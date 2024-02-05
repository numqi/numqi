import torch
import numpy as np
import cvxpy
from tqdm.auto import tqdm
import contextlib
import scipy.linalg
import math

import numqi.gellmann
import numqi.random
import numqi.utils
import numqi.optimize
import numqi.manifold

from ._misc import get_density_matrix_boundary, hf_interpolate_dm, _ree_bisection_solve

def _rand_norm_bounded_unitary(dim, norm2_bound, N0, np_rng):
    # |A|_2 <= |A|_F
    # https://math.stackexchange.com/a/252831
    tmp0 = np_rng.normal(size=(N0,dim,dim)) + 1j*np_rng.normal(size=(N0,dim,dim))
    ret = tmp0 + tmp0.transpose(0,2,1).conj()
    norm = np.linalg.norm(ret, axis=(1,2), ord='fro', keepdims=True)
    ret = scipy.linalg.expm(ret*(1j*norm2_bound/norm))
    return ret


def _cha_reset_state(ketA, ketB, probability, threshold, norm, np_rng, indexR=None):
    mask_low_prob = probability<threshold
    mask_drop = mask_low_prob if (indexR is None) else np.logical_and(mask_low_prob, indexR)
    num_drop = mask_drop.sum()
    num_state,dimA = ketA.shape
    dimB = ketB.shape[1]
    assert ketB.shape[0]==num_state
    if num_drop:
        ind_keep = np.nonzero(np.logical_not(mask_low_prob))[0]
        # TODO choice according to the probability
        ind_select = np_rng.choice(ind_keep, size=num_drop, replace=True)
        tmp0 = _rand_norm_bounded_unitary(dimA, norm, num_drop, np_rng)
        ketA_new = np.einsum(tmp0, [0,1,2], ketA[ind_select], [0,2], [0,1], optimize=True)
        tmp0 = _rand_norm_bounded_unitary(dimB, norm, num_drop, np_rng)
        ketB_new = np.einsum(tmp0, [0,1,2], ketB[ind_select], [0,2], [0,1], optimize=True)
        ret = mask_drop,ketA_new,ketB_new
    else:
        ret = None,None,None
    return ret


class CHABoundaryBagging:
    # CHA with bagging 10.1103/PhysRevA.98.012315
    def __init__(self, dim, num_state=None) -> None:
        assert len(dim)==2
        dimA,dimB = dim
        num_state = 3*(dimA*dimB)**2 if (num_state is None) else num_state
        self.dimA = dim[0]
        self.dimB = dim[1]
        # 3*dimA*dimB*dimA*dimB looks good for 3x3 bipartite system
        self.num_state = num_state

        self.cvx_beta = cvxpy.Variable(name='beta')
        self.cvx_lambda = cvxpy.Variable(num_state, name='lambda')
        self.cvx_A_r = cvxpy.Parameter((num_state,dimA*dimB*dimA*dimB))
        self.cvx_A_i = cvxpy.Parameter((num_state,dimA*dimB*dimA*dimB))
        self.cvx_obj = cvxpy.Maximize(self.cvx_beta)
        self.cvx_problem = None
        self.dm_target = None
        self.ketA = None
        self.ketB = None

    def _rand_init_state(self, np_rng, max_retry):
        assert max_retry>0
        def hf0(sz0, sz1):
            ret = np_rng.normal(size=(sz0, sz1*2)).astype(np.float64, copy=False).view(np.complex128)
            ret /= np.linalg.norm(ret, axis=1, keepdims=True)
            return ret
        for _ in range(max_retry):
            self.ketA = hf0(self.num_state, self.dimA)
            self.ketB = hf0(self.num_state, self.dimB)
            beta = self._cvxpy_solve()
            if (beta is not None) and (not math.isinf(beta)):
                break
        else:
            raise RuntimeError('Failed to find a good initial state')
        ind0 = np.argsort(self.cvx_lambda.value)[::-1]
        self.ketA = self.ketA[ind0]
        self.ketB = self.ketB[ind0]

    def _cvxpy_solve(self):
        N0 = self.dimA*self.dimB
        tmp0 = np.einsum(self.ketA,[0,1],self.ketA.conj(),[0,3],self.ketB,[0,2],self.ketB.conj(),[0,4],[0,1,2,3,4],optimize=True)
        tmp0 = (tmp0.reshape(-1,N0,N0) - np.eye(N0)/N0).reshape(-1,N0*N0)
        self.cvx_A_r.value = tmp0.real
        self.cvx_A_i.value = tmp0.imag
        ret = self.cvx_problem.solve(ignore_dpp=True) #ECOS solver
        # ret and self.cvx_lambda.value could be None if num_state is too small
        return ret

    def set_dm_target(self, dm):
        N0 = self.dimA*self.dimB
        assert dm.shape==(N0,N0)
        assert abs(np.trace(dm)-1) < 1e-10
        assert np.abs(dm-dm.T.conj()).max() < 1e-10
        self.dm_target = dm.copy()
        dm_normed = (dm - np.eye(N0)/N0).reshape(-1) / numqi.gellmann.dm_to_gellmann_norm(dm)
        cvx_constrants = [
            self.cvx_beta*dm_normed.real==self.cvx_lambda @ self.cvx_A_r,
            self.cvx_beta*dm_normed.imag==self.cvx_lambda @ self.cvx_A_i,
            self.cvx_lambda>=0,
            cvxpy.sum(self.cvx_lambda)==1,
        ]
        self.cvx_problem = cvxpy.Problem(self.cvx_obj, cvx_constrants)

    def solve(self, dm, maxiter=150, norm2_init=1, decay_rate=0.97, threshold=1e-7,
                num_init_retry=10, use_tqdm=False, return_info=False, seed=None):
        self.set_dm_target(dm)
        np_rng = numqi.random.get_numpy_rng(seed)
        if num_init_retry>0:
            self._rand_init_state(np_rng, num_init_retry)
        beta_history = [self._cvxpy_solve()]
        assert beta_history[-1] is not None, 'cvxpy solve failed, num_state might be too small'
        norm2_bound = norm2_init
        with (tqdm(range(maxiter)) if use_tqdm else contextlib.nullcontext()) as pbar:
            for _ in (pbar if use_tqdm else range(maxiter)):
                if use_tqdm:
                    pbar.set_postfix_str(f'beta={beta_history[-1]:.5f}, eps={norm2_bound:.4f}')
                mask,tmp2,tmp3 = _cha_reset_state(self.ketA, self.ketB, self.cvx_lambda.value, threshold, norm2_bound, np_rng)
                if mask is not None:
                    self.ketA[mask] = tmp2
                    self.ketB[mask] = tmp3
                norm2_bound *= decay_rate
                beta_history.append(self._cvxpy_solve())
        beta = beta_history[-1]
        if return_info:
            mask = self.cvx_lambda.value > 0
            ret = beta, (self.ketA[mask],self.ketB[mask],self.cvx_lambda.value[mask], beta_history)
        else:
            ret = beta
        return ret


class AutodiffCHAREE(torch.nn.Module):
    def __init__(self, dim0, dim1, num_state=None, distance_kind='ree'):
        r'''Gradient descent model for convex hull approximation to separable states

        Parameters:
            dim0(int): dimension of the first subsystem
            dim1(int): dimension of the second subsystem
            num_state(int): number of states in the convex hull, default to 2*dim0*dim1
            distance_kind(str): 'gellmann' or 'ree', default to 'ree'
        '''
        super().__init__()
        # [2*dA*dB,3*dA*dB] seems to be good enough
        num_state = (2*dim0*dim1) if (num_state is None) else num_state
        distance_kind = distance_kind.lower()
        assert distance_kind in {'gellmann', 'ree'}
        self.distance_kind = distance_kind
        self.num_state = num_state
        self.dim0 = dim0
        self.dim1 = dim1
        self.manifold = numqi.manifold.SeparableDensityMatrix(dim0, dim1, num_state, dtype=torch.complex128)

        self.dm_torch = None
        self.dm_target = None
        self.tr_rho_log_rho = None
        self.expect_op_T_vec = None
        self._torch_logm = ('pade',6,8) #set it by user

    def set_dm_target(self, rho):
        assert (rho.shape[0]==(self.dim0*self.dim1)) and (rho.shape[0]==rho.shape[1])
        self.expect_op_T_vec = None
        self.dm_target = torch.tensor(rho, dtype=torch.complex128)
        self.tr_rho_log_rho = -numqi.utils.get_von_neumann_entropy(rho)

    def set_expectation_op(self, op):
        self.dm_target = None
        self.tr_rho_log_rho = None
        self.expect_op_T_vec = torch.tensor(op.T.reshape(-1), dtype=torch.complex128)

    def forward(self):
        dm_torch = self.manifold().reshape(self.dim0*self.dim1, -1)
        self.dm_torch = dm_torch.detach()
        if self.dm_target is not None:
            if self.distance_kind=='gellmann':
                loss = numqi.gellmann.get_density_matrix_distance2(self.dm_target, dm_torch)
            else:
                loss = numqi.utils.get_relative_entropy(self.dm_target, dm_torch, self.tr_rho_log_rho, self._torch_logm)
        else:
            loss = torch.dot(dm_torch.reshape(-1), self.expect_op_T_vec).real
        return loss

    def get_boundary(self, dm0, xtol=1e-4, converge_tol=1e-10, threshold=1e-7, num_repeat=1, use_tqdm=True, return_info=False, seed=None):
        beta_u = get_density_matrix_boundary(dm0)[1]
        dm0_norm = numqi.gellmann.dm_to_gellmann_norm(dm0)
        np_rng = numqi.random.get_numpy_rng(seed)
        def hf0(beta):
            # use alpha to avoid time-consuming gellmann conversion
            tmp0 = hf_interpolate_dm(dm0, alpha=beta/dm0_norm)
            self.set_dm_target(tmp0)
            theta_optim = numqi.optimize.minimize(self, theta0='uniform',
                        tol=converge_tol, num_repeat=num_repeat, seed=np_rng, print_every_round=0)
            return float(theta_optim.fun)
        beta,history_info = _ree_bisection_solve(hf0, 0, beta_u, xtol, threshold, use_tqdm=use_tqdm)
        ret = (beta,history_info) if return_info else beta
        return ret

    def get_numerical_range(self, op0, op1, num_theta=400, converge_tol=1e-5, num_repeat=1, use_tqdm=True, seed=None):
        np_rng = numqi.random.get_numpy_rng(seed)
        N0 = self.dim0*self.dim1
        assert (op0.shape==(N0,N0)) and (op1.shape==(N0,N0))
        theta_list = np.linspace(0, 2*np.pi, num_theta)
        ret = []
        kwargs = dict(num_repeat=num_repeat, seed=np_rng, print_every_round=0, tol=converge_tol)
        for theta_i in (tqdm(theta_list) if use_tqdm else theta_list):
            # see numqi.entangle.ppt.get_ppt_numerical_range, we use the maximization there
            self.set_expectation_op(-np.cos(theta_i)*op0 - np.sin(theta_i)*op1)
            numqi.optimize.minimize(self, **kwargs)
            rho = self.dm_torch.numpy()
            ret.append([np.trace(x @ rho).real for x in [op0,op1]])
        ret = np.array(ret)
        return ret

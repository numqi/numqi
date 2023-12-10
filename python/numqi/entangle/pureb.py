import torch
import numpy as np
from tqdm.auto import tqdm

import numqi.utils
import numqi.dicke
import numqi.random
import numqi.optimize
import numqi.gellmann
import numqi.manifold

from ._misc import get_density_matrix_boundary, hf_interpolate_dm, _ree_bisection_solve


class PureBosonicExt(torch.nn.Module):
    def __init__(self, dimA, dimB, kext, distance_kind='ree'):
        super().__init__()
        distance_kind = distance_kind.lower()
        assert distance_kind in {'ree','gellmann'}
        self.distance_kind = distance_kind
        Bij = numqi.dicke.get_partial_trace_ABk_to_AB_index(kext, dimB)
        num_dicke = numqi.dicke.get_dicke_number(kext, dimB)
        tmp0 = [torch.int64,torch.int64,torch.complex128]
        self.Bij = [[torch.tensor(y0,dtype=y1) for y0,y1 in zip(x,tmp0)] for x in Bij]
        self.manifold = numqi.manifold.Sphere(dimA*num_dicke, dtype=torch.complex128, method='quotient')
        self.dimA = dimA
        self.dimB = dimB

        self.dm_torch = None
        self.dm_target = None
        self.expect_op_T_vec = None

    def set_dm_target(self, target):
        assert target.ndim in {1,2}
        if target.ndim==1:
            target = target[:,np.newaxis] * target.conj()
        assert (target.shape[0]==target.shape[1])
        self.dm_target = torch.tensor(target, dtype=torch.complex128)

    def set_expectation_op(self, op):
        self.dm_target = None
        self.expect_op_T_vec = torch.tensor(op.T.reshape(-1), dtype=torch.complex128)

    def forward(self):
        tmp1 = self.manifold().reshape(self.dimA,-1)
        self.dm_torch = numqi.dicke.partial_trace_ABk_to_AB(tmp1, self.Bij)
        if self.dm_target is not None:
            if self.distance_kind=='gellmann':
                tmp0 = numqi.gellmann.dm_to_gellmann_basis(self.dm_target)
                tmp1 = numqi.gellmann.dm_to_gellmann_basis(self.dm_torch)
                loss = torch.sum((tmp0-tmp1)**2)
            else:
                loss = numqi.utils.get_relative_entropy(self.dm_target, self.dm_torch, kind='error')
        else:
            loss = torch.dot(self.dm_torch.view(-1), self.expect_op_T_vec).real
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
        N0 = self.dimA*self.dimB
        assert (op0.shape==(N0,N0)) and (op1.shape==(N0,N0))
        theta_list = np.linspace(0, 2*np.pi, num_theta)
        ret = []
        kwargs = dict(num_repeat=num_repeat, seed=np_rng, print_every_round=0, tol=converge_tol)
        for theta_i in (tqdm(theta_list) if use_tqdm else theta_list):
            # see numqi.entangle.get_ppt_numerical_range, we use the maximization there
            self.set_expectation_op(-np.cos(theta_i)*op0 - np.sin(theta_i)*op1)
            numqi.optimize.minimize(self, **kwargs)
            rho = self.dm_torch.detach().numpy()
            ret.append([np.trace(x @ rho).real for x in [op0,op1]])
        ret = np.array(ret)
        return ret

# TODO
# class PureBosonicExtMixed(torch.nn.Module):
#     def __init__(self, dimA, dimB, num_mix) -> None:
#         super().__init__()
#         self.num_mix = num_mix
#         self.dimA = dimA
#         self.dimB = dimB
#         np_rng = np.random.default_rng()
#         tmp0 = np_rng.uniform(num_mix)
#         self.probability = torch.nn.Parameter(torch.tensor(tmp0/tmp0.sum(), dtype=torch.float64))
#         self.pureb_list = torch.nn.ModuleList([numqi.pureb.PureBosonicExt(dimA, dimB) for _ in range(num_mix)])

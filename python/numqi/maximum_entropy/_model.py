import numpy as np
import torch

import numqi.optimize
import numqi.manifold

from ._internal import eigvalsh_largest_power_iteration#, NANGradientToNumber

class MaximumEntropyModel(torch.nn.Module):
    def __init__(self, term_list, use_full=False):
        super().__init__()
        term_np = np.stack(term_list)
        assert (term_np.ndim==3) and (term_np.shape[1]==term_np.shape[2])
        assert np.abs(term_np.transpose(0,2,1).conj()-term_np).max() < 1e-10
        self.term = torch.tensor(term_np, dtype=torch.complex128)
        num_term,dim,_ = self.term.shape
        if use_full:
            self.manifold_PSD = numqi.manifold.Trace1PSD(dim, dtype=torch.complex128)
        else:
            self.theta = torch.nn.Parameter(torch.rand(num_term, dtype=torch.float64))

        self.term_value_target = None
        self.expect_op = None
        self.term_value = None
        self.dm_torch = None

    def set_target(self, x):
        x = np.asarray(x)
        if x.ndim==1:
            self.term_value_target = torch.tensor(x, dtype=torch.float64)
            self.expect_op = None
        else:
            assert (x.ndim==2) and (x.shape[0]==x.shape[1])
            assert np.abs(x-x.T.conj()).max() < 1e-10
            self.term_value_target = None

    def forward(self):
        num_term,dim,_ = self.term.shape
        if hasattr(self, 'manifold_PSD'):
            dm_torch = self.manifold_PSD()
        else:
            tmp0 = (self.theta.to(torch.complex128) @ self.term.reshape(num_term, dim*dim)).reshape(dim,dim)
            dm_torch = numqi.manifold.symmetric_matrix_to_trace1PSD(tmp0)
        self.dm_torch = dm_torch.detach()
        if self.term_value_target is not None:
            expectation = (self.term.reshape(num_term,dim*dim) @ dm_torch.conj().view(dim*dim)).real
            self.term_value = expectation
            loss = torch.sum((expectation - self.term_value_target)**2)
        else:
            assert self.expect_op is not None
            loss = torch.trace(self.expect_op @ dm_torch).real
        return loss

    def get_witness(self, term_value, num_repeat=1, loss_zero_eps=1e-7, rank_zero_eps=1e-4):
        self.set_target(term_value)
        converge_tol = loss_zero_eps/1000
        theta_optim0 = numqi.optimize.minimize(self, theta0='uniform', num_repeat=num_repeat, tol=converge_tol)
        if theta_optim0.fun < loss_zero_eps:
            ret = None,None #not a witness
        else:
            rho = self.dm_torch.detach().numpy().copy()
            EVL,EVC = np.linalg.eigh(rho)
            rank = (EVL>rank_zero_eps).sum()
            if rank>1:
                coeffA = self.term_value.detach().numpy()
                coeffC = theta_optim0.x
            else:
                tmp0 = EVC[:,(-rank):]
                rho_maxent = (tmp0/rank) @ tmp0.T.conj()
                term_value_maxent = np.trace(self.term.numpy() @ rho_maxent, axis1=1, axis2=2).real
                self.set_target(term_value_maxent)
                theta_optim0 = numqi.optimize.minimize(self, theta0='uniform', num_repeat=num_repeat, tol=converge_tol)
                assert theta_optim0.fun < loss_zero_eps
                coeffA = term_value_maxent
                coeffC = theta_optim0.x
            tmp0 = np.trace(self.term.numpy(), axis1=1, axis2=2).real/self.term.shape[1]
            coeffC *= np.sign(np.dot(tmp0 - coeffA, coeffC))
            ret = coeffA, coeffC
            # witness: (x-a)*c >= 0
        return ret


class MaximumEntropyTangentModel(torch.nn.Module):
    def __init__(self, op_list, factor=1, beta=None, grad_nan_to_num=None) -> None:
        super().__init__()
        assert (op_list.ndim==3) and (op_list.shape[1]==op_list.shape[2])
        assert np.abs(op_list-op_list.transpose(0,2,1).conj()).max() < 1e-10
        num_op,dim_op,_ = op_list.shape
        self.op_list = torch.tensor(op_list, dtype=torch.complex128)
        self.num_op = num_op
        self.dim_op = dim_op
        self.factor = float(factor)
        self.beta = beta

        np_rng = np.random.default_rng()
        tmp0 = np_rng.uniform(-1, 1, size=num_op)
        self.theta = torch.nn.Parameter(torch.tensor(tmp0, dtype=torch.float64))
        self.grad_nan_to_num = grad_nan_to_num
        self.eigen_iteration_tol = 1e-7

        self.vecA = None
        self.vecB = None
        self.vecN = None

    def set_target_vec(self, np0):
        assert np0.shape==(self.num_op,)
        assert np.linalg.norm(np0) > 1e-10
        assert not np.iscomplexobj(np0)
        self.vecB = torch.tensor(np0 / np.linalg.norm(np0), dtype=torch.float64)

    def forward(self):
        # make sure torch.dot(vecB,vecN) is always positive
        tmp0 = torch.dot(self.theta, self.vecB)
        tmp1 = self.theta + (torch.nn.functional.softplus(tmp0)-tmp0)*self.vecB
        vecN = tmp1 / torch.linalg.norm(tmp1)
        self.vecN = vecN.detach()

        matH = (vecN.to(self.op_list.dtype) @ self.op_list.reshape(self.num_op,-1)).reshape(self.dim_op, self.dim_op)
        if self.beta is None:
            # if self.grad_nan_to_num is not None:
            #     matH = NANGradientToNumber.apply(matH, self.grad_nan_to_num)
            # EVC = torch.linalg.eigh(matH)[1][:,-1] #take largest eigenvalue
            EVC,num_step = eigvalsh_largest_power_iteration(matH, tag_force_full=True, vec0=None, tol=self.eigen_iteration_tol)
            vecA = ((self.op_list @ EVC) @ EVC.conj()).real
        else:
            rho = numqi.manifold.symmetric_matrix_to_trace1PSD(self.beta*matH)
            vecA = (self.op_list.conj().reshape(self.num_op,-1) @ rho.reshape(-1)).real
        self.vecA = vecA.detach()
        loss = self.factor*torch.dot(vecA, vecN) / (torch.dot(vecN, self.vecB)) # 0.5* for Gell-Mann matrix nomalization
        return loss

    def get_vector(self):
        vecN = self.vecN.numpy().copy()
        vecA = self.vecA.numpy() * self.factor
        return vecA,vecN

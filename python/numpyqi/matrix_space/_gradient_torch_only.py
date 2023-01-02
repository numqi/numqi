import numpy as np
import torch

import torch_wrapper
import scipy.optimize

import numpyqi.param
from ._internal import get_hs_orthogonal_basis
from ..utils import hf_tuple_of_int

class DetectMatrixSpaceRank(torch.nn.Module):
    def __init__(self, matrix_space, rank, input_orth=False, dtype='float64', device='cpu'):
        super().__init__()
        assert dtype in {'float32','float64'}
        self.device = device
        self.dtype = torch.float32 if dtype=='float32' else torch.float64
        self.cdtype = torch.complex64 if dtype=='float32' else torch.complex128
        rank = hf_tuple_of_int(rank)
        assert (len(rank)==1) or (len(rank)==3)
        is_hermite = len(rank)==3
        if input_orth:
            self.matrix_space_orth = torch.tensor(matrix_space, dtype=self.cdtype, device=self.device)
        else:
            tmp0 = get_hs_orthogonal_basis(matrix_space, is_hermite)
            self.matrix_space_orth = torch.tensor(tmp0, dtype=self.cdtype, device=self.device)

        np_rng = np.random.default_rng()
        hf0 = lambda *x: torch.nn.Parameter(torch.tensor(np_rng.normal(size=x), dtype=self.dtype, device=self.device))
        if is_hermite:
            assert len(rank)==3
            self.EVL_free = hf0(rank[0]) if (rank[0]>0) else None
            self.EVL_positive = hf0(rank[1]) if (rank[1]>0) else None
            self.EVL_negative = hf0(rank[2]) if (rank[2]>0) else None
            self.unitary_theta = hf0(*matrix_space.shape[1:])
            self.unitary_theta1 = None
        else:
            assert rank[0]>0
            self.EVL_free = None
            self.EVL_positive = hf0(rank[0])
            self.EVL_negative = None
            self.unitary_theta = hf0(*matrix_space.shape[1:])
            self.unitary_theta1 = hf0(*matrix_space.shape[1:])
        self.matH = None

    def forward(self):
        tmp0 = [
            self.EVL_free,
            None if (self.EVL_positive is None) else torch.nn.functional.softplus(self.EVL_positive),
            None if (self.EVL_negative is None) else (-torch.nn.functional.softplus(self.EVL_negative)),
        ]
        tmp1 = torch.cat([x for x in tmp0 if x is not None])
        EVL_all = tmp1 / torch.linalg.norm(tmp1)

        unitary = numpyqi.param.real_matrix_to_unitary(self.unitary_theta)[:len(EVL_all)]
        if self.unitary_theta1 is None:
            matH = (unitary.T.conj()*EVL_all) @ unitary
        else:
            unitary1 = numpyqi.param.real_matrix_to_unitary(self.unitary_theta1)[:len(EVL_all)]
            matH = (unitary.T.conj()*EVL_all) @ unitary1
        self.matH  = matH
        tmp0 = self.matrix_space_orth.reshape(-1, matH.numel()).conj() @ matH.reshape(-1)
        loss = torch.dot(tmp0.real, tmp0.real) + torch.dot(tmp0.imag, tmp0.imag)
        # if hermite, tmp0.imag is zero
        return loss

    def get_matrix(self):
        ret = self.matH.detach().numpy().copy()
        return ret

    # torch_wrapper.minimize(model, 'normal', num_repeat=3, tol=1e-7, print_freq=20)
    def minimize(self, num_repeat=3, print_freq=-1, tol=1e-7, threshold=None, seed=None):
        # threshold is used for quick return if fun<threshold
        np_rng = np.random.default_rng(seed)
        num_parameter = len(torch_wrapper.get_model_flat_parameter(self))
        hf_model = torch_wrapper.hf_model_wrapper(self)
        loss_list = []
        for _ in range(num_repeat):
            theta0 = np_rng.normal(size=num_parameter)
            hf_callback = torch_wrapper.hf_callback_wrapper(hf_model, print_freq=print_freq)
            theta_optim = scipy.optimize.minimize(hf_model, theta0, jac=True, method='L-BFGS-B', tol=tol, callback=hf_callback)
            loss_list.append(theta_optim)
            if (threshold is not None) and (theta_optim.fun < threshold):
                break
        ret = min(loss_list, key=lambda x: x.fun)
        torch_wrapper.set_model_flat_parameter(self, ret.x)
        return ret


class DetectExtendibleChannel(torch.nn.Module):
    def __init__(self, matrix_space, dtype='float64', device='cpu') -> None:
        super().__init__()
        assert dtype in {'float32','float64'}
        self.device = device
        self.dtype = torch.float32 if dtype=='float32' else torch.float64
        self.cdtype = torch.complex64 if dtype=='float32' else torch.complex128
        self.matrix_subspace_orth = torch.tensor(matrix_space, dtype=self.cdtype, device=self.device)
        np_rng = np.random.default_rng()
        hf_para = lambda *x: torch.nn.Parameter(torch.tensor(np_rng.normal(size=x), dtype=self.dtype, device=self.device))
        # self.unitary_theta = hf_para(*matrix_space.shape[1:])
        self.vec0 = hf_para(matrix_space.shape[1]*2)
        self.vec1 = hf_para(matrix_space.shape[1]*2)
        self.mask = None
        self.factor = 1

    def forward(self):
        # unitary = numpyqi.param.real_matrix_to_unitary(self.unitary_theta)
        # vec0 = unitary[0]
        # vec1 = unitary[1]
        N0 = self.matrix_subspace_orth.shape[1]
        tmp0 = self.vec0[:N0] + 1j*self.vec0[N0:]
        vec0 = tmp0 / torch.linalg.norm(tmp0)
        tmp0 = self.vec1[:N0] + 1j*self.vec1[N0:]
        tmp0 = tmp0 - vec0*torch.dot(vec0.conj(), tmp0)
        vec1 = tmp0 / torch.linalg.norm(tmp0)

        tmp0 = vec1.conj() @ self.matrix_subspace_orth @ vec0
        # tmp0 = unitary[1] @ self.matrix_subspace_orth @ unitary[0]
        loss = torch.dot(tmp0.real, tmp0.real) + torch.dot(tmp0.imag, tmp0.imag)
        # loss = torch.abs(tmp0).sum()
        return loss


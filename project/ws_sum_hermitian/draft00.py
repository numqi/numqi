import numpy as np
import torch
from tqdm import tqdm

import numqi

np_rng = np.random.default_rng()


class TwoHermitianSumModel(torch.nn.Module):
    def __init__(self, N0:int, dtype:str='complex128') -> None:
        # https://doi.org/10.4153/CJM-2010-007-2
        super().__init__()
        N0 = int(N0)
        assert N0>=1
        self.N0 = N0
        assert dtype=='complex128' # TODO real symmetric, complex hermitian
        self.dtype = torch.complex128

        np_rng = np.random.default_rng()
        hf0 = lambda *x: torch.nn.Parameter(torch.tensor(np_rng.uniform(-1,1,size=x), dtype=torch.float64))
        self.thetaU0 = hf0(N0, N0)
        self.thetaU1 = hf0(N0, N0)

    def set_matrix(self, matA=None, matB=None, matC=None):
        shape = self.N0, self.N0
        hf0 = lambda x: np.abs(x-x.T.conj()).max()<1e-10
        if matA is not None:
            assert matA.shape==shape
            assert hf0(matA)
            self.matA = torch.tensor(matA, dtype=self.dtype)
        if matB is not None:
            assert matB.shape==shape
            assert hf0(matB)
            self.matB = torch.tensor(matB, dtype=self.dtype)
        if matC is not None:
            assert matC.shape==shape
            assert hf0(matC)
            self.matC = torch.tensor(matC, dtype=self.dtype)

    def forward(self):
        matU0 = numqi.param.real_matrix_to_special_unitary(self.thetaU0)
        matU1 = numqi.param.real_matrix_to_special_unitary(self.thetaU1)
        matC_target = matU0 @ self.matA @ matU0.T.conj() + matU1 @ self.matB @ matU1.T.conj()
        self.matC_target = matC_target
        tmp0 = (matC_target - self.matC).reshape(-1)
        loss = torch.dot(tmp0, tmp0.conj()).real/len(tmp0)
        return loss



for N0 in [4, 8, 16, 32, 64]:
    # N0 = 50
    model = TwoHermitianSumModel(N0)
    for _ in range(10):
        matA = numqi.random.rand_hermitian_matrix(N0, seed=np_rng)
        matB = numqi.random.rand_hermitian_matrix(N0, seed=np_rng)
        matU0 = numqi.random.rand_unitary_matrix(N0, seed=np_rng)
        matU1 = numqi.random.rand_unitary_matrix(N0, seed=np_rng)
        matC = matU0 @ matA @ matU0.T.conj() + matU1 @ matB @ matU1.T.conj()
        # matC = numqi.random.rand_hermitian_matrix(N0, seed=np_rng)
        # eigA = np.linalg.eigvalsh(matA)
        # eigB = np.linalg.eigvalsh(matB)
        # eigC = np.linalg.eigvalsh(matC)

        model.set_matrix(matA=matA, matB=matB, matC=matC)
        theta_optim = numqi.optimize.minimize(model, tol=1e-12, num_repeat=1, print_every_round=0)
        assert theta_optim.fun < 1e-10, str(theta_optim.fun)
# N0, time (second)
# 4, 0.07
# 8, 0.13
# 16, 0.22
# 32, 0.45
# 64, 1.4

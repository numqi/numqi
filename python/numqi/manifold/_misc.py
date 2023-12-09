import numpy as np
import torch

from ._internal import SpecialOrthogonal, OpenInterval

hf_space_orth = lambda x: np.linalg.eigh(np.eye(x.shape[0]) - x @ x.T.conj())[1][:,x.shape[1]:]

class StiefelManifoldDistanceModel(torch.nn.Module):
    def __init__(self, dim:int, rank:int, parametrize='exp', dtype:torch.dtype=torch.complex128):
        super().__init__()
        assert (rank>0) and (dim>rank)
        self.dim = int(dim)
        self.rank = int(rank)
        assert dtype in {torch.float32,torch.float64,torch.complex64,torch.complex128}
        self.dtype = dtype
        self.manifold = SpecialOrthogonal(dim-rank, method=parametrize, dtype=dtype)
        self.manifold_phase = OpenInterval(-np.pi, np.pi)

    def set_space(self, space0, space1, zero_eps=1e-10):
        assert (space0.shape==(self.dim, self.rank)) and (space0.shape==space1.shape)
        tmp0 = np.eye(self.rank)
        assert np.abs(space0.T.conj() @ space0 - tmp0).max() < zero_eps
        assert np.abs(space1.T.conj() @ space1 - tmp0).max() < zero_eps
        self.space0 = torch.tensor(space0, dtype=self.dtype)
        self.space1 = torch.tensor(space1, dtype=self.dtype)
        self.space0_orth = torch.tensor(hf_space_orth(space0), dtype=self.dtype)
        self.space1_orth = torch.tensor(hf_space_orth(space1), dtype=self.dtype)

    def forward(self):
        tmp0 = torch.exp(1j*self.manifold_phase())
        matU = self.space1_orth @ (tmp0*self.manifold()) @ self.space0_orth.T.conj()
        matU = matU + self.space1 @ self.space0.T.conj()
        self.matU = matU.detach()
        EVL = torch.angle(torch.linalg.eigvals(matU))
        EVL = EVL - EVL.mean() #sometimes might be 2pi
        loss = torch.dot(EVL, EVL)
        return loss

class GrassmannManifoldDistanceModel(torch.nn.Module):
    def __init__(self, dim:int, rank:int, parametrize='exp', dtype:torch.dtype=torch.complex128):
        super().__init__()
        assert (rank>0) and (dim>rank)
        self.dim = int(dim)
        self.rank = int(rank)
        assert dtype in {torch.float32,torch.float64,torch.complex64,torch.complex128}
        self.dtype = dtype
        self.manifold0 = SpecialOrthogonal(rank, method=parametrize, dtype=dtype)
        self.manifold0_phase = OpenInterval(-np.pi, np.pi)
        self.manifold1 = SpecialOrthogonal(dim-rank, method=parametrize, dtype=dtype)
        self.manifold1_phase = OpenInterval(-np.pi, np.pi)

    def set_space(self, space0, space1, zero_eps=1e-10):
        assert (space0.shape==(self.dim, self.rank)) and (space0.shape==space1.shape)
        tmp0 = np.eye(self.rank)
        assert np.abs(space0.T.conj() @ space0 - tmp0).max() < zero_eps
        assert np.abs(space1.T.conj() @ space1 - tmp0).max() < zero_eps
        self.space0 = torch.tensor(space0, dtype=self.dtype)
        self.space1 = torch.tensor(space1, dtype=self.dtype)
        self.space0_orth = torch.tensor(hf_space_orth(space0), dtype=self.dtype)
        self.space1_orth = torch.tensor(hf_space_orth(space1), dtype=self.dtype)

    def forward(self):
        tmp0 = torch.exp(1j*self.manifold0_phase())
        matU = self.space1 @ (tmp0*self.manifold0()) @ self.space0.T.conj()
        tmp0 = torch.exp(1j*self.manifold1_phase())
        matU = matU + self.space1_orth @ (tmp0*self.manifold1()) @ self.space0_orth.T.conj()
        self.matU = matU.detach()
        EVL = torch.angle(torch.linalg.eigvals(matU))
        EVL = EVL - EVL.mean() #sometimes might be 2pi
        loss = torch.dot(EVL, EVL)
        return loss


class TwoHermitianSumModel(torch.nn.Module):
    def __init__(self, dim:int, dtype:torch.dtype=torch.complex128):
        # https://doi.org/10.4153/CJM-2010-007-2
        super().__init__()
        assert dim>=1
        self.dim = dim
        self.dtype = dtype
        self.manifold0 = SpecialOrthogonal(dim, method='exp', dtype=dtype)
        self.manifold1 = SpecialOrthogonal(dim, method='exp', dtype=dtype)

    def set_matrix(self, matA, matB, matC, zero_eps=1e-10):
        assert all((x.ndim==2) and (x.shape==(self.dim, self.dim))
                    and (np.abs(x-x.T.conj()).max()<zero_eps) for x in [matA, matB, matC])
        self.matA = torch.tensor(matA, dtype=self.dtype)
        self.matB = torch.tensor(matB, dtype=self.dtype)
        self.matC = torch.tensor(matC, dtype=self.dtype)

    def forward(self):
        matU0 = self.manifold0()
        matU1 = self.manifold1()
        tmp0 = matU0 @ self.matA @ matU0.T.conj() + matU1 @ self.matB @ matU1.T.conj()
        tmp0 = (tmp0 - self.matC).reshape(-1)
        loss = torch.dot(tmp0, tmp0.conj()).real/tmp0.shape[0]
        return loss

import numpy as np
import torch

import numqi.manifold

class CoherenceFormationModel(torch.nn.Module):
    def __init__(self, dim:int, num_term:int, rank:int|None=None):
        r'''gradient-based optimization model for evaluating Coherence of Formation

        reference: Unified Framework for Calculating Convex Roof Resource Measures
        [arxiv-link](https://arxiv.org/abs/2406.19683)

        Parameters:
            dim (int): the dimension of the Hilbert space
            num_term (int): the number of entries for ensemble decomposition (dimension for the Stiefel matrix)
            rank (int|None): the rank of the density matrix, default is None, which means the full-rank density matrix
        '''
        super().__init__()
        self.dtype = torch.float64
        self.cdtype = torch.complex128
        self.dim = int(dim)
        self.num_term = int(num_term)
        if rank is None:
            rank = self.dim
        assert num_term>=rank
        assert rank!=1, 'for pure state, call "numqi.entangle.get_coherence_of_formation_pure()" instead'
        self.manifold = numqi.manifold.Stiefel(num_term, rank, dtype=self.cdtype, method='polar')
        self.rank = rank

        self._sqrt_rho_Tconj = None
        self._eps = torch.tensor(torch.finfo(self.dtype).smallest_normal, dtype=self.dtype)

    def set_density_matrix(self, rho):
        r'''Set the density matrix

        Parameters:
            rho (np.ndarray): the density matrix, shape=(dim,dim)
        '''
        assert rho.shape == (self.dim, self.dim)
        assert np.abs(rho - rho.T.conj()).max() < 1e-10
        assert abs(np.trace(rho) - 1) < 1e-10
        assert np.linalg.eigvalsh(rho)[0] > -1e-10
        EVL,EVC = np.linalg.eigh(rho)
        EVL = np.maximum(0, EVL[-self.rank:])
        assert abs(EVL.sum()-1) < 1e-10
        EVC = EVC[:,-self.rank:]
        tmp0 = (EVC * np.sqrt(EVL)).reshape(self.dim, self.rank)
        self._sqrt_rho_Tconj = torch.tensor(np.ascontiguousarray(tmp0.T.conj()), dtype=self.cdtype)

    def forward(self):
        mat_st = self.manifold()
        tmp0 = mat_st @ self._sqrt_rho_Tconj
        p_alpha_i = tmp0.real**2 + tmp0.imag**2
        p_alpha = p_alpha_i.sum(axis=1)
        tmp0 = torch.dot(p_alpha, torch.log(torch.maximum(p_alpha, self._eps)))
        tmp1 = p_alpha_i.reshape(-1)
        ret = tmp0 - torch.dot(tmp1, torch.log(torch.maximum(tmp1, self._eps)))
        return ret


class GeometricCoherenceModel(torch.nn.Module):
    def __init__(self, dim:int, num_term:int, temperature:float|None=None, rank:int|None=None):
        r'''gradient-based optimization model for evaluating Geometric Coherence

        reference: Unified Framework for Calculating Convex Roof Resource Measures
        [arxiv-link](https://arxiv.org/abs/2406.19683)

        Parameters:
            dim (int): the dimension of the Hilbert space
            num_term (int): the number of entries for ensemble decomposition (dimension for the Stiefel matrix)
            temperature (float|None): when temperature (T) is a float number, LogSumExp is used to approximate the "max" operation.
                    When `temperature=None`, the "max" operation is used. As T goes to zero, LogSumExp is close to "max",
                    and "max" operation is generally difficult for gradient-based optimization. It's recommended to use
                    some small value for T, e.g. T=0.3, during optimization, then set T=None to get the final result.
            rank (int|None): the rank of the density matrix, default is None, which means the full-rank density matrix
        '''
        super().__init__()
        self.dtype = torch.float64
        self.cdtype = torch.complex128
        self.dim = int(dim)
        self.num_term = int(num_term)
        if rank is None:
            rank = self.dim
        assert num_term>=rank
        assert rank!=1, 'for pure state, call "numqi.entangle.get_geometric_coherence_pure()" instead'
        self.manifold = numqi.manifold.Stiefel(num_term, rank, dtype=self.cdtype, method='polar')
        self.rank = rank
        self.temperature = temperature

        self._sqrt_rho_Tconj = None

    def set_density_matrix(self, rho):
        r'''Set the density matrix

        Parameters:
            rho (np.ndarray): the density matrix, shape=(dim,dim)
        '''
        assert rho.shape == (self.dim, self.dim)
        assert np.abs(rho - rho.T.conj()).max() < 1e-10
        assert abs(np.trace(rho) - 1) < 1e-10
        assert np.linalg.eigvalsh(rho)[0] > -1e-10
        EVL,EVC = np.linalg.eigh(rho)
        EVL = np.maximum(0, EVL[-self.rank:])
        assert abs(EVL.sum()-1) < 1e-10
        EVC = EVC[:,-self.rank:]
        tmp0 = (EVC * np.sqrt(EVL)).reshape(self.dim, self.rank)
        self._sqrt_rho_Tconj = torch.tensor(np.ascontiguousarray(tmp0.T.conj()), dtype=self.cdtype)

    def forward(self, use_temperature:bool=True):
        r'''evaluate the geometric measure for coherence

        Parameters:
            use_temperature (bool): whether to use temperature for LogSumExp, default is True. to get the final result, set it to False

        Returns:
            gmc (torch.Tensor): the (approximated) geometric measure of coherence
        '''
        mat_st = self.manifold()
        tmp0 = mat_st @ self._sqrt_rho_Tconj
        p_alpha_i = tmp0.real**2 + tmp0.imag**2
        if use_temperature and (self.temperature is not None):
            ret = 1 - self.temperature*torch.logsumexp(p_alpha_i/self.temperature, dim=1).sum()
        else:
            ret = 1 - p_alpha_i.max(axis=1)[0].sum()
        return ret

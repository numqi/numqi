import numpy as np
import torch
import scipy.sparse
import opt_einsum

import numqi.manifold


class MagicStabilizerEntropyModel(torch.nn.Module):
    def __init__(self, alpha:float, num_qubit:int, num_term:int, rank:int|None=None, method:str='polar'):
        super().__init__()
        self.dtype = torch.float64
        self.cdtype = torch.complex128
        assert alpha>=2
        self.alpha = float(alpha)
        self.num_qubit = int(num_qubit)
        self.num_term = int(num_term)
        if rank is None:
            rank = 2**self.num_qubit
        assert num_term>=rank
        self.manifold = numqi.manifold.Stiefel(num_term, rank, dtype=self.cdtype, method=method)
        self.rank = rank

        tmp0 = numqi.gate.get_pauli_group(num_qubit, use_sparse=True)
        self.pauli_mat = scipy.sparse.vstack(tmp0, format='csr')

        self._sqrt_rho_Tconj = None
        self._psi_pauli_psi = None
        self.contract_expr = None
        # 1/6 is arbitrarily chosen
        self._eps = torch.tensor(torch.finfo(self.dtype).smallest_normal**(1/6), dtype=self.dtype)
        # self._eps = torch.tensor(10**(-12), dtype=self.dtype)

    def set_density_matrix(self, rho):
        r'''Set the density matrix

        Parameters:
            rho (np.ndarray): the density matrix, shape=(dim,dim)
        '''
        dim = 2**self.num_qubit
        assert rho.shape == (dim, dim)
        assert np.abs(rho - rho.T.conj()).max() < 1e-10
        assert abs(np.trace(rho) - 1) < 1e-10
        assert np.linalg.eigvalsh(rho)[0] > -1e-10
        EVL,EVC = np.linalg.eigh(rho)
        EVL = np.maximum(0, EVL[-self.rank:])
        assert abs(EVL.sum()-1) < 1e-10
        EVC = EVC[:,-self.rank:]
        tmp0 = (EVC * np.sqrt(EVL)).reshape(dim, self.rank)
        self._sqrt_rho_Tconj = torch.tensor(np.ascontiguousarray(tmp0.T.conj()), dtype=self.cdtype)
        tmp1 = (self.pauli_mat @ tmp0).reshape(-1, dim, self.rank)
        self._psi_pauli_psi = torch.tensor(np.einsum(tmp1, [0,1,2], tmp0.conj(), [1,3], [0,3,2], optimize=True), dtype=self.cdtype)
        # man_st, mat_st.conj(), _psi_pauli_psi
        self.contract_expr = opt_einsum.contract_expression([self.num_term,self.rank], [0,1],
                    [self.num_term,self.rank], [0,2], self._psi_pauli_psi, [3,1,2], [3,0], constants=[2])

    def forward(self):
        mat_st = self.manifold()
        tmp0 = ((self.contract_expr(mat_st, mat_st.conj()).real**2)**self.alpha).sum(axis=0)
        psi_tilde = mat_st @ self._sqrt_rho_Tconj
        plist = (psi_tilde.real**2 + psi_tilde.imag**2).sum(axis=1)
        tmp2 = torch.maximum(plist, self._eps)**(1-2*self.alpha)
        loss = - torch.dot(tmp0, tmp2) / (2**self.num_qubit)
        return loss

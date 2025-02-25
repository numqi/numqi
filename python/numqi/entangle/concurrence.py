import numpy as np
import opt_einsum
import torch

from numqi.matrix_space._public import _get_nontrivial_subset_list
import numqi.manifold

def get_concurrence_2qubit(rho:np.ndarray):
    r'''get the concurrence of a 2-qubit density matrix
    [wiki-link](https://en.wikipedia.org/wiki/Concurrence_(quantum_computing))

    Parameters:
        rho (np.ndarray): a 2-qubit density matrix, shape=(4,4)

    Returns:
        ret (float): the concurrence of the 2-qubit density matrix
    '''
    assert (rho.shape==(4,4)) and (np.abs(rho-rho.T.conj()).max()<1e-10)
    tmp0 = np.array([-1,1,1,-1])
    z0 = (tmp0[:,np.newaxis]*tmp0) * rho[::-1,::-1].conj()
    # tmp0 = np.kron(numqi.gate.Y, numqi.gate.Y).real
    # z0_ = tmp0 @ rho.conj() @ tmp0
    # assert np.abs(z0_-z0).max() < 1e-10
    # scipy.linalg.sqrtm is not precise and also slow, so replace it with eigendecomposition
    # tmp0 = scipy.linalg.sqrtm(rho)
    # if tmp0.dtype.name=='complex256':
    #     # scipy-v1.10 bug https://github.com/scipy/scipy/issues/18250
    #     tmp0 = tmp0.astype(np.complex128)
    EVL,EVC = np.linalg.eigh(rho)
    sqrt_rho = (EVC * np.sqrt(np.maximum(0,EVL))) @ EVC.T.conj()
    EVL = np.sqrt(np.maximum(0, np.linalg.eigvalsh(sqrt_rho @ z0 @ sqrt_rho)))
    ret = np.maximum(2*EVL[-1]-EVL.sum(), 0)
    return ret


def get_concurrence_pure(psi:np.ndarray):
    r'''get the concurrence of a bipartite pure state

    Parameters:
        psi (np.ndarray): a pure state, shape=(dimA,dimB)

    Returns:
        ret (float): the concurrence of the bipartite pure state
    '''
    assert (psi.ndim==2)
    if (psi.shape[0]==1) or (psi.shape[1]==1):
        ret = 0
    else:
        if psi.shape[0]<psi.shape[1]:
            tmp0 = psi @ psi.conj().T
        else:
            tmp0 = psi.conj().T @ psi
        tmp1 = tmp0.reshape(-1)
        tmp2 = np.vdot(tmp1, tmp1).real #Frobenius norm, np.trace(tmp1 @ tmp1)
        ret = np.sqrt(2*(1-tmp2))
    return ret


class ConcurrenceModel(torch.nn.Module):
    '''Calculate the concurrence of a 2-qubit density matrix via optimization

    What is the motivation for the definition of concurrence in quantum information?
    [stackexchange-link](https://physics.stackexchange.com/a/46509/283720)
    '''
    def __init__(self, dimA:int, dimB:int, num_term:int, rank:int=None):
        r'''Initialize the model

        Parameters:
            dimA (int): the dimension of the first subsystem
            dimB (int): the dimension of the second subsystem
            num_term (int): the number of terms in the variational ansatz, `num_term` is bounded by (dimA*dimB)**2
            rank (int): the rank of the density matrix
        '''
        super().__init__()
        self.dtype = torch.float64
        self.cdtype = torch.complex128
        self.dimA = dimA
        self.dimB = dimB
        if rank is None:
            rank = dimA*dimB
        self.num_term = num_term
        assert num_term>=rank
        self.rank = rank
        self.manifold = numqi.manifold.Stiefel(num_term, rank, dtype=self.cdtype, method='polar')

        self._sqrt_rho = None
        self._eps = torch.tensor(torch.finfo(self.dtype).smallest_normal, dtype=self.dtype)
        self.contract_expr = None
        self.contract_expr1 = None

    def set_density_matrix(self, rho:np.ndarray):
        r'''Set the density matrix

        Parameters:
            rho (np.ndarray): the density matrix, shape=(dimA*dimB,dimA*dimB)
        '''
        assert rho.shape == (self.dimA*self.dimB, self.dimA*self.dimB)
        assert np.abs(rho - rho.T.conj()).max() < 1e-10
        assert abs(np.trace(rho) - 1) < 1e-10
        assert np.linalg.eigvalsh(rho)[0] > -1e-10
        EVL,EVC = np.linalg.eigh(rho)
        EVL = np.maximum(0, EVL[-self.rank:])
        assert abs(EVL.sum()-1) < 1e-10
        EVC = EVC[:,-self.rank:]
        tmp0 = (EVC * np.sqrt(EVL)).reshape(self.dimA, self.dimB, self.rank)
        self._sqrt_rho = torch.tensor(tmp0, dtype=self.cdtype)
        tmp0 = self._sqrt_rho.conj().resolve_conj()
        if self.dimA<=self.dimB:
            self.contract_expr = opt_einsum.contract_expression(self._sqrt_rho, [0,3,4], tmp0, [1,3,5],
                                [self.num_term,self.rank], [2,4], [self.num_term,self.rank], [2,5], [2,0,1], constants=[0,1])
        else:
            self.contract_expr = opt_einsum.contract_expression(self._sqrt_rho, [3,0,4], tmp0, [3,1,5],
                                [self.num_term,self.rank], [2,4], [self.num_term,self.rank], [2,5], [2,0,1], constants=[0,1])
        tmp0 = min(self.dimA, self.dimB)
        self.contract_expr1 = opt_einsum.contract_expression([self.num_term,tmp0,tmp0], [0,1,2], [self.num_term,tmp0,tmp0], [0,1,2], [0])

    def forward(self):
        mat_st = self.manifold()
        rdm_not_normed = self.contract_expr(mat_st, mat_st.conj(), backend='torch')
        prob = torch.einsum(rdm_not_normed, [0,1,1], [0]).real
        purity = self.contract_expr1(rdm_not_normed, rdm_not_normed.conj()).real
        tmp0 = torch.maximum(self._eps, 2*(prob*prob - purity))
        loss = torch.sqrt(tmp0).sum()
        return loss


def get_generalized_concurrence_pure(psi:np.ndarray):
    """
    Calculate the generalized concurrence for a pure state.

    This function computes the generalized concurrence for a given pure state
    represented by the wavefunction `psi`. The generalized concurrence is a
    measure of entanglement for multipartite quantum systems.

    Parameters:
        psi (np.ndarray): A numpy array representing the pure state wavefunction.
                        The shape of the array should correspond to the dimensions
                        of the subsystems.

    Returns:
        ret(float): The generalized concurrence of the pure state.

    References:
    - Completely entangled subspaces from Moore-like matrices http://doi.org/10.1088/1402-4896/acec15 (Equation 51)
    """
    dim_list = psi.shape
    assert all(x>=2 for x in dim_list) #TODO what if dim(i)==1
    N0 = len(dim_list)
    tmp0 = []
    psi_conj = psi.conj()
    for ind0 in _get_nontrivial_subset_list(N0):
        ind0 = np.array(ind0, dtype=np.int64)
        ind1 = np.array(sorted(set(range(N0))-set(ind0)), dtype=np.int64)
        tmp1 = np.arange(N0, dtype=np.int64)
        tmp1[ind1] += N0
        tmp2 = np.arange(N0, dtype=np.int64)
        tmp2[ind0] += N0
        tmp3 = opt_einsum.contract(psi, list(range(N0)), psi, list(range(N0,2*N0)), psi_conj, tmp1, psi_conj, tmp2, [])
        assert abs(tmp3.imag) < 1e-10
        tmp0.append(tmp3.real)
    ret = 2**(1-N0/2)*np.sqrt(max(0, 2**N0 - 2 - 2*sum(tmp0)))
    return ret


# TODO generalized concurrence for density matrix via convex roof extension
# http://doi.org/10.1088/1402-4896/acec15 (Equation 52)

import numpy as np
import scipy.linalg
import opt_einsum
import torch

import numqi._torch_op
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


def get_eof_pure(psi:np.ndarray, eps:float=1e-10):
    r'''get the entanglement of formation (EOF) of a bipartite pure state

    Parameters:
        psi (np.ndarray): a pure state, shape=(dimA,dimB)
        eps (float): a small number to avoid log(0)

    Returns:
        ret (float): the EOF of the bipartite pure state
    '''
    assert (psi.ndim==2)
    if (psi.shape[0]==1) or (psi.shape[1]==1):
        ret = 0
    else:
        if psi.shape[0]<psi.shape[1]:
            tmp0 = psi @ psi.conj().T
        else:
            tmp0 = psi.conj().T @ psi
        EVL = np.linalg.eigvalsh(tmp0)
        EVL = EVL[EVL>eps]
        ret = -np.dot(EVL, np.log(EVL))
    return ret


def get_eof_2qubit(rho:np.ndarray):
    r'''get the entanglement of formation (EOF) of a 2-qubit density matrix
    [wiki-link](https://en.wikipedia.org/wiki/Entanglement_of_formation)

    Entanglement of Formation of an Arbitrary State of Two Qubits, William K. Wootters
    [doi-link](https://doi.org/10.1103/PhysRevLett.80.2245)

    Parameters:
        rho (np.ndarray): a 2-qubit density matrix, shape=(4,4)

    Returns:
        ret (float): the EOF of the 2-qubit density matrix
    '''
    tmp0 = get_concurrence_2qubit(rho)
    if tmp0==0:
        ret = 0
    else:
        tmp1 = (1 + np.sqrt(1-tmp0*tmp0))/2
        ret = -tmp1*np.log(tmp1) - (1-tmp1)*np.log(1-tmp1)
    return ret


class EntanglementFormationModel(torch.nn.Module):
    '''Calculate the entanglement of formation (EOF) of a bipartite pure state via optimization

    Variational characterizations of separability and entanglement of formation
    [doi-link](https://doi.org/10.1103/PhysRevA.64.052304)
    '''
    def __init__(self, dimA:int, dimB:int, num_term:int, rank:int|None=None):
        r'''Initialize the model

        Parameters:
            dimA (int): the dimension of the first subsystem
            dimB (int): the dimension of the second subsystem
            num_term (int): the number of terms in the variational ansatz, `num_term` is bounded by (dimA*dimB)**2
            rank (int,None): the rank of the density matrix
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
        self.manifold = numqi.manifold.Stiefel(num_term, rank, dtype=self.cdtype, method='polar')
        self.rank = rank

        self._sqrt_rho = None
        self._eps = torch.tensor(torch.finfo(self.dtype).smallest_normal, dtype=self.dtype)
        self.contract_expr = None

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

    def forward(self):
        mat_st = self.manifold()
        rdm_not_normed = self.contract_expr(mat_st, mat_st.conj(), backend='torch')
        EVL = torch.linalg.eigvalsh(rdm_not_normed)
        tmp0 = torch.log(torch.maximum(EVL, self._eps))
        prob = torch.einsum(rdm_not_normed, [0,1,1], [0]).real
        tmp1 = torch.log(torch.maximum(prob, self._eps))
        ret = torch.dot(prob,tmp1) - torch.dot(EVL.reshape(-1), tmp0.reshape(-1))
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

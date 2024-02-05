import numpy as np
import scipy.linalg
import torch

import numqi._torch_op
import numqi.manifold

def get_concurrence_2qubit(rho):
    r'''get the concurrence of a 2-qubit density matrix
    [wiki-link](https://en.wikipedia.org/wiki/Concurrence_(quantum_computing))

    Parameters:
        rho(np.ndarray): a 2-qubit density matrix, shape=(4,4)

    Returns:
        ret(float): the concurrence of the 2-qubit density matrix
    '''
    assert (rho.shape==(4,4)) and (np.abs(rho-rho.T.conj()).max()<1e-10)
    tmp0 = np.array([-1,1,1,-1])
    z0 = (tmp0[:,np.newaxis]*tmp0) * rho[::-1,::-1].conj()
    # tmp0 = np.kron(numqi.gate.Y, numqi.gate.Y).real
    # z0_ = tmp0 @ rho.conj() @ tmp0
    # assert np.abs(z0_-z0).max() < 1e-10
    tmp0 = scipy.linalg.sqrtm(rho)
    if tmp0.dtype.name=='complex256':
        # scipy-v1.10 bug https://github.com/scipy/scipy/issues/18250
        tmp0 = tmp0.astype(np.complex128)
    EVL = np.sqrt(np.maximum(0, np.linalg.eigvalsh(tmp0 @ z0 @ tmp0)))
    ret = np.maximum(2*EVL[-1]-EVL.sum(), 0)
    return ret


def get_concurrence_pure(psi):
    r'''get the concurrence of a bipartite pure state

    Parameters:
        psi(np.ndarray): a pure state, shape=(dimA,dimB)

    Returns:
        ret(float): the concurrence of the bipartite pure state
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


def get_eof_pure(psi, eps=1e-10):
    r'''get the entanglement of formation (EOF) of a bipartite pure state

    Parameters:
        psi(np.ndarray): a pure state, shape=(dimA,dimB)
        eps(float): a small number to avoid log(0)

    Returns:
        ret(float): the EOF of the bipartite pure state
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


def get_eof_2qubit(rho):
    r'''get the entanglement of formation (EOF) of a 2-qubit density matrix
    [wiki-link](https://en.wikipedia.org/wiki/Entanglement_of_formation)

    Parameters:
        rho(np.ndarray): a 2-qubit density matrix, shape=(4,4)

    Returns:
        ret(float): the EOF of the 2-qubit density matrix
    '''
    tmp0 = get_concurrence_2qubit(rho)
    if tmp0==0:
        ret = 0
    else:
        tmp1 = (1 + np.sqrt(1-tmp0*tmp0))/2
        ret = -tmp1*np.log(tmp1) - (1-tmp1)*np.log(1-tmp1)
    return ret


class EntanglementFormationModel(torch.nn.Module):
    def __init__(self, dimA:int, dimB:int, num_term:int, rank:int=None, zero_eps=1e-10):
        # https://doi.org/10.1103/PhysRevA.64.052304
        # TODO pade approximation to avoid 0*log(0) issue
        super().__init__()
        self.dimA = dimA
        self.dimB = dimB
        if rank is None:
            rank = dimA*dimB
        self.num_term = num_term
        assert num_term>=rank
        # num_term bounded by (dimA*dimB)**2 https://doi.org/10.1103%2FPhysRevA.64.052304
        self.manifold = numqi.manifold.Stiefel(num_term, rank, dtype=torch.complex128, method='sqrtm')
        # sometimes fail when method='qr'
        self.rank = rank
        self.zero_eps = torch.tensor(zero_eps, dtype=torch.float64)
        # torch.finfo(torch.float64).eps #TODO
        self.dm0 = None
        self.EVL = None
        self.EVC = None

    def set_density_matrix(self, dm0, zero_eps=1e-10):
        assert dm0.shape == (self.dimA*self.dimB, self.dimA*self.dimB)
        assert np.abs(dm0 - dm0.T.conj()).max() < zero_eps
        assert abs(np.trace(dm0) - 1) < zero_eps
        assert np.linalg.eigvalsh(dm0)[0] > -zero_eps
        EVL,EVC = np.linalg.eigh(dm0)
        assert np.all(EVL[:(-self.rank)] < zero_eps), 'rank mismath'
        self.dm0 = torch.tensor(dm0, dtype=torch.complex128)
        self.EVL = torch.tensor(np.maximum(EVL[(-self.rank):],0), dtype=torch.float64)
        self.EVC = torch.tensor(EVC[:,(-self.rank):], dtype=torch.complex128)

    def forward(self):
        theta1 = self.manifold()
        prob = (theta1*theta1.conj()).real @ self.EVL
        psiAB = (theta1 * torch.sqrt(self.EVL)) @ self.EVC.T.conj() / torch.sqrt(prob).reshape(-1,1)
        tmp1 = psiAB.reshape(-1, self.dimA, self.dimB)
        if self.dimA <= self.dimB:
            rdm = torch.einsum(tmp1, [0,1,2], tmp1.conj(), [0,3,2], [0,1,3])
        else:
            rdm = torch.einsum(tmp1, [0,1,2], tmp1.conj(), [0,1,3], [0,2,3])
        EVL = torch.linalg.eigvalsh(rdm)
        tmp0 = torch.log(torch.maximum(EVL, self.zero_eps))
        ret = -torch.einsum(prob, [0], EVL, [0,1], tmp0, [0,1])
        return ret


class ConcurrenceModel(torch.nn.Module):
    def __init__(self, dimA:int, dimB:int, num_term:int, rank:int=None, zero_eps=1e-10):
        # https://physics.stackexchange.com/a/46509/283720
        # TODO use pade approximation to avoid sqrt(0) issue
        super().__init__()
        self.dimA = dimA
        self.dimB = dimB
        if rank is None:
            rank = dimA*dimB
        self.num_term = num_term
        assert num_term>=rank
        self.rank = rank
        # num_term bounded by (dimA*dimB)**2 https://doi.org/10.1103%2FPhysRevA.64.052304
        self.manifold = numqi.manifold.Stiefel(num_term, rank, dtype=torch.complex128, method='sqrtm')
        self.zero_eps = torch.tensor(zero_eps, dtype=torch.float64)
        self.dm0 = None
        self.EVL = None
        self.EVC = None

    def set_density_matrix(self, dm0, zero_eps=1e-10):
        assert dm0.shape == (self.dimA*self.dimB, self.dimA*self.dimB)
        assert np.abs(dm0 - dm0.T.conj()).max() < zero_eps
        assert abs(np.trace(dm0) - 1) < zero_eps
        assert np.linalg.eigvalsh(dm0)[0] > -zero_eps
        EVL,EVC = np.linalg.eigh(dm0)
        assert np.all(EVL[:(-self.rank)] < zero_eps), 'rank mismath'
        self.dm0 = torch.tensor(dm0, dtype=torch.complex128)
        self.EVL = torch.tensor(np.maximum(EVL[(-self.rank):],0), dtype=torch.float64)
        self.EVC = torch.tensor(EVC[:,(-self.rank):], dtype=torch.complex128)

    def forward(self):
        theta1 = self.manifold()
        prob = (theta1*theta1.conj()).real @ self.EVL
        psiAB = (theta1 * torch.sqrt(self.EVL)) @ self.EVC.T.conj() / torch.sqrt(prob).reshape(-1,1)
        tmp1 = psiAB.reshape(-1, self.dimA, self.dimB)
        if self.dimA <= self.dimB:
            rdm = torch.einsum(tmp1, [0,1,2], tmp1.conj(), [0,3,2], [0,1,3])
        else:
            rdm = torch.einsum(tmp1, [0,1,2], tmp1.conj(), [0,1,3], [0,2,3])
        tmp0 = rdm.reshape(rdm.shape[0], -1)
        tmp1 = 2-2*torch.einsum(tmp0, [0,1], tmp0.conj(), [0,1], [0]).real
        ret = torch.dot(prob, torch.sqrt(torch.maximum(tmp1, self.zero_eps)))
        return ret

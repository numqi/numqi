import numpy as np
import scipy.linalg
import torch

import numqi._torch_op

op_torch_logm = numqi._torch_op.TorchMatrixLogm(num_sqrtm=6, pade_order=8)


def get_concurrence_2qubit(rho):
    # https://en.wikipedia.org/wiki/Concurrence_(quantum_computing)
    assert (rho.shape==(4,4)) and (np.abs(rho-rho.T.conj()).max()<1e-10)
    tmp0 = np.array([-1,1,1,-1])
    z0 = (tmp0[:,np.newaxis]*tmp0) * rho[::-1,::-1].conj()
    # tmp0 = np.kron(numqi.gate.Y, numqi.gate.Y).real
    # z0_ = tmp0 @ rho.conj() @ tmp0
    # assert np.abs(z0_-z0).max() < 1e-10
    tmp0 = scipy.linalg.sqrtm(rho)
    EVL = np.sqrt(np.maximum(0, np.linalg.eigvalsh(tmp0 @ z0 @ tmp0)))
    ret = np.max(2*EVL[-1]-EVL.sum(), 0)
    return ret


def get_concurrence_pure(psi):
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
    # https://en.wikipedia.org/wiki/Entanglement_of_formation
    tmp0 = get_concurrence_2qubit(rho)
    tmp1 = (1 + np.sqrt(1-tmp0*tmp0))/2
    ret = -tmp1*np.log(tmp1) - (1-tmp1)*np.log(1-tmp1)
    return ret


def get_von_neumann_entropy(np0, eps=1e-10):
    assert np.abs(np0-np0.T.conj()).max() < eps
    EVL = np.linalg.eigvalsh(np0)
    ret = -np.dot(EVL, np.log(np.maximum(eps, EVL)))
    return ret


def get_eof_isotropic(dim, alpha):
    # https://doi.org/10.1007/s11704-008-0017-8
    alpha = np.asarray(alpha)
    shape = alpha.shape
    alpha = alpha.reshape(-1)
    ret = np.zeros(alpha.shape[0], dtype=np.float64)
    F = (1+alpha*dim*dim-alpha)/(dim*dim)
    ind0 = np.logical_and(F>1/dim, F<=(4*(dim-1)/(dim*dim)))
    if np.any(ind0):
        gamma = (np.sqrt(F[ind0])+np.sqrt((dim-1)*(1-F[ind0])))**2/dim
        tmp0 = -gamma*np.log(gamma) - (1-gamma)*np.log(1-gamma)
        tmp1 = (1-gamma)*np.log(dim-1)
        ret[ind0] = tmp0 + tmp1
    ind1 = F>(4*(dim-1)/(dim*dim))
    if np.any(ind1):
        ret[ind1] = dim*np.log(dim-1)*(F[ind1]-1)/(dim-2) + np.log(dim)
    ret = ret.reshape(shape)
    return ret

def get_eof_werner(dim, alpha):
    alpha = np.asarray(alpha)
    shape = alpha.shape
    alpha = alpha.reshape(-1)
    a = (1-alpha*dim) / (dim-alpha)
    ret = np.zeros(alpha.shape[0], dtype=np.float64)
    ind0 = a<0
    if np.any(ind0):
        a = a[ind0]
        tmp0 = (1-np.sqrt(1-a*a))/2
        ret[ind0] = -tmp0*np.log(tmp0) - (1-tmp0)*np.log(1-tmp0)
    ret = ret.reshape(shape)
    return ret


class EntanglementFormationModel(torch.nn.Module):
    def __init__(self, dimA, dimB, num_term):
        super().__init__()
        self.dimA = dimA
        self.dimB = dimB
        self.num_term = num_term
        # num_term bounded by (dimA*dimB)**2 https://doi.org/10.1103%2FPhysRevA.64.052304
        np_rng = np.random.default_rng()
        tmp1 = np_rng.uniform(-1, 1, size=(2,num_term,dimA*dimB))
        self.theta = torch.nn.Parameter(torch.tensor(tmp1, dtype=torch.float64))
        self.dm0 = None
        self.EVL = None
        self.EVC = None

    def set_density_matrix(self, dm0):
        assert dm0.shape == (self.dimA*self.dimB, self.dimA*self.dimB)
        assert np.abs(dm0 - dm0.T.conj()).max() < 1e-10
        assert abs(np.trace(dm0) - 1) < 1e-10
        assert np.linalg.eigvalsh(dm0)[0] > -1e-10
        EVL,EVC = np.linalg.eigh(dm0)
        self.dm0 = torch.tensor(dm0, dtype=torch.complex128)
        self.EVL = torch.tensor(np.maximum(EVL,0), dtype=torch.float64)
        self.EVC = torch.tensor(EVC, dtype=torch.complex128)

    def forward(self):
        tmp0 = torch.complex(self.theta[0], self.theta[1])
        theta1 = tmp0 @ torch.linalg.inv(numqi._torch_op.TorchPSDMatrixSqrtm.apply(tmp0.T.conj() @ tmp0))
        prob = (theta1*theta1.conj()).real @ self.EVL
        psiAB = (theta1 * torch.sqrt(self.EVL)) @ self.EVC.T.conj() / torch.sqrt(prob).reshape(-1,1)
        tmp1 = psiAB.reshape(-1, self.dimA, self.dimB)
        if self.dimA <= self.dimB:
            reduced_dm = torch.einsum(tmp1, [0,1,2], tmp1.conj(), [0,3,2], [0,1,3])
        else:
            reduced_dm = torch.einsum(tmp1, [0,1,2], tmp1.conj(), [0,1,3], [0,2,3])
        tmp2 = op_torch_logm(reduced_dm)
        tmp3 = -torch.einsum(reduced_dm, [0,1,2], tmp2, [0,2,1], [0])
        # assert torch.abs(tmp3.imag).max().item() < 1e-10
        ret = torch.dot(prob, tmp3.real)
        return ret

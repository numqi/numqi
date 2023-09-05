import math
import numpy as np
import torch
import scipy.special

import numqi.dicke
import numqi.utils

def get_qudit_H(dimD):
    tmp0 = np.exp(2j*np.pi*np.arange(dimD)/dimD)
    ret = np.vander(tmp0, dimD, increasing=True) / np.sqrt(dimD)
    return ret


def get_qudit_XZ(theta, dim, H=None):
    #period(theta)=1
    is_torch = isinstance(theta, torch.Tensor)
    if H is None:
        H = get_qudit_H(dim)
        if is_torch:
            H = torch.tensor(H, device=theta.device)
    theta = theta.T.reshape(2,-1,1)
    if is_torch:
        Xtheta,Ztheta = torch.exp(2j*np.pi*torch.arange(dim)*theta)
    else:
        Xtheta,Ztheta = np.exp(2j*np.pi*np.arange(dim)*theta)
    Hconj = H.T.conj()
    ret = (H * Xtheta[0]) @ (Hconj*Ztheta[0])
    for ind0 in range(1,Xtheta.shape[0]):
        ret = (H * Xtheta[ind0]) @ (Hconj*Ztheta[ind0]) @ ret
    return ret


def get_dicke_klist(dim, num_qudit):
    hf0 = lambda d,n: [(n,)] if (d<=1) else [(x,)+y for x in range(n+1) for y in hf0(d-1,n-x)]
    klist = hf0(dim, num_qudit)
    return klist


def get_mps_dicke_transform_matrix(dim, num_qudit, num_more_space=1.05, seed=None):
    np_rng = np.random.default_rng(seed)
    num_dicke = int(scipy.special.binom(num_qudit+dim-1, dim-1))
    num_mps = max(int(num_dicke*num_more_space), num_dicke+3)
    mps_alpha = np_rng.normal(size=(num_mps,dim*2)).astype(np.float64, copy=False).view(np.complex128)
    mps_alpha /= np.linalg.norm(mps_alpha, axis=1, keepdims=True)

    klist = np.array(get_dicke_klist(dim, num_qudit))
    binom_term = np.sqrt(get_klist_binom_term(klist))
    matA = np.prod(mps_alpha.reshape(num_mps, 1, dim)**(klist.reshape(1,num_dicke,dim)), axis=2) * binom_term
    U,S,V = np.linalg.svd(matA, full_matrices=False)
    # matA = (U*S) @ V
    assert S[-1]>1e-7, 'not full rank, generate mps again'
    matB = (V.T.conj()/S) @ U.T.conj()
    # assert np.abs(matB @ matA - np.eye(space_dim)).max() < 1e-7
    return mps_alpha,klist,matA,matB


def get_klist_binom_term(klist):
    klist_np = np.array(klist)
    num_qudit = klist_np[0].sum()
    assert np.all(klist_np.sum(axis=1)==num_qudit)
    tmp0 = num_qudit - np.cumsum(np.pad(klist_np, [(0,0),(1,0)], mode='constant', constant_values=0)[:,:-1], axis=1)
    tmp1 = [[(int(y0),int(y1)) for y0,y1 in zip(x0,x1)] for x0,x1 in zip(klist_np,tmp0)]
    tmp2 = {y for x in tmp1 for y in x}
    tmp3 = {x:scipy.special.binom(x[1],x[0]) for x in tmp2}
    ret = np.prod(np.array([[tmp3[y] for y in x] for x in tmp1]), axis=1)
    return ret


def mps_to_dicke(mps_p, mps_alpha, klist, binom_term=None):
    is_torch = isinstance(mps_p, torch.Tensor)
    dimA,num_mps = mps_p.shape
    num_dicke,dim = klist.shape
    num_qudit = klist[0].sum()
    if binom_term is None:
        tmp0 = klist.detach().numpy if is_torch else klist
        binom_term = np.sqrt(get_klist_binom_term(tmp0))
    tmp0 = mps_alpha.reshape(num_mps,1,dim)**(klist.reshape(1,num_dicke,dim))
    if is_torch:
        tmp0 = torch.nan_to_num(tmp0.real, nan=1) + 1j*torch.nan_to_num(tmp0.imag, nan=0)
        tmp1 = torch.prod(tmp0, dim=2) #(0+0j)**0 is nan (not a number)
    else:
        tmp1 = np.prod(np.nan_to_num(tmp0, nan=1), axis=2)
        tmp1 = np.prod(mps_alpha.reshape(num_mps,1,dim)**(klist.reshape(1,num_dicke,dim)), axis=2)
    dicke_p = (mps_p @ tmp1)*binom_term
    return dicke_p

def dicke_to_mps(dicke_p, matB, mps_basis_alpha):
    mps_p,mps_alpha = dicke_p @ matB, mps_basis_alpha
    return mps_p,mps_alpha


def mps_dicke_single_gate(mps_p, mps_alpha, UA, UB):
    ret = UA @ mps_p, mps_alpha @ UB.T
    return ret


def mps_dicke_cnot(dicke_p, klist_permutation_index):
    is_torch = isinstance(dicke_p, torch.Tensor)
    tmp0 = [x[y] for x,y in zip(dicke_p, klist_permutation_index)]
    if is_torch:
        ret = torch.stack(tmp0)
    else:
        ret = np.stack(tmp0)
    return ret


def get_klist_permutation_index(klist):
    klist_dict = {y:x for x,y in enumerate(klist)}
    klist_np = np.array(klist)
    klist_permutation_index = [slice(None)]
    for ind0 in range(1, len(klist[0])):
        tmp0 = np.roll(klist_np, ind0, axis=1).tolist()
        klist_permutation_index.append(np.array([klist_dict[tuple(x)] for x in tmp0]))
    return klist_permutation_index


class QuantumPureBosonicExt(torch.nn.Module):
    def __init__(self, dimA, dimB, num_kext, num_layer, num_XZ=None):
        super().__init__()
        num_XZ = max(math.ceil((dimB*dimB-1)/2), 3) if num_XZ is None else num_XZ
        mps_basis_alpha,klist_np,matA,matB = get_mps_dicke_transform_matrix(dimB, num_kext)
        self.klist = [tuple(x) for x in klist_np.tolist()]
        self.klist_np = klist_np
        self.klist_torch = torch.tensor(klist_np, dtype=torch.int64)
        self.mps_basis_alpha = torch.tensor(mps_basis_alpha, dtype=torch.complex128)
        self.matA = torch.tensor(matA, dtype=torch.complex128)
        self.matB = torch.tensor(matB, dtype=torch.complex128)
        tmp0 = get_klist_permutation_index(self.klist)
        self.klist_permutation_index = [tmp0[0]] + [torch.tensor(x, dtype=torch.int64) for x in tmp0[1:]]
        tmp0 = np.random.rand(num_layer, 2, num_XZ, 2)
        self.theta = torch.nn.Parameter(torch.tensor(tmp0, dtype=torch.float64))

        self.mps_p0 = torch.tensor([1]+[0]*(dimA-1), dtype=torch.complex128).view(dimA,1)
        self.mps_alpha0 = torch.tensor([1]+[0]*(dimB-1), dtype=torch.complex128).view(1,dimB)
        self.weylH = torch.tensor(get_qudit_H(dimB), dtype=torch.complex128)

        self.binom_term = torch.tensor(np.sqrt(get_klist_binom_term(klist_np)), dtype=torch.float64)
        Bij = numqi.dicke.get_partial_trace_ABk_to_AB_index(num_kext, dimB)
        tmp0 = [torch.int64,torch.int64,torch.complex128]
        self.Bij = [[torch.tensor(y0,dtype=y1) for y0,y1 in zip(x,tmp0)] for x in Bij]
        self.dm_torch = None
        self.dm_target = None
        self.expect_op_T_vec = None

    def set_dm_target(self, target):
        assert target.ndim in {1,2}
        if target.ndim==1:
            target = target[:,np.newaxis] * target.conj()
        assert (target.shape[0]==target.shape[1])
        self.dm_target = torch.tensor(target, dtype=torch.complex128)

    def set_expectation_op(self, op):
        self.dm_target = None
        self.expect_op_T_vec = torch.tensor(op.T.reshape(-1), dtype=torch.complex128)

    def forward(self):
        num_layer = self.theta.shape[0]
        mps_p = self.mps_p0
        mps_alpha = self.mps_alpha0
        dimA = mps_p.shape[0]
        dimB = mps_alpha.shape[1]
        for ind0 in range(num_layer):
            UA = get_qudit_XZ(self.theta[ind0,0], dimA, H=self.weylH)
            UB = get_qudit_XZ(self.theta[ind0,1], dimB, H=self.weylH)
            mps_p, mps_alpha = mps_dicke_single_gate(mps_p, mps_alpha, UA, UB)
            dicke_p = mps_to_dicke(mps_p, mps_alpha, self.klist_torch, self.binom_term)
            dicke_p = mps_dicke_cnot(dicke_p, self.klist_permutation_index)
            mps_p,mps_alpha = dicke_to_mps(dicke_p, self.matB, self.mps_basis_alpha)
        self.dm_torch = numqi.dicke.partial_trace_ABk_to_AB(dicke_p, self.Bij)
        if self.dm_target is not None:
            loss = numqi.utils.get_relative_entropy(self.dm_target, self.dm_torch)
        else:
            loss = torch.dot(self.dm_torch.view(-1), self.expect_op_T_vec).real
        return loss

# mps_p (complex128,(NA,N0))
# mps_alpha (complex128,(N0,d))
# klist (int64,(N1,d))
# binom_term (float64,N1)
# dicke_p (complex128,(NA,N1))
# matA (complex128,(N2,N1))
# matB (complex128,(N1,N2))

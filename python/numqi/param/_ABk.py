import itertools
import numpy as np
import scipy.sparse
import torch
import collections

class ABkHermitian(torch.nn.Module):
    def __init__(self, dimA, dimB, kext, dtype=torch.float64, device='cpu'):
        super().__init__()
        np_rng = np.random.default_rng()
        self.index_sym = torch.tensor(ABk_symmetry_index(dimA, dimB, kext), dtype=torch.int64, device=device)
        index_plus,index_minus = ABk_skew_symmetry_index(dimA, dimB, kext)
        self.index_plus = torch.tensor(index_plus, dtype=torch.int64, device=device)
        self.index_minus = torch.tensor(index_minus, dtype=torch.int64, device=device)
        tmp0 = np_rng.uniform(-1, 1, size=self.index_sym.max().item()+1)
        self.theta_sym = torch.nn.Parameter(torch.tensor(tmp0, dtype=dtype, device=device))
        tmp0 = np_rng.uniform(-1, 1, size=self.index_plus.max().item()+1)
        self.theta_skew_sym = torch.nn.Parameter(torch.tensor(tmp0, dtype=dtype, device=device))
        self.c_dtype = torch.complex64 if dtype==torch.float32 else torch.complex128

    def forward(self):
        tmp0 = (self.theta_sym).to(self.c_dtype)[self.index_sym]
        tmp1 = 1j*self.theta_skew_sym.to(self.c_dtype)
        tmp2 = ABk_skew_symmetry_index_to_full(tmp1, self.index_plus, self.index_minus)
        ret = tmp0 + tmp2
        return ret

class ABk2localHermitian(torch.nn.Module):
    def __init__(self, dimA, dimB, kext, dtype=torch.float64, device='cpu'):
        super().__init__()
        np_rng = np.random.default_rng()
        tmp0 = np_rng.uniform(-1, 1, size=(dimA*dimB, dimA*dimB))
        self.matAB_real = torch.nn.Parameter(torch.tensor(tmp0, dtype=dtype, device=device))
        coeff_sym,index_sym = ABk_2local_symmetry_index(dimA, dimB, kext)
        self.coeff_sym = torch.tensor(coeff_sym.toarray(), dtype=dtype, device=device)#.to_sparse_csr()
        self.index_sym = torch.tensor(index_sym, dtype=torch.int64, device=device)
        coeff_skew_sym,index_skew_sym = ABk_2local_skew_symmetry_index(dimA, dimB, kext)
        self.coeff_skew_sym = torch.tensor(coeff_skew_sym.toarray(), dtype=dtype, device=device)#.to_sparse_csr()
        self.index_skew_sym = torch.tensor(index_skew_sym, dtype=torch.int64, device=device)
        self.tril_index0 = torch.triu_indices(dimA*dimB, dimA*dimB, offset=0, device=device)
        self.tril_index1 = torch.triu_indices(dimA*dimB, dimA*dimB, offset=1, device=device)
        self.c_dtype = torch.complex64 if dtype==torch.float32 else torch.complex128

    def forward(self):
        tmp0 = self.matAB_real[self.tril_index0[0], self.tril_index0[1]]
        tmp1 = ABk_2local_index_to_full(tmp0, self.coeff_sym, self.index_sym).to(self.c_dtype)
        tmp2 = self.matAB_real.T[self.tril_index1[0], self.tril_index1[1]]
        tmp3 = ABk_2local_index_to_full(tmp2, self.coeff_skew_sym, self.index_skew_sym).to(self.c_dtype)
        ret = tmp1 + tmp3*1j
        return ret

    def to_AB(self):
        tmp0 = self.matAB_real.detach().cpu().numpy().copy()
        tmp2 = np.triu(tmp0)
        tmp3 = np.tril(tmp0, k=-1).T
        ret = tmp2 + tmp2.T - np.diag(np.diag(tmp2)) + 1j*(tmp3 - tmp3.T)
        return ret

def ABk_permutate(mat, ind0, ind1, dimA, dimB, kext):
    tmp0 = [dimA] + [dimB]*kext + [dimA] + [dimB]*kext
    tmp1 = list(range(2*kext+2))
    tmp1[ind0+1],tmp1[ind1+1] = tmp1[ind1+1],tmp1[ind0+1]
    tmp1[kext+1+ind0+1],tmp1[kext+1+ind1+1] = tmp1[kext+1+ind1+1],tmp1[kext+1+ind0+1]
    ret = mat.reshape(tmp0).transpose(tmp1).reshape(mat.shape)
    return ret


def ABk_symmetry_index(dimA, dimB, kext):
    index_to_set = np.arange((dimA*dimB**kext)**2, dtype=np.int64).reshape(dimA*dimB**kext, -1)
    index_to_set = np.minimum(index_to_set, index_to_set.T)
    tmp0 = [(x,y) for x in range(kext) for y in range(x+1,kext)]
    for ind0,ind1 in tmp0:
        index_to_set = np.minimum(index_to_set, ABk_permutate(index_to_set, ind0, ind1, dimA, dimB, kext))
    tmp0 = index_to_set.reshape(-1)
    tmp1 = np.unique(tmp0)
    tmp2 = -np.ones(tmp1.max()+1, dtype=np.int64)
    tmp2[tmp1] = np.arange(tmp1.shape[0])
    index = tmp2[index_to_set]
    # num_parameter = index.max() + 1
    return index


def ABk_skew_symmetry_index(dimA, dimB, kext):
    assert kext>=1
    tmp0 = dimA*dimB**kext
    index_to_set = np.arange(tmp0**2, dtype=np.int64).reshape(tmp0,tmp0)
    index_to_set = np.minimum(index_to_set, index_to_set.T)
    np.fill_diagonal(index_to_set, 0) #0 is for 0
    tmp0 = np.arange(tmp0)
    factor_set = (2 * (tmp0[:,np.newaxis]>tmp0).astype(np.int32) - 1)*(tmp0[:,np.newaxis]!=tmp0)
    tmp0 = [(x,y) for x in range(kext) for y in range(x+1,kext)]
    for ind0,ind1 in tmp0:
        tmp0 = factor_set + ABk_permutate(factor_set, ind0, ind1, dimA, dimB, kext)
        factor_set = (tmp0>0).astype(np.int64) - (tmp0<0).astype(np.int64)
        index_to_set = np.minimum(index_to_set, ABk_permutate(index_to_set, ind0, ind1, dimA, dimB, kext))
        index_to_set[factor_set==0] = 0
    tmp0 = index_to_set.reshape(-1)
    tmp1 = np.unique(tmp0)
    tmp2 = -np.ones(tmp1.max()+1, dtype=np.int64)
    tmp2[tmp1] = np.arange(tmp1.shape[0])
    index_plus = tmp2[np.maximum(0, index_to_set*factor_set)]
    index_minus = tmp2[np.maximum(0, -index_to_set*factor_set)]
    # num_parameter = index_plus.max()
    return index_plus, index_minus


def ABk_skew_symmetry_index_to_full(vector, index_plus, index_minus):
    if isinstance(vector, torch.Tensor):
        tmp0 = torch.zeros([1], dtype=vector.dtype, device=vector.device)
        tmp1 = torch.concat([tmp0, vector], dim=0)
        ret = tmp1[index_plus] - tmp1[index_minus]
    else:
        tmp0 = np.concatenate([np.zeros([1]), vector], axis=0)
        ret = tmp0[index_plus] - tmp0[index_minus]
    return ret


def unique_index_set(index, offset=0):
    tmp0 = np.unique(index.reshape(-1))
    tmp1 = -np.ones(tmp0.max()+1+offset, dtype=np.int32)
    tmp1[tmp0] = np.arange(len(tmp0)) + offset
    ret = tmp1[index]
    return ret


def ABk_2local_symmetry_index(dimA, dimB, kext):
    assert kext>=1
    num_row = dimA*dimB**kext
    tmp0 = np.arange(1,1+(dimA*dimB)**2, dtype=np.int32).reshape(dimA*dimB, -1)
    index_to_set_AB = unique_index_set(np.minimum(tmp0, tmp0.T), offset=1)
    if kext==1:
        coeff = np.eye(index_to_set_AB.max())
        coeff = scipy.sparse.csr_matrix(coeff)
        index = index_to_set_AB-1
    else:
        ijvalue = []
        for ind0 in range(kext):
            index_ABi = index_to_set_AB
            if ind0>0:
                tmp0 = np.eye(dimB**ind0,dtype=np.int32).reshape(dimB**ind0,1,1,-1,1)
                index_ABi = scipy.sparse.csr_matrix((index_ABi.reshape(dimA, 1, dimB, dimA, 1, dimB) * tmp0).reshape(dimA*dimB**(ind0+1),-1))
            if ind0<kext-1:
                tmp0 = scipy.sparse.eye(dimB**(kext-1-ind0), dtype=np.int32, format='csr')
                index_ABi = scipy.sparse.kron(index_ABi, tmp0, format='csr')
            ijvalue.append(np.stack(scipy.sparse.find(index_ABi)))
        ijvalue = np.concatenate(ijvalue, axis=1)
        ijvalue = ijvalue[:,np.lexsort(ijvalue[::-1])].T
        tmp0 = collections.Counter(tuple(x) for x in ijvalue.tolist())
        z0 = {x:tuple((z[0][2],z[1]) for z in y) for x,y in itertools.groupby(tmp0.items(), lambda x: x[0][:2])}
        hf0 = lambda x: x[1]
        z1 = {x:np.array([z[0][0]*num_row+z[0][1] for z in y]) for x,y in  itertools.groupby(sorted(z0.items(), key=hf0), key=hf0)}
        z2 = {y:x for x,y in enumerate(sorted(z1.keys()), start=1)}
        coeff = np.zeros((len(z2)+1,index_to_set_AB.max()), dtype=np.int32)
        for x,y in z1.items():
            coeff[z2[x], [z[0]-1 for z in x]] = [z[1] for z in x]
        coeff = scipy.sparse.csr_matrix(coeff)
        index = np.zeros(num_row*num_row, dtype=np.int32)
        for x,y in z1.items():
            index[y] = z2[x]
        index = index.reshape(num_row, num_row)
    # num_parameter=coeff.shape[1]
    return coeff, index

def ABk_2local_index_to_full(parameter, coeff, index):
    ret = (coeff @ parameter)[index]
    return ret


def ABk_2local_skew_symmetry_index(dimA, dimB, kext):
    assert kext>=1
    num_row = dimA*dimB**kext
    tmp0 = np.arange(1,1+(dimA*dimB)**2, dtype=np.int32).reshape(dimA*dimB, -1)
    np.fill_diagonal(tmp0, 0) #0 is for 0
    index_to_set_AB = unique_index_set(np.minimum(tmp0, tmp0.T), offset=0)
    tmp0 = np.arange(index_to_set_AB.shape[0])
    index_to_set_AB *= 2*(tmp0[:,np.newaxis]>tmp0)-1

    if kext==1:
        tmp0 = index_to_set_AB.reshape(-1)
        coeff = np.zeros((num_row**2, index_to_set_AB.max()), dtype=np.int32)
        tmp1 = tmp0!=0
        tmp2 = tmp0[tmp1]
        coeff[np.arange(num_row**2)[tmp1], np.abs(tmp2)-1] = 2*(tmp2>0)-1
        coeff = scipy.sparse.csr_matrix(coeff)
        index = np.arange(num_row**2).reshape(num_row,num_row)
        coeff = -coeff
    else:
        ijvalue = []
        for ind0 in range(kext):
            index_ABi = index_to_set_AB
            if ind0>0:
                tmp0 = np.eye(dimB**ind0,dtype=np.int32).reshape(dimB**ind0,1,1,-1,1)
                index_ABi = scipy.sparse.csr_matrix((index_ABi.reshape(dimA, 1, dimB, dimA, 1, dimB) * tmp0).reshape(dimA*dimB**(ind0+1),-1))
            if ind0<kext-1:
                tmp0 = scipy.sparse.eye(dimB**(kext-1-ind0), dtype=np.int32, format='csr')
                index_ABi = scipy.sparse.kron(index_ABi, tmp0, format='csr')
            ijvalue.append(np.stack(scipy.sparse.find(index_ABi)))
        ijvalue = np.concatenate(ijvalue, axis=1)
        ijvalue = ijvalue[:,np.lexsort(ijvalue[::-1])].T

        tmp0 = collections.Counter(tuple(x) for x in ijvalue.tolist())
        z0 = {x:tuple((z[0][2],z[1]) for z in y) for x,y in itertools.groupby(tmp0.items(), lambda x: x[0][:2])}
        hf0 = lambda x: x[1]
        z1 = {x:np.array([z[0][0]*num_row+z[0][1] for z in y]) for x,y in  itertools.groupby(sorted(z0.items(), key=hf0), key=hf0)}
        z2 = {y:x for x,y in enumerate(sorted(z1.keys()), start=1)}
        coeff = np.zeros((len(z2)+1,index_to_set_AB.max()), dtype=np.int32)
        for x,y in z1.items():
            tmp0 = [(abs(z[0]),np.sign(z[0])*z[1]) for z in x]
            tmp1 = [(z0,sum(z2 for _,z2 in z1)) for z0,z1 in itertools.groupby(tmp0, key=lambda z:z[0])]
            coeff[z2[x], [z[0]-1 for z in tmp1]] = [z[1] for z in tmp1]
        coeff = -coeff
        coeff = scipy.sparse.csr_matrix(coeff)
        index = np.zeros(num_row*num_row, dtype=np.int32)
        for x,y in z1.items():
            index[y] = z2[x]
        index = index.reshape(num_row, num_row)
    # num_parameter=coeff.shape[1]
    return coeff,index

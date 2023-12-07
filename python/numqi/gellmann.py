import numpy as np
import functools
import itertools
import torch

def gellmann_matrix(i, j, d):
    # https://en.wikipedia.org/wiki/Gell-Mann_matrices
    # https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices
    # https://github.com/CQuIC/pysme/blob/master/src/pysme/gellmann.py
    # https://arxiv.org/pdf/0806.1174.pdf
    assert (d>0) and (0<=i<d) and (0<=j<d)
    if i > j:
        ind0 = [i, j]
        ind1 = [j, i]
        data = [1j, -1j]
    elif j > i:
        ind0 = [i, j]
        ind1 = [j, i]
        data = [1, 1]
    elif i==j and i==0:
        ind0 = np.arange(d)
        ind1 = ind0
        data = np.ones(d)*(np.sqrt(2/d))
        # norm factor sqrt(2/d) is chosen such that Tr(Mi Mj)=2 delta_ij
    else:
        ind0 = np.arange(i+1)
        ind1 = ind0
        data = np.sqrt(2/(i*(i+1)))*np.array([1]*i + [-i])
    ret = np.zeros((d, d), dtype=np.complex128)
    ret[ind0, ind1] = data
    return ret


@functools.lru_cache
def _all_gellmann_matrix_cache(d, tensor_n, with_I):
    sym_mat = [gellmann_matrix(i,j,d) for i in range(d) for j in range(i+1,d)]
    antisym_mat = [gellmann_matrix(j,i,d) for i in range(d) for j in range(i+1,d)]
    diag_mat = [gellmann_matrix(i,i,d) for i in range(1,d)]
    tmp0 = [gellmann_matrix(0, 0, d)]
    ret = np.stack(sym_mat+antisym_mat+diag_mat+tmp0, axis=0)
    if tensor_n>1:
        tmp0 = [list(range(d**2))]*tensor_n
        ret = np.stack([functools.reduce(np.kron, [ret[y] for y in x]) for x in itertools.product(*tmp0)])
    if not with_I:
        ret = ret[:-1]
    return ret


def all_gellmann_matrix(d, /, tensor_n=1, with_I=True):
    # the last item is identity
    d = int(d)
    tensor_n = int(tensor_n)
    with_I = bool(with_I)
    assert (d>=2) and (tensor_n>=1)
    ret = _all_gellmann_matrix_cache(d, tensor_n, with_I)
    return ret


def matrix_to_gellmann_basis(A, norm_I='sqrt(2/d)'):
    shape0 = A.shape
    N0 = shape0[-1]
    assert norm_I in {'1/d','sqrt(2/d)'}
    factor_I = (1/N0) if norm_I=='1/d' else 1/np.sqrt(2*N0)
    assert len(shape0)>=2 and shape0[-2]==N0
    A = A.reshape(-1,N0,N0)
    if isinstance(A, torch.Tensor):
        indU0,indU1 = torch.triu_indices(N0, N0, offset=1)
        aS = (A + A.transpose(1,2))[:,indU0,indU1]/2
        aA = (A - A.transpose(1,2))[:,indU0,indU1] * (0.5j)
        tmp0 = torch.diagonal(A, dim1=1, dim2=2)
        tmp1 = torch.sqrt(2*torch.arange(1,N0,dtype=torch.float64)*torch.arange(2,N0+1))
        aD = (torch.cumsum(tmp0,dim=1)[:,:-1] - torch.arange(1,N0)*tmp0[:,1:])/tmp1
        aI = torch.einsum(A, [0,1,1], [0]) * factor_I
        ret = torch.concat([aS,aA,aD,aI.view(-1,1)], dim=1)
    else:
        indU0,indU1 = np.triu_indices(N0,1)
        aS = (A + A.transpose(0,2,1))[:,indU0,indU1]/2
        aA = (A - A.transpose(0,2,1))[:,indU0,indU1] * (0.5j)
        tmp0 = np.diagonal(A, axis1=1, axis2=2)
        tmp1 = np.sqrt(2*np.arange(1,N0)*np.arange(2,N0+1))
        aD = (np.cumsum(tmp0,axis=1)[:,:-1] - np.arange(1,N0)*tmp0[:,1:])/tmp1
        aI = np.trace(A, axis1=1, axis2=2) * factor_I
        ret = np.concatenate([aS,aA,aD,aI[:,np.newaxis]], axis=1)
    if len(shape0)==2:
        ret = ret[0]
    else:
        ret = ret.reshape(*shape0[:-2], -1)
    return ret


def gellmann_basis_to_matrix(vec, norm_I='sqrt(2/d)'):
    r'''convert a vector in Gell-Mann basis to a matrix

    ordering: PauliX, PauliY, PauliZ, I

    Parameters:
        vec (np.ndarray,torch.Tensor): vector in Gell-Mann basis
        norm_I (str): normalization of identity matrix, '1/d' or 'sqrt(2/d)'

    Returns:
        ret (np.ndarray,torch.Tensor): matrix
    '''
    shape = vec.shape
    vec = vec.reshape(-1, shape[-1])
    N0 = vec.shape[0]
    N1 = int(np.sqrt(vec.shape[1]))
    assert norm_I in {'1/d','sqrt(2/d)'}
    # 'sqrt(2/d)' tr(Mi Mj)= 2 delta_ij
    factor_I = 1 if norm_I=='1/d' else np.sqrt(2/N1)
    vec0 = vec[:,:(N1*(N1-1)//2)]
    vec1 = vec[:,(N1*(N1-1)//2):(N1*(N1-1))]
    vec2 = vec[:,(N1*(N1-1)):-1]
    vec3 = vec[:,-1:] * factor_I
    assert vec.shape[1]==N1*N1
    if isinstance(vec, torch.Tensor):
        indU0,indU1 = torch.triu_indices(N1,N1,1)
        indU01 = torch.arange(N1*N1).reshape(N1,N1)[indU0,indU1]
        ind0 = torch.arange(N1)
        indU012 = (((N1*N1)*torch.arange(N0).view(-1,1)) + indU01).view(-1)
        zero0 = torch.zeros(N0*N1*N1, dtype=torch.complex128)
        ret0 = torch.scatter(zero0, 0, indU012, (vec0 - 1j*vec1).view(-1)).reshape(N0, N1, N1)
        ret1 = torch.scatter(zero0, 0, indU012, (vec0 + 1j*vec1).view(-1)).reshape(N0, N1, N1).transpose(1,2)
        tmp0 = torch.sqrt(torch.tensor(2,dtype=torch.float64)/(ind0[1:]*(ind0[1:]+1)))
        tmp1 = torch.concat([tmp0*vec2, vec3], axis=1)
        ret2 = torch.diag_embed(tmp1 @ ((ind0.view(-1,1)>=ind0) + torch.diag(-ind0[1:], diagonal=1)).to(tmp1.dtype))
        ret = ret0 + ret1 + ret2
    else:
        ret = np.zeros((N0,N1,N1), dtype=np.complex128)
        indU0,indU1 = np.triu_indices(N1,1)
        # indL0,indL1 = np.tril_indices(N1,-1)
        ind0 = np.arange(N1, dtype=np.int64)
        ret[:,indU0,indU1] = vec0 - 1j*vec1
        tmp0 = np.zeros_like(ret)
        tmp0[:,indU0,indU1] = vec0 + 1j*vec1
        ret += tmp0.transpose(0,2,1)
        tmp1 = np.concatenate([np.sqrt(2/(ind0[1:]*(ind0[1:]+1)))*vec2, vec3], axis=1)
        ret[:,ind0,ind0] = tmp1 @ ((ind0[:,np.newaxis]>=ind0) + np.diag(-ind0[1:], k=1))
    ret = ret[0] if (len(shape)==1) else ret.reshape(*shape[:-1], N1, N1)
    return ret


def dm_to_gellmann_basis(dm, with_rho0=False):
    ret = matrix_to_gellmann_basis(dm)
    if not with_rho0:
        shape = ret.shape
        tmp0 = ret.reshape(-1, shape[-1])[:,:-1].real
        ret = tmp0[0] if (len(shape)==1) else tmp0.reshape(*shape[:-1], -1)
    return ret


def gellmann_basis_to_dm(vec):
    shape = vec.shape
    N0 = int(round(np.sqrt(shape[-1]+1).item()))
    assert shape[-1]==N0*N0-1
    vec = vec.reshape(-1, shape[-1])
    if isinstance(vec, torch.Tensor):
        tmp0 = torch.concat([vec, torch.ones(vec.shape[0], 1, dtype=torch.float64)/N0], axis=1)
    else:
        tmp0 = np.concatenate([vec, np.ones((vec.shape[0],1), dtype=np.float64)/N0], axis=1)
    ret = gellmann_basis_to_matrix(tmp0, norm_I='1/d')
    ret = ret[0] if (len(shape)==1) else ret.reshape(*shape[:-1], *ret.shape[-2:])
    return ret


def dm_to_gellmann_norm(dm):
    shape = dm.shape
    assert shape[-1]==shape[-2]
    N0 = dm.shape[-1]
    dm = dm.reshape(-1, N0, N0)
    tmp0 = (np.trace(dm, axis1=1, axis2=2)/N0).reshape(-1,1,1) * np.eye(N0)
    ret = np.linalg.norm((dm - tmp0).reshape(-1,N0*N0), ord=2, axis=1)/np.sqrt(2)
    ret = ret.item() if (len(shape)==2) else ret.reshape(*shape[:-2])
    ## the following code is equivalent to the above one
    # tmp0 = dm_to_gellmann_basis(dm)
    # shape = tmp0.shape
    # tmp1 = tmp0.reshape(-1, shape[-1])
    # if isinstance(dm, torch.Tensor):
    #     ret = torch.linalg.norm(tmp1, dim=1)
    # else:
    #     ret = np.linalg.norm(tmp1, axis=1)
    # ret = ret.item() if (len(shape)==1) else ret.reshape(*shape[:-1])
    return ret

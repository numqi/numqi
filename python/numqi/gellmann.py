import numpy as np
import functools
import itertools
import torch

def gellmann_matrix(i:int, j:int, d:int):
    r'''get the Gell-Mann matrix
    [wiki-link/Gell-Mann-matrices](https://en.wikipedia.org/wiki/Gell-Mann_matrices)
    [wiki-link/Generalizations-of-Pauli-matrices](https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices)
    [github-link/CQuIC/pysme](https://github.com/CQuIC/pysme/blob/master/src/pysme/gellmann.py)
    [arxiv-link/0806.1174](https://arxiv.org/pdf/0806.1174.pdf)

    normalization condition: Tr(Mi Mj)=2 delta_ij

    Parameters:
        i (int): row indices of Gell-Mann matrix, 0<=i<d
        j (int): column indices of Gell-Mann matrix, 0<=j<d
        d (int): dimension of Gell-Mann matrix

    Returns:
        ret (np.ndarray): Gell-Mann matrix. if `i<j`, Pauli-X like; if `i>j`, Pauli-Y like; if `i=j=0`, identity; if `i=j>0`, Pauli-Z like
    '''
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


def all_gellmann_matrix(d:int, /, tensor_n:int=1, with_I:bool=True):
    r'''get all Gell-Mann matrices

    Parameters:
        d (int): dimension of Gell-Mann matrix
        tensor_n (int): tensor product of Gell-Mann matrices
        with_I (bool): include identity matrix or not. if `True`, the last item is identity matrix.

    Returns:
        ret (np.ndarray): Gell-Mann matrices. Ordering: PauliX, PauliY, PauliZ, I.
            if `tensor_n=1`, shape=(d**2, d, d);
            if `tensor_n>1`, shape=(d**(2n), d**n, d**n)
    '''
    d = int(d)
    tensor_n = int(tensor_n)
    with_I = bool(with_I)
    assert (d>=2) and (tensor_n>=1)
    ret = _all_gellmann_matrix_cache(d, tensor_n, with_I)
    return ret


def matrix_to_gellmann_basis(A:np.ndarray|torch.Tensor):
    r'''convert a matrix to Gell-Mann basis

    ordering: PauliX, PauliY, PauliZ, I

    Parameters:
        A (np.ndarray,torch.Tensor): matrix, support batch, `shape=(...,d,d)`

    Returns:
        ret (np.ndarray,torch.Tensor): vector in Gell-Mann basis, `shape=(...,d**2)`
    '''
    shape0 = A.shape
    N0 = shape0[-1]
    factor_I = 1/np.sqrt(2*N0)
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


def gellmann_basis_to_matrix(vec:np.ndarray|torch.Tensor):
    r'''convert a vector in Gell-Mann basis to a matrix

    ordering: PauliX, PauliY, PauliZ, I

    Parameters:
        vec (np.ndarray,torch.Tensor): vector in Gell-Mann basis, `shape=(...,d**2)`

    Returns:
        ret (np.ndarray,torch.Tensor): matrix, `shape=(...,d,d)`
    '''
    shape = vec.shape
    vec = vec.reshape(-1, shape[-1])
    N0 = vec.shape[0]
    N1 = int(np.sqrt(vec.shape[1]))
    # 'sqrt(2/d)' tr(Mi Mj)= 2 delta_ij
    factor_I = np.sqrt(2/N1)
    vec0 = vec[:,:(N1*(N1-1)//2)]
    vec1 = vec[:,(N1*(N1-1)//2):(N1*(N1-1))]
    vec2 = vec[:,(N1*(N1-1)):-1]
    vec3 = vec[:,-1:] * factor_I
    assert vec.shape[1]==N1*N1
    if isinstance(vec, torch.Tensor):
        isfloat32 = vec1.dtype in (torch.float32,torch.complex64)
        indU0,indU1 = torch.triu_indices(N1,N1,1)
        indU01 = torch.arange(N1*N1).reshape(N1,N1)[indU0,indU1]
        ind0 = torch.arange(N1)
        indU012 = (((N1*N1)*torch.arange(N0).view(-1,1)) + indU01).view(-1)
        zero0 = torch.zeros(N0*N1*N1, dtype=(torch.complex64 if isfloat32 else torch.complex128))
        ret0 = torch.scatter(zero0, 0, indU012, (vec0 - 1j*vec1).view(-1)).reshape(N0, N1, N1)
        ret1 = torch.scatter(zero0, 0, indU012, (vec0 + 1j*vec1).view(-1)).reshape(N0, N1, N1).transpose(1,2)
        tmp0 = torch.sqrt(torch.tensor(2,dtype=(torch.float32 if isfloat32 else torch.float64))/(ind0[1:]*(ind0[1:]+1)))
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


def dm_to_gellmann_basis(dm:np.ndarray|torch.Tensor, with_rho0:bool=False):
    r'''convert density matrix to a vector in Gell-Mann basis

    Parameters:
        dm (np.ndarray,torch.Tensor): density matrix, 2d array (support batch)
        with_rho0 (bool): include rho0 or not. if `True`, the last item is not included.

    Returns:
        ret (np.ndarray,torch.Tensor): real vector in Gell-Mann basis, `shape=(...,d**2-1)` if `with_rho0=False`, otherwise `shape=(...,d**2)`
    '''
    ret = matrix_to_gellmann_basis(dm).real
    if not with_rho0:
        shape = ret.shape
        tmp0 = ret.reshape(-1, shape[-1])[:,:-1]
        ret = tmp0[0] if (len(shape)==1) else tmp0.reshape(*shape[:-1], -1)
    return ret


def gellmann_basis_to_dm(vec:np.ndarray|torch.Tensor):
    r'''convert vector in Gell-Mann basis to a density matrix

    Parameters:
        vec (np.ndarray,torch.Tensor): vector in Gell-Mann basis, `shape=(...,d**2-1)`

    Returns:
        ret (np.ndarray,torch.Tensor): density matrix, `shape=(...,d,d)`
    '''
    shape = vec.shape
    N0 = int(round(np.sqrt(shape[-1]+1).item()))
    assert shape[-1]==N0*N0-1
    vec = vec.reshape(-1, shape[-1])
    tmp0 = 1/np.sqrt(2*N0).item()
    if isinstance(vec, torch.Tensor):
        tmp1 = torch.concat([vec, torch.ones(vec.shape[0], 1, dtype=torch.float64)*tmp0], axis=1)
    else:
        tmp1 = np.concatenate([vec, np.ones((vec.shape[0],1), dtype=np.float64)*tmp0], axis=1)
    ret = gellmann_basis_to_matrix(tmp1)
    ret = ret[0] if (len(shape)==1) else ret.reshape(*shape[:-1], *ret.shape[-2:])
    return ret


def dm_to_gellmann_norm(dm:np.ndarray):
    r'''get the norm of a density matrix in Gell-Mann basis,
    For traceless hermitian matrix, the norm is the Frobenius norm divided by sqrt(2)

    Parameters:
        dm (np.ndarray): density matrix, 2d array (support batch)

    Returns:
        ret (np.ndarray): norm of dm
    '''
    shape = dm.shape
    assert shape[-1]==shape[-2]
    N0 = dm.shape[-1]
    dm = dm.reshape(-1, N0, N0)
    tmp0 = (np.trace(dm, axis1=1, axis2=2)/N0).reshape(-1,1,1) * np.eye(N0)
    ret = np.linalg.norm((dm - tmp0).reshape(-1,N0*N0), ord=2, axis=1)/np.sqrt(2)
    ret = ret[0] if (len(shape)==2) else ret.reshape(*shape[:-2])
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


def get_density_matrix_distance2(rho:np.ndarray, sigma:np.ndarray):
    r'''Get the distance between two density matrices in Gell-Mann basis.
    Equivalent to the Frobenius distance over 2.

    Parameters:
        rho (np.ndarray): density matrix, 2d array
        sigma (np.ndarray): density matrix, 2d array

    Returns:
        ret (float): distance
    '''
    # not support batch
    tmp0 = (rho - sigma).reshape(-1)
    if isinstance(rho, torch.Tensor):
        ret = torch.vdot(tmp0, tmp0).real / 2
        # factor 1/2 is due to the normalization of Gell-Mann basis
    else:
        ret = np.vdot(tmp0, tmp0).real / 2
    ## equivalent to below
    # tmp0 = dm_to_gellmann_basis(rho)
    # tmp1 = dm_to_gellmann_basis(sigma)
    # ret = np.linalg.norm(tmp0-tmp1)**2
    return ret

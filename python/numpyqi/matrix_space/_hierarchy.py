import itertools
import functools
import operator
import numpy as np
import scipy.special

from ._internal import is_linear_independent

hf_kron = lambda x: functools.reduce(np.kron, x)
hf_multiply = lambda x: functools.reduce(operator.mul, x)


# https://arxiv.org/abs/2210.16389v1
# A Complete Hierarchy of Linear Systems for Certifying Quantum Entanglement of Subspaces
@functools.lru_cache
def permutation_with_antisymmetric_factor(N0):
    index = np.array(list(itertools.permutations(list(range(N0)), N0)), dtype=np.int64)
    value = 1-2*(np.sum(np.triu(index[:,:,np.newaxis] > index[:,np.newaxis], 1), axis=(1,2)) % 2)
    return index,value


# bad performance
@functools.lru_cache
def get_antisymmetric_basis(dim, repeat):
    # (ret) (np,float64,(N0,N1))
    #    N0: number of basis
    #    N1=dim**repeat
    assert (0<repeat) and (repeat<=dim)
    permutation_index, permutation_value = permutation_with_antisymmetric_factor(repeat)
    index = np.array(list(itertools.combinations(list(range(dim)), repeat)), dtype=np.int64).T.copy()
    ret = np.zeros((index.shape[1], dim**repeat), dtype=np.float64)
    factor = 1/np.sqrt(scipy.special.factorial(repeat))
    for ind0,value in zip(permutation_index, permutation_value):
        tmp0 = np.ravel_multi_index(index[list(ind0)], [dim]*repeat)
        ret[np.arange(len(tmp0)), tmp0] = value * factor
    return ret


# bad performance
@functools.lru_cache
def get_symmetric_basis(dim, repeat):
    # (ret) (np,float64,(N0,N1))
    #    N0: number of basis
    #    N1=dim**repeat
    assert 0<repeat
    permutation_index = permutation_with_antisymmetric_factor(repeat)[0]
    index = np.array(list(itertools.combinations_with_replacement(list(range(dim)), repeat)), dtype=np.int64).T.copy()
    tmp0 = np.stack([(index==x).sum(axis=0) for x in range(dim)], axis=1)
    factor = np.sqrt(np.product(scipy.special.factorial(tmp0),axis=1) * (1/scipy.special.factorial(repeat)))
    ret = np.zeros((index.shape[1], dim**repeat), dtype=np.float64)
    for ind0 in permutation_index:
        tmp0 = np.ravel_multi_index(index[list(ind0)], [dim]*repeat)
        ret[np.arange(len(tmp0)), tmp0] = factor
    return ret


@functools.lru_cache
def _project_to_antisymmetric_basis_index(dim, repeat):
    permutation_index, permutation_value = permutation_with_antisymmetric_factor(repeat)
    index = np.array(list(itertools.combinations(list(range(dim)), repeat)), dtype=np.int64).T.copy()
    return permutation_index, permutation_value, index


# TODO replace with np.einsum with indexing
def project_to_antisymmetric_basis(np_list):
    repeat = len(np_list)
    dim = np_list[0].shape[0]
    permutation_index, permutation_value, index = _project_to_antisymmetric_basis_index(dim, repeat)
    ret = 0
    factor = 1/np.sqrt(scipy.special.factorial(repeat))
    for ind0,value in zip(permutation_index, permutation_value):
        tmp0 = (np_list[x][index[y]] for x,y in enumerate(ind0))
        # tmp0 = (np_list[y][index[x]] for x,y in enumerate(ind0)) #both two are correct
        ret = ret + (value*factor)*hf_multiply(tmp0)
    return ret


def tensor2d_project_to_antisym_basis(np_list):
    repeat = len(np_list)
    assert repeat>0
    assert all(x.ndim==2 for x in np_list)
    dimA,dimB = np_list[0].shape
    assert all(x.shape==(dimA,dimB) for x in np_list)
    pindex, pvalue, indI = _project_to_antisymmetric_basis_index(dimA, repeat)
    _, _, indJ = _project_to_antisymmetric_basis_index(dimB, repeat) #pindex0==pindex1
    # pindex(np,int,(N0,r))
    # pvalue(np,float,(N0,))
    # indI(np,int,(r,N1))
    # indJ(np,int,(r,N2))
    # ret(np,float,(N1,N2))
    ret = 0
    tmp0 = ((x0,x1,y0,y1) for x0,x1 in zip(pindex, pvalue) for y0,y1 in zip(pindex, pvalue))
    factor = 1/scipy.special.factorial(repeat)
    for ind0,value0,ind1,value1 in tmp0:
        tmp1 = hf_multiply(np_list[x0][indI[x1,:,np.newaxis],indJ[y]] for x0,(x1,y) in enumerate(zip(ind0,ind1)))
        ret = ret + (value0*value1*factor)*tmp1
    return ret


# bad performance, for unittest only
@functools.lru_cache
def naive_antisym_sym_projector(dimA, dimB, r, k):
    tmp0 = get_symmetric_basis(dimA*dimB, r+k)
    N0 = tmp0.shape[0]
    tmp1 = [N0] + [x for _ in range(r+k) for x in [dimA,dimB]]
    tmp2 = [0] + list(range(1,2*r+2*k+1,2)) + list(range(2,2*r+2*k+1,2))
    z0 = tmp0.reshape(tmp1).transpose(tmp2).reshape(N0, -1)
    z1 = get_antisymmetric_basis(dimA, r+1)
    z2 = get_antisymmetric_basis(dimB, r+1)
    ret = np.einsum(z0, [0,7], z0.reshape(N0, dimA**(r+1), dimA**(k-1), dimB**(r+1), dimB**(k-1)), [0,1,2,3,4],
            z1, [5,1], z2, [6,3], [5,2,6,4,7], optimize=True).reshape(-1, z0.shape[1])
    return ret


def naive_tensor2d_project_to_sym_antisym_basis(np_list, r):
    k = len(np_list) - r
    assert (r>0) and (k>0)
    assert all(x.ndim==2 for x in np_list)
    dimA,dimB = np_list[0].shape
    assert all(x.shape==(dimA,dimB) for x in np_list)
    tmp0 = naive_antisym_sym_projector(dimA, dimB, r, k)
    tmp1 = tmp0.reshape([-1]+[dimA]*(r+k)+[dimB]*(r+k)) #N, dimA, dimA, dimA, dimA, dimB, dimB, dimB, dimB
    tmp2 = zip(np_list, list(range(1,r+k+1)), list(range(r+k+1,2*r+2*k+1)))
    tmp3 = [y for x0,x1,x2 in tmp2 for y in (x0,(x1,x2))]
    ret = np.einsum(tmp1, list(range(2*r+2*k+1)), *tmp3, [0], optimize=True)
    return ret


def tensor2d_project_to_sym_antisym_basis(np_list, r):
    k = len(np_list) - r
    assert (r>0) and (k>0)
    assert all(x.ndim==2 for x in np_list)
    dimA,dimB = np_list[0].shape
    assert all(x.shape==(dimA,dimB) for x in np_list)
    ret = 0
    for indI in itertools.permutations(list(range(r+k))):
        np_list_i = [np_list[x] for x in indI]
        tmp0 = tensor2d_project_to_antisym_basis(np_list_i[:(r+1)])
        if k==1:
            ret = ret + tmp0
        elif k==2:
            ret = ret + np.kron(tmp0, np_list_i[-1]).reshape(-1)
        else:
            # TODO optimize using symmetry
            ret = ret + hf_kron([tmp0] + np_list_i[-(k-1):]).reshape(-1)
            # ret = ret + (tmp0[:,np.newaxis]*hf_kron(np_list_i[-(k-1):]).reshape(-1)).reshape(-1)
    ret *= (1/scipy.special.factorial(r+k))
    return ret


def has_rank(matrix_space, rank, hierarchy_k=1, zero_eps=1e-7)->bool:
    assert rank>1
    r = rank-1
    tmp0 = itertools.combinations_with_replacement(list(range(len(matrix_space))), r+hierarchy_k)
    # TODO batch indexing
    if hierarchy_k==1:
        z0 = [tensor2d_project_to_antisym_basis([matrix_space[y] for y in x]) for x in tmp0]
    else:
        z0 = [tensor2d_project_to_sym_antisym_basis([matrix_space[y] for y in x], r) for x in tmp0]
    ret = is_linear_independent(np.stack([x.reshape(-1) for x in z0]), zero_eps)
    # if ret=True, matrix_space must be at least rank
    # if ret=False, not too much can be predicted
    return ret


# TODO tensor2d_project_to_sym_antisym_basis with all in one, make use tensor network
# def tensor2d

# def split_r_k_index(r, k):
#     pass


# np_rng = np.random.default_rng()
# hf_randc = lambda *size: np_rng.normal(size=size) + 1j*np_rng.normal(size=size)

# dimA = 4
# dimB = 4
# rank = 2
# hierarchy_k = 2
# num_matrix = 4




# matrix_subspace = [hf_randc(dimA,dimB) for _ in range(num_matrix)]
# # z0 = is_matrix_space_has_rank(matrix_subspace, rank, hierarchy_k)

# np_list_index_list = list(itertools.combinations_with_replacement(list(range(len(matrix_subspace))), rank+hierarchy_k))
# tmp1 = np.stack([tensor2d_project_to_sym_antisym_basis([matrix_subspace[y] for y in x], rank) for x in np_list_index_list])
# z0 = tmp1.conj() @ tmp1.T

# zc0 = np.zeros_like(z0)

# np_list0 = [matrix_subspace[y] for y in np_list_index_list[0]]
# np_list1 = [matrix_subspace[y] for y in np_list_index_list[1]]


# # N0 = 5
# # N1 = 11
# # zc0 = np.tril(hf_randc(N0,N0))
# # zc1 = np.triu(hf_randc(N0,N1))
# # zc1[N0-1,N0-1] = 0
# # print(np.linalg.svd(zc0 @ zc1)[1])

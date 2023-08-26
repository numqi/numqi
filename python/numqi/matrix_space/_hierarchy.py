import math
import collections
import itertools
import functools
import operator
import numpy as np
import scipy.special
import concurrent.futures

hf_kron = lambda x: functools.reduce(np.kron, x)
hf_multiply = lambda x: functools.reduce(operator.mul, x)

from ._misc import is_vector_linear_independent


# https://arxiv.org/abs/2210.16389v1
# A Complete Hierarchy of Linear Systems for Certifying Quantum Entanglement of Subspaces
def _permutation_antisymmetric_hf0(repeat_list, all_index):
    r = repeat_list[0]
    if r==1:
        assert len(repeat_list)==len(all_index)
        yield from itertools.permutations(all_index)
    else:
        if len(repeat_list)==1:
            assert len(all_index)==r
            yield all_index
        else:
            tmp0 = set(all_index)
            for x in itertools.combinations(all_index, r):
                for y in _permutation_antisymmetric_hf0(repeat_list[1:], tuple(sorted(tmp0 - set(x)))):
                    yield x+y


@functools.lru_cache
def _permutation_with_antisymmetric_factor_on_int_tuple(int_tuple):
    if len(int_tuple)==1:
        index = np.array([[0]], dtype=np.int64)
        value = np.array([1], dtype=np.int64)
    else:
        hf1 = lambda x:x[1]
        tmp0 = itertools.groupby(sorted(enumerate(int_tuple), key=hf1), key=hf1)
        tmp1 = sorted([(x0,[y[0] for y in x1]) for x0,x1 in tmp0], key=lambda x:len(x[1]), reverse=True)
        tmp2 = np.array(list(_permutation_antisymmetric_hf0([len(x[1]) for x in tmp1], tuple(range(len(int_tuple))))), dtype=np.int64)
        index = np.zeros_like(tmp2)
        index[np.arange(len(tmp2))[:,np.newaxis],tmp2] = np.array([y for x in tmp1 for y in x[1]])
        tmp3 = np.array([y[0] for y in sorted(enumerate(index.tolist()), key=lambda x:x[1])])
        index = np.argsort(index[tmp3], axis=1)
        value = 1-2*(np.sum(np.triu(index[:,:,np.newaxis] > index[:,np.newaxis], 1), axis=(1,2)) % 2)
    return index,value

def permutation_with_antisymmetric_factor(x0):
    if (hasattr(x0, '__len__') and len(x0)==1) or ((not hasattr(x0,'__len__')) and (int(x0)==1)):
        index = np.array([[0]], dtype=np.int64)
        value = np.array([1], dtype=np.int64)
    else:
        if hasattr(x0, '__len__'):
            int_tuple_ori = tuple(int(x) for x in x0)
            assert len(int_tuple_ori) >= 2
            tmp0 = {y:x for x,y in enumerate(sorted(set(int_tuple_ori)))}
            int_tuple = tuple(tmp0[x] for x in int_tuple_ori)
            multiplier = np.prod([math.factorial(len(list(y))) for x,y in itertools.groupby(sorted(int_tuple_ori))])
        else:
            x0 = int(x0)
            assert x0 >= 2
            int_tuple = tuple(range(x0))
            multiplier = 1
        index,value = _permutation_with_antisymmetric_factor_on_int_tuple(int_tuple)
        value = value * multiplier
    return index,value

# bad performance
@functools.lru_cache
def get_antisymmetric_basis(dim, rank):
    # (ret) (np,float64,(N0,N1))
    #    N0: number of basis
    #    N1=dim**rank
    assert (0<rank) and (rank<=dim)
    pindex, pvalue = permutation_with_antisymmetric_factor(rank)
    index = np.array(list(itertools.combinations(list(range(dim)), rank)), dtype=np.int64).T.copy()
    ret = np.zeros((index.shape[1], dim**rank), dtype=np.float64)
    factor = 1/np.sqrt(scipy.special.factorial(rank))
    for ind0,value in zip(pindex, pvalue):
        tmp0 = np.ravel_multi_index(index[list(ind0)], [dim]*rank)
        ret[np.arange(len(tmp0)), tmp0] = value * factor
    return ret


@functools.lru_cache
def get_antisymmetric_basis_index(dim, repeat):
    pindex, pvalue = permutation_with_antisymmetric_factor(repeat)
    index = np.array(list(itertools.combinations(list(range(dim)), pindex.shape[1])), dtype=np.int64).T.copy()
    return pindex, pvalue, index


# bad performance
@functools.lru_cache
def get_symmetric_basis(dim, rank):
    # (ret) (np,float64,(N0,N1))
    #    N0: number of basis
    #    N1=dim**rank
    assert 0<rank
    pindex = permutation_with_antisymmetric_factor(rank)[0]
    index = np.array(list(itertools.combinations_with_replacement(list(range(dim)), rank)), dtype=np.int64).T.copy()
    tmp0 = np.stack([(index==x).sum(axis=0) for x in range(dim)], axis=1)
    factor = np.sqrt(np.prod(scipy.special.factorial(tmp0),axis=1) * (1/scipy.special.factorial(rank)))
    ret = np.zeros((index.shape[1], dim**rank), dtype=np.float64)
    for ind0 in pindex:
        tmp0 = np.ravel_multi_index(index[list(ind0)], [dim]*rank)
        ret[np.arange(len(tmp0)), tmp0] = factor
    return ret


@functools.lru_cache
def get_symmetric_basis_index(dim, repeat):
    pindex, _ = permutation_with_antisymmetric_factor(repeat)
    tmp0 = list(itertools.combinations_with_replacement(list(range(dim)), pindex.shape[1]))
    index = np.array(tmp0, dtype=np.int64).T.copy()
    tmp1 = [np.prod([math.factorial(y) for y in collections.Counter(x).values()]) for x in tmp0]
    pvalue = np.sqrt(math.factorial(pindex.shape[1])/np.array(tmp1)) / pindex.shape[0]
    return pindex,pvalue,index


def project_nd_tensor_to_antisymmetric_basis(np0, rank):
    shape = np0.shape
    is_single_item = len(shape)==rank
    assert len(shape)>=rank
    dim = shape[0]
    N0 = 1 if is_single_item else np.prod(shape[rank:])
    assert shape[:rank]==((dim,)*rank)
    np0 = np0.reshape([dim]*rank+[N0])
    if rank>dim:
        ret = np.zeros((0,)+shape[rank:], dtype=np0.dtype)
    else:
        pindex, pvalue, index = get_antisymmetric_basis_index(dim, rank)
        factor = 1/np.sqrt(scipy.special.factorial(rank))
        ret = 0
        for ind0,value in zip(pindex, pvalue):
            ret = ret + (value*factor)*np0[tuple(index[x] for x in ind0)]
        ret = ret.reshape((-1,)+shape[rank:])
    return ret

def project_nd_tensor_to_symmetric_basis(np0, rank):
    shape = np0.shape
    is_single_item = len(shape)==rank
    assert len(shape)>=rank
    dim = shape[0]
    N0 = 1 if is_single_item else np.prod(shape[rank:])
    assert shape[:rank]==((dim,)*rank)
    np0 = np0.reshape([dim]*rank+[N0])
    pindex, pvalue, index = get_symmetric_basis_index(dim, rank)
    ret = pvalue[:,np.newaxis]*sum(np0[tuple(index[x] for x in ind0)] for ind0 in pindex)
    ret = ret.reshape((-1,)+shape[rank:])
    return ret

def project_to_antisymmetric_basis(np_list):
    repeat = len(np_list)
    dim = np_list[0].shape[0]
    pindex, pvalue, index = get_antisymmetric_basis_index(dim, repeat)
    ret = 0
    factor = 1/np.sqrt(scipy.special.factorial(repeat))
    for ind0,value in zip(pindex, pvalue):
        tmp0 = (np_list[x][index[y]] for x,y in enumerate(ind0))
        # tmp0 = (np_list[y][index[x]] for x,y in enumerate(ind0)) #both two are correct
        ret = ret + (value*factor)*hf_multiply(tmp0)
    return ret


def project_to_symmetric_basis(np_list, INDEX=None):
    if INDEX is None:
        # INDEX will be used as equivalent tensor
        INDEX = tuple(range(len(np_list)))
    else:
        INDEX = tuple(int(x) for x in INDEX)
    if len(np_list)==1:
        ret = np_list[0]
    else:
        dim = np_list[0].shape[0]
        assert all(x.shape[0]==dim for x in np_list)
        pindex, pvalue, index = get_symmetric_basis_index(dim, INDEX)
        ret = 0
        for ind0 in pindex:
            tmp0 = (np_list[INDEX[x]][index[y]] for x,y in enumerate(ind0))
            # tmp0 = (np_list[y][index[x]] for x,y in enumerate(ind0)) #both two are correct
            ret = ret + hf_multiply(tmp0)
        ret *= pvalue.reshape([-1]+[1]*len(np_list[0].shape[1:]))
    return ret

def tensor2d_project_to_antisym_basis(np_list, INDEX=None):
    if INDEX is None:
        # INDEX will be used as equivalent tensor
        INDEX = np.arange(len(np_list))
    else:
        INDEX = np.asarray(INDEX, dtype=np.int64)
    repeat = len(INDEX)
    assert repeat>0
    assert all(x.ndim==2 for x in np_list)
    dimA,dimB = np_list[0].shape
    assert all(x.shape==(dimA,dimB) for x in np_list)
    pindex0, pvalue0, indI = get_antisymmetric_basis_index(dimA, tuple(INDEX.tolist()))
    pindex1, pvalue1, indJ = get_antisymmetric_basis_index(dimB, repeat)
    # pindex0,pindex1,pindex2 = [np.argsort(x,axis=1) for x in [pindex0,pindex1,pindex2]]
    # pindex(np,int,(N0,r))
    # pvalue(np,float,(N0,))
    # indI(np,int,(r,N1))
    # indJ(np,int,(r,N2))
    # ret(np,float,(N1,N2))
    ret = 0
    factor = 1/scipy.special.factorial(repeat)
    for ind0,value0 in zip(pindex0, pvalue0):
        tmp0 = [np_list[x0][indI[x1]] for x0,x1 in zip(INDEX,ind0)]
        for ind1,value1 in zip(pindex1, pvalue1):
            ret = ret + (value0*value1*factor)*hf_multiply(x0[:,indJ[x1]] for x0,x1 in zip(tmp0,ind1))
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


def tensor2d_project_to_sym_antisym_basis(np_list, r, INDEX=None):
    if INDEX is None:
        INDEX = np.arange(len(np_list), dtype=np.int64)
    else:
        INDEX = np.asarray(INDEX, dtype=np.int64)
    k = len(INDEX) - r
    if k==1:
        ret = tensor2d_project_to_antisym_basis(np_list, INDEX)
    else:
        assert (r>0) and (k>0)
        assert all(x.ndim==2 for x in np_list)
        dimA,dimB = np_list[0].shape
        assert all(x.shape==(dimA,dimB) for x in np_list)
        ret = 0
        for indI0 in itertools.combinations(list(range(r+k)), r+1):
            tmp0 = tensor2d_project_to_antisym_basis(np_list, INDEX[list(indI0)])
            tmp1 = project_to_symmetric_basis([x.reshape(-1) for x in np_list], INDEX[sorted(set(range(r+k))-set(indI0))])
            ret = ret + tmp0[:,:,np.newaxis]*tmp1
        ret /= scipy.special.binom(r+k, r+1)
    return ret

def has_rank_hierarchical_method(matrix_space, rank, hierarchy_k=1, zero_eps=1e-7, return_space=False, num_worker=None):
    # "r-entangled" (defined in the paper) means that the matrix_space has at-least rank (r+1)
    assert rank>1
    r = rank-1
    tmp0 = itertools.combinations_with_replacement(list(range(len(matrix_space))), r+hierarchy_k)
    # TODO batch indexing
    if num_worker is None:
        z0 = np.stack([tensor2d_project_to_sym_antisym_basis(matrix_space, r, x).reshape(-1) for x in tmp0])
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            job_list = [executor.submit(tensor2d_project_to_sym_antisym_basis, matrix_space, r, x) for x in tmp0]
            z0 = np.stack([x.result().reshape(-1) for x in job_list])
    ret = is_vector_linear_independent(z0, 'complex', zero_eps)
    # if ret=True, matrix_space must be at least rank
    # if ret=False, not too much can be predicted
    if return_space:
        ret = ret, z0
    return ret

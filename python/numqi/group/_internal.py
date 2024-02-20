import math
import itertools
import functools
import numpy as np
import scipy.linalg

# TODO documentation

def cayley_table_to_left_regular_form(index_tuple):
    N0 = len(index_tuple)
    ret = np.zeros((N0,N0,N0), dtype=np.int64)
    tmp0 = np.arange(N0)
    for ind0 in range(N0):
        ret[ind0,np.array(index_tuple[ind0]),tmp0] = 1
    return ret


@functools.lru_cache
def permutation_to_cycle_notation(index_tuple):
    # permutation_to_cycle_notation((2,3,0,1))
    # permutation_to_cycle_notation(((0,2), (1,3), (2,0), (3,1)))
    assert len(index_tuple)>0
    if hasattr(index_tuple[0], '__len__'):
        assert all(len(x)==2 for x in index_tuple)
        z0 = {x:y for x,y in index_tuple}
    else:
        z0 = dict(enumerate(index_tuple))
    z1 = set(z0.keys())
    ret = []
    while len(z1)>0:
        x0 = z1.pop()
        tmp0 = [x0]
        while True:
            x = z0[tmp0[-1]]
            if x==x0:
                break
            else:
                assert x in z1
                tmp0.append(x)
                z1.remove(x)
        ret.append(tmp0)
    ret = tuple(tuple(x) for x in ret)
    return ret


def get_klein_four_group_cayley_table():
    ret = np.array((
        (0,1,2,3),
        (1,0,3,2),
        (2,3,0,1),
        (3,2,1,0),
    ), dtype=np.int64)
    return ret


# def get_dihedral_group_matrix(n):
#     # https://math.stackexchange.com/a/1576762
#     tmp0 = np.zeros((n,n), dtype=np.int64)
#     tmp0[np.arange(1,n+1)%n, np.arange(n)] = 1
#     tmp1 = np.eye(n, dtype=np.int64)[::-1]
#     ret = [np.eye(n, dtype=np.int64)]
#     for _ in range(n-1):
#         ret.append(ret[-1] @ tmp0)
#     ret.append(tmp1)
#     for _ in range(n-1):
#         ret.append(ret[-1] @ tmp0)
#     ret = np.stack(ret)
#     return ret


@functools.lru_cache
def get_dihedral_group_cayley_table(n):
    # https://math.stackexchange.com/a/1576762
    assert n>2
    tmp0 = scipy.linalg.circulant(np.arange(n, dtype=np.int64)).T
    tmp1 = tmp0 @ (np.eye(n, dtype=np.int64)[::-1])
    tmp2 = np.concatenate([tmp0,tmp1], axis=0)
    tmp3 = {tuple(y):x for x,y in enumerate(tmp2.tolist())}
    ret = np.array(tuple([tuple([tmp3[tuple(y)] for y in x]) for x in tmp2[:, tmp2].tolist()]), dtype=np.int64)
    return ret


# cyclic group irrep on C-field is just complex number
# cyclic group irrep on R-field are 2-by-2 matrix
def get_cyclic_group_cayley_table(n):
    assert n>=2
    tmp0 = np.arange(n, dtype=np.int64)
    ret = np.array(tuple(tuple(x) for x in np.remainder(tmp0[:,np.newaxis] + tmp0, n).tolist()), dtype=np.int64)
    return ret


def _dummy_partition(length, hf0):
    ret = []
    ind_start = 0
    ind_end = 1
    while ind_start<length:
        if ind_end==length:
            ret.append(slice(ind_start, ind_end))
            break
        if hf0(ind_start, ind_end):
            ind_end = ind_end + 1
        else:
            ret.append(slice(ind_start,ind_end))
            ind_start,ind_end = ind_end,ind_end+1
    return ret


def _reduce_group_representation_get_matH(np0, zero_eps=1e-4):
    # tmp1 = np.einsum(tmp0, [0,1,2], np0, [0,3,4], [1,4,2,3], optimize=True)
    # tmp2 = tmp1.transpose(0,1,3,2)
    # matH = np.tril(tmp1 + tmp2, k=0) + np.triu(1j*(tmp1-tmp2), k=1)
    # tmp3 = np.abs(matH[0,0,:,:]*(np.eye(dim).reshape(dim,dim,1,1)) - matH).max(axis=(0,1))
    # indR,indS = np.nonzero(tmp3>1e-4)
    # assert len(indR)>0
    # matH = matH[:,:,indR[0],indS[0]]
    dim = np0.shape[0]
    np1 = np0.transpose(1,0,2).copy()
    np2 = np0.transpose(1,2,0).conj().copy()
    eye = np.eye(np0.shape[1])
    ret = None
    for ind0,ind1 in [(x,y) for x in range(dim) for y in range(x,dim)]:
        if ind0==ind1:
            tmp0 = np2[ind0] @ np1[ind0]
            if np.abs(tmp0 - tmp0[0,0]*eye).max() > zero_eps:
                ret = tmp0
                break
        else:
            tmp0 = np2[ind0] @ np1[ind1]
            tmp1 = np2[ind1] @ np1[ind0]
            tmp2 = tmp0 + tmp1
            if np.abs(tmp2-tmp2[0,0]*eye).max() > zero_eps:
                ret = tmp2
                break
            tmp2 = 1j*(tmp0 - tmp1)
            if np.abs(tmp2-tmp2[0,0]*eye).max() > zero_eps:
                ret = tmp2
                break
    return ret


def reduce_group_representation(np0, zero_eps=1e-7):
    # https://sheaves.github.io/Representation-Theory-Decomposing-Representations/
    assert (np0.ndim==3) and (np0.shape[1]==np0.shape[1])
    dim = np0.shape[1]
    if (np.linalg.norm(np.trace(np0, axis1=1, axis2=2))**2/np0.shape[0])<1.5: #character theory
        ret = [np0]
    else:
        tmp0 = np0.transpose(0,2,1).conj()
        assert np.abs(tmp0 @ np0 - np.eye(dim)).max() < zero_eps, 'must be unitary representation'
        matH = _reduce_group_representation_get_matH(np0, np.sqrt(zero_eps))
        EVL,EVC = np.linalg.eigh(matH)
        ind_list = _dummy_partition(dim, lambda x,y: abs(EVL[x]-EVL[y])<1e-4)
        z0 = EVC.T.conj() @ np0 @ EVC
        z1 = []
        for x in ind_list:
            tmp0 = z0[:,x,x]
            if tmp0.shape[1]>1:
                z1 += reduce_group_representation(tmp0)
            else:
                z1.append(tmp0)
        hf0 = lambda x: x.shape[1]
        z1 = [list(y) for x,y in itertools.groupby(sorted(z1, key=hf0), key=hf0)]
        # check the equivalence
        ret = []
        for tmp0 in z1:
            if len(tmp0)==1:
                ret.append(tmp0[0])
            else:
                tmp1 = np.stack([np.trace(x, axis1=1, axis2=2) for x in tmp0])
                tmp2 = np.abs(tmp1 @ tmp1.T.conj())/tmp1.shape[1]
                tmp3 = list(set(tuple(sorted(np.nonzero(x)[0].tolist())) for x in np.abs(tmp2-1)<zero_eps))
                ret += [tmp0[x[0]] for x in tmp3]
    return ret


def to_unitary_representation(np0, return_matP=False):
    assert np0.ndim==3
    tmp0 = np0.reshape(-1,np0.shape[2])
    matP = scipy.linalg.sqrtm(tmp0.T.conj() @ tmp0)
    if matP.dtype.name=='complex256':
        # scipy-v1.10 bug https://github.com/scipy/scipy/issues/18250
        matP = matP.astype(np.complex128)
    np1 = matP @ np0 @ np.linalg.inv(matP)
    ret = (np1,matP) if return_matP else np1
    return ret


def _matrix_block_diagonal_shape_slice(int_list):
    tmp0 = np.cumsum(np.array((0,)+tuple(int_list), dtype=np.int64))
    ret = [slice(x,y) for x,y in zip(tmp0[:-1],tmp0[1:])]
    return ret

def matrix_block_diagonal(*np_list):
    # matrix direct sum
    if len(np_list)==1:
        ret = np_list[0]
    else:
        assert all(x.ndim>=2 for x in np_list)
        shape0_list = np.broadcast_shapes(*[x.shape[:-2] for x in np_list])
        slice1_list = _matrix_block_diagonal_shape_slice([x.shape[-2] for x in np_list])
        slice2_list = _matrix_block_diagonal_shape_slice([x.shape[-1] for x in np_list])
        dtype = functools.reduce(np.promote_types, [x.dtype for x in np_list])
        tmp0 = shape0_list + (slice1_list[-1].stop,slice2_list[-1].stop)
        ret = np.zeros(tmp0, dtype=dtype)
        for ind0 in range(len(np_list)):
            ret[...,slice1_list[ind0],slice2_list[ind0]] = np_list[ind0]
    return ret


def get_charater_and_class(irrep_list, zero_eps=1e-7):
    character = np.stack([np.trace(x, axis1=1, axis2=2) for x in irrep_list])
    N0 = character.shape[1]
    tmp0 = character.T.conj() @ character
    tmp1 = {tuple(np.nonzero(np.abs(x)>(zero_eps*N0))[0].tolist()) for x in tmp0}
    class_list = sorted(tmp1, key=lambda x: (len(x),x))
    character_table = character[:,[x[0] for x in class_list]]
    return character, class_list, character_table


hf_Euler_totient = lambda n: sum(math.gcd(n,x)==1 for x in range(1, n+1))
# TODO sympy.totient()
# https://stackoverflow.com/a/18114286
hf_is_prime = lambda n: (n>=2) and ((n==2) or ((n%2==1) and all(n%x>0 for x in range(3, int(math.sqrt(n))+1, 2))))
# TODO sympy.isprime()
# https://stackoverflow.com/q/18114138
# True: 2,3,5,7
# False: -1,0,1,4,6

# wiki https://en.wikipedia.org/wiki/Multiplicative_group_of_integers_modulo_n
# direct product of cyclic group
@functools.lru_cache
def get_multiplicative_group_cayley_table(n):
    assert n>=3 #TODO
    element = [x for x in range(1, n) if math.gcd(n,x)==1]
    x_to_index = {y:x for x,y in enumerate(element)}
    ret = np.array(tuple(tuple(x_to_index[(x*y)%n] for y in element) for x in element), dtype=np.int64)
    return ret


@functools.lru_cache
def get_quaternion_cayley_table():
    tmp0 = ['1 i j k', 'i -1 k -j', 'j -k -1 i', 'k j -i -1']
    tmp0 = [x.split( ) for x in tmp0]
    hf0 = lambda x: (('-'+x) if (len(x)==1) else x[1])
    tmp1 = tmp0 + [[hf0(y) for y in x] for x in tmp0]
    tmp2 = [(x+[hf0(y) for y in x]) for x in tmp1]
    z0 = {y:x for x,y in enumerate('1 i j k -1 -i -j -k'.split(' '))}
    ret = np.array(tuple(tuple(z0[y] for y in x) for x in tmp2), dtype=np.int64)
    return ret


def get_index_cayley_table(cayley_table):
    np0 = np.asarray(cayley_table)
    ret = np.argsort(np0.reshape(-1))
    return ret


def group_algebra_product(vec0, vec1, cayley_table, use_index=False):
    assert vec0.shape[-1]==vec1.shape[-1]
    dim = vec0.shape[-1]
    if use_index:
        index = np.asarray(cayley_table)
    else:
        index = get_index_cayley_table(cayley_table)
    shape = np.broadcast_shapes(vec0.shape, vec1.shape)
    N0 = np.prod(np.array(shape, dtype=np.int64))
    vec0,vec1 = [np.broadcast_to(x, shape).reshape(-1, dim) for x in (vec0,vec1)]
    ret = (vec0[:,:,np.newaxis] * vec1[:,np.newaxis,:]).reshape(-1, dim*dim)[:, index].reshape(-1, dim, dim).sum(axis=2).reshape(shape)
    return ret


def pretty_print_character_table(character_table, class_list):
    tmp0 = np.round(character_table.real).astype(np.int64)
    character_table_str = [[(str(y0) if abs(y0-y1)<1e-10 else 'xxx') for y0,y1 in zip(x0,x1)] for x0,x1 in zip(tmp0,character_table)]
    print('| $\chi$ | {} |'.format(' | '.join(str(len(x)) for x in class_list)))
    print('| {} |'.format(' | '.join([':-:']*(len(class_list)+1))))
    # assert np.abs(tmp0-character_table).max() < 1e-10
    for ind0 in range(len(character_table_str)):
        tmp1 = ' | '.join(character_table_str[ind0])
        tmp2 = '{' + str(ind0) + '}'
        print(f'| $A_{tmp2}$ | {tmp1} |')

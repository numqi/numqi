import itertools
import functools
import math
import collections
import sympy.combinatorics
import numpy as np

import numqi.utils

# def get_symmetric_group_matrix(n):
#     # TODO seems not correct, we need regular form, this seems to be permutation group
#     ret = []
#     tmp0 = np.arange(n, dtype=np.int64)
#     for x in itertools.permutations(list(range(n))):
#         tmp1 = np.zeros((n,n), dtype=np.int64)
#         tmp1[np.array(x), tmp0] = 1
#         ret.append(tmp1)
#     ret = np.stack(ret)
#     return ret


def permutation_to_cycle_notation(index_tuple:tuple[int]):
    r'''Permutation to cycle notation

    Parameters:
        index_tuple (tuple[int]): tuple of int, e.g. `(2,3,0,1)`

    Returns:
        ret (tuple[tuple[int]]): tuple of tuple of int, e.g. `((0,2),(1,3))`
    '''
    assert len(index_tuple)>0
    z0 = dict(enumerate(numqi.utils.hf_tuple_of_int(index_tuple)))
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


@functools.lru_cache
def _get_symmetric_group_cayley_table_hf0(n:int, alternating:bool):
    if alternating:
        perm_list = []
        for x in itertools.permutations(list(range(n))):
            y = permutation_to_cycle_notation(x)
            if sum((len(x)-1) for x in y)%2==0:
                perm_list.append(x)
    else:
        perm_list = list(itertools.permutations(list(range(n))))
    tmp1 = {y:x for x,y in enumerate(perm_list)}
    tmp2 = np.array(perm_list, dtype=np.int64)
    ret = np.array(tuple([tuple([tmp1[tuple(y)] for y in x]) for x in tmp2[:, tmp2].tolist()]), dtype=np.int64)
    return ret


def get_symmetric_group_cayley_table(n:int, alternating:bool=False):
    r'''Cayley table of symmetric group

    Parameters:
        n (int): the order of symmetric group
        alternating (bool): whether to get the alternating group

    Returns:
        ret (np.ndarray): the Cayley table of symmetric group, `dtype=np.int64`, `shape=(n!,n!)`
    '''
    n = int(n)
    assert n>=2
    alternating = bool(alternating)
    ret = _get_symmetric_group_cayley_table_hf0(n, alternating)
    return ret

# slow, only for N<=50
# @functools.lru_cache
# def get_sym_group_num_irrep(N:int):
#     N = int(N)
#     assert N>=1
#     if N<=3:
#         ret = N
#     else:
#         ret = 0
#         func_stack = [(N,N)]
#         # TODO mn_to_value make a cache
#         while len(func_stack)>0:
#             # print(ret, func_stack)
#             n,m = func_stack.pop()
#             for r in range(n//m+1):
#                 n1,m1 = n-r*m,m-1
#                 if (n1==0) or (n1==1) or (m1==1):
#                     ret = ret + 1
#                 elif n1==2:
#                     ret = ret + 2 #m1 must be greater than 1 here
#                 else:
#                     func_stack.append((n1,m1)) #n1>=3, m1>=2
#     return ret



@functools.lru_cache
def _get_sym_group_num_irrep_hf0(N:int, return_full=False):
    if N<=3:
        if return_full:
            tmp0 = np.array([[1,1,1,1], [1,1,1,1], [1,1,2,2], [1,1,2,3]]) #0,1,2,3
            ret = N, tmp0[:(N+1),:(N+1)]
        else:
            ret = N
    else:
        z0 = np.zeros((N+1,N+1), dtype=np.int64)
        z0[:2] = 1
        z0[:,:2] = 1
        z0[2, 2:] = 2
        for m in range(2, N+1):
            for n in range(3, m):
                z0[n,m] = z0[n,n]
            for n in range(m, N+1):
                r = np.arange(n//m+1)
                z0[n,m] = z0[n-r*m, m-1].sum()
        tmp0 = z0[-1,-1].item()
        ret = (tmp0,z0) if return_full else tmp0
    return ret


def get_sym_group_num_irrep(N:int, return_full:bool=False):
    r'''Number of irreducible representations of symmetric group

    Parameters:
        N (int): the order of symmetric group
        return_full (bool): whether to return the full table

    Returns:
        ret0 (int): the number of irreducible representations
        ret1 (np.ndarray): the full table of the number of irreducible representations, `dtype=np.int64`,
                    `shape=(N+1,N+1)`, only if `return_full` is `True`
    '''
    N = int(N)
    assert N>=1
    return_full = bool(return_full)
    ret = _get_sym_group_num_irrep_hf0(N, return_full)
    return ret

def get_sym_group_young_diagram(N:int):
    r'''Young diagram of symmetric group, almost to N=50 (10 seconds)

    Parameters:
        N (int): the order of symmetric group

    Returns:
        ret (np.ndarray): the Young diagram of symmetric group, `dtype=np.int64`, `shape=(#Young,N)`
    '''
    assert N>=1
    dtype = np.int64
    if N==1:
        ret = np.array([[1]], dtype=dtype)
    elif N==2:
        ret = np.array([(2,0), (1,1)], dtype=dtype)
    elif N==3:
        ret = np.array([(3,0,0), (2,1,0), (1,1,1)], dtype=dtype)
    else:
        _,z1 = get_sym_group_num_irrep(N, return_full=True)
        z0 = dict()
        for m in range(1,N+1):
            tmp0 = np.zeros((1,m), dtype=dtype)
            tmp0[0,0] = 1
            z0[(1,m)] = tmp0
            z0[(m,1)] = np.array([m], dtype=dtype)
        for m in range(2,N+1):
            tmp0 = np.zeros((2,m), dtype=dtype)
            tmp0[0,0] = 2
            tmp0[1,:2] = 1
            z0[(2,m)] = tmp0
        for m in range(2,N+1):
            for n in range(m, N+1):
                np0 = np.zeros((z1[n,m],m), dtype=dtype)
                ind0 = 0
                for r in range(n//m+1):
                    n1,m1 = n-r*m,m-1
                    if n1==0:
                        np0[ind0] = r
                        ind0 = ind0 + 1
                    elif n1==1:
                        np0[ind0] = r
                        np0[ind0,0] += 1
                        ind0 = ind0 + 1
                    elif m1==1:
                        np0[ind0] = r
                        np0[ind0,0] += n1 #n1>=2
                        ind0 = ind0 + 1
                    elif n1==2:
                        np0[ind0:(ind0+2)] = r #m1>=2
                        np0[ind0,0] += 2
                        np0[ind0+1,:2] += 1
                        ind0 = ind0 + 2
                    else:
                        tmp0 = z0[(n1, min(n1,m1))]
                        shape = tmp0.shape
                        np0[ind0:(ind0+shape[0])] = r
                        np0[ind0:(ind0+shape[0]), :shape[1]] += tmp0
                        ind0 = ind0 + shape[0]
                z0[(n,m)] = np0
        ret = z0[(N,N)]
    return ret


def get_young_diagram_mask(young:tuple[int], check:bool=True):
    r'''Young diagram mask

    Parameters:
        young (tuple[int]): the Young diagram, e.g. `(3,2,1)`
        check (bool): whether to check the input

    Returns:
        ret (np.ndarray): the Young diagram mask, `dtype=np.int64`, `shape=(len(young),young[0])`
    '''
    young = np.asarray(young, dtype=np.int64)
    if check:
        check_young_diagram(young)
    tmp0 = np.arange(young[0], dtype=np.int64)
    ret = (young[:,np.newaxis]>tmp0).astype(np.int64)
    return ret

def check_young_diagram(np0:tuple[int]):
    r'''Check the Young diagram

    Parameters:
        np0 (tuple[int]): the Young diagram, e.g. `(3,2,1)`
    '''
    assert len(np0)>0
    np0 = np.asarray(np0, dtype=np.int64)
    assert np.all(np0>0)
    if len(np0)>1:
        assert np.all(np0[:-1] >= np0[1:])


@functools.lru_cache
def _get_hook_length_hf0(*int_tuple, check):
    np0 = np.asarray(int_tuple, dtype=np.int64)
    if check:
        check_young_diagram(np0)
    mask = get_young_diagram_mask(np0, check=False)
    tmp2 = (mask[::-1].cumsum(axis=0)[::-1] + mask[:,::-1].cumsum(axis=1)[:,::-1] - 1)
    tmp3 = collections.Counter(tmp2[mask.astype(np.bool_)].tolist())
    tmp4 = {x:int(1-tmp3.get(x,0)) for x in range(1,np0.sum()+1)}
    ret = math.prod([k**v for k,v in tmp4.items() if v>0]) // math.prod([k**(-v) for k,v in tmp4.items() if v<0])
    return ret

def get_hook_length(*int_tuple:tuple[int], check:bool=True):
    r'''Hook length [wiki-link](https://en.wikipedia.org/wiki/Hook_length_formula)

    Parameters:
        int_tuple (tuple[int]): the Young diagram, e.g. `(3,2,1)`
        check (bool): whether to check the input

    Returns:
        ret (int): the hook length
    '''
    int_tuple = numqi.utils.hf_tuple_of_int(int_tuple)
    check = bool(check)
    ret = _get_hook_length_hf0(*int_tuple, check=check)
    return ret


def get_young_diagram_transpose(np0:tuple[int], check=True):
    r'''Transpose of Young diagram

    Parameters:
        np0 (tuple[int]): the Young diagram, e.g. `(3,2,1)`
        check (bool): whether to check the input

    Returns:
        ret (np.ndarray): the transpose of Young diagram, `dtype=np.int64`, `shape=(len(np0),)`
    '''
    np0 = np.asarray(np0, dtype=np.int64)
    if check:
        check_young_diagram(np0)
    ret = get_young_diagram_mask(np0, check=False).T.sum(axis=1)
    return ret


def _get_bounded_combination_wrapper(gen, bound_i):
    ret = ((*x,y) for x in gen for y in range(max(x[-1]+1, bound_i[0]), bound_i[1]))
    return ret

def _get_bounded_combination(np0, bound):
    # np0(np,int64,N0) increasing order
    # bound(list,(tuple,int,2),N1)
    xy_tuple = ((x,) for x in range(bound[0][0], bound[0][1]))
    for ind in range(1,len(bound)):
        xy_tuple = _get_bounded_combination_wrapper(xy_tuple, bound[ind])
        # the following line is wrong
        # xy_tuple = ((*x,y) for x in xy_tuple for y in range(max(bound[ind][0],x[-1]+1), bound[ind][1]))
    tmp0 = set(range(len(np0)))
    ret = []
    for xy in xy_tuple:
        tmp1 = [max(x-i-1,0) for i,x in enumerate(xy)] #bound for next layer
        ret.append((np0[list(xy)],tmp1,np0[sorted(tmp0-set(xy))]))
    return ret


def _get_all_young_tableaux_hf0(young, index, lower_bound):
    youngT = get_young_diagram_transpose(young, check=False)
    if (len(young)==1):
        ret = index.reshape(1,1,-1)
    elif young[0]==1:
        ret = index.reshape(1,-1,1)
    elif young[1]==1:
        ret = []
        if all(x==0 for x in lower_bound):
            tmp0 = set(index[1:].tolist())
            for x in itertools.combinations(index[1:], young[0]-1):
                tmp1 = np.zeros((len(young), young[0]), dtype=np.int64)
                tmp1[0,0] = index[0]
                tmp1[0,1:] = np.array(x, dtype=np.int64)
                tmp1[1:,0] = np.array(sorted(tmp0-set(x)))
                ret.append(tmp1)
        else:
            bound = list(zip(lower_bound, range(youngT[0], youngT.sum())))
            for xy_i,_,index_i in _get_bounded_combination(index[1:], bound):
                tmp1 = np.zeros((len(young), young[0]), dtype=np.int64)
                tmp1[0,0] = index[0]
                tmp1[0,1:] = xy_i
                tmp1[1:,0] = index_i
                ret.append(tmp1)
        ret = np.stack(ret)
    else:
        upper_bound = young.sum() - np.cumsum(youngT[::-1])[::-1][1:]
        bound = list(zip(lower_bound, upper_bound))
        ret = []
        for xy_i,lower_bound_i,index_i in _get_bounded_combination(index[1:], bound):
            tmp0 = _get_all_young_tableaux_hf0(young[1:], index_i, lower_bound_i[:(young[1]-1)])
            tmp1 = np.zeros((len(tmp0),len(young),young[0]), dtype=np.int64)
            tmp1[:,0,0] = index[0]
            tmp1[:,0,1:] = xy_i
            tmp1[:,1:,:tmp0.shape[2]] = tmp0
            ret.append(tmp1)
        ret = np.concatenate(ret, axis=0)
    return ret

def get_all_young_tableaux(young:tuple[int], check:bool=True):
    r'''All Young tableaux

    Parameters:
        young (tuple[int]): the Young diagram, e.g. `(3,2,1)`
        check (bool): whether to check the input

    Returns:
        ret (np.ndarray): all Young tableaux, `dtype=np.int64`, `shape=(#tableaux,len(young),young[0])`
    '''
    young = np.asarray(young, dtype=np.int64)
    if check:
        check_young_diagram(young)
    index = np.arange(young.sum(), dtype=np.int64)
    lower_bound = [0]*(young[0]-1)
    ret = _get_all_young_tableaux_hf0(young, index, lower_bound)
    return ret


def young_tableau_to_young_symmetrizer(young:tuple[int], tableau:np.ndarray):
    r'''Young tableau to Young symmetrizer

    Parameters:
        young (tuple[int]): the Young diagram, e.g. `(3,2,1)`
        tableau (np.ndarray): the Young tableau, `dtype=np.int64`, `shape=(len(young),young[0])`

    Returns:
        symmetrizer (np.ndarray): the Young symmetrizer, `dtype=np.int64`, `shape=(#?,sum(young))`
        sign (np.ndarray): the sign of Young symmetrizer, `dtype=np.int64`, `shape=(#?,)`
    '''
    young = np.asarray(young, dtype=np.int64)
    youngT = get_young_diagram_transpose(young)
    N0 = young.sum()
    tmp0 = sorted(set(young.tolist()) | set(youngT.tolist()))
    n_to_allperm = {x:np.array(list(itertools.permutations(list(range(x))))) for x in tmp0 if x>1}

    op_row_list = []
    row_list = [tableau[x,:y] for x,y in enumerate(young) if y>1]
    for row_i in row_list:
        tmp0 = []
        for x in n_to_allperm[len(row_i)]:
            tmp1 = np.arange(N0, dtype=np.int64)
            tmp1[row_i] = tmp1[row_i[x]]
            tmp0.append(tmp1)
        op_row_list.append(np.stack(tmp0))
    if len(op_row_list)>0:
        op_row = op_row_list[0]
        for x in op_row_list[1:]:
            op_row = op_row[:,x].reshape(-1,N0)
    else:
        op_row = None

    op_column_list = []
    column_list = [tableau[:y,x] for x,y in enumerate(youngT) if y>1]
    for column_i in column_list:
        tmp0 = []
        for x in n_to_allperm[len(column_i)]:
            tmp1 = np.arange(N0, dtype=np.int64)
            tmp1[column_i] = tmp1[column_i[x]]
            tmp0.append(tmp1)
        op_column_list.append(np.stack(tmp0))
    if len(op_column_list)>0:
        column_parity_list = [np.array([sympy.combinatorics.Permutation(y).parity() for y in x], dtype=np.int64) for x in op_column_list]
        op_column = op_column_list[0]
        op_sign = 1-2*column_parity_list[0]
        # if parity==0, sign=+1
        # if parity==1, sign=-1
        for x,y in zip(op_column_list[1:], column_parity_list[1:]):
            op_column = op_column[:,x].reshape(-1,N0)
            op_sign = (op_sign.reshape(-1,1)*(1-2*y)).reshape(-1)
    else:
        op_column = None
        op_sign = None

    if (op_row is None) and (op_column is None):
        ret = np.arange(N0, dtype=np.int64).reshape(1,-1), np.array([1])
    elif op_row is None:
        ret = op_column,op_sign
    elif op_column is None:
        ret = op_row, np.ones(len(op_row), dtype=np.int64)
    else:
        op_sign = (op_sign.reshape(-1,1)*np.ones(len(op_row),dtype=np.int64)).reshape(-1)
        op_young = op_column[:,op_row].reshape(-1,N0)
        ## no repeat element @YouningLI
        # op_young = [tuple(x) for x in op_column[:,op_row].reshape(-1,N0).tolist()]
        # index = [x[0] for x in sorted(list(enumerate(op_young)), key=lambda x:x[1])]
        # tmp1 = itertools.groupby([(x,op_young[x]) for x in index], key=lambda x:x[1])
        # index_group = [(x0,[y[0] for y in x1]) for x0,x1 in tmp1]
        # op_sign = np.array([sum(op_sign[y] for y in x) for _,x in index_group])
        # op_young = np.array([x[0] for x in index_group], dtype=np.int64)
        ret = op_young,op_sign
    return ret


def print_all_young_tableaux(N0:int):
    r'''Print all Young tableaux

    Parameters:
        N0 (int): the order of symmetric group
    '''
    assert N0>=1
    tmp0 = [tuple(y for y in x if y>0) for x in get_sym_group_young_diagram(N0).tolist()]
    young_tableaux = {x:get_all_young_tableaux(x) for x in tmp0}
    for key,value in young_tableaux.items():
        tmp0 = '[' + ','.join(str(x) for x in key) + ']:'
        print(tmp0, f'#tableaux: {len(value)}')
        mask = get_young_diagram_mask(key)
        for ind0 in range(len(value)):
            for x,y in zip(value[ind0],mask):
                print(x[:y.sum()].tolist())
            print(('=' if ind0==(len(value)-1) else '-')*30)

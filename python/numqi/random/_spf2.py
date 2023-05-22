import numpy as np

from ._internal import get_random_rng, get_numpy_rng
import numqi.group.spf2

def rand_F2(*size, not_zero=False, not_one=False, seed=None):
    r'''generate random uint8 binary ndarray with shape `size` and values in {0,1}

    Parameters:
        *size (tuple of int): shape of the ndarray
        not_zero (bool): if True, the generated ndarray cannot be all zero
        not_one (bool): if True, the generated ndarray cannot be all one
        seed (int,None,numpy.random.Generator): seed for the random number generator

    Returns:
        ret (np.ndarray): random uint8 binary ndarray with shape `size` and values in {0,1}
    '''
    np_rng = get_numpy_rng(seed)
    if not_zero and not_one:
        assert (len(size)>0) and (np.prod(size)>1)
    while True:
        ret = np_rng.integers(0, 2, size=size, dtype=np.uint8)
        if not_zero and np.array_equiv(ret, 0):
            continue
        if not_one and np.array_equiv(ret, 1):
            continue
        break
    return ret


def rand_SpF2_int_tuple(n, seed=None):
    rng = get_random_rng(seed)
    int_base = numqi.group.spf2.get_number(n, kind='base')
    ret = tuple(rng.randint(0,x-1) for x in int_base)
    return ret


def rand_SpF2(n, return_int_tuple=False, seed=None):
    rng = get_random_rng(seed)
    int_base = numqi.group.spf2.get_number(n, kind='base')
    int_tuple = tuple(rng.randint(0,x-1) for x in int_base)
    ret = numqi.group.spf2.from_int_tuple(int_tuple)
    if return_int_tuple:
        ret = int_tuple,ret
    return ret


def rand_Clifford_group(n:int, seed=None):
    assert n>=1
    rng = get_random_rng(seed)
    cli_r = rand_F2(2*n)
    cli_mat = rand_SpF2(n, seed=rng.randint(0, 2**32-1))
    return cli_r, cli_mat

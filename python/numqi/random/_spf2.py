import numpy as np

from ._public import get_random_rng, get_numpy_rng
import numqi.group.spf2
from ..gate._pauli import PauliOperator

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


def rand_SpF2(n:int, return_kind='matrix', seed=None):
    r'''generate random Symplectic matrix over finite field F2

    Parameters:
        n (int): half of the column/row of the matrix
        return_kind (str): return kind, one of {'matrix', 'int_tuple', 'int_tuple-matrix'}
        seed (int,None,numpy.random.Generator): seed for the random number generator

    Returns:
        ret (np.ndarray,tuple[int],tuple[int,np.ndarray]): random Symplectic matrix over finite field F2
            if return_kind=='matrix', return a matrix. shape=(2*n,2*n)
            if return_kind=='int_tuple', return a integer tuple representation. length=2*n
            if return_kind=='int_tuple-matrix', return a integer tuple representation and a matrix
    '''
    return_kind = str(return_kind).lower()
    assert return_kind in {'matrix', 'int_tuple', 'int_tuple-matrix'}
    rng = get_random_rng(seed)
    int_base = numqi.group.spf2.get_number(n, kind='base')
    int_tuple = tuple(rng.randint(0,x-1) for x in int_base)
    if return_kind=='int_tuple':
        ret = int_tuple
    else:
        mat = numqi.group.spf2.from_int_tuple(int_tuple)
        if return_kind=='matrix':
            ret = mat
        else: #int_tuple-matrix
            ret = int_tuple,mat
    return ret


def rand_Clifford_group(n:int, seed=None):
    r'''generate random Clifford group element in the symplectic representation

    Parameters:
        n (int): half of the column/row of the matrix
        seed (int,None,numpy.random.Generator): seed for the random number generator

    Returns:
        cli_r (np.ndarray): shape (`2n`,)
        cli_mat (np.ndarray): shape (`2n`,`2n`)
    '''
    assert n>=1
    rng = get_random_rng(seed)
    cli_r = rand_F2(2*n)
    cli_mat = rand_SpF2(n, seed=rng.randint(0, 2**32-1))
    return cli_r, cli_mat


def rand_pauli(n:int, is_hermitian:bool=None, seed=None):
    r'''generate random Pauli operator

    Parameters:
        n (int): number of qubits
        is_hermitian (bool): if True, the returned Pauli operator is Hermitian,
                if False, anti-Hermitian, if None, either Hermitian or anti-Hermitian
        seed (int,None,numpy.random.Generator): seed for the random number generator

    Returns:
        pauli (numqi.gate.PauliOperator): random Pauli operator
    '''
    assert n>=1
    assert is_hermitian in {None,True,False}
    np_rng = get_numpy_rng(seed)
    F2 = rand_F2(2*n+2, seed=np_rng)
    if is_hermitian is not None:
        tmp0 = np.dot(F2[2:(2+n)], F2[(2+n):]) % 2
        if is_hermitian:
            F2[1] = tmp0
        else:
            F2[1] = 1-tmp0
    ret = PauliOperator(F2)
    return ret

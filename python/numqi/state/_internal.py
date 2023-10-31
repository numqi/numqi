import numpy as np
import scipy.special

def W(n:int):
    ret = np.zeros(2**n, dtype=np.float64)
    ret[2**np.arange(n,dtype=np.int64)] = np.sqrt(1/n)
    return ret


def Wtype(np0):
    np0 = np0 / np.linalg.norm(np0)
    N0 = np0.shape[0]
    ret = np.zeros(2**N0, dtype=np0.dtype)
    ret[2**np.arange(N0)] = np0
    return ret

def GHZ(n:int=2):
    assert n>=1
    ret = np.zeros(2**n, dtype=np.float64)
    ret[[0,-1]] = 1/np.sqrt(2)
    return ret


def get_qubit_dicke_state_gm(n:int, k:int):
    r'''get the geometric measure of the Dicke state

    https://doi.org/10.1063/1.3464263

    Parameters:
        n(int): the number of qubits
        k(int): the number of excitations
    '''
    ret = 1 - scipy.special.binom(n, k) * ((k/n)**k) * (((n-k)/n)**(n-k))
    return ret



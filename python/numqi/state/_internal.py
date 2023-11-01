import numpy as np
import scipy.special

from ..utils import get_relative_entropy

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


def Werner(d:int, alpha:float):
    r'''get the Werner state

    https://en.wikipedia.org/wiki/Werner_state

    https://www.quantiki.org/wiki/werner-state

    alpha = ((1-2*p)*d+1) / (1-2*p+d)

    alpha: [-1,1]

    SEP: [-1,1/d]

    (1,k)-ext: [-1, (k+d^2-d)/(kd+d-1)]

    (1,k)-ext boundary: https://journals.aps.org/pra/abstract/10.1103/PhysRevA.88.032323

    Parameters:
        d(int): the dimension of the Hilbert space
        alpha(float): the parameter of the Werner state

    Returns:
        ret(np.ndarray): the density matrix of the Werner state
    '''
    assert d>1
    assert (-1<=alpha) and (alpha<=1)
    pmat = np.eye(d**2).reshape(d,d,d,d).transpose(0,1,3,2).reshape(d**2,d**2)
    ret = (np.eye(d**2)-alpha*pmat) / (d**2-d*alpha)
    return ret


def get_Werner_ree(d:int, alpha:float):
    r'''get the relative entropy of entanglement (REE) of the Werner state

    Parameters:
        d(int): the dimension of the Hilbert space
        alpha(float): the parameter of the Werner state

    Returns:
        ret(float): the relative entropy of entanglement of the Werner state
    '''
    if alpha<=1/d:
        ret = 0
    else:
        rho0 = Werner(d, alpha)
        rho1 = Werner(d, 1/d)
        ret = get_relative_entropy(rho0, rho1, kind='infinity')
    return ret


def Isotropic(d:int, alpha:float):
    r'''get the isotropic state

    https://www.quantiki.org/wiki/isotropic-state

    alpha: [-1/(d^2-1), 1]

    SEP: [-1/(d^2-1), 1/(d+1)]

    (1,k)-ext: [-1/(d^2-1),(kd+d^2-d-k)/(k(d^2-1))]

    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.88.032323

    Parameters:
        d(int): the dimension of the Hilbert space
        alpha(float): the parameter of the isotropic state

    Returns:
        ret(np.ndarray): the density matrix of the isotropic state
    '''
    assert d>1
    assert ((-1/(d**2-1))<=alpha) and (alpha<=1) #beyond this range, the density matrix is not PSD
    tmp0 = np.eye(d).reshape(-1)
    ret = ((1-alpha)/d**2) * np.eye(d**2) + (alpha/d) * (tmp0[:,np.newaxis]*tmp0)
    return ret


def get_Isotropic_ree(d:int, alpha:float):
    r'''get the relative entropy of entanglement (REE) of the isotropic state

    Parameters:
        d(int): the dimension of the Hilbert space
        alpha(float): the parameter of the isotropic state

    Returns:
        ret(float): the relative entropy of entanglement of the isotropic state
    '''
    if alpha<=1/(d+1):
        ret = 0
    else:
        rho0 = Isotropic(d, alpha)
        rho1 = Isotropic(d, 1/(d+1))
        ret = get_relative_entropy(rho0, rho1, kind='infinity')
    return ret

def maximally_entangled_state(d:int):
    r'''get the maximally entangled state

    https://www.quantiki.org/wiki/maximally-entangled-state

    Parameters:
        d(int): the dimension of the Hilbert space

    Returns:
        ret(np.ndarray): the maximally entangled state, `ret.ndim=1`
    '''
    assert d>1
    ret = np.diag(np.ones(d)*np.sqrt(1/d)).reshape(-1)
    return ret

# TODO AME

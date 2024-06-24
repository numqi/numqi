import numpy as np
import scipy.special

import numqi.utils

from numqi.dicke import Dicke

def W(n:int):
    r'''get the W-state [wiki-link](https://en.wikipedia.org/wiki/W_state)

    $$ |W\rangle=\frac{1}{\sqrt{n}}\left( |100\cdots 0\rangle + |010\cdots 0\rangle + |000\cdots 1\rangle \right) $$

    Parameters:
        n (int): the number of qubits

    Returns:
        ret (np.ndarray): the W-state, `ret.ndim=1`
    '''
    ret = np.zeros(2**n, dtype=np.float64)
    ret[2**np.arange(n,dtype=np.int64)] = np.sqrt(1/n)
    return ret


def Wtype(coeff:np.ndarray):
    r'''get the W-type state

    $$ |W\rangle=\sum_{wt(x)=1} c_i|x\rangle $$

    Parameters:
        coeff (np.ndarray): the coefficients of the W-type state, `coeff.ndim=1`

    Returns:
        ret (np.ndarray): the W-type state, `ret.ndim=1`
    '''
    coeff = coeff / np.linalg.norm(coeff)
    N0 = coeff.shape[0]
    ret = np.zeros(2**N0, dtype=coeff.dtype)
    ret[2**np.arange(N0)] = coeff
    return ret


def get_Wtype_state_GME(a:float, b:float, c:float):
    r'''get the geometric measure of the W-type state
    [arxiv-link](https://arxiv.org/abs/0710.0571)

    Analytic Expressions for Geometric Measure of Three Qubit States

    Parameters:
        a (float): the coefficient of |100>
        b (float): the coefficient of |010>
        c (float): the coefficient of |001>

    Returns:
        ret (float): the geometric measure of the W-type state
    '''
    # TODO broadcast
    assert abs(a*a+b*b+c*c-1) < 1e-10
    r1 = b**2 + c**2 - a**2
    r2 = a**2 + c**2 - b**2
    r3 = a**2 + b**2 - c**2
    if r1 > 0 and r2 > 0 and r3 > 0:
        w = 2*a*b
        tmp0 = (16*a*a*b*b*c*c - w*w + r3*r3) / (w*w - r3*r3)
        ret = 3/4 - tmp0/4
    else:
        ret = 1- max(a**2, b**2, c**2)
    return ret


def GHZ(n:int=2):
    r'''get the GHZ state
    [wiki-link](https://en.wikipedia.org/wiki/Greenberger%E2%80%93Horne%E2%80%93Zeilinger_state)

    Parameters:
        n (int): the number of qubits

    Returns:
        ret (np.ndarray): the GHZ state, `ret.ndim=1`
    '''
    assert n>=1
    ret = np.zeros(2**n, dtype=np.float64)
    ret[[0,-1]] = 1/np.sqrt(2)
    return ret


def Bell(i:int=0):
    r'''get the Bell state

    Parameters:
        i (int): the index of the Bell state, $i\in[0,3]$

    Returns:
        ret (np.ndarray): the Bell state, `ret.ndim=1`
    '''
    i = int(i)
    assert i in [0,1,2,3]
    if i==0:
        ret = np.array([1,0,0,1], dtype=np.float64) / np.sqrt(2)
    elif i==1:
        ret = np.array([1,0,0,-1], dtype=np.float64) / np.sqrt(2)
    elif i==2:
        ret = np.array([0,1,1,0], dtype=np.float64) / np.sqrt(2)
    else:
        ret = np.array([0,1,-1,0], dtype=np.float64) / np.sqrt(2)
    return ret


def get_qubit_dicke_state_GME(n:int, k:int):
    r'''get the geometric measure of entanglement for the Dicke state

    Matrix permanent and quantum entanglement of permutation invariant states [doi-link](https://doi.org/10.1063/1.3464263)

    Parameters:
        n (int): the number of qubits
        k (int): the number of excitations
    '''
    ret = 1 - scipy.special.binom(n, k) * ((k/n)**k) * (((n-k)/n)**(n-k))
    return ret


def Werner(d:int, alpha:float):
    r'''get the Werner state
    [wiki-link](https://en.wikipedia.org/wiki/Werner_state)
    [quantiki-link](https://www.quantiki.org/wiki/werner-state)

    alpha: $[-1,1]$

    SEP: $\left[-1,\frac{1}{d} \right]$

    (1,k)-ext: $\left[-1, \frac{k+d^2-d}{kd+d-1} \right]$

    (1,k)-ext boundary: Compatible quantum correlations: Extension problems for Werner and isotropic states
    [doi-link](https://doi.org/10.1103/PhysRevA.88.032323)

    Parameters:
        d (int): the dimension of the Hilbert space
        alpha (float): the parameter of the Werner state

    Returns:
        ret (np.ndarray): the density matrix of the Werner state
    '''
    assert d>1
    assert (-1<=alpha) and (alpha<=1)
    pmat = np.eye(d**2).reshape(d,d,d,d).transpose(0,1,3,2).reshape(d**2,d**2)
    ret = (np.eye(d**2)-alpha*pmat) / (d**2-d*alpha)
    return ret


def get_Werner_ree(d:int, alpha:float):
    r'''get the relative entropy of entanglement (REE) of the Werner state

    Parameters:
        d (int): the dimension of the Hilbert space
        alpha (float): the parameter of the Werner state

    Returns:
        ret (float): the relative entropy of entanglement of the Werner state
    '''
    if alpha<=1/d:
        ret = 0
    else:
        rho0 = Werner(d, alpha)
        rho1 = Werner(d, 1/d)
        ret = numqi.utils.get_relative_entropy(rho0, rho1)
    return ret


def get_Werner_GME(d:int, alpha:float|np.ndarray):
    r'''get the geometric measure of entanglement (GME) of the Werner state

    Geometric measure of entanglement and applications to bipartite and multipartite quantum states
    [doi-link](https://doi.org/10.1103/PhysRevA.68.042307) (eq-51)

    Parameters:
        d (int): the dimension of the Hilbert space
        alpha (float,np.ndarray): the parameter of the Werner state

    Returns:
        ret (float,np.ndarray): the geometric measure of entanglement of the Werner state
    '''
    assert d>=2
    tmp0 = d - (1-d*d) / (alpha - d)
    ret = (tmp0<=0)*(1-np.sqrt(np.maximum(0,1-tmp0*tmp0)))/2
    return ret


def get_Werner_eof(dim:int, alpha:np.ndarray|float):
    r'''get the entanglement of formation (EOF) of the Werner state

    reference: Entanglement of formation and concurrence for mixed states
    [doi-link](https://doi.org/10.1007/s11704-008-0017-8)

    Parameters:
        dim (int): the dimension of the Hilbert space
        alpha (np.ndarray,float): the parameter of the Werner state

    Returns:
        ret (np.ndarray,float): the entanglement of formation of the Werner state
    '''
    alpha = np.asarray(alpha)
    shape = alpha.shape
    alpha = alpha.reshape(-1)
    a = (1-alpha*dim) / (dim-alpha)
    ret = np.zeros(alpha.shape[0], dtype=np.float64)
    ind0 = a<0
    if np.any(ind0):
        a = a[ind0]
        tmp0 = (1-np.sqrt(1-a*a))/2
        ret[ind0] = -tmp0*np.log(tmp0) - (1-tmp0)*np.log(1-tmp0)
    ret = ret.reshape(shape)
    return ret


def Isotropic(d:int, alpha:float):
    r'''get the isotropic state [quantiki-link](https://www.quantiki.org/wiki/isotropic-state)

    alpha: $\left[-\frac{1}{d^2-1}, 1\right]$

    SEP: $\left[-\frac{1}{d^2-1}, \frac{1}{d+1}\right]$

    (1,k)-ext: $\left[-\frac{1}{d^2-1}, \frac{kd+d^2-d-k}{k(d^2-1)}\right]$

    Compatible quantum correlations: Extension problems for Werner and isotropic states
    [doi-link](https://doi.org/10.1103/PhysRevA.88.032323)

    Parameters:
        d (int): the dimension of the Hilbert space
        alpha (float): the parameter of the isotropic state

    Returns:
        ret (np.ndarray): the density matrix of the isotropic state
    '''
    assert d>1
    assert ((-1/(d**2-1))<=alpha) and (alpha<=1) #beyond this range, the density matrix is not PSD
    tmp0 = np.eye(d).reshape(-1)
    ret = ((1-alpha)/d**2) * np.eye(d**2) + (alpha/d) * (tmp0[:,np.newaxis]*tmp0)
    return ret


def get_Isotropic_ree(d:int, alpha:float):
    r'''get the relative entropy of entanglement (REE) of the isotropic state

    Parameters:
        d (int): the dimension of the Hilbert space
        alpha (float): the parameter of the isotropic state

    Returns:
        ret (float): the relative entropy of entanglement of the isotropic state
    '''
    if alpha<=1/(d+1):
        ret = 0
    else:
        rho0 = Isotropic(d, alpha)
        rho1 = Isotropic(d, 1/(d+1))
        ret = numqi.utils.get_relative_entropy(rho0, rho1)
    return ret

def get_Isotropic_GME(d:int, alpha:float|np.ndarray):
    r'''get the geometric measure of entanglement (GME) of the isotropic state

    Geometric measure of entanglement and applications to bipartite and multipartite quantum states
    [doi-link](https://doi.org/10.1103/PhysRevA.68.042307) (eq-54)

    Parameters:
        d (int): the dimension of the Hilbert space
        alpha (float,np.ndarray): the parameter of the isotropic state

    Returns:
        ret (float,np.ndarray): the geometric measure of entanglement of the isotropic state
    '''
    assert d>=2
    tmp0 = np.clip(alpha + (1-alpha)/(d*d), 0, 1)
    tmp1 = 1 - ((np.sqrt(tmp0) + np.sqrt((1-tmp0)*(d-1)))**2)/d
    ret = (tmp0>=(1/d)) * tmp1
    return ret


def get_Isotropic_eof(dim:int, alpha:np.ndarray|float):
    r'''get the entanglement of formation (EOF) of the isotropic state

    reference: Entanglement of formation and concurrence for mixed states
    [doi-link](https://doi.org/10.1007/s11704-008-0017-8)

    Parameters:
        dim (int): the dimension of the Hilbert space
        alpha (np.ndarray,float): the parameter of the isotropic state

    Returns:
        ret(np.ndarray,float): the entanglement of formation of the isotropic state
    '''
    alpha = np.asarray(alpha)
    shape = alpha.shape
    alpha = alpha.reshape(-1)
    ret = np.zeros(alpha.shape[0], dtype=np.float64)
    F = (1+alpha*dim*dim-alpha)/(dim*dim)
    ind0 = np.logical_and(F>1/dim, F<=(4*(dim-1)/(dim*dim)))
    if np.any(ind0):
        gamma = (np.sqrt(F[ind0])+np.sqrt((dim-1)*(1-F[ind0])))**2/dim
        tmp0 = -gamma*np.log(gamma) - (1-gamma)*np.log(1-gamma)
        tmp1 = (1-gamma)*np.log(dim-1)
        ret[ind0] = tmp0 + tmp1
    ind1 = F>(4*(dim-1)/(dim*dim))
    if np.any(ind1):
        ret[ind1] = dim*np.log(dim-1)*(F[ind1]-1)/(dim-2) + np.log(dim)
    ret = ret.reshape(shape)
    return ret


def maximally_entangled_state(d:int):
    r'''get the maximally entangled state
    [quantiki-link](https://www.quantiki.org/wiki/maximally-entangled-state)

    Parameters:
        d (int): the dimension of the Hilbert space

    Returns:
        ret (np.ndarray): the maximally entangled state, `ret.ndim=1`
    '''
    assert d>1
    ret = np.diag(np.ones(d)*np.sqrt(1/d)).reshape(-1)
    return ret


def maximally_mixed_state(d:int):
    r'''get the maximally mixed state

    Parameters:
        d (int): the dimension of the Hilbert space

    Returns:
        ret (np.ndarray): the maximally mixed state, `ret.ndim=2` of shape $(d^2,d^2)$
    '''
    assert d>=1
    ret = np.eye(d*d) / d*d
    return ret


# TODO AME

def get_2qutrit_Antoine2022(q:float) -> np.ndarray:
    r'''an example of SEP-PPT-NPT states in 2-qutrit system

    reference: Building separable approximations for quantum states via neural networks
    [doi-link](https://doi.org/10.1103/PhysRevResearch.4.023238)

    [0,0.5]: separable
    (0.5,1.5]: PPT
    (1.5,2.5]: NPT

    Parameters:
        q (float): q in [-2.5,2.5]

    Returns:
        rho (np.ndarray): 9x9 density matrix
    '''
    q = float(q)
    assert -2.5<=q<=2.5
    betap = (2.5 + q) / 21
    betam = (2.5 - q) / 21
    np0 = np.diag(np.array([2/21,betam,betap,betap,2/21,betam,betam,betap,2/21], dtype=np.float64))
    np0[[0,0,4,4,8,8],[4,8,0,8,0,4]] = 2/21
    return np0


def get_bes2x4_Horodecki1997(b:float):
    r'''get the 2x4 bound entangled state proposed by Horodecki et al. in 1997

    reference: Separability criterion and inseparable mixed states with positive partial transposition
    [doi-link](https://doi.org/10.1016/S0375-9601(97)00416-7)

    reference: Certifying Quantum Separability with Adaptive Polytopes
    [arxiv-link](https://arxiv.org/abs/2210.10054)

    b in [0,1]

    PPT range of b: [0, 1]

    SEP: b=0 or b=1

    Parameters:
        b (float): the parameter of the state

    Returns:
        ret (np.ndarray): the density matrix of the state, shape=(8,8)
    '''
    assert (b >= 0) and (b <= 1)
    ret = np.eye(8, dtype=np.float64)*(b/(7*b+1))
    ret[[0,1,2,5,6,7], [5,6,7,0,1,2]] = b/(7*b+1)
    ret[[4,7],[4,7]] = (1+b)/(14*b+2)
    ret[[4,7], [7,4]] = np.sqrt(1-b*b)/(14*b+2)
    return ret


def get_bes3x3_Horodecki1997(a:float):
    r'''get the 3x3 bound entangled state proposed by Horodecki et al. in 1997

    reference: Separability criterion and inseparable mixed states with positive partial transposition
    [doi-link](https://doi.org/10.1016/S0375-9601(97)00416-7)

    reference: Certifying Quantum Separability with Adaptive Polytopes
    [arxiv-link](https://arxiv.org/abs/2210.10054)

    a in [0,1]

    PPT range of a: [0, 1]

    SEP: a=0 or a=1

    Parameters:
        a (float): the parameter of the state

    Returns:
        ret (np.ndarray): the density matrix of the state, shape=(9,9)
    '''
    assert (a >= 0) and (a <= 1)
    ret = np.eye(9, dtype=np.float64)*(a/(8*a+1))
    ret[[0,0,4,4,8,8],[4,8,0,8,0,4]] = a/(8*a+1)
    ret[[6,8],[6,8]] = (1+a)/(16*a+2)
    ret[[6,8], [8,6]] = np.sqrt(1-a*a)/(16*a+2)
    return ret


def maximally_coherent_state(d:int, return_dm:bool=False):
    r'''get the maximally coherent state

    reference: [arxiv-link](https://arxiv.org/abs/1503.07103)

    Parameters:
        d (int): the dimension of the Hilbert space
        return_dm (bool): whether to return the density matrix

    Returns:
        ret (np.ndarray): the maximally coherent state, `ret.ndim=1` or `ret.ndim=2`
    '''
    assert d>=1
    if return_dm:
        ret = np.eye(d, dtype=np.float64) / d
    else:
        ret = np.ones(d, dtype=np.float64) / np.sqrt(d)
    return ret


def get_4qubit_special_state_gme(key:str, plist:float|np.ndarray):
    r'''get the geometric measure of entanglement (GME) for some special 4-qubit states

    reference: Multiparticle entanglement under the influence of decoherence
    [doi-link](https://doi.org/10.1103/PhysRevA.78.060301)

    Parameters:
        key (str): the type of the special state, one of {'cluster','ghz','w','dicke'}
        plist (float,np.ndarray): the decoherence parameter, $p\in[0,1]$. For $p=0$, the state is incoherent (diagonal only)

    Returns:
        rho (np.ndarray): the density matrix of the special state, shape=(16,16), or (n,16,16) if `plist.ndim=1`
        gme (float,np.ndarray): GME of the special state, or (n,) if `plist.ndim=1`
    '''
    assert key in {'cluster','ghz','w','dicke'}
    plist = np.asarray(plist)
    assert plist.ndim in {0,1}
    isone = (plist.ndim==0)
    plist = plist.reshape(-1)
    assert np.all(plist >= 0) and np.all(plist <= 1)
    if key == 'cluster':
        # (0000 + 0011 + 1100 - 1111)/2
        psi_cluster = np.zeros(16, dtype=np.float64)
        psi_cluster[[0, 3, 12, 15]] = np.array([1,1,1,-1])/2
        rho = psi_cluster.reshape(-1,1) * psi_cluster.conj()
        gme = (3/8)*(1 + plist - np.sqrt(1+(2-3*plist)*plist))
    elif key == 'ghz':
        # (0000 + 1111)/sqrt(2)
        psi_ghz = numqi.state.GHZ(4)
        rho = psi_ghz.reshape(-1,1) * psi_ghz.conj()
        gme = (1/2)*(1-np.sqrt(1-plist*plist))
    elif key == 'w':
        psi_W = numqi.state.W(4)
        rho = psi_W.reshape(-1,1) * psi_W.conj()
        tmp0 = plist>(2183/2667)
        gme = tmp0 * (37*(81*plist-37)/2816) + (1-tmp0) * (3/8)*(1+plist-np.sqrt(1+(2-3*plist)*plist))
    elif key == 'dicke':
        psi_dicke = numqi.state.Dicke(2, 2)
        rho = psi_dicke.reshape(-1,1) * psi_dicke.conj()
        tmp0 = (plist > 5/7)
        gme = tmp0 * (5*(3*plist-1)/16) + (1-tmp0) * (5/18)*(1+2*plist-np.sqrt(1+(4-5*plist)*plist))
    # dim_list = (2,2,2,2)
    mask_diag = np.eye(rho.shape[0], dtype=np.float64)
    mask_offdiag = 1-mask_diag
    rho_list = rho*mask_diag + (rho*mask_offdiag)*plist.reshape(-1,1,1)
    ret = (rho_list[0], gme[0]) if isone else (rho_list, gme)
    return ret

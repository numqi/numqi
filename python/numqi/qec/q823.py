import numpy as np
import scipy.linalg

import numqi.gate

from ._pauli import hf_pauli

def get_BD38_LP(coeff2=None, sign=None, phase0=None, return_info=True, seed=None):
    matA = np.array([[1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1],
                [1,1,1,1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,-1],
                [1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,-1,-1],
                [1,1,1,-1,1,1,-1,1,-1,-1,-1,1,1,1,-1],
                [1,1,-1,1,1,-1,1,-1,1,-1,-1,1,1,-1,1],
                [1,1,-1,-1,1,-1,-1,1,1,-1,1,-1,-1,1,1],
                [1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,-1,-1],
                [1,-1,1,1,-1,1,1,-1,-1,1,-1,-1,-1,1,1],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
    vecb = np.array([0]*8+[1])
    if (coeff2 is None) or (sign is None) or (phase0 is None):
        np_rng = np.random.default_rng(seed)
    phase0 = np_rng.uniform(0, 2*np.pi) if phase0 is None else float(phase0)
    if sign is None:
        sign = np_rng.integers(2, size=14)*2-1
    else:
        sign = tuple(int(x) for x in sign)
        assert (len(sign)==14) and all(x in [1,-1] for x in sign)
        sign = np.asarray(sign, dtype=np.int64)
    phase = np.concatenate([np.exp(1j*phase0).reshape(-1), sign])
    if coeff2 is not None:
        assert (coeff2.shape==(15,)) and np.all(coeff2>=0)
        assert np.abs(matA @ coeff2 - vecb).max() < 1e-10
        coeff2[0] = 1/38
        coeff2[10] = 4/19
    else:
        matB = scipy.linalg.null_space(matA) #(15,6)
        x0 = np.array([8, 10, 23, 23, 10, 23, 23, 10, 10, 12, 64, 24, 24, 20, 20])/(19*16)
        tmp0 = np_rng.normal(size=matB.shape[1])
        tmp1 = matB @ (tmp0/np.linalg.norm(tmp0))
        ind0 = tmp1>1e-5
        ind1 = tmp1 < -1e-5
        assert np.any(ind0) and np.any(ind1)
        tmp2 = np_rng.uniform((-x0[ind0]/tmp1[ind0]).max(), (-x0[ind1]/tmp1[ind1]).min())
        coeff2 = np.maximum(0, x0 + tmp2*tmp1)
    coeff = np.sqrt(coeff2) * phase
    basis0 = np.zeros((15, 2**8), dtype=np.complex128)
    basis0[np.arange(7), [0, 35, 46, 54, 67, 78, 86]] = 1
    basis0[np.arange(7,15), [105, 113, 124, 153, 165, 197, 234, 242]] = 1j
    basis1 = (hf_pauli('X'*8) @ basis0.T).T
    code = np.stack([coeff@basis0,coeff@basis1], axis=0)
    ret = code
    if return_info:
        su2 = numqi.gate.rz(np.array([2,3,3,4,4,5,7,9])*2*np.pi/19)
        info = dict(su2=su2, coeff=coeff, basis0=basis0, basis1=basis1, coeff2=coeff2, sign=sign, phase0=phase0)
        ret = code, info
    return ret


def get_BD64(theta, sign, return_info:bool=False):
    theta = np.asarray(theta, dtype=np.float64)
    assert theta.shape==(3,)
    if sign is None:
        sign = np.array([1]*9, dtype=np.int64)
    else:
        sign = tuple(int(x) for x in sign)
        assert len(sign) == 9
        assert all(x in [1,-1] for x in sign)
        sign = np.asarray(sign, dtype=np.int64)
    s = np.sqrt
    basis0 = np.zeros((9, 2**8), dtype=np.float64)
    basis0[np.arange(9), [0,14,35,87,102,105,189,197,218]] = 1
    basis1 = (hf_pauli('X'*8) @ basis0.T).T
    coeff = np.array([s(5/32), 1/8*np.exp(1j*theta[2]), s(3/32)*1j, 1/4*1j, s(1/8),
            s(3/64)*np.exp(1j*theta[0]), s(15/64)*1j, 1/4*np.exp(1j*theta[1]), s(13/64)])*sign
    code = np.stack([coeff@basis0, coeff@basis1], axis=0)
    ret = code
    if return_info:
        su2 = numqi.gate.rz(np.array([4,7,10,12,17,23,24,30])*2*np.pi/32) #Z(-2pi/32)
        # TODO weight enumeartor
        info = dict(theta=theta, sign=sign, basis0=basis0, basis1=basis1, coeff=coeff, su2=su2)
        ret = code,info
    return ret


def get_BD72(theta=None, sign=None, return_info=True, seed=None):
    if (theta is None) or (sign is None):
        np_rng = np.random.default_rng(seed)
    if theta is None:
        theta = np_rng.uniform(0, 2*np.pi, size=5)
    else:
        assert len(theta)==5
    if sign is None:
        sign = np_rng.integers(2, size=3)*2-1
    else:
        assert len(sign)==3
        sign = [int(x) for x in sign]
        assert all((x==1 or x==-1) for x in sign)
    hf0 = lambda i: np.exp(1j*theta[i])
    phase = np.array([1,hf0(0),hf0(1),sign[0]*hf0(1),hf0(2),
                hf0(3),1j*sign[1]*hf0(1),hf0(4),1j*sign[2]*hf0(0)])
    coeff = np.sqrt(np.array([1,14,3,13,5,8,10,6,12])/72) * phase
    basis0 = np.zeros((9, 2**8), dtype=np.complex128)
    basis0[np.arange(9), [0,13,60,90,102,150,163,236,241]] = 1
    basis1 = (hf_pauli('XXXXXXXX') @ basis0.T).T
    code = np.stack([coeff@basis0, coeff@basis1], axis=0)
    ret = code
    if return_info:
        su2 = numqi.gate.rz(np.array([3,5,6,8,10,12,13,14])*2*np.pi/36) #rz(-2*np.pi/38)
        info = dict(theta=theta, sign=sign, coeff=coeff, basis0=basis0, basis1=basis1, su2=su2)
        ret = code, info
    return ret


def get_BD74(theta=None, sign=None, return_info=True, seed=None):
    if (theta is None) or (sign is None):
        np_rng = np.random.default_rng(seed)
    if theta is None:
        theta = np_rng.uniform(0, 2*np.pi, size=4)
    else:
        assert len(theta)==4
    if sign is None:
        sign = np_rng.integers(2, size=4)*2-1
    else:
        assert len(sign)==4
        sign = [int(x) for x in sign]
        assert all((x==1 or x==-1) for x in sign)
    hf0 = lambda i: np.exp(1j*theta[i])
    phase = np.array([1, hf0(0), hf0(1), sign[0]*hf0(1), hf0(2),
                       1j*sign[1]*hf0(2), hf0(3), 1j*sign[2]*hf0(1), 1j*sign[3]*hf0(0)])
    coeff = np.sqrt(np.array([1,12,3,11,10,6,7,8,16])/74) * phase
    basis0 = np.zeros((9, 2**8), dtype=np.complex128)
    basis0[np.arange(9), [0,7,60,90,105,142,153,163,244]] = 1
    basis1 = (hf_pauli('XXXXXXXX') @ basis0.T).T
    code = np.stack([coeff@basis0, coeff@basis1], axis=0)
    ret = code
    if return_info:
        su2 = numqi.gate.rz(np.array([4,6,7,9,10,11,12,14])*2*np.pi/37) #rz(-2*np.pi/37)
        info = dict(theta=theta, sign=sign, coeff=coeff, basis0=basis0, basis1=basis1, su2=su2)
        ret = code, info
    return ret


def get_BD76(theta=None, sign=None, return_info=True, seed=None):
    if (theta is None) or (sign is None):
        np_rng = np.random.default_rng(seed)
    if theta is None:
        theta = np_rng.uniform(0, 2*np.pi, size=6)
    else:
        assert len(theta)==6
    if sign is None:
        sign = np_rng.integers(2, size=2)*2-1
    else:
        assert len(sign)==2
        sign = [int(x) for x in sign]
        assert all((x==1 or x==-1) for x in sign)
    hf0 = lambda i: np.exp(1j*theta[i])
    phase = np.array([1,hf0(0),hf0(1),hf0(2),hf0(3),hf0(4),hf0(5),1j*sign[0]*hf0(0),1j*sign[1]*hf0(0)])
    coeff = np.sqrt(np.array([1,18,4,3,7,5,12,10,16])/76)*phase
    basis0 = np.zeros((9, 2**8), dtype=np.complex128)
    basis0[np.arange(9), [0,13,35,60,90,102,150,234,241]] = 1
    basis1 = (hf_pauli('XXXXXXXX') @ basis0.T).T
    code = np.stack([coeff@basis0, coeff@basis1], axis=0)
    ret = code
    if return_info:
        su2 = numqi.gate.rz(np.array([2,4,7,9,10,12,15,16])*2*np.pi/38) #rz(-2*np.pi/38)
        info = dict(theta=theta, sign=sign, coeff=coeff, basis0=basis0, basis1=basis1, su2=su2)
        ret = code, info
    return ret


def get_BD78(theta=None, sign=None, return_info=True, seed=None):
    if (theta is None) or (sign is None):
        np_rng = np.random.default_rng(seed)
    if theta is None:
        theta = np_rng.uniform(0, 2*np.pi, size=6)
    else:
        assert len(theta)==6
    if sign is None:
        sign = np_rng.integers(2, size=2)*2-1
    else:
        assert len(sign)==2
        sign = [int(x) for x in sign]
        assert all((x==1 or x==-1) for x in sign)
    hf0 = lambda i: np.exp(1j*theta[i])
    phase = np.array([1,hf0(0),hf0(1),hf0(2),hf0(3),  1j*sign[0]*hf0(3),hf0(4),1j*sign[1]*hf0(0), hf0(5)])
    coeff = np.sqrt(np.array([1,16,3,5,14,10,9,12,8])/78) * phase
    basis0 = np.zeros((9, 2**8), dtype=np.complex128)
    basis0[np.arange(9), [0,19,60,102,105,142,165,220,242]] = 1
    basis1 = (hf_pauli('XXXXXXXX') @ basis0.T).T
    code = np.stack([coeff@basis0, coeff@basis1], axis=0)
    ret = code
    if return_info:
        su2 = numqi.gate.rz(np.array([3,5,8,9,10,12,14,16])*2*np.pi/39) #rz(-2*np.pi/39)
        info = dict(theta=theta, sign=sign, coeff=coeff, basis0=basis0, basis1=basis1, su2=su2)
        ret = code, info
    return ret


def get_BD80(theta=None, sign=None, return_info=True, seed=None):
    if (theta is None) or (sign is None):
        np_rng = np.random.default_rng(seed)
    if theta is None:
        theta = np_rng.uniform(0, 2*np.pi, size=6)
    else:
        assert len(theta)==6
    if sign is None:
        sign = np_rng.integers(2, size=2)*2-1
    else:
        assert len(sign)==2
        sign = [int(x) for x in sign]
        assert all((x==1 or x==-1) for x in sign)
    hf0 = lambda i: np.exp(1j*theta[i])
    phase = np.array([1,hf0(0),hf0(1),hf0(2),hf0(3),  hf0(4),1j*sign[0]*hf0(2),1j*sign[1]*hf0(2),hf0(5)])
    coeff = np.sqrt(np.array([1,2,8,18,5, 6,11,15,14])/80)*phase
    basis0 = np.zeros((9, 2**8), dtype=np.complex128)
    basis0[np.arange(9), [0,14,21,58,102,105,165,195,220]] = 1
    basis1 = (hf_pauli('XXXXXXXX') @ basis0.T).T
    code = np.stack([coeff@basis0, coeff@basis1], axis=0)
    ret = code
    if return_info:
        su2 = numqi.gate.rz(np.array([2,5,6,8,11,14,15,18])*2*np.pi/40) #rz(-2*np.pi/40)
        info = dict(theta=theta, sign=sign, coeff=coeff, basis0=basis0, basis1=basis1, su2=su2)
        ret = code, info
    return ret


def get_BD84(theta=None, sign:int=None, return_info=True, seed=None):
    if (theta is None) or (sign is None):
        np_rng = np.random.default_rng(seed)
    if theta is None:
        theta = np_rng.uniform(0, 2*np.pi, size=7)
    else:
        assert len(theta)==7
    if sign is None:
        sign = (np_rng.integers(2)*2-1).item()
    else:
        sign = int(sign)
        assert sign in [1, -1]
    hf0 = lambda i: np.exp(1j*theta[i])
    phase = np.array([1,hf0(0),hf0(1),hf0(2),hf0(3),  hf0(4),hf0(5),hf0(6),1j*sign*hf0(5)])
    coeff = np.sqrt(np.array([10,4,2,11,12,3,17,9,16])/84)*phase
    basis0 = np.zeros((9, 2**8), dtype=np.complex128)
    basis0[np.arange(9), [0,21,59,90,102,105,143,188,241]] = 1
    basis1 = (hf_pauli('XXXXXXXX') @ basis0.T).T
    code = np.stack([coeff@basis0, coeff@basis1], axis=0)
    ret = code
    if return_info:
        su2 = numqi.gate.rz(np.array([4,10,13,18,22,27,34,39])*2*np.pi/42) #rz(-2*np.pi/42)
        info = dict(theta=theta, sign=sign, coeff=coeff, basis0=basis0, basis1=basis1, su2=su2)
        ret = code, info
    return ret

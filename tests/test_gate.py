import numpy as np
import scipy.linalg

import numpyqi

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

PauliX = np.array([[0,1], [1,0]])
PauliY = np.array([[0,-1j], [1j,0]])
PauliZ = np.array([[1,0], [0,-1]])


def test_pauli_exponential():
    np_rng = np.random.default_rng()
    para = np_rng.normal(size=3)

    ret0 = numpyqi.gate.pauli_exponential(*para)
    tmp0 = (numpyqi.gate.X, numpyqi.gate.Y, numpyqi.gate.Z)
    tmp1 = para[0]*np.sin(para[1])*np.cos(para[2]), para[0]*np.sin(para[1])*np.sin(para[2]), para[0]*np.cos(para[1])
    ret_ = scipy.linalg.expm(1j*sum(x*y for x,y in zip(tmp0,tmp1)))
    assert np.abs(ret_-ret0).max() < 1e-7


def test_rzz():
    np_rng = np.random.default_rng()
    para = np_rng.normal(size=1)
    ret0 = numpyqi.gate.rzz(para)
    ret_ = scipy.linalg.expm(-1j*para*np.kron(numpyqi.gate.Z, numpyqi.gate.Z)/2)
    assert np.abs(ret0-ret_).max() < 1e-7


def test_rx():
    tmp0 = np.random.randn(5)
    ret_ = np.stack([scipy.linalg.expm(-1j*x/2*PauliX) for x in tmp0])
    ret0 = numpyqi.gate.rx(tmp0)
    assert hfe(ret_, ret0) < 1e-7


def test_ry():
    tmp0 = np.random.randn(5)
    ret_ = np.stack([scipy.linalg.expm(-1j*x/2*PauliY) for x in tmp0])
    ret0 = numpyqi.gate.ry(tmp0)
    assert hfe(ret_, ret0) < 1e-7


def test_rz():
    tmp0 = np.random.randn(5)
    ret_ = np.stack([scipy.linalg.expm(-1j*x/2*PauliZ) for x in tmp0])
    ret0 = numpyqi.gate.rz(tmp0)
    assert hfe(ret_, ret0) < 1e-7


def test_u3():
    parameter = np.random.randn(3, 5)
    ret_ = []
    for theta,phi,lambda_ in parameter.T:
        tmp0 = np.exp(0.5j*(phi+lambda_))
        ret_.append(tmp0*numpyqi.gate.rz(phi) @ numpyqi.gate.ry(theta) @ numpyqi.gate.rz(lambda_))
    ret_ = np.stack(ret_)
    ret0 = numpyqi.gate.u3(*parameter)
    assert hfe(ret_, ret0) < 1e-7

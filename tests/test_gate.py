import numpy as np
import scipy.linalg

import numqi

try:
    import torch
except ImportError:
    torch = None

np_rng = np.random.default_rng()

PauliX = np.array([[0,1], [1,0]])
PauliY = np.array([[0,-1j], [1j,0]])
PauliZ = np.array([[1,0], [0,-1]])


def test_pauli_exponential():
    np_rng = np.random.default_rng()
    para = np_rng.normal(size=3)

    ret0 = numqi.gate.pauli_exponential(*para)
    tmp0 = (numqi.gate.X, numqi.gate.Y, numqi.gate.Z)
    tmp1 = para[0]*np.sin(para[1])*np.cos(para[2]), para[0]*np.sin(para[1])*np.sin(para[2]), para[0]*np.cos(para[1])
    ret_ = scipy.linalg.expm(1j*sum(x*y for x,y in zip(tmp0,tmp1)))
    assert np.abs(ret_-ret0).max() < 1e-7


def test_rzz():
    np_rng = np.random.default_rng()
    para = np_rng.normal(size=1)
    ret0 = numqi.gate.rzz(para)
    ret_ = scipy.linalg.expm(-1j*para*np.kron(numqi.gate.Z, numqi.gate.Z)/2)
    assert np.abs(ret0-ret_).max() < 1e-7


def test_rx():
    tmp0 = np.random.randn(5)
    ret_ = np.stack([scipy.linalg.expm(-1j*x/2*PauliX) for x in tmp0])
    ret0 = numqi.gate.rx(tmp0)
    assert np.abs(ret_ - ret0).max() < 1e-7


def test_ry():
    tmp0 = np.random.randn(5)
    ret_ = np.stack([scipy.linalg.expm(-1j*x/2*PauliY) for x in tmp0])
    ret0 = numqi.gate.ry(tmp0)
    assert np.abs(ret_ - ret0).max() < 1e-7


def test_rz():
    tmp0 = np.random.randn(5)
    ret_ = np.stack([scipy.linalg.expm(-1j*x/2*PauliZ) for x in tmp0])
    ret0 = numqi.gate.rz(tmp0)
    assert np.abs(ret_ - ret0).max() < 1e-7


def test_u3():
    parameter = np.random.randn(3, 5)
    ret_ = []
    for theta,phi,lambda_ in parameter.T:
        tmp0 = np.exp(0.5j*(phi+lambda_))
        ret_.append(tmp0*numqi.gate.rz(phi) @ numqi.gate.ry(theta) @ numqi.gate.rz(lambda_))
    ret_ = np.stack(ret_)
    ret0 = numqi.gate.u3(*parameter)
    assert np.abs(ret_ - ret0).max() < 1e-7


def test_quditXZH():
    for d in range(2,7):
        X = numqi.gate.get_quditX(d)
        Z = numqi.gate.get_quditZ(d)
        H = numqi.gate.get_quditH(d)
        assert np.abs(X - H @ Z @ H.T.conj()).max() < 1e-10


def test_rx_qudit():
    theta = np_rng.uniform(0, 1, size=23)
    z0 = numqi.gate.rx(theta)
    z1 = numqi.gate._internal._rx_qudit(theta, 2)
    assert np.abs(z0-z1).max() < 1e-10

    for d in range(2,6):
        theta = np_rng.uniform(0, 1, size=23)
        z0 = numqi.gate.rx(theta, d)
        z1 = numqi.gate.rx(theta+4*np.pi, d)
        assert np.abs(np.linalg.det(z0)-1).max() < 1e-10
        assert np.abs(z0-z1).max() < 1e-10

        if torch is not None:
            z2 = numqi.gate.rx(torch.tensor(theta,dtype=torch.float64), d).numpy()
            assert np.abs(z0-z2).max() < 1e-10


def test_rz_qudit():
    theta = np_rng.uniform(0, 1, size=23)
    z0 = numqi.gate.rz(theta)
    z1 = numqi.gate._internal._rz_qudit(theta, 2, diag_only=False)
    assert np.abs(z0-z1).max() < 1e-10

    for d in range(2, 7):
        theta = np_rng.uniform(0, 1, size=23)
        z0 = numqi.gate.rx(theta, d)
        H = numqi.gate.get_quditH(d)
        z1 = numqi.gate.rz(theta, d, diag_only=False)
        assert np.abs(z0 - H @ z1 @ H.T.conj()).max() < 1e-10

        if torch is not None:
            z2 = numqi.gate.rz(torch.tensor(theta, dtype=torch.float64), d, diag_only=False).numpy()
            assert np.abs(z1-z2).max() < 1e-10

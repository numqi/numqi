import numpy as np
import scipy.linalg
import torch

import numqi

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

        z2 = numqi.gate.rz(torch.tensor(theta, dtype=torch.float64), d, diag_only=False).numpy()
        assert np.abs(z1-z2).max() < 1e-10


def test_PauliOperator_convert():
    example_list = [
        (numqi.gate.I, [0,0,0,0]),
        (numqi.gate.X, [0,0,1,0]),
        (numqi.gate.Y, [0,1,1,1]),
        (numqi.gate.Z, [0,0,0,1]),
        (np.kron(numqi.gate.X, numqi.gate.X), [0,0,1,1,0,0]),
        (np.kron(numqi.gate.X, numqi.gate.Y), [0,1,1,1,0,1]),
        (1j * np.kron(numqi.gate.Y, numqi.gate.Y), [1,1,1,1,1,1]),
    ]
    for np0,index in example_list:
        ret_ = np.array(index, dtype=np.uint8)
        ret0 = numqi.gate.PauliOperator.from_full_matrix(np0).F2
        assert np.array_equal(ret_, ret0)

        tmp0 = numqi.gate.PauliOperator.from_F2(ret_).full_matrix
        ret1 = numqi.gate.PauliOperator.from_full_matrix(tmp0).F2
        assert np.array_equal(ret_, ret1)

    for _ in range(10):
        pauli0 = numqi.random.rand_pauli(5)
        pauli1 = numqi.gate.PauliOperator.from_F2(pauli0.F2)
        pauli2 = numqi.gate.PauliOperator.from_str(pauli0.str_, sign=pauli0.sign)
        pauli3 = numqi.gate.PauliOperator.from_np_list(pauli0.np_list, sign=pauli0.sign)
        pauli4 = numqi.gate.PauliOperator.from_full_matrix(pauli0.full_matrix)
        assert np.all(pauli0.F2 == pauli1.F2)
        assert np.all(pauli0.F2 == pauli2.F2)
        assert np.all(pauli0.F2 == pauli3.F2)
        assert np.all(pauli0.F2 == pauli4.F2)

def test_PauliOperator_matmul():
    for num_qubit in [1,2,3]:
        for _ in range(10):
            pauli0 = numqi.random.rand_pauli(num_qubit)
            pauli1 = numqi.random.rand_pauli(num_qubit)
            pauli2 = pauli0 @ pauli1
            ret_ = numqi.gate.PauliOperator.from_full_matrix(pauli0.full_matrix @ pauli1.full_matrix)
            assert np.all(pauli2.F2==ret_.F2)


def test_PauliOperator_commutate():
    for num_qubit in [1,2,3]:
        for _ in range(10):
            pauli0 = numqi.random.rand_pauli(num_qubit)
            pauli1 = numqi.random.rand_pauli(num_qubit)
            np0 = pauli0.full_matrix
            np1 = pauli1.full_matrix
            ret_ = np.abs(np0 @ np1 - np1 @ np0).max() < 1e-10
            ret0 = pauli0.commutate_with(pauli1)
            assert ret_ == ret0


def test_PauliOperator_inverse():
    for num_qubit in [1,2,3]:
        for _ in range(10):
            pauli0 = numqi.random.rand_pauli(num_qubit)
            pauli1 = pauli0.inverse()
            pauli2 = pauli0 @ pauli1
            pauli3 = pauli1 @ pauli0
            assert np.all(pauli2.F2==0)
            assert np.all(pauli3.F2==0)

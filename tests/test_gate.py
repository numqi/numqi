import random
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
        assert np.abs(X - H.T.conj() @ Z @ H).max() < 1e-10

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
        assert np.abs(z0 - H.T.conj() @ z1 @ H).max() < 1e-10

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

    # test batch
    pauli_F2 = numqi.random.rand_F2(2, 3, 10)
    pauli_str,pauli_sign = numqi.gate.pauli_F2_to_str(pauli_F2)
    ret0 = numqi.gate.pauli_str_to_F2(pauli_str, pauli_sign)
    assert np.all(ret0==pauli_F2)


def test_pauli_str_to_index():
    rng = random.Random()
    num_qubit = 34
    x0 = rng.randint(0, 4**num_qubit)
    x1 = numqi.gate.pauli_index_to_str(x0, num_qubit)
    assert numqi.gate.pauli_str_to_index(x1)==x0

    num_qubit = 28 #should be less than 31
    x0 = np_rng.integers(0, 4**num_qubit, size=(2,3)).astype(np.uint64)
    x1 = numqi.gate.pauli_index_to_str(x0, num_qubit)
    assert np.all(numqi.gate.pauli_str_to_index(x1)==x0)


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


def test_pauli_F2_to_index():
    np_rng = np.random.default_rng()
    for num_qubit in [3,4,5]:
        N0 = 4**num_qubit
        np0 = np_rng.integers(0, N0, size=23).astype(np.uint64)
        np1 = numqi.gate.pauli_index_to_F2(np0, num_qubit, with_sign=False)
        np2 = numqi.gate.pauli_F2_to_index(np1, with_sign=False)
        assert np.array_equal(np0, np2)

        np0 = np_rng.integers(0, N0, size=23).astype(np.uint64)
        np1 = numqi.gate.pauli_index_to_F2(np0, num_qubit, with_sign=True)
        tmp0 = np.array([numqi.gate.PauliOperator(x).sign for x in np1])
        assert np.abs(tmp0-1).max() < 1e-10


def test_get_pauli_subset_equivalent():
    ud_subset = (0, 1, 2, 3, 4, 9, 10, 11, 13, 14, 15)
    equivalent_set = numqi.gate.get_pauli_subset_equivalent(ud_subset, num_qubit=2)
    ret_ = {
        (0, 1, 2, 3, 4, 9, 10, 11, 13, 14, 15),
        (0, 1, 2, 3, 5, 6, 7, 8, 13, 14, 15),
        (0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12),
        (0, 1, 4, 6, 7, 8, 10, 11, 12, 14, 15),
        (0, 2, 4, 5, 7, 8, 9, 11, 12, 13, 15),
        (0, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14),
    }
    assert equivalent_set==ret_
    # all 2-qubit UD measurement schemes of size 11 (x6)

    ud_subset = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 13, 14, 15)
    equivalent_set = numqi.gate.get_pauli_subset_equivalent(ud_subset, num_qubit=2)
    ret_ = {
        (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 13, 14, 15),
        (0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14),
        (0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 14, 15),
        (0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 15),
        (0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 13, 14, 15),
        (0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15),
        (0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14),
        (0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 15),
        (0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 14, 15),
        (0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15),
        (0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15),
        (0, 1, 2, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15),
        (0, 1, 2, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        (0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14),
        (0, 1, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15),
        (0, 1, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15),
        (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
        (0, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15),
        (0, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15),
        (0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    }
    assert equivalent_set==ret_
    # all 2-qubit UD measurement schemes of size 13 (x20)

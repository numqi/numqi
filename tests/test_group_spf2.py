import random
import numpy as np

import numqi

rng = random.Random()
np_rng = np.random.default_rng()


def test_get_number():
    tmp0 = [(1,6), (2,120), (3,2016), (4,32640), (5,523776), (6,8386560)]
    tmp1 = numqi.group.spf2.get_number(6, kind='coset')
    assert tuple(x[1] for x in tmp0)==tmp1


def test_int_to_bit():
    n = 8
    for _ in range(10):
        x_int = rng.randint(0, 2**n-1)
        tmp0 = bin(x_int)[2:].rjust(n, '0')
        ret_ = np.array([x=='1' for x in tmp0[::-1]], dtype=np.uint8)
        ret0 = numqi.group.spf2.int_to_bitarray(x_int, n)
        assert np.array_equal(ret_, ret0)
        assert numqi.group.spf2.bitarray_to_int(ret_)==x_int


def test_schmidt_orthogonalization():
    sigmax = np.array([[0,1],[1,0]],dtype=np.uint8)
    for N0 in range(4, 16, 2):
        npLambda = np.kron(sigmax, np.eye(N0,dtype=np.uint8))
        vec_in = [numqi.random.rand_F2(2*N0) for _ in range(2*N0)]
        vec_out = np.stack(numqi.group.spf2.schmidt_orthogonalization(vec_in))
        assert (vec_out.shape[0]%2)==0
        tmp0 = np.kron(sigmax, np.eye(vec_out.shape[0]//2, dtype=np.uint8))
        assert np.array_equal((vec_out @ npLambda @ vec_out.T) % 2, tmp0)


def test_find_transvection():
    for N0 in range(2, 14, 2): #even
        for _ in range(min(4**N0, 500)):
            v0 = numqi.random.rand_F2(N0, not_zero=True)
            v1 = numqi.random.rand_F2(N0, not_zero=True)
            h0,h1 = numqi.group.spf2.find_transvection(v0, v1)
            assert np.array_equal(numqi.group.spf2.transvection(v0, h0, h1), v1)


def test_from_int_tuple_and_to_int_tuple():
    # N0=1000 is almost the limit of what can be done on my laptop
    sigmax = np.array([[0,1],[1,0]],dtype=np.uint8)
    for N0 in range(1,10):
        npLambda = np.kron(sigmax, np.eye(N0,dtype=np.uint8))
        for _ in range(10):
            int_tuple,np0 = numqi.random.rand_SpF2(N0, return_int_tuple=True)
            assert np.all(((np0 @ npLambda @ np0.T)%2)==npLambda)
            assert np.all(((np0.T @ npLambda @ np0)%2)==npLambda) #x in Sp, then x^T in Sp
            assert numqi.group.spf2.to_int_tuple(np0)==int_tuple

            np0 = numqi.random.rand_SpF2(N0) @ numqi.random.rand_SpF2(N0)
            assert np.array_equal((np0 @ npLambda @ np0.T)%2, npLambda) #x and y in Sp, then xy in Sp


def test_inverse():
    for N0 in range(1, 10):
        for _ in range(10):
            np0 = numqi.random.rand_SpF2(N0)
            np1 = numqi.group.spf2.inverse(np0)
            assert np.array_equal((np0 @ np1)%2, np.eye(2*N0, dtype=np.uint8))
            assert np.array_equal((np1 @ np0)%2, np.eye(2*N0, dtype=np.uint8))


def test_apply_clifford_on_pauli():
    for num_qubit in [1,2,3]:
        all_integer = tuple(range(1<<(2*num_qubit+2)))
        pauli_bit_list = [numqi.group.spf2.int_to_bitarray(x, 2*num_qubit+2) for x in all_integer]
        for _ in range(100):
            cli_r,cli_mat = numqi.random.rand_Clifford_group(num_qubit)
            z0 = [numqi.group.spf2.bitarray_to_int(numqi.sim.clifford.apply_clifford_on_pauli(x, cli_r, cli_mat)) for x in pauli_bit_list]
            assert tuple(sorted(z0))==all_integer

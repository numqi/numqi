import numpy as np

import numqi

np_rng = np.random.default_rng()

def test_clifford_array_to_F2():
    example_list = [
        (numqi.gate.H, [0,0], [[0,1],[1,0]]),
        (numqi.gate.X, [0,1], [[1,0],[0,1]]),
        (numqi.gate.Y, [1,1], [[1,0],[0,1]]),
        (numqi.gate.Z, [1,0], [[1,0],[0,1]]),
        (numqi.gate.S, [0,0], [[1,0],[1,1]]),
        (numqi.gate.CNOT, [0,0,0,0], [[1,0,0,0],[1,1,0,0],[0,0,1,1],[0,0,0,1]]),
    ]
    for np0, cli_r, cli_mat in example_list:
        cli_r = np.array(cli_r, dtype=np.uint8)
        cli_mat = np.array(cli_mat, dtype=np.uint8)
        ret0, ret1 = numqi.sim.clifford.clifford_array_to_F2(np0)
        assert np.array_equal(ret0, cli_r)
        assert np.array_equal(ret1, cli_mat)

        pauli_array = numqi.gate.get_pauli_group(len(cli_r)//2, kind='numpy')
        tmp0 = [numqi.gate.PauliOperator.from_full_matrix(x).F2 for x in pauli_array]
        ret0 = [numqi.sim.clifford.apply_clifford_on_pauli(x, cli_r, cli_mat) for x in tmp0]
        tmp0 = np.einsum(np0, [0,1], pauli_array, [2,1,3], np0.conj(), [4,3], [2,0,4], optimize=True)
        ret_ = [numqi.gate.PauliOperator.from_full_matrix(x).F2 for x in tmp0]
        assert all(np.array_equal(x, y) for x,y in zip(ret0, ret_))


def test_clifford_multiply():
    for N0 in [1,2,3,4,5,6]:
        for _ in range(10):
            rx = numqi.random.rand_F2(2*N0)
            Sx = numqi.random.rand_SpF2(N0)
            ry = numqi.random.rand_F2(2*N0)
            Sy = numqi.random.rand_SpF2(N0)
            rz,Sz = numqi.sim.clifford.clifford_multiply(rx, Sx, ry, Sy)
            for _ in range(10):
                pauli = numqi.random.rand_F2(2*N0+2)

                tmp0 = numqi.sim.clifford.apply_clifford_on_pauli(pauli, rx, Sx)
                ret_ = numqi.sim.clifford.apply_clifford_on_pauli(tmp0, ry, Sy)
                ret0 = numqi.sim.clifford.apply_clifford_on_pauli(pauli, rz, Sz)
                assert np.array_equal(ret_, ret0)


def test_CliffordCircuit():
    num_qubit = 10
    num_depth = 5

    circ = numqi.sim.CliffordCircuit()
    for _ in range(num_depth):
        for _ in range(2):
            for ind0 in range(num_qubit):
                circ.random_one_qubit_gate(ind0) #IXYZHS
        for _ in range(2*num_qubit):
            ind0,ind1 = np_rng.choice(num_qubit, size=2, replace=False) #CX CY CZ
            circ.random_two_qubit_gate(ind0, ind1)

    for _ in range(10):
        pauli_F2 = numqi.random.rand_pauli(num_qubit, is_hermitian=True).F2
        q0 = numqi.random.rand_haar_state(2**num_qubit)

        q1 = circ.to_universal_circuit().apply_state(q0)
        tmp0 = numqi.gate.PauliOperator.from_F2(pauli_F2).op_list
        ret_ = numqi.sim.state.inner_product_psi0_O_psi1(q1, q1, tmp0).real

        tmp0 = numqi.gate.PauliOperator.from_F2(circ.apply_pauli_F2(pauli_F2)).op_list
        ret0 = numqi.sim.state.inner_product_psi0_O_psi1(q0, q0, tmp0)
        assert abs(ret0.imag) < 1e-10
        assert abs(ret_-ret0) < 1e-10

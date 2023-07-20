import numpy as np

import numqi

def test_pauli_array_to_F2():
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
        ret0 = numqi.sim.clifford.pauli_array_to_F2(np0)
        assert np.array_equal(ret_, ret0)

        ret1 = numqi.sim.clifford.pauli_array_to_F2(numqi.sim.clifford.pauli_F2_to_array(ret_))
        assert np.array_equal(ret_, ret1)


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
        tmp0 = [numqi.sim.clifford.pauli_array_to_F2(x) for x in pauli_array]
        ret0 = [numqi.sim.clifford.apply_clifford_on_pauli(x, cli_r, cli_mat) for x in tmp0]
        tmp0 = np.einsum(np0, [0,1], pauli_array, [2,1,3], np0.conj(), [4,3], [2,0,4], optimize=True)
        ret_ = [numqi.sim.clifford.pauli_array_to_F2(x) for x in tmp0]
        assert all(np.array_equal(x, y) for x,y in zip(ret0, ret_))

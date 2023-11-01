import numpy as np

import numqi.gate
import numqi.random

# see numqi.group.spf2

def apply_clifford_on_pauli(pauli_bit, cli_r, cli_mat):
    for x in [pauli_bit, cli_r, cli_mat]:
        assert x.dtype.type==np.uint8
    assert pauli_bit.max()<=1
    N0 = cli_r.shape[0]//2
    XZin = pauli_bit[2:]
    XZout = (cli_mat @ XZin)%2
    delta = pauli_bit[1] + np.dot((cli_mat[:N0]*XZin).reshape(-1), cli_mat[N0:].reshape(-1))
    bit1 = delta%2
    tmp0 = cli_mat * XZin
    tmp_jk = tmp0[N0:].T @ tmp0[:N0]
    tmp1 = ((delta%4)//2).astype(np.uint8)
    bit0 = (pauli_bit[0] + np.dot(XZin, cli_r) + np.triu(tmp_jk, 1).sum() + tmp1) % 2
    ret = np.concatenate([np.array([bit0,bit1], dtype=np.uint8), XZout], axis=0)
    return ret


def clifford_array_to_F2(np0):
    assert (np0.ndim==2) and (np0.shape[0]==np0.shape[1]) and np0.shape[0]>=2
    N0 = int(np.log2(np0.shape[0]))
    assert 2**N0 == np0.shape[0]
    assert np.abs(np0 @ np0.T.conj() - np.eye(np0.shape[0])).max() < 1e-12

    cli_r = np.zeros(2*N0, dtype=np.uint8)
    cli_mat = np.zeros((2*N0,2*N0), dtype=np.uint8)
    for ind0 in range(N0):
        tmp0 = np0.reshape(2**N0, 2**ind0, 2, 2**(N0-ind0-1))

        tmp1 = np.einsum(tmp0, [0,1,2,3], numqi.gate.X, [2,4], tmp0.conj(), [5,1,4,3], [0,5], optimize=True)
        Xbit = numqi.gate.PauliOperator.from_full_matrix(tmp1).F2
        cli_mat[:,ind0] = Xbit[2:]
        cli_r[ind0] = (Xbit[0] + (np.dot(Xbit[2:(N0+2)], Xbit[(N0+2):]) % 4)//2) % 2

        tmp1 = np.einsum(tmp0, [0,1,2,3], numqi.gate.Z, [2,4], tmp0.conj(), [5,1,4,3], [0,5], optimize=True)
        Zbit = numqi.gate.PauliOperator.from_full_matrix(tmp1).F2
        cli_mat[:,ind0+N0] = Zbit[2:]
        cli_r[ind0+N0] = (Zbit[0] + (np.dot(Zbit[2:(N0+2)], Zbit[(N0+2):]) % 4)//2) % 2
    return cli_r,cli_mat


def clifford_multiply(rx, Sx, ry, Sy):
    # z=y \circ x
    assert rx.shape[0]%2==0
    N0 = rx.shape[0]//2
    assert (rx.shape==(2*N0,)) and (Sx.shape==(2*N0,2*N0))
    assert (ry.shape==(2*N0,)) and (Sy.shape==(2*N0,2*N0))
    assert all(x.dtype.type==np.uint8 for x in [rx,ry,Sx,Sy])
    Sz = (Sy @ Sx) % 2
    tmp0 = np.einsum(Sx[:N0], [0,1], Sx[N0:], [0,1], [1], optimize=True)
    tmp1 = np.einsum(Sx, [0,1], Sy[:N0], [2,0], Sy[N0:], [2,0], [1], optimize=True)
    tmp2 = np.einsum(Sz[:N0], [0,1], Sz[N0:], [0,1], [1], optimize=True)
    assert np.all((tmp0+tmp1+tmp2)%2==0)
    delta = ((tmp0 + tmp1 - tmp2)%4).astype(np.uint8)
    # alpha=0 j=1 k=2 i=3
    tmp0 = np.triu(np.ones(2*N0, dtype=np.uint8), k=1)
    tmp1 = np.einsum(Sx, [1,0], Sx, [2,0], Sy[N0:], [3,1], Sy[:N0], [3,2], tmp0, [1,2], [0], optimize=True)
    rz = (rx + ry@Sx + tmp1 + delta//2) % 2
    return rz, Sz


def _clifford_circuit_single_qubit_gate(key):
    def hf0(self, index):
        index = int(index)
        assert index>=0
        self.gate_index_list.append((key, index))
    return hf0

def _clifford_circuit_two_qubit_gate(key):
    def hf0(self, index0, index1):
        index0 = int(index0)
        index1 = int(index1)
        assert (index0>=0) and (index1>=0) and (index0!=index1)
        self.gate_index_list.append((key, index0, index1))
    return hf0

_basic_clifford_dict = {
    'X': numqi.gate.X,
    'Y': numqi.gate.Y,
    'Z': numqi.gate.Z,
    'H': numqi.gate.H,
    'S': numqi.gate.S,
    'CX': numqi.gate.CNOT,
    'CY': np.block([[numqi.gate.I, numqi.gate.I*0], [numqi.gate.I*0, numqi.gate.Y]]),
    'CZ': np.block([[numqi.gate.I, numqi.gate.I*0], [numqi.gate.I*0, numqi.gate.Z]]),
}
_basic_clifford_dagger_f2_cache = dict()
def _basic_clifford_dagger_f2(key):
    if key in _basic_clifford_dagger_f2_cache:
        ret = _basic_clifford_dagger_f2_cache[key]
    else:
        ret = clifford_array_to_F2(_basic_clifford_dict[key].T.conj())
        _basic_clifford_dagger_f2_cache[key] = ret
    return ret


class CliffordCircuit:
    _single_gate_list = ['I', 'X', 'Y', 'Z', 'H', 'S']
    _two_qubit_gate_list = ['CX', 'CY', 'CZ']

    def __init__(self, seed=None):
        self.gate_index_list = []
        self.np_rng = numqi.random.get_numpy_rng(seed)
        self._R = None
        self._S = None

    def I(self, *index):
        pass

    X = _clifford_circuit_single_qubit_gate('X')
    Y = _clifford_circuit_single_qubit_gate('Y')
    Z = _clifford_circuit_single_qubit_gate('Z')
    H = _clifford_circuit_single_qubit_gate('H')
    S = _clifford_circuit_single_qubit_gate('S')
    CX = _clifford_circuit_two_qubit_gate('CX')
    CY = _clifford_circuit_two_qubit_gate('CY')
    CZ = _clifford_circuit_two_qubit_gate('CZ')
    CNOT = CX

    @property
    def num_qubit(self):
        ret = max(y for x in self.gate_index_list for y in x[1:]) + 1
        return ret

    def random_one_qubit_gate(self, index):
        tmp0 = self._single_gate_list[self.np_rng.integers(0, len(self._single_gate_list))]
        getattr(self, tmp0)(index)

    def random_two_qubit_gate(self, index0, index1):
        assert index0!=index1
        tmp0 = self._two_qubit_gate_list[self.np_rng.integers(0, len(self._two_qubit_gate_list))]
        getattr(self, tmp0)(index0, index1)

    def to_symplectic_form(self):
        if self._R is None:
            num_qubit = self.num_qubit
            R0 = np.zeros(2*num_qubit, dtype=np.uint8)
            S0 = np.eye(2*num_qubit, dtype=np.uint8)
            retR = R0.copy()
            retS = S0.copy()
            for gate in self.gate_index_list[::-1]:
                if len(gate)==2: #single qubit gate
                    index = np.array([gate[1], gate[1]+num_qubit], dtype=np.int32)
                else:
                    assert len(gate)==3
                    index = np.array([gate[1], gate[2], gate[1]+num_qubit, gate[2]+num_qubit], dtype=np.int32)
                tmp0 = _basic_clifford_dagger_f2(gate[0])
                tmpR = R0.copy()
                tmpS = S0.copy()
                tmpR[index] = tmp0[0]
                tmpS[index[:,np.newaxis], index] = tmp0[1]
                retR, retS = clifford_multiply(retR, retS, tmpR, tmpS)
            self._R = retR
            self._S = retS
            ret = retR,retS
        else:
            ret = self._R, self._S
        return ret

    def apply_pauli_F2(self, pauli_F2):
        retR,retS = self.to_symplectic_form()
        ret = apply_clifford_on_pauli(pauli_F2, retR, retS)
        return ret

    def to_universal_circuit(self):
        ret = numqi.sim.Circuit()
        tmp0 = {'X': numqi.gate.X, 'Y': numqi.gate.Y, 'Z': numqi.gate.Z, 'H': numqi.gate.H,
                'S': numqi.gate.S, 'CX': numqi.gate.X, 'CY': numqi.gate.Y, 'CZ': numqi.gate.Z}
        for gate in self.gate_index_list:
            if len(gate)==2: #single qubit gate
                ret.single_qubit_gate(tmp0[gate[0]], gate[1])
            else:
                assert len(gate)==3
                ret.controlled_single_qubit_gate(tmp0[gate[0]], gate[1], gate[2])
        return ret

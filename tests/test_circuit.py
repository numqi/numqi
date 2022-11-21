import numpy as np
import pytest

import numpyqi

try:
    import torch
    import torch_wrapper
    from _circuit_torch_utils import DummyQNN
except ImportError:
    torch = None
    torch_wrapper = None
    DummyQNN = None


def _vqe00_generate_RB_state(num_qubit):
    def hf_bin_pad(x, N):
        tmp0 = bin(x)[2:]
        ret = '0'*(N-len(tmp0)) + tmp0
        return ret
    tmp0 = num_qubit//2
    tmp1 = [hf_bin_pad(x, tmp0) for x in range(2**tmp0)]
    tmp2 = [(''.join(('1' if y=='0' else '0') for y in x))[::-1] for x in tmp1]
    tmp3 = [int(x+y,2) for x,y in zip(tmp1,tmp2)]
    hf0 = lambda x: sum(y=='1' for y in x)%2==0
    tmp4 = np.array([(1 if hf0(x) else -1) for x in tmp1]) / np.sqrt(2**tmp0)
    ret = np.zeros(2**num_qubit, dtype=np.complex128)
    ret[tmp3] = tmp4
    return ret


def build_dummy_circuit(num_depth, num_qubit):
    circ = numpyqi.circuit.Circuit(default_requires_grad=True)
    for ind0 in range(num_depth):
        tmp0 = list(range(0, num_qubit-1, 2)) + list(range(1, num_qubit-1, 2))
        for ind1 in tmp0:
            circ.ry(ind1)
            circ.ry(ind1+1)
            circ.rz(ind1)
            circ.rz(ind1+1)
            circ.cnot(ind1, ind1+1)
            circ.double_qubit_gate(numpyqi.random.rand_haar_unitary(4,4), ind1, ind1+1)
    return circ


@pytest.mark.skipif(torch is None, reason='pytorch is not installed')
def test_dummy_circuit():
    num_qubit = 5
    num_depth = 3
    zero_eps = 1e-7
    target_state = _vqe00_generate_RB_state(num_qubit)

    circuit = build_dummy_circuit(num_depth, num_qubit)
    model = DummyQNN(circuit, target_state)
    torch_wrapper.check_model_gradient(model)


def test_circuit_to_unitary():
    num_qubit = 5
    circ = build_dummy_circuit(num_depth=3, num_qubit=num_qubit)
    np0 = numpyqi.random.rand_haar_state(2**num_qubit)

    ret_ = circ.apply_state(np0)

    unitary_matrix = circ.to_unitary()
    ret0 = unitary_matrix @ np0
    assert np.abs(unitary_matrix @ unitary_matrix.T.conj() - np.eye(2**num_qubit)).max() < 1e-7
    assert np.abs(ret_-ret0).max() < 1e-7

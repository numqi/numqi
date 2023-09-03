import numpy as np
import torch

import numqi

np_rng = np.random.default_rng()

class DummyQNNModel(torch.nn.Module):
    def __init__(self, circuit):
        super().__init__()
        self.circuit_torch = numqi.sim.CircuitTorchWrapper(circuit)
        self.num_qubit = circuit.num_qubit
        np_rng = np.random.default_rng()
        tmp0 = np_rng.normal(size=2**self.num_qubit) + 1j*np_rng.normal(size=2**self.num_qubit)
        self.target_state = torch.tensor(tmp0 / np.linalg.norm(tmp0), dtype=torch.complex128)

        self.q0 = torch.empty(2**self.num_qubit, dtype=torch.complex128, requires_grad=False)

    def forward(self):
        self.q0[:] = 0
        self.q0[0] = 1
        q0 = self.circuit_torch(self.q0)
        inner_product = torch.dot(self.target_state.conj(), q0)
        loss = (inner_product*inner_product.conj()).real
        return loss


def build_dummy_circuit(num_depth, num_qubit, seed=None):
    np_rng = numqi.random.get_numpy_rng(seed)
    circ = numqi.sim.Circuit(default_requires_grad=True)
    hf0 = lambda x: np_rng.uniform(0, 2*np.pi, size=x)
    for _ in range(num_depth):
        tmp0 = list(range(0, num_qubit-1, 2)) + list(range(1, num_qubit-1, 2))
        for ind1 in tmp0:
            circ.ry(ind1, hf0(1))
            circ.ry(ind1+1, hf0(1))
            circ.rz(ind1, hf0(1))
            circ.rz(ind1+1, hf0(1))
            circ.rzz((ind1, ind1+1), hf0(1))
            circ.cnot(ind1, ind1+1)
            tmp0 = numqi.random.rand_unitary_matrix(2, seed=np_rng)
            circ.controlled_single_qubit_gate(tmp0, (ind1+1,(ind1-1)%num_qubit), ind1)
            circ.double_qubit_gate(numqi.random.rand_unitary_matrix(4, seed=np_rng), ind1, ind1+1)
    return circ


def test_dummy_circuit():
    num_qubit = 5
    num_depth = 2
    circuit = build_dummy_circuit(num_depth, num_qubit)
    model = DummyQNNModel(circuit)
    numqi.optimize.check_model_gradient(model)


def test_circuit_to_unitary():
    num_qubit = 5
    num_depth = 3
    circ = build_dummy_circuit(num_depth, num_qubit)
    np0 = numqi.random.rand_haar_state(2**num_qubit)

    ret_ = circ.apply_state(np0)

    unitary_matrix = circ.to_unitary()
    ret0 = unitary_matrix @ np0
    assert np.abs(unitary_matrix @ unitary_matrix.T.conj() - np.eye(2**num_qubit)).max() < 1e-7
    assert np.abs(ret_-ret0).max() < 1e-7


def test_measure_gate():
    # bell state
    circ = numqi.sim.Circuit(default_requires_grad=False)
    circ.H(0)
    circ.cnot(0, 1)
    gate_measure = circ.measure(index=(0,1))
    q0 = numqi.sim.state.new_base(num_qubit=2)
    for _ in range(10): #randomness in measure gate, so we repeat here
        q1 = circ.apply_state(q0)
        assert tuple(gate_measure.bitstr) in {(0,0),(1,1)}
        assert np.abs(gate_measure.probability-np.abs([0.5,0,0,0.5])).max() < 1e-7
        if tuple(gate_measure.bitstr)==(0,0):
            assert np.abs(q1-np.array([1,0,0,0])).max() < 1e-7
        else:
            assert np.abs(q1-np.array([0,0,0,1])).max() < 1e-7

    # GHZ state
    circ = numqi.sim.Circuit(default_requires_grad=False)
    circ.H(0)
    circ.cnot(0, 1)
    circ.cnot(1, 2)
    gate_measure = circ.measure(index=(0,1,2))
    q0 = numqi.sim.state.new_base(num_qubit=3)
    for _ in range(10):
        q1 = circ.apply_state(q0)
        assert tuple(gate_measure.bitstr) in {(0,0,0),(1,1,1)}
        assert np.abs(gate_measure.probability-np.abs([0.5,0,0,0,0,0,0,0.5])).max() < 1e-7
        if tuple(gate_measure.bitstr)==(0,0,0):
            assert np.abs(q1-np.array([1,0,0,0,0,0,0,0])).max() < 1e-7
        else:
            assert np.abs(q1-np.array([0,0,0,0,0,0,0,1])).max() < 1e-7


def hf_ry_rx(alpha, beta):
    r'''
    ry(beta) * rx(alpha)
    '''
    if isinstance(alpha, torch.Tensor):
        assert isinstance(beta, torch.Tensor)
        assert alpha.dtype==beta.dtype
        if alpha.dtype==torch.float32:
            alpha = alpha*torch.tensor(1, dtype=torch.complex64)
            beta = beta*torch.tensor(1, dtype=torch.complex64)
        else:
            assert alpha.dtype==torch.float64
            alpha = alpha*torch.tensor(1, dtype=torch.complex128)
            beta = beta*torch.tensor(1, dtype=torch.complex128)
        cosa,sina,cosb,sinb = torch.cos(alpha/2),torch.sin(alpha/2),torch.cos(beta/2),torch.sin(beta/2)
        cc,cs,sc,ss = cosa*cosb,cosa*sinb,sina*cosb,sina*sinb
        ret = torch.stack([cc+1j*ss,-1j*sc-cs,cs-1j*sc,cc-1j*ss], dim=-1).view(*alpha.shape, 2, 2)
    else:
        alpha = np.asarray(alpha)
        beta = np.asarray(beta)
        # assert alpha.ndim<=1 and beta.ndim<=1
        cosa,sina,cosb,sinb = np.cos(alpha/2),np.sin(alpha/2),np.cos(beta/2),np.sin(beta/2)
        cc,cs,sc,ss = cosa*cosb,cosa*sinb,sina*cosb,sina*sinb
        ret = np.stack([cc+1j*ss,-1j*sc-cs,cs-1j*sc,cc-1j*ss], axis=-1).reshape(*alpha.shape, 2, 2)
    return ret


class RyRxGate(numqi.sim.ParameterGate):
    def __init__(self, index, alpha=0, beta=0, requires_grad=True):
        super().__init__(kind='unitary', hf0=hf_ry_rx, args=(alpha,beta), name='ry_rx', requires_grad=requires_grad)
        self.index = index, #must be tuple of int


def test_custom_gate_without_torch():
    alpha,beta = np_rng.uniform(0, 2*np.pi, 2)
    tmp0 = np_rng.uniform(size=2) + 1j*np_rng.uniform(size=2)
    q0 = tmp0 / np.linalg.norm(tmp0)

    circ = numqi.sim.Circuit(default_requires_grad=False)
    circ.register_custom_gate('ry_rx', RyRxGate)
    circ.ry_rx(0, alpha, beta)
    q1 = circ.apply_state(q0)

    q2 = numqi.gate.ry(beta) @ (numqi.gate.rx(alpha) @ q0)
    assert np.abs(q1-q2).max() < 1e-10


def test_custom_gate_with_torch():
    alpha,beta = np_rng.uniform(0, 2*np.pi, 2)
    circ = numqi.sim.Circuit(default_requires_grad=False)
    circ.register_custom_gate('ry_rx', RyRxGate)
    circ.ry_rx(0, alpha, beta)
    model = DummyQNNModel(circ)
    numqi.optimize.check_model_gradient(model)


def test_toffoli_gate_decomposition():
    # wiki/Toffoli-gate https://en.wikipedia.org/wiki/Toffoli_gate
    np0 = np.block([[np.eye(6),np.zeros((6,2))],[np.zeros((2,6)),np.array([[0,1],[1,0]])]])

    circ0 = numqi.sim.Circuit(default_requires_grad=False)
    circ0.toffoli((0,1), 2)
    np1 = circ0.to_unitary()
    assert np.abs(np0-np1).max() < 1e-10

    circ1 = numqi.sim.Circuit(default_requires_grad=False)
    circ1.H(2)
    circ1.cnot(1, 2)
    T_dagger = numqi.gate.T.conj()
    circ1.single_qubit_gate(T_dagger, 2)
    circ1.cnot(0, 2)
    circ1.T(2)
    circ1.cnot(1, 2)
    circ1.single_qubit_gate(T_dagger, 2)
    circ1.cnot(0, 2)
    circ1.T(1)
    circ1.T(2)
    circ1.cnot(0, 1)
    circ1.H(2)
    circ1.T(0)
    circ1.single_qubit_gate(T_dagger, 1)
    circ1.cnot(0, 1)
    np2 = circ1.to_unitary()
    assert np.abs(np0-np2).max() < 1e-10


class ClassicalControlGate:
    def __init__(self, gateM, op, index, name='classical_control_gate'):
        self.gateM = gateM
        self.op = op
        self.index = index
        self.name = name
        self.requires_grad = False
        self.kind = 'custom'

    def forward(self, q0):
        if self.gateM.bitstr[0]==1:
            q0 = numqi.sim.state.apply_gate(q0, self.op, self.index)
        return q0


def test_teleportation():
    # https://nbviewer.org/urls/qutip.org/qutip-tutorials/tutorials-v4/quantum-circuits/teleportation.ipynb
    circ = numqi.sim.Circuit(default_requires_grad=False)
    circ.register_custom_gate('classical_control_gate', ClassicalControlGate)
    # alice: 0, 1
    # bob: 2
    circ.H(1)
    circ.cnot(1, 2)
    circ.cnot(0, 1)
    circ.H(0)
    gate_M0 = circ.measure(1)
    gate_M1 = circ.measure(0)
    circ.classical_control_gate(gate_M0, numqi.gate.X, 2)
    circ.classical_control_gate(gate_M1, numqi.gate.Z, 2)

    q0 = numqi.random.rand_state(2)
    tmp0 = np.zeros(4, dtype=np.complex128)
    tmp0[0] = 1
    tmp1 = (q0[:,np.newaxis] * tmp0).reshape(-1)
    q1 = circ.apply_state(tmp1)
    _,S,V = np.linalg.svd(q1.reshape(4, 2), full_matrices=False)
    assert S[1] < 1e-7 #product state
    assert abs(abs(np.vdot(V[0], q0)) - 1) < 1e-7 #fidelity 1


def test_build_graph_state():
    for N0 in [3,5,7]:
        adjacent_mat = numqi.random.rand_adjacent_matrix(N0)
        q0, stabilizer_circ_list = numqi.sim.build_graph_state(adjacent_mat, return_stabilizer_circ=True)
        z0 = np.array([np.vdot(q0, x.apply_state(q0)) for x in stabilizer_circ_list])
        assert np.abs(z0-1).max() < 1e-10

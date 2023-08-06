import numpy as np

try:
    import torch
except ImportError:
    torch = None

import numqi.gate
import numqi.channel
import numqi.sim.state
from numqi.utils import hf_tuple_of_int, hf_tuple_of_any

from ._internal import Gate, ParameterGate

if torch is not None:
    from ._torch_utils import CircuitTorchWrapper
else:
    CircuitTorchWrapper = None

CANONICAL_GATE_KIND = {'unitary','control','measure'}
# TODO kraus


class MeasureGate:
    def __init__(self, index, seed=None, name='measure'):
        self.kind = 'measure'
        self.name = name
        self.requires_grad = False
        index = numqi.utils.hf_tuple_of_int(index)
        assert all(x==y for x,y in zip(sorted(index),index)), 'index must be sorted'
        self.index = index
        self.np_rng = numqi.random.get_numpy_rng(seed)

        self.bitstr = None
        self.probability = None

    def forward(self, q0):
        self.bitstr,self.probability,q1 = numqi.sim.state.measure_quantum_vector(q0, self.index, self.np_rng)
        return q1


def _unitary_gate(name_, array, num_index):
    def hf0(self, /, *index, name=name_):
        gate = Gate('unitary', array, name=name)
        index = hf_tuple_of_int(index)
        assert len(index)==num_index
        self.gate_index_list.append((gate, index))
        return gate
    return hf0


def _control_gate(name_, array, num_control, num_target):
    def hf0(self, control_qubit, target_qubit, name=name_):
        control_qubit = set(sorted(hf_tuple_of_int(control_qubit)))
        target_qubit = hf_tuple_of_int(target_qubit)
        assert len(control_qubit)==num_control
        assert len(target_qubit)==num_target
        assert all((x not in control_qubit) for x in target_qubit) and len(target_qubit)==len(set(target_qubit))
        gate = Gate('control', array, name=name)
        self.gate_index_list.append((gate, (control_qubit,target_qubit)))
        return gate
    return hf0


def _unitary_parameter_gate(name_, hf0, num_index, num_parameter):
    def hf1(self, index, args=None, name=name_, requires_grad=None):
        if requires_grad is None:
            requires_grad = self.default_requires_grad
        if args is None:
            args = (0.,)*num_parameter #initialize to zero
        else:
            args = hf_tuple_of_any(args, type_=float) #convert float/int into tuple
        gate = ParameterGate('unitary', hf0, args, name=name, requires_grad=requires_grad)
        index = hf_tuple_of_int(index)
        assert len(index)==num_index
        self.gate_index_list.append((gate, index))
        return gate
    return hf1

def _control_parameter_gate(name_, hf0, num_parameter):
    def hf1(self, control_qubit, target_qubit, args=None, name=name_, requires_grad=None):
        if requires_grad is None:
            requires_grad = self.default_requires_grad
        if args is None:
            args = (0.,)*num_parameter #initialize to zero
        else:
            args = hf_tuple_of_any(args, type_=float) #convert float/int into tuple
        control_qubit = set(sorted(hf_tuple_of_int(control_qubit)))
        target_qubit = hf_tuple_of_int(target_qubit)
        assert all((x not in control_qubit) for x in target_qubit) and len(target_qubit)==len(set(target_qubit))
        gate = ParameterGate('control', hf0, args, name=name, requires_grad=requires_grad)
        self.gate_index_list.append((gate, (control_qubit,target_qubit)))
        return gate
    return hf1


def _kraus_gate(name_, hf0):
    def hf1(self, index, args, name=name_):
        kop = hf0(*args)
        gate = Gate('kraus', kop, requires_grad=False, name=name)
        self.gate_index_list.append((gate, hf_tuple_of_int(index)))
        return gate
    return hf1


class Circuit:
    def __init__(self, default_requires_grad=False):
        self.gate_index_list = []
        if torch is None:
            assert not default_requires_grad, 'pytorch is required for default_requires_grad=True'
        self.default_requires_grad = default_requires_grad

    def append_gate(self, gate, index):
        if gate.kind=='unitary':
            index = hf_tuple_of_int(index)
        elif gate.kind=='control':
            assert len(index)==2
            control_qubit = set(sorted(hf_tuple_of_int(index[0])))
            target_qubit = hf_tuple_of_int(index[1])
            assert all((x not in control_qubit) for x in target_qubit) and len(target_qubit)==len(set(target_qubit))
            index = control_qubit,target_qubit
        self.gate_index_list.append((gate,index))

    def extend_circuit(self, circ0):
        for gate_i,index_i in circ0.gate_index_list:
            self.append_gate(gate_i,index_i)

    def register_custom_gate(self, name, gate_class):
        # gate_class could not be child of Gate if not is_pgate
        # gate_class must be child of ParameterGate if is_pgate
        is_pgate = issubclass(gate_class, ParameterGate)
        def hf0(self, *args, **kwargs):
            gate = gate_class(*args, **kwargs)
            # TODO WARNING default_requires_grad is not checked here
            index = gate.index if (gate.kind in CANONICAL_GATE_KIND) else ()
            #leave index to empty for unkown gate
            self.gate_index_list.append((gate,index))
            return gate
        assert not hasattr(self, name)
        # https://stackoverflow.com/a/1015405
        tmp0 = hf0.__get__(self, self.__class__)
        setattr(self, name, tmp0)

    def single_qubit_gate(self, np0, ind0, name='single'):
        assert np0.shape==(2,2)
        gate = Gate('unitary', np0, requires_grad=False, name=name)
        index = int(ind0),
        self.gate_index_list.append((gate, index))
        return gate

    def double_qubit_gate(self, np0, ind0, ind1, name='double'):
        assert np0.shape==(4,4)
        gate = Gate('unitary', np0, requires_grad=False, name=name)
        index = int(ind0),int(ind1)
        self.gate_index_list.append((gate, index))
        return gate

    def triple_qubit_gate(self, np0, ind0, ind1, ind2, name='triple'):
        assert np0.shape==(8,8)
        gate = Gate('unitary', np0, requires_grad=False, name=name)
        index = int(ind0),int(ind1),int(ind2)
        self.gate_index_list.append((gate, index))
        return gate

    def quadruple_qubit_gate(self, np0, ind0, ind1, ind2, ind3, name='quadruple'):
        assert np0.shape==(16,16)
        gate = Gate('unitary', np0, requires_grad=False, name=name)
        index = int(ind0),int(ind1),int(ind2),int(ind3)
        self.gate_index_list.append((gate, index))
        return gate

    def controlled_single_qubit_gate(self, np0, ind_control_set, ind_target, name='control'):
        ind_control_set = set(sorted(hf_tuple_of_int(ind_control_set)))
        ind_target = hf_tuple_of_int(ind_target)
        assert len(ind_target)==1
        assert all((x not in ind_control_set) for x in ind_target) and len(ind_target)==len(set(ind_target))
        assert np0.shape==(2,2)
        gate = Gate('control', np0, requires_grad=False, name=name)
        self.gate_index_list.append((gate, (ind_control_set, ind_target)))
        return gate

    def controlled_double_qubit_gate(self, np0, ind_control_set, ind_target, name='control'):
        ind_control_set = set(sorted(hf_tuple_of_int(ind_control_set)))
        ind_target = hf_tuple_of_int(ind_target)
        assert len(ind_target)==2
        assert all((x not in ind_control_set) for x in ind_target) and len(ind_target)==len(set(ind_target))
        assert np0.shape==(4,4)
        gate = Gate('control', np0, requires_grad=False, name=name)
        self.gate_index_list.append((gate, (ind_control_set, ind_target)))
        return gate

    X = _unitary_gate('X', numqi.gate.pauli.sx, 1)
    Y = _unitary_gate('Y', numqi.gate.pauli.sy, 1)
    Z = _unitary_gate('Z', numqi.gate.pauli.sz, 1)
    H = _unitary_gate('H', numqi.gate.H, 1)
    S = _unitary_gate('S', numqi.gate.S, 1)
    T = _unitary_gate('T', numqi.gate.T, 1)
    Swap = _unitary_gate('Swap', numqi.gate.Swap, 2)

    cnot = _control_gate('cnot', numqi.gate.pauli.sx, 1, 1)
    cx = cnot
    cy = _control_gate('cy', numqi.gate.pauli.sy, 1, 1)
    cz = _control_gate('cz', numqi.gate.pauli.sz, 1, 1)
    toffoli = _control_gate('toffoli', numqi.gate.pauli.sx, 2, 1)

    rx = _unitary_parameter_gate('rx', numqi.gate.rx, 1, 1)
    ry = _unitary_parameter_gate('ry', numqi.gate.ry, 1, 1)
    rz = _unitary_parameter_gate('rz', numqi.gate.rz, 1, 1)
    u3 = _unitary_parameter_gate('u3', numqi.gate.u3, 1, 3)
    rzz = _unitary_parameter_gate('rzz', numqi.gate.rzz, 2, 1)

    crx = _control_parameter_gate('crx', numqi.gate.rx, 1)
    cry = _control_parameter_gate('cry', numqi.gate.ry, 1)
    crz = _control_parameter_gate('crz', numqi.gate.rz, 1)
    cu3 = _control_parameter_gate('cu3', numqi.gate.u3, 3)

    dephasing = _kraus_gate('dephasing', numqi.channel.hf_dephasing_kraus_op)
    depolarizing = _kraus_gate('depolarizing', numqi.channel.hf_depolarizing_kraus_op)
    amplitude_damping = _kraus_gate('amplitude_damping', numqi.channel.hf_amplitude_damping_kraus_op)

    def measure(self, index, seed=None, name='measure'):
        gate = MeasureGate(index, seed, name)
        self.gate_index_list.append((gate,gate.index))
        return gate

    def to_unitary(self):
        assert all(x[0].kind!='measure' for x in self.gate_index_list)
        num_qubit = self.num_qubit
        num_state = 2**num_qubit
        ret = np.eye(num_state, dtype=np.complex128)
        for ind0 in range(num_state):
            ret[ind0] = self.apply_state(ret[ind0])
        ret = ret.T.copy()
        return ret

    @property
    def num_qubit(self):
        assert len(self.gate_index_list)>0
        ret = 0
        for gate_i,index_i in self.gate_index_list:
            if gate_i.kind in CANONICAL_GATE_KIND:
                if gate_i.kind=='control':
                    ret = max(ret, max(index_i[0]), max(index_i[1]))
                else:
                    ret = max(ret, max(index_i))
        ret = ret + 1
        return ret

    def shift_qubit_index_(self, delta):
        if delta!=0:
            for ind0 in range(len(self.gate_index_list)):
                gate_i,index_i = self.gate_index_list[ind0]
                if gate_i.kind in CANONICAL_GATE_KIND:
                    if gate_i.kind=='unitary':
                        self.gate_index_list[ind0] = gate_i, tuple(x+delta for x in index_i)
                    elif gate_i.kind=='control':
                        self.gate_index_list[ind0] = gate_i, ({(x+delta) for x in index_i[0]}, tuple((x+delta) for x in index_i[1]))
                    elif gate_i.kind=='measure':
                        assert index_i==gate_i.index
                        tmp0 = tuple(x+delta for x in index_i)
                        self.gate_index_list[ind0] = gate_i, tmp0
                        gate_i.index = tmp0

    def apply_state(self, q0):
        for gate,index in self.gate_index_list:
            if gate.kind=='unitary':
                q0 = numqi.sim.state.apply_gate(q0, gate.array, index)
            elif gate.kind=='control':
                q0 = numqi.sim.state.apply_control_n_gate(q0, gate.array, index[0], index[1])
            elif gate.kind=='measure':
                q0 = gate.forward(q0)
            elif gate.kind=='custom':
                q0 = gate.forward(q0)
            else:
                assert False, f'{gate} not supported'
        return q0

# TODO ch see qiskit
# TODO when should we use torch, when should we use numpy only

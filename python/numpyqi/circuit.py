import inspect
import numpy as np

try:
    import torch
except ImportError:
    torch = None

import numpyqi.state
import numpyqi.gate
import numpyqi.channel

from .utils import hf_tuple_of_int, hf_tuple_of_any

CANONICAL_GATE_KIND = {'unitary','control'}

def circuit_apply_state(q0, gate_index_list):
    for gate_i,index_i in gate_index_list:
        if gate_i.kind=='unitary':
            q0 = numpyqi.state.apply_gate(q0, gate_i.array, index_i)
        elif gate_i.kind=='control':
            q0 = numpyqi.state.apply_control_n_gate(q0, gate_i.array, index_i[0], index_i[1])
        elif gate_i.kind=='custom':
            q0 = gate_i.forward(q0)
        else:
            assert False, f'{gate_i} not supported'
    return q0


def circuit_apply_state_grad(q0_conj, q0_grad, gate_index_list):
    op_grad_list = []
    for gate_i,index_i in reversed(gate_index_list):
        if gate_i.kind=='control':
            q0_conj, q0_grad, op_grad = numpyqi.state.apply_control_n_gate_grad(
                    q0_conj, q0_grad, gate_i.array, index_i[0], index_i[1], tag_op_grad=gate_i.requires_grad)
        elif gate_i.kind=='unitary':
            q0_conj, q0_grad, op_grad = numpyqi.state.apply_gate_grad(q0_conj,
                    q0_grad, gate_i.array, index_i, tag_op_grad=gate_i.requires_grad)
        elif gate_i.kind=='custom':
            q0_conj, q0_grad, op_grad = gate_i.grad_backward(q0_conj, q0_grad)#TODO
        else:
            raise KeyError(f'not recognized gate "{gate_i}"')
        if op_grad is not None:
            gate_i.grad += op_grad
        op_grad_list.append(op_grad)
    ret = q0_conj,q0_grad,op_grad_list[::-1]
    return ret

def _unitary_gate(name_, array, num_index):
    def hf0(self, /, *index, name=name_):
        gate = numpyqi.gate.Gate('unitary', array, name=name)
        index = hf_tuple_of_int(index)
        assert len(index)==num_index
        self.gate_index_list.append((gate, hf_tuple_of_int(index)))
        return gate
    return hf0


def _control_gate(name_, array):
    def hf0(self, control_qubit, target_qubit, name=name_):
        control_qubit = set(sorted(hf_tuple_of_int(control_qubit)))
        target_qubit = hf_tuple_of_int(target_qubit)
        assert all((x not in control_qubit) for x in target_qubit) and len(target_qubit)==len(set(target_qubit))
        gate = numpyqi.gate.Gate('control', array, name=name)
        self.gate_index_list.append((gate, (control_qubit,target_qubit)))
        return gate
    return hf0


def _unitary_parameter_gate(name_, hf0, num_index):
    num_parameter = len(inspect.signature(hf0).parameters)
    def hf1(self, index, args=None, name=name_, requires_grad=None):
        if requires_grad is None:
            requires_grad = self.default_requires_grad
        if args is None:
            args = (0.,)*num_parameter #initialize to zero
        else:
            args = hf_tuple_of_any(args, type_=float) #convert float/int into tuple
        gate = numpyqi.gate.ParameterGate('unitary', hf0, args, name=name, requires_grad=requires_grad)
        self.check_parameter_gate(gate)
        index = hf_tuple_of_int(index)
        assert len(index)==num_index
        self.gate_index_list.append((gate, index))
        return gate
    return hf1

def _control_parameter_gate(name_, hf0):
    num_parameter = len(inspect.signature(hf0).parameters)
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
        gate = numpyqi.gate.ParameterGate('control', hf0, args, name=name, requires_grad=requires_grad)
        self.check_parameter_gate(gate)
        self.gate_index_list.append((gate, (control_qubit,target_qubit)))
        return gate
    return hf1


def _kraus_gate(name_, hf0):
    def hf1(self, index, args, name=name_):
        kop = hf0(*args)
        gate = numpyqi.gate.Gate('kraus', kop, requires_grad=False, name=name)
        self.gate_index_list.append((gate, hf_tuple_of_int(index)))
        return gate
    return hf1


class Circuit:
    def __init__(self, default_requires_grad=False):
        self.gate_index_list = []
        if torch is None:
            assert not default_requires_grad, 'pytorch is required for default_requires_grad=True'
        self.default_requires_grad = default_requires_grad
        self.name_to_pgate = dict()

    def register_custom_gate(self, name, gate_class):
        # gate_class could not be child of cupysim.gate.Gate if not is_pgate
        # gate_class must be child of cupysim.gate.ParameterGate if is_pgate
        is_pgate = issubclass(gate_class, numpyqi.gate.ParameterGate)
        def hf0(self, *args, **kwargs):
            gate = gate_class(*args, **kwargs)
            if is_pgate:
                # TODO WARNING default_requires_grad is not checked here
                self.check_parameter_gate(gate)
            index = gate.index if (gate.kind in CANONICAL_GATE_KIND) else ()
            #leave index to empty for unkown gate
            self.gate_index_list.append((gate,index))
            return gate
        assert not hasattr(self, name)
        # https://stackoverflow.com/a/1015405
        tmp0 = hf0.__get__(self, self.__class__)
        setattr(self, name, tmp0)

    def check_parameter_gate(self, pgate):
        index = len(self.gate_index_list)
        num_parameter = len(pgate.args)
        name = pgate.name
        if pgate.name in self.name_to_pgate:
            tmp0 = self.gate_index_list[self.name_to_pgate[name]['index'][0]][0]
            assert tmp0.hf0 == pgate.hf0
            self.name_to_pgate[name]['index'].append(index)
            if pgate.requires_grad and (id(pgate) not in self.name_to_pgate[name]['grad_index_set']):
                self.name_to_pgate[name]['grad_index'].append(index)
                self.name_to_pgate[name]['grad_index_set'].add(id(pgate))
        else:
            self.name_to_pgate[name] = {
                'index': [index],
                'grad_index': [index] if pgate.requires_grad else [],
                'grad_index_set': {id(pgate)} if pgate.requires_grad else set(),
                'num_parameter': num_parameter,
                'hf0': pgate.hf0,
            }

    def single_qubit_gate(self, np0, ind0, requires_grad=False, name='single'):
        assert np0.shape==(2,2)
        gate = numpyqi.gate.Gate('unitary', np0, requires_grad, name=name)
        index = int(ind0),
        self.gate_index_list.append((gate, index))
        return gate

    def double_qubit_gate(self, np0, ind0, ind1, requires_grad=False, name='double'):
        assert np0.shape==(4,4)
        gate = numpyqi.gate.Gate('unitary', np0, requires_grad, name=name)
        index = int(ind0),int(ind1)
        self.gate_index_list.append((gate, index))
        return gate

    def controlled_single_qubit_gate(self, np0, ind_control_set, ind_target, requires_grad=False, name='control'):
        ind_control_set = set(sorted(hf_tuple_of_int(ind_control_set)))
        ind_target = hf_tuple_of_int(ind_target)
        assert len(ind_target)==1
        assert all((x not in ind_control_set) for x in ind_target) and len(ind_target)==len(set(ind_target))
        assert np0.shape==(2,2)
        gate = numpyqi.gate.Gate('control', np0, requires_grad, name=name)
        self.gate_index_list.append((gate, (ind_control_set, ind_target)))
        return gate

    def controlled_double_qubit_gate(self, np0, ind_control_set, ind_target, requires_grad=False, name='control'):
        ind_control_set = set(sorted(hf_tuple_of_int(ind_control_set)))
        ind_target = hf_tuple_of_int(ind_target)
        assert len(ind_target)==2
        assert all((x not in ind_control_set) for x in ind_target) and len(ind_target)==len(set(ind_target))
        assert np0.shape==(4,4)
        gate = numpyqi.gate.Gate('control', np0, requires_grad, name=name)
        self.gate_index_list.append((gate, (ind_control_set, ind_target)))
        return gate

    X = _unitary_gate('X', numpyqi.gate.pauli.sx, 1)
    Y = _unitary_gate('Y', numpyqi.gate.pauli.sy, 1)
    X = _unitary_gate('Z', numpyqi.gate.pauli.sz, 1)
    H = _unitary_gate('H', numpyqi.gate.H, 1)
    S = _unitary_gate('S', numpyqi.gate.S, 1)
    T = _unitary_gate('T', numpyqi.gate.T, 1)
    Swap = _unitary_gate('Swap', numpyqi.gate.Swap, 1)

    cnot = _control_gate('cnot', numpyqi.gate.pauli.sx)
    cx = cnot
    cy = _control_gate('cy', numpyqi.gate.pauli.sy)
    cz = _control_gate('cz', numpyqi.gate.pauli.sz)

    rx = _unitary_parameter_gate('rx', numpyqi.gate.rx, 1)
    ry = _unitary_parameter_gate('ry', numpyqi.gate.ry, 1)
    rz = _unitary_parameter_gate('rz', numpyqi.gate.rz, 1)
    u3 = _unitary_parameter_gate('u3', numpyqi.gate.u3, 1)
    rzz = _unitary_parameter_gate('rzz', numpyqi.gate.rzz, 2)

    crx = _control_parameter_gate('crx', numpyqi.gate.rx)
    cry = _control_parameter_gate('cry', numpyqi.gate.rx)
    crz = _control_parameter_gate('crz', numpyqi.gate.rx)
    cu3 = _control_parameter_gate('cu3', numpyqi.gate.u3)

    dephasing = _kraus_gate('dephasing', numpyqi.channel.hf_dephasing_kraus_op)
    depolarizing = _kraus_gate('depolarizing', numpyqi.channel.hf_depolarizing_kraus_op)
    amplitude_damping = _kraus_gate('amplitude_damping', numpyqi.channel.hf_amplitude_damping_kraus_op)

    def to_unitary(self):
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
                    if gate_i.kind!='control':
                        self.gate_index_list[ind0] = gate_i, tuple(x+delta for x in index_i)
                    else:
                        self.gate_index_list[ind0] = gate_i, ({(x+delta) for x in index_i[0]}, tuple((x+delta) for x in index_i[1]))

    def init_theta_torch(self, force=False):
        ret = force or any((len(x['grad_index'])>0) and ('theta_torch' not in x)
                        for x in self.name_to_pgate.values())
        if ret:
            for key,value in self.name_to_pgate.items():
                if len(value['grad_index'])==0:
                    continue
                tmp1 = np.array([self.gate_index_list[x][0].args for x in value['index']])
                # tmp1 = np_rng.uniform(min_, max_, size=(len(value['grad_index']),value['num_parameter']))
                if 'theta_torch' in self.name_to_pgate[key]:
                    self.name_to_pgate[key]['theta_torch'].data[:] = torch.tensor(tmp1, dtype=torch.float64, requires_grad=True)
                else:
                    theta_torch = torch.nn.Parameter(torch.tensor(tmp1, dtype=torch.float64, requires_grad=True))
                    self.name_to_pgate[key]['theta_torch'] = theta_torch
            self.update_gate()
        return ret

    def get_trainable_parameter(self):
        self.init_theta_torch()
        tmp0 = {k:v['theta_torch'] for k,v in self.name_to_pgate.items() if ('theta_torch' in v)}
        ret = torch.nn.ParameterDict(tmp0)
        return ret

    def update_gate(self):
        self.init_theta_torch()
        for key,value in self.name_to_pgate.items():
            if len(value['grad_index'])==0:
                continue
            theta_torch = value['theta_torch']
            gate_torch = value['hf0'](*theta_torch.T)
            theta_np = theta_torch.detach().numpy()
            gate_np = gate_torch.detach().numpy()
            for ind0,ind1 in enumerate(value['grad_index']):
                self.gate_index_list[ind1][0].set_args(theta_np[ind0], gate_np[ind0])
            self.name_to_pgate[key]['gate_torch'] = gate_torch

    def apply_state(self, q0, tag_update=True):
        tmp0 = self.init_theta_torch()
        if (not tmp0) and tag_update:
            self.update_gate()
        ret = circuit_apply_state(q0, self.gate_index_list)
        return ret

    def _gate_grad(self):
        for value in self.name_to_pgate.values():
            if len(value['grad_index'])==0:
                continue
            tmp0 = torch.tensor(np.stack([self.gate_index_list[x][0].grad for x in value['grad_index']], axis=0))
            value['gate_torch'].backward(tmp0)

    def zero_grad_(self):
        for x in self.name_to_pgate.values():
            for y in x['grad_index']:
                self.gate_index_list[y][0].zero_grad_()

    def apply_state_grad(self, q0, q0_grad, tag_zero_grad=True):
        if tag_zero_grad:
            self.zero_grad_()
        q0_conj = np.conj(q0)
        q0_conj,q0_grad,op_grad_list = circuit_apply_state_grad(q0_conj, q0_grad, self.gate_index_list)
        self._gate_grad()
        q0 = np.conj(q0_conj)
        return q0, q0_grad, op_grad_list

# TODO ch see qiskit
# TODO when should we use torch, when should we use numpy only

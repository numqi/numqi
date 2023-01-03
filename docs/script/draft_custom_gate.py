import functools
import itertools
import numpy as np

import numpyqi


@functools.lru_cache
def _measure_quantum_vector_hf0(num_qubit, index):
    tmp0 = sorted(set(index))
    assert (len(tmp0)==len(index)) and all(x==y for x,y in zip(tmp0,index))
    shape = [2]*num_qubit
    kind = np.zeros(len(shape), dtype=np.int64)
    kind[list(index)] = 1
    kind = kind.tolist()
    hf0 = lambda x: x[0]
    hf1 = lambda x: int(np.prod([y for _,y in x]))
    z0 = [(k,hf1(x)) for k,x in itertools.groupby(zip(kind,shape), key=hf0)]
    shape = tuple(x[1] for x in z0)
    keep_dim = tuple(x for x,y in enumerate(z0) if y[0]==1)
    reduce_dim = tuple(x for x,y in enumerate(z0) if y[0]==0)
    return shape,keep_dim,reduce_dim


def measure_quantum_vector(q0, index, seed=None):
    np_rng = numpyqi.random.get_numpy_rng(seed)
    index = numpyqi.utils.hf_tuple_of_int(index)
    assert all(x==y for x,y in zip(sorted(index),index)), 'index must be sorted'
    num_qubit = numpyqi.utils.hf_num_state_to_num_qubit(q0.shape[0])
    shape,keep_dim,reduce_dim = _measure_quantum_vector_hf0(num_qubit, index)
    q1 = q0.reshape(shape)
    prob = np.linalg.norm(q1, axis=reduce_dim).reshape(-1)**2
    ind1 = np_rng.choice(len(prob), p=prob)
    bitstr = [int(x) for x in bin(ind1)[2:].rjust(len(shape),'0')]
    ind1a = np.unravel_index(ind1, tuple(shape[x] for x in keep_dim))
    ind2 = [slice(None)]*len(shape)
    for x,y in zip(keep_dim, ind1a):
        ind2[x] = y
    ind2 = tuple(ind2)
    q2 = np.zeros_like(q1)
    q2[ind2] = q1[ind2] / np.sqrt(prob[ind1])
    q2 = q2.reshape(-1)
    return bitstr,prob,q2


class MeasureGate:
    def __init__(self, index, seed=None, name='measure'):
        self.kind = 'custom'
        self.name = name
        self.requires_grad = False
        index = numpyqi.utils.hf_tuple_of_int(index)
        assert all(x==y for x,y in zip(sorted(index),index)), 'index must be sorted'
        self.index = index
        self.np_rng = numpyqi.random.get_numpy_rng(seed)

        self.bitstr = None
        self.probability = None

    def forward(self, q0):
        self.bitstr,self.probability,q1 = measure_quantum_vector(q0, self.index, self.np_rng)
        return q1

    # def grad_backward():

num_qubit = 5
num_layer = 3
measure_gate_list = []
circ = numpyqi.circuit.Circuit()
circ.register_custom_gate('measure', MeasureGate)
np_rng = np.random.default_rng()

for _ in range(num_layer):
    for ind0 in range(num_qubit):
        circ.u3(ind0, args=np_rng.uniform(0, 2*np.pi, size=3))
    tmp0 = list(range(0, num_qubit-1, 2)) + list(range(1, num_qubit-1, 2))
    for ind0 in tmp0:
        circ.cnot(ind0, ind0+1)
    measure_gate_list.append(circ.measure(index=(0,1)))

q0 = numpyqi.state.new_base(num_qubit)
q1 = circ.apply_state(q0)
print(np.linalg.norm(q1)) #1
print('probability:', measure_gate_list[0].probability)
for ind0,gate_i in enumerate(measure_gate_list):
    print(f'[gate-{ind0}] bitstr:', gate_i.bitstr)

import numpy as np

import numpyqi

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
        self.bitstr,self.probability,q1 = numpyqi.sim.state.measure_quantum_vector(q0, self.index, self.np_rng)
        return q1

## Bell state
circ = numpyqi.sim.Circuit()
# actually, we have .measure() gate. For demonstration, we name it as .measure_custom()
circ.register_custom_gate('measure_custom', MeasureGate)
circ.H(0)
circ.cnot(0, 1)
gate_measure = circ.measure_custom(index=(0,1))
# gate_measure = circ.measure(index=(0,1))

q0 = numpyqi.sim.state.new_base(num_qubit=2)
q1 = circ.apply_state(q0)
assert tuple(gate_measure.bitstr) in {(0,0),(1,1)} #result must be either 00 or 11
assert np.abs(gate_measure.probability-np.abs([0.5,0,0,0.5])).max() < 1e-7
if tuple(gate_measure.bitstr)==(0,0):
    assert np.abs(q1-np.array([1,0,0,0])).max() < 1e-7
else:
    assert np.abs(q1-np.array([0,0,0,1])).max() < 1e-7


## GHZ state
circ = numpyqi.sim.Circuit()
circ.register_custom_gate('measure_custom', MeasureGate)
circ.H(0)
circ.cnot(0, 1)
circ.cnot(1, 2)
gate_measure = circ.measure_custom(index=(0,1,2))

q0 = numpyqi.sim.state.new_base(num_qubit=3)
q1 = circ.apply_state(q0)
assert tuple(gate_measure.bitstr) in {(0,0,0),(1,1,1)}
assert np.abs(gate_measure.probability-np.abs([0.5,0,0,0,0,0,0,0.5])).max() < 1e-7
if tuple(gate_measure.bitstr)==(0,0,0):
    assert np.abs(q1-np.array([1,0,0,0,0,0,0,0])).max() < 1e-7
else:
    assert np.abs(q1-np.array([0,0,0,0,0,0,0,1])).max() < 1e-7


## measure with many layers
num_qubit = 5
num_layer = 3
measure_gate_list = []
circ = numpyqi.sim.Circuit()
circ.register_custom_gate('measure_custom', MeasureGate)
np_rng = np.random.default_rng()

for _ in range(num_layer):
    for ind0 in range(num_qubit):
        circ.u3(ind0, args=np_rng.uniform(0, 2*np.pi, size=3))
    tmp0 = list(range(0, num_qubit-1, 2)) + list(range(1, num_qubit-1, 2))
    for ind0 in tmp0:
        circ.cnot(ind0, ind0+1)
    measure_gate_list.append(circ.measure_custom(index=(0,1)))

q0 = numpyqi.sim.state.new_base(num_qubit)
q1 = circ.apply_state(q0)
print(np.linalg.norm(q1)) #1
print('probability:', measure_gate_list[0].probability)
for ind0,gate_i in enumerate(measure_gate_list):
    print(f'[gate-{ind0}] bitstr:', gate_i.bitstr)

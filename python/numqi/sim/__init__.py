from .state import new_base
from .circuit import Circuit, CircuitTorchWrapper
from .clifford import CliffordCircuit
from ._internal import Gate, ParameterGate

from . import state
from . import dm
from . import circuit
from . import clifford


# TODO redesign numqi.sim.Circuit
'''
circ = numqi.sim。Circuit(requires_grad=True)
circ.rx(3) #trainable parameter
circ.cnot(0, 1)
circ.rx(3, circ.P['rx'][0]) #placeholder
circ.two_qubit_gate(0, 1, circ.P['u4'][0])
circ.setP(rx=xxx, u4=yyy)
circ(q0)

circ_torch = numqi.sim.CircuitTorchWrapper(circ)
circ_torch.setP(rx=xxx, u4=yyy)
q1 = circ_torch(q0)
loss = hf0(q1)
loss.backward()
'''

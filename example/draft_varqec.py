import numpy as np

import numqi

np_rng = np.random.default_rng()

def build_circuit(num_depth, num_qubit):
    prime = [x for x in range(3, num_qubit) if np.gcd(x,num_qubit)==1]
    assert len(prime)>=1
    prime = prime[0]
    circ = numqi.sim.Circuit(default_requires_grad=True)
    for _ in range(num_depth):
        for x in range(num_qubit):
            circ.u3(x)
        tmp0 = np.mod(-np.arange(num_qubit+1), num_qubit)
        for x,y in zip(tmp0[:-1],tmp0[1:]):
            circ.cu3(x, y)
        for x in range(num_qubit):
            circ.u3(x)
        tmp0 = np.mod(-np.arange(num_qubit+1)*prime, num_qubit)
        for x,y in zip(tmp0[:-1],tmp0[1:]):
            circ.cu3(x, y)
    return circ

str_qecc = '((5,2,3))' #'((6,2,de(2)=4))'
tmp0 = numqi.qec.parse_str_qecc(str_qecc)
num_qubit = tmp0['num_qubit']
num_logical_dim = tmp0['num_logical_dim']
distance = tmp0['distance']
weight_z = tmp0['weight_z']
num_layer = 5

if weight_z is None:
    error_list = numqi.qec.make_error_list(num_qubit, distance)
else:
    error_list = numqi.qec.make_asymmetric_error_set(num_qubit, distance, weight_z)

circuit = build_circuit(num_layer, num_qubit)
model = numqi.qec.VarQEC(circuit, num_logical_dim, error_list, loss_type='L2')
theta_optim = numqi.optimize.minimize(model, ('uniform',0,2*np.pi), num_repeat=1, tol=1e-10, print_freq=20)
code0 = model.get_code()
theta_optim = numqi.optimize.minimize(model, ('uniform',0,2*np.pi), num_repeat=1, tol=1e-10, print_freq=20)
code1 = model.get_code()

model1 = numqi.qec.QECCEqualModel(code0, code1)
tmp0 = numqi.optimize.minimize(model, ('uniform',-1,1), num_repeat=1, tol=1e-10, print_freq=20)
if tmp0.fun<1e-5:
    print('equivalent QECC')

model = numqi.qec.VarQECUnitary(num_qubit, num_logical_dim, error_list)
theta_optim = numqi.optimize.minimize(model, ('uniform',-1,1), num_repeat=1, tol=1e-10, print_freq=20)
z0 = model.get_code()
numqi.qec.degeneracy(z0[0])

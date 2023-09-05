import numpy as np

import numqi


def get_hamiltonian(num_qubit):
    assert num_qubit >= 2
    ham_drift = 0
    tmp0 = sum(np.kron(x,x) for x in [numqi.gate.X,numqi.gate.Y,numqi.gate.Z])/2
    for ind0 in range(num_qubit-1):
        tmp1 = tmp0
        if ind0 > 0:
            tmp1 = np.kron(np.eye(2**ind0), tmp1)
        if ind0 < (num_qubit-2):
            tmp1 = np.kron(tmp1, np.eye(2**(num_qubit-ind0-2)))
        ham_drift = ham_drift + tmp1

    ham_control = []
    for ind0 in range(num_qubit):
        tmp0 = numqi.gate.X/2
        tmp1 = numqi.gate.Y/2
        if ind0 > 0:
            tmp0 = np.kron(np.eye(2**ind0), tmp0)
            tmp1 = np.kron(np.eye(2**ind0), tmp1)
        if ind0 < (num_qubit-1):
            tmp0 = np.kron(tmp0, np.eye(2**(num_qubit-ind0-1)))
            tmp1 = np.kron(tmp1, np.eye(2**(num_qubit-ind0-1)))
        ham_control.append(tmp0)
        ham_control.append(tmp1)
    ham_control = np.stack(ham_control, axis=0)
    return ham_drift, ham_control

def test_GrapeModel():
    num_qubit = 5
    tspan = np.linspace(0, 10, 101)
    q0 = numqi.random.rand_state(2**num_qubit)
    q1 = numqi.random.rand_state(2**num_qubit)
    ham_drift, ham_control = get_hamiltonian(num_qubit)

    # smooth_weight=0 converge in 60 steps
    # smooth_weight=0.001 converge requires more than 1k steps
    model = numqi.optimal_control.GrapeModel(ham_drift, ham_control, tspan, smooth_weight=0)
    model.set_state_vector(q0, q1)
    theta_optim = numqi.optimize.minimize(model, ('uniform',-0.001,0.001), num_repeat=3,
                    tol=1e-10, early_stop_threshold=-1+1e-7, print_freq=0, method='L-BFGS-B')
    assert theta_optim.fun < (-1+1e-7)

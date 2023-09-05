import numpy as np

import numqi


def hf_grover_oracle_wrapper(x:int):
    def hf0(q0):
        q0 = q0.copy()
        q0[x] *= -1
        return q0
    return hf0


num_qubit = 4
num_query = 3
model = numqi.query.QueryGroverModel(num_qubit, num_query, use_fractional=False)
theta_optim = numqi.optimize.minimize(model, theta0=('uniform', -1, 1), tol=1e-10, num_repeat=3, early_stop_threshold=1e-4, print_freq=500)
print('error rate:', model.error_rate)
print('loss function:', theta_optim.fun)

if theta_optim.fun < 1e-5:
    np_rng = np.random.default_rng()
    xstar = np_rng.integers(2**num_qubit)
    hf_oracle = hf_grover_oracle_wrapper(xstar)

    unitary_list = numqi.param.real_matrix_to_special_unitary(model.theta.detach().numpy()).transpose(0,2,1)
    q0 = np.zeros(2**num_qubit, dtype=np.complex128)
    q0[0] = 1
    for ind0 in range(num_query):
        q0 = unitary_list[ind0] @ q0
        q0 = hf_oracle(q0)
    q0 = unitary_list[-1] @ q0
    prob = (q0.conj() * q0).real
    x_found = np.argmax(prob)
    assert x_found==xstar
    assert abs(prob.max() - 1) < 1e-4

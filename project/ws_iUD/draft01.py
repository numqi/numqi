import numpy as np
import numqi

hf_trace0 = lambda x: x-(np.trace(x)/x.shape[0])*np.eye(x.shape[0])

dim = 3
num_op = 4
psi = numqi.random.rand_state(dim)
op_list = np.stack([hf_trace0(numqi.random.rand_hermite_matrix(dim)) for _ in range(num_op)])

expectation_value = ((op_list @ psi) @ psi.conj()).real
model = numqi.unique_determine.FindStateWithOpModel(op_list, use_dm=False)
model.set_expectation(expectation_value)
fidelity_list = []
for _ in range(10):
    theta_optim = numqi.optimize.minimize(model, theta0='uniform', num_repeat=1, tol=1e-10, print_every_round=0)
    psi_new = model.get_state()
    if theta_optim.fun < 1e-10:
        fidelity_list.append(abs(np.vdot(psi, psi_new))**2)
print(min(fidelity_list), len(fidelity_list))

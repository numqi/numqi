import itertools
import functools
import numpy as np
import numqi

hf_trace0 = lambda x: x-(np.trace(x)/x.shape[0])*np.eye(x.shape[0])

def get_Wtype_state(np0):
    np0 = np0 / np.linalg.norm(np0)
    N0 = np0.shape[0]
    ret = np.zeros(2**N0, dtype=np0.dtype)
    ret[2**np.arange(N0)] = np0
    return ret

def get_nlocal_measurement_set(num_party, num_local):
    if not hasattr(num_local, '__len__'):
        num_local_list = int(num_local),
    else:
        num_local_list = tuple(sorted({int(x) for x in num_local}))
    assert all(0<=x<=num_party for x in num_local_list)
    ret = []
    pauli = [numqi.gate.X, numqi.gate.Y, numqi.gate.Z]
    for num_local in num_local_list:
        if num_local==0:
            ret.append(np.eye(2**num_party))
        else:
            for ind0 in itertools.combinations(range(num_party), num_local):
                tmp0 = [tuple(range(3))]*num_local
                for ind1 in itertools.product(*tmp0):
                    tmp1 = [numqi.gate.I for _ in range(num_party)]
                    for x,y in zip(ind0,ind1):
                        tmp1[x] = pauli[y]
                    for x0,x1 in zip(ind0, ind1):
                        tmp1[x0] = pauli[x1]
                    ret.append(functools.reduce(np.kron, tmp1))
    ret = np.stack(ret)
    return ret


dim = 3
num_op = 4
psi = numqi.random.rand_haar_state(dim)
op_list = np.stack([hf_trace0(numqi.random.rand_hermitian_matrix(dim)) for _ in range(num_op)])

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



np_rng = np.random.default_rng()
hf_randc = lambda *x: np_rng.normal(size=x) + 1j*np_rng.normal(size=x)

num_qubit = 5
tmp0 = hf_randc(num_qubit)
np0 = hf_randc(num_qubit)/np.linalg.norm(tmp0)
op_list = get_nlocal_measurement_set(num_qubit, (0,1,2))
psi = get_Wtype_state(np0)

expectation_value = ((op_list @ psi) @ psi.conj()).real
model = numqi.unique_determine.FindStateWithOpModel(op_list, use_dm=False)
model.set_expectation(expectation_value)
fidelity_list = []
for _ in range(30):
    theta_optim = numqi.optimize.minimize(model, theta0='uniform', num_repeat=1, tol=1e-10, print_every_round=0)
    psi_new = model.get_state()
    if theta_optim.fun < 1e-7:
        fidelity_list.append(abs(np.vdot(psi, psi_new))**2)
print(min(fidelity_list), len(fidelity_list))

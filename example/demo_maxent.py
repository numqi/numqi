import time
import functools
import numpy as np

import numqi

hf_kron = lambda *x: functools.reduce(np.kron, x)
hf_trace0 = lambda x: x-(np.trace(x)/x.shape[0])*np.eye(x.shape[0])

np_rng = np.random.default_rng()


def demo_2qubit_pauli():
    op_list = [
        hf_kron(numqi.gate.X, numqi.gate.X),
        hf_kron(numqi.gate.Z, numqi.gate.I),
    ]
    model = numqi.maximum_entropy.MaximumEntropyModel(op_list)

    term_value = np.array([0.5, 0.5])
    model.set_target(term_value)
    theta_optim0 = numqi.optimize.minimize(model, theta0='uniform', num_repeat=1, tol=1e-10)
    model.theta
    model.dm_torch
    model.term_value

    term_value = np.array([2, 2])
    model.set_target(term_value)
    theta_optim0 = numqi.optimize.minimize(model, theta0='uniform', num_repeat=1, tol=1e-10)

    term_value_target, term_value_list, EVL_list = numqi.maximum_entropy.get_maximum_entropy_model_boundary(model, radius=1.5)
    term_value = np.array([-1.3,-1.5])
    coeffA, coeffC = model.get_witness(term_value)
    fig,ax = numqi.maximum_entropy.draw_maximum_entropy_model_boundary(term_value_target, term_value_list,
                    EVL_list, witnessA=coeffA, witnessC=coeffC)
    fig.savefig('tbd00.png', dpi=200)
    # fig.savefig(hf_data('maxent_2qubit_pauli.png'), dpi=200)


def demo_3qubit_2local_random():
    np_rng = np.random.default_rng(233) #233
    tmp0 = [
        hf_kron(numqi.random.rand_hermite_matrix(4, seed=np_rng), numqi.gate.I),
        hf_kron(numqi.gate.I, numqi.random.rand_hermite_matrix(4, seed=np_rng)),
    ]
    tmp0 = [hf_trace0(x) for x in tmp0]
    op_list = [x*(2/np.linalg.norm(x.reshape(-1))) for x in tmp0] #make it norm-2, better in plotting
    model = numqi.maximum_entropy.MaximumEntropyModel(op_list)

    term_value_target, term_value_list, EVL_list = numqi.maximum_entropy.get_maximum_entropy_model_boundary(model, radius=1.5, index=(0,1))
    term_value = np.array([-1.4,-0.5])
    coeffA, coeffC = model.get_witness(term_value)
    assert coeffA is not None
    fig,ax = numqi.maximum_entropy.draw_maximum_entropy_model_boundary(term_value_target, term_value_list,
                    EVL_list, witnessA=coeffA, witnessC=coeffC)
    fig.savefig('tbd00.png', dpi=200)
    # fig.savefig('maxent_3qubit_2local_random.png', dpi=200)


def demo_3qubit_2local():
    num_qubit = 3
    op_list = numqi.maximum_entropy.get_1dchain_2local_pauli_basis(num_qubit, with_I=False)
    model = numqi.maximum_entropy.MaximumEntropyModel(op_list)

    state = np.array([0,1,1,0,1,0,0,0])/np.sqrt(3) #W-state
    term_value_target = ((op_list @ state) @ state.conj()).real
    model.set_target(term_value_target)
    theta_optim0 = numqi.optimize.minimize(model, theta0='uniform', num_repeat=3, tol=1e-12)
    if theta_optim0.fun < 1e-9: #fail to converge sometime, just re-run it
        rho = model.dm_torch.detach().numpy()
        rank = np.sum(np.linalg.eigvalsh(rho)>1e-4)
        assert rank==1

    state = np.array([1,0,0,0,0,0,0,1])/np.sqrt(2) #GHZ-state
    term_value_target = ((op_list @ state) @ state.conj()).real
    model.set_target(term_value_target)
    theta_optim0 = numqi.optimize.minimize(model, theta0='uniform', num_repeat=3, tol=1e-12)
    if theta_optim0.fun < 1e-9: #fail to converge sometime, just re-run it
        rho = model.dm_torch.detach().numpy()
        rank = np.sum(np.linalg.eigvalsh(rho)>1e-4)
        assert rank==2

        coeffA,coeffC = model.get_witness(term_value_target*1.1)
        assert coeffA is not None
        for _ in range(1000):
            rho = numqi.random.rand_density_matrix(2**num_qubit)
            z0 = np.trace(op_list @ rho, axis1=1, axis2=2).real
            assert np.dot(z0 - coeffA, coeffC) >= 0


def demo_4qubit_entanglement_monogamy():
    num_qubit = 4
    op_2qubit_list = numqi.maximum_entropy.get_1dchain_2local_pauli_basis(2, with_I=False)
    op_list = numqi.maximum_entropy.get_1dchain_2local_pauli_basis(num_qubit, with_I=False)
    model = numqi.maximum_entropy.MaximumEntropyModel(op_list)

    state = np.array([1,0,0,1])/np.sqrt(2) #Bell state
    tmp0 = ((op_2qubit_list @ state) @ state.conj()).real
    term_value_target = np.tile(tmp0, num_qubit-1)

    # due to entanglement monogamy, this must be a witness
    coeffA,coeffC = model.get_witness(term_value_target*1.1)
    assert coeffA is not None
    for _ in range(1000):
        rho = numqi.random.rand_density_matrix(2**num_qubit)
        z0 = np.trace(op_list @ rho, axis1=1, axis2=2).real
        assert np.dot(z0 - coeffA, coeffC) >= 0



def demo_4qubit_2local_random():
    np_rng = np.random.default_rng(233)
    num_qubit = 4
    op_2qubit_list = numqi.maximum_entropy.get_1dchain_2local_pauli_basis(2, with_I=False)
    op_list = numqi.maximum_entropy.get_1dchain_2local_pauli_basis(num_qubit, with_I=False)
    model = numqi.maximum_entropy.MaximumEntropyModel(op_list)

    tmp0 = [numqi.random.rand_density_matrix(4, seed=np_rng) for _ in range(num_qubit-1)]
    term_value = np.concatenate([np.trace(op_2qubit_list@x, axis1=1, axis2=2).real for x in tmp0], axis=0)

    index = np.sort(np_rng.permutation(len(term_value))[:2])
    term_value_target, term_value_list, EVL_list = numqi.maximum_entropy.get_maximum_entropy_model_boundary(
            model, radius=1.2, index=index, term_value_target=term_value, tol=1e-7, num_repeat=3, num_point=50)
    fig,ax = numqi.maximum_entropy.draw_maximum_entropy_model_boundary(term_value_target, term_value_list, EVL_list, index=index, rank_radius=-1)
    fig.savefig('tbd00.png')

    coeffA,coeffC = model.get_witness(term_value_target[20])
    if coeffA is not None:
        fig,ax = numqi.maximum_entropy.draw_maximum_entropy_model_boundary(term_value_target, term_value_list, EVL_list,
                    index=index, witnessA=coeffA, witnessC=coeffC, rank_radius=-1)
        fig.savefig('tbd00.png', dpi=200)
        # fig.savefig('maxent_4qubit_2local_random.png', dpi=200)
        for _ in range(1000):
            rho = numqi.random.rand_density_matrix(2**num_qubit)
            z0 = np.trace(op_list @ rho, axis1=1, axis2=2).real
            assert np.dot(z0 - coeffA, coeffC) >= 0


def demo_benchmark_time():
    num_qubit_list = list(range(3,9))
    num_repeat = 3

    op_2qubit_list = numqi.maximum_entropy.get_1dchain_2local_pauli_basis(2, with_I=False)
    time_maxent = []
    time_sdp = []
    for num_qubit in num_qubit_list:
        op_list = numqi.maximum_entropy.get_1dchain_2local_pauli_basis(num_qubit, with_I=False)

        for _ in range(num_repeat):
            # make sure always has solution
            rho_target = numqi.random.rand_density_matrix(2**num_qubit, seed=np_rng)
            term_value_target = np.trace(op_list @ rho_target, axis1=1, axis2=2).real

            model = numqi.maximum_entropy.MaximumEntropyModel(op_list)
            model.set_target(term_value_target)
            t0 = time.time()
            while True:
                theta_optim0 = numqi.optimize.minimize(model, theta0='uniform', num_repeat=1, tol=1e-7)
                if theta_optim0.fun < 1e-6:
                    break
            time_maxent.append(time.time() - t0)

            t0 = time.time()
            # z0 = numqi.maximum_entropy.sdp_op_list_solve(op_list, term_value_target)
            z0 = numqi.maximum_entropy.sdp_2local_rdm_solve(term_value_target)
            time_sdp.append(time.time() - t0)
            print(num_qubit, time_maxent[-1], time_sdp[-1])
    time_maxent = np.array(time_maxent).reshape(-1, num_repeat)
    time_sdp = np.array(time_sdp).reshape(-1, num_repeat)

    print('| #qubit | maxent time (s) | sdp time (s)')
    print('| :-: | :-: | :-: |')
    for ind0 in range(len(time_maxent)):
        tmp0 = num_qubit_list[ind0]
        tmp1 = time_maxent[ind0].mean()
        tmp2 = time_sdp[ind0].mean()
        print(f'| {tmp0} | {tmp1:.3f} | {tmp2:.3f} |')


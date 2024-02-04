import numpy as np

import numqi

np_rng = np.random.default_rng()


def test_density_matrix_recovery_SDP():
    num_qubit = 3
    noise_rate = 0.01
    cvxpy_eps = 1e-6
    pauli_matrix_list = numqi.gate.get_pauli_group(num_qubit, use_sparse=False)
    tmp0 = numqi.unique_determine.load_pauli_ud_example(num_qubit)
    tmp1 = tmp0[np_rng.integers(0, len(tmp0))]
    matrix_subspace = numqi.unique_determine.get_matrix_list_indexing(pauli_matrix_list, tmp1)

    state0 = numqi.random.rand_haar_state(2**num_qubit)
    measure_no_noise = ((matrix_subspace @ state0) @ state0.conj()).real
    tmp0 = np_rng.normal(size=len(matrix_subspace))
    measure_with_noise = measure_no_noise + tmp0*(noise_rate/np.linalg.norm(tmp0))

    rho, eps = numqi.unique_determine.density_matrix_recovery_SDP(matrix_subspace, measure_with_noise, converge_eps=cvxpy_eps)
    tmp0 = np.linalg.norm(np.trace(matrix_subspace @ rho, axis1=1, axis2=2).real - measure_with_noise)
    assert abs(eps - tmp0) < 1e-5
    assert eps < noise_rate #mostly should be fine


def test_pauli_UDP_is_UDA():
    # (num_qubit, time-per-cpu): (3,6s) (4,40s)
    num_repeat_dict = {2:50, 3:50, 4:100} #, 5:100

    for num_qubit,num_repeat in num_repeat_dict.items():
        pauli_matrix_list = numqi.gate.get_pauli_group(num_qubit, use_sparse=True)
        tmp0 = numqi.unique_determine.load_pauli_ud_example(num_qubit)
        ind0 = np.sort(np_rng.choice(np.arange(len(tmp0)), size=1, replace=False))
        matB_list = [numqi.unique_determine.get_matrix_list_indexing(pauli_matrix_list, tmp0[x]) for x in ind0]
        z0 = numqi.unique_determine.check_UD('uda', matB_list, num_repeat=num_repeat, num_worker=1)
        # num_worker=1 pytest seems to limit the cpu usage, so num_worker>1 doesn't help
        assert all(x[0] for x in z0)



def test_get_qutrit_projector_basis():
    matrix_subspace = numqi.unique_determine.get_qutrit_projector_basis(num_qutrit=1)
    tmp0 = matrix_subspace.reshape(-1, 9)
    # full rank
    assert np.all(np.linalg.eigvalsh(tmp0.T @ tmp0.conj())>0)


def test_get_chebshev_orthonormal():
    for dim in range(3,7):
        alpha = np_rng.uniform(0, np.pi)
        _, basis_list = numqi.unique_determine.get_chebshev_orthonormal(dim, alpha, return_basis=True)
        for x in basis_list:
            assert np.abs(x @ x.T.conj() - np.eye(x.shape[0])).max() < 1e-10

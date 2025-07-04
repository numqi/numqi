import numpy as np
import numqi

np_rng = np.random.default_rng()

def test_weight_enumerator_transform_matrix():
    for n in [5,6,7]:
        mat_dict = numqi.qec.get_weight_enumerator_transform_matrix(n, kind='numpy')
        for key in ['M', "M1", "M2"]:
            np0 = mat_dict[key]
            assert np.abs(np0 @ np0 - np.eye(n+1, dtype=np.float64)).max() < 1e-10, f"self-inverse failed for {key}"
        for key in ['T1', 'T2', 'T3']:
            np0 = mat_dict[key]
            np1 = mat_dict['i' + key]
            assert np.abs(np0 @ np1 - np.eye(n+1, dtype=np.float64)).max() < 1e-10, f"inverse failed for {key}"

        assert np.abs(mat_dict['T1'] @ mat_dict['M'] @ mat_dict['iT1'] - mat_dict['M1']).max() < 1e-10, "M/M1 relation failed"
        assert np.abs(mat_dict['T2'] @ mat_dict['M'] @ mat_dict['iT2'] - mat_dict['M2']).max() < 1e-10, "M/M2 relation failed"
        assert np.abs(mat_dict['T3'] @ mat_dict['M1'] @ mat_dict['iT3'] - mat_dict['M2']).max() < 1e-10, "M1/M2 relation failed"

    # import qsalto #pip install qsalto
    # n = 7
    # ret = get_weight_enumerator_transform_matrix(n, kind='numpy')
    # for key,value in ret.items():
    #     assert np.abs(getattr(qsalto, key)(n) - value).max() < 1e-10


def test_get_knill_laflamme_matrix_indexing_over_vector():
    index_KL_matrix = numqi.qec.get_knill_laflamme_matrix_indexing_over_vector(num_qubit=7, distance=3)

    code,info = numqi.qec.get_code_subspace('723cyclic', lambda2=np_rng.uniform(0,7))
    error_scipy = numqi.qec.make_pauli_error_list_sparse(7, distance=3, kind='scipy-csr01')[1]
    lambda_a = ((error_scipy @ code[0]).reshape(-1,2**7) @ code[0].conj()).real
    lambda_ab = np.concat([np.ones(1), lambda_a], axis=0)[index_KL_matrix].reshape(-1, round(np.sqrt(index_KL_matrix.size)))

    error_scipy_a = numqi.qec.make_pauli_error_list_sparse(7, distance=2, kind='scipy-csr01')[1]
    tmp0 = np.concatenate([code[:1],(error_scipy_a @ code[0]).reshape(-1, 2**7)], axis=0)
    lambda_ab_ = tmp0.conj() @ tmp0.T
    assert np.abs(lambda_ab_.imag).max() < 1e-12
    lambda_ab_ = lambda_ab_.real
    assert np.abs(lambda_ab-lambda_ab_).max() < 1e-12

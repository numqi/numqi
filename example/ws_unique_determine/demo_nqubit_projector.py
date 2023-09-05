import numpy as np

import numqi


def get_qubit_projector_basis(num_qubit):
    assert num_qubit>=1
    hf0 = lambda x,y,z=1: np.array([[x*np.conj(x),x*np.conj(y)], [y*np.conj(x), y*np.conj(y)]])/(z*z)
    s2 = np.sqrt(2)
    tmp0 = [(1,0), (0,1), (1,1,s2), (1,-1,s2), (1,1j,s2), (1,-1j,s2)]
    np0 = np.stack([hf0(*x) for x in tmp0])
    ret = np0
    for _ in range(num_qubit-1):
        tmp0 = np.einsum(ret, [0,1,2], np0, [3,4,5], [0,3,1,4,2,5], optimize=True)
        tmp1 = [x*y for x,y in zip(ret.shape, np0.shape)]
        ret = tmp0.reshape(tmp1)
    ret = np.concatenate([np.eye(ret.shape[1])[np.newaxis], ret], axis=0)
    return ret


## 2 qubits, non-orthogonal matrix, UDA!=UDP
num_qubit = 2
matrix_subspace = get_qubit_projector_basis(num_qubit)
z0 = numqi.unique_determine.find_UDP_over_matrix_basis(num_round=10, matrix_basis=matrix_subspace, indexF=[0],
            num_repeat=100, num_random_select=15, tag_reduce=True, early_stop_threshold=0.0001, num_worker=10)
matB_list = [matrix_subspace[x] for x in z0]
z1 = numqi.unique_determine.check_UDP_matrix_subspace(matB_list, num_repeat=1000, num_worker=19, early_stop_threshold=0.0001, converge_tol=1e-7, dtype='float64')
z2 = numqi.unique_determine.check_UDA_matrix_subspace(matB_list, num_repeat=1000, num_worker=19, early_stop_threshold=0.0001, converge_tol=1e-7, dtype='float64')

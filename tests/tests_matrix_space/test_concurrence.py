import numpy as np

import numqi


def test_SubspaceGeneralizedConcurrenceModel_3qubit():
    # http://doi.org/10.1088/1402-4896/acec15 eq(59)
    s16 = 1/np.sqrt(6)
    s12 = 1/np.sqrt(2)
    index_value = [
        ([0,0,0,1], 2*s16), ([0,0,1,0], -s16), ([0,1,0,0], -s16),
        ([1,0,1,1], 2*s16), ([1,1,0,1], -s16), ([1,1,1,0], -s16),
        ([2,0,1,0], s12), ([2,1,0,0], -s12),
        ([3,1,0,1], s12), ([3,1,1,0], -s12),
    ]
    subspace = np.zeros((4,2,2,2), dtype=np.complex128)
    index = np.array([x[0] for x in index_value], dtype=np.int64).T
    subspace[index[0], index[1], index[2], index[3]] = np.array([x[1] for x in index_value])

    model = numqi.matrix_space.SubspaceGeneralizedConcurrenceModel(len(subspace))
    model.set_subspace_basis(subspace, use_tensor5=True)
    theta_optim = numqi.optimize.minimize(model, 'uniform', num_repeat=3, tol=1e-12)
    assert abs(theta_optim.fun-5/6) < 1e-8


def test_SubspaceGeneralizedConcurrenceModel_2qutrit():
    # http://doi.org/10.1088/1402-4896/acec15 eq(60)
    s16 = 1/np.sqrt(6)
    s12 = 1/np.sqrt(2)
    index_value = [
        ([0,0,2], 2*s16), ([0,1,1], -s16), ([0,2,0], -s16),
        ([1,0,1], s12), ([1,1,0], -s12),
        ([2,1,2], s12), ([2,2,1], -s12),
        ([3,1,1], s12), ([3,2,0], -s12),
    ]
    subspace = np.zeros((4,3,3), dtype=np.complex128)
    index = np.array([x[0] for x in index_value], dtype=np.int64).T
    subspace[index[0], index[1], index[2]] = np.array([x[1] for x in index_value])

    model = numqi.matrix_space.SubspaceGeneralizedConcurrenceModel(len(subspace))
    model.set_subspace_basis(subspace, use_tensor5=True)
    theta_optim = numqi.optimize.minimize(model, 'uniform', num_repeat=3, tol=1e-12)
    assert abs(theta_optim.fun-1/3) < 1e-8

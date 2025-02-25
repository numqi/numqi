import numpy as np

import numqi

def test_get_generalized_concurrence_pure():
    # http://doi.org/10.1088/1402-4896/acec15 eq(59)
    a1 = 0.577328529773946
    a2 = -0.293019231228375
    a3 = 0.502495600880907
    a4 = -0.284309298545589
    a5 = -0.497466177336762
    a6 = -0.005029423544141
    b1 = 0.247747741831176
    b2 = -0.575500049928480
    b3 = -0.475301477765220
    b4 = 0.327752308097298
    b5 = -0.046190705711075
    b6 = 0.521492183476296
    index_value = [
        ([0,0,0,1], a1), ([0,0,1,0], a2), ([0,0,1,1], a3), ([0,1,0,0], a4), ([0,1,0,1], a5), ([0,1,1,0], a6),
        ([1,0,0,1], b1), ([1,0,1,0], b2), ([1,0,1,1], b3), ([1,1,0,0], b4), ([1,1,0,1], b5), ([1,1,1,0], b6),
        ([2,0,0,1], -b6), ([2,0,1,0], -b5), ([2,0,1,1], b4), ([2,1,0,0], -b3), ([2,1,0,1], b2), ([2,1,1,0], b1),
        ([3,0,0,1], a6), ([3,0,1,0], a5), ([3,0,1,1], -a4), ([3,1,0,0], a3), ([3,1,0,1], -a2), ([3,1,1,0], -a1),
    ]
    psi_list = np.zeros((4,2,2,2), dtype=np.complex128)
    index = np.array([x[0] for x in index_value], dtype=np.int64).T
    psi_list[index[0], index[1], index[2], index[3]] = np.array([x[1] for x in index_value])

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

    tmp0 = psi_list.reshape(4,8)
    assert np.abs(np.linalg.norm(tmp0, axis=1)-1).max() < 1e-10
    assert np.abs(np.linalg.norm(subspace.reshape(4,8) @ tmp0.T.conj(), axis=0)-1).max() < 1e-10
    for x in psi_list:
        assert abs(numqi.entangle.get_generalized_concurrence_pure(x)**2 - 5/6) < 1e-10


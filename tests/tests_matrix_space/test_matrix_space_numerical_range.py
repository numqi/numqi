import numpy as np

import numqi

np_rng = np.random.default_rng()

def test_get_matrix_numerical_range_along_direction():
    dimA = 3
    dimB = 5
    matA = np_rng.normal(size=(dimA*dimB,dimA*dimB))
    matA = matA + matA.T
    matA_pt = matA.reshape(dimA,dimB,dimA,dimB).transpose(0,3,2,1).reshape(dimA*dimB,dimA*dimB)
    alpha_list = np_rng.uniform(0, 2*np.pi, size=100)

    ret = []
    for x in alpha_list:
        tmp0 = matA*np.cos(x) + 1j*matA_pt*np.sin(x)
        ret.append(numqi.matrix_space.get_matrix_numerical_range_along_direction(tmp0, x, kind='max')[0])
        ret.append(numqi.matrix_space.get_matrix_numerical_range_along_direction(tmp0, x, kind='min')[0])
    ret = np.array(ret).reshape(-1, 2)
    # 1e-9 might fail sometimes
    assert np.std(ret, axis=0).max() < 1e-8


def test_get_real_bipartite_numerical_range():
    dimA = 3
    dimB = 5
    matA = np_rng.normal(size=(dimA, dimB,dimA, dimB))
    matA = matA + matA.transpose(2,3,0,1) #for method="eigen", the matrix must be symmetric

    for _ in range(10):
        for kind in ['min','max']:
            ret0 = numqi.matrix_space.get_real_bipartite_numerical_range(matA, kind=kind, method='rotation')
            ret1 = numqi.matrix_space.get_real_bipartite_numerical_range(matA, kind=kind, method='eigen')
            assert abs(ret0-ret1) < 1e-9


def test_detect_real_matrix_subspace_rank_one():
    # https://arxiv.org/abs/2212.12811 example 3
    # span_R(X iY)
    matrix_subspace = np.stack([np.array([[1,0],[0,1]]), np.array([[0,-1],[1,0]])])
    tag_rank_one,upper_bound = numqi.matrix_space.detect_real_matrix_subspace_rank_one(matrix_subspace)
    assert not tag_rank_one
    assert abs(upper_bound-0.5) < 1e-6


def test_get_real_bipartite_numerical_range01():
    # https://arxiv.org/abs/2212.12811 figure 4
    c_list = np.linspace(0, 1, 30)
    w1i_bound_list = []
    for c in c_list:
        hf0 = lambda x: np.array([
                [x[0,0]+c*x[1,1]+x[2,2], -x[0,1], -x[0,2]],
                [-x[1,0], x[0,0]+x[1,1]+c*x[2,2], -x[1,2]],
                [-x[2,0], -x[2,1], c*x[0,0]+x[1,1]+x[2,2]],
            ])
        choi_op = numqi.channel.hf_channel_to_choi_op(hf0, 3)
        w1i_bound_list.append(numqi.matrix_space.get_real_bipartite_numerical_range(choi_op, kind='min', method='eigen'))
    w1i_bound_list = np.array(w1i_bound_list)
    assert w1i_bound_list[c_list<1/4].max() < 0
    assert w1i_bound_list[c_list>1/4].min() > 0

import numpy as np

import numqi


def test_get_bipartition_list():
    for n in [2,3,4,5,6,7]:
        assert len(numqi.matrix_space.get_bipartition_list(n))==(2**(n-1)-1)
    assert numqi.matrix_space.get_bipartition_list(2)==[(0,)]
    assert numqi.matrix_space.get_bipartition_list(3)==[(0,), (1,), (2,)]
    assert numqi.matrix_space.get_bipartition_list(4)==[(0,), (1,), (2,), (3,), (0,1), (0,2), (0,3)]
    assert numqi.matrix_space.get_bipartition_list(5)==[(0,), (1,), (2,), (3,), (4,), (0,1), (0,2), (0,3),
                                      (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]


def test_get_GES_Maciej2019():
    for num_party in [2,3]:
        for dim in [3,4,5]:
            z0 = numqi.matrix_space.get_GES_Maciej2019(dim, num_party=num_party)
            assert z0.shape[0]==(dim-1)**(num_party-1)
            tmp0 = z0.reshape(z0.shape[0], -1)
            assert np.abs(tmp0.conj() @ tmp0.T - np.eye(tmp0.shape[0])).max() < 1e-10


def test_get_geometric_measure_ppt():
    theta_list = np.linspace(0, np.pi, 5, endpoint=False)
    for dimB in [3,4]:
        dim_list = [2, dimB]
        ret_ppt = []
        for theta_i in theta_list:
            matrix_subspace = numqi.matrix_space.get_GES_Maciej2019(dimB, num_party=2, theta=theta_i)
            ret_ppt.append(numqi.matrix_space.get_geometric_measure_ppt(matrix_subspace, dim_list))
        ret_ppt = np.array(ret_ppt)
        ret_analytical = numqi.matrix_space.get_GM_Maciej2019(dimB, theta_list)
        assert np.abs(ret_ppt-ret_analytical).max()<1e-4
    pass

def test_get_generalized_geometric_measure_ppt():
    theta_list = np.linspace(0, np.pi, 5, endpoint=False)
    num_party = 3
    for dimB in [3,4]:
        bipartition_list = [tuple(range(x)) for x in range(1,num_party)]
        dim_list = [2,dimB,dimB]
        ret_ppt = []
        for theta_i in theta_list:
            matrix_subspace = numqi.matrix_space.get_GES_Maciej2019(dimB, num_party=num_party, theta=theta_i)
            ret_ppt.append(numqi.matrix_space.get_generalized_geometric_measure_ppt(matrix_subspace, dim_list, bipartition_list))
        ret_ppt = np.array(ret_ppt)
        ret_analytical = numqi.matrix_space.get_GM_Maciej2019(dimB, theta_list)
        assert np.abs(ret_ppt-ret_analytical).max()<1e-4


def test_get_qubit_dicke_state_gm():
    for num_qubit in [3,5]:
        dim_list = [2]*num_qubit
        dicke_basis = numqi.dicke.get_dicke_basis(num_qubit, dim=2)
        model = numqi.matrix_space.DetectCanonicalPolyadicRankModel(dim_list, rank=1)
        kwargs = dict(theta0='uniform', tol=1e-14, num_repeat=3, print_every_round=0, early_stop_threshold=1e-14)
        ret0 = []
        dicke_k_list = np.arange(num_qubit+1)
        for dicke_k in dicke_k_list:
            model.set_target(dicke_basis[dicke_k].reshape(dim_list))
            ret0.append(numqi.optimize.minimize(model, **kwargs).fun)
        ret0 = np.array(ret0)
        ret_ = numqi.state.get_qubit_dicke_state_GME(num_qubit, dicke_k_list)
        assert np.abs(ret0-ret_).max() < 1e-10

import numpy as np

import numqi

np_rng = np.random.default_rng()

def test_werner_gme():
    alpha_list = np_rng.uniform(0, 1, 10)
    dim = 3

    model = numqi.entangle.DensityMatrixGMEModel([dim,dim], num_ensemble=27, rank=9)
    ret = []
    for alpha_i in alpha_list:
        model.set_density_matrix(numqi.state.Werner(dim, alpha=alpha_i))
        ret.append(numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
    ret = np.array(ret)
    ret_analytical = numqi.state.get_Werner_GME(dim, alpha_list)
    assert np.abs(ret-ret_analytical).max() < 1e-7


def test_isotropic_gme():
    alpha_list = np_rng.uniform(0, 1, 10)
    dim = 3

    model = numqi.entangle.DensityMatrixGMEModel([dim,dim], num_ensemble=27, rank=9)
    ret = []
    for alpha_i in alpha_list:
        model.set_density_matrix(numqi.state.Isotropic(dim, alpha=alpha_i))
        ret.append(numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
    ret = np.array(ret)
    ret_analytical = numqi.state.get_Isotropic_GME(dim, alpha_list)
    assert np.abs(ret-ret_analytical).max() < 1e-7


def test_2qubit_gme():
    model = numqi.entangle.DensityMatrixGMEModel(dim_list=[2,2], num_ensemble=12, rank=4)
    for _ in range(10):
        rho = numqi.random.rand_density_matrix(4)
        model.set_density_matrix(rho)
        ret = numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun
        ret_ = numqi.entangle.get_gme_2qubit(rho)
        assert abs(ret-ret_) < 1e-7

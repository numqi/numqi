import numpy as np

import numqi

np_rng = np.random.default_rng()

def test_werner_gme():
    alpha_list = np_rng.uniform(0, 1, 10)
    dim = 3

    model = numqi.entangle.DensityMatrixGMEModel([dim,dim], num_ensemble=27)
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

    model = numqi.entangle.DensityMatrixGMEModel([dim,dim], num_ensemble=27)
    ret = []
    for alpha_i in alpha_list:
        model.set_density_matrix(numqi.state.Isotropic(dim, alpha=alpha_i))
        ret.append(numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
    ret = np.array(ret)
    ret_analytical = numqi.state.get_Isotropic_GME(dim, alpha_list)
    assert np.abs(ret-ret_analytical).max() < 1e-7


def test_2qubit_gme():
    model = numqi.entangle.DensityMatrixGMEModel(dim_list=[2,2], num_ensemble=12)
    for _ in range(10):
        rho = numqi.random.rand_density_matrix(4)
        model.set_density_matrix(rho)
        ret = numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun
        ret_ = numqi.entangle.get_gme_2qubit(rho)
        assert abs(ret-ret_) < 1e-7


def test_gme_4qubit():
    # https://doi.org/10.1103/PhysRevA.78.060301
    # (0000 + 0011 + 1100 - 1111)/2
    psi_cluster = np.zeros(16, dtype=np.float64)
    psi_cluster[[0, 3, 12, 15]] = np.array([1,1,1,-1])/2
    rho_cluster = psi_cluster.reshape(-1,1) * psi_cluster.conj()

    # (0000 + 1111)/sqrt(2)
    psi_ghz = numqi.state.GHZ(4)
    rho_ghz = psi_ghz.reshape(-1,1) * psi_ghz.conj()

    psi_W = numqi.state.W(4)
    rho_W = psi_W.reshape(-1,1) * psi_W.conj()

    psi_dicke = numqi.state.Dicke(2, 2)
    rho_dicke = psi_dicke.reshape(-1,1) * psi_dicke.conj()

    xlist = np_rng.uniform(0, 1, size=5)
    ret_cluster = (3/8)*(1 + xlist - np.sqrt(1+(2-3*xlist)*xlist))
    ret_ghz = (1/2)*(1-np.sqrt(1-xlist*xlist))
    tmp0 = xlist>(2183/2667)
    ret_W = tmp0 * (37*(81*xlist-37)/2816) + (1-tmp0) * (3/8)*(1+xlist-np.sqrt(1+(2-3*xlist)*xlist))
    tmp0 = (xlist > 5/7)
    ret_dicke = tmp0 * (5*(3*xlist-1)/16) + (1-tmp0) * (5/18)*(1+2*xlist-np.sqrt(1+(4-5*xlist)*xlist))
    ret_ = np.stack([ret_cluster, ret_ghz, ret_W, ret_dicke])

    model = numqi.entangle.DensityMatrixGMEModel(dim_list=[2,2,2,2], num_ensemble=32)
    mask_diag = np.eye(rho_cluster.shape[0], dtype=np.float64)
    mask_offdiag = 1-mask_diag
    ret_model = []
    for rho in [rho_cluster, rho_ghz, rho_W, rho_dicke]:
        for x in xlist:
            model.set_density_matrix(rho*mask_diag + rho*mask_offdiag*x)
            ret_model.append(numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
    ret_model = np.array(ret_model).reshape(4,-1)

    assert np.abs(ret_-ret_model).max() < 1e-7

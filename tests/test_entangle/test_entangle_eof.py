import numpy as np
import scipy.linalg
import torch

import numqi

np_rng = np.random.default_rng()

def test_eof_A_B():
    dimA = 4
    dimB = 3
    q0 = numqi.random.rand_state(dimA*dimB).reshape(dimA, dimB)
    rhoA = np.einsum(q0.conj(), [0,1], q0, [0,2], [1,2], optimize=True)
    rhoB = np.einsum(q0.conj(), [0,1], q0, [2,1], [0,2], optimize=True)
    ret0 = numqi.entangle.get_von_neumann_entropy(rhoA)
    ret1 = numqi.entangle.get_von_neumann_entropy(rhoB)
    assert abs(ret0-ret1) < 1e-10


def test_EntanglementFormationModel_separable():
    num_sample = 5
    for dimA,dimB in [(2,2),(3,4),(4,3)]:
        num_term = 2*dimA*dimB
        for _ in range(num_sample):
            model = numqi.entangle.EntanglementFormationModel(dimA, dimB, num_term)
            dm0 = numqi.random.rand_separable_dm(dimA, dimB, k=dimA*dimB)
            model.set_density_matrix(dm0)
            theta_optim = numqi.optimize.minimize(model, num_repeat=3, print_freq=0, tol=1e-10)
            assert theta_optim.fun < 1e-7

def test_EntanglementFormationModel_isotropic():
    num_sample = 10
    for dim in [2,3]:
        # alpha_list = np.sort(np_rng.uniform(-1/(dim*dim-1), 1, size=num_sample))
        alpha_list = np.linspace(-1/(dim*dim-1), 1, num_sample)
        ret_ = numqi.state.get_Isotropic_eof(dim, alpha_list)

        model = numqi.entangle.EntanglementFormationModel(dim, dim, 2*dim*dim)
        ret0 = []
        kwargs = dict(num_repeat=3, print_freq=0, tol=1e-10, print_every_round=0)
        for alpha_i in alpha_list:
            model.set_density_matrix(numqi.state.Isotropic(dim, alpha_i))
            theta_optim = numqi.optimize.minimize(model, **kwargs)
            ret0.append(theta_optim.fun)
        ret0 = np.array(ret0)
        assert np.abs(ret_-ret0).max() < 1e-7


def test_EntanglementFormationModel_werner():
    num_sample = 10
    for dim in [2,3]:
        alpha_list = np.linspace(-1, 1, num_sample)
        ret_ = numqi.state.get_Werner_eof(dim, alpha_list)

        model = numqi.entangle.EntanglementFormationModel(dim, dim, 2*dim*dim)
        ret0 = []
        kwargs = dict(num_repeat=3, print_freq=0, tol=1e-10, print_every_round=0)
        for alpha_i in alpha_list:
            model.set_density_matrix(numqi.state.Werner(dim, alpha_i))
            theta_optim = numqi.optimize.minimize(model, **kwargs)
            ret0.append(theta_optim.fun)
        ret0 = np.array(ret0)
        assert np.abs(ret_-ret0).max() < 1e-7

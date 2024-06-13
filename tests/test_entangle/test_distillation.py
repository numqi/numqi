import numpy as np
import torch

import numqi

def test_get_PPT_entanglement_cost_bound_werner():
    dim = 3
    # compared with fig1 in https://doi.org/10.1103/PhysRevLett.90.027901
    z0 = numqi.entangle.get_PPT_entanglement_cost_bound(numqi.state.Werner(dim, alpha=1), dim, dim)[0]
    ret_ = 0.7369655941662062
    assert np.abs(z0-ret_) < 1e-10


def test_get_bi_negativity_werner():
    # bi-negativity of Werner states are positive,
    # https://doi.org/10.1103/PhysRevLett.90.027901
    for dim in [3,4,5]:
        alpha_list = np.linspace(-1, 1, 10)
        for alpha_i in alpha_list:
            rho = numqi.state.Werner(dim, alpha_i)
            assert np.linalg.eigvalsh(numqi.entangle.get_binegativity(rho, dim, dim))[0] > -1e-7


def test_get_binegativity():
    # https://oqp.iqoqi.oeaw.ac.at/qubit-bi-negativity
    # two-qubit density matrix's bi-negativity are all positive
    rho = numqi.random.rand_density_matrix(4)
    z0 = numqi.entangle.get_binegativity(rho, 2, 2)
    assert np.abs(z0-z0.T.conj()).max()<1e-10
    assert np.linalg.eigvalsh(z0)[0] > -1e-7

    dimA = 3
    dimB = 4
    rho = numqi.random.rand_density_matrix(dimA*dimB)
    z0 = numqi.entangle.get_binegativity(rho, dimA, dimB)
    z1 = numqi.entangle.get_binegativity(torch.tensor(rho, dtype=torch.complex128), dimA, dimB).numpy()
    assert np.abs(z0-z1).max() < 1e-10

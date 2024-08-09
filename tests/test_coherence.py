import numpy as np
import torch

import numqi

try:
    import mosek
    USE_MOSEK = True
except ImportError:
    USE_MOSEK = False

def test_CoherenceFormationModel_1qubit():
    dim = 2
    model = numqi.coherence.CoherenceFormationModel(dim, num_term=3*dim)
    for _ in range(10):
        rho = numqi.random.rand_density_matrix(2)
        ret_ = numqi.coherence.get_coherence_of_formation_1qubit(rho)
        model.set_density_matrix(rho)
        theta_optim = numqi.optimize.minimize(model, 'uniform', num_repeat=3, tol=1e-10, print_every_round=0)
        assert abs(ret_-theta_optim.fun) < 1e-7


def test_get_coherence_of_formation_1qubit():
    for _ in range(3):
        psi = numqi.random.rand_haar_state(2)
        ret_ = numqi.coherence.get_coherence_of_formation_pure(psi)
        tmp0 = psi.reshape(-1,1) * psi.conj()
        ret0 = numqi.coherence.get_coherence_of_formation_1qubit(tmp0)
        assert abs(ret_-ret0) < 1e-10


def test_GeometricCoherenceModel():
    dim = 2
    model = numqi.coherence.GeometricCoherenceModel(dim, num_term=4*dim, temperature=0.3)
    for _ in range(10):
        rho = numqi.random.rand_density_matrix(2)
        ret_ = numqi.coherence.get_geometric_coherence_1qubit(rho)
        model.set_density_matrix(rho)
        _ = numqi.optimize.minimize(model, 'uniform', num_repeat=3, tol=1e-10, print_every_round=0)
        with torch.no_grad():
            ret0 = model(use_temperature=False).item()
        assert abs(ret_-ret0) < 1e-7


def test_get_geometric_coherence_1qubit():
    for _ in range(3):
        psi = numqi.random.rand_haar_state(2)
        ret_ = numqi.coherence.get_geometric_coherence_pure(psi)
        tmp0 = psi.reshape(-1,1) * psi.conj()
        ret0 = numqi.coherence.get_geometric_coherence_1qubit(tmp0)
        assert abs(ret_-ret0) < 1e-10


def test_get_maximally_coherent_state_mixed_coherence():
    dim = 3
    alpha_list = np.linspace(0, 1, 10)
    model = numqi.coherence.GeometricCoherenceModel(dim, num_term=4*dim, temperature=0.3)
    kwargs = dict(theta0='uniform', num_repeat=3, tol=1e-10, print_every_round=0)
    gmc_list = []
    for alpha_i in alpha_list:
        model.set_density_matrix(numqi.state.maximally_coherent_state(dim, alpha=alpha_i))
        _ = numqi.optimize.minimize(model, **kwargs).fun
        with torch.no_grad():
            gmc_list.append(model(use_temperature=False).item())
    gmc_ = numqi.state.get_maximally_coherent_state_gmc(dim, alpha_list)
    assert np.abs(gmc_list-gmc_).max() < 1e-7


def test_get_geometric_coherence_sdp():
    dim = 3
    alpha_list = np.linspace(0, 0.999, 10) #seem a large error for alpha=1
    gmc_ = numqi.state.get_maximally_coherent_state_gmc(dim, alpha_list)
    tmp0 = np.stack([numqi.state.maximally_coherent_state(dim, alpha=x) for x in alpha_list])
    gmc_list = numqi.coherence.get_geometric_coherence_sdp(tmp0)
    assert np.abs(gmc_-gmc_list).max() < (1e-6 if USE_MOSEK else 1e-4)

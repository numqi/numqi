import numpy as np

import numqi

def test_get_density_matrix_boundary():
    for N0 in [3,4,5]:
        dm0 = numqi.random.rand_density_matrix(N0)
        dm0_norm = numqi.gellmann.dm_to_gellmann_norm(dm0)
        for x0 in [None,dm0_norm]:
            beta_l,beta_u = numqi.entangle.get_density_matrix_boundary(dm0, dm_norm=x0)
            assert beta_l<0
            assert beta_u>0
            assert np.linalg.eigvalsh(numqi.utils.hf_interpolate_dm(dm0, beta=beta_l, dm_norm=x0))[0] > -1e-7
            assert np.linalg.eigvalsh(numqi.utils.hf_interpolate_dm(dm0, beta=beta_u, dm_norm=x0))[0] > -1e-7


def test_entangle_check_swap_witness():
    # if is_seperable state
    for _ in range(100):
        tmp0 = numqi.random.rand_bipartite_state(2, 2, k=1, return_dm=True)
        assert numqi.entangle.check_swap_witness(tmp0)
        tmp0 = numqi.random.rand_separable_dm(dimA=3, dimB=3, k=2)
        assert numqi.entangle.check_swap_witness(tmp0)


def test_isotropic_state():
    np_rng = np.random.default_rng()
    for d in range(2,10):
        alpha = np_rng.uniform(-(1/(d**2-1)), 1)
        rho = numqi.state.Isotropic(d, alpha)
        u0 = numqi.random.rand_haar_unitary(d, seed=np_rng)
        tmp0 = np.kron(u0, u0.conj())
        assert np.abs(tmp0 @ rho @ tmp0.T.conj() - rho).max() < 1e-7
    for d in range(2, 10):
        rho_ent = numqi.state.Isotropic(d, np_rng.uniform(1/(d+1), 1))
        rho_sep = numqi.state.Isotropic(d, np_rng.uniform(-(1/(d**2-1)), 1/(d+1)))
        assert numqi.entangle.is_ppt(rho_sep, (d,d))
        assert not numqi.entangle.is_ppt(rho_ent, (d,d))
        assert numqi.entangle.check_reduction_witness(rho_sep, (d,d))
        assert not numqi.entangle.check_reduction_witness(rho_ent, (d,d))


def test_werner_state():
    np_rng = np.random.default_rng()
    for d in range(2, 10):
        alpha = np_rng.uniform(-1, 1)
        rho = numqi.state.Werner(d, alpha)
        u0 = numqi.random.rand_haar_unitary(d, seed=np_rng)
        tmp0 = np.kron(u0, u0)
        assert np.abs(tmp0 @ rho @ tmp0.T.conj() - rho).max() < 1e-7
    zero_eps = 1e-4 # for numerical stability
    for d in range(2, 10):
        alpha_sep = np_rng.uniform(-1, 1/d-zero_eps)
        alpha_ent = np_rng.uniform(1/d, 1)
        rho_ent = numqi.state.Werner(d, alpha_ent)
        rho_sep = numqi.state.Werner(d, alpha_sep)
        assert numqi.entangle.is_ppt(rho_sep, (d,d))
        assert not numqi.entangle.is_ppt(rho_ent, (d,d))
        assert numqi.entangle.check_reduction_witness(rho_sep, (d,d))
        tmp0 = numqi.entangle.check_reduction_witness(rho_ent, (d,d))
        if d==2:
            assert not tmp0 #only d==2 is correct
        else:
            assert tmp0

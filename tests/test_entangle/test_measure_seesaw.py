import numpy as np
import scipy.linalg
import torch

import numqi

np_rng = np.random.default_rng()
hf_randc = lambda *sz: np_rng.normal(size=sz) + 1j * np_rng.normal(size=sz)
hf_norm = lambda x: x/np.linalg.norm(x.reshape(-1), ord=2)


def test_get_GME_pure_seesaw_2party():
    dimA = 3
    dimB = 4
    num_repeat = 3
    np0 = hf_norm(hf_randc(dimA,dimB))
    kwargs = dict(converge_eps=1e-12, maxiter=3000)
    ret0 = []
    for _ in range(num_repeat):
        psi_list = [hf_norm(hf_randc(x)) for x in [dimA,dimB]]
        ret0.append(numqi.entangle.measure_seesaw._get_GME_pure_seesaw_hf0(np0, psi_list=psi_list, **kwargs))
    ret0,(psiA0,psiB0) = min(ret0, key=lambda x: x[0])
    tmp0 = psiA0.reshape(-1,1) @ psiB0.reshape(1,-1)
    tmp1 = 1-abs(np.vdot(np0.reshape(-1), tmp0))**2
    assert np.abs(ret0-tmp1) < 1e-7

    U,S,V = np.linalg.svd(np0, full_matrices=False)
    psiA = U[:,0]
    psiB = V[0]
    assert abs(1-S[0]**2-ret0) < 1e-7
    assert abs(abs(np.vdot(psiA, psiA0))-1) < 1e-7
    assert abs(abs(np.vdot(psiB, psiB0))-1) < 1e-7


def test_get_GME_pure_seesaw_dicke():
    num_qubit = 4
    klist = list(range(num_qubit+1))

    ret_ = np.array([numqi.state.get_qubit_dicke_state_GME(num_qubit,x) for x in klist])
    np0_list = [numqi.dicke.Dicke(num_qubit-x,x) for x in klist]
    kwargs = dict(converge_eps=1e-10, maxiter=1000, num_repeat=1)
    ret0 = np.array([numqi.entangle.get_GME_pure_seesaw(x.reshape([2]*num_qubit), **kwargs)[0] for x in np0_list])
    assert np.abs(ret_-ret0).max() < 1e-7


def test_seesaw_step1_Uhlmann_theorem():
    N0 = 7
    N1 = 3
    matM = hf_randc(N1, N0)

    matX,matP = scipy.linalg.polar(matM, side='left')
    assert np.abs(matP-matP.T.conj()).max() < 1e-10
    assert np.linalg.eigvalsh(matP)[0] > -1e-7
    assert np.abs(matP @ matX - matM).max() < 1e-10

    hf0 = lambda X: abs(np.trace(matM @ X))**2
    ret_ = hf0(matX.T.conj())
    for _ in range(10):
        assert ret_ >= hf0(numqi.manifold.to_stiefel_polar(np_rng.normal(size=2*N0*N1), dim=N0, rank=N1))

    theta = torch.tensor(numqi.manifold.from_stiefel_polar(matX.T.conj()), dtype=torch.float64, requires_grad=True)
    tmp0 = numqi.manifold.to_stiefel_polar(theta, N0, N1)
    loss = torch.abs(torch.trace(torch.tensor(matM, dtype=torch.complex128) @ tmp0))**2
    loss.backward()
    assert torch.abs(theta.grad).max().item() < 1e-10


def test_seesaw_step3_Cauchy_Schwarz():
    # https://en.wikipedia.org/wiki/Cauchy%E2%80%93Schwarz_inequality
    N0 = 7
    vecS = np_rng.uniform(0, 2, size=N0)

    hf0 = lambda x: np.dot(vecS, x)**2
    theta = vecS / np.linalg.norm(vecS, ord=2)
    ret_ = hf0(theta)
    for _ in range(10):
        tmp0 = np_rng.uniform(0, 1, size=N0)
        assert hf0(tmp0/tmp0.sum()) <= ret_

    x0 = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    loss = torch.dot(torch.tensor(vecS, dtype=torch.float64), x0/torch.linalg.norm(x0, ord=2))**2
    loss.backward()
    assert torch.abs(x0.grad).max().item() < 1e-10


def test_get_GME_subspace_seesaw_vs_GD():
    case_list = [(2,2,2), (2,2,4),(2,2,6),(2,3,4),(2,3,6),(2,3,8),(3,3,6)]
    # ,(3,3,8),(3,4,7),(4,4,7),(4,5,10)
    kwargs_gd = dict(theta0='uniform', num_repeat=5, tol=1e-12, print_every_round=0)
    kwargs_seesaw = dict(converge_eps=1e-12, num_repeat=5, maxiter=2000)
    for dimA,dimB,dimC in case_list:
        np_list = numqi.matrix_space.get_completed_entangled_subspace((dimA, dimB, dimC), kind='quant-ph/0409032')[0]
        model = numqi.matrix_space.DetectCanonicalPolyadicRankModel((dimA, dimB, dimC), rank=1)
        model.set_target(np_list)
        theta_optim = numqi.optimize.minimize(model, **kwargs_gd)
        gme = numqi.entangle.get_GME_subspace_seesaw(np_list, **kwargs_seesaw)[0]
        assert abs(theta_optim.fun-gme) < 1e-7



def test_get_GME_seesaw_random():
    dim = 3
    kwargs = dict(dim_list=(dim,dim), maxiter=2000, converge_eps=1e-12, num_repeat=3, return_info=True)
    for _ in range(3):
        rho = numqi.random.rand_density_matrix(dim*dim)
        ret0,info = numqi.entangle.get_GME_seesaw(rho, **kwargs)

        coeffq = info['coeffq']
        phiA,phiB = info['phi_list']
        sigma = np.einsum(coeffq, [0], phiA, [0,1], phiB, [0,2],
            phiA.conj(), [0,3], phiB.conj(), [0,4], [1,2,3,4], optimize=True).reshape(dim*dim, dim*dim)
        ret1 = 1 - numqi.utils.get_fidelity(rho, sigma)
        assert abs(ret1-ret0) < 1e-4

        # convex roof extension
        matX = info['matX']
        assert np.abs(matX.T @ matX.conj() - np.eye(matX.shape[1])).max() < 1e-10
        EVL,EVC = np.linalg.eigh(rho)
        tmp0 = EVC*np.sqrt(EVL) @ matX.T.conj()
        assert np.abs(tmp0 @ tmp0.T.conj() - rho).max() < 1e-10
        coeffp = np.linalg.norm(tmp0, axis=0, ord=2)
        psi_list = (tmp0 / coeffp).T.reshape(-1, dim, dim)
        gme_psi_i = np.array([(1-np.linalg.svd(x, compute_uv=False)[0]**2) for x in psi_list])
        ret2 = np.dot(coeffp**2, gme_psi_i)
        assert abs(ret2-ret0) < 1e-4

def test_get_GME_seesaw_isotropic():
    dim = 3
    alpha_list = np_rng.uniform(0, 1, 10)
    kwargs = dict(dim_list=(dim,dim), maxiter=2000, converge_eps=1e-10, num_repeat=3)
    for alpha_i in alpha_list:
        rho = numqi.state.Isotropic(dim, alpha_i)
        ret0 = numqi.entangle.get_GME_seesaw(rho, **kwargs)
        ret_ = numqi.state.get_Isotropic_GME(dim, alpha_i)
        assert np.abs(ret_-ret0) < 1e-7

def test_get_GME_seesaw_GHZ():
    tmp0 = numqi.state.GHZ(3)
    zero_eps = 1e-4
    rho_ghz = tmp0.reshape(-1,1)*tmp0.conj()
    kwargs = dict(dim_list=(2,2,2), maxiter=2000, converge_eps=1e-10, num_repeat=3)

    plist = np_rng.uniform(0, 0.2-zero_eps, 5)
    ret_seesaw = np.array([numqi.entangle.get_GME_seesaw(numqi.entangle.hf_interpolate_dm(rho_ghz, alpha=x), **kwargs) for x in plist])
    assert np.abs(ret_seesaw).max() < 1e-7

    plist = np_rng.uniform(0.2+zero_eps, 1, 5)
    ret_seesaw = np.array([numqi.entangle.get_GME_seesaw(numqi.entangle.hf_interpolate_dm(rho_ghz, alpha=x), **kwargs) for x in plist])
    assert np.abs(ret_seesaw).max() > 1e-7

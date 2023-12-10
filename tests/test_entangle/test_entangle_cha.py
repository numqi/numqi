import numpy as np
import torch

import numqi

if torch.get_num_threads()!=1:
    torch.set_num_threads(1)


def test_AutodiffCHAREE_werner():
    alpha_list = np.linspace(0, 1, 5, endpoint=False)
    for dim in [3,4,5]:
        # dim=3, 2 seconds
        # dim=4, 4 seconds
        # dim=5, 10 seconds
        ree_analytical = np.array([numqi.state.get_Werner_ree(dim, x) for x in alpha_list])

        model = numqi.entangle.AutodiffCHAREE(dim, dim, distance_kind='ree')
        # num_repeat=1 should be good for most cases, to avoid unittest failure, use num_repeat=3
        kwargs = dict(theta0='uniform', tol=1e-12, num_repeat=3, print_every_round=0)
        ree_cha = []
        for x in alpha_list:
            model.set_dm_target(numqi.state.Werner(dim, x))
            ree_cha.append(float(numqi.optimize.minimize(model, **kwargs).fun))
        ree_cha = np.array(ree_cha)
        assert np.abs(ree_cha-ree_analytical).max() < 1e-8 #1e-9 fails sometimes


def test_AutodiffCHAREE_boundary():
    rho = numqi.entangle.load_upb('tiles', return_bes=True)[1]
    # rho_norm = numqi.gellmann.dm_to_gellmann_norm(rho)
    model = numqi.entangle.AutodiffCHAREE(3, 3, distance_kind='ree')
    beta = model.get_boundary(rho, xtol=1e-4, threshold=1e-7, converge_tol=1e-10, num_repeat=3, use_tqdm=False)
    assert abs(beta-0.22792) < 5e-4 #fails sometimes for 3e-4
    # beta=0.8649*rho_norm=0.2279211623566359 https://arxiv.org/abs/1705.01523
    # beta=0.2279208149384356

    model = numqi.entangle.AutodiffCHAREE(3, 3, distance_kind='gellmann')
    beta = model.get_boundary(rho, xtol=1e-4, threshold=1e-7, converge_tol=1e-10, num_repeat=3, use_tqdm=False)
    assert abs(beta-0.22792) < 5e-4


def test_convex_hull_approximation_iterative():
    dm0 = numqi.entangle.load_upb('tiles', return_bes=True)[1]
    beta,history_info = numqi.entangle.CHABoundaryBagging((3,3)).solve(dm0, maxiter=150, return_info=True, use_tqdm=False)
    assert abs(beta-0.2279211623566359) < 3e-4
    # beta=0.8649*rho_norm=0.2279211623566359 https://arxiv.org/abs/1705.01523

    ketA,ketB,lambda_,beta_history = history_info
    assert abs(lambda_.sum()-1) < 1e-6
    ret_ = numqi.entangle.hf_interpolate_dm(dm0, beta=beta)
    ret0 = np.einsum(lambda_,[0],ketA,[0,1],ketA.conj(),[0,3],ketB,[0,2],ketB.conj(),[0,4],[1,2,3,4],optimize=True).reshape(dm0.shape)
    assert np.abs(ret_-ret0).max() < 1e-6

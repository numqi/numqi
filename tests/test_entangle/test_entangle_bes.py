import numpy as np
import torch

import numqi


def test_tiles_sixparam_equivalent():
    tmp0 = np.array([1,1,0,1,1,0])*3*np.pi/4
    dm_sixparam = numqi.entangle.load_upb('sixparam', tmp0, return_bes=True)[1]
    dm_tiles = numqi.entangle.load_upb('tiles', return_bes=True)[1]

    model = numqi.entangle.LocalUnitaryEquivalentModel(3, 3, num_term=1)
    model.set_density_matrix(dm_sixparam, dm_tiles)
    # theta_optim = numqi.optimize.minimize(model, num_repeat=3, print_every_round=1, tol=1e-15)
    tmp0 = [0, 0, 0, -0.729011064926899, -1.7599883993660514, -1.7599884089794073, 0, 0,
        -1.4466899565146252, 0.5992386021310881, -0.5992385953110727, 0.15404236098120472,
        0.371891174123775, 0.3718911684969829, 0, 0.6919411325088392]
    numqi.optimize.set_model_flat_parameter(model, np.array(tmp0))
    assert model().item() < 1e-12


def test_pyramid_sixparam_equivalent():
    tmp0 = np.array([1,1,0,1,1,0])*np.arccos((np.sqrt(5)-1)/2)
    dm_sixparam = numqi.entangle.load_upb('sixparam', tmp0, return_bes=True)[1]
    dm_pyramid = numqi.entangle.load_upb('pyramid', return_bes=True)[1]

    model = numqi.entangle.LocalUnitaryEquivalentModel(3, 3, num_term=1)
    model.set_density_matrix(dm_sixparam, dm_pyramid)
    # theta_optim = numqi.optimize.minimize(model, num_repeat=3, print_every_round=1, tol=1e-15)
    tmp0 = [-1.3617766992911065, -0.43429912791906067, 0.26613885602984894, 0.19735746689907954,
        -0.6188289411326084, -1.0098369306252757, 0.693859843451564, 0.8333938247200651,
        1.033595598875421, -0.3056032042357522, 0.7377914028632555, -0.29860275163511957,
        -1.0099191230071312, -0.4183221753228916, 1.033595627906017, -0.7179825078795885]
    numqi.optimize.set_model_flat_parameter(model, np.array(tmp0))
    assert model().item() < 1e-12

    # tiles and pyramid are not LOCC-1 equivalent
    # dm_tiles = numqi.entangle.load_upb('tiles', return_bes=True)[1]
    # dm_pyramid = numqi.entangle.load_upb('pyramid', return_bes=True)[1]
    # model = numqi.entangle.LocalUnitaryEquivalentModel(3, 3, num_term=4)
    # model.set_density_matrix(dm_tiles, dm_pyramid)
    # theta_optim = numqi.optimize.minimize(model, num_repeat=10, print_every_round=1, tol=1e-8)

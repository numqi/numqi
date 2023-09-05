import numpy as np
import torch

import numqi


def test_tiles_sixparam_equivalent():
    tmp0 = np.array([1,1,0,1,1,0])*3*np.pi/4
    dm_sixparam = numqi.entangle.load_upb('sixparam', tmp0, return_bes=True)[1]
    dm_tiles = numqi.entangle.load_upb('tiles', return_bes=True)[1]

    model = numqi.entangle.DensityMatrixLocalUnitaryEquivalentModel(3, 3, num_term=1)
    # model.set_density_matrix(dm_sixparam, dm_tiles)
    # theta_optim = numqi.optimize.minimize(model, num_repeat=3, print_every_round=1, tol=1e-15)
    model.theta0.data[:] = torch.tensor(np.array([-0.025531755705645025, 0, 0,
                2.503770731178937, -0.025531558237369988, 0,
                -2.5037707437499157, 1.0370957995734835, -0.02553154471604701]).reshape(1,3,3), dtype=torch.float64)
    model.theta1.data[:] = torch.tensor(np.array([-1.1626236711113576, 1.1984770956258217, 1.198477417653706,
                -1.7599884404997281, 0.035853501728184745, 2.8933798904340082,
                1.7599883417355267, -0.7290111851923249, 0.03585351949428765]).reshape(1,3,3), dtype=torch.float64)
    model.set_density_matrix(dm_sixparam, dm_tiles)
    ret0 = model().item()
    assert ret0 < 1e-12


def test_pyramid_sixparam_equivalent():
    tmp0 = np.array([1,1,0,1,1,0])*np.arccos((np.sqrt(5)-1)/2)
    dm_sixparam = numqi.entangle.load_upb('sixparam', tmp0, return_bes=True)[1]
    dm_pyramid = numqi.entangle.load_upb('pyramid', return_bes=True)[1]

    model = numqi.entangle.DensityMatrixLocalUnitaryEquivalentModel(3, 3, num_term=1)
    # model.set_density_matrix(dm_sixparam, dm_pyramid)
    # theta_optim = numqi.optimize.minimize(model, num_repeat=3, print_every_round=1, tol=1e-15)
    model.theta0.data[:] = torch.tensor(np.array([-0.3993527741548174, 0, 0,
            -0.408851233847488, -0.39935278448346234, 0,
            -0.653202896453, -1.0659302409655058, -0.39935277138091396]).reshape(1,3,3), dtype=torch.float64)
    model.theta1.data[:] = torch.tensor(np.array([-0.4123748243866279, 0.2938536956157055, 0.3609794586009344,
            -0.6736232425391058, 0.16434548485938258, 1.503587580033367,
            0.5483599701517647, -0.13164963517886363, 0.47587913399762616]).reshape(1,3,3), dtype=torch.float64)
    model.set_density_matrix(dm_sixparam, dm_pyramid)
    ret0 = model().item()
    assert ret0 < 1e-12

    # tiles and pyramid are not LOCC-1 equivalent
    # dm_tiles = numqi.entangle.load_upb('tiles', return_bes=True)[1]
    # dm_pyramid = numqi.entangle.load_upb('pyramid', return_bes=True)[1]
    # model = numqi.entangle.DensityMatrixLocalUnitaryEquivalentModel(3, 3, num_term=4)
    # model.set_density_matrix(dm_tiles, dm_pyramid)
    # theta_optim = numqi.optimize.minimize(model, num_repeat=10, print_every_round=1, tol=1e-8)

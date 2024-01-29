import time
import numpy as np

import numqi

np_rng = np.random.default_rng()


def demo_misc00():
    dimA = 3
    dimB = 3
    dimC = 3
    np_list = numqi.matrix_space.get_completed_entangled_subspace((dimA, dimB, dimC), kind='quant-ph/0409032')[0]

    # mat0 = np_list.reshape(-1,dimA*dimB, dimC)
    # # mat0 = np_list.transpose(0,1,3,2).reshape(-1,dimA*dimC,dimB)
    # z0 = numqi.matrix_space.has_rank_hierarchical_method(mat0, rank=2, hierarchy_k=4)

    # mat0 = np_list.reshape(-1,dimA*dimB, dimC)
    # basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(mat0, field='complex')
    # model = numqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=1)
    # theta_optim = numqi.optimize.minimize(model, 'uniform', num_repeat=3, tol=1e-7)

    # mat0 = np_list.reshape(-1,dimA*dimB, dimC)
    # basis_orth = numqi.matrix_space.get_vector_orthogonal_basis(np_list.reshape(np_list[0], -1)).reshape(-1, dimA,dimB,dimC)
    # basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(mat0, field='complex')
    # basis_orth = basis_orth.reshape(-1, dimA, dimB, dimC)


    print(f'[{dimA}x{dimB}x{dimC}] detect rank=2')
    t0 = time.time()
    model = numqi.matrix_space.DetectCanonicalPolyadicRankModel([dimA,dimB,dimC], rank=2)
    model.set_target(np_list)
    theta_optim = numqi.optimize.minimize(model, 'uniform', num_repeat=3, tol=1e-12)
    print(f'elapsed time: {time.time()-t0:3f}s', '\n')

    print(f'[{dimA}x{dimB}x{dimC}] detect rank=1')
    model = numqi.matrix_space.DetectCanonicalPolyadicRankModel([dimA,dimB,dimC], rank=1)
    model.set_target(np_list)
    theta_optim = numqi.optimize.minimize(model, 'uniform', num_repeat=3, tol=1e-12)
    print(f'elapsed time: {time.time()-t0:3f}s', '\n')


def demo_time_usage():
    case_list = [(2,2,2,2), (2,2,3,2), (2,2,4,2), (2,2,5,2), (2,2,6,2), (2,2,7,2),
                (2,2,8,2), (2,2,9,2), (2,3,3,3), (2,3,4,3), (2,3,5,3), (3,3,3,4)]
    kwargs = dict(theta0='uniform', num_repeat=3, tol=1e-12, print_every_round=0)
    for dimA,dimB,dimC,_ in case_list:
        np_list = numqi.matrix_space.get_completed_entangled_subspace((dimA, dimB, dimC), kind='quant-ph/0409032')[0]
        t0 = time.time()
        model = numqi.matrix_space.DetectCanonicalPolyadicRankModel([dimA,dimB,dimC], rank=2)
        model.set_target(np_list)
        theta_optim2 = numqi.optimize.minimize(model, **kwargs)
        model = numqi.matrix_space.DetectCanonicalPolyadicRankModel([dimA,dimB,dimC], rank=1)
        model.set_target(np_list)
        theta_optim1 = numqi.optimize.minimize(model, **kwargs)
        tmp0 = time.time()-t0
        print(f'[{dimA}x{dimB}x{dimC}][{tmp0:.3f}s] loss(r=2)= {theta_optim2.fun:.4e}, loss(r=1)= {theta_optim1.fun:.4e}')
        # [2x2x2][0.106s] loss(r=2)= 1.0103e-14, loss(r=1)= 2.5000e-01
        # [2x2x3][0.092s] loss(r=2)= 8.8818e-15, loss(r=1)= 9.9242e-02
        # [2x2x4][0.109s] loss(r=2)= 1.8430e-14, loss(r=1)= 4.5039e-02
        # [2x2x5][0.101s] loss(r=2)= 5.4401e-15, loss(r=1)= 2.2597e-02
        # [2x2x6][0.155s] loss(r=2)= 3.6549e-13, loss(r=1)= 1.2303e-02
        # [2x2x7][0.203s] loss(r=2)= 4.9494e-13, loss(r=1)= 7.1707e-03
        # [2x2x8][0.193s] loss(r=2)= 2.4558e-13, loss(r=1)= 4.4234e-03
        # [2x2x9][0.235s] loss(r=2)= 8.7186e-13, loss(r=1)= 2.8611e-03
        # [2x3x3][0.111s] loss(r=2)= 7.5162e-14, loss(r=1)= 3.5569e-02
        # [2x3x4][0.117s] loss(r=2)= 3.1197e-14, loss(r=1)= 1.4135e-02
        # [2x3x5][0.186s] loss(r=2)= 2.3336e-12, loss(r=1)= 6.1101e-03
        # [3x3x3][0.110s] loss(r=2)= 4.3077e-14, loss(r=1)= 1.1905e-02

def demo_completed_entangled_subspace():
    # TODO (3,3), (4,4) bug
    case_list = [(3,4), (2,2,2), (3,3,3), (4,4,4)]

    kwargs = dict(num_repeat=7, tol=1e-14, print_every_round=0, early_stop_threshold=1e-12)
    for dim_tuple in case_list:
        matrix_subspace, matrix_subspace_orth, space_char,_ = numqi.matrix_space.get_completed_entangled_subspace(dim_tuple, kind='quant-ph/0405077')

        model_rank1 = numqi.matrix_space.DetectCanonicalPolyadicRankModel(dim_tuple, rank=1)
        model_rank1.set_target(matrix_subspace)
        theta_optim1 = numqi.optimize.minimize(model_rank1, **kwargs)

        model_rank2 = numqi.matrix_space.DetectCanonicalPolyadicRankModel(dim_tuple, rank=2)
        model_rank2.set_target(matrix_subspace)
        theta_optim2 = numqi.optimize.minimize(model_rank2, **kwargs)

        ind0 = np_rng.permutation(len(matrix_subspace_orth))[0]
        model_rank1.set_target(np.concatenate([matrix_subspace, matrix_subspace_orth[ind0:ind0+1]], axis=0))
        theta_optim3 = numqi.optimize.minimize(model_rank1, **kwargs)
        tmp0 = f'dim_tuple={dim_tuple}, space_char={space_char}, degree={matrix_subspace.shape[0]}/{matrix_subspace_orth.shape[0]}'
        tmp1 = ', '.join([f'{x.fun:.7g}' for x in [theta_optim1,theta_optim2,theta_optim3]])
        print(f'{tmp0}\t loss={tmp1}')
        assert theta_optim1.fun > 1e-7 #mostly should be, but not always
        assert theta_optim2.fun < 1e-10 #should always be
        assert theta_optim3.fun < 1e-10 #should always be
        # dim_tuple=(3, 4), space_char=C, degree=6/6       loss=0.04116747, 3.330669e-16, 1.776357e-15
        # dim_tuple=(2, 2, 2), space_char=C, degree=4/4    loss=0.25, -4.440892e-16, -4.440892e-16
        # dim_tuple=(3, 3, 3), space_char=C, degree=20/7   loss=0.01190476, 6.661338e-16, 8.256729e-13
        # dim_tuple=(4, 4, 4), space_char=C, degree=54/10  loss=0.000433599, 3.441691e-14, 2.265632e-12

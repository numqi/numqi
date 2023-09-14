import numpy as np

import numqi


np_rng = np.random.default_rng()

case_list = [(3,3), (3,4), (4,4), (2,2,2), (3,3,3), (4,4,4)]

kwargs = dict(num_repeat=7, tol=1e-14, print_every_round=0, early_stop_threshold=1e-12)
for dim_tuple in case_list:
    matrix_subspace, matrix_subspace_orth, space_char,_ = numqi.matrix_space.get_completed_entangled_subspace(dim_tuple, kind='quant-ph/0405077')

    model = numqi.matrix_space.DetectCPRankModel(matrix_subspace_orth, rank=1)
    theta_optim1 = numqi.optimize.minimize(model, **kwargs)
    model = numqi.matrix_space.DetectCPRankModel(matrix_subspace_orth, rank=2)
    theta_optim2 = numqi.optimize.minimize(model, **kwargs)
    ind0 = np_rng.permutation(len(matrix_subspace_orth))[:-1]
    model = numqi.matrix_space.DetectCPRankModel(matrix_subspace_orth[ind0], rank=1)
    theta_optim3 = numqi.optimize.minimize(model, **kwargs)
    tmp0 = f'dim_tuple={dim_tuple}, space_char={space_char}, degree={matrix_subspace.shape[0]}/{matrix_subspace_orth.shape[0]}'
    tmp1 = ', '.join([f'{x.fun:.7g}' for x in [theta_optim1,theta_optim2,theta_optim3]])
    print(f'{tmp0}\t loss={tmp1}')
    assert theta_optim1.fun > 1e-7 #mostly should be, but not always
    assert theta_optim2.fun < 1e-10 #should always be
    assert theta_optim3.fun < 1e-10 #should always be
# dim_tuple=(3, 3), space_char=C_T, degree=1/5     loss=0.1818182, 7.681028e-16, 4.317868e-16
# dim_tuple=(3, 4), space_char=C, degree=6/6       loss=0.04116747, 1.696002e-15, 1.980355e-15
# dim_tuple=(4, 4), space_char=C_T, degree=3/7     loss=0.02859548, 8.980346e-14, 7.370078e-16
# dim_tuple=(2, 2, 2), space_char=C, degree=4/4    loss=0.25, 4.130567e-17, 8.353551e-19
# dim_tuple=(3, 3, 3), space_char=C, degree=20/7   loss=0.01190476, 4.785023e-15, 2.827872e-17
# dim_tuple=(4, 4, 4), space_char=C, degree=54/10  loss=0.000433599, 1.45477e-13, 3.335669e-14
## TODO strange, loss minimum value is almost irrelevent to the seed

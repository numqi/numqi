import numpy as np
from tqdm import tqdm

import numqi

np_rng = np.random.default_rng()
hf_randc = lambda *size: np_rng.normal(size=size) + 1j*np_rng.normal(size=size)

def demo_matrix_subspace_XZ_R_XZ_C():
    # XZ_R
    matrix_subspace,field = numqi.matrix_space.get_matrix_subspace_example('XZ_R')
    basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matrix_subspace, field=field)
    model = numqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=(0,1,0))
    theta_optim010 = numqi.optimize.minimize(model, 'normal', num_repeat=3, tol=1e-7)
    model = numqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=(0,2,0))
    theta_optim020 = numqi.optimize.minimize(model, 'normal', num_repeat=3, tol=1e-7)
    model = numqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=(0,1,1))
    theta_optim011 = numqi.optimize.minimize(model, 'normal', num_repeat=3, tol=1e-7)
    matH,coeff,residual = model.get_matrix(theta_optim011.x, matrix_subspace)
    print(f'space={space_char} basis.shape={basis.shape} basis_orth.shape={basis_orth.shape}')
    print(f'loss(010): {theta_optim010.fun}') #1.0
    print(f'loss(020): {theta_optim020.fun}') #1.0
    print(f'loss(011): {theta_optim011.fun}, residual={residual}') #1e-9
    # space=R_T basis.shape=(2, 2, 2) basis_orth.shape=(1, 2, 2)
    # loss(010): 0.9999999999999971
    # loss(020): 1.0000000672122786
    # loss(011): 3.222853017830347e-26, residual=1.6095880932274593e-26

    # XZ_C
    matrix_subspace,field = numqi.matrix_space.get_matrix_subspace_example('XZ_C')
    basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matrix_subspace, field=field)
    model = numqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=1)
    theta_optim1 = numqi.optimize.minimize(model, 'normal', num_repeat=3, tol=1e-7)
    matH,coeff,residual = model.get_matrix(theta_optim1.x, matrix_subspace)
    print(f'space={space_char} basis.shape={basis.shape} basis_orth.shape={basis_orth.shape}')
    print(f'loss(1): {theta_optim1.fun}, residual={residual}') #1e-9
    # space=C_T basis.shape=(2, 2, 2) basis_orth.shape=(1, 2, 2)
    # loss(011): 8.314191254116187e-15, residual=4.1570956204761624e-15


def demo_matrix_space_UDA():
    # UDA
    pauli_str = 'XX XY'
    rank_space,rank_space_orth = numqi.gate.pauli_str_to_matrix(pauli_str, return_orth=True)

    # span_R(C_H)
    basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(rank_space, field='real')

    model = numqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=(0,3,1))
    theta_optim031 = numqi.optimize.minimize(model, 'normal', num_repeat=3, tol=1e-7)
    model = numqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=(0,4,0))
    theta_optim040 = numqi.optimize.minimize(model, 'normal', num_repeat=3, tol=1e-7)
    model = numqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=(0,2,2))
    theta_optim022 = numqi.optimize.minimize(model, 'normal', num_repeat=3, tol=1e-7)
    matH,coeff,residual = model.get_matrix(theta_optim022.x, rank_space)
    print(f'space={space_char} basis.shape={basis.shape} basis_orth.shape={basis_orth.shape}')
    print(f'loss(031): {theta_optim031.fun}') #0.5
    print(f'loss(040): {theta_optim040.fun}') #1.0
    print(f'loss(022): {theta_optim022.fun}, residual={residual}') #1e-9

    # span_C(C_H)
    basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(rank_space, field='complex')

    model = numqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=1)
    theta_optim1 = numqi.optimize.minimize(model, 'normal', num_repeat=3, tol=1e-7)
    model = numqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=2)
    theta_optim2 = numqi.optimize.minimize(model, 'normal', num_repeat=3, tol=1e-7)
    matH,coeff,residual = model.get_matrix(theta_optim2.x, rank_space)
    print(f'space={space_char} basis.shape={basis.shape} basis_orth.shape={basis_orth.shape}')
    print(f'loss(1): {theta_optim1.fun}') #0.5
    print(f'loss(2): {theta_optim2.fun}, residual={residual}') #0



def demo_hierarchical_XZ_over_real():
    pauli_str = 'X Z'
    matA,matB = numqi.gate.pauli_str_to_matrix(pauli_str, return_orth=True)
    for k in range(1, 8):
        tag,z0 = numqi.matrix_space.has_rank_hierarchical_method(matA, rank=2,
                        hierarchy_k=k, use_real_field=True, return_space=True, num_worker=10)
        print(k, tag, z0.shape, np.linalg.eigvalsh(z0@z0.T))
    # 1, False (3, 2) [0. 0. 2.]
    # 2 False (4, 8) [0 0  2.22222222e+00  2.22222222e+00]
    # 3 False (5, 32) [0  0  1.000e+00  4.000e+00  4.222e+00]
    # 4 False (6, 128) [0  0  1.197e+00  1.197e+00  8.083e+00  8.083e+00]
    # 5 False (7, 512) [0  0  8.493e-01  1.778e+00  2.098e+00  1.607e+01  1.608e+01]
    # 6 False (8, 2048) [0 0 9.711e-01 9.711e-01 3.410e+00 3.410e+00 3.207e+01 3.207e+01]
    # 7 False (9, 8192) [0 0 7.976e-01 1.223e+00 1.615e+00 6.122e+00 6.141e+00 6.408e+01 6.408e+01]

    pauli_str = 'XX XY'
    matA,matB = numqi.gate.pauli_str_to_matrix(pauli_str, return_orth=True)
    for k in range(1,7):
        tag,z0 = numqi.matrix_space.has_rank_hierarchical_method(matA, rank=4,
                        hierarchy_k=k, use_real_field=True, return_space=True, num_worker=10)
        print(k, tag, z0.shape, np.linalg.eigvalsh(z0@z0.T))
    # 1 False [-2.681e-16  0.000e+00  1.844e-17  7.976e-17  2.111e+00]
    # 2 False [-1.233e-16 -3.619e-18  6.405e-17  4.757e-16  4.320e+00  4.320e+00]
    # 3 False [-2.211e-15 -1.636e-16 -1.182e-16  1.991e-16  2.098e+00  1.607e+01  1.664e+01]
    # 4 False (8,8192) [-8.531e-15 -2.173e-15  4.899e-16  9.227e-16  4.973e+00  4.973e+00  6.464e+01  6.464e+01]
    # 5 False [-3.004e-14 -5.318e-16  3.281e-15  1.230e-14  5.145e+00  1.633e+01  1.894e+01  2.573e+02  2.574e+02]
    # 6 False [-4.415e-14 -3.072e-14  1.263e-13  1.504e-13  1.174e+01  1.174e+01  6.660e+01  6.660e+01  1.027e+03  1.027e+03]


def demo_hierarchical_method_random_matrix_subspace():
    dimA = 4
    dimB = 4
    rank = 2
    hierarchy_k = 2
    num_matrix = 9 #(da-rank+1)(db-rank+1)
    ret = []
    for _ in tqdm(range(int(100))):
        matrix_space = [hf_randc(dimA,dimB) for _ in range(num_matrix)]
        ret.append(numqi.matrix_space.has_rank(matrix_space, rank, hierarchy_k))

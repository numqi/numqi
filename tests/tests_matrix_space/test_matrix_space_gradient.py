import numpy as np
import torch

import numqi


def test_matrix_subspace_XZ_R_XZ_C():
    # XZ_R
    matrix_subspace,field = numqi.matrix_space.get_matrix_subspace_example('XZ_R')
    basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matrix_subspace, field=field)
    model = numqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=(0,1,1))
    theta_optim011 = numqi.optimize.minimize(model, 'normal', num_repeat=10, tol=1e-10, early_stop_threshold=1e-7)
    matH,coeff,residual = model.get_matrix(matrix_subspace)
    assert theta_optim011.fun < 1e-7
    assert residual < 1e-7

    # XZ_C
    matrix_subspace,field = numqi.matrix_space.get_matrix_subspace_example('XZ_C')
    basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matrix_subspace, field=field)
    model = numqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=1)
    theta_optim1 = numqi.optimize.minimize(model, 'normal', num_repeat=10, tol=1e-10, early_stop_threshold=1e-7)
    matH,coeff,residual = model.get_matrix(matrix_subspace)
    assert theta_optim1.fun < 1e-7
    assert residual < 1e-7


def test_matrix_subspace_real_complex_field_2qubit():
    pauli_str = 'XX XY'.split(' ')
    matrix_subspace = np.stack([numqi.gate.PauliOperator.from_str(x).full_matrix for x in pauli_str])

    # span_R(C_H)
    basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matrix_subspace, field='real')
    model = numqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=(0,2,2))
    theta_optim022 = numqi.optimize.minimize(model, 'normal', num_repeat=10, tol=1e-10, early_stop_threshold=1e-7)
    matH,coeff,residual = model.get_matrix(matrix_subspace)
    assert theta_optim022.fun < 1e-7
    assert residual < 1e-7

    # span_C(C_H)
    basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matrix_subspace, field='complex')
    model = numqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=2)
    theta_optim2 = numqi.optimize.minimize(model, 'normal', num_repeat=10, tol=1e-10, early_stop_threshold=1e-7)
    matH,coeff,residual = model.get_matrix(matrix_subspace)
    assert theta_optim2.fun < 1e-7
    assert residual < 1e-7


def test_matrix_subspace_misc():
    dim = 4
    num_op = 7

    tmp0 = numqi.random.rand_haar_state(dim) #make it definite contains one rank-1 state
    tmp1 = [numqi.random.rand_hermitian_matrix(dim) for _ in range(num_op-1)] + [tmp0.reshape(-1,1)*tmp0.conj()]
    tmp2 = numqi.random.rand_special_orthogonal_matrix(num_op, tag_complex=False)
    matrix_subspace = (tmp2 @ np.stack(tmp1).reshape(num_op,-1)).reshape(-1, dim, dim)

    basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matrix_subspace, field='real')
    model = numqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=(1,0,0))
    theta_optim022 = numqi.optimize.minimize(model, 'normal', num_repeat=20, tol=1e-12, early_stop_threshold=1e-10)
    matH,coeff,residual = model.get_matrix(matrix_subspace)
    assert theta_optim022.fun < 1e-7
    assert residual < 1e-7
    EVL = np.linalg.eigvalsh(matH)
    assert abs(np.linalg.norm(EVL)-1) < 1e-7
    assert abs(np.abs(EVL).max()-1) < 1e-7
    tmp0 = (coeff @ matrix_subspace.reshape(-1,dim*dim)).reshape(dim,dim)
    assert np.abs(tmp0-matH).max() < 2e-7


def test_matrix_subspace_sparse():
    pauli_str = ['XX', 'XY']
    pauli_str_orth = set(numqi.gate.get_pauli_group(2, kind='str')) - set(pauli_str)
    matrix_subspace = np.stack([numqi.gate.PauliOperator.from_str(x).full_matrix for x in pauli_str])
    matrix_subspace_orth = np.stack([numqi.gate.PauliOperator.from_str(x).full_matrix for x in pauli_str_orth])

    model = numqi.matrix_space.DetectRankModel(torch.tensor(matrix_subspace_orth).to_sparse(), space_char='C_H', rank=(0,2,2))
    theta_optim022 = numqi.optimize.minimize(model, 'normal', num_repeat=10, tol=1e-10, early_stop_threshold=1e-7)
    matH,coeff,residual = model.get_matrix(matrix_subspace)
    assert theta_optim022.fun < 1e-7
    assert residual < 1e-7


def test_DetectCanonicalPolyadicRankModel_W_state():
    Wstate = np.zeros((2,2,2), dtype=np.float64)
    Wstate[0,0,1] = 1/np.sqrt(3)
    Wstate[0,1,0] = 1/np.sqrt(3)
    Wstate[1,0,0] = 1/np.sqrt(3)

    model = numqi.matrix_space.DetectCanonicalPolyadicRankModel([2,2,2], 1)
    model.set_target(Wstate)
    kwargs = dict(theta0='uniform', tol=1e-12, num_repeat=3, print_every_round=1, early_stop_threshold=1e-14)
    theta_optim = numqi.optimize.minimize(model, **kwargs)
    assert abs(theta_optim.fun - 5/9) < 1e-10

    # border_rank=2, CP_rank=3
    model = numqi.matrix_space.DetectCanonicalPolyadicRankModel([2,2,2], 2)
    model.set_target(Wstate)
    kwargs = dict(theta0='uniform', tol=1e-12, num_repeat=5, print_every_round=1, early_stop_threshold=1e-14)
    theta_optim = numqi.optimize.minimize(model, **kwargs)
    assert theta_optim.fun < 1e-8


def test_DetectCanonicalPolyadicRankModel_ghz_state():
    ghz = np.zeros((2,2,2))
    ghz[0,0,0] = 1/np.sqrt(2)
    ghz[1,1,1] = 1/np.sqrt(2)
    seed_map = {4:233}
    for num_copy in [2,3,4]:
        ghz_ncopy = ghz.copy()
        for _ in range(num_copy-1):
            tmp0 = np.einsum(ghz_ncopy, [0,1,2], ghz, [3,4,5], [0,3,1,4,2,5], optimize=True)
            tmp1 = ghz_ncopy.shape[0] * ghz.shape[0]
            ghz_ncopy = tmp0.reshape(tmp1,tmp1,tmp1)
        model = numqi.matrix_space.DetectCanonicalPolyadicRankModel(ghz_ncopy.shape, 2**num_copy)
        model.set_target(ghz_ncopy)
        kwargs = dict(theta0='uniform', tol=1e-14, num_repeat=10, print_every_round=0, early_stop_threshold=1e-10)
        theta_optim = numqi.optimize.minimize(model, **kwargs, seed=seed_map.get(num_copy))
        assert theta_optim.fun < 1e-10

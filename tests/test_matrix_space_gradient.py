import numpy as np
import pytest

try:
    import torch
except ImportError:
    torch = None

import numqi


@pytest.mark.skipif(torch is None, reason='pytorch is not installed')
def test_matrix_subspace_XZ_R_XZ_C():
    # XZ_R
    matrix_subspace,field = numqi.matrix_space.get_matrix_subspace_example('XZ_R')
    basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matrix_subspace, field=field)
    model = numqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=(0,1,1))
    theta_optim011 = numqi.optimize.minimize(model, 'normal', num_repeat=10, tol=1e-10, early_stop_threshold=1e-7)
    matH,coeff,residual = model.get_matrix(theta_optim011.x, matrix_subspace)
    assert theta_optim011.fun < 1e-7
    assert residual < 1e-7

    # XZ_C
    matrix_subspace,field = numqi.matrix_space.get_matrix_subspace_example('XZ_C')
    basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matrix_subspace, field=field)
    model = numqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=1)
    theta_optim1 = numqi.optimize.minimize(model, 'normal', num_repeat=10, tol=1e-10, early_stop_threshold=1e-7)
    matH,coeff,residual = model.get_matrix(theta_optim1.x, matrix_subspace)
    assert theta_optim1.fun < 1e-7
    assert residual < 1e-7


@pytest.mark.skipif(torch is None, reason='pytorch is not installed')
def test_matrix_subspace_UDA():
    pauli_str = 'XX XY'
    matrix_subspace,matrix_subspace_orth = numqi.gate.pauli_str_to_matrix(pauli_str, return_orth=True)

    # span_R(C_H)
    basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matrix_subspace, field='real')
    model = numqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=(0,2,2))
    theta_optim022 = numqi.optimize.minimize(model, 'normal', num_repeat=10, tol=1e-10, early_stop_threshold=1e-7)
    matH,coeff,residual = model.get_matrix(theta_optim022.x, matrix_subspace)
    assert theta_optim022.fun < 1e-7
    assert residual < 1e-7

    # span_C(C_H)
    basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matrix_subspace, field='complex')
    model = numqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=2)
    theta_optim2 = numqi.optimize.minimize(model, 'normal', num_repeat=10, tol=1e-10, early_stop_threshold=1e-7)
    matH,coeff,residual = model.get_matrix(theta_optim2.x, matrix_subspace)
    assert theta_optim2.fun < 1e-7
    assert residual < 1e-7


@pytest.mark.skipif(torch is None, reason='pytorch is not installed')
def test_matrix_subspace_misc():
    dim = 4
    num_op = 7

    tmp0 = numqi.random.rand_state(dim) #make it definite contains one rank-1 state
    tmp1 = [numqi.random.rand_hermite_matrix(dim) for _ in range(num_op-1)] + [tmp0.reshape(-1,1)*tmp0.conj()]
    tmp2 = numqi.random.rand_unitary_matrix(num_op, tag_complex=False)
    matrix_subspace = (tmp2 @ np.stack(tmp1).reshape(num_op,-1)).reshape(-1, dim, dim)

    basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matrix_subspace, field='real')
    model = numqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=(1,0,0))
    theta_optim022 = numqi.optimize.minimize(model, 'normal', num_repeat=20, tol=1e-12, early_stop_threshold=1e-10)
    matH,coeff,residual = model.get_matrix(theta_optim022.x, matrix_subspace)
    assert theta_optim022.fun < 1e-7
    assert residual < 1e-7
    EVL = np.linalg.eigvalsh(matH)
    assert abs(np.linalg.norm(EVL)-1) < 1e-7
    assert abs(np.abs(EVL).max()-1) < 1e-7
    tmp0 = (coeff @ matrix_subspace.reshape(-1,dim*dim)).reshape(dim,dim)
    assert np.abs(tmp0-matH).max() < 2e-7


@pytest.mark.skipif(torch is None, reason='pytorch is not installed')
def test_matrix_subspace_sparse():
    pauli_str = 'XX XY'
    matrix_subspace,matrix_subspace_orth = numqi.gate.pauli_str_to_matrix(pauli_str, return_orth=True)

    model = numqi.matrix_space.DetectRankModel(torch.tensor(matrix_subspace_orth).to_sparse(), space_char='C_H', rank=(0,2,2))
    theta_optim022 = numqi.optimize.minimize(model, 'normal', num_repeat=10, tol=1e-10, early_stop_threshold=1e-7)
    matH,coeff,residual = model.get_matrix(theta_optim022.x, matrix_subspace)
    assert theta_optim022.fun < 1e-7
    assert residual < 1e-7

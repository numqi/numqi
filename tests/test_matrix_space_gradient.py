import numpy as np
import pytest

try:
    import torch
    import torch_wrapper
except ImportError:
    torch = None
    torch_wrapper = None

import numpyqi


@pytest.mark.skipif(torch is None, reason='pytorch is not installed')
def test_matrix_subspace_XZ_R_XZ_C():
    # XZ_R
    matrix_subspace,field = numpyqi.matrix_space.get_matrix_subspace_example('XZ_R')
    basis,basis_orth,space_char = numpyqi.matrix_space.get_matrix_orthogonal_basis(matrix_subspace, field=field)
    model = numpyqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=(0,1,1))
    theta_optim011 = torch_wrapper.minimize(model, 'normal', num_repeat=10, tol=1e-10, early_stop_threshold=1e-7)
    matH,coeff,residual = model.get_matrix(theta_optim011.x, matrix_subspace)
    assert theta_optim011.fun < 1e-7
    assert residual < 1e-7

    # XZ_C
    matrix_subspace,field = numpyqi.matrix_space.get_matrix_subspace_example('XZ_C')
    basis,basis_orth,space_char = numpyqi.matrix_space.get_matrix_orthogonal_basis(matrix_subspace, field=field)
    model = numpyqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=1)
    theta_optim1 = torch_wrapper.minimize(model, 'normal', num_repeat=10, tol=1e-10, early_stop_threshold=1e-7)
    matH,coeff,residual = model.get_matrix(theta_optim1.x, matrix_subspace)
    assert theta_optim011.fun < 1e-7
    assert residual < 1e-7


@pytest.mark.skipif(torch is None, reason='pytorch is not installed')
def test_matrix_subspace_UDA():
    pauli_str = 'XX XY'
    rank_space,rank_space_orth = numpyqi.gate.pauli_str_to_matrix(pauli_str, return_orth=True)

    # span_R(C_H)
    basis,basis_orth,space_char = numpyqi.matrix_space.get_matrix_orthogonal_basis(rank_space, field='real')
    model = numpyqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=(0,2,2))
    theta_optim022 = torch_wrapper.minimize(model, 'normal', num_repeat=10, tol=1e-10, early_stop_threshold=1e-7)
    matH,coeff,residual = model.get_matrix(theta_optim022.x, rank_space)
    assert theta_optim022.fun < 1e-7
    assert residual < 1e-7

    # span_C(C_H)
    basis,basis_orth,space_char = numpyqi.matrix_space.get_matrix_orthogonal_basis(rank_space, field='complex')
    model = numpyqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=2)
    theta_optim2 = torch_wrapper.minimize(model, 'normal', num_repeat=10, tol=1e-10, early_stop_threshold=1e-7)
    matH,coeff,residual = model.get_matrix(theta_optim2.x, rank_space)
    assert theta_optim2.fun < 1e-7
    assert residual < 1e-7

import numpy as np
import torch

import numqi


def test_VarQECUnitary_623_min_max():
    num_qubit = 6
    dimK = 2
    distance = 3
    pauli_str, error_torch = numqi.qec.make_pauli_error_list_sparse(num_qubit, distance, kind='torch-csr01')
    model = numqi.qec.VarQECUnitary(num_qubit, dimK, error_torch)
    kwargs_constraint = dict(theta0='uniform', num_repeat=10, tol=1e-14,
                        constraint_penalty=10, constraint_p=1.4, constraint_threshold=1e-12)

    model.set_lambda_target('min')
    theta_optim = numqi.optimize.minimize(model, **kwargs_constraint)
    assert theta_optim is not None, 'fail to satisfy constraint'
    assert abs(theta_optim.fun-0.6) < 1e-7

    model.set_lambda_target('max')
    theta_optim = numqi.optimize.minimize(model, **kwargs_constraint)
    assert theta_optim is not None, 'fail to satisfy constraint'
    assert abs(theta_optim.fun+1) < 1e-7

    model.set_lambda_target(0.6)
    kwargs = dict(theta0='uniform', num_repeat=10, tol=1e-20, early_stop_threshold=1e-18)
    theta_optim = numqi.optimize.minimize(model, **kwargs)
    assert abs(theta_optim.fun) < 1e-10

    # num_qubit = 7
    # dimK = 2
    # distance = 3
    # pauli_str, error_torch = numqi.qec.make_pauli_error_list_sparse(num_qubit, distance, kind='torch-csr01')
    # model = numqi.qec.VarQECUnitary(num_qubit, dimK, error_torch)
    # kwargs_constraint = dict(theta0='uniform', num_repeat=10, tol=1e-14,
    #                     constraint_penalty=10, constraint_p=1.4, constraint_threshold=1e-12)
    # # hard to converge for lambda='min'
    # model.set_lambda_target('max')
    # theta_optim = numqi.optimize.minimize(model, **kwargs_constraint)


# theta_optim = numqi.optimize.minimize(model, 'uniform', num_repeat=20, tol=1e-14, early_stop_threshold=1e-13)

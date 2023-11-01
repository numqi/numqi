import functools
import numpy as np
import torch

import numqi

np_rng = np.random.default_rng()
hf_kron = lambda *x: functools.reduce(np.kron, x)

def test_sdp_2local_rdm_solve():
    num_qubit = 3
    rho_target = numqi.random.rand_density_matrix(2**num_qubit, seed=np_rng)
    op_list = numqi.maximum_entropy.get_1dchain_2local_pauli_basis(num_qubit, with_I=False)
    term_value_target = np.trace(op_list @ rho_target, axis1=1, axis2=2).real
    z0 = numqi.maximum_entropy.sdp_2local_rdm_solve(term_value_target)
    assert np.abs(np.trace(op_list @ z0, axis1=1, axis2=2).real-term_value_target).max() < 1e-6


def test_sdp_op_list_solve():
    num_op = 10
    num_qubit = 3
    np_rng = np.random.default_rng(seed=233)

    rho = numqi.random.rand_density_matrix(2**num_qubit, seed=np_rng)
    op_list = np.stack([numqi.random.rand_hermitian_matrix(2**num_qubit, seed=np_rng) for _ in range(num_op)])
    term_value = np.trace(op_list @ rho, axis1=1, axis2=2).real

    z0 = numqi.maximum_entropy.sdp_op_list_solve(op_list, term_value)
    assert np.abs(np.trace(op_list @ z0, axis1=1, axis2=2) - term_value).max() < 1e-6


def test_maximum_entropy_model_2qubit():
    op_list = [
        hf_kron(numqi.gate.X, numqi.gate.X),
        hf_kron(numqi.gate.Z, numqi.gate.I),
    ]
    model = numqi.maximum_entropy.MaximumEntropyModel(op_list)

    for _ in range(3):
        tmp0 = np_rng.uniform(0, 0.9)
        tmp1 = np_rng.uniform(0, 2*np.pi)
        term_value = np.array([tmp0*np.cos(tmp1), tmp0*np.sin(tmp1)])
        model.set_target(term_value)
        theta_optim0 = numqi.optimize.minimize(model, theta0='uniform', num_repeat=10, tol=1e-10, print_every_round=0, print_freq=0)
        # always should converge to 0
        assert theta_optim0.fun < 1e-7

        tmp0 = np_rng.uniform(1.1, 2)
        tmp1 = np_rng.uniform(0, 2*np.pi)
        term_value = np.array([tmp0*np.cos(tmp1), tmp0*np.sin(tmp1)])
        model.set_target(term_value)
        theta_optim0 = numqi.optimize.minimize(model, theta0='uniform', num_repeat=10, tol=1e-10, print_every_round=0, print_freq=0)
        assert theta_optim0.fun > 0.01


def test_get_ABk_gellmann_preimage_op():
    for dimA,dimB,kext in [(2,2,3),(3,2,3),(3,3,3)]:
        N0 = dimA*dimB*dimA*dimB-1
        ret0 = numqi.maximum_entropy.get_ABk_gellmann_preimage_op(dimA, dimB, kext, kind='boson')
        N1 = numqi.dicke.get_dicke_number(kext, dimB)*dimA
        assert ret0.shape==(N0,N1,N1)
        assert np.abs(ret0-ret0.transpose(0,2,1).conj()).max() < 1e-10
        assert np.abs(np.trace(ret0, axis1=1, axis2=2)).max() < 1e-10

        ret0 = numqi.maximum_entropy.get_ABk_gellmann_preimage_op(dimA, dimB, kext, kind='symmetric')
        N1 = dimA*(dimB**kext)
        assert ret0.shape==(N0,N1,N1)
        assert np.abs(ret0-ret0.transpose(0,2,1).conj()).max() < 1e-10
        assert np.abs(np.trace(ret0, axis1=1, axis2=2)).max() < 1e-10


def test_eigvalsh_largest_power_iteration():
    np_rng = np.random.default_rng()
    # openblas-macos seems to be really slow on the follow test
    for dim in [2,4,8,16,32,64,128]:#,256,512,1024]:
        tmp0 = np_rng.normal(size=(dim,dim))+1j*np_rng.normal(size=(dim,dim))
        np0 = tmp0+tmp0.T.conj()
        ret_ = np.linalg.eigvalsh(np0).max()
        torch0 = torch.tensor(np0, dtype=torch.complex128)
        EVC,num_step = numqi.maximum_entropy.eigvalsh_largest_power_iteration(torch0, maxiter=1000, tol=1e-7)
        EVC = EVC.detach().cpu().numpy()
        error = abs(((np0 @ EVC) @ EVC.conj()).real - ret_)
        assert error < 1e-10
        print(dim, num_step, error)

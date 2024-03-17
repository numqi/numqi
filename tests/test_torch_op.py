import numpy as np
import scipy.linalg
import torch

import numqi

np_rng = np.random.default_rng()


def test_PSDMatrixSqrtm():
    N0 = 4
    np0 = np_rng.normal(size=(N0,N0)) + 1j*np_rng.normal(size=(N0,N0))
    np1 = np_rng.normal(size=(N0,N0)) + 1j*np_rng.normal(size=(N0,N0))
    torch0_r = torch.tensor(np0.real, dtype=torch.float64, requires_grad=True)
    torch0_i = torch.tensor(np0.imag, dtype=torch.float64, requires_grad=True)
    tmp0 = torch.complex(torch0_r,torch0_i)
    torch1 = numqi._torch_op.PSDMatrixSqrtm.apply(tmp0 @ tmp0.T.conj())
    loss = (torch1*torch.tensor(np1, dtype=torch.complex128)).real.sum()
    loss.backward()
    ret0 = torch0_r.grad.numpy() + 1j*torch0_i.grad.numpy()

    def hf0(x):
        tmp0 = x @ x.T.conj()
        EVL,EVC = np.linalg.eigh(tmp0)
        sqrt_x = (EVC * np.sqrt(np.maximum(0,EVL))) @ EVC.T.conj()
        ret = (sqrt_x*np1).real.sum()
        return ret
    # hf0 = lambda x: (scipy.linalg.sqrtm(x @ x.T.conj())*np1).real.sum()
    ret_ = numqi.optimize.finite_difference_central(hf0, np0, zero_eps=1e-4)
    assert np.abs(ret_-ret0).max() < 1e-6


def test_PSDMatrixLogm():
    N0 = 4
    np0 = np_rng.normal(size=(N0,N0)) + 1j*np_rng.normal(size=(N0,N0))
    np1 = np_rng.normal(size=(N0,N0)) + 1j*np_rng.normal(size=(N0,N0))
    torch0_r = torch.tensor(np0.real, dtype=torch.float64, requires_grad=True)
    torch0_i = torch.tensor(np0.imag, dtype=torch.float64, requires_grad=True)
    tmp0 = torch.complex(torch0_r,torch0_i)
    op_torch_logm = numqi._torch_op.PSDMatrixLogm(num_sqrtm=6, pade_order=8)
    torch1 = op_torch_logm(tmp0 @ tmp0.T.conj())
    loss = (torch1*torch.tensor(np1, dtype=torch.complex128)).real.sum()
    loss.backward()
    ret0 = torch0_r.grad.numpy() + 1j*torch0_i.grad.numpy()

    hf0 = lambda x: (scipy.linalg.logm(x @ x.T.conj())*np1).real.sum()
    ret_ = numqi.optimize.finite_difference_central(hf0, np0, zero_eps=1e-4)
    assert np.abs(ret_-ret0).max() < 1e-6

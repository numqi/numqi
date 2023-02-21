import numpy as np
import pytest

try:
    import torch
except ImportError:
    torch = None

import numpyqi

np_rng = np.random.default_rng()

@pytest.mark.skipif(torch==None, reason="requires torch")
def test_real_matrix_to_trace1_PSD():
    np_rng = np.random.default_rng()
    N0 = 5
    zero_eps = 1e-4
    matA = np_rng.normal(size=(N0,N0))
    grad_coeff = np_rng.normal(size=(N0,N0)) + 1j*np_rng.normal(size=(N0,N0))

    for use_cholesky in [True, False]:
        torch0 = torch.tensor(matA, requires_grad=True, dtype=torch.float64)
        tmp0 = numpyqi.param.real_matrix_to_trace1_PSD(torch0, use_cholesky=use_cholesky)
        loss = (tmp0 * torch.tensor(grad_coeff, dtype=torch.complex128)).sum().real
        loss.backward()
        grad = torch0.grad.detach().numpy().copy()

        hf0 = lambda x: (numpyqi.param.real_matrix_to_trace1_PSD(x, use_cholesky=use_cholesky)*grad_coeff).sum().real
        loss_ = hf0(matA)
        grad_ = np.zeros((N0,N0), dtype=np.float64)
        tmp0 = ((x,y) for x in range(N0) for y in range(N0))
        for ind0 in range(N0):
            for ind1 in range(N0):
                tmp0,tmp1 = [matA.copy() for _ in range(2)]
                tmp0[ind0,ind1] += zero_eps
                tmp1[ind0,ind1] -= zero_eps
                grad_[ind0,ind1] = (hf0(tmp0)-hf0(tmp1))/(2*zero_eps)
        assert np.abs(loss.item()-loss_).max() < 1e-10
        assert np.abs(grad-grad_).max() < 1e-6


def test_real_matrix_to_special_unitary():
    N0 = 23
    for N1 in range(2,7):
        np0 = np_rng.uniform(-1, 1, size=(N0,N1,N1))

        #SO(n)
        np1 = numpyqi.param.real_matrix_to_special_unitary(np0, tag_real=True)
        assert np.abs((np1 @ np1.transpose(0,2,1)) - np.eye(N1)).max() < 1e-10
        assert np.abs(np.linalg.det(np1)-1).max() < 1e-10

        #SU(n)
        np1 = numpyqi.param.real_matrix_to_special_unitary(np0, tag_real=False)
        assert np.abs((np1 @ np1.transpose(0,2,1).conj()) - np.eye(N1)).max() < 1e-10
        assert np.abs(np.linalg.det(np1)-1).max() < 1e-10

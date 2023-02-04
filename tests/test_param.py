import numpy as np
import pytest

try:
    import torch
except ImportError:
    torch = None

import numpyqi

np_rng = np.random.default_rng()

@pytest.mark.skipif(torch==None, reason="requires torch")
def test_hermitian_matrix_to_PSD_shift_max_eig():
    np_rng = np.random.default_rng()
    dim = 5
    tmp0 = np_rng.uniform(-1, 1, size=(dim,dim)) + 1j*np_rng.uniform(-1, 1, size=(dim,dim))
    matA = (tmp0 + tmp0.T.conj())/2
    z0 = numpyqi.param.hermitian_matrix_to_PSD(matA, shift_max_eig=True)

    tmp0 = torch.tensor(matA, dtype=torch.complex128, requires_grad=True)
    z1 = numpyqi.param.hermitian_matrix_to_PSD(tmp0, shift_max_eig=True)
    torch.abs(z1).sum().backward()
    z1_grad = tmp0.grad.detach().numpy().copy()

    tmp0 = torch.tensor(matA, dtype=torch.complex128, requires_grad=True)
    tmp1 = numpyqi.param.hermitian_matrix_to_PSD(tmp0, shift_max_eig=False)
    z2 = tmp1 / torch.trace(tmp1)
    torch.abs(z2).sum().backward()
    z2_grad = tmp0.grad.detach().numpy().copy()
    assert np.abs(z1.detach().numpy()-z0).max() < 1e-10
    assert np.abs(z1_grad-z2_grad).max() < 1e-10


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

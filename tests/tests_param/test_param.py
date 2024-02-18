import numpy as np
import torch

import numqi

np_rng = np.random.default_rng()

def test_real_matrix_to_trace1_PSD():
    np_rng = np.random.default_rng()
    N0 = 5
    zero_eps = 1e-4
    matA = np_rng.normal(size=(N0,N0))
    grad_coeff = np_rng.normal(size=(N0,N0)) + 1j*np_rng.normal(size=(N0,N0))

    for use_cholesky in [True, False]:
        torch0 = torch.tensor(matA, requires_grad=True, dtype=torch.float64)
        tmp0 = numqi.param.real_matrix_to_trace1_PSD(torch0, use_cholesky=use_cholesky)
        loss = (tmp0 * torch.tensor(grad_coeff, dtype=torch.complex128)).sum().real
        loss.backward()
        grad = torch0.grad.detach().numpy().copy()

        hf0 = lambda x: (numqi.param.real_matrix_to_trace1_PSD(x, use_cholesky=use_cholesky)*grad_coeff).sum().real
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
        np1 = numqi.param.real_matrix_to_special_unitary(np0, tag_real=True)
        assert np.abs((np1 @ np1.transpose(0,2,1)) - np.eye(N1)).max() < 1e-10
        assert np.abs(np.linalg.det(np1)-1).max() < 1e-10

        #SU(n)
        np1 = numqi.param.real_matrix_to_special_unitary(np0, tag_real=False)
        assert np.abs((np1 @ np1.transpose(0,2,1).conj()) - np.eye(N1)).max() < 1e-10
        assert np.abs(np.linalg.det(np1)-1).max() < 1e-10


def test_matrix_to_kraus_op():
    dim_in = 5
    dim_out = 3
    rank = 3
    size = rank,dim_out,dim_in

    # float64
    np0 = np_rng.uniform(-1,1,size=size)
    np1 = numqi.param.matrix_to_kraus_op(np0)
    assert np1.dtype.type==np0.dtype.type
    tmp0 = np.einsum(np1, [0,1,2], np1.conj(), [0,1,3], [2,3], optimize=True)
    assert np.abs(tmp0-np.eye(dim_in)).max() < 1e-10
    torch0 = torch.tensor(np0, dtype=torch.float64)
    torch1 = numqi.param.matrix_to_kraus_op(torch0)
    assert torch1.dtype==torch.float64
    tmp0 = torch.einsum(torch1, [0,1,2], torch1.conj(), [0,1,3], [2,3]).numpy()
    assert np.abs(tmp0-np.eye(dim_in)).max() < 1e-10

    # complex128
    np0 = np_rng.uniform(-1,1,size=size) + 1j*np_rng.uniform(-1,1,size=size)
    np1 = numqi.param.matrix_to_kraus_op(np0)
    assert np1.dtype.type==np0.dtype.type
    tmp0 = np.einsum(np1, [0,1,2], np1.conj(), [0,1,3], [2,3], optimize=True)
    assert np.abs(tmp0-np.eye(dim_in)).max() < 1e-10
    torch0 = torch.tensor(np0, dtype=torch.complex128)
    torch1 = numqi.param.matrix_to_kraus_op(torch0)
    assert torch1.dtype==torch.complex128
    tmp0 = torch.einsum(torch1, [0,1,2], torch1.conj(), [0,1,3], [2,3]).numpy()
    assert np.abs(tmp0-np.eye(dim_in)).max() < 1e-10


def test_matrix_to_stiefel():
    N0 = 4
    N1 = 5
    N2 = 3
    size = N0,N1,N2

    # float64
    np0 = np_rng.uniform(-1,1,size=size)
    np1 = numqi.param.matrix_to_stiefel(np0)
    assert np1.dtype.type==np0.dtype.type
    tmp0 = np.einsum(np1, [0,1,2], np1.conj(), [0,1,3], [0,2,3], optimize=True)
    assert np.abs(tmp0-np.eye(N2)).max() < 1e-10
    torch0 = torch.tensor(np0, dtype=torch.float64)
    torch1 = numqi.param.matrix_to_stiefel(torch0)
    assert torch1.dtype==torch.float64
    tmp0 = torch.einsum(torch1, [0,1,2], torch1.conj(), [0,1,3], [0,2,3]).numpy()
    assert np.abs(tmp0-np.eye(N2)).max() < 1e-10

    # complex128
    np0 = np_rng.uniform(-1,1,size=size) + 1j*np_rng.uniform(-1,1,size=size)
    np1 = numqi.param.matrix_to_stiefel(np0)
    assert np1.dtype.type==np0.dtype.type
    tmp0 = np.einsum(np1, [0,1,2], np1.conj(), [0,1,3], [0,2,3], optimize=True)
    assert np.abs(tmp0-np.eye(N2)).max() < 1e-10
    torch0 = torch.tensor(np0, dtype=torch.complex128)
    torch1 = numqi.param.matrix_to_stiefel(torch0)
    assert torch1.dtype==torch.complex128
    tmp0 = torch.einsum(torch1, [0,1,2], torch1.conj(), [0,1,3], [0,2,3]).numpy()
    assert np.abs(tmp0-np.eye(N2)).max() < 1e-10


def test_matrix_to_choi_op():
    dim_in = 5
    dim_out = 3
    rank = 4

    tmp0 = np_rng.uniform(-1,1,size=(2,rank,dim_out,dim_in))
    np0 = tmp0[0] + 1j*tmp0[1]
    choi_op = numqi.param.matrix_to_choi_op(np0)
    assert np.abs(choi_op-choi_op.transpose(2,3,0,1).conj()).max() < 1e-10
    assert np.linalg.eigvalsh(choi_op.reshape(dim_out*dim_in,-1)).min() >= -1e-10
    assert np.abs(np.trace(choi_op, axis1=0, axis2=2) - np.eye(dim_in)).max() < 1e-10

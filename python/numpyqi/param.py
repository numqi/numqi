import numpy as np
import scipy.sparse.linalg

try:
    import torch
    from ._torch_op import TorchPSDMatrixSqrtm
except ImportError:
    torch = None
    TorchPSDMatrixSqrtm = None

from .utils import is_torch


def _real_matrix_to_PSD_cholesky(matA, shift_max_eig, tag_real):
    shape = matA.shape
    matA = matA.reshape(-1, shape[-1], shape[-1])
    N0 = len(matA)
    if is_torch(matA):
        if tag_real:
            tmp0 = torch.tril(matA)
        else:
            tmp0 = torch.tril(matA) + 1j*(torch.triu(matA, 1).transpose(1,2))
        ret = tmp0 @ tmp0.transpose(1,2).conj()
        if shift_max_eig:
            ret = ret / torch.linalg.norm(matA, dim=(1,2), keepdims=True)**2
    else:
        if tag_real:
            tmp0 = np.tril(matA)
        else:
            tmp0 = np.tril(matA) + 1j*(np.triu(matA, 1).transpose(0,2,1))
        ret = tmp0 @ tmp0.transpose(0,2,1).conj()
        if shift_max_eig:
            ret = ret / np.linalg.norm(matA, axis=(1,2), keepdims=True)**2
    ret = ret.reshape(*shape)
    return ret


def real_matrix_to_PSD(matA, shift_max_eig=True, tag_real=False, use_cholesky=False):
    if use_cholesky:
        ret = _real_matrix_to_PSD_cholesky(matA, shift_max_eig, tag_real)
    shape = matA.shape
    matA = matA.reshape(-1, shape[-1], shape[-1])
    N0 = len(matA)
    if is_torch(matA):
        tmp1 = torch.triu(matA)
        if tag_real:
            tmp2 = (tmp1 + tmp1.transpose(1,2))
        else:
            tmp0 = torch.tril(matA, -1)
            tmp2 = 1j*(tmp0 - tmp0.transpose(1,2)) + (tmp1 + tmp1.transpose(1,2))
        if shift_max_eig:
            tmp3 = tmp2.detach().cpu().numpy()
            EVL = [scipy.sparse.linalg.eigsh(tmp3[x], k=1, which='LA', return_eigenvectors=False)[0] for x in range(N0)]
            eye_mat = torch.eye(tmp2.shape[1], device=matA.device)
            ret = torch.stack([torch.linalg.matrix_exp(tmp2[x]-EVL[x]*eye_mat) for x in range(N0)])
    else:
        tmp1 = np.triu(matA)
        if tag_real:
            tmp2 = (tmp1 + tmp1.transpose(0,2,1))
        else:
            tmp0 = np.tril(matA, -1)
            tmp2 = 1j*(tmp0 - tmp0.transpose(0,2,1)) + (tmp1 + tmp1.transpose(0,2,1))
        if shift_max_eig:
            EVL = [scipy.sparse.linalg.eigsh(tmp2[x], k=1, which='LA', return_eigenvectors=False)[0] for x in range(N0)]
            eye_mat = np.eye(tmp2.shape[1])
            ret = np.stack([scipy.linalg.expm(tmp2[x]-EVL[x]*eye_mat) for x in range(N0)])
        else:
            ret = np.stack([scipy.linalg.expm(x) for x in tmp2])
    ret = ret.reshape(*shape)
    return ret


def real_matrix_to_choi_op(matA, dim_in, use_cholesky=False):
    assert matA.ndim==2 #TODO support batch
    shape = matA.shape
    matA = matA.reshape(-1, shape[-1], shape[-1])
    assert shape[-1]%dim_in==0
    dim_out = shape[-1]//dim_in
    if is_torch(matA):
        tmp0 = torch.tril(matA, -1)
        tmp1 = torch.triu(matA)
        tmp2 = 1j*(tmp0 - tmp0.transpose(1,2)) + (tmp1 + tmp1.transpose(1,2))
        # tmp3 = [torch.lobpcg(hf_complex_to_real(x.detach()),k=1)[0] for x in tmp2]
        tmp3 = [scipy.sparse.linalg.eigsh(tmp2[0].detach().cpu().numpy(), k=1, which='LA', return_eigenvectors=False)[0]]
        # torch.lobpcg(tmp2, k=1)
        # tmp3 = torch.max(torch.diagonal(tmp2.real, dim1=1, dim2=2), dim=1)[0]
        tmp4 = torch.eye(tmp2.shape[1], device=matA.device)
        mat0 = torch.stack([torch.linalg.matrix_exp(tmp2[x]-tmp3[x]*tmp4) for x in range(len(tmp3))])
        # mat0 = torch.stack([torch.linalg.matrix_exp(x) for x in tmp2])
        tmp1 = torch.einsum(mat0.reshape(dim_in, dim_out, dim_in, dim_out), [0,1,2,1], [0,2])
        if use_cholesky:
            tmp2 = torch.linalg.inv(torch.linalg.cholesky_ex(tmp1, upper=True)[0])
        else:
            tmp2 = torch.linalg.inv(TorchPSDMatrixSqrtm.apply(tmp1))
        ret = torch.einsum(mat0.reshape(dim_in,dim_out,dim_in,dim_out), [0,1,2,3],
                tmp2.conj(), [0,4], tmp2, [2,5], [4,1,5,3]).reshape(dim_in*dim_out,-1)
    else:
        tmp0 = np.tril(matA, -1)
        tmp1 = np.triu(matA)
        tmp2 = 1j*(tmp0 - tmp0.transpose(0,2,1)) + (tmp1 + tmp1.transpose(0,2,1))

        mat0 = np.stack([scipy.linalg.expm(x) for x in tmp2])
        # mat0 = scipy.linalg.expm(tmp2) #TODO scipy-v1.9
        tmp1 = np.einsum(mat0.reshape(dim_in, dim_out, dim_in, dim_out), [0,1,2,1], [0,2], optimize=True)
        if use_cholesky:
            tmp2 = np.linalg.inv(scipy.linalg.cholesky(tmp1))
        else:
            tmp2 = np.linalg.inv(scipy.linalg.sqrtm(tmp1))
        ret = np.einsum(mat0.reshape(dim_in,dim_out,dim_in,dim_out), [0,1,2,3],
                tmp2.conj(), [0,4], tmp2, [2,5], [4,1,5,3], optimize=True).reshape(dim_in*dim_out,-1)
    ret = ret.reshape(*shape)
    return ret


def real_matrix_to_orthogonal(matA):
    # TODO merge into real_matrix_to_unitary
    shape = matA.shape
    matA = matA.reshape(-1, shape[-1], shape[-1])
    if isinstance(matA, torch.Tensor):
        tmp0 = torch.triu(matA, 1)
        tmp1 = tmp0 - tmp0.transpose(1,2)
        ret = torch.stack([torch.linalg.matrix_exp(x) for x in tmp1])
    else:
        tmp0 = np.triu(matA, 1)
        tmp1 = tmp0 - tmp0.transpose(0,2,1)
        # ret = np.stack([scipy.linalg.expm(x) for x in tmp1])
        ret = scipy.linalg.expm(tmp1) #TODO scipy-v1.9
    ret = ret.reshape(*shape)
    return ret


def real_matrix_to_unitary(matA, with_phase=False, tag_real=False):
    if tag_real: #Special Orthogonal
        ret = real_matrix_to_choi_op(matA)
        return ret
    shape = matA.shape
    matA = matA.reshape(-1, shape[-1], shape[-1])
    if is_torch(matA):
        tmp0 = torch.tril(matA, -1)
        tmp1 = torch.triu(matA)
        if with_phase:
            tmp3 = tmp1
        else:
            tmp2 = torch.diagonal(tmp1, dim1=-2, dim2=-1).sum(dim=1).reshape(-1,1,1)/shape[-1]
            tmp3 = tmp1 - tmp2*torch.eye(shape[-1], device=matA.device)
        tmp4 = 1j*(tmp0 - tmp0.transpose(1,2)) + (tmp3 + tmp3.transpose(1,2))
        ret = torch.linalg.matrix_exp(1j*tmp4)
    else:
        tmp0 = np.tril(matA, -1)
        tmp1 = np.triu(matA)
        if not with_phase:
            tmp1 = tmp1 - np.trace(tmp1, axis1=-2, axis2=-1).reshape(-1,1,1)/shape[-1]*np.eye(shape[-1])
        tmp2 = 1j*(tmp0 - tmp0.transpose(0,2,1)) + (tmp1 + tmp1.transpose(0,2,1))
        ret = np.stack([scipy.linalg.expm(1j*x) for x in tmp2])
        # ret = scipy.linalg.expm(1j*tmp2) #TODO scipy-v1.9
    ret = ret.reshape(*shape)
    return ret


def real_to_kraus_op(mat, dim_in, dim_out):
    # this method is in-efficient
    # batched version might have memory issue
    assert (mat.ndim==2) and (mat.shape[0]==mat.shape[1]) and (mat.shape[0]%(dim_in*dim_out)==0)
    matU = real_matrix_to_unitary(mat, with_phase=True)
    ret = matU[:,:dim_in].reshape(-1, dim_out, dim_in)
    return ret


def PSD_to_choi_op(matA, dim_in, use_cholesky=False):
    assert matA.ndim==2 # TODO batch
    shape = matA.shape
    matA = matA.reshape(-1, shape[-1], shape[-1])
    N0 = matA.shape[0]
    assert shape[-1]%dim_in==0
    dim_out = shape[-1]//dim_in
    if is_torch(matA):
        tmp1 = torch.einsum(matA.reshape(N0, dim_in, dim_out, dim_in, dim_out), [0,1,2,3,2], [0,1,3])
        if use_cholesky:
            tmp2 = torch.stack([torch.linalg.inv(torch.linalg.cholesky_ex(x, upper=True)[0]) for x in tmp1])
        else:
            tmp2 = torch.stack([torch.linalg.inv(TorchPSDMatrixSqrtm.apply(x)) for x in tmp1])
        ret = torch.einsum(matA.reshape(N0, dim_in,dim_out,dim_in,dim_out), [6,0,1,2,3],
                tmp2.conj(), [6,0,4], tmp2, [6,2,5], [6,4,1,5,3]).reshape(N0, dim_in*dim_out,-1)
    else:
        tmp1 = np.einsum(matA.reshape(N0, dim_in, dim_out, dim_in, dim_out), [0,1,2,3,2], [0,1,3], optimize=True)
        if use_cholesky:
            tmp2 = np.stack([np.linalg.inv(scipy.linalg.cholesky(x)) for x in tmp1])
        else:
            tmp2 = np.stack([np.linalg.inv(scipy.linalg.sqrtm(x)) for x in tmp1])
        ret = np.einsum(matA.reshape(N0,dim_in,dim_out,dim_in,dim_out), [6,0,1,2,3],
                tmp2.conj(), [6,0,4], tmp2, [6,2,5], [6,4,1,5,3], optimize=True).reshape(N0, dim_in*dim_out, dim_in*dim_out)
    ret = ret.reshape(*shape)
    return ret

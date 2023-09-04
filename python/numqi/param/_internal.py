import numpy as np
import scipy.sparse.linalg
import torch

from .._torch_op import TorchPSDMatrixSqrtm


def _real_matrix_to_trace1_PSD_cholesky(matA, tag_real):
    shape = matA.shape
    matA = matA.reshape(-1, shape[-1], shape[-1])
    N0 = len(matA)
    if isinstance(matA, torch.Tensor):
        if tag_real:
            tmp0 = torch.tril(matA)
        else:
            tmp0 = torch.tril(matA) + 1j*(torch.triu(matA, 1).transpose(1,2))
        ret = tmp0 @ tmp0.transpose(1,2).conj()
        ret = ret / torch.linalg.norm(matA, dim=(1,2), keepdims=True)**2 #shift_max_eig
    else:
        if tag_real:
            tmp0 = np.tril(matA)
        else:
            tmp0 = np.tril(matA) + 1j*(np.triu(matA, 1).transpose(0,2,1))
        ret = tmp0 @ tmp0.transpose(0,2,1).conj()
        ret = ret / np.linalg.norm(matA, axis=(1,2), keepdims=True)**2 #shift_max_eig
    ret = ret.reshape(*shape)
    return ret


def real_matrix_to_trace1_PSD(matA, tag_real=False, use_cholesky=False):
    if use_cholesky:
        ret = _real_matrix_to_trace1_PSD_cholesky(matA, tag_real)
    tmp0 = real_matrix_to_hermitian(matA, tag_real)
    ret = hermitian_matrix_to_trace1_PSD(tmp0)
    return ret


def real_matrix_to_hermitian(matA, tag_real=False):
    shape = matA.shape
    matA = matA.reshape(-1, shape[-1], shape[-1])
    if isinstance(matA, torch.Tensor):
        tmp1 = torch.triu(matA)
        if tag_real:
            ret = (tmp1 + tmp1.transpose(1,2))
        else:
            tmp0 = torch.tril(matA, -1)
            ret = 1j*(tmp0 - tmp0.transpose(1,2)) + (tmp1 + tmp1.transpose(1,2))
    else:
        tmp1 = np.triu(matA)
        if tag_real:
            ret = (tmp1 + tmp1.transpose(0,2,1))
        else:
            tmp0 = np.tril(matA, -1)
            ret = 1j*(tmp0 - tmp0.transpose(0,2,1)) + (tmp1 + tmp1.transpose(0,2,1))
    ret = ret.reshape(*shape)
    return ret


_hf_trace1_np = lambda x: x/np.trace(x)
_hf_trace1_torch = lambda x: x/torch.trace(x)

def hermitian_matrix_to_trace1_PSD(matA):
    shape = matA.shape
    matA = matA.reshape(-1, shape[-1], shape[-1])
    tag_is_torch = isinstance(matA, torch.Tensor)
    N0,N1,_ = matA.shape
    assert N1>=1
    if N1==1:
        ret = torch.ones_like(matA) if tag_is_torch else np.ones_like(matA)
    else:
        tmp3 = matA.detach().cpu().numpy() if tag_is_torch else matA
        if N1==2:
            EVL = np.linalg.eigvalsh(tmp3)[:,-1]
        else:
            EVL = [scipy.sparse.linalg.eigsh(tmp3[x], k=1, which='LA', return_eigenvectors=False)[0] for x in range(N0)]
        if tag_is_torch:
            eye_mat = torch.eye(matA.shape[1], dtype=matA.dtype, device=matA.device)
            ret = torch.stack([_hf_trace1_torch(torch.linalg.matrix_exp(matA[x]-EVL[x]*eye_mat)) for x in range(N0)])
        else:
            eye_mat = np.eye(matA.shape[1])
            ret = np.stack([_hf_trace1_np(scipy.linalg.expm(matA[x]-EVL[x]*eye_mat)) for x in range(N0)])
    ret = ret.reshape(*shape)
    return ret


def real_matrix_to_choi_op(matA, dim_in, use_cholesky=False):
    assert matA.ndim==2 #TODO support batch
    shape = matA.shape
    matA = matA.reshape(-1, shape[-1], shape[-1])
    assert shape[-1]%dim_in==0
    dim_out = shape[-1]//dim_in
    if isinstance(matA, torch.Tensor):
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
            tmp2 = np.linalg.inv(scipy.linalg.sqrtm(tmp1).astype(tmp1.dtype))
            # TODO .astype(xxx.dtype) scipy-v1.10 bug https://github.com/scipy/scipy/issues/18250
        ret = np.einsum(mat0.reshape(dim_in,dim_out,dim_in,dim_out), [0,1,2,3],
                tmp2.conj(), [0,4], tmp2, [2,5], [4,1,5,3], optimize=True).reshape(dim_in*dim_out,-1)
    ret = ret.reshape(*shape)
    return ret


def real_matrix_to_special_unitary(matA, tag_real=False):
    assert matA.shape[-1]==matA.shape[-2]
    shape = matA.shape
    matA = matA.reshape(-1, shape[-1], shape[-1])
    if isinstance(matA, torch.Tensor):
        if tag_real:
            tmp0 = torch.triu(matA, 1)
            tmp1 = tmp0 - tmp0.transpose(1,2)
            # torch.linalg.matrix_exp for a batch of input will lead to memory issue, so use torch.stack()
            ret = torch.stack([torch.linalg.matrix_exp(tmp1[x]) for x in range(len(tmp1))])
        else:
            tmp0 = torch.tril(matA, -1)
            tmp1 = torch.triu(matA)
            tmp2 = torch.diagonal(tmp1, dim1=-2, dim2=-1).mean(dim=1).reshape(-1,1,1)
            tmp3 = tmp1 - tmp2*torch.eye(shape[-1], device=matA.device)
            tmp4 = 1j*(tmp0 - tmp0.transpose(1,2)) + (tmp3 + tmp3.transpose(1,2))
            ret = torch.stack([torch.linalg.matrix_exp(1j*tmp4[x]) for x in range(len(tmp4))])
    else:
        if tag_real:
            tmp0 = np.triu(matA, 1)
            tmp1 = tmp0 - tmp0.transpose(0,2,1)
            ret = np.stack([scipy.linalg.expm(x) for x in tmp1])
            # ret = scipy.linalg.expm(tmp1) #TODO scipy-v1.9
        else:
            tmp0 = np.tril(matA, -1)
            tmp1 = np.triu(matA)
            tmp1 = tmp1 - (np.trace(tmp1, axis1=-2, axis2=-1).reshape(-1,1,1)/shape[-1])*np.eye(shape[-1])
            tmp2 = 1j*(tmp0 - tmp0.transpose(0,2,1)) + (tmp1 + tmp1.transpose(0,2,1))
            ret = np.stack([scipy.linalg.expm(1j*x) for x in tmp2])
            # ret = scipy.linalg.expm(1j*tmp2) #TODO scipy-v1.9
    ret = ret.reshape(shape)
    return ret


def real_to_kraus_op(mat, dim_in, dim_out):
    # this method is in-efficient
    # batched version might have memory issue
    assert (mat.ndim==2) and (mat.shape[0]==mat.shape[1]) and (mat.shape[0]%(dim_in*dim_out)==0)
    matU = real_matrix_to_special_unitary(mat)
    ret = matU[:,:dim_in].reshape(-1, dim_out, dim_in)
    return ret


def PSD_to_choi_op(matA, dim_in, use_cholesky=False):
    assert matA.ndim==2 # TODO batch
    shape = matA.shape
    matA = matA.reshape(-1, shape[-1], shape[-1])
    N0 = matA.shape[0]
    assert shape[-1]%dim_in==0
    dim_out = shape[-1]//dim_in
    if isinstance(matA, torch.Tensor):
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
            tmp2 = np.stack([np.linalg.inv(scipy.linalg.sqrtm(x).astype(x.dtype)) for x in tmp1])
            # TODO .astype(xxx.dtype) scipy-v1.10 bug https://github.com/scipy/scipy/issues/18250
        ret = np.einsum(matA.reshape(N0,dim_in,dim_out,dim_in,dim_out), [6,0,1,2,3],
                tmp2.conj(), [6,0,4], tmp2, [6,2,5], [6,4,1,5,3], optimize=True).reshape(N0, dim_in*dim_out, dim_in*dim_out)
    ret = ret.reshape(*shape)
    return ret


def get_rational_orthogonal2_matrix(m, n):
    # https://en.wikipedia.org/wiki/Pythagorean_triple
    m = int(m)
    n = int(n)
    assert (m!=0) and (n!=0) and (abs(m)!=abs(n))
    a = m*m - n*n
    b = 2*m*n
    c = m*m + n*n
    # print(a,b,c)
    st = a/c
    ct = b/c
    ret = np.array([[ct,st],[-st,ct]])
    return ret

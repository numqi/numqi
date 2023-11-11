import numpy as np
import scipy.sparse.linalg
import scipy.special
import torch
import opt_einsum

from .._torch_op import TorchPSDMatrixSqrtm


def real_to_bounded(theta, lower:float, upper:float):
    r'''map a real number into a bounded interval (lower,upper) using sigmoid function

    Parameters:
        theta (np.ndarray,torch.tensor): array of any shape.
        lower (float): lower bound.
        upper (float): upper bound.

    Returns:
        ret (np.ndarray,torch.tensor): array of the same shape as `theta`.
    '''
    assert lower < upper
    if isinstance(theta, torch.Tensor):
        tmp0 = torch.sigmoid(theta) #1/(1+exp(-theta))
    else:
        tmp0 = scipy.special.expit(theta)
    ret = tmp0 * (upper - lower) + lower
    return ret


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
    r'''map a real matrix to a positive semi-definite matrix with trace 1 (density matrix)

    Parameters:
        matA (np.ndarray,torch.tensor): a real matrix of shape (...,N0,N0) where `N0` is the dimension.
                If `tag_real=True`, only the upper triangular part (include diagonal element) of `matA` is used.
        tag_real (bool): If `tag_real=True`, the output is real. Otherwise the output is complex.
        use_cholesky (bool): If `use_cholesky=True`, the Cholesky decomposition scheme is used to map `matA`.
                Otherwise the matrix exponential scheme is used (time-consuming).

    Returns:
        ret (np.ndarray,torch.tensor): a matrix of shape (...,N0,N0) which satisfies the positive semi-definite condition.
    '''
    if use_cholesky:
        ret = _real_matrix_to_trace1_PSD_cholesky(matA, tag_real)
    tmp0 = real_matrix_to_hermitian(matA, tag_real)
    ret = hermitian_matrix_to_trace1_PSD(tmp0)
    return ret


def real_matrix_to_hermitian(matA, tag_real=False):
    r'''map a real matrix to a Hermitian matrix

    Parameters:
        matA (np.ndarray,torch.tensor): a real matrix of shape (...,N0,N0) where `N0` is the dimension.
                If `tag_real=True`, only the upper triangular part (include diagonal element) of `matA` is used.
        tag_real (bool): If `tag_real=True`, the output is real. Otherwise the output is complex.

    Returns:
        ret (np.ndarray,torch.tensor): a matrix of shape (...,N0,N0) which satisfies the Hermitian condition.
    '''
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
        if N1<=5: #5 is chosen intuitively
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


def matrix_to_stiefel(mat, method:str='sqrtm'):
    r'''map a matrix to the Stiefel manifold

    wiki-link: https://en.wikipedia.org/wiki/Stiefel_manifold

    $$ \left\{ x\in\mathbb{K}^{m\times n}:x^\dagger x=I_n \right\}$$

    where $\mathbb{K}=\mathbb{R}$ or $\mathbb{C}$. Matrix square root is used so the backward might be
    several times slower than the forward. QR decomposition

    Parameters:
        mat (np.ndarray,torch.tensor): a matrix of shape (...,m,n). WARNING, it must be `m>=n`.
                It can be real or complex.
        method (str): 'sqrtm' or 'qr'. We observed that 'sqrtm' is faster in optimization.

    Returns:
        ret (np.ndarray,torch.tensor): a matrix of shape (...,m,n) on the Stiefel manifold.
                It is real if `mat` is real, and complex if `mat` is complex.
    '''
    assert mat.ndim>=2
    assert method in {'sqrtm','qr'}
    if method=='sqrtm':
        shape = mat.shape
        dim0,dim1 = shape[-2:]
        assert dim0>=dim1, f'row number should be smaller than column number, but got ({dim0},{dim1})'
        mat = mat.reshape(-1, dim0, dim1)
        tmp0 = opt_einsum.contract(mat, [0,1,2], mat.conj(), [0,1,3], [0,2,3])
        if isinstance(mat, torch.Tensor):
            tmp1 = torch.stack([torch.linalg.inv(TorchPSDMatrixSqrtm.apply(x)) for x in tmp0])
        else:
            tmp1 = np.stack([np.linalg.inv(scipy.linalg.sqrtm(x).astype(x.dtype)) for x in tmp0])
        ret = opt_einsum.contract(mat, [0,1,2], tmp1, [0,3,2], [0,1,3])
        ret = ret.reshape(*shape)
    else:
        # TODO poor efficiency in L-BFGS-B optimization
        # the forward and backward time is still okay compared with matrix-square-root,
        # but the time for L-BFGS-B optimization is much longer (strange)
        if isinstance(mat, torch.Tensor):
            ret = torch.linalg.qr(mat, mode='reduced')[0]
        else:
            ret = np.linalg.qr(mat, mode='reduced')[0]
    return ret


def matrix_to_kraus_op(mat, method='sqrtm'):
    r'''map a matrix to a Kraus operator

    For a quantum channel with output dimension m, input dimension n, and rank l (number of Kraus operator),
    the Kraus operator satisfies

    $$\left\{ x\in\mathbb{C}^{l\times m\times n}: \sum_{ij} x_{iju}x^*_{ijv}=\delta_{uv} \right\}$$

    Parameters:
        mat (np.ndarray,torch.tensor): a matrix of shape (...,l,m,n) where `l` is the number of Kraus operators,
                `m` is the output dimension, and `n` is the input dimension. WARNING, it must be `m>=n`.
        method (str): 'sqrtm' or 'qr'. We observed that 'sqrtm' is faster in optimization.

    Returns:
        ret (np.ndarray,torch.tensor): a matrix of shape (...,l,m,n) which satisfies the Kraus operator condition.
    '''
    assert mat.ndim>=3
    shape = mat.shape
    rank,dim_out,dim_in = shape[-3:]
    ret = matrix_to_stiefel(mat.reshape(-1, rank*dim_out, dim_in), method).reshape(*shape)
    return ret


def matrix_to_choi_op(mat, method='sqrtm'):
    r'''map a matrix to a Choi operator

    For a quantum channel with output dimension m, input dimension n, and rank l, the Choi operator satisfies

    $$\left\{ x\in\mathbb{C}^{m\times n\times m\times n}:x\succeq 0 \sum_i x_{iuiv}=\delta_{uv} \right\}$$

    where $\succeq 0$ means positive semi-definite.

    Parameters:
        mat (np.ndarray,torch.tensor): a matrix of shape (...,l,m,n) where `l` is the number of Kraus operators,
                `m` is the output dimension, and `n` is the input dimension. WARNING, it must be `m>=n`.
        method (str): 'sqrtm' or 'qr'. We observed that 'sqrtm' is faster in optimization.

    Returns:
        ret (np.ndarray,torch.tensor): a matrix of shape (...,m,n,m,n) which satisfies the Choi operator condition.
    '''
    assert mat.ndim>=3
    shape = mat.shape
    rank,dim_out,dim_in = shape[-3:]
    tmp0 = matrix_to_stiefel(mat.reshape(-1, rank*dim_out, dim_in), method).reshape(-1,rank,dim_out,dim_in)
    tmp1 = opt_einsum.contract(tmp0, [0,1,2,3], tmp0.conj(), [0,1,4,5], [2,3,4,5])
    ret = tmp1.reshape(shape[:-3]+(dim_out,dim_in,dim_out,dim_in))
    return ret


def real_matrix_to_special_unitary(matA, tag_real:bool=False):
    r'''map a real matrix to a special unitary matrix

    Parameters:
        matA (np.ndarray,torch.tensor): a real matrix of shape (...,N0,N0) where `N0` is the dimension.
        tag_real (bool): If `tag_real=True`, the output is real (special orthogonal matrix),
                otherwise the output is complex (special unitary matrix).
                If `tag_real=True`, only the upper triangular part (not include diagonal element) of `matA` is used.

    Returns:
        ret (np.ndarray,torch.tensor): a matrix of shape (...,N0,N0) which satisfies the special unitary condition.
    '''
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


# def PSD_to_choi_op(matA, dim_in, use_cholesky=False):
#     assert matA.ndim==2 # TODO batch
#     shape = matA.shape
#     matA = matA.reshape(-1, shape[-1], shape[-1])
#     N0 = matA.shape[0]
#     assert shape[-1]%dim_in==0
#     dim_out = shape[-1]//dim_in
#     if isinstance(matA, torch.Tensor):
#         tmp1 = torch.einsum(matA.reshape(N0, dim_in, dim_out, dim_in, dim_out), [0,1,2,3,2], [0,1,3])
#         if use_cholesky:
#             tmp2 = torch.stack([torch.linalg.inv(torch.linalg.cholesky_ex(x, upper=True)[0]) for x in tmp1])
#         else:
#             tmp2 = torch.stack([torch.linalg.inv(TorchPSDMatrixSqrtm.apply(x)) for x in tmp1])
#         ret = torch.einsum(matA.reshape(N0, dim_in,dim_out,dim_in,dim_out), [6,0,1,2,3],
#                 tmp2.conj(), [6,0,4], tmp2, [6,2,5], [6,4,1,5,3]).reshape(N0, dim_in*dim_out,-1)
#     else:
#         tmp1 = np.einsum(matA.reshape(N0, dim_in, dim_out, dim_in, dim_out), [0,1,2,3,2], [0,1,3], optimize=True)
#         if use_cholesky:
#             tmp2 = np.stack([np.linalg.inv(scipy.linalg.cholesky(x)) for x in tmp1])
#         else:
#             tmp2 = np.stack([np.linalg.inv(scipy.linalg.sqrtm(x).astype(x.dtype)) for x in tmp1])
#             # TODO .astype(xxx.dtype) scipy-v1.10 bug https://github.com/scipy/scipy/issues/18250
#         ret = np.einsum(matA.reshape(N0,dim_in,dim_out,dim_in,dim_out), [6,0,1,2,3],
#                 tmp2.conj(), [6,0,4], tmp2, [6,2,5], [6,4,1,5,3], optimize=True).reshape(N0, dim_in*dim_out, dim_in*dim_out)
#     ret = ret.reshape(*shape)
#     return ret


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


# TODO see ws00

# def real_to_povm():
#     pass

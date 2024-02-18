import numpy as np
import scipy.special
import torch

import numqi.gellmann
import numqi._torch_op

_CPU = torch.device('cpu')

def _hf_para(dtype, requires_grad, *size):
    # the value does not matter for that it's initialized in numqi.optimize.minimize
    ret = torch.nn.Parameter(torch.rand(*size, dtype=dtype)-0.5, requires_grad=requires_grad)
    return ret

class PositiveReal(torch.nn.Module):
    def __init__(self, batch_size:(int|None)=None, method:str='softplus',
                requires_grad:bool=True, dtype:torch.dtype=torch.float64, device:torch.device=_CPU):
        r'''positive real number

        Parameters:
            batch_size (int,None): batch size.
            method (str): method to map real vector to a positive real number.
                'softplus': softplus function.
                'exp': exponential function.
            requires_grad (bool): whether to track the gradients of the parameters.
            dtype (torch.dtype): data type of the parameters, either torch.float32 or torch.float64
            device (torch.device): device of the parameters.
        '''
        assert dtype in {torch.float32,torch.float64}
        assert (batch_size is None) or (batch_size>0)
        assert method in {'softplus','exp'}
        assert isinstance(device, torch.device)
        super().__init__()
        self.theta = _hf_para(dtype, requires_grad, 1 if (batch_size is None) else batch_size).to(device)
        self.method = method
        self.batch_size = batch_size

    def forward(self):
        if self.method=='softplus':
            ret = to_positive_real_softplus(self.theta)
        else:
            ret = to_positive_real_exp(self.theta)
        return ret

def to_positive_real_softplus(theta):
    r'''map real vector to a positive real number using softplus function

    Parameters:
        theta (np.ndarray,torch.Tensor): array of any shape.

    Returns:
        ret (np.ndarray,torch.Tensor): array of the same shape as `theta`.
    '''
    if isinstance(theta, torch.Tensor):
        ret = torch.nn.functional.softplus(theta)
    else:
        ret = _np_softplus(theta)
    return ret


def to_positive_real_exp(theta):
    r'''map real vector to a positive real number using exponential function

    Parameters:
        theta (np.ndarray,torch.Tensor): array of any shape.

    Returns:
        ret (np.ndarray,torch.Tensor): array of the same shape as `theta`.
    '''
    if isinstance(theta, torch.Tensor):
        ret = torch.exp(theta)
    else:
        ret = np.exp(theta)
    return ret

class OpenInterval(torch.nn.Module):
    def __init__(self, lower:float, upper:float, batch_size:(int|None)=None,
                requires_grad:bool=True, dtype:torch.dtype=torch.float64, device:torch.device=_CPU):
        r'''open interval manifold (lower,upper) using sigmoid function

        Parameters:
            lower (float): lower bound.
            upper (float): upper bound.
            requires_grad (bool): whether to track the gradients of the parameters.
            dtype (torch.dtype): data type of the parameters, either torch.float32 or torch.float64
            device (torch.device): device of the parameters.
        '''
        assert dtype in (torch.float32, torch.float64)
        assert (batch_size is None) or (batch_size>0)
        assert isinstance(device, torch.device)
        super().__init__()
        self.lower = torch.tensor(lower, dtype=dtype, device=device)
        self.upper = torch.tensor(upper, dtype=dtype, device=device)
        self.theta = _hf_para(dtype, requires_grad, 1 if (batch_size is None) else batch_size).to(device)
        self.batch_size = batch_size

    def forward(self):
        tmp0 = self.theta[0] if (self.batch_size is None) else self.theta
        ret = to_open_interval(tmp0, self.lower, self.upper)
        return ret


def to_open_interval(theta, lower:float, upper:float):
    r'''map a real number into a bounded interval (lower,upper) using sigmoid function

    Parameters:
        theta (np.ndarray,torch.tensor): array of any shape.
        lower (float): lower bound.
        upper (float): upper bound.

    Returns:
        ret (np.ndarray,torch.tensor): array of the same shape as `theta`.
    '''
    # exponential overflow is handled by scipy.special.expit/torch.sigmoid
    if isinstance(theta, torch.Tensor):
        tmp0 = torch.sigmoid(theta) #1/(1+exp(-theta))
    else:
        tmp0 = scipy.special.expit(theta)
    ret = tmp0 * (upper - lower) + lower
    return ret


class Trace1PSD(torch.nn.Module):
    def __init__(self, dim:int, rank:(int|None)=None, batch_size:(int|None)=None,
                method:str='cholesky', requires_grad:bool=True, dtype:torch.dtype=torch.float64, device:torch.device=_CPU):
        r'''positive semi-definite (PSD) matrix with trace 1 of rank `rank` using Cholesky decomposition

        Parameters:
            dim (int): dimension of the matrix.
            rank (int): rank of the matrix.
            batch_size (int,None): batch size.
            method (str): method to map real vector to a PSD matrix.
                'cholesky': Cholesky decomposition.
                'ensemble': ensemble decomposition.
            requires_grad (bool): whether to track the gradients of the parameters.
            dtype (torch.dtype): data type of the parameters
                torch.float32 / torch.float64: real PSD matrix
                torch.complex64 / torch.complex128: complex PSD matrix
            device (torch.device): device of the parameters.
        '''
        super().__init__()
        assert method in {'cholesky','ensemble'}
        assert dim>=2
        assert dtype in {torch.float32,torch.float64,torch.complex64,torch.complex128}
        assert (batch_size is None) or (batch_size>0)
        assert isinstance(device, torch.device)
        is_real = dtype in {torch.float32,torch.float64}
        if rank is None:
            rank = dim
        tmp0 = torch.float32 if (dtype in {torch.float32,torch.complex64}) else torch.float64
        if method=='cholesky':
            N0 = (rank*(2*dim-rank+1))//2
            tmp1 = N0 if is_real else (2*N0-rank)
        else: #ensemble
            tmp1 = (rank+dim*rank) if is_real else (rank+2*dim*rank)
        tmp2 = (tmp1,) if (batch_size is None) else (batch_size, tmp1)
        self.theta = _hf_para(tmp0, requires_grad, *tmp2).to(device)
        self.dim = int(dim)
        self.rank = int(rank)
        self.dtype = dtype
        self.method = method
        self.batch_size = batch_size

    def forward(self):
        if self.method=='cholesky':
            ret = to_trace1_psd_cholesky(self.theta, self.dim, self.rank)
        else: #ensemble
            ret = to_trace1_psd_ensemble(self.theta, self.dim, self.rank)
        return ret


def _np_softplus(x):
    tmp0 = np.sign(x)
    ret = np.log1p(np.exp(-tmp0 * x)) + (1+tmp0)/2 * x
    return ret


def to_trace1_psd_ensemble(theta, dim:int, rank:(int|None)=None):
    r'''map real vector to a positive semi-definite (PSD) matrix with trace 1 using ensemble method

    Parameters:
        theta (np.ndarray,torch.Tensor): if `ndim>1`, then the last dimension will be expanded to the matrix
                and the rest dimensions will be batch dimensions
        dim (int): dimension of the matrix.
        rank (int): rank of the matrix.

    Returns:
        ret (np.ndarray,torch.Tensor): array of shape `theta.shape[:-1]+(dim,dim)`
    '''
    if rank is None:
        rank = dim
    if theta.shape[-1]==(rank+dim*rank):
        is_real = True
    else:
        assert theta.shape[-1]==(rank+2*dim*rank)
        is_real = False
    shape = theta.shape
    theta = theta.reshape(-1, shape[-1])
    theta_p = to_discrete_probability_softmax(theta[:,:rank])
    theta_psi = to_sphere_quotient(theta[:,rank:].reshape(theta.shape[0]*rank, -1), is_real).reshape(-1, rank, dim)
    if isinstance(theta, torch.Tensor):
        ret = torch.einsum(theta_p, [0,1], theta_psi, [0,1,2], theta_psi.conj(), [0,1,3], [0,2,3])
    else:
        ret = np.einsum(theta_p, [0,1], theta_psi, [0,1,2], theta_psi.conj(), [0,1,3], [0,2,3], optimize=True)
    ret = ret.reshape(*shape[:-1], dim, dim)
    return ret


def to_trace1_psd_cholesky(theta, dim:int, rank:(int|None)=None):
    r'''map real vector to a positive semi-definite (PSD) matrix with trace 1 of rank `rank` using Cholesky decomposition

    Parameters:
        theta (np.ndarray,torch.Tensor): if `ndim>1`, then the last dimension will be expanded to the matrix
                and the rest dimensions will be batch dimensions.
        dim (int): dimension of the matrix.
        rank (int): rank of the matrix.

    Returns:
        ret (np.ndarray,torch.Tensor): array of shape `theta.shape[:-1]+(dim,dim)`
    '''
    if rank is None:
        rank = dim
    N0 = (rank*(2*dim-rank+1))//2
    if theta.shape[-1]==N0:
        is_real = True
    else:
        is_real = False
        assert theta.shape[-1]==(2*N0-rank), f'shape "{N0}" for real case or "{2*N0-rank}" for complex case expected, got {theta.shape[-1]}'
    shape = theta.shape
    theta = theta.reshape(-1, shape[-1])
    N1 = theta.shape[0]
    is_torch = isinstance(theta, torch.Tensor)
    if is_torch:
        assert (theta.dtype==torch.float32) or (theta.dtype==torch.float64)
        indexL = torch.tril_indices(dim, rank, -1)
        indexD = torch.arange(rank, dtype=torch.int32, device=theta.device)
        tmp0 = theta[:,rank:]
        tmp1 = torch.nn.functional.softplus(theta[:,:rank])
        norm_factor = torch.sqrt(torch.linalg.norm(tmp0, axis=1)**2 + torch.linalg.norm(tmp1, axis=1)**2).reshape(-1,1)
        tmp2 = (theta.dtype if is_real else (torch.complex64 if (theta.dtype==torch.float32) else torch.complex128))
        tmp3 = torch.zeros(N1, dim, rank, dtype=tmp2, device=theta.device)
        tmp3[:,indexD,indexD] = (tmp1 if is_real else tmp1.to(tmp3.dtype)) / norm_factor
        if is_real:
            tmp3[:,indexL[0],indexL[1]] = tmp0 / norm_factor
        else:
            tmp3[:,indexL[0],indexL[1]] = torch.complex(tmp0[:,:(N0-rank)], tmp0[:,(N0-rank):]) / norm_factor
        ret = tmp3 @ tmp3.transpose(1,2).conj()
    else:
        assert (theta.dtype.type==np.float32) or (theta.dtype.type==np.float64)
        indexL = np.tril_indices(dim, -1, rank)
        indexD = np.arange(rank, dtype=np.int32)
        tmp0 = theta[:,rank:]
        tmp1 = _np_softplus(theta[:,:rank])
        norm_factor = np.sqrt(np.linalg.norm(tmp0, axis=1)**2 + np.linalg.norm(tmp1, axis=1)**2).reshape(-1,1)
        tmp2 = (theta.dtype if is_real else (np.complex64 if (theta.dtype.type==np.float32) else np.complex128))
        tmp3 = np.zeros((N1, dim, rank), dtype=tmp2)
        tmp3[:,indexD,indexD] = tmp1 / norm_factor
        if is_real:
            tmp3[:,indexL[0],indexL[1]] = tmp0 / norm_factor
        else:
            tmp3[:,indexL[0],indexL[1]] = (tmp0[:,:(N0-rank)] + 1j* tmp0[:,(N0-rank):]) / norm_factor
        ret = tmp3 @ tmp3.transpose(0,2,1).conj()
    ret = ret.reshape(*shape[:-1], dim, dim)
    return ret


class SymmetricMatrix(torch.nn.Module):
    def __init__(self, dim:int, batch_size:(int|None)=None, is_trace0=False, is_norm1=False,
                    requires_grad:bool=True, dtype:torch.dtype=torch.float64, device:torch.device=_CPU):
        r'''symmetric matrix, hermitian, is_trace0, is_norm1

        Parameters:
            dim (int): dimension of the matrix.
            batch_size (int,None): batch size.
            is_trace0 (bool): whether the trace of the matrix is 0.
            is_norm1 (bool): whether the frobenius norm of the matrix is 1.
            requires_grad (bool): whether to track the gradients of the parameters.
            dtype (torch.dtype): data type of the parameters
                torch.float32 / torch.float64: real symmetric matrix
                torch.complex64 / torch.complex128: complex hermitian matrix
            device (torch.device): device of the parameters.
        '''
        super().__init__()
        assert dim>=2
        assert dtype in {torch.float32,torch.float64,torch.complex64,torch.complex128}
        assert (batch_size is None) or (batch_size>0)
        assert isinstance(device, torch.device)
        is_real = dtype in {torch.float32,torch.float64}
        tmp0 = torch.float32 if (dtype in {torch.float32,torch.complex64}) else torch.float64
        N0 = (dim*(dim+1))//2 if is_real else dim*dim
        if is_trace0:
            N0 = N0 - 1
        tmp1 = (N0,) if (batch_size is None) else (batch_size, N0)
        self.theta = _hf_para(tmp0, requires_grad, *tmp1).to(device)
        self.is_trace0 = is_trace0
        self.is_norm1 = is_norm1
        self.dim = int(dim)
        self.dtype = dtype
        self.is_real = is_real
        self.batch_size = batch_size

    def forward(self):
        ret = to_symmetric_matrix(self.theta, self.dim, self.is_trace0, self.is_norm1)
        return ret


def to_symmetric_matrix(theta, dim:int, is_trace0=False, is_norm1=False):
    r'''map real vector to a symmetric matrix

    Parameters:
        theta (np.ndarray,torch.Tensor): if `ndim>1`, then the last dimension will be expanded to the matrix
                and the rest dimensions will be batch dimensions.
                symmetric matrix: (dim*(dim+1))//2
                hermitian matrix: dim*dim
                traceless symmetric matrix: (dim*(dim+1))//2-1
                traceless hermitian matrix: dim*dim-1
        dim (int): dimension of the matrix.
        is_trace0 (bool): whether the trace of the matrix is 0.
        is_norm1 (bool): whether the frobenius norm of the matrix is 1.

    Returns:
        ret (np.ndarray,torch.Tensor): array of shape `theta.shape[:-1]+(dim,dim)`
    '''
    N0 = (dim*(dim-1))//2
    if theta.shape[-1]==((N0+dim) - (1 if is_trace0 else 0)):
        is_real = True
    else:
        assert theta.shape[-1]==(dim*dim - (1 if is_trace0 else 0))
        is_real = False
    shape = theta.shape
    theta = theta.reshape(-1, shape[-1])
    N1 = theta.shape[0]
    if isinstance(theta, torch.Tensor):
        if is_real:
            if is_trace0:
                tmp0 = torch.zeros(N1, N0, dtype=theta.dtype, device=theta.device)
                tmp1 = torch.zeros(N1, 1, dtype=theta.dtype, device=theta.device)
                tmp2 = torch.concat([theta[:,:N0], tmp0, theta[:,N0:], tmp1], axis=1)
                ret = numqi.gellmann.gellmann_basis_to_matrix(tmp2).real
            else:
                indexU = torch.triu_indices(dim,dim)
                ret = torch.zeros(N1, dim, dim, dtype=theta.dtype, device=theta.device)
                ret[:,indexU[0],indexU[1]] = theta
                ret = ret + ret.transpose(1,2)
        else:
            if is_trace0:
                tmp0 = torch.zeros(N1, 1, dtype=theta.dtype, device=theta.device)
                ret = numqi.gellmann.gellmann_basis_to_matrix(torch.concat([theta, tmp0], axis=1))
            else:
                theta = theta.reshape(-1, dim, dim)
                tmp0 = torch.triu(theta)
                tmp1 = torch.tril(theta,-1)
                ret = tmp0 + tmp0.transpose(1,2) + 1j*(tmp1 - tmp1.transpose(1,2))
        if is_norm1:
            ret = ret / torch.linalg.norm(ret, dim=(1,2), keepdims=True)
    else: #numpy
        if is_real:
            if is_trace0:
                tmp0 = np.zeros((N1, N0))
                tmp1 = np.zeros((N1, 1))
                tmp2 = np.concatenate([theta[:,:N0], tmp0, theta[:,N0:], tmp1], axis=1)
                ret = numqi.gellmann.gellmann_basis_to_matrix(tmp2).real
            else:
                indexU = np.triu_indices(dim)
                ret = np.zeros((N1, dim, dim), dtype=theta.dtype)
                ret[:,indexU[0],indexU[1]] = theta
                ret = ret + ret.transpose(0,2,1)
        else:
            if is_trace0:
                tmp0 = np.zeros((N1, 1))
                ret = numqi.gellmann.gellmann_basis_to_matrix(np.concatenate([theta, tmp0], axis=1))
            else:
                theta = theta.reshape(-1, dim, dim)
                tmp0 = np.triu(theta)
                tmp1 = np.tril(theta,-1)
                ret = tmp0 + tmp0.transpose(0,2,1) + 1j*(tmp1 - tmp1.transpose(0,2,1))
        if is_norm1:
            ret = ret / np.linalg.norm(ret, axis=(1,2), keepdims=True)
    ret = ret.reshape(*shape[:-1], dim, dim)
    return ret


class Ball(torch.nn.Module):
    def __init__(self, dim:int, batch_size:(int|None)=None,
                requires_grad:bool=True, dtype:torch.dtype=torch.float64, device:torch.device=_CPU):
        r'''ball manifold

        Parameters:
            dim (int): dimension of the ball.
            batch_size (int,None): batch size.
            requires_grad (bool): whether to track the gradients of the parameters.
            dtype (torch.dtype): data type of the parameters, either torch.float32 or torch.float64
            device (torch.device): device of the parameters.
        '''
        super().__init__()
        assert dim>=2
        assert dtype in {torch.float32,torch.float64,torch.complex64,torch.complex128}
        assert (batch_size is None) or (batch_size>0)
        assert isinstance(device, torch.device)
        if dtype in {torch.float32,torch.float64}:
            self.is_real = True
            tmp0 = torch.float32 if (dtype==torch.float32) else torch.float64
            tmp1 = dim
        else:
            self.is_real = False
            tmp0 = torch.float32 if (dtype==torch.complex64) else torch.float64
            tmp1 = 2*dim
        tmp2 = (tmp1,) if (batch_size is None) else (batch_size, tmp1)
        self.theta = _hf_para(tmp0, requires_grad, *tmp2).to(device)
        self.dim = int(dim)
        self.dtype = dtype
        self.batch_size = batch_size

    def forward(self):
        ret = to_ball(self.theta, self.is_real)
        return ret

def to_ball(theta, is_real:bool=True):
    r'''map real vector to a point in the ball

    Parameters:
        theta (np.ndarray,torch.Tensor): if `ndim>1`, then the last dimension will be expanded to the point
                and the rest dimensions will be batch dimensions.
        is_real (bool): whether the output is real

    Returns:
        ret (np.ndarray,torch.Tensor): array of shape `theta.shape[:-1]+(dim,)`
    '''
    if isinstance(theta, torch.Tensor):
        tmp0 = torch.linalg.norm(theta, dim=-1, keepdims=True)
        ret = theta * (tmp0 / (1+tmp0))
        if not is_real:
            tmp0 = theta.shape[-1]//2
            ret = torch.complex(ret[...,:tmp0], ret[...,tmp0:])
    else:
        tmp0 = np.linalg.norm(theta, axis=-1, keepdims=True)
        ret = theta * (tmp0 / (1+tmp0))
        if not is_real:
            tmp0 = theta.shape[-1]//2
            ret = ret[...,:tmp0] + 1j* ret[...,tmp0:]
    return ret


class Sphere(torch.nn.Module):
    def __init__(self, dim:int, batch_size:(int|None)=None, method:str='quotient',
                requires_grad:bool=True, dtype:torch.dtype=torch.float64, device:torch.device=_CPU):
        r'''sphere manifold

        Parameters:
            dim (int): dimension of the sphere.
            batch_size (int,None): batch size.
            method (str): method to map real vector to a point on the sphere.
                'quotient': quotient map.
                'coordinate': cosine and sine functions.
            requires_grad (bool): whether to track the gradients of the parameters.
            dtype (torch.dtype): data type of the parameters, either torch.float32 or torch.float64
            device (torch.device): device of the parameters.
        '''
        super().__init__()
        assert dim>=2
        assert dtype in {torch.float32,torch.float64,torch.complex64,torch.complex128}
        assert (batch_size is None) or (batch_size>0)
        assert method in {'quotient','coordinate'}
        assert isinstance(device, torch.device)
        if dtype in {torch.float32,torch.float64}:
            self.is_real = True
            tmp0 = torch.float32 if (dtype==torch.float32) else torch.float64
            tmp1 = dim if (method=='quotient') else (dim-1)
        else:
            self.is_real = False
            tmp0 = torch.float32 if (dtype==torch.complex64) else torch.float64
            tmp1 = 2*dim if (method=='quotient') else (2*dim-1)
        tmp2 = (tmp1,) if (batch_size is None) else (batch_size, tmp1)
        self.theta = _hf_para(tmp0, requires_grad, *tmp2).to(device)
        self.dim = int(dim)
        self.batch_size = batch_size
        self.dtype = dtype
        self.method = method

    def forward(self):
        if self.method=='quotient':
            ret = to_sphere_quotient(self.theta, self.is_real)
        else:
            ret = to_sphere_coordinate(self.theta, self.is_real)
        return ret


def to_sphere_quotient(theta, is_real:bool=True):
    r'''map real vector to a point on the sphere via quotient

    Parameters:
        theta (np.ndarray,torch.Tensor): if `ndim>1`, then the last dimension will be expanded to the point
                and the rest dimensions will be batch dimensions.
        is_real (bool): whether the output is real

    Returns:
        ret (np.ndarray,torch.Tensor): array of shape `theta.shape[:-1]+(dim,)`
    '''
    if isinstance(theta, torch.Tensor):
        ret = theta / torch.linalg.norm(theta, dim=-1, keepdims=True)
        if not is_real:
            tmp0 = theta.shape[-1]//2
            ret = torch.complex(ret[...,:tmp0], ret[...,tmp0:])
    else:
        ret = theta / np.linalg.norm(theta, axis=-1, keepdims=True)
        if not is_real:
            tmp0 = theta.shape[-1]//2
            ret = ret[...,:tmp0] + 1j* ret[...,tmp0:]
    return ret

def to_sphere_coordinate(theta, is_real:bool=True):
    r'''map real vector to a point on the sphere via cosine and sine functions

    reference: A Derivation of n-Dimensional Spherical Coordinates
    [doi-link](https://doi.org/10.2307/2308932)

    Parameters:
        theta (np.ndarray,torch.Tensor): if `ndim>1`, then the last dimension will be expanded to the point
                and the rest dimensions will be batch dimensions.
        is_real (bool): whether the output is real

    Returns:
        ret (np.ndarray,torch.Tensor): array of shape `theta.shape[:-1]+(dim,)`
    '''
    shape = theta.shape
    theta = theta.reshape(-1, shape[-1])
    if isinstance(theta, torch.Tensor):
        ct = torch.cos(theta)
        st = torch.sin(theta)
        if shape[-1]==1:
            ret = torch.concat([ct,st], dim=1)
        else:
            tmp0 = torch.cumprod(st, dim=1)
            ret = torch.concat([ct[:,:1],ct[:,1:]*tmp0[:,:-1],tmp0[:,-1:]], dim=1)
        if not is_real:
            tmp0 = ret.shape[1]//2
            ret = torch.complex(ret[:,:tmp0], ret[:,tmp0:])
    else: #numpy
        ct = np.cos(theta)
        st = np.sin(theta)
        if shape[-1]==1:
            ret = np.concatenate([ct,st], axis=1)
        else:
            tmp0 = np.cumprod(st, axis=1)
            ret = np.concatenate([ct[:,:1],ct[:,1:]*tmp0[:,:-1],tmp0[:,-1:]], axis=1)
        if not is_real:
            tmp0 = ret.shape[1]//2
            ret = ret[:,:tmp0] + 1j* ret[:,tmp0:]
    ret = ret.reshape(*shape[:-1], -1)
    return ret


class DiscreteProbability(torch.nn.Module):
    def __init__(self, dim:int, batch_size:(int|None)=None, method:str='softmax', weight=None,
                requires_grad:bool=True, dtype:torch.dtype=torch.float64, device:torch.device=_CPU):
        r'''discrete probability distribution

        Parameters:
            dim (int): dimension of the probability vector.
            batch_size (int,None): batch size.
            method (str): method to map real vector to a probability vector.
                'softmax': softmax function.
                'sphere': quotient map.
            weight (np.ndarray,torch.Tensor): weight of each dimension.
            requires_grad (bool): whether to track the gradients of the parameters.
            dtype (torch.dtype): data type of the parameters, either torch.float32 or torch.float64
            device (torch.device): device of the parameters.
        '''
        super().__init__()
        assert dim>=2
        assert dtype in {torch.float32,torch.float64}
        assert (batch_size is None) or (batch_size>0)
        assert method in {'softmax','sphere'}
        assert isinstance(device, torch.device)
        tmp0 = (dim,) if (batch_size is None) else (batch_size, dim)
        self.theta = _hf_para(dtype, requires_grad, *tmp0).to(device)
        if weight is not None:
            assert (weight.min()>0) and weight.shape==(dim,)
            self.weight_inv = torch.tensor(1/weight, dtype=dtype, device=device)
        else:
            self.weight_inv = None
        self.dim = int(dim)
        self.dtype = dtype
        self.method = method
        self.batch_size = batch_size

    def forward(self):
        if self.method=='softmax':
            ret = to_discrete_probability_softmax(self.theta)
        else: #sphere
            ret = to_discrete_probability_sphere(self.theta)
        if self.weight_inv is not None:
            ret = ret * self.weight_inv
        return ret

def to_discrete_probability_sphere(theta):
    r'''map real vector to a point on the discrete probability via square of sphere

    Parameters:
        theta (np.ndarray,torch.Tensor): if `ndim>1`, then the last dimension will be expanded to the point
                and the rest dimensions will be batch dimensions.

    Returns:
        ret (np.ndarray,torch.Tensor): array of shape `theta.shape`
    '''
    tmp0 = to_sphere_quotient(theta, is_real=True)
    ret = tmp0*tmp0
    return ret

def to_discrete_probability_softmax(theta):
    r'''map real vector to a probability vector using softmax function

    Parameters:
        theta (np.ndarray,torch.Tensor): if `ndim>1`, then the last dimension will be expanded to the probability vector
                and the rest dimensions will be batch dimensions.

    Returns:
        ret (np.ndarray,torch.Tensor): array of shape `theta.shape`
    '''
    if isinstance(theta, torch.Tensor):
        ret = torch.nn.functional.softmax(theta, dim=-1)
    else:
        ret = scipy.special.softmax(theta, axis=-1)
    return ret

class Stiefel(torch.nn.Module):
    def __init__(self, dim:int, rank:int, batch_size:(int|None)=None, method:str='qr',
                requires_grad:bool=True, dtype:torch.dtype=torch.float64, device:torch.device=_CPU):
        r'''Stiefel manifold

        Parameters:
            dim (int): dimension of the matrix.
            rank (int): rank of the matrix.
            batch_size (int,None): batch size.
            method (str): method to map real vector to a Stiefel matrix.
                'choleskyL': Cholesky decomposition.
                'qr': QR decomposition.
                'sqrtm': square root of a matrix.
                'so-exp': exponential map of special orthogonal group.
                'so-cayley': Cayley transform of special orthogonal group.
            requires_grad (bool): whether to track the gradients of the parameters.
            dtype (torch.dtype): data type of the parameters
                torch.float32 / torch.float64: real Stiefel matrix
                torch.complex64 / torch.complex128: complex Stiefel matrix
            device (torch.device): device of the parameters.
        '''
        super().__init__()
        assert (dim>=2) and (rank>=1) and (rank<=dim)
        assert dtype in {torch.float32,torch.float64,torch.complex64,torch.complex128}
        assert (batch_size is None) or (batch_size>0)
        assert isinstance(device, torch.device)
        # choleskyL is really bad
        assert method in {'choleskyL','qr','so-exp','so-cayley','sqrtm'}
        if method in {'qr','sqrtm'}:
            tmp0 = dim*rank if (dtype in {torch.float32,torch.float64}) else 2*dim*rank
        elif method=='choleskyL':
            tmp0 = (dim*rank-((rank*(rank+1))//2)) * (1 if (dtype in {torch.float32,torch.float64}) else 2)
        elif method in {'so-exp','so-cayley'}: #special orthogonal (SO)
            tmp0 = ((dim*(dim-1))//2) if (dtype in {torch.float32,torch.float64}) else (dim*dim-1)
        tmp1 = (tmp0,) if (batch_size is None) else (batch_size, tmp0)
        tmp2 = torch.float32 if (dtype in {torch.float32,torch.complex64}) else torch.float64
        self.theta = _hf_para(tmp2, requires_grad, *tmp1).to(device)
        self.dim = int(dim)
        self.rank = int(rank)
        self.dtype = dtype
        self.method = method
        self.batch_size = batch_size

    def forward(self):
        if self.method=='choleskyL':
            ret = to_stiefel_choleskyL(self.theta, self.dim, self.rank)
        elif self.method=='qr': #qr
            ret = to_stiefel_qr(self.theta, self.dim, self.rank)
        elif self.method=='sqrtm':
            ret = to_stiefel_sqrtm(self.theta, self.dim, self.rank)
        elif self.method=='so-exp': #so
            ret = to_special_orthogonal_exp(self.theta, self.dim)[...,:self.rank]
        elif self.method=='so-cayley':
            ret = to_special_orthogonal_cayley(self.theta, self.dim)[...,:self.rank]
        return ret

def to_stiefel_sqrtm(theta, dim:int, rank:int):
    r'''map real vector to a Stiefel manifold via square root of a matrix

    Parameters:
        theta (np.ndarray,torch.Tensor): if `ndim>1`, then the last dimension will be expanded to the matrix
                and the rest dimensions will be batch dimensions.
        dim (int): dimension of the matrix.
        rank (int): rank of the matrix.

    Returns:
        ret (np.ndarray,torch.Tensor): array of shape `theta.shape[:-1]+(dim,rank)`
    '''
    assert rank<=dim
    shape = theta.shape
    if shape[-1]==dim*rank: #real
        is_real = True
    else:
        assert shape[-1]==2*dim*rank
        is_real = False
    theta = theta.reshape(-1,shape[-1])
    if isinstance(theta, torch.Tensor):
        if is_real:
            mat = theta.reshape(-1, dim, rank)
        else:
            tmp0 = theta.reshape(-1,2,dim,rank)
            mat = torch.complex(tmp0[:,0], tmp0[:,1])
        if rank==1:
            ret = mat / torch.linalg.norm(mat, axis=1, keepdims=True)
        else:
            tmp0 = torch.linalg.inv(numqi._torch_op.PSDMatrixSqrtm.apply(mat.transpose(1,2).conj() @ mat))
            ret = mat @ tmp0
    else: #numpy
        if is_real:
            mat = theta.reshape(-1, dim, rank)
        else:
            tmp0 = theta.reshape(-1,2,dim,rank)
            mat = tmp0[:,0] + 1j*tmp0[:,1]
        if rank==1:
            ret = mat / np.linalg.norm(mat, axis=1, keepdims=True)
        else:
            # scipy.linalg.sqrtm is slow, so we use np.linalg.eigh here
            EVL,EVC = np.linalg.eigh(mat.transpose(0,2,1).conj() @ mat)
            tmp0 = (EVC*np.sqrt(1/EVL).reshape(-1,1,rank)) @ EVC.transpose(0,2,1).conj()
            ret = mat @ tmp0
    ret = ret.reshape(*shape[:-1], dim, rank)
    return ret

def to_stiefel_choleskyL(theta, dim:int, rank:int):
    r'''map real vector to a Stiefel manifold via Cholesky decomposition

    minimum parameters but bad convergance

    Parameters:
        theta (np.ndarray,torch.Tensor): the last dimension will be expanded to the matrix
                and the rest dimensions will be batch dimensions. For real case, the last dimension
                should be `dim*rank-((rank*(rank+1))//2)`, and for complex case, the last dimension
                should be `2*dim*rank-rank*(rank+1)`.
        dim (int): dimension of the matrix.
        rank (int): rank of the matrix.

    Returns:
        ret (np.ndarray,torch.Tensor): array of shape `theta.shape[:-1]+(dim,rank)`
    '''
    N0 = (rank*(rank+1))//2
    assert rank<=dim
    if theta.shape[-1]==dim*rank-N0:
        is_real = True
    else:
        assert theta.shape[-1]==2*dim*rank-2*N0
        is_real = False
    shape = theta.shape
    theta = theta.reshape(-1, shape[-1])
    N1 = N0 - rank
    N2 = theta.shape[0]
    if isinstance(theta, torch.Tensor):
        device = theta.device
        matL = torch.eye(rank, dtype=theta.dtype, device=device).reshape(1,rank,rank).repeat(N2,1,1)
        indexL = torch.tril_indices(rank,rank,-1)
        matL[:,indexL[0],indexL[1]] = theta[:,:N1]
        if not is_real:
            tmp0 = torch.zeros(N2, rank, rank, dtype=theta.dtype, device=device)
            tmp0[:,indexL[0],indexL[1]] = theta[:,N1:(2*N1)]
            matL = torch.complex(matL, tmp0)
        if rank<dim:
            if is_real:
                matL = torch.concat([matL, theta[:,N1:].reshape(-1,dim-rank,rank)], axis=1)
            else:
                tmp0 = theta[:,(2*N1):].reshape(N2, 2, dim-rank, rank)
                matL = torch.concat([matL, torch.complex(tmp0[:,0], tmp0[:,1])], axis=1)
        tmp0 = torch.linalg.inv(torch.linalg.cholesky_ex(matL.transpose(1,2).conj() @ matL)[0].transpose(1,2).conj())
        ret = matL @ tmp0
    else: #numpy
        matL = np.tile(np.eye(rank, dtype=theta.dtype).reshape(1,rank,rank), (N2,1,1))
        indexL = np.tril_indices(rank,-1)
        matL[:,indexL[0],indexL[1]] = theta[:,:N1]
        if not is_real:
            tmp0 = np.zeros((N2, rank, rank), dtype=theta.dtype)
            tmp0[:,indexL[0],indexL[1]] = theta[:,N1:(2*N1)]
            matL = matL + 1j*tmp0
        if rank<dim:
            if is_real:
                matL = np.concatenate([matL, theta[:,N1:].reshape(-1,dim-rank,rank)], axis=1)
            else:
                tmp0 = theta[:,(2*N1):].reshape(N2,2,dim-rank,rank)
                matL = np.concatenate([matL, tmp0[:,0] + 1j*tmp0[:,1]], axis=1)
        tmp0 = np.linalg.inv(np.linalg.cholesky(matL.transpose(0,2,1).conj() @ matL).transpose(0,2,1).conj())
        ret = matL @ tmp0
    ret = ret.reshape(*shape[:-1], dim, rank)
    return ret

def to_stiefel_qr(theta, dim:int, rank:int):
    r'''map real vector to a Stiefel manifold via QR decomposition

    Parameters:
        theta (np.ndarray,torch.Tensor): the last dimension will be expanded to the matrix
                and the rest dimensions will be batch dimensions. For real case, the last dimension
                should be `dim*rank`, and for complex case, the last dimension should be `2*dim*rank`.
        dim (int): dimension of the matrix.
        rank (int): rank of the matrix.

    Returns:
        ret (np.ndarray,torch.Tensor): array of shape `theta.shape[:-1]+(dim,rank)`
    '''
    assert dim>=rank
    is_torch = isinstance(theta, torch.Tensor)
    shape = theta.shape
    if shape[-1]==dim*rank: #real
        mat = theta.reshape(-1,dim,rank)
    else: #complex
        assert shape[-1]==2*dim*rank
        tmp0 = theta.reshape(-1,2,dim,rank)
        if is_torch:
            mat = torch.complex(tmp0[:,0], tmp0[:,1])
        else:
            mat = tmp0[:,0] + 1j* tmp0[:,1]
    mat = mat.reshape(*shape[:-1], dim, rank)
    # TODO poor efficiency in L-BFGS-B optimization
    # the forward and backward time is still okay compared with matrix-cholesky,
    # but the time for L-BFGS-B optimization is much longer (strange)
    if isinstance(mat, torch.Tensor):
        ret = torch.linalg.qr(mat, mode='reduced')[0]
    else:
        ret = np.linalg.qr(mat, mode='reduced')[0]
    return ret

class SpecialOrthogonal(torch.nn.Module):
    def __init__(self, dim:int, batch_size:(int|None)=None, method:str='exp', cayley_order:int=2,
                    requires_grad:bool=True, dtype:torch.dtype=torch.float64, device:torch.device=_CPU):
        r'''orthogonal matrix

        Parameters:
            dim (int): dimension of the matrix.
            batch_size (int,None): batch size.
            method (str): method to map real vector to an orthogonal matrix.
                'exp': exponential map.
                'cayley': cayley transformation
            requires_grad (bool): whether to track the gradients of the parameters.
            dtype (torch.dtype): data type of the parameters
                torch.float32 / torch.float64: SO(d) manifold,
                torch.complex64 / torch.complex128: SU(d) manifold
            device (torch.device): device of the parameters.
        '''
        super().__init__()
        assert method in {'exp', 'cayley'}
        assert dim>=2
        assert dtype in {torch.float32,torch.float64,torch.complex64,torch.complex128}
        assert cayley_order>=1
        assert isinstance(device, torch.device)
        is_real = dtype in {torch.float32,torch.float64}
        assert (batch_size is None) or (batch_size>0)
        tmp0 = torch.float32 if (dtype in {torch.float32,torch.complex64}) else torch.float64
        tmp1 = (dim*(dim-1)//2) if is_real else dim*dim-1
        tmp2 = (tmp1,) if (batch_size is None) else (batch_size, tmp1)
        self.theta = _hf_para(tmp0, requires_grad, *tmp2).to(device)
        self.dim = int(dim)
        self.method = method
        self.cayley_order = cayley_order
        self.dtype = dtype
        self.batch_size = batch_size

    def forward(self):
        if self.method=='exp':
            ret = to_special_orthogonal_exp(self.theta, self.dim)
        else:
            ret = to_special_orthogonal_cayley(self.theta, self.dim, self.cayley_order)
        return ret

def to_special_orthogonal_exp(theta, dim:int):
    r'''map real vector to a special orthogonal (unitary) manifold via exponential map

    Parameters:
        theta (np.ndarray,torch.Tensor): if `ndim>1`, then the last dimension will be expanded to the matrix
                and the rest dimensions will be batch dimensions.
                special orthogonal matrix: (dim*(dim-1))//2
                special unitary matrix: dim*dim-1
        dim (int): dimension of the matrix.

    Returns:
        ret (np.ndarray,torch.Tensor): array of shape `theta.shape[:-1]+(dim,dim)`
    '''
    assert dim>=2
    N0 = dim*(dim-1)//2
    shape = theta.shape
    if shape[-1]==N0:
        is_real = True
    else:
        assert shape[-1]==(dim*dim-1)
        is_real = False
    theta = theta.reshape(-1, shape[-1])
    N1 = theta.shape[0]
    if isinstance(theta, torch.Tensor):
        device = theta.device
        if is_real:
            tmp0 = torch.zeros(N1, N0, dtype=theta.dtype, device=device)
            tmp1 = torch.zeros(N1, dim, dtype=theta.dtype, device=device)
            mat = numqi.gellmann.gellmann_basis_to_matrix(torch.concat([theta, tmp0, tmp1], axis=1)).imag
        else:
            tmp0 = torch.zeros(N1, 1, dtype=theta.dtype, device=device)
            mat = 1j*numqi.gellmann.gellmann_basis_to_matrix(torch.concat([theta, tmp0], axis=1))
        # batch-version matrix-exp is memory-consuming in pytorch
        ret = torch.stack([torch.linalg.matrix_exp(mat[x]) for x in range(len(mat))])
    else: #numpy
        if is_real:
            tmp0 = np.zeros((N1, N0), dtype=theta.dtype)
            tmp1 = np.zeros((N1, dim), dtype=theta.dtype)
            mat = numqi.gellmann.gellmann_basis_to_matrix(np.concatenate([theta, tmp0, tmp1], axis=1)).imag
        else:
            tmp0 = np.zeros((N1, 1), dtype=theta.dtype)
            mat = 1j*numqi.gellmann.gellmann_basis_to_matrix(np.concatenate([theta, tmp0], axis=1))
        ret = np.stack([scipy.linalg.expm(x) for x in mat])
        # ret = scipy.linalg.expm(mat) #TODO scipy-v1.9 https://github.com/scipy/scipy/issues/12838
    ret = ret.reshape(*shape[:-1], dim, dim)
    return ret

def to_special_orthogonal_cayley(theta, dim:int, order:int=2):
    r'''map real vector to a special orthogonal (unitary) manifold via Cayley transform

    [wiki/Cayley-transform](https://en.wikipedia.org/wiki/Cayley_transform)

    Parameters:
        theta (np.ndarray,torch.Tensor): if `ndim>1`, then the last dimension will be expanded to the matrix
                and the rest dimensions will be batch dimensions.
                special orthogonal matrix: (dim*(dim-1))//2
                special unitary matrix: dim*dim-1
        dim (int): dimension of the matrix.
        order (int): order of the Cayley transformation.

    Returns:
        ret (np.ndarray,torch.Tensor): array of shape `theta.shape[:-1]+(dim,dim)`
    '''
    assert dim>=2
    assert order>=1
    N0 = dim*(dim-1)//2
    shape = theta.shape
    if shape[-1]==N0:
        is_real = True
    else:
        assert shape[-1]==(dim*dim-1)
        is_real = False
    theta = theta.reshape(-1, shape[-1])
    N1 = theta.shape[0]
    if isinstance(theta, torch.Tensor):
        device = theta.device
        if is_real:
            tmp0 = torch.zeros(N1, N0, dtype=theta.dtype, device=device)
            tmp1 = torch.zeros(N1, dim, dtype=theta.dtype, device=device)
            mat = numqi.gellmann.gellmann_basis_to_matrix(torch.concat([theta, tmp0, tmp1], axis=1)).imag
        else:
            tmp0 = torch.zeros(N1, 1, dtype=theta.dtype, device=device)
            mat = 1j*numqi.gellmann.gellmann_basis_to_matrix(torch.concat([theta, tmp0], axis=1))
        tmp0 = torch.eye(dim, dtype=theta.dtype, device=device)
        tmp1 = torch.linalg.inv(tmp0+mat) @ (tmp0-mat)
        ret = tmp1
        for _ in range(order-1):
            ret = ret @ tmp1
    else: #numpy
        if is_real:
            tmp0 = np.zeros((N1, N0), dtype=theta.dtype)
            tmp1 = np.zeros((N1, dim), dtype=theta.dtype)
            mat = numqi.gellmann.gellmann_basis_to_matrix(np.concatenate([theta, tmp0, tmp1], axis=1)).imag
        else:
            tmp0 = np.zeros((N1, 1), dtype=theta.dtype)
            mat = 1j*numqi.gellmann.gellmann_basis_to_matrix(np.concatenate([theta, tmp0], axis=1))
        tmp0 = np.eye(dim, dtype=theta.dtype)
        tmp1 = np.linalg.inv(tmp0+mat) @ (tmp0-mat)
        ret = tmp1
        for _ in range(order-1):
            ret = ret @ tmp1
    ret = ret.reshape(*shape[:-1], dim, dim)
    return ret

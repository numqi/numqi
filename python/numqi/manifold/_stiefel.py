import numpy as np
import torch

from ._internal import _CPU, _hf_para
from ._internal import to_special_orthogonal_exp, to_special_orthogonal_cayley
import numqi._torch_op


class Stiefel(torch.nn.Module):
    def __init__(self, dim:int, rank:int, batch_size:(int|None)=None, method:str='polar',
                requires_grad:bool=True, dtype:torch.dtype=torch.float64, device:torch.device=_CPU):
        r'''Stiefel manifold

        Parameters:
            dim (int): dimension of the matrix.
            rank (int): rank of the matrix.
            batch_size (int,None): batch size.
            method (str): method to map real vector to a Stiefel matrix.
                'choleskyL': Cholesky decomposition.
                'euler': Euler-Hurwitz angles.
                'qr': QR decomposition.
                'polar': square root of a matrix.
                'so-exp': exponential map of special orthogonal group.
                'so-cayley': Cayley transform of special orthogonal group.
            requires_grad (bool): whether to track the gradients of the parameters.
            dtype (torch.dtype): data type of the parameters
                torch.float32 / torch.float64: real Stiefel matrix
                torch.complex64 / torch.complex128: complex Stiefel matrix
            device (torch.device): device of the parameters.
        '''
        # TODO polar decomposition https://openreview.net/forum?id=5mtwoRNzjm
        super().__init__()
        assert (dim>=2) and (rank>=1) and (rank<=dim)
        assert dtype in {torch.float32,torch.float64,torch.complex64,torch.complex128}
        assert (batch_size is None) or (batch_size>0)
        assert isinstance(device, torch.device)
        # choleskyL is really bad
        assert method!='euler', '[TODO] euler is not correctly implemented' #TODO
        assert method in {'choleskyL','qr','so-exp','so-cayley','polar','euler'}
        if method in {'qr','polar'}:
            tmp0 = dim*rank if (dtype in {torch.float32,torch.float64}) else 2*dim*rank
        elif method=='choleskyL':
            tmp0 = (dim*rank-((rank*(rank+1))//2)) * (1 if (dtype in {torch.float32,torch.float64}) else 2)
        elif method in {'so-exp','so-cayley'}: #special orthogonal (SO)
            tmp0 = ((dim*(dim-1))//2) if (dtype in {torch.float32,torch.float64}) else (dim*dim-1)
        else: #euler
            tmp0 = (dim*rank-rank*(rank+1)//2) if (dtype in {torch.float32,torch.float64}) else (2*dim*rank-rank*(rank+1))
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
        elif self.method=='polar':
            ret = to_stiefel_polar(self.theta, self.dim, self.rank)
        elif self.method=='so-exp': #so
            ret = to_special_orthogonal_exp(self.theta, self.dim)[...,:self.rank]
        elif self.method=='so-cayley':
            ret = to_special_orthogonal_cayley(self.theta, self.dim)[...,:self.rank]
        else: #euler
            ret = to_stiefel_euler(self.theta, self.dim, self.rank)
        return ret

def to_stiefel_polar(theta, dim:int, rank:int):
    r'''map real vector to a Stiefel manifold via polar decomposition

    [wiki-link](https://en.wikipedia.org/wiki/Stiefel_manifold)

    $$ \left\{ x\in\mathbb{K}^{m\times n}:x^\dagger x=I_n \right\}$$

    where $\mathbb{K}=\mathbb{R}$ or $\mathbb{C}$. Matrix square root is used so the backward might be
    several times slower than the forward. QR decomposition

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
    # TODO poor efficiency in L-BFGS-B optimization
    # the forward and backward time is still okay compared with matrix-square-root,
    # but the time for L-BFGS-B optimization is much longer (strange)
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


def _to_stiefel_euler_real(theta, dim, rank):
    batch = theta.shape[0]
    tmp0 = np.cumsum(np.arange(dim-rank, dim)).tolist()
    theta_list = [theta[:,x:y] for x,y in zip([0]+tmp0,tmp0)]
    ret = None
    if isinstance(theta, torch.Tensor):
        for theta_i in theta_list:
            N0 = theta_i.shape[1]
            ct = torch.cos(theta_i)
            st = torch.sin(theta_i)
            cum_st = torch.cumprod(st, dim=1)
            rowJ = torch.concat([ct[:,:1], ct[:,1:]*cum_st[:,:-1], cum_st[:,-1:]], dim=1).reshape(batch,N0+1,1)
            if ret is None:
                ret = rowJ
            else:
                zi = []
                tmp0 = 0*ret[:,0], ret[:,0]
                for indI in range(N0):
                    zi.append(ct[:,indI]*tmp0[0] - st[:,indI]*tmp0[1])
                    tmp1 = ct[:,indI]*tmp0[1] + st[:,indI]*tmp0[0]
                    if indI+1 < N0:
                        tmp0 = tmp1, ret[:,indI+1]
                    else:
                        zi.append(tmp1)
                ret = torch.concat([rowJ, torch.stack(zi, dim=1)], dim=2)
    else:
        for theta_i in theta_list:
            N0 = theta_i.shape[1]
            ct = np.cos(theta_i)
            st = np.sin(theta_i)
            cum_st = np.cumprod(st, axis=1)
            rowJ = np.concatenate([ct[:,:1], ct[:,1:]*cum_st[:,:-1], cum_st[:,-1:]], axis=1).reshape(batch,N0+1,1)
            if ret is None:
                ret = rowJ
            else:
                zi = []
                tmp0 = 0*ret[:,0], ret[:,0]
                for indI in range(N0):
                    zi.append(ct[:,indI]*tmp0[0] - st[:,indI]*tmp0[1])
                    tmp1 = ct[:,indI]*tmp0[1] + st[:,indI]*tmp0[0]
                    if indI+1 < N0:
                        tmp0 = tmp1, ret[:,indI+1]
                    else:
                        zi.append(tmp1)
                ret = np.concatenate([rowJ, np.stack(zi, axis=1)], axis=2)
    return ret

def _to_stiefel_euler_complex(theta, dim, rank):
    # TODO add phase
    # TODO manually backward
    batch = theta.shape[0]
    theta = theta.reshape(batch, -1, 2)
    tmp0 = np.cumsum(np.arange(dim-rank, dim)).tolist()
    theta_list = [(theta[:,x:y,0],theta[:,x:y,1]) for x,y in zip([0]+tmp0,tmp0)]
    ret = None
    if isinstance(theta, torch.Tensor):
        for theta_i,phi_i in theta_list:
            N0 = theta_i.shape[1]
            tmp0 = phi_i[:,:1]*0
            cum_expp = torch.exp(1j*(torch.cumsum(torch.concat([tmp0, phi_i], dim=1), dim=1) - torch.concat([phi_i, tmp0], dim=1)))
            expp = torch.exp(1j*phi_i)
            ct = torch.cos(theta_i)
            st = torch.sin(theta_i)
            cum_st = torch.cumprod(st, dim=1)
            rowJ = (torch.concat([ct[:,:1], ct[:,1:]*cum_st[:,:-1], cum_st[:,-1:]],dim=1)*cum_expp).reshape(batch,N0+1,1)
            if ret is None:
                ret = rowJ
            else:
                zi = []
                tmp0 = 0*ret[:,0], ret[:,0]
                for indI in range(N0):
                    zi.append((ct[:,indI]/expp[:,indI])*tmp0[0] - (st[:,indI]/expp[:,indI])*tmp0[1])
                    tmp1 = (ct[:,indI]*expp[:,indI])*tmp0[1] + (st[:,indI]*expp[:,indI])*tmp0[0]
                    if indI+1 < N0:
                        tmp0 = tmp1, ret[:,indI+1]
                    else:
                        zi.append(tmp1)
                ret = torch.concat([rowJ, torch.stack(zi, dim=1)], dim=2)
    else:
        for theta_i,phi_i in theta_list:
            N0 = theta_i.shape[1]
            tmp0 = np.zeros((batch,1))
            cum_expp = np.exp(1j*(np.cumsum(np.concatenate([tmp0, phi_i], axis=1), axis=1) - np.concatenate([phi_i, tmp0], axis=1)))
            expp = np.exp(1j*phi_i)
            ct = np.cos(theta_i)
            st = np.sin(theta_i)
            cum_st = np.cumprod(st, axis=1)
            rowJ = (np.concatenate([ct[:,:1], ct[:,1:]*cum_st[:,:-1], cum_st[:,-1:]],axis=1)*cum_expp).reshape(batch,N0+1,1)
            if ret is None:
                ret = rowJ
            else:
                zi = []
                tmp0 = 0*ret[:,0], ret[:,0]
                for indI in range(N0):
                    zi.append((ct[:,indI]/expp[:,indI])*tmp0[0] - (st[:,indI]/expp[:,indI])*tmp0[1])
                    tmp1 = (ct[:,indI]*expp[:,indI])*tmp0[1] + (st[:,indI]*expp[:,indI])*tmp0[0]
                    if indI+1 < N0:
                        tmp0 = tmp1, ret[:,indI+1]
                    else:
                        zi.append(tmp1)
                ret = np.concatenate([rowJ, np.stack(zi, axis=1)], axis=2)
    return ret

def to_stiefel_euler(theta:np.ndarray|torch.Tensor, dim:int, rank:int):
    r'''map real vector to a Stiefel manifold via Euler-Hurwitz angles

    Numerical evaluation of convex-roof entanglement measures with applications to spin rings
    [doi-link](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.80.042301)

    real Stiefel: theta [0,pi/2]

    complex Stiefel: theta [0,pi/2], phi (-pi,pi)

    Parameters:
        theta (np.ndarray,torch.Tensor): if `ndim>1`, then the last dimension will be expanded to the matrix
                and the rest dimensions will be batch dimensions.
        dim (int): dimension of the matrix.
        rank (int): rank of the matrix.

    Returns:
        ret (np.ndarray,torch.Tensor): array of shape `theta.shape[:-1]+(dim,rank)`
    '''
    assert (theta.ndim==1) or (theta.ndim==2)
    shape = theta.shape
    if theta.ndim==1:
        theta = theta.reshape(1, -1)
    else:
        theta = theta.reshape(-1, theta.shape[-1])
    N0 = dim*rank - rank*(rank+1)//2
    assert (theta.shape[1]==N0) or (theta.shape[1]==2*N0)
    if theta.shape[1]==N0:
        ret = _to_stiefel_euler_real(theta, dim, rank)
    else:
        ret = _to_stiefel_euler_complex(theta, dim, rank)
    ret = ret.reshape(shape[:-1] + (dim,rank))
    return ret


def from_stiefel_euler(np0:np.ndarray, zero_eps:float=1e-10):
    r'''map a Stiefel manifold to real vector via Euler-Hurwitz angles

    Numerical evaluation of convex-roof entanglement measures with applications to spin rings
    [doi-link](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.80.042301)

    complex Stiefel ordering: [theta,phi,theta,phi,...]

    Parameters:
        np0 (np.ndarray): array of shape (dim,rank)
        zero_eps (float): small number to avoid division by zero.

    Returns:
        ret (np.ndarray): array of shape (N0,) where N0=dim*rank-rank*(rank+1)//2 for real Stiefel,
            and N0=2*dim*rank-rank*(rank+1) for complex Stiefel.
    '''
    assert isinstance(np0, np.ndarray) and (np0.ndim==2)
    dim,rank = np0.shape
    isreal = not np.iscomplexobj(np0)
    ret = []
    np1 = np0.copy()
    for indJ in range(rank):
        for indI in range(dim-1,indJ,-1):
            x = np1[indI-1,indJ]
            y = np1[indI,indJ]
            if isreal:
                theta = np.arctan2(y, x) if (abs(x) > zero_eps) else np.pi/2
                ct = np.cos(theta)
                st = np.sin(theta)
                tmp0 = np.array([[ct, st], [-st, ct]])
                ret.append(theta)
            else:
                rx = np.abs(x)
                if rx>zero_eps:
                    tmp0 = y / x
                    theta = np.arctan(np.abs(tmp0))
                    phi = np.angle(tmp0)/2
                else:
                    theta = np.pi/2
                    phi = 0
                ct = np.cos(theta)
                st = np.sin(theta)
                expp = np.exp(1j*phi)
                tmp0 = np.array([[ct*expp, st/expp], [-st*expp, ct/expp]])
                ret.append(theta)
                ret.append(phi)
            np1[(indI-1):(indI+1)] = tmp0 @ np1[(indI-1):(indI+1)]
    if isreal:
        ret = np.array(ret[::-1])
    else:
        ret = np.array(ret).reshape(-1,2)[::-1].reshape(-1)
    # sign = np.sign(np.diag(np1))
    # for real Stiefel, seems sign are always 1, we don't see negative sign
    phase = np.angle(np.diag(np1))
    return ret,phase

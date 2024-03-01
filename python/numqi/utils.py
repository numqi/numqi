import functools
import collections
import numpy as np
import torch

import numqi._torch_op

@functools.lru_cache(maxsize=128)
def _hf_num_state_to_num_qubit_hf0(num_state:int, kind:str):
    if kind=='exact':
        ret:int = round(float(np.log2(num_state)))
        assert 2**ret==num_state
    elif kind=='ceil':
        ret = int(np.ceil(np.log2(num_state)))
    else: #floor
        ret = int(np.floor(np.log2(num_state)))
    return ret


def hf_num_state_to_num_qubit(num_state:int, kind:str='exact'):
    r'''convert the number of states to the number of qubits

    Parameters:
        num_state (int): number of states
        kind (str): 'exact', 'ceil', 'floor'

    Returns:
        ret(int): number of qubits
    '''
    num_state = int(num_state)
    kind = str(kind)
    assert (num_state>0) and (kind in {'exact','ceil','floor'})
    ret = _hf_num_state_to_num_qubit_hf0(num_state, kind)
    return ret

def hf_tuple_of_any(x, type_=None):
    hf0 = lambda x: x if (type_ is None) else type_(x)
    if isinstance(x,collections.abc.Iterable):
        if isinstance(x, np.ndarray):
            ret = [hf0(y) for y in np.nditer(x)]
        else:
            # error when x is np.array(0)
            ret = tuple(hf0(y) for y in x)
    else:
        ret = hf0(x),
    return ret

hf_tuple_of_int = lambda x: hf_tuple_of_any(x, type_=int)


def hf_complex_to_real(x):
    r'''convert a complex matrix to a real matrix

    `complex -> [[real,-imag], [imag,real]]`

    $$ A\in\mathbb{C}^{m\times n}\mapsto\begin{bmatrix} \Re[A] & -\Im[A]\\ \Im[A] & \Re[A] \end{bmatrix}\in\mathbb{R}^{2m\times2n} $$

    Parameters:
        x (np.ndarray,torch.Tensor): a complex matrix, shape=(...,dim0,dim1), support batch

    Returns:
        ret (np.ndarray,torch.Tensor): a real matrix, shape=(...,2*dim0,2*dim1)
    '''
    dim0,dim1 = x.shape[-2:]
    shape = x.shape[:-2]
    x = x.reshape(-1, dim0, dim1)
    # ret = np.block([[x.real,-x.imag],[x.imag,x.real]])
    if isinstance(x, torch.Tensor):
        tmp0 = torch.concat([x.real, -x.imag], dim=2)
        tmp1 = torch.concat([x.imag, x.real], dim=2)
        ret = torch.concat([tmp0,tmp1], dim=1)
    else:
        ret = np.block([[x.real,-x.imag],[x.imag,x.real]])
    ret = ret.reshape(shape+(2*dim0,2*dim1))
    return ret


def hf_real_to_complex(x):
    r'''convert a real matrix to a complex matrix

    `[[real,-imag], [imag,real]] -> complex`

    Parameters:
        x (np.ndarray,torch.Tensor): a real matrix, shape=(...,2*dim0,2*dim1), support batch

    Returns:
        ret (np.ndarray,torch.Tensor): a complex matrix, shape=(...,dim0,dim1)
    '''
    assert (x.shape[-2]%2==0) and (x.shape[-1]%2==0)
    dim0 = x.shape[-2]//2
    dim1 = x.shape[-1]//2
    ret = x[...,:dim0,:dim1] + 1j*x[...,dim0:,:dim1]
    return ret


# def state_to_dm(ket):
#     ret = ket[:,np.newaxis] * ket.conj()
#     return ret


def partial_trace(rho:np.ndarray, dim:tuple[int], keep_index:set[int]):
    r'''partial trace of a density matrix

    Parameters:
        rho (np.ndarray): a density matrix, shape=(*dim,*dim)
        dim (tuple[int]): shape of the density matrix
        keep_index (set[int]): the indices to keep

    Returns:
        ret (np.ndarray): the partial trace of the density matrix, shape=(*dim[keep_index], *dim[keep_index])
    '''
    if not isinstance(keep_index, collections.abc.Iterable):
        keep_index = {int(keep_index)}
    N0 = len(dim)
    keep_index = sorted(set(keep_index))
    rho = rho.reshape(*dim, *dim)
    assert all(0<=x<N0 for x in keep_index)
    tmp0 = list(range(N0))
    tmp1 = list(range(N0,2*N0))
    tmp2 = set(range(N0))-set(keep_index)
    for x in tmp2:
        tmp1[x] = x
    tmp3 = list(keep_index) + [x+N0 for x in keep_index]
    N1 = np.prod([dim[x] for x in keep_index])
    ret = np.einsum(rho, tmp0+tmp1, tmp3, optimize=True).reshape(N1, N1)
    return ret


def get_fidelity(rho0:np.ndarray|torch.Tensor, rho1:np.ndarray|torch.Tensor):
    r'''get the fidelity of two density matrices or pure states

    Parameters:
        rho0 (np.ndarray,torch.Tensor): pure state (ndim=1) or density matrix (ndim=2)
        rho1 (np.ndarray,torch.Tensor): pure state or density matrix

    Returns:
        ret (float,torch.Tensor): the fidelity of the two states
    '''
    ndim0 = rho0.ndim
    ndim1 = rho1.ndim
    assert (ndim0 in {1,2}) and (ndim1 in {1,2})
    if isinstance(rho0, torch.Tensor):
        if ndim0==1 and ndim1==1:
            ret = torch.abs(torch.vdot(rho0, rho1))**2
        elif ndim0==1 and ndim1==2:
            ret = torch.vdot(rho0, rho1 @ rho0).real
        elif ndim0==2 and ndim1==1:
            ret = torch.vdot(rho1, rho0 @ rho1).real
        else:
            EVL0,EVC0 = torch.linalg.eigh(rho0)
            zero = torch.tensor(0.0, device=rho0.device)
            tmp0 = torch.sqrt(torch.maximum(zero, EVL0))
            tmp1 = (tmp0.reshape(-1,1) * EVC0.T.conj()) @ rho1 @ (EVC0 * tmp0)
            tmp2 = torch.linalg.eigvalsh(tmp1)
            ret = torch.sum(torch.sqrt(torch.maximum(zero, tmp2)))**2
    else:
        if ndim0==1 and ndim1==1:
            ret = abs(np.vdot(rho0, rho1))**2
        elif ndim0==1 and ndim1==2:
            ret = np.vdot(rho0, rho1 @ rho0).real.item()
        elif ndim0==2 and ndim1==1:
            ret = np.vdot(rho1, rho0 @ rho1).real.item()
        else:
            EVL0,EVC0 = np.linalg.eigh(rho0)
            tmp0 = np.sqrt(np.maximum(0, EVL0))
            tmp1 = (tmp0[:,np.newaxis] * EVC0.T.conj()) @ rho1 @ (EVC0 * tmp0)
            tmp2 = np.linalg.eigvalsh(tmp1)
            ret = np.sum(np.sqrt(np.maximum(0, tmp2)))**2
    return ret


def get_von_neumann_entropy(rho:np.ndarray|torch.Tensor, _torch_logm:str|tuple='eigen'):
    r'''get the von Neumann entropy of a density matrix
    [wiki-link](https://en.wikipedia.org/wiki/Von_Neumann_entropy)

    Parameters:
        rho (np.ndarray,torch.Tensor): a density matrix, shape=(dim,dim)
        _torch_logm (str,tuple): 'eigen' or ('pade',num_sqrtm,pade_order), 'pade' is used only when requires_grad

    Returns:
        ret (float): the von Neumann entropy of the density matrix
    '''
    shape = rho.shape
    assert (len(shape)>=2) and (shape[-1]==shape[-2])
    dim = shape[-1]
    rho = rho.reshape(-1, dim, dim)
    if isinstance(rho, torch.Tensor):
        if rho.requires_grad and (_torch_logm!='eigen'):
            op_logm = numqi._torch_op.get_PSDMatrixLogm(int(_torch_logm[1]), int(_torch_logm[2]))
            log_rho = op_logm(rho)
            ret = -torch.einsum(rho.reshape(-1,dim*dim).conj(), [0,1], log_rho.reshape(-1,dim*dim), [0,1], [0]).real
            # ret = -torch.vdot(rho.reshape(-1), log_rho.reshape(-1)).real
        else:
            eps = torch.tensor(torch.finfo(rho.dtype).eps)
            EVL = torch.maximum(torch.linalg.eigvalsh(rho), eps)
            ret = -torch.einsum(EVL, [0,1], torch.log(EVL), [0,1], [0])
            # ret = -torch.dot(EVL, torch.log(EVL))
    else:
        EVL = np.maximum(np.linalg.eigvalsh(rho), np.finfo(rho.dtype).eps)
        ret = - np.einsum(EVL, [0,1], np.log(EVL), [0,1], [0], optimize=True)
        # ret = -np.dot(EVL, np.log(EVL))
    if len(shape)==2:
        ret = ret[0]
    else:
        ret = ret.reshape(shape[:-2])
    return ret


def get_Renyi_entropy(rho:np.ndarray|torch.Tensor, alpha:float):
    r'''get the Renyi entropy of a density matrix

    Parameters:
        rho (np.ndarray,torch.Tensor): a density matrix, shape=(dim,dim)
        alpha (float): the order of the Renyi entropy

    Returns:
        ret (float): the Renyi entropy of the density matrix
    '''
    assert (alpha!=1) and (alpha>0)
    if isinstance(rho, torch.Tensor):
        EVL = torch.linalg.eigvalsh(rho)
        ret = torch.log((EVL**alpha).sum()) / (1-alpha)
    else:
        EVL = np.linalg.eigvalsh(rho)
        ret = np.log((EVL**alpha).sum()) / (1-alpha)
    return ret


def get_purification(rho:np.ndarray, dimR:int|None=None, seed=None):
    r'''get the purification of a density matrix.
    all purification are connected by Stiefel manifold

    Parameters:
        rho (np.ndarray): a density matrix, shape=(dim,dim)
        dimR (int,None): the dimension of the purification, dimR>=dim, if None, dimR=dim
        seed (int,None): random seed

    Returns:
        ret (np.ndarray): a purification of the density matrix, shape=(dim,dimR)
    '''
    assert (rho.ndim==2) and (np.abs(rho-rho.T.conj()).max()<1e-10)
    assert abs(np.trace(rho)-1).max() < 1e-10
    ret = np.linalg.cholesky(rho)
    if dimR is not None:
        dim = rho.shape[0]
        assert dimR>=dim
        np_rng = np.random.default_rng(seed)
        tmp0 = np_rng.normal(size=(dimR*2*dim))
        tmp1 = numqi.manifold.to_stiefel_qr(tmp0, dim=dimR, rank=dim)
        ret = ret @ tmp1.T
    return ret


def get_trace_distance(rho:np.ndarray, sigma:np.ndarray):
    r'''get the trace distance of two density matrices.
    abs is not a good choice for loss function, so no torch-version

    Parameters:
        rho (np.ndarray): a density matrix, shape=(dim,dim)
        sigma (np.ndarray): a density matrix, shape=(dim,dim)

    Returns:
        ret (float): the trace distance of the density matrices
    '''
    tmp0 = rho - sigma
    assert (tmp0.ndim==2) and (np.abs(tmp0-tmp0.T.conj()).max()<1e-10)
    ret = np.abs(np.linalg.eigvalsh(tmp0)).sum() / 2
    return ret


def get_purity(rho):
    r'''get the purity of a density matrix

    Parameters:
        rho (np.ndarray,torch.Tensor): a density matrix, shape=(dim,dim)

    Returns:
        ret (float): the purity of the density matrix
    '''
    assert (rho.ndim==2) and (rho.shape[0]==rho.shape[1])
    # ret = np.trace(rho @ rho).real
    tmp0 = rho.reshape(-1)
    if isinstance(rho, torch.Tensor):
        ret = torch.vdot(tmp0, tmp0)
    else:
        ret = np.vdot(tmp0, tmp0)
    assert abs(ret.imag.item()) < 1e-10
    return ret.real


def _get_psd_logm(mat, method):
    if isinstance(mat, torch.Tensor):
        eps = torch.tensor(torch.finfo(mat.dtype).eps)
        if method=='eigen':
            EVL,EVC = torch.linalg.eigh(mat)
            ret = (EVC * torch.log(torch.maximum(eps, EVL))) @ EVC.T.conj()
        else:
            assert (len(method)==3) and (method[0]=='pade')
            logm_op = numqi._torch_op.get_PSDMatrixLogm(int(method[1]), int(method[2]))
            ret = logm_op(mat)
    else: #numpy
        eps = np.finfo(mat.dtype).eps
        EVL,EVC = np.linalg.eigh(mat)
        ret = (EVC * np.log(np.maximum(eps, EVL))) @ EVC.T.conj()
    return ret


def get_relative_entropy(rho, sigma, tr_rho_log_rho:float=None, _torch_logm=('pade',6,8)):
    r'''get the relative entropy of two density matrices
    [wiki-link](https://en.wikipedia.org/wiki/Quantum_relative_entropy)

    $$ S(\rho,\sigma) = \mathrm{Tr}(\rho \log\rho - \rho \log\sigma) $$

    Parameters:
        rho (np.ndarray,torch.Tensor): a density matrix, shape=(dim,dim)
        sigma (np.ndarray,torch.Tensor): a density matrix, shape=(dim,dim)
        tr_rho_log_rho (float,None): tr(rho log(rho)), if None, calculate it
        _torch_logm (str,tuple): 'eigen' or ('pade',num_sqrtm,pade_order), 'pade' is used only when requires_grad

    Returns:
        ret (float,torch.Tensor): the relative entropy of the density matrices
    '''
    is_torch = isinstance(rho, torch.Tensor)
    if is_torch:
        eps = torch.tensor(torch.finfo(rho.dtype).eps)
        assert (_torch_logm=='eigen') or ((len(_torch_logm)==3) and (_torch_logm[0]=='pade'))
        if (rho.requires_grad or sigma.requires_grad) and (_torch_logm!='eigen'):
            tmp0 = numqi._torch_op.get_PSDMatrixLogm(int(_torch_logm[1]), int(_torch_logm[2]))
            log_sigma = tmp0(sigma)
        else:
            EVL,EVC = torch.linalg.eigh(sigma)
            log_sigma = (EVC * torch.log(torch.maximum(eps, EVL))) @ EVC.T.conj()
        ret = - torch.vdot(rho.reshape(-1), log_sigma.reshape(-1)).real
        if tr_rho_log_rho is None:
            EVL = torch.maximum(eps, torch.linalg.eigvalsh(rho))
            ret = ret + torch.dot(EVL, torch.log(EVL))
        else:
            ret = tr_rho_log_rho + ret
    else: #numpy
        eps = np.finfo(rho.dtype).eps
        EVL,EVC = np.linalg.eigh(sigma)
        log_sigma = (EVC * np.log(np.maximum(eps, EVL))) @ EVC.T.conj()
        ret = - np.vdot(rho.reshape(-1), log_sigma.reshape(-1)).real
        if tr_rho_log_rho is None:
            EVL = np.maximum(eps, np.linalg.eigvalsh(rho))
            ret = ret + np.dot(EVL, np.log(EVL))
        else:
            ret = tr_rho_log_rho + ret
    return ret

def get_tetrahedron_POVM(num_qubit:int=1):
    r'''Tetrahedron POVM

    [wiki-link](https://en.wikipedia.org/wiki/SIC-POVM)

    Parameters:
        num_qubit (int): number of qubits

    Returns:
        ret (np.ndarray): shape=(N, m, m) where `N=4**num_qubit`, `m=2**num_qubit`
    '''
    a = np.sqrt(2)/3
    b = np.sqrt(2/3)
    vec = 1/4 * np.array([[1,0,0,1], [1,2*a,0,-1/3], [1,-a,b,-1/3], [1,-a,-b,-1/3]])
    tmp0 = np.array([[1,0,0,1], [0,1,1,0], [0,-1j,1j,0], [1,0,0,-1]]).reshape(4,2,2)
    mat = np.einsum(vec, [0,1], tmp0, [1,2,3], [0,2,3], optimize=True)
    ret = mat
    for _ in range(num_qubit-1):
        ret = np.einsum(ret, [0,1,2], mat, [3,4,5], [0,3,1,4,2,5], optimize=True).reshape(-1, ret.shape[1]*2, ret.shape[2]*2)
    return ret


def is_positive_semi_definite(np0:np.ndarray, shift:float=0.0, hermitian_eps:float|None=None):
    '''check whether a matrix is positive semi-definite.
    Cholesky decomposition is used to check the positive semi-definite property.

    Determining whether a symmetric matrix is positive-definite
    [stackexchange-link](https://math.stackexchange.com/a/13311)

    A practical way to check if a matrix is positive-definite
    [stackexchange-link](https://math.stackexchange.com/a/87538)

    Parameters:
        np0 (np.ndarray): matrix, must be Hermitian
        shift (float): shift the matrix by a scalar, e.g. `shift=1e-10` or `shift=-1e-10`
        hermitian_eps (float|None): threshold for the Hermitian property, raise AssertionError if fail. if `None`, then skip the check

    Returns:
        tag (bool): whether the matrix is positive semi-definite
    '''
    if hermitian_eps is not None:
        assert np.abs(np0-np0.T.conj()).max() < hermitian_eps
    if shift!=0:
        np0 = np0 + shift*np.eye(np0.shape[0])
    try:
        np.linalg.cholesky(np0)
        ret = True
    except np.linalg.LinAlgError:
        ret = False
    return ret

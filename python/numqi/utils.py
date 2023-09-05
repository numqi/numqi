import functools
import collections
import numpy as np
import scipy.linalg
import torch

from ._torch_op import TorchMatrixLogm

@functools.lru_cache(maxsize=128)
def hf_num_state_to_num_qubit(num_state:int, kind:str='exact'):
    assert kind in {'exact','ceil','floor'}
    if kind=='exact':
        ret = round(float(np.log2(num_state)))
        assert abs(2**ret-num_state)<1e-7
    elif kind=='ceil':
        ret = int(np.ceil(np.log2(num_state)))
    else: #floor
        ret = int(np.floor(np.log2(num_state)))
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
    assert (x.shape[-2]%2==0) and (x.shape[-1]%2==0)
    dim0 = x.shape[-2]//2
    dim1 = x.shape[-1]//2
    ret = x[...,:dim0,:dim1] + 1j*x[...,dim0:,:dim1]
    return ret


# def state_to_dm(ket):
#     ret = ket[:,np.newaxis] * ket.conj()
#     return ret


def partial_trace(rho, dim, keep_index):
    if not isinstance(keep_index, collections.abc.Iterable):
        keep_index = [keep_index]
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


def get_fidelity(rho0, rho1):
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
            # sqrt_rho = TorchMatrixSqrtm.apply(rho)
            # ret = torch.trace(TorchMatrixSqrtm.apply(sqrt_rho @ sigma @ sqrt_rho))
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


def get_purity(rho):
    # ret = np.trace(rho @ rho).real
    ret = np.dot(rho.reshape(-1), rho.reshape(-1).conj()).real
    return ret


_ree_op_torch_logm = TorchMatrixLogm(num_sqrtm=6, pade_order=8)

def get_relative_entropy(rho0, rho1, kind='error', zero_tol=1e-5):
    is_torch = isinstance(rho0, torch.Tensor)
    if is_torch:
        tmp0 = _ree_op_torch_logm(rho0)
        tmp1 = _ree_op_torch_logm(rho1)
        ret = torch.trace(rho0 @ tmp0) - torch.trace(rho0 @ tmp1)
    else:
        tmp0 = scipy.linalg.logm(rho0)
        tmp1 = scipy.linalg.logm(rho1)
        ret = np.trace(rho0 @ tmp0) - np.trace(rho0 @ tmp1)
    if abs(ret.imag.item())>zero_tol:
        assert kind in {'error', 'infinity', 'ignore'}
        if kind=='error':
            raise ValueError('quantum-relative-entropy be infinty')
        elif kind=='infinity':
            ret = torch.inf if is_torch else np.inf
    ret = ret.real
    return ret

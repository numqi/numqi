import numpy as np
import scipy.linalg

try:
    import torch
except ImportError:
    torch = None

from .utils import is_torch
from .gate import pauli

def hf_dephasing_kraus_op(noise_rate):
    ret = [
        np.sqrt(1-noise_rate)*np.eye(2),
        np.sqrt(noise_rate)*pauli.sz,
    ]
    return ret


def hf_depolarizing_kraus_op(noise_rate):
    ret = [
        np.sqrt(1-3*noise_rate/4)*np.eye(2),
        np.sqrt(noise_rate/4)*pauli.sx,
        np.sqrt(noise_rate/4)*pauli.sy,
        np.sqrt(noise_rate/4)*pauli.sz,
    ]
    return ret


def hf_amplitude_damping_kraus_op(noise_rate):
    ret = [
        np.array([[1,0], [0,np.sqrt(1-noise_rate)]]),
        np.array([[0,np.sqrt(noise_rate)], [0,0]]),
    ]
    return ret


def apply_kraus_op(rho, op):
    ret = sum(x@rho@x.T.conj() for x in op)
    return ret


def kraus_op_to_choi_op(op):
    # op(np,complex,(N0,dim_out,dim_in)))
    # (ret)(np,complex,(dim_in*dim_out,dim_in*dim_out))
    if is_torch(op):
        tmp0 = op.transpose(1,2).reshape(op.shape[0], -1)
    else:
        tmp0 = op.transpose(0,2,1).reshape(op.shape[0], -1)
    ret = tmp0.T @ tmp0.conj()
    # tmp0 = [x.reshape(-1) for x in op]
    # ret = sum(x[:,np.newaxis]*x.conj() for x in tmp0)
    return ret


def apply_choi_op(op, rho):
    assert (rho.ndim==2) and (rho.shape[0]==rho.shape[1])
    assert (op.ndim==2) and (op.shape[0]==op.shape[1]) and (op.shape[0]%rho.shape[0]==0)
    dim0 = rho.shape[0]
    dim1 = op.shape[0]//dim0
    if is_torch(op):
        ret = torch.einsum(op.reshape(dim0,dim1,dim0,dim1), [0,1,2,3], rho, [0,2], [1,3])
    else:
        ret = np.einsum(op.reshape(dim0,dim1,dim0,dim1), [0,1,2,3], rho, [0,2], [1,3], optimize=True)
    return ret

def kraus_op_to_super_op(op):
    ret = sum(np.kron(x,x.conj()) for x in op)
    return ret


def apply_super_op(op, rho):
    assert (rho.ndim==2) and (rho.shape[0]==rho.shape[1])
    dim0 = rho.shape[0]
    assert (op.ndim==2) and (op.shape[1]==dim0*dim0)
    dim1 = int(np.sqrt(op.shape[0]))
    assert op.shape[0]==dim1*dim1
    ret = (op @ rho.reshape(-1)).reshape(dim1, dim1)
    return ret

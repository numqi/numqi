import numpy as np
import torch

import numqi.gate
import numqi.utils

def hf_dephasing_kraus_op(noise_rate):
    ret = np.stack([
        np.sqrt(1-noise_rate)*np.eye(2),
        np.sqrt(noise_rate)*numqi.gate.Z,
    ], axis=0)
    return ret


def hf_depolarizing_kraus_op(noise_rate):
    ret = np.stack([
        np.sqrt(1-3*noise_rate/4)*np.eye(2),
        np.sqrt(noise_rate/4)*numqi.gate.X,
        np.sqrt(noise_rate/4)*numqi.gate.Y,
        np.sqrt(noise_rate/4)*numqi.gate.Z,
    ], axis=0)
    return ret


def hf_amplitude_damping_kraus_op(noise_rate):
    assert 0<=noise_rate<=1
    ret = np.array([
        [[1,0], [0,np.sqrt(1-noise_rate)]],
        [[0,np.sqrt(noise_rate)], [0,0]],
    ])
    return ret


def kraus_op_to_choi_op(op):
    # op(np,complex,(N0,dim_out,dim_in)))
    # (ret)(np,complex,(dim_in*dim_out,dim_in*dim_out))
    if isinstance(op, torch.Tensor):
        tmp0 = op.transpose(1,2).reshape(op.shape[0], -1)
    else:
        tmp0 = op.transpose(0,2,1).reshape(op.shape[0], -1)
    ret = tmp0.T @ tmp0.conj()
    # tmp0 = [x.reshape(-1) for x in op]
    # ret = sum(x[:,np.newaxis]*x.conj() for x in tmp0)
    return ret


def kraus_op_to_super_op(op):
    ret = sum(np.kron(x,x.conj()) for x in op)
    return ret


def choi_op_to_kraus_op(op, dim_in, zero_eps=1e-10):
    # choi_op(dim_in*dim_out, dim_in*dim_out)
    # (ret)kraus_op(-1,dim_out,dim_in)
    assert (op.ndim==2) and (op.shape[0]==op.shape[1]) and (op.shape[0]%dim_in==0)
    dim_out = op.shape[0]//dim_in
    # TODO torch
    EVL,EVC = np.linalg.eigh(op)
    N0 = (EVL<zero_eps).sum() #sorted in order
    ret = (EVC[:,N0:]*np.sqrt(EVL[N0:])).reshape(dim_in, dim_out, -1).transpose(2,1,0)
    return ret


def choi_op_to_super_op(op, dim_in):
    # choi_op(dim_in*dim_out, dim_in*dim_out)
    # (ret)super_op(dim_out*dim_out, dim_in*dim_in)
    assert (op.ndim==2) and (op.shape[0]==op.shape[1]) and (op.shape[0]%dim_in==0)
    dim_out = op.shape[0]//dim_in
    ret = op.reshape(dim_in,dim_out,dim_in,dim_out).transpose(1,3,0,2).reshape(dim_out*dim_out,dim_in*dim_in)
    return ret


def super_op_to_choi_op(op):
    # super_op(dim_out*dim_out, dim_in*dim_in)
    # (ret)choi_op(dim_in*dim_out, dim_in*dim_out)
    assert op.ndim==2
    dim_in = int(np.sqrt(op.shape[1]))
    dim_out = int(np.sqrt(op.shape[0]))
    assert op.shape==(dim_out*dim_out, dim_in*dim_in)
    ret = op.reshape(dim_out,dim_out,dim_in,dim_in).transpose(2,0,3,1).reshape(dim_in*dim_out,dim_in*dim_out)
    return ret


def super_op_to_kraus_op(op, zero_eps=1e-10):
    assert op.ndim==2
    dim_in = int(np.sqrt(op.shape[1]))
    choi_op = super_op_to_choi_op(op)
    ret = choi_op_to_kraus_op(choi_op, dim_in, zero_eps)
    return ret


def apply_kraus_op(op, rho):
    ret = sum(x@rho@x.T.conj() for x in op)
    return ret


def apply_choi_op(op, rho):
    assert (rho.ndim==2) and (rho.shape[0]==rho.shape[1])
    assert (op.ndim==2) and (op.shape[0]==op.shape[1]) and (op.shape[0]%rho.shape[0]==0)
    din = rho.shape[0]
    dout = op.shape[0]//din
    if isinstance(op, torch.Tensor):
        ret = torch.einsum(op.reshape(din,dout,din,dout), [0,1,2,3], rho, [0,2], [1,3])
    else:
        ret = np.einsum(op.reshape(din,dout,din,dout), [0,1,2,3], rho, [0,2], [1,3], optimize=True)
    return ret

def apply_super_op(op, rho):
    assert (rho.ndim==2) and (rho.shape[0]==rho.shape[1])
    dim0 = rho.shape[0]
    assert (op.ndim==2) and (op.shape[1]==dim0*dim0)
    dim1 = int(np.sqrt(op.shape[0]))
    assert op.shape[0]==dim1*dim1
    ret = (op @ rho.reshape(-1)).reshape(dim1, dim1)
    return ret


# bad performance
def hf_channel_to_kraus_op(hf_channel, dim_in):
    r'''convert hf_channel (linear map) to kraus_op

    warning, not all hf_channel can be converted to kraus_op (see https://arxiv.org/abs/2212.12811 eq16)
    '''
    super_op = []
    for ind0 in range(dim_in):
        for ind1 in range(dim_in):
            tmp0 = np.zeros((dim_in,dim_in))
            tmp0[ind0,ind1] = 1
            super_op.append(hf_channel(tmp0))
    super_op = np.stack(super_op, axis=2).reshape(-1,dim_in*dim_in)
    kraus_op = super_op_to_kraus_op(super_op)
    return kraus_op

# bad performance
def hf_channel_to_choi_op(hf0, dim_in):
    ret = []
    tmp0 = np.zeros((dim_in,dim_in), dtype=np.float64)
    for ind0 in range(dim_in):
        for ind1 in range(dim_in):
            tmp0[ind0,ind1] = 1
            ret.append(hf0(tmp0))
            tmp0[ind0,ind1] = 0
    dim_out = ret[0].shape[0]
    ret = np.stack(ret, axis=0).reshape(dim_in,dim_in,dim_out,dim_out).transpose(0,2,1,3)
    return ret


def choi_op_to_bloch_map(op):
    assert (op.ndim==4) #(in,out,in,out)
    assert (op.shape[0]==op.shape[2]) and (op.shape[1]==op.shape[3])
    din,dout = op.shape[:2]
    tmp0 = op.transpose(1,3,2,0).reshape(dout*dout, din, din)
    tmp1 = numqi.gellmann.matrix_to_gellmann_basis(tmp0)
    op_gm = numqi.gellmann.matrix_to_gellmann_basis(tmp1.T.reshape(-1,dout,dout)).real.T
    matA = op_gm[:-1,:-1] * 2
    vecb = op_gm[:-1,-1] * np.sqrt(2/din)
    return matA, vecb

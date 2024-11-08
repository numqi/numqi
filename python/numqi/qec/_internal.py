import itertools
import numpy as np

import torch

import numqi.gate
import numqi.utils
import numqi.sim

from ._pauli import hf_pauli

# def degeneracy(code_i):
#     # only works for d=3
#     num_qubit = numqi.utils.hf_num_state_to_num_qubit(code_i.shape[0])
#     error_list = make_error_list(num_qubit, distance=2) + [()] #identity
#     mat = np.zeros([len(error_list), len(error_list)], dtype=np.complex128)
#     for ind0 in range(len(error_list)):
#         q0 = code_i
#         for ind_op,op_i in error_list[ind0]:
#             q0 = numqi.sim.state.apply_gate(q0, op_i, ind_op)
#         for ind1 in range(len(error_list)):
#             q1 = code_i
#             for ind_op,op_i in error_list[ind1]:
#                 q1 = numqi.sim.state.apply_gate(q1, op_i, ind_op)
#             mat[ind0,ind1] = np.vdot(q0, q1)
#     EVL = np.linalg.eigvalsh(mat)
#     return EVL


def generate_code_np(circ, num_logical_dim):
    num_qubit = circ.num_qubit
    ret = []
    for ind0 in range(num_logical_dim):
        q0 = np.zeros(2**num_qubit, dtype=np.complex128)
        q0[ind0] = 1
        circ.apply_state(q0)
        ret.append(circ.apply_state(q0))
    ret = np.stack(ret, axis=0)
    return ret


class _KnillLaflammeInnerProductTorchOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q0_torch, op_list):
        q0 = q0_torch.detach().numpy()
        num_logical_dim = q0.shape[0]
        num_logical_qubit = numqi.utils.hf_num_state_to_num_qubit(num_logical_dim, kind='exact')
        ret = []
        q0_conj = q0.conj()
        for op_sequence in op_list:
            q1 = q0.reshape(-1)
            for (ind1,op_i) in op_sequence:
                q1 = numqi.sim.state.apply_gate(q1, op_i, [x+num_logical_qubit for x in ind1])
            ret.append(q0_conj @ q1.reshape(num_logical_dim,-1).T)
        ret = torch.from_numpy(np.stack(ret, axis=0))
        ctx.save_for_backward(q0_torch)
        ctx._pyqet_data = dict(op_list=op_list)
        return ret

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        q0 = ctx.saved_tensors[0].detach().numpy()
        grad_output = grad_output.detach().numpy()
        op_list = ctx._pyqet_data['op_list']
        num_logical_dim = q0.shape[0]
        num_logical_qubit = numqi.utils.hf_num_state_to_num_qubit(num_logical_dim, kind='exact')
        q0_grad = np.zeros_like(q0)
        hf0 = lambda x: x.reshape(num_logical_dim, -1)
        for ind0 in range(len(op_list)):
            q1 = q0.reshape(-1)
            for ind1,op_i in op_list[ind0]:
                q1 = numqi.sim.state.apply_gate(q1, op_i, [x+num_logical_qubit for x in ind1])
            q0_grad += grad_output[ind0].conj() @ hf0(q1)

            q1 = q0.reshape(-1)
            for ind1,op_i in reversed(op_list[ind0]):
                q1 = numqi.sim.state.apply_gate(q1, op_i.T.conj(), [x+num_logical_qubit for x in ind1])
            q0_grad += grad_output[ind0].T @ hf0(q1)
        q0_grad = torch.from_numpy(q0_grad)
        return q0_grad,None


def knill_laflamme_inner_product(q0, op_list):
    if isinstance(q0, torch.Tensor):
        ret = _KnillLaflammeInnerProductTorchOp.apply(q0, op_list)
    else:
        assert q0.ndim==2 #np.ndarray
        num_logical_dim = q0.shape[0]
        num_logical_qubit = numqi.utils.hf_num_state_to_num_qubit(num_logical_dim, kind='exact')
        ret = []
        q0_conj = q0.conj()
        for op_sequence in op_list:
            q1 = q0.reshape(-1)
            for (ind1,op_i) in op_sequence:
                q1 = numqi.sim.state.apply_gate(q1, op_i, [x+num_logical_qubit for x in ind1])
            ret.append(q0_conj @ q1.reshape(num_logical_dim,-1).T)
        ret = np.stack(ret)
    return ret

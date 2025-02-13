import numpy as np
import torch
import typing

import numqi.utils
import numqi.sim.state

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


class _Function_knill_laflamme_hermite_mul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matA, psi):
        r'''Compute the Knill-Laflamme-Hermite inner product.

        Parameters:
            matA (torch.Tensor): sparse csr-format tensor of shape `(num_op*2**n, 2**n)`. must be Hermitian.
            psi (torch.Tensor): code space of shape `(2**n, dimK)`.

        Returns:
            lambda_aij (torch.Tensor): inner product of shape `(num_op, dimK, dimK)`.
        '''
        assert not matA.requires_grad
        dimN,dimK = psi.shape
        num_op = matA.numel()//(dimN*dimN)
        A_psi = (matA @ psi).reshape(num_op, dimN, dimK)
        lambda_aij = psi.T.conj() @ A_psi
        ctx.A_psi = A_psi
        return lambda_aij

    @staticmethod
    def backward(ctx, grad):
        A_psi = ctx.A_psi
        tmp0 = grad.transpose(1,2).conj() + grad
        grad_psi = torch.einsum(A_psi, [0,1,2], tmp0, [0,2,3], [1,3])
        # if not Hermitian
        # Adag_psi = (matA.transpose(1,2).conj() @ psi).reshape(-1, dimN, dimK)
        # tmp0 = torch.einsum(A_psi, [0,1,2], grad.conj(), [0,3,2], [1,3])
        # grad_psi = tmp0 + torch.einsum(Adag_psi, [0,1,2], grad, [0,2,3], [1,3])
        return None, grad_psi

tmp0 = typing.Annotated[typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor], '''Compute the Knill-Laflamme-Hermite inner product.

Parameters:
    matA (torch.Tensor): sparse csr-format tensor of shape `(num_op*2**n, 2**n)`. must be Hermitian.
    psi (torch.Tensor): code space of shape `(2**n, dimK)`.

Returns:
    lambda_aij (torch.Tensor): inner product of shape `(num_op, dimK, dimK)`.
''']
knill_laflamme_hermite_mul:tmp0 = _Function_knill_laflamme_hermite_mul.apply

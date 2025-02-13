import numpy as np
import torch

import numqi.utils
import numqi.manifold
import numqi.sim

from ._grad import knill_laflamme_inner_product, knill_laflamme_hermite_mul

def knill_laflamme_loss(inner_product, kind='L2'):
    assert kind in {'L1','L2'}
    assert inner_product.ndim==3
    num_logical_dim = inner_product.shape[1]
    if isinstance(inner_product, torch.Tensor):
        mask = torch.triu(torch.ones(num_logical_dim, num_logical_dim,
                 dtype=torch.complex128, device=inner_product.device), diagonal=1)
        hf0 = lambda x: x if (kind=='L1') else torch.square(x)
        tmp0 = hf0(torch.abs(inner_product*mask)).sum()
        tmp1 = torch.diagonal(inner_product, dim1=1, dim2=2)
        tmp2 = hf0(torch.abs(tmp1 - tmp1.mean(dim=1, keepdim=True))).sum()
        loss = tmp0 + tmp2
    else:
        mask = np.triu(np.ones((num_logical_dim,num_logical_dim)), k=1)
        hf0 = lambda x: x if (kind=='L1') else np.square(x)
        tmp0 = hf0(np.abs(inner_product*mask)).sum()
        tmp1 = np.diagonal(inner_product, axis1=1, axis2=2)
        tmp2 = hf0(np.abs(tmp1 - tmp1.mean(axis=1, keepdims=True))).sum()
        loss = tmp0 + tmp2
    return loss


class QECCEqualModel(torch.nn.Module):
    def __init__(self, code0, code1, device='cpu'):
        super().__init__()
        assert code0.shape==code1.shape
        device = torch.device(device)
        self.num_qubit = numqi.utils.hf_num_state_to_num_qubit(code0.shape[1], kind='exact')
        self.manifold = numqi.manifold.SpecialOrthogonal(2, batch_size=self.num_qubit, dtype=torch.complex128, device=device)
        self.code0 = torch.tensor(code0, dtype=torch.complex128, device=device)
        self.code1 = torch.tensor(code1, dtype=torch.complex128, device=device)

    def forward(self):
        N0 = len(self.code0)
        unitary = self.manifold()
        code0 = self.code0
        for ind0 in range(self.num_qubit):
            tmp0 = code0.reshape(N0*(2**ind0), 2, 2**(self.num_qubit-ind0-1))
            code0 = torch.einsum(tmp0, [0,1,2], unitary[ind0], [1,3], [0,3,2]).reshape(N0, -1)
        tmp0 = (code0.conj() @ self.code1.T).reshape(-1)
        loss = N0 - torch.vdot(tmp0, tmp0).real
        return loss


class VarQECUnitary(torch.nn.Module):
    def __init__(self, num_qubit:int, num_logical_dim:int, error_torch:torch.tensor):
        r'''Variational method to find a quantum error correcting code.

        Parameters:
            num_qubit (int): number of qubits.
            num_logical_dim (int): number of logical qubits.
            error_torch (torch.tensor): error tensor of shape `(N0, 2**num_qubit, 2**num_qubit)` or `(N0*2**num_qubit, 2**num_qubit)`.
                    Recommend to use `numqi.qec.make_pauli_error_list_sparse` to generate sparse torch tensor for performance.
        '''
        super().__init__()
        self.num_logical_dim = num_logical_dim
        self.num_qubit = num_qubit
        self.error_torch = error_torch.clone().to(torch.complex128)
        self.manifold = numqi.manifold.Stiefel(2**num_qubit, rank=self.num_logical_dim, dtype=torch.complex128)
        self.ind_triu = torch.triu_indices(num_logical_dim, num_logical_dim, offset=1)
        self.lambda_target = None

    def set_lambda_target(self, x:None|str|float|np.ndarray):
        r'''Set the target of the lambda.

        Parameters:
            x (None|str|float|np.ndarray): the objective function depends on the lambda. if `lambda` is
                - None: the objective function is to satisfy the Knill-Laflamme condition only (just search for a code).
                - 'min': minimize the lambda^2 while satisfying the Knill-Laflamme condition.
                - 'max': maximize the lambda^2 while satisfying the Knill-Laflamme condition.
                - float: minimize the difference between lambda^2 and the target while satisfying the Knill-Laflamme condition.
                        Be careful, the input is the square of the lambda, not the lambda itself (different from the code in `QEC-and-RDM`).
                - np.ndarray: search for a code with the given lambda vector.
        '''
        if x is None:
            self.lambda_target = None
        elif isinstance(x, str):
            assert x in ['min', 'max']
            self.lambda_target = x
        else: #float
            self.lambda_target = torch.tensor(x, dtype=torch.float64).reshape(-1)

    def forward(self, return_info:bool=False):
        q0 = self.manifold()
        lambda_aij = knill_laflamme_hermite_mul(self.error_torch, q0)
        # lambda_aij = q0.T.conj() @ (self.error_torch @ q0).reshape(-1, *q0.shape) #equivalent but much slower
        lambda_ai = torch.diagonal(lambda_aij, dim1=1, dim2=2).real
        lambda_a = lambda_ai.mean(dim=1)
        constraint = [
            lambda_aij[:,self.ind_triu[0],self.ind_triu[1]],
            lambda_ai - lambda_a.reshape(-1,1),
        ]
        lambda2 = torch.dot(lambda_a, lambda_a)
        loss = None
        if self.lambda_target == 'min':
            loss = lambda2
        elif self.lambda_target == 'max':
            loss = -lambda2
        elif isinstance(self.lambda_target, torch.Tensor):
            constraint.append(lambda2-self.lambda_target)

        if loss is None:
            loss = sum([torch.vdot(x.reshape(-1), x.reshape(-1)).real for x in constraint])
            constraint = []
        if return_info:
            info = dict(q0=q0.detach().numpy().copy(), lambda_aij=lambda_aij.detach().numpy().copy(), lambda2=lambda2.item())
            ret = (loss, info) if (len(constraint)==0) else (loss,constraint,info)
        else:
            ret = loss if (len(constraint)==0) else (loss,constraint)
        return ret


## TODO replace with Stiefel manifold
# class VarQECUnitary(torch.nn.Module):
#     def __init__(self, num_qubit, num_logical_dim, error_list, loss_type='L2'):
#         super().__init__()
#         self.num_logical_dim = num_logical_dim
#         self.num_logical_dim_ceil = 2**numqi.utils.hf_num_state_to_num_qubit(num_logical_dim, kind='ceil')
#         self.num_qubit = num_qubit
#         self.error_list = error_list
#         # self.error_list_torch = [[(y0,torch.tensor(y1,dtype=torch.complex128)) for y0,y1 in x] for x in error_list]
#         assert loss_type in {'L1','L2'}
#         self.loss_type = loss_type
#         self.manifold = numqi.manifold.Stiefel(2**num_qubit, rank=self.num_logical_dim_ceil, dtype=torch.complex128)
#         self.q0_torch = None

#     def forward(self):
#         q0_torch = self.manifold().T
#         self.q0_torch = q0_torch.detach()
#         tmp0 = knill_laflamme_inner_product(q0_torch, self.error_list)
#         inner_product = tmp0[:,:self.num_logical_dim,:self.num_logical_dim]
#         loss = knill_laflamme_loss(inner_product, self.loss_type)
#         return loss

#     def get_code(self):
#         with torch.no_grad():
#             self()
#         ret = self.q0_torch.cpu()[:self.num_logical_dim].numpy().reshape(-1, 2**self.num_qubit).copy()
#         return ret

class VarQEC(torch.nn.Module):
    def __init__(self, circuit, num_logical_dim, error_list, loss_type='L2'):
        super().__init__()
        num_logical_qubit = numqi.utils.hf_num_state_to_num_qubit(num_logical_dim, kind='ceil')
        num_qubit = circuit.num_qubit
        circuit.shift_qubit_index_(num_logical_qubit)
        self.circuit_torch = numqi.sim.CircuitTorchWrapper(circuit)
        self.num_qubit = num_qubit
        self.num_logical_dim = num_logical_dim
        self.num_logical_qubit = num_logical_qubit
        self.error_list = error_list
        assert loss_type in {'L1','L2'}
        self.loss_type = loss_type

        self.q0 = torch.empty(2**self.num_logical_qubit, 2**self.num_qubit, dtype=torch.complex128)
        self.inner_product_torch = None

    def _run_circuit(self):
        self.q0[:] = 0
        self.q0[np.arange(self.num_logical_dim), np.arange(self.num_logical_dim)] = 1
        q0 = self.circuit_torch(self.q0.reshape(-1)).reshape(-1, 2**self.num_qubit)
        return q0

    def get_code(self):
        with torch.no_grad():
            q0 = self._run_circuit()
        ret = q0.reshape(-1, 2**self.num_qubit)[:self.num_logical_dim].numpy().copy()
        return ret

    def forward(self):
        q0 = self._run_circuit()
        inner_product = knill_laflamme_inner_product(q0, self.error_list)
        loss = knill_laflamme_loss(inner_product[:,:self.num_logical_dim,:self.num_logical_dim], self.loss_type)
        return loss

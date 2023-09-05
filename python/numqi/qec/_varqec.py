import numpy as np
import torch

import numqi.utils
import numqi.param
import numqi.sim

from ._internal import knill_laflamme_inner_product

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
        self.device = device
        self.num_qubit = numqi.utils.hf_num_state_to_num_qubit(code0.shape[1], kind='exact')
        np_rng = np.random.default_rng()
        tmp0 = np_rng.uniform(-1, 1, (self.num_qubit, 2, 2))
        self.theta = torch.nn.Parameter(torch.tensor(tmp0, dtype=torch.float64, device=self.device))
        self.code0 = torch.tensor(code0, dtype=torch.complex128, device=self.device)
        self.code1 = torch.tensor(code1, dtype=torch.complex128, device=self.device)

    def forward(self):
        N0 = len(self.code0)
        unitary = numqi.param.real_matrix_to_special_unitary(self.theta)
        code0 = self.code0
        for ind0 in range(self.num_qubit):
            tmp0 = code0.reshape(N0*(2**ind0), 2, 2**(self.num_qubit-ind0-1))
            code0 = torch.einsum(tmp0, [0,1,2], unitary[ind0], [1,3], [0,3,2]).reshape(N0, -1)
        loss = torch.sum(1-torch.linalg.norm(code0.conj() @ self.code1.T, dim=1)**2)
        return loss

class VarQECUnitary(torch.nn.Module):
    def __init__(self, num_qubit, num_logical_dim, error_list, device='cpu', loss_type='L2'):
        super().__init__()
        assert device=='cpu'
        self.num_logical_dim = num_logical_dim
        self.num_logical_dim_ceil = 2**numqi.utils.hf_num_state_to_num_qubit(num_logical_dim, kind='ceil')
        self.num_qubit = num_qubit
        self.device = device
        self.error_list = error_list
        # self.error_list_torch = [[(y0,torch.tensor(y1,dtype=torch.complex128,device=device)) for y0,y1 in x] for x in error_list]
        assert loss_type in {'L1','L2'}
        self.loss_type = loss_type

        np_rng = np.random.default_rng()
        tmp0 = np_rng.normal(size=(2**num_qubit,2**num_qubit))
        self.theta = torch.nn.Parameter(torch.tensor(tmp0, dtype=torch.float64, device=device))
        self.q0_torch = None

    def forward(self):
        self.q0_torch = numqi.param.real_matrix_to_special_unitary(self.theta)[:self.num_logical_dim_ceil]
        tmp0 = knill_laflamme_inner_product(self.q0_torch, self.error_list)
        inner_product = tmp0[:,:self.num_logical_dim,:self.num_logical_dim]
        loss = knill_laflamme_loss(inner_product, self.loss_type)
        return loss

    def get_code(self):
        self()
        ret = self.q0_torch.detach().cpu()[:self.num_logical_dim].numpy().reshape(-1, 2**self.num_qubit).copy()
        return ret

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

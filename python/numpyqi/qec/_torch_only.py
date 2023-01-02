import numpy as np
import torch

import numpyqi.utils
import numpyqi.param

class QECCEqualModel(torch.nn.Module):
    def __init__(self, code0, code1, device='cpu'):
        super().__init__()
        assert code0.shape==code1.shape
        self.device = device
        self.num_qubit = numpyqi.utils.hf_num_state_to_num_qubit(code0.shape[1], kind='exact')
        np_rng = np.random.default_rng()
        tmp0 = np_rng.uniform(-1, 1, (self.num_qubit, 2, 2))
        self.theta = torch.nn.Parameter(torch.tensor(tmp0, dtype=torch.float64, device=self.device))
        self.code0 = torch.tensor(code0, dtype=torch.complex128, device=self.device)
        self.code1 = torch.tensor(code1, dtype=torch.complex128, device=self.device)

    def forward(self):
        N0 = len(self.code0)
        unitary = numpyqi.param.real_matrix_to_unitary(self.theta)
        code0 = self.code0
        for ind0 in range(self.num_qubit):
            tmp0 = code0.reshape(N0*(2**ind0), 2, 2**(self.num_qubit-ind0-1))
            code0 = torch.einsum(tmp0, [0,1,2], unitary[ind0], [1,3], [0,3,2]).reshape(N0, -1)
        loss = torch.sum(1-torch.linalg.norm(code0.conj() @ self.code1.T, dim=1)**2)
        return loss

class VarQECUnitary(torch.nn.Module):
    def __init__(self, num_qubit, num_logical_dim, error_list, device='cpu', loss_type='L2'):
        super().__init__()
        self.num_logical_dim = num_logical_dim
        self.num_logical_dim_ceil = 2**numpyqi.utils.hf_num_state_to_num_qubit(num_logical_dim, kind='ceil')
        self.num_qubit = num_qubit
        self.device = device
        self.error_list = error_list
        self.error_list_torch = [[(y0,torch.tensor(y1,dtype=torch.complex128,device=device)) for y0,y1 in x] for x in error_list]
        assert loss_type in {'L1','L2'}
        self.loss_type = loss_type

        np_rng = np.random.default_rng()
        tmp0 = np_rng.normal(size=(2**num_qubit,2**num_qubit))
        self.theta = torch.nn.Parameter(torch.tensor(tmp0, dtype=torch.float64, device=device))
        self.q0_torch = None

    def forward(self):
        self.q0_torch = numpyqi.param.real_matrix_to_unitary(self.theta)[:self.num_logical_dim_ceil]
        tmp0 = numpyqi.qec.knill_laflamme_inner_product(self.q0_torch, self.error_list_torch)
        inner_product = tmp0[:,:self.num_logical_dim,:self.num_logical_dim]
        loss = numpyqi.qec.knill_laflamme_loss(inner_product, self.loss_type)
        return loss

    def get_code(self):
        self()
        ret = self.q0_torch.detach().cpu()[:self.num_logical_dim].numpy().reshape(-1, 2**self.num_qubit).copy()
        return ret

class VarQEC(torch.nn.Module):
    def __init__(self, circuit, num_logical_dim, error_list, loss_type='L2'):
        super().__init__()
        num_logical_qubit = numpyqi.utils.hf_num_state_to_num_qubit(num_logical_dim, kind='ceil')
        num_qubit = circuit.num_qubit
        circuit.shift_qubit_index_(num_logical_qubit)
        self.circuit = circuit
        self.theta = circuit.get_trainable_parameter()
        self.num_qubit = num_qubit
        self.num_logical_dim = num_logical_dim
        self.num_logical_qubit = num_logical_qubit
        self.error_list = error_list
        assert loss_type in {'L1','L2'}
        self.loss_type = loss_type

        self.q0 = np.zeros((2**self.num_logical_qubit, 2**self.num_qubit), dtype=np.complex128)
        self.inner_product_torch = None

    def get_code(self):
        self()
        ret = self.q0.reshape(-1, 2**self.num_qubit)[:self.num_logical_dim].copy()
        return ret

    def forward(self):
        self.q0[:] = 0
        self.q0[np.arange(self.num_logical_dim), np.arange(self.num_logical_dim)] = 1
        self.q0 = self.circuit.apply_state(self.q0.reshape(-1)).reshape(-1, 2**self.num_qubit)
        inner_product = numpyqi.qec.knill_laflamme_inner_product(self.q0, self.error_list)

        self.inner_product_torch = torch.tensor(inner_product[:,:self.num_logical_dim,:self.num_logical_dim], dtype=torch.complex128)
        self.inner_product_torch.requires_grad_()
        loss = numpyqi.qec.knill_laflamme_loss(self.inner_product_torch, self.loss_type)
        return loss

    def grad_backward(self, loss):
        loss.backward()
        term_grad = self.inner_product_torch.grad.detach().numpy()
        tmp0 = 2**self.num_logical_qubit-self.num_logical_dim
        if tmp0!=0:
            term_grad = np.pad(term_grad, [(0,0),(0,tmp0),(0,tmp0)])
        q0_grad = numpyqi.qec.knill_laflamme_inner_product_grad(self.q0, self.error_list, term_grad)
        tmp0 = self.circuit.apply_state_grad(self.q0.reshape(-1), q0_grad.reshape(-1))
        # self.q0, q0_grad, op_grad_list
        self.q0 = tmp0[0].reshape(self.q0.shape)

import numpy as np
import torch

import numqi.sim
import numqi.utils
import numqi.manifold

from .utils import get_measure_matrix, get_xbit

class QueryGroverModel(torch.nn.Module):
    def __init__(self, num_qubit, num_query, use_fractional=False, dtype='float64', device='cpu'):
        super().__init__()
        assert dtype in {'float32', 'float64'}
        assert device in {'cpu', 'cuda'}
        self.dtype = torch.float32 if dtype=='float32' else torch.float64
        self.cdtype = torch.complex64 if dtype=='float32' else torch.complex128
        self.device = torch.device(device)
        self.num_qubit = num_qubit
        self.manifold_SU = numqi.manifold.SpecialOrthogonal(dim=2**num_qubit, batch_size=num_query+1, dtype=self.cdtype, device=self.device)
        if use_fractional:
            self.alpha = torch.nn.Parameter(torch.rand(num_query, dtype=self.dtype, device=self.device))
            self.mask = 1-torch.eye(2**num_qubit, dtype=self.dtype, device=self.device)
        else:
            self.alpha = None
            self.mask = 1-2*torch.eye(2**num_qubit, dtype=self.dtype, device=self.device)
        self.q0 = torch.zeros(2**num_qubit, 2**num_qubit, dtype=self.cdtype, device=self.device)
        self.error_rate = None

    def forward(self):
        self.q0[:] = 0
        self.q0[:,0] = 1
        unitary = self.manifold_SU()
        q0 = self.q0
        for ind0 in range(len(unitary)-1):
            q0 = q0 @ unitary[ind0]
            if self.alpha is None:
                q0 = q0*self.mask
            else:
                tmp0 = torch.exp((-1j*np.pi)*self.alpha[ind0]) * torch.ones(q0.shape[0], device=self.device)
                q0 = q0 * (self.mask + torch.diag(tmp0))
        q0 = q0 @ unitary[-1]
        probability_sum,error_rate = diagonal_to_probability_sum(q0)
        self.error_rate = error_rate
        loss = 1 - probability_sum/2**self.num_qubit
        return loss

class GroverOracle:
    def __init__(self, num_qubit):
        self.kind = 'custom'
        self.name = 'grover_oracle'
        self.requires_grad = False
        self.num_qubit = num_qubit
        self.index0 = np.arange(2**num_qubit)

    def forward(self, q0):
        q0 = q0.reshape(2**self.num_qubit, -1).copy()
        assert q0.shape[0]==q0.shape[1]
        q0[self.index0, self.index0] = -q0[self.index0, self.index0]
        return q0.reshape(-1)

    def grad_backward(self, q0_conj, q0_grad):
        q0_conj = q0_conj.reshape(2**self.num_qubit, -1).copy()
        q0_grad = q0_grad.reshape(2**self.num_qubit, -1).copy()
        q0_conj[self.index0, self.index0] = -q0_conj[self.index0, self.index0]
        q0_grad[self.index0, self.index0] = -q0_grad[self.index0, self.index0]
        op_grad = None
        return q0_conj.reshape(-1),q0_grad.reshape(-1),op_grad


hf_fractional_grover = lambda x: torch.exp(-1j*np.pi*x) if isinstance(x, torch.Tensor) else np.exp(-1j*np.pi*x)

class FractionalGroverOracle(numqi.sim.ParameterGate):
    def __init__(self, num_qubit, theta=0, requires_grad=True):
        self.hf0 = hf_fractional_grover
        self.kind = 'custom'
        self.args = float(theta), #tuple
        self.array = self.hf0(theta)
        self.index = 0#useless
        self.requires_grad = requires_grad
        self.name = 'fractional_grover_oracle'
        self.num_qubit = num_qubit
        self.index0 = np.arange(2**num_qubit)

    def forward(self, q0):
        q0 = q0.reshape(2**self.num_qubit, -1).copy()
        assert q0.shape[0]==q0.shape[1]
        q0[self.index0, self.index0] *= self.array
        return q0.reshape(-1)

    def set_args(self, args, array=None):
        self.args = numqi.utils.hf_tuple_of_any(args, float)
        if array is None:
            self.array = self.hf0(*args)
        else:
            self.array = np.asarray(array)

    def grad_backward(self, q0_conj, q0_grad):
        q0_conj = q0_conj.reshape(2**self.num_qubit, -1).copy()
        q0_grad = q0_grad.reshape(2**self.num_qubit, -1).copy()
        q0_conj[self.index0, self.index0] *= self.array
        op_grad = None
        if self.requires_grad:
            op_grad = np.dot(q0_conj[self.index0, self.index0], q0_grad[self.index0, self.index0])
        q0_grad[self.index0, self.index0] *= self.array.conj()
        return q0_conj.reshape(-1),q0_grad.reshape(-1),op_grad

    def copy(self):
        ret = FractionalGroverOracle(self.num_qubit, theta=self.args[0], requires_grad=self.requires_grad)
        return ret


def diagonal_to_probability_sum(q0):
    assert (q0.ndim==2) and (q0.shape[0]==q0.shape[1])
    if isinstance(q0, torch.Tensor):
        tmp0 = torch.diag(q0)
        prob_sum = torch.vdot(tmp0, tmp0).real
        error_rate = 1 - torch.abs(tmp0.detach()).min()**2 #no need to gradient backward
    else: #numpy
        tmp0 = np.diag(q0)
        prob_sum = np.vdot(tmp0, tmp0).real
        error_rate = 1 - np.abs(tmp0).min()**2 #no need to gradient backward
    return prob_sum, error_rate


class QueryGroverQuantumModel(torch.nn.Module):
    def __init__(self, circuit):
        super().__init__()
        self.num_qubit = circuit.num_qubit
        self.num_bitstring = circuit.num_qubit
        circuit.shift_qubit_index_(self.num_bitstring)
        self.circuit_torch = numqi.sim.CircuitTorchWrapper(circuit)

        self.q0 = torch.empty(2**self.num_bitstring,2**self.num_qubit, dtype=torch.complex128)
        self.error_rate = None

    def forward(self):
        self.q0[:] = 0
        self.q0[:,0] = 1
        q0 = self.circuit_torch(self.q0.reshape(-1)).reshape(-1, 2**self.num_qubit)
        probability_sum,error_rate = diagonal_to_probability_sum(q0)
        self.error_rate = error_rate.item()
        loss = 1 - probability_sum/2**self.num_qubit
        return loss


class HammingQueryQuditModel(torch.nn.Module):
    def __init__(self, num_query, dim_query, partition, bitmap, num_XZ=None,
                use_fractional=True, alpha_upper_bound=None):
        super().__init__()
        num_bit = numqi.utils.hf_num_state_to_num_qubit(bitmap.shape[0], kind='exact')
        partition = np.array(partition)
        dim_total = partition.sum()
        assert dim_total%dim_query==0
        if use_fractional:
            self.alpha = torch.nn.Parameter(torch.ones(num_query, dtype=torch.float64))
        else:
            self.alpha = torch.ones(num_query, dtype=torch.float64)
        np_rng = np.random.default_rng()
        hf0 = lambda *size,x0,x1: torch.nn.Parameter(torch.tensor(np_rng.uniform(x0, x1, size), dtype=torch.float64))
        if num_XZ is not None:
            self.x_theta = hf0(num_query+1, num_XZ, x0=0, x1=2*np.pi)
            self.z_theta = hf0(num_query+1, num_XZ, x0=0, x1=2*np.pi)
        else:
            self.manifold = numqi.manifold.SpecialOrthogonal(dim=dim_total, batch_size=num_query+1, dtype=torch.complex128)
        self.measure_matrix_T = torch.tensor(get_measure_matrix(bitmap, partition).T.copy(), dtype=torch.float64)
        self.num_bit = num_bit
        self.dim_total = dim_total
        self.dim_query = dim_query

        self.x_bit = torch.tensor(get_xbit(num_bit, dim_query), dtype=torch.int64)
        assert (alpha_upper_bound is None) or (alpha_upper_bound>0)
        self.alpha_upper_bound = alpha_upper_bound

    def forward(self):
        if hasattr(self, 'x_theta'):
            XZ = []
            for xi,zi in zip(self.x_theta,self.z_theta):
                tmp0 = numqi.gate.rx(xi, self.dim_total) * (numqi.gate.rz(zi, self.dim_total, diag_only=True).reshape(-1, 1, self.dim_total))
                tmp1 = tmp0[0]
                for x in tmp0[1:]:
                    tmp1 = x @ tmp1
                XZ.append(tmp1)
        else:
            XZ = self.manifold()
        alpha = self.alpha % 2 #TODO remove this
        oracle = [torch.exp((-1j*np.pi*x)*self.x_bit) for x in alpha]
        q0 = torch.zeros(self.dim_total, 2**self.num_bit, dtype=torch.complex128)
        q0[0] = 1
        for ind0 in range(len(oracle)):
            q0 = XZ[ind0] @ q0
            tmp0 = oracle[ind0].T.reshape(self.dim_query,1,-1)
            q0 = (tmp0 * q0.reshape(self.dim_query, -1, q0.shape[1])).reshape(self.dim_total, -1)
        q0 = XZ[ind0+1] @ q0
        tmp0 = (q0*q0.conj()).real
        loss = 1-torch.sum(self.measure_matrix_T*tmp0)/q0.shape[1]
        self.error_rate = (1-torch.min(torch.sum(self.measure_matrix_T*tmp0.detach(),dim=0))).item()
        if self.alpha_upper_bound is not None:
            loss = loss + torch.square(torch.nn.functional.relu(alpha.sum()-self.alpha_upper_bound)).sum()
            # loss = loss + torch.square(torch.nn.functional.relu(alpha-0.9)).sum()
        return loss

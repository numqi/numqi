import numpy as np
import torch

import numqi

class DummyQNNModel(torch.nn.Module):
    def __init__(self, circuit):
        super().__init__()
        self.circuit_torch = numqi.sim.CircuitTorchWrapper(circuit)
        self.num_qubit = circuit.num_qubit
        np_rng = np.random.default_rng()
        tmp0 = np_rng.normal(size=2**self.num_qubit) + 1j*np_rng.normal(size=2**self.num_qubit)
        self.target_state = torch.tensor(tmp0 / np.linalg.norm(tmp0), dtype=torch.complex128)

        self.q0 = torch.empty(2**self.num_qubit, dtype=torch.complex128, requires_grad=False)

    def forward(self):
        self.q0[:] = 0
        self.q0[0] = 1
        q0 = self.circuit_torch(self.q0)
        inner_product = torch.dot(self.target_state.conj(), q0)
        loss = (inner_product*inner_product.conj()).real
        return loss


class Rosenbrock(torch.nn.Module):
    def __init__(self, num_parameter=3) -> None:
        super().__init__()
        assert num_parameter>1
        np_rng = np.random.default_rng()
        self.theta = torch.nn.Parameter(torch.tensor(np_rng.normal(size=num_parameter), dtype=torch.float64))
        # solution [1,1,...,1] 0

    def forward(self):
        tmp0 = self.theta[1:] - self.theta[:-1]
        tmp1 = 1-self.theta
        ret = 100*torch.dot(tmp0, tmp0) + torch.dot(tmp1,tmp1)
        return ret

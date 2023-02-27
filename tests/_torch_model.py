import numpy as np
import torch

import numpyqi

class DummyQNNModel(torch.nn.Module):
    def __init__(self, circuit):
        super().__init__()
        self.circuit = circuit
        self.theta = circuit.get_trainable_parameter()
        self.num_qubit = circuit.num_qubit
        np_rng = np.random.default_rng()
        tmp0 = np_rng.normal(size=2**self.num_qubit) + 1j*np_rng.normal(size=2**self.num_qubit)
        self.target_state = (tmp0 / np.linalg.norm(tmp0)).astype(np.complex128)

        self.q0 = None
        self.q0_grad = None

    def forward(self):
        self.q0 = numpyqi.sim.state.new_base(self.num_qubit, dtype=np.complex128)
        self.q0 = self.circuit.apply_state(self.q0)
        self.inner_product = numpyqi.sim.state.inner_product(self.q0, self.target_state)
        loss = abs(self.inner_product)**2
        return loss

    def grad_backward(self, loss=None):
        # loss is necessary for numpyqi.optimize.check_model_gradient
        q0_grad,_ = numpyqi.sim.state.inner_product_grad(self.q0, self.target_state, 2*self.inner_product, tag_grad=(True,False))
        self.q0, self.q0_grad, op_grad_list = self.circuit.apply_state_grad(self.q0, q0_grad)
        return op_grad_list


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

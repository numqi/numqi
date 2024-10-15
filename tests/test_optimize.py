import numpy as np
import torch

import numqi

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

def test_gradient_correct():
    model = Rosenbrock(num_parameter=5)
    numqi.optimize.check_model_gradient(model, zero_eps=1e-4)


class DummyModel02(torch.nn.Module):
    def __init__(self):
        super().__init__()
        np_rng = np.random.default_rng()
        hf0 = lambda *x: torch.nn.Parameter(torch.tensor(np_rng.uniform(-1,1,size=x), dtype=torch.float64))
        self.theta = hf0(2)
        self.manifold_positive = numqi.manifold.PositiveReal(method='softplus', dtype=torch.float64)

    def forward(self):
        loss = self.theta[0]**2 + self.theta[1]**2
        p_positive = self.manifold_positive()
        constraint = self.theta[0] - self.theta[1] - 1 - p_positive
        return loss,constraint


def test_optimize_minimize_with_constraint():
    model = DummyModel02()
    theta_optim = numqi.optimize.minimize(model, num_repeat=5, tol=1e-15, method='L-BFGS-B',
                    constraint_penalty=10, constraint_p=1.1, constraint_threshold=1e-14)
    loss,constraint = model()
    assert abs(loss.item()-0.5) < 1e-5 #about 1e-7
    assert constraint.item()**2 < 1e-14

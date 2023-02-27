import numpy as np
import scipy.optimize
import torch

import numpyqi



np_rng = np.random.default_rng()

# or use numpyqi.optimize.minimize(model) for convenience
def hf_demo(model, print_freq=10, history_info=None):
    hf_model = numpyqi.optimize.hf_model_wrapper(model)
    hf_callback = numpyqi.optimize.hf_callback_wrapper(hf_model, history_info, print_freq=print_freq)
    num_parameter = len(numpyqi.optimize.get_model_flat_parameter(model))
    theta0 = np_rng.uniform(size=num_parameter)
    theta_optim = scipy.optimize.minimize(hf_model, theta0, method='L-BFGS-B', jac=True, callback=hf_callback, tol=1e-20)
    print(theta_optim.x, theta_optim.fun)


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

class Rastrigin(torch.nn.Module):
    def __init__(self, num_parameter=3):
        super().__init__()
        assert num_parameter>1
        np_rng = np.random.default_rng()
        self.theta = torch.nn.Parameter(torch.tensor(np_rng.normal(size=num_parameter), dtype=torch.float64))
        # solution [0,0,...,0], 0

    def forward(self):
        A = 10
        tmp0 = A*len(self.theta) #a constant shifts the minimum to zero
        tmp1 = torch.dot(self.theta,self.theta)
        tmp2 = A*torch.sum(torch.cos(2*np.pi*self.theta))
        ret = tmp0 + tmp1 - tmp2
        return ret


class Ackley(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        np_rng = np.random.default_rng()
        self.theta = torch.nn.Parameter(torch.tensor(np_rng.normal(size=2), dtype=torch.float64))
        # solution [0,0] 0

    def forward(self):
        x,y = self.theta
        tmp0 = -20*torch.exp(-0.2*torch.sqrt(0.5*(x*x + y*y)))
        tmp1 = -torch.exp(0.5*(torch.cos(2*np.pi*x)+torch.cos(2*np.pi*y))) + np.e + 20
        ret = tmp0 + tmp1
        return ret


class Beale(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        np_rng = np.random.default_rng()
        self.theta = torch.nn.Parameter(torch.tensor(np_rng.normal(size=2), dtype=torch.float64))
        # [3,0.5] 0

    def forward(self):
        x,y = self.theta
        ret = (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
        return ret


model = Rosenbrock(num_parameter=10)
theta_optim = numpyqi.optimize.minimize(model, num_repeat=1, tol=1e-10, print_freq=20)
# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] 3.959370430722476e-21


model = Rastrigin(num_parameter=10)
theta_optim = numpyqi.optimize.minimize(model, num_repeat=1, tol=1e-10, print_freq=20)
# almost impossible to find the global minimum


model = Ackley()
theta_optim = numpyqi.optimize.minimize(model, num_repeat=1, tol=1e-10, print_freq=20)
# [0,0] 0


model = Beale()
theta_optim = numpyqi.optimize.minimize(model, num_repeat=1, tol=1e-10, print_freq=20)
# [3, 0.5] 0

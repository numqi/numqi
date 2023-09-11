import numpy as np
import torch
import scipy.optimize

import numqi


class AugmentedLagrangianModel(torch.nn.Module):
    def __init__(self, model, penalty_weight=1):
        super().__init__()
        self.model = model
        assert penalty_weight>=0
        self.penalty_weight = torch.tensor(penalty_weight, dtype=torch.float64)
        np_rng = np.random.default_rng()
        hf0 = lambda *x: torch.nn.Parameter(torch.tensor(np_rng.uniform(-1,1,size=x), dtype=torch.float64))
        if hasattr(model, 'constraint_equality'):
            with torch.no_grad():
                tmp0 = model.constraint_equality()
            assert isinstance(tmp0, torch.Tensor) and (tmp0.ndim==1) and (not torch.is_complex(tmp0))
            self.theta_lambda_equality = hf0(len(tmp0))
        else:
            self.theta_lambda_equality = None
        # TODO: inequality constraint
        if hasattr(model, 'constraint_inequality'):
            with torch.no_grad():
                tmp0 = model.constraint_inequality()
            assert isinstance(tmp0, torch.Tensor) and (tmp0.ndim==1) and (not torch.is_complex(tmp0))
            self.theta_lambda_inequality = hf0(len(tmp0))
        else:
            self.theta_lambda_inequality = None

        self._equality = None
        self._inequality = None

    def set_optimize_lambda(self, tag):
        if self.theta_lambda_equality is not None:
            self.theta_lambda_equality.requires_grad_(tag)
        if self.theta_lambda_inequality is not None:
            self.theta_lambda_inequality.requires_grad_(tag)

    def forward(self):
        fval = self.model()
        if self.theta_lambda_equality is not None:
            tmp0 = self.model.constraint_equality()
            self._equality = tmp0.detach()
            fval = fval + torch.dot(self.theta_lambda_equality, tmp0) + (self.penalty_weight/2) * torch.dot(tmp0, tmp0)
        if self.theta_lambda_inequality is not None:
            tmp0 = self.model.constraint_inequality()
            tmp0 = torch.maximum(tmp0, torch.zeros_like(tmp0))
            self._inequality = tmp0.detach()
            fval = fval + torch.dot(self.theta_lambda_inequality, tmp0) + (self.penalty_weight/2) * torch.dot(tmp0, tmp0)
        return fval


def augumented_lagrangian_minimize(model, num_step, tol=1e-7, penalty_weight=1, method='L-BFGS-B', maxiter=None, seed=None):
    model_AL = AugmentedLagrangianModel(model, penalty_weight)
    model_AL.set_optimize_lambda(False)
    hf_model = numqi.optimize.hf_model_wrapper(model_AL)
    options = dict() if maxiter is None else {'maxiter':maxiter}
    tag_equality = (model_AL.theta_lambda_equality is not None)
    tag_inequality = (model_AL.theta_lambda_inequality is not None)
    for ind0 in range(num_step):
        model_AL.set_optimize_lambda(False)
        theta0 = numqi.optimize.get_model_flat_parameter(model_AL)
        theta_optim = scipy.optimize.minimize(hf_model, theta0, jac=True, method=method, tol=tol, options=options)
        model_AL.set_optimize_lambda(True)

        if tag_equality and (model_AL.theta_lambda_equality.grad is not None):
            model_AL.theta_lambda_equality.grad.zero_()
        if tag_inequality and (model_AL.theta_lambda_inequality.grad is not None):
            model_AL.theta_lambda_inequality.grad.zero_()
        model_AL().backward()
        if tag_equality:
            model_AL.theta_lambda_equality.data[:] += model_AL.penalty_weight * model_AL.theta_lambda_equality.grad
        if tag_inequality:
            model_AL.theta_lambda_inequality.data[:] += model_AL.penalty_weight * model_AL.theta_lambda_inequality.grad

        message = f'[step={ind0}], fval={theta_optim.fun:.10f}'
        if tag_equality:
            tmp0 = torch.linalg.norm(model_AL._equality).item()
            tmp1 = torch.linalg.norm(model_AL.theta_lambda_equality.detach()).item()
            message = message + f' |equality|={tmp0:.7g}, |lambda(equality)|={tmp1:.7g}'
        if tag_inequality:
            tmp2 = torch.linalg.norm(model_AL._inequality).item()
            tmp3 = torch.linalg.norm(model_AL.theta_lambda_inequality.detach()).item()
            message = message + f' |inequality|={tmp2:.7g}, |lambda(inequality)|={tmp3:.7g}'
        print(message)
    model_AL.set_optimize_lambda(False)
    hf_model(theta_optim.x, tag_grad=False) #set theta and model.property (sometimes)
    return theta_optim


class DummyModel00(torch.nn.Module):
    def __init__(self):
        super().__init__()
        np_rng = np.random.default_rng()
        hf0 = lambda *x: torch.nn.Parameter(torch.tensor(np_rng.uniform(-1,1,size=x), dtype=torch.float64))
        self.theta = hf0(2)

    def forward(self):
        loss = self.theta[0]**2 + self.theta[1]**2
        return loss

    # def constraint_equality(self):
    #     ret = (self.theta[0] + self.theta[1] - 1).reshape(-1)
    #     return ret

    def constraint_inequality(self):
        # h(x) <= 0
        ret = (1 - self.theta[0] + self.theta[1]).reshape(-1)
        return ret


# https://en.wikipedia.org/wiki/Test_functions_for_optimization
class RosenBrockCubicLineModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        np_rng = np.random.default_rng()
        self.theta = torch.nn.Parameter(torch.tensor(np_rng.uniform(-2,2,size=2), dtype=torch.float64))

    def forward(self):
        x,y = self.theta
        loss = (1-x)**2 + 100*(y-x**2)**2
        return loss

    def constraint_inequality(self):
        x,y = self.theta
        ret = torch.stack([(x-1)**3-y+1, x+y-2]).reshape(-1)
        return ret

class RosenBrockDisk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        np_rng = np.random.default_rng()
        self.theta = torch.nn.Parameter(torch.tensor(np_rng.uniform(-2,2,size=2), dtype=torch.float64))

    def forward(self):
        x,y = self.theta
        loss = (1-x)**2 + 100*(y-x**2)**2
        return loss

    def constraint_inequality(self):
        x,y = self.theta
        ret = (x*x + y*y -  2).reshape(-1)
        return ret

class MishraBirdModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        np_rng = np.random.default_rng()
        self.theta = torch.nn.Parameter(torch.tensor(np_rng.uniform(-10,0,size=2), dtype=torch.float64))

    def forward(self):
        x,y = self.theta
        cx = torch.cos(x)
        sy = torch.sin(y)
        loss = sy * torch.exp((1-cx)**2) + cx * torch.exp((1-sy)**2) + (x-y)**2
        return loss

    def constraint_inequality(self):
        x,y = self.theta
        ret = ((x+5)**2 + (y+5)**2 - 25).reshape(-1)
        return ret

def demo_test_function():
    model = DummyModel00()
    theta_optim = augumented_lagrangian_minimize(model, num_step=30)

    model = RosenBrockCubicLineModel()
    theta_optim = augumented_lagrangian_minimize(model, num_step=30, penalty_weight=1, tol=1e-7)
    # f(1,1)=0

    model = RosenBrockDisk()
    theta_optim = augumented_lagrangian_minimize(model, num_step=30, penalty_weight=1, tol=1e-10)
    # f(1,1)=0

    model = MishraBirdModel()
    theta_optim = augumented_lagrangian_minimize(model, num_step=30, penalty_weight=1, tol=1e-7)
    # f(-3.1302468,-1.5821422)=-106.7645367

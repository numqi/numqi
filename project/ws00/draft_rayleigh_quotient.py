import numpy as np
import matplotlib.pyplot as plt
import torch

import numqi

np_rng = np.random.default_rng()

# not help
class SphereRemoveGrad(torch.autograd.Function):
    @staticmethod
    def forward(x):
        ret = x
        return ret

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        tmp0 = torch.dot(x.detach(), x.detach())
        grad_x = grad_output - (torch.dot(x, grad_output) / (tmp0)) * x
        return grad_x, None

class DummyMinEigen(torch.nn.Module):
    def __init__(self, mat):
        super().__init__()
        self.mat = torch.tensor(mat, dtype=torch.float64)
        self.theta = torch.nn.Parameter(torch.randn(mat.shape[0], dtype=torch.float64))
        # self.manifold = numqi.manifold.Sphere(mat.shape[0], dtype=torch.float64)
        self.vec = None

    def forward(self):
        theta = self.theta
        # theta = SphereRemoveGrad.apply(theta)
        vec = theta / torch.linalg.norm(theta)
        # vec = self.manifold()
        self.vec = vec.detach()
        loss = torch.vdot(vec, (self.mat @ vec)).real
        return loss

# see explanation in optimization algorithm on matrix manifold @proposition-2.1.2
N0 = 128
tmp0 = np_rng.normal(size=(N0,N0))
mat = (tmp0 + tmp0.T) / 2

model = DummyMinEigen(mat)
callback = numqi.optimize.MinimizeCallback(print_freq=1, extra_key=['grad_norm','path'], tag_print=False)
theta_optim = numqi.optimize.minimize(model, theta0='uniform', num_repeat=1, callback=callback, tol=1e-14)
# numqi.optimize.minimize_adam(model, num_step=1000, optim_args=('adam',0.01,0.001))
EVL = theta_optim.fun
EVC = model.vec.numpy()
EVL_ = np.linalg.eigvalsh(mat)[0]
print('error(EVL)', np.abs(EVL-EVL_))
print('mae(EVC)', np.abs(mat @ EVC - EVC * EVL).max())

fig,(ax0,ax1) = plt.subplots(1, 2, figsize=(8,4.5))
ax0.plot(np.array(callback.state['fval'])-EVL_, label='fval-optimal')
ax0.plot(callback.state['grad_norm'], label='norm(grad)')
ax0.grid()
ax0.legend()
ax0.set_yscale('log')
ax0.set_xlabel('step')
ax1.plot([np.linalg.norm(x) for x in callback.state['path']], label='norm(path)')
ax1.set_yscale('log')
ax1.grid()
ax1.set_xlabel('step')
ax1.set_ylabel('norm(path)')
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)


N0 = 128
tmp0 = np_rng.normal(size=(N0,N0))
mat = (tmp0 + tmp0.T) / 2

model = DummyMinEigen(mat)
hf_model = numqi.optimize.hf_model_wrapper(model)
x0 = numqi.optimize.get_model_flat_parameter(model)

fval0,grad0 = hf_model(x0)
hess = numqi.optimize.get_model_hessian(model)
delta0 = np.linalg.solve(hess, grad0)

assert np.abs(hess @ x0 + grad0).max() < 1e-10

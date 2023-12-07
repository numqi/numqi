import numpy as np
import matplotlib.pyplot as plt
import torch

import numqi


class DummyMinEigen(torch.nn.Module):
    def __init__(self, mat):
        super().__init__()
        self.mat = torch.tensor(mat, dtype=torch.float64)
        self.theta = torch.nn.Parameter(torch.randn(mat.shape[0], dtype=torch.float64))
        # self.manifold = numqi.manifold.Sphere(mat.shape[0], dtype=torch.float64)
        self.vec = None

    def forward(self):
        vec = self.theta / torch.linalg.norm(self.theta)
        # vec = self.manifold()
        self.vec = vec.detach()
        loss = torch.vdot(vec, (self.mat @ vec)).real
        return loss

np_rng = np.random.default_rng()

N0 = 1000
tmp0 = np_rng.normal(size=(N0,N0))
mat = (tmp0 + tmp0.T) / 2

model = DummyMinEigen(mat)
callback = numqi.optimize.MinimizeCallback(print_freq=1, extra_key='grad_norm', tag_print=False)
theta_optim = numqi.optimize.minimize(model, theta0='uniform', num_repeat=1, callback=callback, tol=1e-14)
EVL = theta_optim.fun
EVC = model.vec.numpy()
EVL_ = np.linalg.eigvalsh(mat)[0]
print('error(EVL)', np.abs(EVL-EVL_))
print('mae(EVC)', np.abs(mat @ EVC - EVC * EVL).max())

fig,(ax0,ax1) = plt.subplots(1, 2, figsize=(8,4.5))
ax0.plot(callback.state['fval'])
ax0.axhline(EVL_, linestyle=':', color='red', label='minimum eigen values')
ax0.legend()
ax0.set_xlabel('step')
ax0.set_ylabel('loss')
ax1.plot(callback.state['grad_norm'])
ax1.set_yscale('log')
ax1.set_xlabel('step')
ax1.set_ylabel('gradient norm')
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)

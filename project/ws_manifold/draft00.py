import numpy as np
import matplotlib.pyplot as plt
import torch

import numqi


class DummyMinEigen(torch.nn.Module):
    def __init__(self, dim:int):
        super().__init__()
        self.dim = int(dim)
        self.theta = torch.nn.Parameter(torch.randn(self.dim, dtype=torch.float64))
        self.mat = None
        self.vec = None

    def set_matrix(self, np0):
        assert np0.shape == (self.dim, self.dim)
        assert np.abs(np0.imag).max() < 1e-10
        assert np.abs(np0-np0.T).max() < 1e-10
        self.mat = torch.tensor(np0, dtype=torch.float64)

    def forward(self):
        vec = self.theta / torch.linalg.norm(self.theta)
        self.vec = vec.detach()
        loss = torch.vdot(vec, (self.mat @ vec)).real
        return loss

np_rng = np.random.default_rng()

N0 = 1000
tmp0 = np_rng.normal(size=(N0,N0))
mat = (tmp0 + tmp0.T) / 2

model = DummyMinEigen(N0)
model.set_matrix(mat)
callback = numqi.optimize.MinimizeCallback(print_freq=1, extra_key='grad_norm')
theta_optim = numqi.optimize.minimize(model, theta0='uniform', num_repeat=1, callback=callback, tol=1e-14)
EVL = theta_optim.fun
EVC = model.vec.numpy()
print(np.abs(mat @ EVC - EVC * EVL).max())
EVL_ = np.linalg.eigvalsh(mat)[0]

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

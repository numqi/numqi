import numpy as np
import torch

import numqi

# TODO test logm via eigen-decomposition and sqrtm

np_rng = np.random.default_rng(234)

dimA = 3
dimB = 3
kext = 16

# rho = numqi.random.rand_density_matrix(dimA*dimB, seed=np_rng)
shape = dimA*dimB, dimA*dimB-1
tmp0 = np_rng.normal(size=shape) + np_rng.normal(size=shape)
tmp1 = tmp0 @ tmp0.T.conj()
rho = tmp1 / np.trace(tmp1)

model = numqi.entangle.PureBosonicExt(dimA, dimB, kext=kext, distance_kind='ree')
model.set_dm_target(rho)
theta_optim = numqi.optimize.minimize(model, tol=1e-8, num_repeat=3, print_every_round=1, seed=np_rng, print_freq=200)
# [round=0] min(f)=0.005731086287363407, current(f)=0.005731086287363407
# [round=1] min(f)=0.005731086287363407, current(f)=0.005744899288925298
# [round=2] min(f)=0.005731086287363407, current(f)=0.0057466038596716285


# EVL,EVC = torch.linalg.eigh(x)
# tmp0 = torch.log(torch.maximum(torch.zeros(1, dtype=EVL.dtype, device=EVL.device), EVL))
# ret = ((EVC*tmp0) @ EVC.T.conj())


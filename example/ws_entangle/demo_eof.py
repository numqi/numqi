import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import numqi

np_rng = np.random.default_rng()

# werner state
dim = 3
alpha_list = np.linspace(-1, 1, 200)
ret_ = numqi.entangle.eof.get_eof_werner(dim, alpha_list)

dm_list = [numqi.state.Werner(dim, x) for x in alpha_list]
model = numqi.entangle.eof.EntanglementFormationModel(dim, dim, 2*dim*dim)
ret0 = []
kwargs = dict(num_repeat=3, print_freq=0, tol=1e-7, print_every_round=0)
for x in tqdm(dm_list):
    model.set_density_matrix(x)
    theta_optim = numqi.optimize.minimize(model, **kwargs)
    ret0.append(theta_optim.fun)
ret0 = np.array(ret0)

fig,ax = plt.subplots()
ax.plot(alpha_list, ret_, 'x', label='analytical')
ax.plot(alpha_list, ret0, label='variational')
ax.set_xlabel('alpha')
ax.set_yscale('log')
ax.set_ylabel('EOF')
ax.set_title(f'isotropic({dim})')
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)

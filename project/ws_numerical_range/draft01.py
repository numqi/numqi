import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import numqi

np_rng = np.random.default_rng()


class MaxEigenDegeneracyModel(torch.nn.Module):
    def __init__(self, matrix_space, indI=-3):
        super().__init__()
        assert matrix_space.ndim==3
        assert np.abs(matrix_space-matrix_space.transpose(0,2,1).conj()).max() < 1e-10
        self.matrix_space = torch.tensor(matrix_space)
        self.matA = torch.tensor(matrix_space.transpose(1,2,0), dtype=torch.complex128)
        N0 = matrix_space.shape[0]
        self.indI = indI
        tmp0 = np.random.default_rng().uniform(-1, 1, size=N0)
        self.theta = torch.nn.Parameter(torch.tensor(tmp0/np.linalg.norm(tmp0), dtype=torch.float64))

    def forward(self):
        tmp0 = self.theta / torch.linalg.norm(self.theta)
        EVL = torch.linalg.eigvalsh(self.matA @ tmp0.to(self.matA.dtype))
        loss = (EVL[-1] - EVL[self.indI])**2
        return loss


dimA = 2
dimB = 2
kext = 4

Brsab = numqi.dicke.get_partial_trace_ABk_to_AB_index(kext, dimB,  return_tensor=True)

matG = numqi.gellmann.all_gellmann_matrix(dimA*dimB, with_I=False)

matGext = np.einsum(matG.reshape(-1,dimA,dimB,dimA,dimB), [0,1,2,3,4],
        Brsab, [2,4,5,6], [0,1,5,3,6], optimize=True).reshape(matG.shape[0], dimA*Brsab.shape[2], -1)
model = MaxEigenDegeneracyModel(matGext, indI=-3)
model_pureb = numqi.entangle.PureBosonicExt(dimA, dimB, kext, distance_kind='ree')

theta_optim = numqi.optimize.minimize(model, num_repeat=1, tol=1e-12, print_freq=10)

for ind_round in range(100):
    loss = numqi.optimize.minimize_adam(model, num_step=1000, theta0='uniform')
    theta0 = numqi.optimize.get_model_flat_parameter(model)
    theta_optim = numqi.optimize.minimize(model, theta0=theta0, num_repeat=1, tol=1e-12, print_freq=40)

    tmp0 = model.theta.detach().numpy().copy()
    coeff = tmp0 / np.linalg.norm(tmp0)
    EVL = np.linalg.eigvalsh(np.einsum(coeff, [0], matGext, [0,1,2], [1,2], optimize=True))
    # print(f'[ind_round={ind_round}] EVL:', EVL.tolist())

    tmp0 = numqi.gellmann.gellmann_basis_to_dm(coeff)
    beta_u = numqi.entangle.get_density_matrix_boundary(tmp0)[1]
    dm_target = numqi.entangle.hf_interpolate_dm(tmp0, beta=beta_u)
    model_pureb.set_dm_target(dm_target)
    # ree = numqi.optimize.minimize(model_pureb, num_repeat=3, tol=1e-10, print_every_round=0).fun
    ree = numqi.entangle.get_ABk_symmetric_extension_ree(dm_target, (dimA,dimB), kext, use_ppt=False, use_tqdm=True)
    print(f'[ind_round={ind_round}] REE:', ree)
    if ree>1e-3:
        break

# some random direction
for ind_round in range(100):
    tmp0 = np_rng.uniform(-1, 1, size=len(matG))
    coeff = tmp0 / np.linalg.norm(tmp0)
    EVL = np.linalg.eigvalsh(np.einsum(coeff, [0], matGext, [0,1,2], [1,2], optimize=True))
    # print(f'[ind_round={ind_round}] EVL:', EVL.tolist())

    tmp0 = numqi.gellmann.gellmann_basis_to_dm(coeff)
    beta_u = numqi.entangle.get_density_matrix_boundary(tmp0)[1]
    dm_target = numqi.entangle.hf_interpolate_dm(tmp0, beta=beta_u)
    model_pureb.set_dm_target(dm_target)
    # ree = numqi.optimize.minimize(model_pureb, num_repeat=3, tol=1e-10, print_every_round=0).fun
    ree = numqi.entangle.get_ABk_symmetric_extension_ree(dm_target, (dimA,dimB), kext, use_ppt=False, use_tqdm=True)
    print(f'[ind_round={ind_round}] REE:', ree)
    if ree > 1e-3:
        break


beta_kext = model_pureb.get_boundary(dm_target, xtol=1e-4, threshold=1e-7, converge_tol=1e-10, num_repeat=3)
# beta_ppt = numqi.entangle.get_ppt_boundary(dm_target, (dimA,dimB))[1]
beta_list = np.linspace(beta_kext-0.01, min(beta_u,beta_kext+0.1), 100)
dm_target_list = [numqi.entangle.hf_interpolate_dm(tmp0, beta=x) for x in beta_list]

ree_pureb = []
for dm_i in tqdm(dm_target_list):
    model_pureb.set_dm_target(dm_i)
    ree_pureb.append(numqi.optimize.minimize(model_pureb, num_repeat=3, tol=1e-10, print_every_round=0).fun)
ree_pureb = np.array(ree_pureb)

ree_irrep = numqi.entangle.get_ABk_symmetric_extension_ree(dm_target_list, (dimA,dimB), kext, use_ppt=False, use_tqdm=True)

fig, ax = plt.subplots()
ind0 = slice(None,None)
ax.plot(beta_list[ind0], ree_pureb[ind0], 'x', label=f'PureB(k={kext})')
ax.plot(beta_list[ind0], ree_irrep[ind0], label=f'BosonExt(k={kext})')
ax.set_title(f'bipartite {dimA}x{dimB}')
ax.set_xlabel(r'$\beta$')
ax.set_ylabel('REE')
ax.set_ylim(1e-12, None)
ax.set_yscale('log')
ax.legend()
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)

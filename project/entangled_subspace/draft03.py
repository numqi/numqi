import numpy as np
import torch
from tqdm import tqdm
import opt_einsum
import matplotlib.pyplot as plt

import numqi


class DensityMatrixGMEModel(torch.nn.Module):
    def __init__(self, dim_list:tuple[int], num_ensemble:int, rank:int, CPrank:int=1):
        super().__init__()
        dim_list = tuple(int(x) for x in dim_list)
        assert (len(dim_list)>=2) and all(x>=2 for x in dim_list)
        self.dim_list = dim_list
        N0 = np.prod(np.array(dim_list))
        self.num_ensemble = int(num_ensemble)
        assert rank<=N0
        self.rank = int(rank)
        assert CPrank>=1
        self.CPrank = int(CPrank)

        self.manifold_stiefel = numqi.manifold.Stiefel(num_ensemble, rank, dtype=torch.complex128, method='sqrtm')
        # methods='QR' seems really bad
        self.manifold_psi = torch.nn.ModuleList([numqi.manifold.Sphere(x, batch_size=num_ensemble*CPrank, dtype=torch.complex128) for x in dim_list])
        if CPrank>1:
            self.manifold_coeff = numqi.manifold.PositiveReal(num_ensemble*CPrank, dtype=torch.float64)
            N1 = len(dim_list)
            tmp0 = [(num_ensemble,rank),(num_ensemble,rank)] + [(num_ensemble,rank,x) for x in dim_list] + [(num_ensemble,rank,x) for x in dim_list]
            tmp1 = [(N1,N1+1),(N1,N1+2)] + [(N1,N1+1,x) for x in range(N1)] + [(N1,N1+2,x) for x in range(N1)]
            self.contract_psi_psi = opt_einsum.contract_expression(*[y for x in zip(tmp0,tmp1) for y in x], [N1])

        self._sqrt_rho = None
        self.contract_expr = None
        self.contract_coeff = None

    def set_density_matrix(self, rho:np.ndarray):
        N0 = np.prod(np.array(self.dim_list))
        assert rho.shape == (N0, N0)
        assert np.abs(rho-rho.T.conj()).max() < 1e-10
        EVL,EVC = np.linalg.eigh(rho)
        EVL = np.maximum(0, EVL[-self.rank:])
        assert abs(EVL.sum()-1) < 1e-10
        EVC = EVC[:,-self.rank:]
        tmp0 = (EVC * np.sqrt(EVL)).reshape(*self.dim_list, self.rank)
        self._sqrt_rho = torch.tensor(tmp0, dtype=torch.complex128)
        N1 = len(self.dim_list)
        if self.CPrank==1:
            tmp0 = [(N1+1,x) for x in range(N1)]
            tmp1 = [(self.num_ensemble,x) for x in self.dim_list]
            tmp2 = [y for x in zip(tmp1,tmp0) for y in x]
            self.contract_expr = opt_einsum.contract_expression(self._sqrt_rho, tuple(range(N1+1)),
                                [self.num_ensemble,self.rank], (N1+1,N1), *tmp2, (N1+1,), constants=[0])
        else:
            tmp0 = [((N1+1,N1+2))] + [(N1+1,N1+2,x) for x in range(N1)]
            tmp1 = [(self.num_ensemble,self.CPrank)] + [(self.num_ensemble,self.CPrank,x) for x in self.dim_list]
            tmp2 = [y for x in zip(tmp1,tmp0) for y in x]
            self.contract_expr = opt_einsum.contract_expression(self._sqrt_rho, tuple(range(N1+1)),
                                [self.num_ensemble,self.rank], (N1+1,N1), *tmp2, (N1+1,), constants=[0])

    def forward(self):
        matX = self.manifold_stiefel()
        psi_list = [x() for x in self.manifold_psi]
        if self.CPrank>1:
            coeff = self.manifold_coeff().reshape(self.num_ensemble, self.CPrank)
            psi_list = [x.reshape(self.num_ensemble,self.CPrank,-1) for x in psi_list]
            psi_conj_list = [x.conj().resolve_conj() for x in psi_list]
            psi_psi = self.contract_psi_psi(coeff, coeff, *psi_list, *psi_conj_list).real
            coeff = coeff / torch.sqrt(psi_psi).reshape(-1,1)
            tmp2 = self.contract_expr(matX, coeff, *psi_list, backend='torch')
        else:
            tmp2 = self.contract_expr(matX, *psi_list, backend='torch')
        loss = 1-torch.vdot(tmp2,tmp2).real
        return loss

np_rng = np.random.default_rng()

## W state
psiW = numqi.state.W(3)
rhoW = psiW.reshape(-1,1) * psiW.conj()
model = DensityMatrixGMEModel(dim_list=[2,2,2], num_ensemble=4, rank=2)
model.set_density_matrix(rhoW)
loss = model()
theta_optim = numqi.optimize.minimize(model, num_repeat=3, tol=1e-10)


## Werner
alpha_list = np.linspace(0,1,32)[1:-1]
dim = 3
num_ensemble = 27
rank = 9

model = DensityMatrixGMEModel([dim,dim], num_ensemble, rank)
ret = []
for alpha_i in tqdm(alpha_list):
    model.set_density_matrix(numqi.state.Werner(d=3, alpha=alpha_i))
    ret.append(numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
ret = np.array(ret)

fig,ax = plt.subplots()
ax.plot(alpha_list, ret)
tmp0 = dim - (1-dim*dim) / (alpha_list - dim)
tmp1 = 0.5*(1-np.sqrt(np.maximum(0,1-tmp0*tmp0))) #https://doi.org/10.1103/PhysRevA.68.042307
ax.plot(alpha_list, tmp1, 'x')
ax.axvline(1/dim, color='k', linestyle='--')
ax.set_yscale('log')
fig.savefig('tbd00.png', dpi=100)


## sigma=2
dimA = 4
dimB = 4
tmp0 = [
    [(0,0,1), (1,1,1), (2,2,1), (3,3,1)],
    [(0,1,1), (1,2,1), (2,3,1), (3,0,1)],
    [(0,2,1), (1,3,1), (2,0,1), (3,1,-1)],
]
matrix_subspace = np.stack([numqi.matrix_space.build_matrix_with_index_value(dimA, dimB, x) for x in tmp0])
rho = np.einsum(matrix_subspace, [0,1,2], matrix_subspace.conj(), [0,3,4], [1,2,3,4], optimize=True).reshape(dimA*dimB,dimA*dimB) / 12
rank = (np.linalg.eigvalsh(rho)>1e-5).sum()
CPrank = 2
model = DensityMatrixGMEModel([dimA,dimB], num_ensemble=32, rank=rank, CPrank=CPrank)
model.set_density_matrix(rho)
loss = model()
theta_optim = numqi.optimize.minimize(model, num_repeat=10, tol=1e-10)


## upb
rho_bes = numqi.entangle.load_upb('tiles', return_bes=True)[1]
alpha_list = np.linspace(0,1,32)[1:-1]
dim = 3
num_ensemble = 27
rank = 9

model = DensityMatrixGMEModel([dim,dim], num_ensemble, rank)
ret = []
for alpha_i in tqdm(alpha_list):
    model.set_density_matrix(numqi.entangle.hf_interpolate_dm(rho_bes, alpha=alpha_i))
    ret.append(numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
ret = np.array(ret)

fig,ax = plt.subplots()
ax.plot(alpha_list, ret)
ax.axvline(0.8647, color='k', linestyle='--')
ax.set_yscale('log')
fig.savefig('tbd00.png', dpi=100)


## stiefel manifold, polar decomposition
# https://openreview.net/forum?id=5mtwoRNzjm

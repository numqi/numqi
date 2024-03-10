import numpy as np
import torch
import opt_einsum
from tqdm import tqdm
import matplotlib.pyplot as plt

import numqi

np_rng = np.random.default_rng()

# linear entropy http://dx.doi.org/10.1103/PhysRevLett.114.160501

def test_linear_entropy_tensor_network():
    dim0 = 3
    dim1 = 4

    rho = numqi.random.rand_density_matrix(dim0*dim1)
    EVL,EVC = np.linalg.eigh(rho)
    assert np.abs(rho - (EVC*EVL) @ EVC.T.conj()).max() < 1e-10

    manifold = numqi.manifold.Stiefel(2*dim0*dim1, rank=dim0*dim1, dtype=torch.complex128)
    mat_st = manifold().detach().numpy()

    z0 = (EVC*np.sqrt(EVL)) @ mat_st.T
    plist = np.linalg.norm(z0, ord=2, axis=0)**2
    psilist = (z0 / np.sqrt(plist)).T
    z1 = np.einsum(plist, [0], psilist, [0,1], psilist.conj(), [0,2], [1,2], optimize=True)
    assert np.abs(z1 - rho).max() < 1e-10
    tmp0 = psilist.reshape(-1, dim0, dim1)
    rdm = np.einsum(tmp0, [0,1,2], tmp0.conj(), [0,3,2], [0,1,3], optimize=True)
    tmp1 = np.linalg.norm(rdm, ord='fro', axis=(1,2))**2
    ret_ = np.dot(plist, tmp1)

    tmp0 = (EVC * np.sqrt(EVL)).reshape(dim0, dim1, -1)
    tmp1 = np.einsum(tmp0, [0,3,4], tmp0.conj(), [1,3,5], mat_st, [2,4], mat_st.conj(), [2,5], [2,0,1], optimize=True)
    rdm1 = tmp1 / np.trace(tmp1, axis1=1, axis2=2).reshape(-1,1,1)
    assert np.abs(rdm1 - rdm).max() < 1e-10
    tmp2 = np.linalg.norm(tmp1, ord='fro', axis=(1,2))**2
    ret0 = (tmp2 / np.trace(tmp1, axis1=1, axis2=2)).sum()
    # ret0 = np.dot(plist, tmp2)
    assert abs(ret_ - ret0) < 1e-10


class DensityMatrixLinearEntropyModel(torch.nn.Module):
    def __init__(self, dim:tuple[int], num_ensemble:int, rank:int=None, kind:str='convex'):
        super().__init__()
        assert kind in {'convex','concave'}
        self.dtype = torch.float64
        self.cdtype = torch.complex128
        dim = tuple(int(x) for x in dim)
        assert (len(dim)==2) and all(x>=2 for x in dim)
        self.dim0 = dim[0]
        self.dim1 = dim[1]
        self.num_ensemble = int(num_ensemble)
        if rank is None:
            rank = dim[0]*dim[1]
        assert rank<=dim[0]*dim[1]
        self.rank = int(rank)
        self.manifold_stiefel = numqi.manifold.Stiefel(num_ensemble, rank, dtype=self.cdtype)

        self._sqrt_rho = None
        self._eps = torch.tensor(torch.finfo(self.dtype).eps, dtype=self.dtype)
        self._sign = 1 if (kind=='convex') else -1

    def set_density_matrix(self, rho:np.ndarray):
        assert rho.shape==(self.dim0*self.dim1, self.dim0*self.dim1)
        assert np.abs(rho-rho.T.conj()).max() < 1e-10
        EVL,EVC = np.linalg.eigh(rho)
        EVL = np.maximum(0, EVL[-self.rank:])
        assert abs(EVL.sum()-1) < 1e-10
        EVC = EVC[:,-self.rank:]
        tmp0 = (EVC * np.sqrt(EVL)).reshape(self.dim0, self.dim1, -1)
        self._sqrt_rho = torch.tensor(tmp0, dtype=self.cdtype)
        tmp0 = self._sqrt_rho.conj().resolve_conj()
        if self.dim0<=self.dim1:
            self.contract_expr = opt_einsum.contract_expression(self._sqrt_rho, [0,3,4], tmp0, [1,3,5],
                                [self.num_ensemble,self.rank], [2,4], [self.num_ensemble,self.rank], [2,5], [2,0,1], constants=[0,1])
        else:
            self.contract_expr = opt_einsum.contract_expression(self._sqrt_rho, [3,0,4], tmp0, [3,1,5],
                                [self.num_ensemble,self.rank], [2,4], [self.num_ensemble,self.rank], [2,5], [2,0,1], constants=[0,1])
        tmp0 = min(self.dim0, self.dim1)
        self.contract_expr1 = opt_einsum.contract_expression([self.num_ensemble,tmp0,tmp0], [0,1,2], [self.num_ensemble,tmp0,tmp0], [0,1,2], [0])

    def forward(self):
        mat_st = self.manifold_stiefel()
        tmp0 = self.contract_expr(mat_st, mat_st.conj(), backend='torch')
        # rdm = tmp0 / torch.einsum(tmp0, [0,1,1], [0]).reshape(-1,1,1)
        tmp1 = torch.maximum(self._eps, torch.einsum(tmp0, [0,1,1], [0]).real)
        loss = self._sign * (1 - (self.contract_expr1(tmp0, tmp0.conj()).real / tmp1).sum())
        return loss


alpha_list = np.linspace(0, 1, 100)
dim = 3

model = DensityMatrixLinearEntropyModel([dim,dim], num_ensemble=27, kind='convex')
ret0 = []
for alpha_i in tqdm(alpha_list):
    model.set_density_matrix(numqi.state.Werner(dim, alpha=alpha_i))
    ret0.append(numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
ret0 = np.array(ret0)

model = DensityMatrixLinearEntropyModel([dim,dim], num_ensemble=27, kind='concave')
ret1 = []
for alpha_i in tqdm(alpha_list):
    model.set_density_matrix(numqi.state.Werner(dim, alpha=alpha_i))
    ret1.append(-numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
ret1 = np.array(ret1)


fig,ax = plt.subplots()
ax.axvline(1/dim, color='r')
ax.plot(alpha_list, ret0, label='convex')
ax.plot(alpha_list, ret1, label='concave')
ax.legend()
# ax.set_yscale('log')
ax.set_xlabel('alpha')
ax.set_ylabel('linear entropy')
ax.set_title(f'Werner({dim})')
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)

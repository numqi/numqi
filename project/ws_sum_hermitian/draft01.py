import numpy as np
import scipy.linalg
import torch

import numqi
import numqi._torch_op

np_rng = np.random.default_rng()


# TODO manifold
class SpaceDistanceModel(torch.nn.Module):
    def __init__(self, dim, degree, dtype:str='complex128'):
        super().__init__()
        assert (degree>0) and (dim>degree)
        self.dim = int(dim)
        self.degree = int(degree)
        assert dtype=='complex128' # TODO real symmetric, complex hermitian
        self.dtype = torch.complex128

        np_rng = np.random.default_rng()
        hf0 = lambda *x: torch.nn.Parameter(torch.tensor(np_rng.uniform(-1,1,size=x), dtype=torch.float64))
        self.thetaU0 = hf0(degree, degree)
        self.thetaU1 = hf0(dim-degree, dim-degree)

        self.space0 = None
        self.space0_orth = None
        self.space1 = None
        self.space1_orth = None
        self.matH = None
        ## bad, PSD matrix only
        # self.torch_logm_op = numqi._torch_op.PSDMatrixLogm(num_sqrtm=6, pade_order=8)


    def set_space(self, space0, space1):
        assert space0.shape==(self.degree, self.dim)
        assert space1.shape==(self.degree, self.dim)
        tmp0 = np.eye(self.degree)
        assert np.abs(space0.conj() @ space0.T - tmp0).max() < 1e-10
        assert np.abs(space1.conj() @ space1.T - tmp0).max() < 1e-10
        self.space0 = torch.tensor(space0, dtype=self.dtype)
        self.space1 = torch.tensor(space1, dtype=self.dtype)
        tmp0 = np.linalg.eigh(np.eye(self.dim) - space0.T @ space0.conj())[1][:, self.degree:].T
        self.space0_orth = torch.tensor(tmp0, dtype=self.dtype)
        tmp0 = np.linalg.eigh(np.eye(self.dim) - space1.T @ space1.conj())[1][:, self.degree:].T
        self.space1_orth = torch.tensor(tmp0, dtype=self.dtype)

    def forward(self, tag_set_matH=False):
        matA = numqi.param.real_matrix_to_special_unitary(self.thetaU0)
        matB = numqi.param.real_matrix_to_special_unitary(self.thetaU1)
        matU = self.space1.T @ matA @ self.space0.conj() + self.space1_orth.T @ matB @ self.space0_orth.conj()
        if tag_set_matH:
            tmp0,EVC = torch.linalg.eig(matU)
            EVC = EVC.detach()
            tmp0 = torch.angle(tmp0)
            EVL = tmp0 - tmp0.mean()
            self.matH = (EVC * EVL.detach()) @ EVC.T.conj()
        else:
            tmp0 = torch.angle(torch.linalg.eigvals(matU)) #(-pi,pi]
            EVL = tmp0 - tmp0.mean()
        loss = torch.dot(EVL, EVL)
        return loss

    def get_unitary(self):
        with torch.no_grad():
            self(tag_set_matH=True)
        ret = scipy.linalg.expm(1j*self.matH.numpy())
        return ret


dim = 5
degree = 3

space0 = numqi.random.rand_unitary_matrix(dim, seed=np_rng)[:degree]

tmp0 = np_rng.normal(size=(dim,dim)) + 1j*np_rng.normal(size=(dim,dim))
tmp0 = tmp0 + tmp0.T.conj()
tmp0 = tmp0 - (np.trace(tmp0)/dim)*np.eye(dim)
matH = tmp0 * (0.1/np.linalg.norm(tmp0, ord='fro'))
space1 = space0 @ scipy.linalg.expm(1j*matH)
# space1 = numqi.random.rand_unitary_matrix(degree) @ space1

model = SpaceDistanceModel(dim, degree)
model.set_space(space0, space1)
theta_optim = numqi.optimize.minimize(model, tol=1e-7, num_repeat=30)

# loss_history = []
# for _ in range(100):
#     loss = numqi.optimize.minimize_adam(model, num_step=1000, theta0='uniform', optim_args=('adam',0.01), tqdm_update_freq=0)
#     loss_history.append(loss)
#     print(loss, min(loss_history))

z0 = model.get_unitary()
z0 = scipy.linalg.expm(1j*matH.T)
z233 = np.abs((space0 @ z0.T) @ space1.T.conj())**2
assert np.abs(z233.sum(axis=0)-1).max() < 1e-7
assert np.abs(z233.sum(axis=1)-1).max() < 1e-7


# EVL,EVC = np.linalg.eigh(np.eye(dim) - space0.T @ space0.conj())
# space0_orth = EVC[:, degree:].T
# EVL,EVC = np.linalg.eigh(np.eye(dim) - space1.T @ space1.conj())
# space1_orth = EVC[:, degree:].T

# z0 = numqi.random.rand_unitary_matrix(degree, seed=np_rng)
# z1 = numqi.random.rand_unitary_matrix(dim-degree, seed=np_rng)
# matU = space1.T @ z0 @ space0.conj() + space1_orth.T @ z1 @ space0_orth.conj()
# z233 = np.abs((space0 @ matU.T) @ space1.T.conj())**2
# print(z233.sum(axis=0), z233.sum(axis=1))
# matH = scipy.linalg.logm(matU)/1j
# matH = matH - (np.trace(matH)/dim)*np.eye(dim)
# tmp0 = matH.reshape(-1)
# loss = np.dot(tmp0, tmp0.conj()).real

import numpy as np
import torch

import numqi

np_rng = np.random.default_rng()

pcoeff = np.array([-0.91255, 0.32392, 0.24965])
qcoeff = np.array([-0.91795, -0.03220, 0.20626, -0.28660, -0.17789])

basis = numqi.dicke.get_dicke_basis(4, dim=2)[::-1]
for x in basis:
    print(np.nonzero(x)[0])

state0 = pcoeff[0]*basis[0] + pcoeff[1]*basis[2] + pcoeff[2]*basis[4]
state1 = qcoeff @ basis
state0 /= np.linalg.norm(state0)
state1 /= np.linalg.norm(state1)

tmp0 = state0.reshape(4,-1)
rdm0 = tmp0 @ tmp0.T.conj()
tmp1 = state1.reshape(4,-1)
rdm1 = tmp1 @ tmp1.T.conj()
np.abs(rdm0 - rdm1).max()

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pcoeff = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
        self.qcoeff = torch.nn.Parameter(torch.zeros(10, dtype=torch.float64))
        self.basis = torch.tensor(numqi.dicke.get_dicke_basis(4, dim=2)[::-1].copy(), dtype=torch.complex128)

    def forward(self):
        pcoeff,qcoeff = self._get_coeff()
        state0 = pcoeff[0]*self.basis[0] + pcoeff[1]*self.basis[2] + pcoeff[2]*self.basis[4]
        state1 = qcoeff @ self.basis
        tmp0 = state0.reshape(4,4)
        tmp1 = state1.reshape(4,4)
        tmp2 = (tmp0 @ tmp0.T.conj() - tmp1 @ tmp1.T.conj()).reshape(-1)
        loss = torch.vdot(tmp2, tmp2).real
        return loss

    def _get_coeff(self):
        pcoeff = self.pcoeff / torch.linalg.norm(self.pcoeff)
        qcoeff = self.qcoeff / torch.linalg.norm(self.qcoeff)
        qcoeff = torch.complex(qcoeff[:5], 0*qcoeff[5:])
        return pcoeff,qcoeff

    def get_coeff(self):
        with torch.no_grad():
            pcoeff,qcoeff = self._get_coeff()
            pcoeff = pcoeff.numpy()
            qcoeff = qcoeff.numpy()
            qcoeff /= np.exp(1j*np.angle(qcoeff[2]))
        return pcoeff,qcoeff

model = DummyModel()
theta0 = np.concatenate([pcoeff, qcoeff.real, qcoeff.imag])
numqi.optimize.set_model_flat_parameter(model, theta0)
model()
z0_list = []
for _ in range(10):
    tmp0 = theta0 + np_rng.uniform(-3e-3, 3e-3, size=theta0.shape)
    theta_optim = numqi.optimize.minimize(model, theta0=tmp0, num_repeat=1, tol=1e-25)
    z0,z1 = model.get_coeff()
    print(z0.tolist())
    print(z1.tolist())
    z0_list.append(z0)
# 0.9125528072653093, 0.3239316864746705, -0.24963100057861232
# 0.9179494546971481, 0.032200387931413534j, 0.20627576340298437, -0.28659936775379397j,0.17788492199710595
for x in z0_list:
    print(x.tolist())

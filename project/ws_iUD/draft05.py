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
tmp0 = theta0 + 0*np_rng.uniform(-5e-2, 5e-2, size=theta0.shape)
theta_optim = numqi.optimize.minimize(model, theta0=tmp0, num_repeat=1, tol=1e-25)
model.get_coeff()[0].tolist()


line_list = [model.get_coeff()[0].tolist()]
data_list = [numqi.optimize.get_model_flat_parameter(model)]
fidelity_list = []
for index in range(100000):
    tmp0 = data_list[-1] + np.array([np_rng.uniform(0.0003, 0.001)]+[0]*(len(theta0)-1))
    theta_optim = numqi.optimize.minimize(model, theta0=tmp0, num_repeat=1, tol=1e-25, print_every_round=0)
    tmpP,tmpQ = model.get_coeff()
    fidelity = np.dot(tmpP, tmpQ[[0,2,4]].real)
    if index%500==0:
        print(index)
    if (fidelity < 0.999) and (np.dot(tmpP, line_list[-1])>0.999) and (tmpP[0]>line_list[-1][0]):
        print(f'[{index}][fidelity={fidelity:.6f}][loss={theta_optim.fun:.3g}] {np.around(tmpP,7).tolist()}')
        line_list.append(tmpP.tolist())
        data_list.append(numqi.optimize.get_model_flat_parameter(model))
        fidelity_list.append(fidelity)
    else:
        break

x0,x1,x2 = line_list[-1]
basis = numqi.dicke.get_dicke_basis(4, dim=2)[::-1]
tmp0 = (x0*basis[0] + x1*basis[2] + x2*basis[4]).reshape(4,4)
np.linalg.eigvalsh(tmp0 @ tmp0.T).tolist()

z0_list = []
z1_list = []
for _ in range(100):
    tmp0 = theta0 + np_rng.uniform(-5e-2, 5e-2, size=theta0.shape)
    theta_optim = numqi.optimize.minimize(model, theta0=tmp0, num_repeat=1, tol=1e-25)
    z0,z1 = model.get_coeff()
    print(z0.tolist())
    print(z1.tolist())
    z1_list.append(theta_optim.fun)
    z0_list.append(z0)
print(max(z1_list))
z2 = np.stack(z0_list)
z3 = (z2.T @ z2) / len(z2)
EVL,EVC = np.linalg.eigh(z3)
# 0.9125528072653093, 0.3239316864746705, -0.24963100057861232
# 0.9179494546971481, 0.032200387931413534j, 0.20627576340298437, -0.28659936775379397j,0.17788492199710595
for x in z0_list:
    print(x.tolist())


abc = np.array([0.010069255168374126, -0.5923864433733188, 0.8055910326014458])
basis = numqi.dicke.get_dicke_basis(4, dim=2)[::-1]
tmp0 = np.linalg.eigh(np.eye(3) - abc.reshape(-1,1)*abc)[1][:,1:].T
theta_list = np.linspace(0, 2*np.pi, 100)
z4 = np.cos(theta_list.reshape(-1,1))*tmp0[0] + np.sin(theta_list.reshape(-1,1))*tmp0[1]

z5_list = []
for x0,x1,x2 in z4:
    tmp0 = (x0*basis[0] + x1*basis[2] + x2*basis[4]).reshape(4,4)
    z5_list.append(np.linalg.eigvalsh(tmp0 @ tmp0.T).tolist())
z5_list = np.stack(z5_list)
import matplotlib.pyplot as plt
fig,ax = plt.subplots()
ax.plot(theta_list, z5_list)
fig.savefig('tbd00.png', dpi=200)


class DummyModel233(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
        self.basis = torch.tensor(numqi.dicke.get_dicke_basis(4, dim=2)[::-1].copy(), dtype=torch.float64)

    def forward(self):
        coeff = self.theta / torch.linalg.norm(self.theta)
        state = (coeff[0]*self.basis[0] + coeff[1]*self.basis[2] + coeff[2]*self.basis[4]).reshape(4,4)
        tmp0 = (state @ state.T) * 3
        tmp1 = (tmp0@tmp0 - tmp0).reshape(-1)
        loss = torch.dot(tmp1,tmp1)
        return loss

model = DummyModel233()
theta_optim = numqi.optimize.minimize(model, theta0='uniform', num_repeat=3, tol=1e-20)
tmp0 = model.theta.detach().numpy()
z0 = tmp0 / np.linalg.norm(tmp0)
print(z0.tolist())



class DummyModel(torch.nn.Module):
    def __init__(self, pcoeff):
        super().__init__()
        self.basis = torch.tensor(numqi.dicke.get_dicke_basis(4, dim=2)[::-1].copy(), dtype=torch.float64)
        pcoeff = torch.tensor(pcoeff/np.linalg.norm(pcoeff), dtype=torch.float64)
        pstate = (pcoeff[0]*self.basis[0] + pcoeff[1]*self.basis[2] + pcoeff[2]*self.basis[4]).reshape(4,4)
        self.prdm = pstate @ pstate.T
        self.pcoeff = pcoeff

        self.qcoeff = torch.nn.Parameter(torch.ones(5, dtype=torch.float64))

    def forward(self):
        qcoeff = self.qcoeff / torch.linalg.norm(self.qcoeff)
        qstate = (qcoeff @ self.basis).reshape(4,4)
        tmp1 = (self.prdm - qstate @ qstate.T).reshape(-1)
        fidelity = (self.pcoeff[0] * qcoeff[0] + self.pcoeff[1] * qcoeff[2] + self.pcoeff[2] * qcoeff[4])**2
        loss = torch.dot(tmp1, tmp1).real + fidelity
        return loss

tmp0 = np.array([-0.5, -np.sqrt(2)/2, 0.5])
model = DummyModel(tmp0)
theta_optim = numqi.optimize.minimize(model, theta0='uniform', num_repeat=3, tol=1e-20)
tmp0 = model.qcoeff.detach().numpy()
tmp0 = tmp0 / np.linalg.norm(tmp0)
print(tmp0)

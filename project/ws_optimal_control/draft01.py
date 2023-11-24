# https://docs.q-ctrl.com/boulder-opal/user-guides/how-to-automate-closed-loop-hardware-optimization
import numpy as np
import scipy.linalg
import torch
import matplotlib.pyplot as plt

import numqi

USE_QCTRL = True

def generate_exp_data(omega, time_duration, target_gate, ham_model, ham_unknown, np_rng):
    assert (omega.ndim==1) or (omega.ndim==2)
    is_single_item = omega.ndim==1
    omega = omega.reshape(-1, omega.shape[-1])
    batch_size,num_segment = omega.shape
    delta_t = time_duration / num_segment
    N1 = target_gate.shape[0]
    unitary = np.eye(N1)
    for x in range(num_segment):
        if USE_QCTRL:
            unitary = scipy.linalg.expm((-1j*omega[:,x]*delta_t).reshape(-1,1,1) * (ham_model + ham_unknown)) @ unitary
        else:
            unitary = scipy.linalg.expm(-1j*delta_t*(omega[:,x].reshape(-1,1,1) * ham_model + ham_unknown)) @ unitary
    tmp2 = np.trace((target_gate.T.conj() @ unitary), axis1=1, axis2=2)
    infidelity = (1 - np.abs(tmp2)**2 / N1**2)
    # such a noise is not realistic, but it is enough for demonstration
    ret = np.clip(infidelity + np_rng.normal(0, scale=0.01, size=infidelity.shape), 0, 1)
    if is_single_item:
        ret = ret.item()
    return ret


class DummyControlModel(torch.nn.Module):
    def __init__(self, ham_model, target_gate, num_segment, time_duration):
        super().__init__()
        N0 = ham_model.shape[0]
        assert (ham_model.ndim==2) and (ham_model.shape==(N0,N0))
        assert np.abs(ham_model-ham_model.T.conj()).max() < 1e-10
        assert (target_gate.shape==(N0,N0))
        assert np.abs(target_gate @ target_gate.T.conj() - np.eye(N0)).max() < 1e-10
        # TODO bound
        self.ham_model = torch.tensor(ham_model, dtype=torch.complex128)
        np_rng = np.random.default_rng()
        hf0 = lambda *x: torch.nn.Parameter(torch.tensor(np_rng.uniform(-0.1, 0.1, size=x), dtype=torch.float64))
        self.omega = hf0(num_segment) #MHz
        self.para_ham_unknown = hf0(N0, N0)
        self.delta_t = time_duration / num_segment
        self.target_gate = torch.tensor(target_gate, dtype=torch.complex128)

        self.ham_unknown = None
        self.infedility = None
        self.is_calibration = False
        self.omega_data = None
        self.infedility_data = None

    def set_calibration(self, omega_data, infedility_data):
        self.is_calibration = True
        self.para_ham_unknown.requires_grad_(True)
        self.omega.requires_grad_(False)
        assert (omega_data.ndim==2) and (omega_data.shape[1]==self.omega.shape[0])
        assert (infedility_data.ndim==1) and (omega_data.shape[0]==infedility_data.shape[0])
        self.omega_data = torch.tensor(omega_data, dtype=torch.float64)
        self.infedility_data = torch.tensor(infedility_data, dtype=torch.float64)

    def get_optimal_control(self):
        self.is_calibration = False
        self.para_ham_unknown.requires_grad_(False)
        self.omega.requires_grad_(True)
        theta_optim = numqi.optimize.minimize(self, theta0=('uniform',-0.1,0.1), num_repeat=3, tol=1e-7)
        return theta_optim

    def forward(self):
        N1 = self.omega.shape[0]
        N2 = self.target_gate.shape[0]
        unitary = torch.eye(N2, dtype=torch.complex128)
        tmp0 = numqi.param.real_matrix_to_hermitian(self.para_ham_unknown)
        ham_unknown = tmp0 - torch.trace(tmp0)*torch.eye(N2, dtype=torch.complex128)/N2
        self.ham_unknown = ham_unknown.detach()
        if self.is_calibration:
            assert self.omega_data is not None
            for x in range(N1):
                if USE_QCTRL:
                    unitary = torch.linalg.matrix_exp((-1j*self.delta_t)*self.omega_data[:,x].reshape(-1,1,1) * (self.ham_model + ham_unknown)) @ unitary
                else:
                    unitary = torch.linalg.matrix_exp((-1j*self.delta_t)*(self.omega_data[:,x].reshape(-1,1,1) * self.ham_model + ham_unknown)) @ unitary
            tmp0 = torch.diagonal(self.target_gate.T.conj() @ unitary, dim1=1, dim2=2).sum(dim=1)
            infidelity = 1 - (tmp0 * tmp0.conj()).real/(N2*N2)
            self.infedility = infidelity.detach()
            loss = torch.mean((self.infedility_data - infidelity)**2)
        else:
            for x in range(N1):
                if USE_QCTRL:
                    unitary = torch.linalg.matrix_exp((-1j*self.delta_t)*self.omega[x] * (self.ham_model + ham_unknown)) @ unitary
                else:
                    unitary = torch.linalg.matrix_exp((-1j*self.delta_t)*(self.omega[x] * self.ham_model + ham_unknown)) @ unitary
            tmp0 = torch.trace(self.target_gate.T.conj() @ unitary)
            loss = 1 - (tmp0*tmp0.conj()).real / (N2*N2)
        return loss


u = -0.5846363798417062
phi = 2.86514324198955
ham_unknown = (u*numqi.gate.Z + np.sqrt(1-u**2)*(np.cos(phi)*numqi.gate.X + np.sin(phi)*numqi.gate.Y)) / 8
ham_model = 0.5 * numqi.gate.X
target_gate = numqi.gate.X
## such a Hamiltonian is really boring, for it's commutative at different time
# omega(t) * (ham_model + ham_unknown)

np_rng = np.random.default_rng()

batch_size = 20
num_segment = 10
time_duration = 1 #microsecond

model = DummyControlModel(ham_model, target_gate, num_segment, time_duration)

omega_data_list = []
infedility_data_list = []
omega_optim_list = []
omega_cur = None
for ind_round in range(10):
    print(f'round {ind_round}')
    if omega_cur is None:
        omega_data =  np.pi/num_segment * np.linspace(-1, 1, batch_size)[:,None] * np.ones(num_segment) #MHz
    else:
        omega_data = omega_cur + np_rng.normal(0, scale=0.1, size=(batch_size,num_segment))
    infedility_data = generate_exp_data(omega_data, time_duration, target_gate, ham_model, ham_unknown, np_rng)
    omega_data_list.append(omega_data)
    infedility_data_list.append(infedility_data)

    model.set_calibration(np.concatenate(omega_data_list, axis=0), np.concatenate(infedility_data_list, axis=0))
    theta_optim = numqi.optimize.minimize(model, theta0=('uniform',-0.1,0.1), num_repeat=3, tol=1e-7)

    omega_optim_list.append(model.get_optimal_control())
    omega_cur = omega_optim_list[-1].x
# there is a factor 2 in original qctrl documentation which i merge into omega.
# since Hamiltonian is commutative for different time, the gradient for omega at different time is just the same value,
# then the optimized omega will be almost a constant for different time.


fig,ax = plt.subplots()
xdata = np.arange(len(omega_optim_list)) + 1
ydata = [x.mean() for x in infedility_data_list]
ax.plot(xdata, ydata, label='exp')
ydata = [x.fun for x in omega_optim_list]
ax.plot(xdata, ydata, label='opt')
ax.set_xlabel('round')
ax.set_ylabel('infedility')
ax.set_yscale('log')
ax.legend()
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)

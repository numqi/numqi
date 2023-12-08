import numpy as np
import torch
import opt_einsum
import matplotlib.pyplot as plt

import numqi

# http://arxiv.org/abs/quant-ph/0304102v1
# Capacities of Quantum Channels and How to Find Them

op_torch_logm = numqi._torch_op.PSDMatrixLogm(num_sqrtm=6, pade_order=8)

def get_von_neumann_entropy(rho):
    log_rho = op_torch_logm(rho)
    if rho.ndim==2:
        ret = -torch.vdot(log_rho.reshape(-1), rho.reshape(-1)).real
    else:
        shape = rho.shape[:-2]
        N0 = rho.shape[-1]
        ret = -(rho.reshape(-1,1,N0*N0).conj() @ log_rho.reshape(-1,N0*N0,1)).real.reshape(*shape)
    return ret

class ChannelCapacity1Inf(torch.nn.Module):
    def __init__(self, kop, num_state:int):
        super().__init__()
        assert kop.ndim==3
        tmp0 = np.einsum(kop, [0,1,2], kop.conj(), [0,1,3], [2,3], optimize=True)
        assert np.abs(tmp0-np.eye(tmp0.shape[0])).max() < 1e-10
        _,dim_in,dim_out = kop.shape
        self.dim_out = kop.shape[1]
        self.dim_in = kop.shape[2]
        self.kop = torch.tensor(kop, dtype=torch.complex128)
        self.kop_conj = self.kop.conj().resolve_conj()

        np_rng = np.random.default_rng()
        hf0 = lambda *x: torch.nn.Parameter(torch.tensor(np_rng.uniform(-1,1,size=x), dtype=torch.float64))
        self.theta_coeff = hf0(num_state)
        self.theta_state = hf0(num_state, self.dim_in, 2)
        self.rho = None

        self.contract0 = opt_einsum.contract_expression(kop.shape, [0,1,2], kop.shape, [0,3,4], [dim_in,dim_in], [2,4], [1,3])
        self.contract1 = opt_einsum.contract_expression(kop.shape, [0,1,2], kop.shape, [0,3,4], [num_state,dim_in], [5,2], [num_state,dim_in], [5,4], [5,1,3])


    def forward(self):
        tmp0 = torch.nn.functional.softplus(self.theta_coeff)
        coeff = tmp0 / tmp0.sum()
        tmp0 = self.theta_state / torch.linalg.norm(self.theta_state, axis=(1,2), keepdims=True)
        state = torch.complex(tmp0[:,:,0], tmp0[:,:,1])
        rho = (state.T * coeff) @ state.conj()
        self.rho = rho.detach()
        rho_out = self.contract0(self.kop, self.kop_conj, rho)

        tmp0 = self.contract1(self.kop, self.kop_conj, state, state.conj())
        ret = -(get_von_neumann_entropy(rho_out) - torch.dot(coeff, get_von_neumann_entropy(tmp0)))
        # the capacity is to maximize the output entropy, which is equivalent to minimize the negative output entropy
        return ret


noise_rate = 0.233
kop = np.stack(numqi.channel.hf_amplitude_damping_kraus_op(noise_rate), axis=0)

num_state_list = np.arange(2, 20, 2)
kwargs = dict(theta0='uniform', num_repeat=3, tol=1e-10, print_every_round=0)
loss_list = []
for num_state in num_state_list:
    model = ChannelCapacity1Inf(kop, num_state)
    loss_list.append(-numqi.optimize.minimize(model, **kwargs).fun)
loss_list = np.array(loss_list)

fig,ax = plt.subplots()
ax.plot(num_state_list, loss_list, 'o-')
ax.set_xlabel('num_state')
ax.set_ylabel(r'capacity $C_{1,\infty}$')
ax.set_title(r'amplitude-damping channel $\eta=0.233$')
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)


num_state = 6
noise_rate_list = np.linspace(0, 1, 20)

hf_channel_list = [
    numqi.channel.hf_amplitude_damping_kraus_op,
    numqi.channel.hf_depolarizing_kraus_op,
    numqi.channel.hf_dephasing_kraus_op,
]

loss_list = []
kwargs = dict(theta0='uniform', num_repeat=3, tol=1e-10, print_every_round=0)
for hf_channel in hf_channel_list:
    for noise_rate in noise_rate_list:
        kop = np.stack(hf_channel(noise_rate), axis=0)
        model = ChannelCapacity1Inf(kop, num_state)
        loss_list.append(-numqi.optimize.minimize(model, **kwargs).fun)
loss_list = np.array(loss_list).reshape(len(hf_channel_list), -1)


fig,ax = plt.subplots()
for ind0,name in enumerate(['amplitude-damping', 'depolarizing', 'phase-damping']):
    ax.plot(noise_rate_list, loss_list[ind0], 'o-', label=name)
ax.set_xlabel('noise rate')
ax.set_ylabel(r'capacity $C_{1,\infty}$')
ax.legend()
ax.grid()
fig.tight_layout()
fig.savefig('tbd01.png', dpi=200)

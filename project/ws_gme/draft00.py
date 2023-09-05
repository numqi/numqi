import itertools
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import opt_einsum

import pyqet

def get_bipartition_list(num_part):
    ret = []
    for r in range(1, num_part//2+1):
        for x in itertools.combinations(range(num_part), r):
            tmp0 = set(x)
            tmp1 = sorted(tmp0), sorted(set(range(num_part)) - tmp0)
            if (len(tmp1[0])>len(tmp1[1])) or (len(tmp1[0])==len(tmp1[1]) and (0 in tmp1[1])):
                tmp1 = tmp1[1],tmp1[0]
            ret.append(tuple(tuple(sorted(x)) for x in tmp1))
    hf0 = lambda x: (len(x[0]),) + tuple(x[0]) + tuple(x[1])
    ret = sorted(set(ret), key=hf0)
    return ret


def test_get_bipartition_list():
    tmp0 = {((0,),(1,))}
    assert set(get_bipartition_list(2))==tmp0
    tmp0 = {((0,),(1,2)), ((1,),(0,2)), ((2,),(0,1))}
    assert set(get_bipartition_list(3))==tmp0
    tmp0 = {
        ((0,), (1, 2, 3)),
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
        ((1,), (0, 2, 3)),
        ((2,), (0, 1, 3)),
        ((3,), (0, 1, 2)),
    }
    assert set(get_bipartition_list(4))==tmp0


class AutodiffCHAGMEModel(torch.nn.Module):
    def __init__(self, dim_list, num_state=None):
        super().__init__()
        dim_list = [int(x) for x in dim_list]
        assert (len(dim_list)>=2) and all(x>1 for x in dim_list)
        dim_total = np.prod(np.array(dim_list, dtype=np.int64))
        self.dim_list = dim_list
        self.dim_total = dim_total

        num_part = len(dim_list)
        partition_list = get_bipartition_list(num_part)
        if num_state is None:
            num_state = [2*dim_total] * len(partition_list)
        else:
            num_state = [int(x) for x in num_state]
            assert len(num_state)==len(partition_list)
            assert all(x>0 for x in num_state)
        self.partition_list = partition_list
        self.num_state = num_state

        np_rng = np.random.default_rng()
        hf0 = lambda *size: torch.nn.Parameter(torch.tensor(np_rng.uniform(-1,1,size=size), dtype=torch.float64, requires_grad=True))
        z0 = []
        contract_expr_list = []
        for num0,(ind0,ind1) in zip(num_state,partition_list):
            dim0 = np.prod(np.array([dim_list[x] for x in ind0], dtype=np.int64))
            dim1 = np.prod(np.array([dim_list[x] for x in ind1], dtype=np.int64))
            z0.append((hf0(num0), hf0(2, num0, dim0), hf0(2, num0, dim1)))
            shape0 = [num0] + [self.dim_list[x] for x in ind0]
            shape1 = [num0] + [self.dim_list[x] for x in ind1]
            contract_expr = opt_einsum.contract_expression(
                [num0], [2*num_part],
                shape0, [2*num_part] + list(ind0),
                shape1, [2*num_part] + list(ind1),
                shape0, [2*num_part] + [num_part+x for x in ind0],
                shape1, [2*num_part] + [num_part+x for x in ind1],
                list(range(2*num_part)),
            )
            contract_expr_list.append((shape0,shape1,contract_expr))
        self.theta_p = hf0(len(partition_list))
        self.theta_q = torch.nn.ParameterList([x[0] for x in z0])
        self.theta_psi0 = torch.nn.ParameterList([x[1] for x in z0])
        self.theta_psi1 = torch.nn.ParameterList([x[2] for x in z0])
        self.contract_expr_list = contract_expr_list

        self.dm_torch = None
        self.probability_p = None
        self.probability_q_list = None
        self.dm_target = None

    def set_dm_target(self, dm):
        assert (dm.ndim==2) and (dm.shape[0]==dm.shape[1]) and (dm.shape[0]==self.dim_total)
        assert np.abs(dm - dm.T.conj()).max() < 1e-10
        assert np.linalg.eigvalsh(dm)[0] > -1e-10
        self.dm_target = torch.tensor(dm, dtype=torch.complex128)

    def forward(self):
        probability_q_list = []
        dm_torch = 0
        prob_p = torch.nn.functional.softmax(self.theta_p, dim=0)
        for ind0 in range(len(self.partition_list)):
            shape0,shape1,contract_expr = self.contract_expr_list[ind0]
            prob_q = torch.nn.functional.softmax(self.theta_q[ind0], dim=0)
            psi0 = torch.complex(self.theta_psi0[ind0][0], self.theta_psi0[ind0][1]).reshape(*shape0)
            psi1 = torch.complex(self.theta_psi1[ind0][0], self.theta_psi1[ind0][1]).reshape(*shape1)
            tmp0 = contract_expr(prob_q, psi0, psi1, psi0.conj(), psi1.conj())
            dm_torch = dm_torch + tmp0.reshape(self.dim_total, self.dim_total) * prob_p[ind0]
            probability_q_list.append(prob_q.detach())
        self.dm_torch = dm_torch.detach()
        self.probability_p = prob_p.detach()
        self.probability_q_list = probability_q_list
        tmp0 = (self.dm_target - dm_torch).reshape(-1)
        loss = 2 * torch.dot(tmp0, tmp0.conj()).real
        return loss


def get_ghz_state(num_part=3, return_dm=False):
    assert num_part>=2
    ret = np.zeros(2**num_part, dtype=np.float64)
    ret[0] = 1/np.sqrt(2)
    ret[-1] = 1/np.sqrt(2)
    if return_dm:
        ret = ret.reshape(-1,1) * ret
    return ret


num_part = 5
dim_list = [2]*num_part
dm_ghz = get_ghz_state(num_part, return_dm=True)

alpha_list = np.linspace(0, 1, 101)
dm_target_list = [pyqet.entangle.hf_interpolate_dm(dm_ghz, alpha=x) for x in alpha_list]

ret_list = []
model = AutodiffCHAGMEModel(dim_list)
# kwargs = dict(theta0='uniform', num_repeat=3, tol=1e-10, print_every_round=0)
kwargs = dict(num_step=10000, theta0='uniform', tqdm_update_freq=200, early_stop_threshold=1e-11)
prob_list = []
for ind0,dm_target_i in enumerate(dm_target_list):
    model.set_dm_target(dm_target_i)
    # theta_optim = pyqet.optimize.minimize(model, **kwargs)
    # ret_list.append(theta_optim.fun)
    loss = pyqet.optimize.minimize_adam(model, **kwargs)
    print(f"loss={loss:.2e}, alpha={alpha_list[ind0]:.3f}")
    ret_list.append(loss)
    prob_list.append(model.probability_p.numpy().copy())
# with dm_target_list as pbar:
#     for ind0,dm_target_i in enumerate(pbar):
#         model.set_dm_target(dm_target_i)
#         # theta_optim = pyqet.optimize.minimize(model, **kwargs)
#         # ret_list.append(theta_optim.fun)
#         loss = pyqet.optimize.minimize_adam(model, **kwargs)
#         pbar.set_postfix(loss=f'{loss:.2e}', alpha=f'{alpha_list[ind0]:.3f}')
#         ret_list.append(loss)
#         prob_list.append(model.probability_p.numpy().copy())
ret_list = np.array(ret_list)
prob_list = np.stack(prob_list)

fig,ax = plt.subplots()
ax.plot(alpha_list, ret_list)
ax.set_xlabel('alpha')
ax.set_ylabel('loss')
ax.set_yscale('log')
tmp0 = r'$\mathcal{H}_2^{\otimes' + str(num_part) + '}$'
ax.set_title(f'{tmp0} GHZ state (GME boundary)')
ax.grid()
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)


fig,ax = plt.subplots(figsize=(8,2))
himage = ax.imshow(prob_list.T, vmin=0, vmax=1, interpolation='nearest', origin='lower', aspect=1.4)
ax.set_xlabel('alpha')
fig.colorbar(himage)
fig.tight_layout()
fig.savefig('tbd01.png', dpi=200)

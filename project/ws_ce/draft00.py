import numpy as np
import functools
import opt_einsum
import torch
import scipy.optimize
from tqdm import tqdm

import numqi

hf_trace = lambda x,y: np.dot(x.reshape(-1), y.T.reshape(-1))

def _concentratable_entanglement_part_slow(rho, alpha):
    assert (rho.ndim==2) and rho.shape[0]==rho.shape[1]
    num_qubit = numqi.utils.hf_num_state_to_num_qubit(rho.shape[0])
    alpha_set = set(alpha)
    if (len(alpha_set)==0) or (len(alpha_set)==num_qubit):
        return 1
    alpha = sorted(alpha_set)
    alpha_orth = [x for x in range(num_qubit) if x not in alpha_set]
    tmp0 = list(range(2*num_qubit))
    for x in alpha_orth:
        tmp0[num_qubit+x] = x
    tmp1 = list(alpha) + [(x+num_qubit) for x in alpha]
    rho_alpha = opt_einsum.contract(rho.reshape([2]*(2*num_qubit)), tmp0, tmp1).reshape(2**len(alpha), -1)
    ret = np.vdot(rho_alpha.reshape(-1), rho_alpha.reshape(-1)).real
    return ret

@functools.lru_cache
def _concentratable_entanglement_get_index(num_qubit, dim_tuple=None):
    alpha_beta_list = []
    if dim_tuple is None:
        dim_tuple = (2,)*num_qubit
    else:
        assert len(dim_tuple)==num_qubit
    for x in range(1, 2**num_qubit):
        alpha = bin(x)[2:].rjust(num_qubit, '0')
        weight = sum(y=='1' for y in alpha)
        if weight<=num_qubit//2:
            beta = ''.join(('1' if y=='0' else '0') for y in alpha)
            if (2*weight!=num_qubit) or alpha[0]=='1':
                alpha_beta_list.append((alpha, beta))
    alpha_beta_list = tuple(alpha_beta_list)
    index_list = []
    for alpha,_ in alpha_beta_list:
        shape,ind0,ind1,ind_output,ind_cur,x = [],[],[],[],num_qubit,0
        while x<num_qubit:
            if (x==0) or (alpha[x]!=alpha[x-1]):
                shape.append(dim_tuple[x])
                ind0.append(x)
                if alpha[x]=='0':
                    ind1.append(x)
                else:
                    ind1.append(ind_cur)
                    ind_output.append(x)
                    ind_cur = ind_cur + 1
            else: #alpha[x]==alpha[x-1]
                shape[-1] *= dim_tuple[x]
            x = x + 1
        ind_output += list(range(num_qubit, ind_cur))
        index_list.append((tuple(shape),tuple(ind0),tuple(ind1),tuple(ind_output)))
    index_list = tuple(index_list)
    return alpha_beta_list,index_list


@functools.lru_cache
def _concentratable_entanglement_get_bitstr(num_qubit, alpha_tuple):
    ret = []
    for ind0 in range(2**len(alpha_tuple)):
        tmp0 = ['0' for _ in range(num_qubit)]
        ind0 = bin(ind0)[2:].rjust(num_qubit, '0')
        for x,y in zip(ind0, alpha_tuple):
            if x=='1':
                tmp0[y] = '1'
        ret.append(''.join(tmp0))
    return tuple(ret)

def get_concentratable_entanglement(psi, alpha_tuple=None):
    is_torch = numqi.utils.is_torch(psi)
    assert psi.ndim==1
    num_qubit = numqi.utils.hf_num_state_to_num_qubit(psi.shape[0])
    if alpha_tuple is None:
        alpha_tuple = tuple(range(num_qubit))
    alpha_beta_list,index_list = _concentratable_entanglement_get_index(num_qubit)
    bitstr_list = _concentratable_entanglement_get_bitstr(num_qubit, alpha_tuple)
    bitstr_to_id = dict()
    for x,(y,z) in enumerate(alpha_beta_list):
        bitstr_to_id[y] = x
        bitstr_to_id[z] = x
    bitstr_to_id['0'*num_qubit] = len(alpha_beta_list)
    bitstr_to_id['1'*num_qubit] = len(alpha_beta_list)

    if is_torch:
        dtype_map = {torch.complex64:torch.float32, torch.complex128:torch.float64}
    else:
        dtype_map = {np.complex64:np.float32, np.complex128:np.float64}
    id_to_value = [None for _ in range(len(alpha_beta_list))] + [1.0]
    psi_conj = psi.conj()
    for bitstr in bitstr_list:
        if id_to_value[bitstr_to_id[bitstr]] is None:
            shape,ind0,ind1,ind_output = index_list[bitstr_to_id[bitstr]]
            tmp0 = psi.reshape(shape)
            tmp1 = psi_conj.reshape(shape)
            # tmp2 = opt_einsum.contract(tmp0, ind0, tmp1, ind1, ind_output).reshape(-1)
            if is_torch:
                tmp2 = torch.einsum(tmp0, ind0, tmp1, ind1, ind_output).reshape(-1)
                id_to_value[bitstr_to_id[bitstr]] = torch.linalg.norm(tmp2)**2
            else:
                tmp2 = np.einsum(tmp0, ind0, tmp1, ind1, ind_output, optimize=True).reshape(-1)
                tmp2 = tmp2.view(dtype_map[tmp2.dtype.type])
                id_to_value[bitstr_to_id[bitstr]] = np.dot(tmp2, tmp2)
    ret = {x:id_to_value[bitstr_to_id[x]] for x in bitstr_list}
    ce = 1-sum([ret[x] for x in bitstr_list])/2**len(alpha_tuple)
    return ce,ret

class ConcentratableEntanglementModel(torch.nn.Module):
    def __init__(self, num_qubit) -> None:
        super().__init__()
        np_rng = np.random.default_rng()
        self.num_qubit = num_qubit
        # hf0 = lambda *x: torch.nn.Parameter(torch.tensor(np_rng.uniform(-1, 1, size=x), dtype=torch.float64))
        # self.radius = hf0(2**num_qubit)
        # self.angle = hf0(2**num_qubit)
        self.theta = torch.nn.Parameter(torch.tensor(np_rng.uniform(-1, 1, size=2*2**num_qubit), dtype=torch.float64))

    def forward(self):
        psi = self.get_pure_state()
        ce,_ = get_concentratable_entanglement(psi)
        loss = -ce
        return loss

    def get_pure_state(self):
        # tmp0 = torch.exp(1j*self.angle)
        # psi = self.radius * tmp0 / torch.linalg.norm(self.radius)
        N0 = self.theta.shape[0]//2
        tmp0 = self.theta / torch.linalg.norm(self.theta)
        psi = tmp0[:N0] + 1j*tmp0[N0:]
        return psi

    # TODO use numqi.optimize.minimize
    def minimize(self, num_repeat=3, tol=1e-7, print_freq=-1, seed=None):
        np_rng = numqi.random.get_numpy_rng(seed)
        num_parameter = len(numqi.optimize.get_model_flat_parameter(self))
        hf_model = numqi.optimize.hf_model_wrapper(self)
        ret = []
        min_fun = 1
        for ind0 in range(num_repeat):
            theta0 = np_rng.uniform(-1, 1, size=num_parameter)
            hf_callback = numqi.optimize.hf_callback_wrapper(hf_model, print_freq=print_freq)
            theta_optim = scipy.optimize.minimize(hf_model, theta0, jac=True, method='L-BFGS-B', tol=tol, callback=hf_callback)
            ret.append(theta_optim)
            min_fun = min(min_fun, theta_optim.fun)
            print(ind0, theta_optim.fun, min_fun)
        ret = min(ret, key=lambda x: x.fun)
        numqi.optimize.set_model_flat_parameter(self, ret.x)
        return ret

    # TODO use numqi.optimize.minimize_adam
    def minimize_adam(self, num_step, lr=0.001):
        # optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.8)
        with tqdm(range(num_step)) as pbar:
            for ind0 in pbar:
                optimizer.zero_grad()
                loss = self()
                loss.backward()
                optimizer.step()
                if ind0%10==0:
                    pbar.set_postfix(ce=f'{-loss.item():.12f}')

# dim0 = 4
# dim1 = 4
# pure_state = numqi.random.rand_state(dim0*dim1)
# dm0 = pure_state[:,np.newaxis]*pure_state.conj()

# z0 = _concentratable_entanglement_part_slow(dm0, [0,2])
# z1 = _concentratable_entanglement_part_slow(dm0, [1,3])
# assert abs(z0-z1)<1e-10

# z2 = get_concentratable_entanglement(pure_state)
# z3 = get_concentratable_entanglement(torch.tensor(pure_state))

num_qubit = 7

model = ConcentratableEntanglementModel(num_qubit)
# theta_optim = model.minimize(num_repeat=30000, tol=1e-10)
# print(theta_optim.fun)


from zzz import to_pickle, from_pickle
# to_pickle(theta0=theta0, theta1=theta1)
np_rng = numqi.random.get_numpy_rng(None)
num_parameter = len(numqi.optimize.get_model_flat_parameter(model))
hf_model = numqi.optimize.hf_model_wrapper(model)
theta0 = from_pickle('theta0')
theta1 = from_pickle('theta1')
# theta0 = np_rng.uniform(-1, 1, size=num_parameter)
# theta1 = np_rng.uniform(-1, 1, size=num_parameter)
alpha_list = np.linspace(0, 1, 100)
hf_interp = lambda x: theta0*(1-x) + theta1*x
ret = []
for ind0,alpha_i in enumerate(alpha_list):
    theta_i = hf_interp(alpha_i)
    hf_callback = numqi.optimize.hf_callback_wrapper(hf_model, print_freq=-1)
    theta_optim = scipy.optimize.minimize(hf_model, theta_i, jac=True, method='L-BFGS-B', tol=1e-10, callback=hf_callback)
    ret.append(theta_optim)
    print(ind0, theta_optim.fun)


import matplotlib.pyplot as plt
plt.ion()
fig,ax = plt.subplots()
ax.plot(alpha_list, [x.fun for x in ret])
fig.tight_layout()
fig.savefig('tbd00.png', dpi=100)

# np_rng = np.random.default_rng()
# num_parameter = len(numqi.optimize.get_model_flat_parameter(model))
# theta0 = np_rng.uniform(-1, 1, size=num_parameter)
# numqi.optimize.set_model_flat_parameter(model, theta0)
# model.minimize_adam(10000, lr=0.1)

# from zzz import to_pickle
# to_pickle(ce7=theta_optim)
# 2 0.25
# 3 0.37499999999914957
# 4 0.49999999574509135
# 5 0.6249999955678904
# 6 0.7187499938911843
# 7 0.7745901636077773
# 8 0.8281247453070576
# 9 0.8662233227306793

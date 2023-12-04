import numpy as np
import functools
import opt_einsum
import torch

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

    if isinstance(psi,torch.Tensor):
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
            if isinstance(psi,torch.Tensor):
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
    def __init__(self, num_qubit):
        super().__init__()
        np_rng = np.random.default_rng()
        self.theta = torch.nn.Parameter(torch.tensor(np_rng.uniform(-1, 1, size=(2,2**num_qubit)), dtype=torch.float64))

    def forward(self):
        tmp0 = self.theta / torch.linalg.norm(self.theta)
        psi = torch.complex(tmp0[0], tmp0[1])
        loss = -get_concentratable_entanglement(psi)[0]
        return loss

# dim0 = 4
# dim1 = 4
# pure_state = numqi.random.rand_state(dim0*dim1)
# dm0 = pure_state[:,np.newaxis]*pure_state.conj()

# z0 = _concentratable_entanglement_part_slow(dm0, [0,2])
# z1 = _concentratable_entanglement_part_slow(dm0, [1,3])
# assert abs(z0-z1)<1e-10

# z2 = get_concentratable_entanglement(pure_state)
# z3 = get_concentratable_entanglement(torch.tensor(pure_state))
# TODO unittest

num_qubit = 2
model = ConcentratableEntanglementModel(num_qubit)
theta_optim = numqi.optimize.minimize(model, 'uniform', num_repeat=10, tol=1e-10)
# 2 0.25
# 3 0.37499999999914957
# 4 0.49999999574509135
# 5 0.6249999955678904
# 6 0.7187499938911843
# 7 0.7745901636077773
# 8 0.8281247453070576
# 9 0.8662233227306793

import time
import pickle
import itertools
import numpy as np
import torch
import opt_einsum
import concurrent.futures

if not torch.get_num_threads()!=1:
    torch.set_num_threads(1)

import numqi

def get_keep_list(num_party):
    keep_list = [tuple(x) for x in itertools.combinations(range(num_party), num_party//2)]
    if num_party % 2 == 0:
        tmp0 = set(range(num_party))
        tmp1 = []
        for x in keep_list:
            y = tuple(sorted(tmp0 - set(x)))
            tmp1.append(((x,y) if (min(x)<min(y)) else (y,x)))
        keep_list = [x[0] for x in sorted(set(tmp1))]
    return keep_list

class AbsolutelyMaximallyEntangledTensorStateModel(torch.nn.Module):
    def __init__(self, num_party:int, dim:int, dim_bond:int, dtype='float64'):
        super().__init__()
        assert num_party>=2
        assert dim>=2
        assert dtype in {'float32','float64'}
        self.dtype = torch.float32 if (dtype=='float32') else torch.float64
        self.cdtype = torch.complex64 if (dtype=='float32') else torch.complex128
        self.num_party = int(num_party)
        self.dim = int(dim)
        self.dim_bond = int(dim_bond)
        np_rng = np.random.default_rng()
        hf0 = lambda *x: torch.nn.Parameter(torch.tensor(np_rng.uniform(-1,1,size=x), dtype=self.dtype))
        self.theta_list = torch.nn.ParameterList([hf0(2, dim_bond, dim) for _ in range(num_party)])
        self.target = torch.eye(dim**(num_party//2), dtype=self.cdtype)/(dim**(num_party//2))

        n = num_party
        r = num_party//2
        tmp0 = [y for x in range(num_party) for y in ([dim_bond,dim],(num_party,x))]
        tmp1 = [y for x in range(num_party) for y in ([dim_bond,dim],(num_party+1,x))]
        self.contract_factor2 = opt_einsum.contract_expression(*tmp0, *tmp1, [])

        self.keep_list = get_keep_list(num_party)
        self.contract_list = []
        for keep in self.keep_list:
            tmp0 = [y for x in range(num_party) for y in ([dim_bond,dim],(num_party,x))]
            tmp1 = np.arange(num_party)
            tmp1[list(keep)] += num_party + 2
            tmp2 = [y for x in tmp1 for y in ([dim_bond,dim],(num_party+1,x))]
            tmp3 = list(keep) + [x+num_party+2 for x in keep]
            self.contract_list.append(opt_einsum.contract_expression(*tmp0, *tmp2, tmp3))

    def forward(self):
        theta_list = [torch.complex(x[0],x[1]) for x in self.theta_list]
        theta_conj_list = [x.conj().resolve_conj() for x in theta_list]
        factor2 = self.contract_factor2(*theta_list, *theta_conj_list).real
        loss = 0
        for contract_i in self.contract_list:
            tmp0 = contract_i(*theta_list, *theta_conj_list).reshape(self.target.shape) / factor2
            tmp1 = (tmp0 - self.target).reshape(-1)
            loss = loss + torch.vdot(tmp1, tmp1).real
        return loss


np_rng = np.random.default_rng()
hf_randc = lambda *x: np_rng.uniform(-1,1,size=x) + 1j*np_rng.uniform(-1,1,size=x)

dim = 6
num_party = 4
dim_bond = 6**3
num_repeat = 4

model = AbsolutelyMaximallyEntangledTensorStateModel(num_party, dim, dim_bond)
kwargs = dict(theta0='uniform', tol=1e-7, early_stop_threshold=None, print_every_round=1)
theta_optim = numqi.optimize.minimize(model, num_repeat=100, **kwargs)

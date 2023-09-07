import numpy as np
import opt_einsum
import torch

import numqi

def get_all_non_isomorphic_graph(num_node:int):
    # https://www.graphclasses.org/smallgraphs.html
    assert num_node in {2,3,4,5}
    if num_node==2:
        graph = [[], [(0,1)]]
    elif num_node==3:
        graph = [
            [], [(0,1)], [(0,2), (1,2)],
            [(0,1)], [(0,1),(1,2)],
        ]
    elif num_node==4:
        graph = [
            [], [(0,1),(1,2),(0,2),(0,3),(1,3),(2,3)],
            [(0,1)], [(0,1),(0,2),(1,2),(3,1),(3,2)],
            [(0,1),(1,2)], [(0,1),(0,2),(1,2),(0,3)],
            [(0,1),(2,3)], [(0,1),(1,2),(2,3),(0,3)],
            [(0,1),(0,2),(0,3)], [(0,1),(1,2),(0,2)],
            [(0,1),(1,2),(2,3)],
        ]
    elif num_node==5:
        graph = [
            [], [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)],
            [(0,1)], [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4)],
            [(0,1),(1,2)], [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3),(0,4),(1,4)],
            [(0,1),(2,3)], [(0,1),(1,2),(2,3),(0,3),(0,4),(1,4),(0,4),(1,4)],
            [(0,1),(0,2),(0,3)], [(0,1),(1,2),(2,3),(0,3),(0,4),(1,4),(0,4)],
            [(0,1),(1,2),(3,4)], [(0,1),(1,2),(2,3),(0,3),(0,4),(1,4),(2,4)],
            [(0,1),(1,2),(2,3)], [(0,1),(1,2),(2,3),(0,4),(1,4),(2,4),(3,4)],
            [(0,1),(1,2),(0,2)], [(0,1),(1,2),(0,2),(3,0),(3,1),(4,0),(4,1)],
            [(0,1),(0,2),(0,3),(0,4)], [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)],
            [(0,1),(1,2),(2,3),(3,0)], [(0,1),(0,2),(0,3),(0,4),(1,2),(3,4)],
            [(0,1),(0,2),(0,3),(3,4)], [(0,1),(1,2),(2,3),(3,0),(1,3),(0,4)],
            [(0,1),(1,2),(2,0),(0,3)], [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3)],
            [(0,1),(1,2),(2,3),(3,4)], [(0,1),(1,2),(2,3),(3,0),(0,4),(1,4)],
            [(0,1),(1,2),(2,0),(3,4)], [(0,1),(1,2),(2,3),(3,0),(0,4),(1,4)],
            [(0,1),(1,2),(2,3),(3,0),(0,4)], [(0,1),(0,2),(0,3),(1,2),(3,4)],
            [(0,1),(1,2),(2,0),(0,3),(1,4)],
            [(0,1),(0,2),(0,3),(0,4),(1,2)], [(0,1),(1,2),(2,3),(3,0),(0,2)],
            [(0,1),(1,2),(2,3),(3,4),(4,0)],
        ]
    ret = []
    for edge_list in graph:
        tmp0 = np.zeros((num_node,num_node), dtype=np.uint8)
        if len(edge_list):
            tmp1 = np.array(edge_list)
            tmp0[tmp1[:,0], tmp1[:,1]] = 1
            tmp0[tmp1[:,1], tmp1[:,0]] = 1
        ret.append(tmp0)
    ret = np.stack(ret, axis=0)
    return graph,ret

# a canonical polyadic (CP) tensor decomposition [30, 31]. The CP rank r = rank(|ψ〉) of a ten
class CanonicalPolyadicRankModel(torch.nn.Module):
    def __init__(self, dim_list, rank:int):
        super().__init__()
        dim_list = tuple(int(x) for x in dim_list)
        assert len(dim_list)>=3
        assert all(x>1 for x in dim_list)
        assert rank>=1
        self.dim_list = dim_list
        self.rank = rank
        np_rng = np.random.default_rng()
        hf0 = lambda *x: torch.nn.Parameter(torch.tensor(np_rng.uniform(-1,1,size=x), dtype=torch.float64))
        self.theta_psi = torch.nn.ParameterList([hf0(rank,x,2) for x in dim_list])
        self.theta_coeff = hf0(rank,2)

        N0 = len(dim_list)
        tmp0 = [(rank,),(rank,)] + [(rank,x) for x in dim_list] + [(rank,x) for x in dim_list]
        tmp1 = [(N0,),(N0+1,)] + [(N0,x) for x in range(N0)] + [(N0+1,x) for x in range(N0)]
        self.contract_psi_psi = opt_einsum.contract_expression(*[y for x in zip(tmp0,tmp1) for y in x], [])
        tmp0 = [dim_list,(rank,)] + [(rank,x) for x in dim_list]
        tmp1 = [tuple(range(N0)),(N0,)] + [(N0,x) for x in range(N0)]
        self.contract_target_psi = opt_einsum.contract_expression(*[y for x in zip(tmp0,tmp1) for y in x], [])

        self.target = None
        self.traget_conj = None

    def set_target(self, np0):
        assert np0.shape==self.dim_list
        self.target = torch.tensor(np0 / np.linalg.norm(np0.reshape(-1)), dtype=torch.complex128)
        self.target_conj = self.target.conj().resolve_conj()

    def forward(self):
        tmp0 = (x/torch.linalg.norm(x,axis=(1,2),keepdims=True) for x in self.theta_psi)
        theta_psi = [torch.complex(x[:,:,0],x[:,:,1]) for x in tmp0]
        theta_coeff = torch.complex(self.theta_coeff[:,0], self.theta_coeff[:,1])
        theta_psi_conj = [x.conj().resolve_conj() for x in theta_psi]
        psi_psi = self.contract_psi_psi(theta_coeff, theta_coeff.conj(), *theta_psi, *theta_psi_conj).real
        target_psi = self.contract_target_psi(self.target_conj, theta_coeff, *theta_psi)
        loss = 1 - (target_psi.real**2 + target_psi.imag**2) / psi_psi
        return loss

case_list = {
    2: [1,2],
    3: [1,2],
    4: [1,2,3,4],
    5: [1,2,3,4,5,6],
}

num_qubit = 3
rank_list = case_list[num_qubit]
dim_list = [2]*num_qubit

edge_list,adjacent_list = get_all_non_isomorphic_graph(num_qubit)

model_list = [CanonicalPolyadicRankModel(dim_list,x) for x in rank_list]
kwargs = dict(theta0='uniform', tol=1e-10, num_repeat=3, print_every_round=0)
loss_list = []
for edge_i,adjacent_mat in zip(edge_list, adjacent_list):
    q0 = numqi.sim.build_graph_state(adjacent_mat).reshape(dim_list)
    tmp0 = []
    for model in model_list:
        model.set_target(q0)
        theta_optim = numqi.optimize.minimize(model, **kwargs)
        tmp0.append(theta_optim.fun)
    tmp1 = ' '.join([f'{x:.3g}' for x in tmp0])
    print(f'[{edge_i}] {tmp1}')
    loss_list.append(tmp0)
loss_list = np.array(loss_list)
## #qubits=3
# [[0.  0. ]
#  [0.5 0. ]
#  [0.5 0. ]
#  [0.5 0. ]
#  [0.5 0. ]]

## #qubits=4
# [[0.   0.   0.   0.  ]
#  [0.5  0.   0.   0.  ]
#  [0.5  0.   0.   0.  ]
#  [0.75 0.5  0.25 0.  ]
#  [0.5  0.   0.   0.  ]
#  [0.75 0.5  0.25 0.  ]
#  [0.75 0.5  0.25 0.  ]
#  [0.75 0.5  0.25 0.  ]
#  [0.5  0.   0.   0.  ]
#  [0.5  0.   0.   0.  ]
#  [0.75 0.5  0.25 0.  ]]

## #qubits=5
# [[0.    0.    0.    0.    0.    0.   ]
#  [0.5   0.    0.    0.    0.    0.   ]
#  [0.5   0.    0.    0.    0.    0.   ]
#  [0.75  0.5   0.25  0.    0.    0.   ]
#  [0.5   0.    0.    0.    0.    0.   ]
#  [0.75  0.5   0.25  0.    0.    0.   ]
#  [0.75  0.5   0.25  0.    0.    0.   ]
#  [0.869 0.688 0.5   0.25  0.125 0.   ]
#  [0.5   0.    0.    0.    0.    0.   ]
#  [0.869 0.688 0.5   0.25  0.125 0.   ]
#  [0.75  0.5   0.25  0.    0.    0.   ]
#  [0.75  0.5   0.25  0.    0.    0.   ]
#  [0.75  0.5   0.25  0.    0.    0.   ]
#  [0.869 0.688 0.5   0.25  0.125 0.   ]
#  [0.5   0.    0.    0.    0.    0.   ]
#  [0.75  0.5   0.25  0.    0.    0.   ]
#  [0.5   0.    0.    0.    0.    0.   ]
#  [0.5   0.    0.    0.    0.    0.   ]
#  [0.75  0.5   0.25  0.    0.    0.   ]
#  [0.75  0.5   0.25  0.    0.    0.   ]
#  [0.75  0.5   0.25  0.    0.    0.   ]
#  [0.75  0.5   0.25  0.    0.    0.   ]
#  [0.75  0.5   0.25  0.    0.    0.   ]
#  [0.75  0.5   0.25  0.    0.    0.   ]
#  [0.75  0.5   0.25  0.    0.    0.   ]
#  [0.869 0.688 0.5   0.25  0.125 0.   ]
#  [0.75  0.5   0.25  0.    0.    0.   ]
#  [0.869 0.688 0.5   0.25  0.125 0.   ]
#  [0.75  0.5   0.25  0.    0.    0.   ]
#  [0.75  0.5   0.25  0.    0.    0.   ]
#  [0.75  0.5   0.25  0.    0.    0.   ]
#  [0.75  0.5   0.25  0.    0.    0.   ]
#  [0.75  0.5   0.25  0.    0.    0.   ]
#  [0.869 0.688 0.5   0.25  0.125 0.   ]]

import numpy as np
import opt_einsum
import torch

import numqi

hf_np_softplus = lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
# https://stackoverflow.com/a/51828104/7290857
# torch.nn.functional.softplus

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

# a canonical polyadic (CP) tensor decomposition
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
        self.theta_coeff = hf0(rank)

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
        theta_coeff = torch.nn.functional.softplus(self.theta_coeff).to(theta_psi[0].dtype)
        # theta_coeff = torch.exp(self.theta_coeff).to(theta_psi[0].dtype) #not help
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
'''
# qubits=3
[[0.  0. ]
 [0.5 0. ]
 [0.5 0. ]
 [0.5 0. ]
 [0.5 0. ]]

# qubits=4
[[0.   0.   0.   0.  ]
 [0.5  0.   0.   0.  ]
 [0.5  0.   0.   0.  ]
 [0.75 0.5  0.25 0.  ]
 [0.5  0.   0.   0.  ]
 [0.75 0.5  0.25 0.  ]
 [0.75 0.5  0.25 0.  ]
 [0.75 0.5  0.25 0.  ]
 [0.5  0.   0.   0.  ]
 [0.5  0.   0.   0.  ]
 [0.75 0.5  0.25 0.  ]]

# qubits=5
[[0.    0.    0.    0.    0.    0.   ]
 [0.5   0.    0.    0.    0.    0.   ]
 [0.5   0.    0.    0.    0.    0.   ]
 [0.75  0.5   0.25  0.    0.    0.   ]
 [0.5   0.    0.    0.    0.    0.   ]
 [0.75  0.5   0.25  0.    0.    0.   ]
 [0.75  0.5   0.25  0.    0.    0.   ]
 [0.869 0.688 0.5   0.25  0.125 0.   ]
 [0.5   0.    0.    0.    0.    0.   ]
 [0.869 0.688 0.5   0.25  0.125 0.   ]
 [0.75  0.5   0.25  0.    0.    0.   ]
 [0.75  0.5   0.25  0.    0.    0.   ]
 [0.75  0.5   0.25  0.    0.    0.   ]
 [0.869 0.688 0.5   0.25  0.125 0.   ]
 [0.5   0.    0.    0.    0.    0.   ]
 [0.75  0.5   0.25  0.    0.    0.   ]
 [0.5   0.    0.    0.    0.    0.   ]
 [0.5   0.    0.    0.    0.    0.   ]
 [0.75  0.5   0.25  0.    0.    0.   ]
 [0.75  0.5   0.25  0.    0.    0.   ]
 [0.75  0.5   0.25  0.    0.    0.   ]
 [0.75  0.5   0.25  0.    0.    0.   ]
 [0.75  0.5   0.25  0.    0.    0.   ]
 [0.75  0.5   0.25  0.    0.    0.   ]
 [0.75  0.5   0.25  0.    0.    0.   ]
 [0.869 0.688 0.5   0.25  0.125 0.   ]
 [0.75  0.5   0.25  0.    0.    0.   ]
 [0.869 0.688 0.5   0.25  0.125 0.   ]
 [0.75  0.5   0.25  0.    0.    0.   ]
 [0.75  0.5   0.25  0.    0.    0.   ]
 [0.75  0.5   0.25  0.    0.    0.   ]
 [0.75  0.5   0.25  0.    0.    0.   ]
 [0.75  0.5   0.25  0.    0.    0.   ]
 [0.869 0.688 0.5   0.25  0.125 0.   ]]
'''



def get_line_graph(N0, ring=False):
    assert N0>=2
    edge_list = [(x,x+1) for x in range(N0-1)]
    if ring:
        edge_list.append((N0-1,0))
    ret = np.zeros((N0,N0), dtype=np.uint8)
    tmp1 = np.array([x[0] for x in edge_list], dtype=np.int64)
    tmp2 = np.array([x[1] for x in edge_list], dtype=np.int64)
    ret[tmp1,tmp2] = 1
    ret[tmp2,tmp1] = 1
    return edge_list,ret


num_qubit_list = list(range(3,7))
loss_list = []
kwargs = dict(theta0='uniform', tol=1e-10, num_repeat=10, print_every_round=1)
# kwargs = dict(theta0='uniform', tol=1e-12, num_repeat=5, print_every_round=1, print_freq=100)
for num_qubit in num_qubit_list:
    dim_list = [2]*num_qubit
    edge_list,adjacent_mat = get_line_graph(num_qubit, ring=True)
    q0 = numqi.sim.build_graph_state(adjacent_mat).reshape(dim_list)

    tmp0 = []
    for rank in range(1, 2**num_qubit):
        model = CanonicalPolyadicRankModel(dim_list, rank)
        model.set_target(q0)
        theta_optim = numqi.optimize.minimize(model, **kwargs)
        # numqi.optimize.minimize_adam(model, num_step=10000, theta0=('uniform',-1,1), optim_args=('adam',0.03, 1e-5))
        tmp0.append(theta_optim.fun)
        print(rank, theta_optim.fun)
        # print(np.sort(torch.nn.functional.softplus(model.theta_coeff.detach()).numpy()))
        if theta_optim.fun < 1e-7:
            break
    loss_list.append(tmp0)
    print(f'[num_qubit={num_qubit}][{1}-{len(loss_list[-1])}]', loss_list[-1])


# ring
'''
[num_qubit=3][1-2] [0.5000000000000195, 4.3789416537265424e-12]
[num_qubit=4][1-4] [0.7500000000000033, 0.5000000000411531, 0.2500000000286372, 1.0265899241801435e-11]
[num_qubit=5][1-6] [0.8685541442350628, 0.687500075547633, 0.5000003097814772, 0.2500039980299097, 0.1250000678220633, 2.9184266114867796e-10]
[num_qubit=6][1-8] [0.8750000000000262, 0.7500000000006245, 0.6250000000037773, 0.5000000000190146, 0.37500000000153033, 0.25000000002330014, 0.1250000000182523, 1.316124986772138e-10]
[num_qubit=7]
    [10](0.12500095493000252) 0.00481488 0.00481504 1.45475031 1.45485428 1.45522579 1.45540526 1.45565424 1.45577496 1.4561975  1.4562562
    [11](0.06250138130834926) 0.00694776 0.00973924 0.00973955 1.34516742 1.34567765 1.78595738 1.78598496 1.84270201 1.86365576 2.28920792 2.28960064
    [12](7.482570119066168e-12) 0.59473852 0.59474111 0.81681943 0.81682345 0.81682418 0.81683191 0.81684954 0.81685196 0.81685285 0.81685984 1.71558841 1.71567163
[num_qubit=8][13-16] 0.1875000001826701 0.12500000002549527 0.06250000004499379 7.992795314493151e-11

17 0.29061279384504135
18 0.1875102763652855
19 0.15626209969378912
20 0.1764150393567715
21 0.14540931842678684
22 0.12205324317623023
23 0.031252138009318364
24 3.759466379982257e-06
25 0.03125617156736249
26 4.421121223230351e-06
27 0.03157221838513313
28 1.9713170983370887e-06
29 1.0107359393884963e-11
[num_qubit=9]
'''

# Wstate
w_state = np.zeros((2,2,2), dtype=np.float64)
w_state[[1,0,0], [0,1,0], [0,0,1]] = np.sqrt(1/3)

# state = np.einsum(w_state, [0,1,2], w_state, [3,4,5], [0,3,1,4,2,5], optimize=True).reshape(4,4,4)
# dim_list = [4,4,4]
# rank_list = [5,6,7]
state = np.einsum(w_state, [0,1,2], w_state, [3,4,5], list(range(6)), optimize=True)
dim_list = [2]*6
rank_list = [6,7,8]
kwargs = dict(theta0='uniform', tol=1e-14, num_repeat=10, print_every_round=1, early_stop_threshold=1e-12)
# kwargs = dict(theta0='uniform', tol=1e-12, num_repeat=5, print_every_round=1, print_freq=100)
for rank in rank_list:
    model = CanonicalPolyadicRankModel(dim_list, rank)
    model.set_target(state)
    theta_optim = numqi.optimize.minimize(model, **kwargs)
    print(rank, theta_optim.fun)
    print(np.sort(torch.nn.functional.softplus(model.theta_coeff.detach()).numpy()))
# https://arxiv.org/abs/1708.08578 The tensor rank of tensor product of two three-qubit W states is eight
'''
# 3 partites
5 9.31416822558262e-08 #0.05118167 0.06394317 2.03486349 2.12729523 2.14487255
6 4.951769805305872e-08 #0.13986747 0.23906225 0.49913582 0.52315527 0.99002567 1.43312482
7 1.5276668818842154e-13  #0.59346762 0.62571718 0.68700606 0.99066373 1.02368863 1.40575347 1.94539052

# 6 partites
6 2.0267438061161158e-08 #0.61095522 0.73198293 0.78672375 0.9301251  1.18042963 1.18188821
7 1.3047962377221722e-08 #0.16529814 0.33182699 0.52472523 1.00901532 1.02033939 1.3496718 2.34745586
8 6.437073096776658e-13 #0.12079513 0.30880243 0.50404039 0.50504966 0.51098676 0.86430407 1.22066444 1.42203985
'''

# np0 = np.zeros((5,5,5), dtype=np.float64)
# np0[0,0,0] = 1
# np0[0,1,2] = 1
# np0[1,2,0] = 1
# np0[2,0,1] = 1
# np0[1,3,2] = 1
# np0[3,2,1] = 1
# np0[4,4,4] = 1
# np0 = np0/np.linalg.norm(np0.reshape(-1))
# model = CanonicalPolyadicRankModel([5,5,5], 6)
# model.set_target(np0)
# kwargs = dict(theta0='uniform', tol=1e-14, num_repeat=10, print_every_round=1, early_stop_threshold=1e-12)
# theta_optim = numqi.optimize.minimize(model, **kwargs)


np0 = np.zeros((2,2,2))
np0[0,0,0] = 1/np.sqrt(2)
np0[1,1,1] = 1/np.sqrt(2)
zc0 = np0.copy()
num_copy = 5
for _ in range(num_copy-1):
    tmp0 = np.einsum(zc0, [0,1,2], np0, [3,4,5], [0,3,1,4,2,5], optimize=True)
    tmp1 = zc0.shape[0] * np0.shape[0]
    zc0 = tmp0.reshape(tmp1,tmp1,tmp1)
dim_list = zc0.shape
model = CanonicalPolyadicRankModel(dim_list, 2**num_copy-1)
model.set_target(zc0)
kwargs = dict(theta0='uniform', tol=1e-14, num_repeat=100, print_every_round=1, early_stop_threshold=1e-12)
theta_optim = numqi.optimize.minimize(model, **kwargs)


np0 = np.zeros((2,2,2), dtype=np.float64)
np0[0,0,1] = 1/np.sqrt(3)
np0[0,1,0] = 1/np.sqrt(3)
np0[1,0,0] = 1/np.sqrt(3)
model = CanonicalPolyadicRankModel([2,2,2], 2)
model.set_target(np0)
kwargs = dict(theta0='uniform', tol=1e-14, num_repeat=100, print_every_round=1, early_stop_threshold=1e-14)
theta_optim = numqi.optimize.minimize(model, **kwargs)



N0 = 3
tmp0 = np.eye(N0)
np0 = np.einsum(tmp0, [0,1], tmp0, [2,3], tmp0, [4,5], [0,5,1,2,3,4], optimize=True).reshape(N0*N0,N0*N0,N0*N0)
dim_list = [N0*N0,N0*N0,N0*N0]
model = CanonicalPolyadicRankModel(dim_list, 24)
model.set_target(np0)
kwargs = dict(theta0='uniform', tol=1e-14, num_repeat=10, print_every_round=1, early_stop_threshold=1e-14, print_freq=500)
theta_optim = numqi.optimize.minimize(model, **kwargs)
'''
# N0=2
R=6: 0.12500000000008005
R=7: 6.772360450213455e-14: 0.34486097 0.52890762 0.54696604 0.58723123 0.72860213 1.28302568 1.5255219

# N0=3
R=27: 3.9187764144799075e-10
'''

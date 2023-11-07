import torch
import opt_einsum
import numpy as np

import pymanopt

np_rng = np.random.default_rng()

def get_W(n:int):
    ret = np.zeros(2**n, dtype=np.float64)
    ret[2**np.arange(n,dtype=np.int64)] = np.sqrt(1/n)
    return ret


num_qubit = 3
w_state = torch.tensor(get_W(num_qubit), dtype=torch.complex128)
dim_list = [2]*num_qubit
rank = 2
man_positive = pymanopt.manifolds.Positive(1, rank)
# man_positive.random_point(dim)
tmp0 = [pymanopt.manifolds.ComplexCircle(x) for x in dim_list]
man_CP = pymanopt.manifolds.Product([man_positive] + [x for x in tmp0 for _ in range(rank)])

N0 = len(dim_list)
tmp0 = [(rank,),(rank,)] + [(rank,x) for x in dim_list] + [(rank,x) for x in dim_list]
tmp1 = [(N0,),(N0+1,)] + [(N0,x) for x in range(N0)] + [(N0+1,x) for x in range(N0)]
contract_psi_psi = opt_einsum.contract_expression(*[y for x in zip(tmp0,tmp1) for y in x], [])

tmp0 = [dim_list,(rank,)] + [(rank,x) for x in dim_list]
tmp1 = [tuple(range(N0)),(N0,)] + [(N0,x) for x in range(N0)]
contract_target_psi = opt_einsum.contract_expression(*[y for x in zip(tmp0,tmp1) for y in x], [])

target_conj = w_state.reshape([2]*num_qubit).conj()

@pymanopt.function.pytorch(man_CP)
def cost(*args):
    coeff = args[0][0].to(torch.complex128)
    rank = len(coeff)
    assert (len(args)-1) % rank == 0
    tmp0 = (len(args)-1)//rank
    psi_list = [torch.stack(args[(1+x*rank):(1+(x+1)*rank)]) for x in range(tmp0)]
    # complex circle is sqrt(2) normalized
    psi_list = [x/torch.linalg.norm(x,axis=1,keepdims=True) for x in psi_list]
    psi_conj_list = [x.conj() for x in psi_list]
    psi_psi = contract_psi_psi(coeff, coeff.conj(), *psi_list, *psi_conj_list).real
    tmp0 = contract_target_psi(target_conj, coeff, *psi_list)
    loss = 1 - (tmp0.conj() * tmp0).real / psi_psi
    return loss

problem = pymanopt.Problem(man_CP, cost)
optimizer = pymanopt.optimizers.SteepestDescent()
# optimizer = pymanopt.optimizers.TrustRegions() #need hessian but not support complex
result = optimizer.run(problem)

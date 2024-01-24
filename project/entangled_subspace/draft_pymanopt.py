import torch
import opt_einsum
import numpy as np

import pymanopt

np_rng = np.random.default_rng()

def get_W(n:int):
    ret = np.zeros(2**n, dtype=np.float64)
    ret[2**np.arange(n,dtype=np.int64)] = np.sqrt(1/n)
    return ret


def build_CanonicalPolyadicRank_pymanopt_problem(np0, dim_list, rank:int, zero_eps:float=1e-7, bipartition=None):
    dim_list = tuple(int(x) for x in dim_list)
    dim_list_ori = dim_list
    if bipartition is not None:
        bipartition = tuple(sorted({int(x) for x in bipartition}))
        assert (len(bipartition)>=1) and (bipartition[0]>=0) and (bipartition[-1]<len(dim_list))
        tmp0 = np.prod([dim_list[x] for x in bipartition]).item()
        dim_list = tmp0, np.prod(dim_list).item()//tmp0
    assert len(dim_list)>=2
    assert all(x>1 for x in dim_list)
    assert rank>=1

    N0 = len(dim_list)
    tmp0 = [(rank,),(rank,)] + [(rank,x) for x in dim_list] + [(rank,x) for x in dim_list]
    tmp1 = [(N0,),(N0+1,)] + [(N0,x) for x in range(N0)] + [(N0+1,x) for x in range(N0)]
    contract_psi_psi = opt_einsum.contract_expression(*[y for x in zip(tmp0,tmp1) for y in x], [])

    if (np0.shape==dim_list_ori) or ((np0.shape[1:]==dim_list_ori) and np0.shape[0]==1):
        if np0.shape[1:]==dim_list_ori:
            np0 = np0[0]
        if bipartition is not None:
            tmp0 = bipartition + tuple(sorted(set(range(len(dim_list_ori))) - set(bipartition)))
            np0 = np0.reshape(dim_list_ori).transpose(tmp0).reshape(dim_list)
        target = torch.tensor(np0 / np.linalg.norm(np0.reshape(-1)), dtype=torch.complex128)
        tmp0 = [dim_list,(rank,)] + [(rank,x) for x in dim_list]
        tmp1 = [tuple(range(N0)),(N0,)] + [(N0,x) for x in range(N0)]
        contract_target_psi = opt_einsum.contract_expression(*[y for x in zip(tmp0,tmp1) for y in x], [])
    elif np0.shape[1:]==dim_list_ori:
        _,s,v = np.linalg.svd(np0.reshape(np0.shape[0], -1), full_matrices=False)
        np0 = v[s>zero_eps].reshape(-1, *dim_list_ori)
        if bipartition is not None:
            tmp0 = bipartition + tuple(sorted(set(range(len(dim_list_ori))) - set(bipartition)))
            np0 = np0.transpose([0]+[x+1 for x in tmp0]).reshape(-1, *dim_list)
        target = torch.tensor(np0, dtype=torch.complex128)
        tmp0 = [np0.shape,(rank,)] + [(rank,x) for x in dim_list]
        tmp1 = [tuple(range(N0+1)),(N0+1,)] + [(N0+1,x+1) for x in range(N0)]
        contract_target_psi = opt_einsum.contract_expression(*[y for x in zip(tmp0,tmp1) for y in x], [0])
    else:
        tmp0 = (-1,) + dim_list_ori
        assert False, f'invalid shape, np0.shape should be "({dim_list_ori})" or "({tmp0})", but got "{np0.shape}"'
    target_conj = target.conj().resolve_conj()


    tmp0 = pymanopt.manifolds.Positive(1, rank)
    tmp1 = [pymanopt.manifolds.Sphere(2*x) for x in dim_list]
    man_CP = pymanopt.manifolds.Product([tmp0] + [x for x in tmp1 for _ in range(rank)])

    @pymanopt.function.pytorch(man_CP)
    def cost(*args):
        coeff = args[0][0].to(torch.complex128)
        assert (len(args)-1) % rank == 0
        tmp0 = (len(args)-1)//rank
        tmp1 = [torch.complex(x[::2], x[1::2]) for x in args[1:]]
        psi_list = [torch.stack(tmp1[(x*rank):((x+1)*rank)]) for x in range(tmp0)]
        psi_conj_list = [x.conj() for x in psi_list]
        psi_psi = contract_psi_psi(coeff, coeff.conj(), *psi_list, *psi_conj_list).real
        target_psi = contract_target_psi(target_conj, coeff, *psi_list)
        if target_psi.ndim==1:
            loss = 1 - torch.vdot(target_psi, target_psi).real / psi_psi
        else:
            loss = 1 - (target_psi.real**2 + target_psi.imag**2) / psi_psi
        return loss
    problem = pymanopt.Problem(man_CP, cost)
    return problem



def demo_w_state():
    num_qubit = 5
    w_state = get_W(num_qubit).reshape([2]*num_qubit)
    dim_list = [2]*num_qubit
    rank = 2

    problem = build_CanonicalPolyadicRank_pymanopt_problem(w_state, dim_list, rank)
    # optimizer = pymanopt.optimizers.SteepestDescent()
    optimizer = pymanopt.optimizers.TrustRegions()
    result = optimizer.run(problem)



def demo_matrix_matmul():
    N0 = 2
    tmp0 = np.eye(N0)
    np0 = np.einsum(tmp0, [0,1], tmp0, [2,3], tmp0, [4,5], [0,5,1,2,3,4], optimize=True).reshape(N0*N0,N0*N0,N0*N0)
    dim_list = [N0*N0,N0*N0,N0*N0]
    problem = build_CanonicalPolyadicRank_pymanopt_problem(np0, dim_list, rank=7)
    optimizer = pymanopt.optimizers.TrustRegions()
    result = optimizer.run(problem)

    # import numqi
    # model = numqi.matrix_space.DetectCanonicalPolyadicRankModel(dim_list, 24)
    # model.set_target(np0)
    # kwargs = dict(theta0='uniform', tol=1e-14, num_repeat=3, print_every_round=1, early_stop_threshold=1e-14, print_freq=500)
    # theta_optim = numqi.optimize.minimize(model, **kwargs)

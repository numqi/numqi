import itertools
import numpy as np
import scipy.special

import numqi.utils

try:
    import torch
except ImportError:
    torch = None

def dicke_state(num_qubit, k, return_dm=False):
    # http://arxiv.org/abs/1904.07358
    num_term = int(np.prod(np.arange(k+1,num_qubit+1)) / np.prod(np.arange(1,num_qubit-k+1))) #binom{n}{k}
    tmp0 = np.zeros((num_term,int(np.ceil(num_qubit/8))*8), dtype=np.bool_)
    ind0 = np.repeat(np.arange(num_term, dtype=np.int64), k)
    ind1 = tmp0.shape[1]-1 - np.array(list(itertools.combinations(range(num_qubit), k)), dtype=np.int64)[:,::-1].reshape(-1)
    tmp0[ind0,ind1] = 1
    index = np.packbits(tmp0, axis=1)
    if index.shape[1]>1:
        assert index.shape[1]<=8
        if index.shape[1]<8:
            index = np.concatenate([np.zeros((index.shape[0],8-index.shape[1]), dtype=np.uint8), index], axis=1)
        index = index[:,::-1].copy().view(np.int64).reshape(-1)
    else:
        index = index.astype(np.int64).reshape(-1)
    ret = np.zeros(2**num_qubit, dtype=np.float64)
    ret[index] = 1/np.sqrt(num_term)
    if return_dm:
        ret = ret[:,np.newaxis]*ret
    return ret


def mixed_dicke_state(prob, all_dicke_state=None, tag_flat=False):
    is_torch = numqi.utils.is_torch(prob)
    assert prob.ndim==2
    num_qubitA = round(np.log2(prob.shape[0]).item())
    num_qubitB = prob.shape[1]-1
    prob = prob/prob.sum()
    assert prob.shape[0]==2**num_qubitA
    ret = []
    if all_dicke_state is None:
        all_dicke_state = np.stack([dicke_state(num_qubitB, k, return_dm=True) for k in range(num_qubitB+1)])
        if is_torch:
            all_dicke_state = torch.tensor(all_dicke_state)
    if is_torch:
        ret = (prob @ all_dicke_state.view(all_dicke_state.shape[0],-1)).view(prob.shape[0], *all_dicke_state.shape[1:])
    else:
        ret = np.einsum(prob, [0,1], all_dicke_state, [1,2,3], [0,2,3], optimize=True)
    if tag_flat:
        assert not is_torch
        ret_full = np.zeros((2**num_qubitA,2**num_qubitB,2**num_qubitA,2**num_qubitB), dtype=ret.dtype)
        tmp0 = np.arange(2**num_qubitA)
        ret_full[tmp0,:,tmp0] = ret
        ret_full = ret_full.reshape(2**(num_qubitA+num_qubitB),-1)
        ret = ret_full
    return ret


def dicke_state_partial_trace(num_qubit):
    # assert num_qubit
    tmp0 = np.arange(num_qubit+1)
    a00 = (num_qubit - tmp0) / num_qubit
    a01 = np.sqrt(tmp0[1:]*(tmp0[:0:-1]))/num_qubit
    a10 = a01
    a11 = tmp0/num_qubit
    return a00,a01,a10,a11

def partial_trace_AC_to_AB(state, dicke_a_vec=None):
    is_torch = numqi.utils.is_torch(state)
    assert state.ndim==2
    dimA,dimC = state.shape
    if dicke_a_vec is None:
        a00,a01,a10,a11 = dicke_state_partial_trace(dimC-1)
        if is_torch:
            a00,a01,a10,a11 = [torch.tensor(x,dtype=torch.complex128) for x in [a00,a01,a10,a11]]
    else:
        a00,a01,a10,a11 = dicke_a_vec
    state_conj = state.conj()
    rho00 = (state * a00) @ state_conj.T
    rho11 = (state * a11) @ state_conj.T
    rho01 = (state[:,:-1] * a01) @ state_conj[:,1:].T
    rho10 = (state[:,1:] * a10) @ state_conj[:,:-1].T
    if is_torch:
        ret = torch.stack([rho00,rho01,rho10,rho11], dim=2).reshape(dimA,dimA,2,2).transpose(1,2).reshape(dimA*2,dimA*2)
    else:
        ret = np.stack([rho00,rho01,rho10,rho11], axis=2).reshape(dimA,dimA,2,2).transpose(0,2,1,3).reshape(dimA*2,dimA*2)
    return ret


def qudit_dicke_state(*seq):
    # use this in unittest only, it's slow
    seq = np.asarray([int(x) for x in seq], dtype=np.int64)
    d = len(seq)
    num_qudit = seq.sum()
    tmp0 = [int(x) for x,y in enumerate(seq) for _ in range(y)]
    tmp1 = np.unique(np.array(list(itertools.permutations(tmp0)), dtype=np.int64), axis=0)
    tmp2 = d**(np.arange(num_qudit)[::-1])
    ret = np.zeros(d**num_qudit, dtype=np.float64)
    ret[tmp1 @ tmp2] = 1/np.sqrt(len(tmp1))
    return ret


def qudit_dicke_state_partial_trace(d, num_qudit):
    assert (d>1) and (num_qudit>=1)
    hf0 = lambda d,n: [(n,)] if (d<=1) else [(x,)+y for x in range(n+1) for y in hf0(d-1,n-x)]
    klist = hf0(d, num_qudit)
    # numerical issue if using factorial
    # len_klist = scipy.special.factorial(num_qudit+d-1) // (scipy.special.factorial(d-1) * scipy.special.factorial(num_qudit))
    len_klist = scipy.special.binom(num_qudit+d-1, d-1)
    assert len(klist)==len_klist
    klist_to_index = {y:x for x,y in enumerate(klist)}
    klist_to_ij = dict()
    klist_np = np.array(klist, dtype=np.int64)
    ret = []
    for ind0 in range(d):
        for ind1 in range(d):
            if ind0==ind1:
                tmp0 = np.arange(len(klist))
                tmp1 = np.arange(len(klist))
                tmp2 = klist_np[:,ind0] / num_qudit
                ret.append((tmp0,tmp1,tmp2))
                for x,y in enumerate(klist):
                    klist_to_ij[(y,y,ind0*d+ind1)] = x
            else:
                tmp0 = []
                for x,y in enumerate(klist):
                    tmp1 = list(y)
                    tmp1[ind0] -= 1
                    tmp1[ind1] += 1
                    tmp1 = tuple(tmp1)
                    if tmp1 in klist_to_index:
                        tmp0.append((x,klist_to_index[tmp1]))
                for x,(y0,y1) in enumerate(tmp0):
                    klist_to_ij[(klist[y0],klist[y1],ind0*d+ind1)] = x
                tmp1 = np.array([x[0] for x in tmp0], dtype=np.int64)
                tmp2 = np.array([x[1] for x in tmp0], dtype=np.int64)
                tmp3 = np.sqrt(klist_np[tmp1,ind0]*klist_np[tmp2,ind1])/num_qudit
                ret.append((tmp1,tmp2,tmp3))
    return ret,klist,klist_to_ij


def qudit_partial_trace_AC_to_AB(state, dicke_Bij):
    is_torch = numqi.utils.is_torch(state)
    assert state.ndim==2
    dimA,dimC = state.shape
    dimB = int(np.sqrt(len(dicke_Bij)))
    assert len(dicke_Bij)==dimB*dimB
    ret = []
    state_conj = state.conj()
    for ind0,ind1,value in dicke_Bij:
        ret.append((state[:,ind0] * value) @ state_conj[:,ind1].T)
    if is_torch:
        ret = torch.stack(ret, dim=2).reshape(dimA,dimA,dimB,dimB).transpose(1,2).reshape(dimA*dimB,dimA*dimB)
    else:
        ret = np.stack(ret, axis=2).reshape(dimA,dimA,dimB,dimB).transpose(0,2,1,3).reshape(dimA*dimB,dimA*dimB)
    return ret

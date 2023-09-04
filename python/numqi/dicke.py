import itertools
import numpy as np
import scipy.special
import torch

import numqi.utils

def get_dicke_basis(num_qudit:int, dim:int):
    '''return Dicke basis for n qudits

    see http://arxiv.org/abs/1904.07358 for more information

    Parameters:
        num_qudit (int): number of qudits
        dim (int): dimension of each qudit

    Returns:
        ret (np.ndarray): shape (binom{num_qudit+dim-1}{num_qudit}, dim**num_qudit) float64
    '''
    # use this in unittest only, it's slow and memory consuming
    assert dim>=2
    assert num_qudit>=1
    klist = get_dicke_klist(num_qudit, dim)
    ret = np.zeros((len(klist), dim**num_qudit), dtype=np.float64)
    base = dim**(np.arange(num_qudit)[::-1])
    for ind0,ki in enumerate(klist):
        tmp0 = [int(x) for x,y in enumerate(ki) for _ in range(y)]
        tmp1 = np.unique(np.array(list(itertools.permutations(tmp0)), dtype=np.int64), axis=0)
        ret[ind0, tmp1 @ base] = 1/np.sqrt(len(tmp1))
    return ret


def get_dicke_klist(num_qudit:int, dim:int):
    '''return klist for n qudit Dicke state

    Parameters:
        num_qudit (int): number of qudit
        dim (int): dimension of qudit

    Returns:
        ret (list): list of klist, each element is a tuple of int.
                the list is of length `scipy.special.binom(n+d-1, d-1)`.
                the tuple is of length `dim` and the sum of each tuple is `num_qudit`
    '''
    assert dim>=2
    assert num_qudit>=1
    hf0 = lambda d,n: [(n,)] if (d<=1) else [(x,)+y for x in range(n+1) for y in hf0(d-1,n-x)]
    ret = hf0(dim, num_qudit)
    return ret



def get_qubit_dicke_partial_trace(num_qubit:int):
    '''return partial trace of qubit Dicke state

    Parameters:
        num_qubit (int): number of qubit

    Returns:
        a00 (np.ndarray): shape (num_qubit+1,) float64
        a01 (np.ndarray): shape (num_qubit,) float64 `a01=a10`
        a11 (np.ndarray): shape (num_qubit+1,) float64
    '''
    assert num_qubit>1
    tmp0 = np.arange(num_qubit+1)
    a00 = tmp0 / num_qubit
    a01 = np.sqrt(tmp0[1:]*(tmp0[:0:-1]))/num_qubit
    # a10 = a01
    a11 = (num_qubit - tmp0)/num_qubit
    return a00,a01,a11


# old name: qudit_dicke_state_partial_trace
def get_partial_trace_ABk_to_AB_index(num_qudit:int, dim:int, return_tensor=False):
    r'''return index for partial trace of qudit Dicke state `numqi.dicke.partial_trace_ABk_to_AB`

    $$ B_{rsab} = \mathrm{Tr}_{A^{n-1}} \left[ \langle r|D_{na}\rangle\langle D_{nb}| s\rangle \right] $$

    Parameters:
        num_qudit (int): number of qudit
        dim (int): dimension of qudit
        return_tensor (bool): if True, return $B_{rsab}$, otherwise return indexing used for $B_{rsab}$.
                `return_tensor=True` is useful for unittest. `return_tensor=False` use less memory

    Returns:
        ret (list,np.ndarray): if `return_tensor=False`, list of tuple, each tuple is of length 3,
            and the first two elements are list of int, the third element is np.ndarray of `float64`.
            if `return_tensor=True`, return $B_{rsab}$, shape (dim, dim, #klist, #klist)
    '''
    assert (dim>1) and (num_qudit>=1)
    klist = get_dicke_klist(num_qudit, dim)
    len_klist = get_dicke_number(num_qudit, dim)
    assert len(klist)==len_klist
    klist_to_index = {y:x for x,y in enumerate(klist)}
    klist_np = np.array(klist, dtype=np.int64)
    Bij = []
    for ind0 in range(dim):
        for ind1 in range(dim):
            if ind0==ind1:
                tmp0 = np.arange(len_klist)
                tmp1 = np.arange(len_klist)
                tmp2 = klist_np[:,ind0] / num_qudit
                Bij.append((tmp0,tmp1,tmp2))
            else:
                tmp0 = []
                for x,y in enumerate(klist):
                    tmp1 = list(y)
                    tmp1[ind0] -= 1
                    tmp1[ind1] += 1
                    tmp1 = tuple(tmp1)
                    if tmp1 in klist_to_index:
                        tmp0.append((x,klist_to_index[tmp1]))
                tmp1 = np.array([x[0] for x in tmp0], dtype=np.int64)
                tmp2 = np.array([x[1] for x in tmp0], dtype=np.int64)
                tmp3 = np.sqrt(klist_np[tmp1,ind0]*klist_np[tmp2,ind1])/num_qudit
                Bij.append((tmp1,tmp2,tmp3))
    if return_tensor:
        Brsab = np.zeros((dim*dim,len_klist,len_klist), dtype=np.complex128)
        for ind0 in range(dim*dim):
            indI,indJ,value = Bij[ind0]
            Brsab[ind0,indI,indJ] = value
        Brsab = Brsab.reshape(dim,dim,len_klist,len_klist)
        ret = Brsab
    else:
        ret = Bij
    return ret


def get_dicke_number(num_qudit:int, dim:int):
    # numerical issue if using factorial
    ret = int(scipy.special.binom(num_qudit+dim-1, dim-1))
    return ret


# old name: qudit_partial_trace_AC_to_AB
def partial_trace_ABk_to_AB(state, dicke_Bij):
    assert state.ndim==2
    dimA,dimBk = state.shape
    dimB = int(np.sqrt(len(dicke_Bij)))
    assert len(dicke_Bij)==dimB*dimB
    ret = []
    state_conj = state.conj()
    for ind0,ind1,value in dicke_Bij:
        ret.append((state[:,ind0] * value) @ state_conj[:,ind1].T)
    if isinstance(state, torch.Tensor):
        ret = torch.stack(ret, dim=2).reshape(dimA,dimA,dimB,dimB).transpose(1,2).reshape(dimA*dimB,dimA*dimB)
    else:
        ret = np.stack(ret, axis=2).reshape(dimA,dimA,dimB,dimB).transpose(0,2,1,3).reshape(dimA*dimB,dimA*dimB)
    return ret

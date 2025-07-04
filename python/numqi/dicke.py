import functools
import math
import itertools
import numpy as np
import scipy.special
import scipy.sparse
import torch
import opt_einsum


def _dicke_hf0(klist, base, dim, num_qudit):
    ret = np.zeros(dim**num_qudit, dtype=np.float64)
    tmp0 = [int(x) for x,y in enumerate(klist) for _ in range(y)]
    tmp1 = np.unique(np.array(list(itertools.permutations(tmp0)), dtype=np.int64), axis=0)
    ret[tmp1 @ base] = 1/np.sqrt(len(tmp1))
    return ret


def Dicke(*klist:tuple[int]):
    r'''return Dicke state for n qudits

    see [arxiv-link](http://arxiv.org/abs/1904.07358) for more information

    Parameters:
        klist (tuple[int]): list of int, each int is the number of qudit in each level, `dim=len(klist)`, `num_qudit=sum(klist)`

    Returns:
        ret (np.ndarray): shape (dim**num_qudit)
    '''
    klist = tuple(int(x) for x in klist)
    assert (len(klist)>=2) and all(x>=0 for x in klist)
    dim = len(klist)
    num_qudit = sum(klist)
    base = dim**(np.arange(num_qudit)[::-1])
    ret = _dicke_hf0(klist, base, dim, num_qudit)
    return ret


def get_dicke_basis(num_qudit:int, dim:int):
    r'''return Dicke basis for n qudits

    see [arxiv-link](http://arxiv.org/abs/1904.07358) for more information

    Parameters:
        num_qudit (int): number of qudits
        dim (int): dimension of each qudit

    Returns:
        ret (np.ndarray): shape (binom{num_qudit+dim-1}{num_qudit}, dim**num_qudit) float64
    '''
    # use this in unittest only, it's slow and memory consuming
    assert (dim>=2) and (num_qudit>=1)
    if dim==2: #just for speed
        if num_qudit==1:
            ret = 1-np.eye(2, dtype=np.float64)
        else:
            base = 2**(np.arange(num_qudit)[::-1])
            ret = np.zeros((num_qudit+1, 2**num_qudit), dtype=np.float64)
            ret[0,-1] = 1 #1111 first
            ret[-1,0] = 1
            for ind0 in range(1, num_qudit):
                ind1 = base[np.array(list(itertools.combinations(range(num_qudit), num_qudit-ind0)))].sum(axis=1)
                ret[ind0, ind1] = 1/np.sqrt(scipy.special.binom(num_qudit, ind0))
    else:
        klist = get_dicke_klist(num_qudit, dim)
        base = dim**(np.arange(num_qudit)[::-1])
        ret = np.stack([_dicke_hf0(x, base, dim, num_qudit) for x in klist])
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
    r'''return partial trace of qudit Dicke state

    Parameters:
        state (np.ndarray or torch.Tensor): input state, shape (dimA, dimBk)
        dicke_Bij (list): list of tuples, each tuple contains two lists of int and one np.ndarray of float64

    Returns:
        ret (np.ndarray or torch.Tensor): partial trace result, shape (dimA*dimB, dimA*dimB)
    '''
    assert state.ndim==2
    dimA, dimBk = state.shape
    dimB = int(np.sqrt(len(dicke_Bij)))
    assert len(dicke_Bij) == dimB * dimB
    ret = []
    state_conj = state.conj()
    for ind0, ind1, value in dicke_Bij:
        ret.append((state[:, ind0] * value) @ state_conj[:, ind1].T)
    if isinstance(state, torch.Tensor):
        ret = torch.stack(ret, dim=2).reshape(dimA, dimA, dimB, dimB).transpose(1, 2).reshape(dimA * dimB, dimA * dimB)
    else:
        ret = np.stack(ret, axis=2).reshape(dimA, dimA, dimB, dimB).transpose(0, 2, 1, 3).reshape(dimA * dimB, dimA * dimB)
    return ret


def get_qubit_dicke_rdm_tensor(n:int, rdm:int):
    assert 1<=rdm<n
    hf0 = lambda x: np.sqrt(1-np.arange(x)/x)
    hf0a = lambda x: np.concatenate([np.diag(hf0(x)), np.zeros((1,x))], axis=0)
    hf1 = lambda x: np.sqrt(np.arange(1,x+1)/x)
    hf1a = lambda x: np.concatenate([np.zeros((1,x)), np.diag(hf1(x))], axis=0)
    hf2 = lambda x: np.einsum(x, [0,1], x, [2,3], [0,2,1,3], optimize=True)
    ret = hf2(hf0a(rdm+1)) + hf2(hf1a(rdm+1))
    for x in range(rdm+1, n):
        tmp0 = hf2(hf0a(x+1)) + hf2(hf1a(x+1))
        ret = np.einsum(ret, [0,1,2,3], tmp0, [4,5,0,1], [4,5,2,3], optimize=True)
    return ret


def get_qubit_dicke_rdm_pauli_tensor(n:int, rdm:int, kind:str='numpy'):
    assert 1<=rdm<n
    assert kind in {'numpy', 'torch', 'scipy-csr0', 'scipy-csr01', 'torch-csr0', 'torch-csr01'}
    _pauli_dict = {'X':np.array([[0,1],[1,0]]), 'Y':np.array([[0,-1j],[1j,0]]), 'Z':np.array([[1,0],[0,-1]])}
    Tuab_list = []
    factor_list = []
    pauli_str_list = []
    for wt in range(1, rdm+1):
        Tabrs = get_qubit_dicke_rdm_tensor(n, wt)
        basis = get_dicke_basis(wt, 2)[::-1].reshape([-1]+[2]*wt) #TODO, skip creating this tensor
        factor_common = scipy.special.binom(n, wt)
        pauli_str_list.append([''.join(x) for x in itertools.combinations_with_replacement('XYZ', wt)])
        for xyz in pauli_str_list[-1]:
            tmp0 = [sum(1 for x in xyz if s==x) for s in 'XYZ']
            factor_list.append(math.factorial(wt)//(math.factorial(tmp0[0])*math.factorial(tmp0[1])*math.factorial(tmp0[2]))*factor_common)
            tmp1 = [(_pauli_dict[x], 4+2*i, 5+2*i) for i,x in enumerate(xyz)]
            tmp2 = [2]+[x[2] for x in tmp1]
            tmp3 = [3] + [x[1] for x in tmp1]
            tmp4 = [y for x in tmp1 for y in (x[0], (x[1],x[2]))]
            Tuab_list.append(np.einsum(Tabrs, [0,1,2,3], basis, tmp2, basis, tmp3, *tmp4, [0,1], optimize=True))
    Tuab_list = np.stack(Tuab_list, axis=0)
    factor_list = np.array(factor_list, dtype=np.float64)
    weight_count = {(i+1):len(x) for i,x in enumerate(pauli_str_list)}
    pauli_str_list = [y for x in pauli_str_list for y in x]
    if kind=='torch':
        Tuab_list = torch.tensor(Tuab_list, dtype=torch.complex128)
        factor_list = torch.tensor(factor_list, dtype=torch.float64)
    elif kind in ('scipy-csr0', 'scipy-csr01', 'torch-csr0', 'torch-csr01'):
        shape = (Tuab_list.shape[0], (n+1)**2) if kind[-1]=='0' else (Tuab_list.shape[0]*(n+1), n+1)
        Tuab_list = scipy.sparse.csr_array(Tuab_list.reshape(shape))
        if kind in ('torch-csr0', 'torch-csr01'):
            factor_list = torch.tensor(factor_list, dtype=torch.float64)
            tmp0 = torch.tensor(Tuab_list.indptr, dtype=torch.int64)
            tmp1 = torch.tensor(Tuab_list.indices, dtype=torch.int64)
            tmp2 = torch.tensor(Tuab_list.data, dtype=torch.complex128)
            Tuab_list = torch.sparse_csr_tensor(tmp0, tmp1, tmp2, dtype=torch.complex128)
    return Tuab_list, factor_list, pauli_str_list, weight_count


@functools.lru_cache
def _u2_to_dicke_info(n:int, tag_torch:bool):
    zero_n_int = np.arange(n+1, dtype=np.int64)
    I = zero_n_int.reshape(-1,1,1)
    J = zero_n_int.reshape(1,-1,1)
    M = zero_n_int.reshape(1,1,-1)
    tmp0 = np.sqrt(scipy.special.binom(n, zero_n_int))
    binom_nij = tmp0 / tmp0.reshape(-1,1)
    pascal = scipy.linalg.pascal(n+1, kind='lower', exact=False)
    tmp0 = binom_nij.reshape(n+1,n+1,1) * (pascal * pascal[n-J, np.maximum(I-M,0)])
    ret = {'binom':tmp0, 'mask':(J>=M)[0], 'ind0':np.clip(n-J-I+M,0,n), 'ind12':np.maximum((J-M)[0],0)}
    if tag_torch:
        tmp0 = {'binom':torch.float64, 'mask':torch.bool, 'ind0':torch.int64, 'ind12':torch.int64}
        ret = {k:torch.tensor(v, dtype=tmp0[k]) for k,v in ret.items()}
    return ret

def u2_to_dicke_info(n:int, tag_torch:bool=False):
    assert n>=2
    ret = _u2_to_dicke_info(int(n), bool(tag_torch))
    return ret

_torch_cdtype = {torch.float32:torch.complex64, torch.complex64:torch.complex64,
                 torch.float64:torch.complex128, torch.complex128:torch.complex128}

def u2_to_dicke(np0:np.ndarray|torch.Tensor, n:int, _info:dict|None=None):
    assert (np0.shape==(2,2)) and (n>=1)
    if n==1:
        ret = np0
    else:
        is_torch = isinstance(np0, torch.Tensor)
        if _info is None:
            _info = u2_to_dicke_info(n, is_torch)
        else:
            assert is_torch == isinstance(_info['binom'], torch.Tensor)
        if is_torch:
            cdtype = _torch_cdtype[np0.dtype]
            det_factor = torch.sqrt((np0[0,0]*np0[1,1] - np0[0,1]*np0[1,0]).to(cdtype))
            tmp0 = (np0/det_factor).reshape(4,1) #CAUTION: d(x^n)/dx is bad defined when (x=0,n=1)
            tmp1 = torch.concat([tmp0, tmp0**torch.arange(2,n+1, dtype=torch.int64)], axis=1)
            # tmp0 = (np0/det_factor).reshape(4,1)**torch.arange(1,n+1, dtype=torch.int64)
            abcd = torch.concat([torch.ones(4,1,dtype=cdtype),tmp1], axis=1)
        else:
            det_factor = np.sqrt(np.complex128(np0[0,0]*np0[1,1] - np0[0,1]*np0[1,0]))
            tmp0 = (np0/det_factor).reshape(4,1)**np.arange(1,n+1, dtype=np.int64)
            abcd = np.concatenate([np.ones((4,1)),tmp0], axis=1)
        # ijm
        term2 = abcd[0][_info['ind0']]
        term3 = (abcd[1][_info['ind12']] * _info['mask'])
        term4 = (abcd[2][_info['ind12']] * _info['mask'])
        ret = opt_einsum.contract(_info['binom'], [0,1,2], term2, [0,1,2], term3, [1,2], term4, [0,2], abcd[3], [2], [0,1])
        ret = ret * (det_factor**n)
    return ret

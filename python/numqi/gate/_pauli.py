import time
import sys
import contextlib
import collections
import functools
import itertools
import numpy as np
import scipy.sparse
from tqdm.auto import tqdm
import functools

from ._internal import pauli
import numqi.group.spf2

endianness_map = {
    '>': 'big',
    '<': 'little',
    '=': sys.byteorder,
    '|': 'not applicable',
}


hf_kron = lambda x: functools.reduce(np.kron, x)

_one_pauli_str_to_np = dict(zip('IXYZ', [pauli.s0, pauli.sx, pauli.sy, pauli.sz]))
# do not use this "pauli" below for that this name might be redefined in functions below

## pauli representation
# str+sign: 'XIZYX', 1j
# index+sign: 0 1
# F2
# np.ndarray



@functools.lru_cache
def get_pauli_group(num_qubit, /, kind='numpy', use_sparse=False):
    # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    # II IX IY IZ XI XX XY XZ YI YX YY YZ ZI ZX ZY ZZ
    assert kind in {'numpy','str','str_to_index'}
    if use_sparse:
        assert kind=='numpy'
    if kind=='numpy':
        if use_sparse:
            # @20230309 scipy.sparse.kron have not yet been ported https://docs.scipy.org/doc/scipy/reference/sparse.html
            hf0 = lambda x,y: scipy.sparse.coo_array(scipy.sparse.kron(x,y,format='coo'))
            hf_kron = lambda x: functools.reduce(hf0, x)
            tmp0 = [scipy.sparse.coo_array(_one_pauli_str_to_np[x]) for x in 'IXYZ']
            tmp1 = [(0,1,2,3)]*num_qubit
            ret = [hf_kron([tmp0[y] for y in x]) for x in itertools.product(*tmp1)]
            # x = ret[0]
            # x.toarray()[x.row, x.col] #x.data
        else:
            hf_kron = lambda x: functools.reduce(np.kron, x)
            tmp0 = [_one_pauli_str_to_np[x] for x in 'IXYZ']
            tmp1 = [(0,1,2,3)]*num_qubit
            ret = np.stack([hf_kron([tmp0[y] for y in x]) for x in itertools.product(*tmp1)])
    else:
        tmp0 = tuple(''.join(x) for x in itertools.product(*['IXYZ']*num_qubit))
        if kind=='str':
            ret = tmp0
        else: #str_to_index
            ret = {y:x for x,y in enumerate(tmp0)}
    return ret

def _pauli_index_int_to_str(index:int, num_qubit:int):
    assert 0<=index<=4**num_qubit
    ret = ''
    tmp0 = 'IXYZ'
    for _ in range(num_qubit):
        ret = tmp0[index%4] + ret
        index //= 4
    return ret

def _pauli_str_to_index_int(str_:str):
    tmp0 = {'I':0, 'X':1, 'Y':2, 'Z':3}
    ret = 0
    for x in str_:
        ret = ret*4 + tmp0[x]
    return ret

def pauli_index_to_str(index:int|np.ndarray, num_qubit:int):
    r'''convert Pauli operator in the index representation to string representation

    Parameters:
        index (int|np.ndarray): Pauli index, {0,1,2,3}, support batch
        num_qubit (int): number of qubit

    Returns:
        pauli_str (str|np.ndarray): Pauli string, e.g. 'XIZYX'
    '''

    if isinstance(index, int):
        ret = _pauli_index_int_to_str(index, num_qubit)
    else:
        assert isinstance(index, np.ndarray)
        shape = index.shape
        tmp0 = index.reshape(-1).astype(np.uint64).tolist()
        ret = np.array([_pauli_index_int_to_str(x, num_qubit) for x in tmp0], dtype=f'U{num_qubit}')
        ret = ret.reshape(shape)
    return ret

def pauli_str_to_index(str_:str|np.ndarray):
    r'''convert Pauli operator in the string representation to index representation

    Parameters:
        str_ (str|np.ndarray): Pauli string, e.g. 'XIZYX', uppercase only, support batch

    Returns:
        index (int|np.ndarray): Pauli index, {0,1,2,3}
    '''
    if isinstance(str_, str):
        ret = _pauli_str_to_index_int(str_)
    else:
        assert isinstance(str_, np.ndarray) and (str_.dtype.kind=='U')
        shape = str_.shape
        str_ = str_.reshape(-1).tolist()
        num_qubit = len(str_[0])
        assert 0<num_qubit<=32
        ret = np.array([_pauli_str_to_index_int(x) for x in str_], dtype=np.uint64)
        ret = ret.reshape(shape)
    return ret



def _pauli_index_int_to_F2(index:int, num_qubit:int, with_sign:bool):
    tmp0 = _pauli_index_int_to_str(index, num_qubit)
    ret = pauli_str_to_F2(tmp0)
    if with_sign==False:
        ret = ret[2:]
    return ret


def pauli_index_to_F2(index, num_qubit:int, with_sign:bool=True):
    r'''convert Pauli operator in the index representation to F2 representation

    Parameters:
        index (int|np.ndarray): Pauli index, {0,1,2,3}, support batch
        num_qubit (int): number of qubit
        with_sign (bool): if True, return sign as well

    Returns:
        np0 (np.ndarray): shape (`2n+2`,), dtype=np.uint8, support batch
    '''
    if isinstance(index, int):
        ret = _pauli_index_int_to_F2(index, num_qubit, with_sign)
    else:
        assert num_qubit <= 32
        index = np.asarray(index, dtype=np.uint64)
        shape = index.shape
        index = index.reshape(-1)
        if endianness_map[index.dtype.byteorder]=='little':
            index = index.byteswap()
        index_z2 = np.unpackbits(index.view(np.uint8).reshape(index.shape[0],-1), axis=1, bitorder='big')[:,-(2*num_qubit):].reshape(-1,num_qubit,2)
        ret = np.zeros((len(index_z2)*num_qubit, 2), dtype=np.uint8)
        for x,y in [((0,1),(1,0)), ((1,0),(1,1)), ((1,1),(0,1))]:
            tmp0 = np.all(index_z2==np.array(x, dtype=np.uint8), axis=2).reshape(-1)
            ret[tmp0] = np.array(y, dtype=np.uint8)
        ret = ret.reshape(-1, num_qubit, 2).transpose(0,2,1).reshape(-1,2*num_qubit)
        if with_sign:
            tmp0 = np.einsum(ret[:,:num_qubit], [0,1], ret[:,num_qubit:], [0,1], [0], optimize=True) % 4
            ret = np.concatenate([np.stack([tmp0//2, tmp0%2], axis=1), ret], axis=1)
        ret = ret.reshape(shape+(ret.shape[-1],))
    return ret


def pauli_F2_to_index(np0:np.ndarray, with_sign:bool=True):
    assert (np0.dtype.type==np.uint8) and (np0.ndim>=1) and (np0.shape[-1]%2==0)
    if with_sign:
        np0 = np0[...,2:]
    assert np0.shape[-1]>=2
    num_qubit = np0.shape[-1]//2
    if np0.ndim==1:
        ret = 0
        tmp0 = {(0,0):0, (1,0):1, (0,1):3, (1,1):2}
        for x,y in zip(np0[:num_qubit].tolist(), np0[num_qubit:].tolist()):
            ret = ret*4 + tmp0[(x,y)]
    else:
        shape = np0.shape
        np0 = np0.reshape(-1,2*num_qubit)
        N0 = np0.shape[0]
        np0 = np0.reshape(N0,2,num_qubit).transpose(0,2,1).reshape(N0*num_qubit,2)
        np1 = np.zeros_like(np0)
        for x,y in [((0,1),(1,0)), ((1,0),(1,1)), ((1,1),(0,1))]:
            ind0 = np.all(np0==np.array(y,dtype=np.uint8), axis=1)
            np1[ind0] = np.array(x, dtype=np.uint8)
        tmp0 = 1<<np.arange(2*num_qubit)[::-1]
        ret = np1.reshape(N0, num_qubit*2) @ tmp0
        ret = ret.reshape(shape[:-1])
    return ret


def pauli_F2_to_str(np0):
    r'''convert Pauli operator in the F2 representation to string representation

    Parameters:
        np0 (np.ndarray): shape (`2n+2`,), dtype=np.uint8, support batch

    Returns:
        pauli_str (str): Pauli string, e.g. 'XIZYX'
        sign (complex): coefficient of Pauli string, {1, i, -1, -i}
    '''
    assert (np0.dtype.type==np.uint8) and (np0.ndim>=1) and (np0.shape[-1]%2==0) and (np0.shape[-1]>=2)
    is_single = (np0.ndim==1)
    shape = np0.shape[:-1]
    np0 = np0.reshape(-1, np0.shape[-1])
    N0 = (np0.shape[-1]-2)//2
    bitX = np0[:,2:(2+N0)]
    bitZ = np0[:, (2+N0):]
    tmp1 = (2*np0[:,0] + np0[:,1] + 3*np.einsum(bitX,[0,1],bitZ,[0,1],[0],optimize=True))%4 #XZ=-iY
    sign = 1j**tmp1
    tmp0 = {(0,0):'I', (1,0):'X', (0,1):'Z', (1,1):'Y'}
    pauli_str = [''.join(tmp0[(int(x),int(y))] for x,y in zip(X,Z)) for X,Z in zip(bitX,bitZ)]
    if is_single:
        pauli_str = pauli_str[0]
        sign = sign.item()
    else:
        pauli_str = np.array(pauli_str, dtype=np.dtype(f'U{N0}')).reshape(shape)
        sign = sign.reshape(shape)
    return pauli_str, sign


def pauli_str_to_F2(pauli_str:str|np.ndarray, sign=1):
    r'''convert Pauli string to Pauli operator in the F2 representation

    Parameters:
        pauli_str (str): Pauli string, e.g. 'XIZYX', uppercase only
        sign (complex): sign of Pauli string, {1, i, -1, -i}

    Returns:
        np0 (np.ndarray): shape (`2n+2`,), dtype=np.uint8
    '''
    is_single = isinstance(pauli_str, str)
    if is_single:
        N0 = len(pauli_str)
        pauli_str = [pauli_str]
        sign_ri = [(int(sign.real),int(sign.imag))]
    else:
        assert isinstance(pauli_str, np.ndarray) and (pauli_str.dtype.kind=='U')
        shape = pauli_str.shape
        pauli_str = pauli_str.reshape(-1).tolist()
        N0 = len(pauli_str[0])
        sign = np.broadcast_to(np.asarray(sign), shape).reshape(-1)
        sign_ri = [(int(x.real),int(x.imag)) for x in sign]
    tmp0 = set('IXYZ')
    assert all(set(x)<=tmp0 for x in pauli_str)
    bitX = np.array([[(1 if ((y=='X') or (y=='Y')) else 0) for y in x] for x in pauli_str], dtype=np.uint8)
    bitZ = np.array([[(1 if ((y=='Z') or (y=='Y')) else 0) for y in x] for x in pauli_str], dtype=np.uint8)

    tmp0 = {(1,0):0, (0,1):1, (-1,0):2, (0,-1):3}
    tmp1 = (np.einsum(bitX, [0,1], bitZ, [0,1], [0], optimize=True) + np.array([tmp0[x] for x in sign_ri], dtype=np.uint8)) % 4
    ret = np.concatenate([np.stack([tmp1//2, tmp1%2], axis=1), bitX, bitZ], axis=1)
    if is_single:
        ret = ret[0]
    else:
        ret = ret.reshape(shape+(2*N0+2,))
    return ret

class PauliOperator:
    def __init__(self, pauli_F2):
        assert (pauli_F2.dtype.type==np.uint8)
        assert (pauli_F2.ndim==1) and (pauli_F2.shape[0]%2==0) and (pauli_F2.shape[0]>=2)
        self.F2 = pauli_F2
        self.num_qubit = len(pauli_F2)//2 - 1
        self._str = None
        self._sign = None
        self._np_list = None

    @property
    def sign(self):
        if self._sign is None:
            self._str,self._sign = pauli_F2_to_str(self.F2)
        return self._sign

    @property
    def str_(self):
        if self._str is None:
            self._str,self._sign = pauli_F2_to_str(self.F2)
        return self._str

    @property
    def np_list(self):
        if self._np_list is None:
            self._np_list = tuple(_one_pauli_str_to_np[x] for x in self.str_)
        return list(self._np_list)

    def __len__(self):
        return self.num_qubit

    def __matmul__(self, b):
        assert isinstance(b, PauliOperator) and (self.num_qubit==b.num_qubit)
        tmp0 = (self.F2 + b.F2) % 2
        tmp1 = np.dot(self.F2[(2+self.num_qubit):], b.F2[2:(2+self.num_qubit)]) % 2
        tmp2 = (self.F2[1] + b.F2[1])//2
        tmp0[0] = (tmp0[0] + tmp1 + tmp2) % 2
        ret = PauliOperator(tmp0)
        return ret

    def commutate_with(self, b):
        assert isinstance(b, PauliOperator) and (self.num_qubit==b.num_qubit)
        N0 = self.num_qubit
        ret = (np.dot(self.F2[2:(2+N0)], b.F2[(2+N0):]) + np.dot(self.F2[(2+N0):], b.F2[2:(2+N0)])) % 2 == 0
        return ret

    def inverse(self):
        tmp0 = self.F2.copy()
        tmp0[0] = (self.F2[0] + self.F2[1] + np.dot(self.F2[2:(2+self.num_qubit)], self.F2[(2+self.num_qubit):])) % 2
        ret = PauliOperator(tmp0)
        return ret

    def __str__(self):
        sign_ri = int(self.sign.real), int(self.sign.imag)
        tmp0 = {(1,0):'', (0,1):'i', (-1,0):'-', (0,-1):'-i'}[sign_ri]
        tmp1 = ','.join([str(x) for x in self.F2])
        ret = tmp0 + self.str_ + f' [{tmp1}]'
        return ret

    __repr__ = __str__

    @staticmethod
    def from_F2(np0):
        ret = PauliOperator(np0)
        return ret

    @staticmethod
    def from_str(pauli_str, sign=1):
        np0 = pauli_str_to_F2(pauli_str, sign=sign)
        ret = PauliOperator(np0)
        return ret

    @staticmethod
    def from_index(index:int, num_qubit:int):
        ret = PauliOperator(_pauli_index_int_to_F2(index, num_qubit, with_sign=True))
        return ret

    @staticmethod
    def from_np_list(np_list, sign=1):
        # .T for trace(AB)= dot(vec(A.T), vec(B))
        _pauli_np_list = np.stack([_one_pauli_str_to_np[x].T for x in 'IXYZ'], axis=2).reshape(4, 4)
        tmp0 = np.stack(np_list, axis=0).reshape(-1, 4) @ _pauli_np_list
        assert (tmp0.imag.max() < 1e-10) and (tmp0.imag.min() > -1e-10) #pure real
        tmp0 = tmp0.real
        assert np.abs(tmp0.sum(axis=1) - 2).max() < 1e-10
        assert np.abs(tmp0.max(axis=1) - 2).max() < 1e-10
        ind0 = np.argmax(tmp0, axis=1).tolist()
        tmp1 = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}
        pauli_str = ''.join(tmp1[x] for x in ind0)
        np0 = pauli_str_to_F2(pauli_str, sign=sign)
        ret = PauliOperator(np0)
        return ret

    @staticmethod
    def from_full_matrix(np0):
        # use this for unittest only, for that this is not efficient
        assert (np0.ndim==2) and (np0.shape[0]==np0.shape[1]) and (np0.shape[0]>=2)
        N0 = int(np.log2(np0.shape[0]))
        assert 2**N0 == np0.shape[0]
        IXYZ = [_one_pauli_str_to_np[x]*y for x,y in zip('IXYZ',[1,1,-1j,1])]
        bitXZ = np.zeros(2*N0, dtype=np.uint8)
        for ind0 in range(N0):
            if np0.shape[0]==2:
                np1 = np0.reshape(-1)
            else:
                tmp0 = np0.reshape(2,2**(N0-ind0-1),2,2**(N0-ind0-1))
                tmp1 = np.einsum(tmp0, [0,1,2,3], tmp0.conj(), [4,1,5,3], [0,2,4,5], optimize=True)
                EVL,EVC = np.linalg.eigh(tmp1.reshape(4,4)/np0.shape[0])
                assert np.abs(EVL - np.array([0,0,0,1])).max() < 1e-7
                np1 = EVC[:,3] * np.sqrt(2)
                # tmp2.reshape(2,2) #should be one of the I,X,Y,Z (ignore phase factor)
            for ind1,pauli in enumerate(IXYZ):
                if abs(abs(np.vdot(pauli.reshape(-1), np1))-2) < 1e-7:
                    bitXZ[ind0] = 1 if (ind1 in (1,2)) else 0
                    bitXZ[ind0+N0] = 1 if (ind1 in (2,3)) else 0
                    tmp0 = np0.reshape(2,2**(N0-ind0-1),2,2**(N0-ind0-1))
                    np0 = np.einsum(tmp0, [0,1,2,3], pauli, [0,2], [1,3], optimize=True)/2
                    break
            else: #no break path
                assert False, 'not a Pauli operator'
        assert (np0.shape==(1,1)) and (abs(abs(np0.item())-1)<1e-7)
        tmp0 = round(np.angle(np0.item())*2/np.pi) % 4
        ret = PauliOperator(np.array([tmp0>>1, tmp0&1] + list(bitXZ), dtype=np.uint8))
        return ret

    @property
    def op_list(self):
        ret = [(self.sign,[(y,x) for x,y in enumerate(self.np_list)])]
        return ret

    @property
    def full_matrix(self):
        # use this for unittest only, for that this is not efficient
        int_to_map = {
            0: _one_pauli_str_to_np['I'],
            1: _one_pauli_str_to_np['Z'],
            2: _one_pauli_str_to_np['X'],
            3: _one_pauli_str_to_np['Y'],
        }
        N0 = len(self.str_)
        tmp0 = (self.F2[2:(2+N0)]*2 + self.F2[(2+N0):]).tolist()
        np0 = int_to_map[tmp0[0]]
        for x in tmp0[1:]:
            np0 = np.kron(np0, int_to_map[x])
        ret = self.sign*np0
        return ret


def get_pauli_subset_equivalent(subset:tuple[int], num_qubit:int, use_tqdm:bool=False):
    first_element = tuple(sorted(subset))
    equivalent_set = {first_element}
    first_element_GF4 = pauli_index_to_F2(first_element, num_qubit, with_sign=False)
    tmp0 = [((1<<(2*x))-1) for x in range(1,num_qubit+1)]
    tmp1 = [(1<<(2*x-1)) for x in range(1,num_qubit+1)]
    order_a_b_list = [y for x in zip(tmp0,tmp1) for y in x]
    tmp0 = itertools.product(*[range(x) for x in order_a_b_list])
    total = functools.reduce(lambda x,y:x*y, order_a_b_list, 1)
    if use_tqdm:
        tmp1 = tqdm(tmp0, total=total, desc='found=0')
    else:
        tmp1 = contextlib.nullcontext(tmp0)
    with tmp1 as pbar:
        for ind0,ind_sp2n in enumerate(pbar):
            tmp0 = numqi.group.spf2.from_int_tuple(ind_sp2n)
            tmp1 = np.sort(pauli_F2_to_index((first_element_GF4 @ tmp0) % 2, with_sign=False))
            equivalent_set.add(tuple(tmp1.tolist()))
            if use_tqdm and ((ind0%1000==0) or (ind0==total-1)):
                pbar.set_description(f'found={len(equivalent_set)}')
    return equivalent_set


# TODO unittest
def get_pauli_subset_stabilizer(subset:tuple[int], num_qubit:int, print_every_N:int=10000):
    first_element = tuple(sorted(subset))
    first_element_GF4 = pauli_index_to_F2(first_element, num_qubit, with_sign=False)
    tmp0 = [((1<<(2*x))-1) for x in range(1,num_qubit+1)]
    tmp1 = [(1<<(2*x-1)) for x in range(1,num_qubit+1)]
    order_a_b_list = [y for x in zip(tmp0,tmp1) for y in x]
    t0 = time.time()
    last_print_N0 = 0
    stabilizer_list = []
    for ind_sp2n in itertools.product(*[range(x) for x in order_a_b_list]):
        tmp0 = numqi.group.spf2.from_int_tuple(ind_sp2n)
        tmp1 = np.sort(pauli_F2_to_index((first_element_GF4 @ tmp0) % 2, with_sign=False))
        if tuple(tmp1.tolist())==first_element:
            stabilizer_list.append(ind_sp2n)
        N0 = len(stabilizer_list)
        if (print_every_N>0) and (N0%print_every_N==0) and (N0>last_print_N0):
            print(f'[{time.time()-t0:.1f}s] {N0}')
            last_print_N0 = N0
    return stabilizer_list


# TODO unittest
def get_pauli_all_subset_equivalent(num_qubit:int, order_start:int=1, order_end:int|None=None, use_tqdm:bool=False):
    num_pauli = 4**num_qubit
    assert num_qubit in {2,3} #>=3 is too slow
    if order_end is None:
        order_end = num_pauli
    all_equivalent_set = dict()
    for subset_order in range(max(1,order_start), min(order_end, num_pauli)):
        subset_list = list(itertools.combinations(list(range(num_pauli)), subset_order))
        z0 = []
        for subset_i in (tqdm(subset_list) if use_tqdm else subset_list):
            for x in z0:
                if subset_i in x:
                    break
            else:
                z0.append(get_pauli_subset_equivalent(subset_i, num_qubit))
        tmp0 = collections.Counter([len(x) for x in z0])
        tmp0 = sorted(tmp0.items(),key=lambda x:x[0])
        tmp0 = '+'.join([f'{v}x{k}' for k,v in tmp0])
        print(f'#[len={subset_order}]={len(z0)}, {tmp0}')
        all_equivalent_set[subset_order] = z0
    return all_equivalent_set

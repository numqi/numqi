import functools
import itertools
import numpy as np
import scipy.sparse

from ._internal import pauli

hf_kron = lambda x: functools.reduce(np.kron, x)

_one_pauli_str_to_np = dict(zip('IXYZ', [pauli.s0, pauli.sx, pauli.sy, pauli.sz]))
# do not use this "pauli" below for that this name might be redefined in functions below

# TODO to be replaced by PauliOperator
def pauli_str_to_matrix(pauli_str, return_orth=False):
    #'XX YZ IZ'
    pauli_str = sorted(set(pauli_str.split()))
    num_qubit = len(pauli_str[0])
    assert all(len(x)==num_qubit for x in pauli_str)
    matrix_space = np.stack([hf_kron([_one_pauli_str_to_np[y] for y in x]) for x in pauli_str])
    if return_orth:
        pauli_str_orth = sorted(set(get_pauli_group(num_qubit, kind='str')) - set(pauli_str))
        matrix_space_orth = np.stack([hf_kron([_one_pauli_str_to_np[y] for y in x]) for x in pauli_str_orth])
        ret = matrix_space,matrix_space_orth
    else:
        ret = matrix_space
    return ret


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


def pauli_index_str_convert(terms, num_qubit=None):
    if isinstance(terms[0], str):
        num_qubit = len(terms[0])
        tmp0 = get_pauli_group(num_qubit, kind='str_to_index')
        ret = [tmp0[x] for x in terms]
    else:
        assert num_qubit is not None
        tmp0 = get_pauli_group(num_qubit, kind='str')
        ret = [tmp0[x] for x in terms]
    return ret


def pauli_F2_to_str(np0):
    r'''convert Pauli operator in the F2 representation to string representation

    Parameters:
        np0 (np.ndarray): shape (`2n+2`,), dtype=np.uint8

    Returns:
        pauli_str (str): Pauli string, e.g. 'XIZYX'
        sign (complex): coefficient of Pauli string, {1, i, -1, -i}
    '''
    assert (np0.dtype.type==np.uint8)
    assert (np0.ndim==1) and (np0.shape[0]%2==0) and (np0.shape[0]>=2)
    num_qubit = (np0.shape[0]-2)//2
    bitX = np0[2:(2+num_qubit)]
    bitZ = np0[(2+num_qubit):]
    tmp1 = (2*np0[0] + np0[1] + 3*np.dot(bitX,bitZ))%4 #XZ=-iY
    sign = 1j**tmp1
    tmp0 = {(0,0):'I', (1,0):'X', (0,1):'Z', (1,1):'Y'}
    pauli_str = ''.join(tmp0[(int(x),int(y))] for x,y in zip(bitX,bitZ))
    return pauli_str, sign


def pauli_str_to_F2(pauli_str:str, sign=1):
    r'''convert Pauli string to Pauli operator in the F2 representation

    Parameters:
        pauli_str (str): Pauli string, e.g. 'XIZYX'
        sign (complex): sign of Pauli string, {1, i, -1, -i}

    Returns:
        np0 (np.ndarray): shape (`2n+2`,), dtype=np.uint8
    '''
    pauli_str = pauli_str.upper()
    assert set(pauli_str) <= set('IXYZ')
    bitX = np.array([(1 if ((x=='X') or (x=='Y')) else 0) for x in pauli_str], dtype=np.uint8)
    bitZ = np.array([(1 if ((x=='Z') or (x=='Y')) else 0) for x in pauli_str], dtype=np.uint8)

    sign_ri = int(sign.real), int(sign.imag)
    tmp0 = {(1,0):0, (0,1):1, (-1,0):2, (0,-1):3}
    assert sign_ri in tmp0, 'sign should be one of {1, i, -1, -i}'
    tmp1 = (np.dot(bitX,bitZ) + tmp0[sign_ri]) % 4
    ret = np.concatenate([np.array([tmp1//2, tmp1%2], dtype=np.uint8), bitX, bitZ])
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

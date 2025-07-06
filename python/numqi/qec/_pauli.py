import sympy
import functools
import itertools
import numpy as np
import scipy.sparse
import scipy.special
import torch
from tqdm.auto import tqdm

import numqi.gate
import numqi._internal

from .gf4 import str_to_gf4, gf4_to_str

# TODO sparse for performance
def hf_pauli(x:str):
    r'''convert Pauli string to matrix, should NOT be used in performance-critical code

    Parameters:
        x (str): Pauli string, e.g. 'IXYZ'

    Returns:
        ret (np.ndarray): shape=(2**n,2**n), Pauli matrix
    '''
    assert (len(x)>=1) and (set(x) <= {'I','X','Y','Z'})
    tmp0 = {'I':np.eye(2), 'X':numqi.gate.X, 'Y':numqi.gate.Y, 'Z':numqi.gate.Z}
    ret = tmp0[x[0]]
    for x0 in x[1:]:
        ret = np.kron(ret, tmp0[x0])
    return ret

def _get_pauli_with_weight_sparse_hf0(num_qubit, weight, tag_neighbor=False):
    assert (num_qubit>=1) and (weight>=0) and (weight<=num_qubit)
    if weight==0:
        ret = ['I'*num_qubit, scipy.sparse.eye(2**num_qubit, dtype=np.complex128, format='csr')]
    else:
        pauli = [
            scipy.sparse.eye(2, dtype=np.complex128, format='csr'),
            scipy.sparse.csr_array(([1,1], ([0,1], [1,0])), shape=(2,2), dtype=np.complex128),
            scipy.sparse.csr_array(([-1j,1j], ([0,1], [1,0])), shape=(2,2), dtype=np.complex128),
            scipy.sparse.csr_array(([1,-1], ([0,1], [0,1])), shape=(2,2), dtype=np.complex128),
        ]
        hf0 = lambda x,y: scipy.sparse.kron(x, y, format='csr')
        hf_kron = lambda *x: functools.reduce(hf0, x[1:], x[0])
        error_list = []
        str_list = []
        if tag_neighbor:
            index_qubit_list = (tuple(sorted([(x+y)%num_qubit for y in range(weight)])) for x in range(num_qubit))
        else:
            index_qubit_list = itertools.combinations(range(num_qubit), r=weight)
        for index_qubit in index_qubit_list:
            for index_gate in itertools.product([0,1,2], repeat=weight):
                tmp1 = [pauli[0] for _ in range(num_qubit)]
                tmp2 = ['I']*num_qubit
                for x,y in zip(index_qubit,index_gate):
                    tmp1[x] = pauli[y+1]
                    tmp2[x] = 'XYZ'[y]
                error_list.append(hf_kron(*tmp1))
                str_list.append(''.join(tmp2))
        error_list = scipy.sparse.vstack(error_list, format='csr')
        ret = str_list, error_list
    return ret


def get_pauli_with_weight_sparse(num_qubit:int, weight:int, tag_neighbor:bool=False):
    r'''get pauli operator with given ewight

    Parameters:
        num_qubit (int): number of qubit
        weight (int): weight of pauli operator
        tag_neighbor (bool): if True, the qubit is connected in a ring

    Returns:
        str_list (list[str]): list of string of pauli operator
        error_list (scipy.sparse.csr_array): pauli operator of shape `(N0*2**num_qubit, 2**num_qubit)`, where `N0` is the number of pauli operator with given weight
    '''
    key = f'pauli_csr01_n{num_qubit}_wt{weight}_neighbor{int(tag_neighbor)}'
    if numqi._internal.is_key_in_disk(key):
        np_list = numqi._internal.load_from_disk(key)
        str_list = numqi._internal.load_from_disk(key+'_str').astype(str).tolist()
    else:
        str_list,np_list = _get_pauli_with_weight_sparse_hf0(int(num_qubit), int(weight), bool(tag_neighbor))
        numqi._internal.save_to_disk(key, np_list)
        tmp0 = np.array(str_list, dtype=f'|S{num_qubit}')
        numqi._internal.save_to_disk(key+'_str', tmp0, overwrite=True)
    return str_list,np_list


def pauli_csr_to_kind(x0:scipy.sparse.csr_array, kind:str):
    r'''convert scipy.sparse.csr_array to desired kind

    Parameters:
        x0 (scipy.sparse.csr_array): input matrix
        kind (str): kind of output, 'numpy', 'torch', 'scipy-csr0', 'scipy-csr01', 'torch-csr0', 'torch-csr01'
                numpy: np.ndarray full matrix
                torch: torch.Tensor full matrix
                scipy-csr0: scipy.sparse.csr_matrix of shape `(N0, 4**num_qubit)`
                scipy-csr01: scipy.sparse.csr_matrix of shape `(N0*2**num_qubit, 2**num_qubit)`
                torch-csr0: torch.sparse_csr_tensor of shape `(N0, 4**num_qubit)`
                torch-csr01: torch.sparse_csr_tensor of shape `(N0*2**num_qubit, 2**num_qubit)`

    Returns:
        ret (np.ndarray|torch.Tensor|scipy.sparse.csr_matrix): output matrix
    '''
    assert kind in {'numpy', 'torch', 'scipy-csr0', 'scipy-csr01', 'torch-csr0', 'torch-csr01'}
    assert isinstance(x0, scipy.sparse.csr_array)
    num_qubit = round(np.log2(x0.shape[1]))
    assert 2**num_qubit == x0.shape[1]
    if kind=='numpy':
        ret = x0.todense().reshape(-1, 2**num_qubit, 2**num_qubit)
    elif kind=='torch':
        ret = torch.tensor(x0.todense().reshape(-1, 2**num_qubit, 2**num_qubit))
    elif kind=='scipy-csr0':
        ret = x0.reshape(-1, 4**num_qubit).tocsr()
    elif kind=='scipy-csr01':
        ret = x0
    elif kind in {'torch-csr0', 'torch-csr01'}:
        if kind=='torch-csr0':
            x0 = x0.reshape(-1, 4**num_qubit).tocsr()
        tmp0 = torch.tensor(x0.indptr, dtype=torch.int64)
        tmp1 = torch.tensor(x0.indices, dtype=torch.int64)
        tmp2 = torch.tensor(x0.data, dtype=torch.complex128)
        ret = torch.sparse_csr_tensor(tmp0, tmp1, tmp2, dtype=torch.complex128)
    return ret

def make_pauli_error_list_sparse(num_qubit:int, distance:int, kind:str='torch-csr01', tag_neighbor:bool=False):
    '''make Pauli error operator sorted by weight

    Parameters:
        num_qubit (int): number of qubits
        distance (int): distance of QECC
        kind (str): kind of output, 'numpy', 'torch', 'scipy-csr0', 'scipy-csr01', 'torch-csr0', 'torch-csr01'
        tag_neighbor (bool): if True, consider only neighbor errors

    Returns:
        error_str_list (list[str]): list of Pauli error string
        error_list (np.ndarray|torch.Tensor|scipy.sparse.csr_matrix): list of Pauli error operator. When `kind=`
                numpy: np.ndarray full matrix
                torch: torch.Tensor full matrix
                scipy-csr0: scipy.sparse.csr_matrix of shape `(N0*2**num_qubit, 2**num_qubit)`
                scipy-csr01: scipy.sparse.csr_matrix of shape `(N1, 4**num_qubit)`
                torch-csr0: torch.sparse_csr_tensor of shape `(N0*2**num_qubit, 2**num_qubit)`
                torch-csr01: torch.sparse_csr_tensor of shape `(N1, 4**num_qubit)`
    '''
    # kind: numpy torch scipy-csr0 scipy-csr01 torch-csr0 torch-csr01
    assert kind in {'numpy', 'torch', 'scipy-csr0', 'scipy-csr01', 'torch-csr0', 'torch-csr01'}
    assert (distance>1)
    error_list = []
    str_list = []
    for weight in range(1, distance):
        tmp0,tmp1 = get_pauli_with_weight_sparse(num_qubit, weight, tag_neighbor)
        str_list.extend(tmp0)
        error_list.append(tmp1)
    x0 = scipy.sparse.vstack(error_list, format='csr')
    ret = str_list, pauli_csr_to_kind(x0, kind)
    return ret

# deprecated
def make_error_list(num_qubit, distance, op_list=None, tag_full=False):
    assert distance>1
    if op_list is None:
        op_list = [numqi.gate.X, numqi.gate.Y, numqi.gate.Z]
    ret = []
    for weight in range(1, distance):
        for index_qubit in itertools.combinations(range(num_qubit), r=weight):
            for gate in itertools.product(op_list, repeat=weight):
                ret.append([([x],y) for x,y in zip(index_qubit,gate)])
    if tag_full:
        hf_kron = lambda *x: functools.reduce(np.kron, x[1:], x[0])
        tmp0 = ret
        ret = []
        for op0 in tmp0:
            tmp1 = [numqi.gate.pauli.s0 for _ in range(num_qubit)]
            for y,z in op0:
                tmp1[y[0]] = z
            ret.append(hf_kron(*tmp1))
    return ret


def hf_split_element(np0, num_of_each):
    # split np0 (len(np0)=N) into len(num_of_each) sets, elements in each set is unordered
    assert len(num_of_each) > 0
    assert sum(num_of_each) <= len(np0)
    tmp0 = list(range(np0.shape[0]))
    tmp1 = np.ones(np0.shape[0], dtype=np.bool_)
    if num_of_each[0]==0:
        if len(num_of_each)==1:
            yield (),
        else:
            tmp0 = (),
            for x in hf_split_element(np0, num_of_each[1:]):
                yield tmp0+x
    else:
        for ind0 in itertools.combinations(tmp0, num_of_each[0]):
            ind0 = list(ind0)
            if len(num_of_each)==1:
                yield tuple(np0[ind0].tolist()),
            else:
                tmp1[ind0] = False
                tmp2 = np0[tmp1]
                tmp1[ind0] = True
                tmp3 = tuple(np0[ind0].tolist()),
                for x in hf_split_element(tmp2, num_of_each[1:]):
                    yield tmp3 + x
# z0 = hf_split_element(np.arange(num_qubit), [0,1,0])

def make_asymmetric_error_set(num_qubit, distance, weight_z=1):
    # 0<=nx+ny+nz<=num_qubit
    # 0<=nx,ny,nz
    # nx+ny+cz*nz<distance
    assert weight_z>0
    ret = []
    for nxy in range(min(num_qubit,distance)):
        tmp0 = int(np.ceil((distance-nxy)/weight_z))
        for nz in range(min(num_qubit-nxy+1, tmp0)):
            if (nxy==0) and (nz==0):
                continue
            for nx in range(nxy+1):
                ny = nxy - nx
                for indx,indy,indz in hf_split_element(np.arange(num_qubit), [nx,ny,nz]):
                    tmp0 = [([x],numqi.gate.X) for x in indx]
                    tmp1 = [([x],numqi.gate.Y) for x in indy]
                    tmp2 = [([x],numqi.gate.Z) for x in indz]
                    ret.append(tmp0+tmp1+tmp2)
    return ret


def _get_weight_enumerator_circuit(code, index, use_tqdm, tagB):
    num_qubit = numqi.utils.hf_num_state_to_num_qubit(code.shape[1], kind='exact')
    num_logical_dim = code.shape[0]
    num_logical_qubit = numqi.utils.hf_num_state_to_num_qubit(num_logical_dim, kind='ceil')
    if 2**num_logical_qubit > num_logical_dim:
        code = np.pad(code, [(0,2**num_logical_qubit-num_logical_dim),(0,0)], mode='constant', constant_values=0)
    code_conj = code.conj()
    op_list = [numqi.gate.X, numqi.gate.Y, numqi.gate.Z]
    retA = np.zeros(len(index), dtype=np.float64)
    retB = np.zeros(len(index), dtype=np.float64)
    for ind0,weight in enumerate(index):
        if weight==0:
            retA[ind0] = 1
            if tagB:
                retB[ind0] = 1
        else:
            tmp0 = itertools.combinations(range(num_qubit), r=weight)
            index_gate_generator = ((x,y) for x in tmp0 for y in itertools.product(op_list, repeat=weight))
            if use_tqdm:
                total = len(op_list)**weight * int(round(scipy.special.binom(num_qubit,weight)))
                index_gate_generator = tqdm(index_gate_generator, desc=f'weight={weight}', total=total)
            for index_qubit,gate in index_gate_generator:
                gate_list = [([x+num_logical_qubit],y) for x,y in zip(index_qubit,gate)]
                q0 = code.reshape(-1)
                for ind_op,op_i in gate_list:
                    q0 = numqi.sim.state.apply_gate(q0, op_i, ind_op)
                tmp0 = code_conj.reshape(-1, 2**num_qubit) @ q0.reshape(-1, 2**num_qubit).T
                retA[ind0] += abs(np.trace(tmp0).item())**2/num_logical_dim**2
                if tagB:
                    retB[ind0] += np.vdot(tmp0.reshape(-1), tmp0.reshape(-1)).real.item()/num_logical_dim
    return retA,retB


def _get_weight_enumerator_sparse(code, wt_to_pauli_dict, index, tagB):
    dim,num_state = code.shape
    num_qubit = numqi.utils.hf_num_state_to_num_qubit(num_state)
    if wt_to_pauli_dict is None:
        wt_to_pauli_dict = {x:get_pauli_with_weight_sparse(num_qubit,x)[1] for x in (set(index)-{0})}
    retA = []
    retB = []
    for x in index:
        if x==0:
            retA.append(1)
            if tagB:
                retB.append(1)
        else:
            pauli = wt_to_pauli_dict[x]
            tmp0 = code.conj() @ (pauli @ code.T).reshape(-1,num_state,dim)
            # tmp0 = tmp0.transpose(1,0,2,3).reshape(-1,dim,dim)
            tmp1 = np.diagonal(tmp0, axis1=1, axis2=2).real.sum(axis=1)
            retA.append((np.dot(tmp1, tmp1)/(dim*dim)))
            if tagB:
                tmp1 = tmp0.reshape(-1)
                retB.append(np.vdot(tmp1, tmp1).real / dim)
    return retA,retB

def get_weight_enumerator(code, wt_to_pauli_dict:dict|None=None, index:int|list[int]|None=None, tagB:bool=True, use_circuit:bool=False, use_tqdm:bool=False):
    r'''get Shor-Laflamme quantum weight enumerator

    [arxiv-link](https://arxiv.org/abs/quant-ph/9610040)

    Parameters:
        code (np.ndarray): shape=(dim, 2**num_qubit), code space
        wt_to_pauli_dict (dict,None): dict of weight to pauli operator, {weight: pauli_operator}, generated from
                `numqi.qec.get_pauli_with_weight_sparse`. For performance, it is recommended to pre-calculate this dict
        index (int,list[int],None): weight of pauli operator, if None, return all weight
        tagB (bool): if True, return both A and B
        use_circuit (bool): if True, use circuit to calculate
        use_tqdm (bool): if True, plot progress bar, only valid when `use_circuit=True`

    Returns:
        retA (np.ndarray,float): Shor-Laflamme quantum weight enumerator
        retB (np.ndarray,float): Shor-Laflamme dual quantum weight enumerator
    '''
    num_qubit = numqi.utils.hf_num_state_to_num_qubit(code.shape[1])
    if index is None:
        index = tuple(range(num_qubit+1))
    isone = not hasattr(index, '__len__')
    if isone:
        index = int(index),
    if use_circuit:
        retA,retB = _get_weight_enumerator_circuit(code, index, use_tqdm, tagB)
    else:
        retA,retB = _get_weight_enumerator_sparse(code, wt_to_pauli_dict, index, tagB)
    if tagB:
        ret = np.array(retA), np.array(retB)
    else:
        ret = np.array(retA)
    if isone:
        ret = (ret[0].item(), ret[1].item()) if tagB else ret.item()
    return ret


@functools.lru_cache
def _get_weight_enumerator_transform_matrix_hf0(n:int):
    binom = functools.lru_cache(lambda n,k: (sympy.binomial(n,k) if k<=n else 0))
    pow3 =  functools.lru_cache(lambda n: 3**n)

    matM = sympy.Matrix([[sum(binom(n-j, i-l) * binom(j, l) * ((-1)**l) * pow3(i-l)
                    for l in range(max(0,i+j-n), min(i,j)+1))/2**n for j in range(n+1)] for i in range(n+1)]) #M eq(7)
    matMprime = sympy.Matrix([[1 if (i+j==n) else 0 for j in range(n+1)] for i in range(n+1)]) #M1 eq(8)
    matMtilde = sympy.Matrix([[(-1)**(n+i) if (i==j) else 0 for j in range(n+1)] for i in range(n+1)]) #M2 eq(9)

    matTprime = sympy.Matrix([[(2**(n-i))*binom(n-j,n-i)/binom(n,i) for j in range(n+1)] for i in range(n+1)]) # T1 eq(29)
    matTprime_inv = sympy.Matrix([[(binom(n,j)*binom(n-j,n-i)*(-1)**(i+j))/2**(n-j) for j in range(n+1)] for i in range(n+1)]) # iT1 eq(30)

    matTtilde = sympy.Matrix([[sum(binom(n-j,i-l)*binom(j,l)*((-1)**(j-l))*pow3(i-l)
        for l in range(max(0,i+j-n),min(i,j)+1))/2**n for j in range(n+1)] for i in range(n+1)]) # T2 eq(39)
    matTtilde_inv = sympy.Matrix([[sum(binom(n-j,i-l)*binom(j,l)*((-1)**(i-l))*pow3(i-l)
        for l in range(max(0,i+j-n),min(i,j)+1))/2**n for j in range(n+1)] for i in range(n+1)]) # iT2 eq(73)

    matTptilde = sympy.Matrix([[sum(binom(n-j,i-l)*binom(j,l)*((-1)**(j-l))
        for l in range(max(0,i+j-n),min(i,j)+1))*binom(n,j)/2**n for j in range(n+1)] for i in range(n+1)]) # T3 eq(A1)
    matTptilde_inv = sympy.Matrix([[sum(binom(n-j,i-l)*binom(j,l)*((-1)**(i-l))
        for l in range(max(0,i+j-n),min(i,j)+1))/binom(n,i) for j in range(n+1)] for i in range(n+1)]) # iT3 eq(A2)
    ret = dict(M=matM, M1=matMprime, M2=matMtilde, T1=matTprime, iT1=matTprime_inv,
                T2=matTtilde, iT2=matTtilde_inv, T3=matTptilde, iT3=matTptilde_inv)
    return ret

def get_weight_enumerator_transform_matrix(n:int, kind:str='sympy'):
    r'''get the transform matrix for various quantum weight enumerator

    reference: [arxiv-link](https://arxiv.org/abs/2408.16914) Experimental measurement and a physical interpretation of quantum shadow enumerators.
    also see package `qsalto`

    Parameters:
        n (int): number of qubits
        kind (str): kind of output, 'sympy', 'numpy'. If 'sympy', return sympy.Matrix, if 'numpy', return numpy.ndarray

    Returns:
        ret (dict): dictionary of transform matrix, keys are 'M', 'M1', 'M2', 'T1', 'iT1', 'T2', 'iT2', 'T3', 'iT3'
    '''
    assert kind in ['sympy','numpy']
    ret = _get_weight_enumerator_transform_matrix_hf0(int(n))
    if kind=='numpy':
        ret = {k: np.array([[float(y) for y in x] for x in v.tolist()], dtype=np.float64) for k,v in ret.items()}
    return ret


@functools.lru_cache
def _get_knill_laflamme_matrix_indexing_over_vector_hf0(num_qubit:int, distance:int):
    str_d_list = []
    for weight in range(distance): #include weight=0
        for index_qubit in itertools.combinations(range(num_qubit), r=weight):
            for index_gate in itertools.product([0,1,2], repeat=weight):
                tmp2 = ['I']*num_qubit
                for x,y in zip(index_qubit,index_gate):
                    tmp2[x] = 'XYZ'[y]
                str_d_list.append(''.join(tmp2))

    N0 = sum([scipy.special.comb(num_qubit, x, exact=True)*(3**x) for x in range(distance//2+1)])
    str_d2_list = str_d_list[:N0]

    tmp0 = str_to_gf4(str_d2_list)
    tmp1 = gf4_to_str(((tmp0.reshape(-1, 1, 2*num_qubit) + tmp0) % 2).reshape(-1, 2*num_qubit))
    tmp2 = {x:i for i,x in enumerate(str_d_list)}
    index = np.array([tmp2[x] for x in tmp1], dtype=np.int64)
    index.setflags(write=False)
    return index


def get_knill_laflamme_matrix_indexing_over_vector(num_qubit:int, distance:int):
    assert (distance%2)==1
    ret = _get_knill_laflamme_matrix_indexing_over_vector_hf0(int(num_qubit), int(distance))
    return ret


def get_qweA_kernel(basis0, basis1, tag_real_coeff, zero_eps=1e-10):
    assert (basis0.ndim==2) and (basis0.shape==basis1.shape)
    num_qubit = numqi.utils.hf_num_state_to_num_qubit(basis0.shape[1], kind='exact')
    wt_to_pauli_dict = {x:get_pauli_with_weight_sparse(num_qubit, weight=x)[1] for x in range(num_qubit+1)}
    N0 = len(basis0)
    mask = np.triu(np.ones((N0,N0), dtype=np.int64) * 2, 1) + np.eye(N0, dtype=np.int64)
    qweA_kernel_dict = dict()
    for wt in range(1, num_qubit+1):
        pauli = wt_to_pauli_dict[wt]
        z0 = basis0.conj() @ (pauli @ basis0.T).reshape(-1,2**7,N0) + basis1.conj() @ (pauli @ basis1.T).reshape(-1,2**7,N0)
        tmp0 = np.einsum(z0, [0,1,2], z0, [0,3,4], [1,3,2,4], optimize=True) / 4
        tmp0 = (tmp0 + tmp0.transpose(1,0,2,3)) / 2
        tmp0 = (tmp0 + tmp0.transpose(0,1,3,2)) / 2
        tmp0 = tmp0 * mask.reshape(N0,N0,1,1) * mask.reshape(1,1,N0,N0)
        if tag_real_coeff:
            tmp1 = dict()
            for i0 in np.stack(np.nonzero(np.abs(tmp0)>zero_eps), axis=1).tolist():
                i1 = tuple(sorted(i0))
                tmp1[i1] = tmp0[tuple(i0)] + tmp1.get(i1, 0)
            tmp1 = list(tmp1.items())
            key = np.array([x[0] for x in tmp1]).T
            value = np.array([x[1] for x in tmp1])
            tmp0 = np.zeros_like(tmp0)
            if len(key):
                tmp0[key[0], key[1], key[2], key[3]] = value
        if np.abs(tmp0.imag).max() < zero_eps:
            tmp0 = tmp0.real
        if (not np.iscomplexobj(tmp0)) and (np.abs(tmp0 - np.around(tmp0).astype(np.int64)).max() < zero_eps):
            tmp0 = np.around(tmp0).astype(np.int64)
        qweA_kernel_dict[wt] = tmp0
    # A2_kernel = qweA_kernel_dict[2]
    # hf0 = lambda x: '{' + f'{x[0]+1},{x[1]+1},{x[2]+1},{x[3]+1},{x[4]}' + '}'
    # x0 = [hf0(tuple(i)+(A2_kernel[tuple(i)],)) for i in np.stack(np.nonzero(A2_kernel), axis=1)]
    # ','.join(x0)
    return qweA_kernel_dict

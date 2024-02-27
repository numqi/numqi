import itertools
import functools
import numpy as np
import opt_einsum

from numqi.utils import hf_num_state_to_num_qubit, hf_tuple_of_int
import numqi.random

@functools.lru_cache
def _reduce_shape_index_hf0(shape, index):
    ret = []
    for x0,x1 in itertools.groupby(zip(shape,index), key=lambda x: x[1]==None):
        x1 = list(x1)
        shape_i = [x[0] for x in x1]
        if x0: #index=None
            tmp0 = functools.reduce(lambda y0,y1: y0*y1, shape_i)
            ret.append((tmp0, slice(None)))
        else:
            index_i = [x[1] for x in x1]
            tmp0 = np.cumprod(np.array([1]+shape_i[::-1]))
            tmp1 = np.dot(tmp0[:-1], np.array(index_i[::-1]))
            ret.append((tmp0[-1], tmp1))
    reduced_shape = tuple(x[0] for x in ret)
    reduced_index = tuple(x[1] for x in ret)
    return reduced_shape, reduced_index


def reduce_shape_index(shape:tuple[int], index:tuple[int|None]):
    r'''reduce the shape and index

    1. group index by None

    2. group index by integer

    Parameters:
        shape (tuple[int]): the shape of the tensor
        index (tuple[int|None]): the index list

    Returns:
        reduced_shape (tuple[int]): the reduced shape
        reduced_index (tuple[int|None]): the reduced index list
    '''
    shape = tuple(int(x) for x in shape)
    assert (len(shape)>0) and all(x>1 for x in shape)
    N0 = len(shape)
    index = tuple((None if x is None else int(x)) for x in index)
    assert (len(index)==N0) and all(((y is None) or (0<=y<=x)) for x,y in zip(shape,index))
    ret = _reduce_shape_index_hf0(shape, index)
    return ret


def new_base(num_qubit:int, dtype=np.complex128):
    r'''return the base state of the qubit quantum system

    Parameters:
        num_qubit (int): the number of qubits
        dtype (dtype): the data type of the base state

    Returns:
        ret (np.ndarray): the base state
    '''
    ret = np.zeros(2**num_qubit, dtype=dtype)
    ret[0] = 1
    return ret


def apply_gate(q0:np.ndarray, op:np.ndarray, index:int|tuple[int]):
    r'''apply the gate to the quantum vector

    Parameters:
        q0 (np.ndarray): the quantum vector, `ndim=1`
        op (np.ndarray): the gate, `ndim=2`
        index (int,tuple[int]): the index of the qubits to apply the gate, count from left to right |0123>

    Returns:
        ret (np.ndarray): the quantum vector after applying the gate
    '''
    assert q0.ndim==1
    index = hf_tuple_of_int(index)
    num_state = len(q0)
    num_qubit = hf_num_state_to_num_qubit(num_state)
    N0 = len(index)
    assert all(isinstance(x,int) and (0<=x) and (x<num_qubit) for x in index)
    assert len(index)==len(set(index))
    assert (op.ndim==2) and (op.shape[0]==op.shape[1]) and (op.shape[0]==2**N0)
    tmp0 = q0.reshape([2 for _ in range(num_qubit)])
    tmp1 = list(range(num_qubit))
    tmp2 = op.reshape([2 for _ in range(2*N0)])
    tmp3 = tuple(range(num_qubit,num_qubit+N0))
    tmp4 = {x:y for x,y in zip(index,tmp3)}
    tmp5 = [(tmp4[x] if x in tmp4 else x) for x in range(num_qubit)]
    ret = opt_einsum.contract(tmp0, tmp1, tmp2, tmp3+tuple(index), tmp5).reshape(-1)
    return ret

def apply_gate_grad(q0_conj:np.ndarray, q0_grad:np.ndarray, op:np.ndarray, index:int|tuple[int], tag_op_grad:bool=True):
    r'''gradient back propagation of apply_gate

    Parameters:
        q0_conj (np.ndarray): the conjugate of the quantum vector, `ndim=1`
        q0_grad (np.ndarray): the gradient of the quantum vector, `ndim=1`
        op (np.ndarray): the gate, `ndim=2`
        index (int,tuple[int]): the index of the qubits to apply the gate
        tag_op_grad (bool): whether to calculate the gradient of the gate

    Returns:
        q0_conj (np.ndarray): the conjugate of the quantum vector before applying the gate
        q0_grad (np.ndarray): the gradient of the quantum vector before applying the gate
        op_grad (np.ndarray,None): the gradient of the gate, None if `tag_op_grad=False`
    '''
    q0_conj = apply_gate(q0_conj, op.T, index)
    if tag_op_grad:
        num_state = len(q0_conj)
        num_qubit = hf_num_state_to_num_qubit(num_state)
        tmp0 = q0_grad.reshape([2 for _ in range(num_qubit)])
        tmp1 = list(range(num_qubit))
        tmp2 = q0_conj.reshape([2 for _ in range(num_qubit)])
        tmp3 = list(range(num_qubit))
        for x,y in enumerate(index):
            tmp3[y] = num_qubit + x
        tmp4 = list(index) + list(range(num_qubit,num_qubit+len(index)))
        op_grad = opt_einsum.contract(tmp0, tmp1, tmp2, tmp3, tmp4).reshape(op.shape)
    else:
        op_grad = None
    q0_grad = apply_gate(q0_grad, op.T.conj(), index)
    return q0_conj, q0_grad, op_grad


def _control_n_index(num_qubit, ind_control_set, ind_target):
    tmp0 = [x for x in range(num_qubit) if x not in ind_control_set]
    index_map = {y:x for x,y in enumerate(tmp0)}
    ind_target_new = [index_map[x] for x in ind_target]
    index_list = [None]*num_qubit
    for x in ind_control_set:
        index_list[x] = 1
    shape0,index_tuple0 = reduce_shape_index((2,)*num_qubit, tuple(index_list))
    return shape0, index_tuple0, ind_target_new


def apply_control_n_gate(q0:np.ndarray, op:np.ndarray, ind_control_set:int|set[int], ind_target:int|tuple[int]):
    r'''apply the n-controlled gate to the quantum vector

    Parameters:
        q0 (np.ndarray): the quantum vector, `ndim=1`
        op (np.ndarray): the gate, `ndim=2`
        ind_control_set (int,set[int]): the index of the control qubits
        ind_target (int,tuple[int]): the index of the target qubits

    Returns:
        ret (np.ndarray): the quantum vector after applying the gate
    '''
    if not hasattr(ind_control_set, '__len__'):
        ind_control_set = {int(ind_control_set)}
    else:
        tmp0 = set(hf_tuple_of_int(ind_control_set))
        assert len(tmp0)==len(ind_control_set)
        ind_control_set = tmp0
    ind_target = hf_tuple_of_int(ind_target)
    num_qubit = hf_num_state_to_num_qubit(q0.size)
    assert len(ind_target)==len(set(ind_target))
    assert all((x not in ind_control_set) for x in ind_target)
    shape0, index_tuple0, ind_target_new = _control_n_index(num_qubit, ind_control_set, ind_target)
    ret = q0.copy()
    tmp0 = q0.reshape(shape0)[index_tuple0]
    ret.reshape(shape0)[index_tuple0] = apply_gate(tmp0.reshape(-1), op, ind_target_new).reshape(tmp0.shape)
    return ret


def apply_control_n_gate_grad(q0_conj:np.ndarray, q0_grad:np.ndarray, op:np.ndarray,
            ind_control_set:int|set[int], ind_target:int|tuple[int], tag_op_grad:bool=True):
    r'''gradient back propagation of apply_control_n_gate

    Parameters:
        q0_conj (np.ndarray): the conjugate of the quantum vector, `ndim=1`
        q0_grad (np.ndarray): the gradient of the quantum vector, `ndim=1`
        op (np.ndarray): the gate, `ndim=2`
        ind_control_set (int,set[int]): the index of the control qubits
        ind_target (int,tuple[int]): the index of the target qubits
        tag_op_grad (bool): whether to calculate the gradient of the gate

    Returns:
        q0_conj (np.ndarray): the conjugate of the quantum vector before applying the gate
        q0_grad (np.ndarray): the gradient of the quantum vector before applying the gate
        op_grad (np.ndarray,None): the gradient of the gate, None if `tag_op_grad=False`
    '''
    if not hasattr(ind_control_set, '__len__'):
        ind_control_set = {int(ind_control_set)}
    else:
        tmp0 = set(hf_tuple_of_int(ind_control_set))
        assert len(tmp0)==len(ind_control_set)
        ind_control_set = tmp0
    ind_target = hf_tuple_of_int(ind_target)
    q0_conj = apply_control_n_gate(q0_conj, op.T, ind_control_set, ind_target)
    if tag_op_grad:
        num_qubit = hf_num_state_to_num_qubit(q0_conj.size)
        shape0, index_tuple0, ind_target_new = _control_n_index(num_qubit, ind_control_set, ind_target)
        num_qubit_new = num_qubit - len(ind_control_set)
        tmp0 = q0_grad.reshape(shape0)[index_tuple0].reshape([2]*num_qubit_new)
        tmp1 = list(range(num_qubit_new))
        tmp2 = q0_conj.reshape(shape0)[index_tuple0].reshape([2]*num_qubit_new)
        tmp3 = list(range(num_qubit_new))
        for x,y in enumerate(ind_target_new):
            tmp3[y] = num_qubit_new + x
        tmp4 = list(ind_target_new) + list(range(num_qubit_new,num_qubit_new+len(ind_target_new)))
        op_grad = opt_einsum.contract(tmp0, tmp1, tmp2, tmp3, tmp4).reshape(op.shape)
    else:
        op_grad = None
    q0_grad = apply_control_n_gate(q0_grad, op.T.conj(), ind_control_set, ind_target)
    return q0_conj, q0_grad, op_grad


# TODO torch.autograd.Function
def inner_product_psi0_O_psi1(psi0:np.ndarray, psi1:np.ndarray, op_list:list[list[tuple]]):
    r'''calculate the inner product of <psi0|O|psi1>

    Parameters:
        psi0 (np.ndarray): the quantum vector, `ndim=1`
        psi1 (np.ndarray): the quantum vector, `ndim=1`
        op_list (list[list[tuple]]): the operator list. The first level of list is the sum of the operator
                and the second level of list is matrix multiplication (from left to right). Each tuple is a gate and the index

    Returns:
        ret (np.ndarray): the inner product, `ndim=1` of the length equal to the number of operators (first level of list)
    '''
    ret = []
    for term_i in op_list:
        tmp_psi1 = psi1 #no need to copy
        for tmp0 in reversed(term_i):
            tmp_psi1 = apply_gate(tmp_psi1, tmp0[0], tuple(tmp0[1:]))
        ret.append(np.vdot(psi0, tmp_psi1))
    ret = np.array(ret)
    return ret


def reduce_to_probability(q0:np.ndarray, keep_index_set:set[int]):
    r'''reduce the quantum vector to the probability

    Parameters:
        q0 (np.ndarray): the quantum vector, `ndim=1`
        keep_index_set (set[int]): the index to keep

    Returns:
        ret (np.ndarray): the probability
    '''
    num_qubit = hf_num_state_to_num_qubit(q0.size)
    assert isinstance(keep_index_set,set) and all(0<=x for x in keep_index_set) and all(x<num_qubit for x in keep_index_set)
    tmp0 = (np.abs(q0)**2).reshape([2]*num_qubit)
    tmp1 = list(range(num_qubit))
    tmp2 = sorted(keep_index_set)
    ret = opt_einsum.contract(tmp0, tmp1, tmp2).reshape(-1)
    return ret


def inner_product(q0, q1):
    ret = np.vdot(q0, q1) #np.dot(q0.conj(), q1)
    return ret


def inner_product_grad(q0, q1, c_grad=1, tag_grad=(True,False)):
    if tag_grad[0]:
        q0_grad = q1 * np.conj(c_grad)
    else:
        q0_grad = None
    if tag_grad[1]:
        q1_grad = q0 * c_grad
    else:
        q1_grad = None
    return q0_grad, q1_grad


@functools.lru_cache
def _measure_quantum_vector_hf0(num_qubit, index):
    tmp0 = sorted(set(index))
    assert (len(tmp0)==len(index)) and all(x==y for x,y in zip(tmp0,index))
    shape = [2]*num_qubit
    kind = np.zeros(len(shape), dtype=np.int64)
    kind[list(index)] = 1
    kind = kind.tolist()
    hf0 = lambda x: x[0]
    hf1 = lambda x: int(np.prod([y for _,y in x]))
    z0 = [(k,hf1(x)) for k,x in itertools.groupby(zip(kind,shape), key=hf0)]
    shape = tuple(x[1] for x in z0)
    keep_dim = tuple(x for x,y in enumerate(z0) if y[0]==1)
    reduce_dim = tuple(x for x,y in enumerate(z0) if y[0]==0)
    return shape,keep_dim,reduce_dim


def measure_quantum_vector(q0:np.ndarray, index:int|tuple[int], seed:int|None|np.random.Generator=None):
    r'''measure the quantum vector

    Parameters:
        q0 (np.ndarray): the quantum vector, `ndim=1`
        index (int,tuple[int]): the index to measure, must be sorted (ascending)
        seed (int,None,np.random.Generator): the random seed

    Returns:
        bitstr (list[int]): the measurement result
        prob (np.ndarray): the probability of each result
        q1 (np.ndarray): the quantum vector after measurement
    '''
    np_rng = numqi.random.get_numpy_rng(seed)
    index = numqi.utils.hf_tuple_of_int(index)
    assert all(x==y for x,y in zip(sorted(index),index)), 'index must be sorted'
    num_qubit = numqi.utils.hf_num_state_to_num_qubit(q0.shape[0])
    shape,keep_dim,reduce_dim = _measure_quantum_vector_hf0(num_qubit, index)
    q1 = q0.reshape(shape)
    if len(reduce_dim)>0:
        prob = np.linalg.norm(q1, axis=reduce_dim).reshape(-1)**2
    else:
        prob = np.abs(q1.reshape(-1))**2
    ind1 = np_rng.choice(len(prob), p=prob)
    bitstr = [int(x) for x in bin(ind1)[2:].rjust(len(index),'0')]
    ind1a = np.unravel_index(ind1, tuple(shape[x] for x in keep_dim))
    ind2 = [slice(None)]*len(shape)
    for x,y in zip(keep_dim, ind1a):
        ind2[x] = y
    ind2 = tuple(ind2)
    q2 = np.zeros_like(q1)
    q2[ind2] = q1[ind2] / np.sqrt(prob[ind1])
    q2 = q2.reshape(-1)
    return bitstr,prob,q2

# TODO docs/script/draft_custom_gate.py include measure here

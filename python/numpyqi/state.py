import numpy as np
import opt_einsum

from .utils import hf_num_state_to_num_qubit, reduce_shape_index_list, hf_tuple_of_int


def new_base(num_qubit, dtype=np.complex128):
    ret = np.zeros(2**num_qubit, dtype=dtype)
    ret[0] = 1
    return ret


def apply_gate(q0, op, index):
    index = hf_tuple_of_int(index)
    num_state = len(q0)
    num_qubit = hf_num_state_to_num_qubit(num_state)
    N0 = len(index)
    assert num_state==(2**num_qubit)
    assert all(isinstance(x,int) and (0<=x) and (x<num_qubit) for x in index)
    assert len(index)==len(set(index))
    assert op.ndim==2 and op.shape[0]==op.shape[1]
    assert op.shape[0]==2**N0
    tmp0 = q0.reshape([2 for _ in range(num_qubit)])
    tmp1 = list(range(num_qubit))
    tmp2 = op.reshape([2 for _ in range(2*N0)])
    tmp3 = tuple(range(num_qubit,num_qubit+N0))
    tmp4 = {x:y for x,y in zip(index,tmp3)}
    tmp5 = [(tmp4[x] if x in tmp4 else x) for x in range(num_qubit)]
    ret = opt_einsum.contract(tmp0, tmp1, tmp2, tmp3+tuple(index), tmp5).reshape(-1)
    return ret

def apply_gate_grad(q0_conj, q0_grad, op, index, tag_op_grad=True):
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
    shape0,index_tuple0 = reduce_shape_index_list((2,)*num_qubit, tuple(index_list))
    return shape0, index_tuple0, ind_target_new

def apply_control_n_gate(q0, op, ind_control_set, ind_target):
    num_state = q0.size
    num_qubit = hf_num_state_to_num_qubit(num_state)
    assert len(ind_target)==len(set(ind_target))
    assert all((x not in ind_control_set) for x in ind_target)
    shape0, index_tuple0, ind_target_new = _control_n_index(num_qubit, ind_control_set, ind_target)
    ret = q0.copy()
    tmp0 = q0.reshape(shape0)[index_tuple0]
    ret.reshape(shape0)[index_tuple0] = apply_gate(tmp0.reshape(-1), op, ind_target_new).reshape(tmp0.shape)
    return ret


def apply_control_n_gate_grad(q0_conj, q0_grad, op, ind_control_set, ind_target, tag_op_grad=True):
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


def inner_product_psi0_O_psi1(psi0, psi1, operator_list):
    '''
    psi0(np,?,(N0,))
    psi1(np,?,(N0,))
    operator_list(list,(tuple,%,2))
        %0(float): coefficient
        %1(list,(tuple,%,?))
            %0(np,?,(N1,N1))
            %1...(tuple,int)
    '''
    psi0_conjugate = np.conjugate(psi0)
    ret = 0
    for coefficient,term_i in operator_list:
        tmp_psi1 = psi1 #no need to copy
        for tmp0 in reversed(term_i):
            tmp_psi1 = apply_gate(tmp_psi1, tmp0[0], list(tmp0[1:]))
        ret = ret + coefficient * np.dot(psi0_conjugate, tmp_psi1)
    return ret

# TODO to be replaced by inner_product_psi0_O_psi1
def operator_expectation(q0, operator, qubit_sequence):
    tmp0 = apply_gate(q0, operator, qubit_sequence)
    ret = np.dot(np.conjugate(q0), tmp0)
    return ret


def to_dm(q0):
    ret = q0[:,np.newaxis] * np.conjugate(q0)
    return ret


def reduce_to_probability(q0, keep_index_set):
    num_qubit = hf_num_state_to_num_qubit(q0.size)
    assert isinstance(keep_index_set,set) and all(0<=x for x in keep_index_set) and all(x<num_qubit for x in keep_index_set)
    tmp0 = (np.abs(q0)**2).reshape([2]*num_qubit)
    tmp1 = list(range(num_qubit))
    tmp2 = sorted(keep_index_set)
    ret = opt_einsum.contract(tmp0, tmp1, tmp2).reshape(-1)
    return ret


def inner_product(q0, q1):
    ret = np.dot(q0. conj(), q1)
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

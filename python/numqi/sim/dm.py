import numpy as np
import opt_einsum

from numqi.utils import hf_num_state_to_num_qubit

# TODO merge with circuit

def new_base(num_qubit:int, dtype=np.complex128):
    r'''return the base density matrix of the qubit quantum system

    Parameters:
        num_qubit (int): the number of qubits
        dtype (dtype): the data type of the base state

    Returns:
        ret (np.ndarray): the base state, `ret.shape==(2**num_qubit,2**num_qubit)`
    '''
    ret = np.zeros((2**num_qubit,2**num_qubit), dtype=dtype)
    ret[0,0] = 1
    return ret


def apply_gate(dm:np.ndarray, op:np.ndarray, index:int|tuple[int]):
    r'''apply a gate to the density matrix

    Parameters:
        dm (np.ndarray): the density matrix, `dm.shape==(2**num_qubit,2**num_qubit)`
        op (np.ndarray): the gate, `op.shape==(2**N0,2**N0)`
        index (int|tuple[int]): the qubit index to apply the gate, `0<=index<num_qubit`, count from left to right |0123>

    Returns:
        ret (np.ndarray): the density matrix after applying the gate
    '''
    num_state = len(dm)
    assert dm.ndim==2 and dm.shape==(num_state,num_state)
    num_qubit = hf_num_state_to_num_qubit(num_state)
    N0 = len(index)
    assert num_state==(2**num_qubit)
    assert all(isinstance(x,int) and (0<=x) and (x<num_qubit) for x in index)
    assert len(index)==len(set(index))
    assert op.ndim==2 and op.shape[0]==op.shape[1]
    assert op.shape[0]==2**N0

    tmp0 = dm.reshape([2 for _ in range(num_qubit)]+[num_state])
    tmp1 = list(range(num_qubit+1))
    tmp2 = op.reshape([2 for _ in range(2*N0)])
    tmp3 = list(range(num_qubit+1,num_qubit+1+N0))
    tmp4 = {x:y for x,y in zip(index,tmp3)}
    tmp5 = [(tmp4[x] if x in tmp4 else x) for x in range(num_qubit+1)]
    dm_with_left_op = opt_einsum.contract(tmp0, tmp1, tmp2, tmp3+index, tmp5).reshape(num_state,num_state)

    tmp0 = dm_with_left_op.reshape([num_state] + [2 for _ in range(num_qubit)])
    tmp1 = list(range(num_qubit+1))
    tmp2 = np.conjugate(op).reshape([2 for _ in range(2*N0)])
    tmp3 = list(range(num_qubit+1,num_qubit+1+N0))
    index_plus1 = [x+1 for x in index]
    tmp4 = {x:y for x,y in zip(index_plus1,tmp3)}
    tmp5 = [(tmp4[x] if x in tmp4 else x) for x in range(num_qubit+1)]
    ret = opt_einsum.contract(tmp0, tmp1, tmp2, tmp3+index_plus1, tmp5).reshape(num_state,num_state)
    return ret


# def apply_asymmetric_gate(dm:np.ndarray, op0:np.ndarray, index0:int|tuple[int], op1:np.ndarray, index1:int|tuple[int]):
#     r'''apply asymmetric operator to the density matrix

#     Parameters:
#         dm (np.ndarray): the density matrix, `dm.shape==(2**num_qubit,2**num_qubit)`
#         op0 (np.ndarray): the left operator, `op0.shape==(2**N0,2**N0)`
#         index0 (int|tuple[int]): the qubit index to apply the left operator
#         op1 (np.ndarray): the right operator, `op1.shape==(2**N1,2**N1)`
#         index1 (int|tuple[int]): the qubit index to apply the right operator

#     '''
#     num_state = len(dm)
#     num_qubit = hf_num_state_to_num_qubit(num_state)
#     tmp0 = apply_gate(dm.reshape(-1), op0, index0)
#     ret = apply_gate(tmp0.reshape(-1), op1, [x+num_qubit for x in index1]).reshape(num_state, num_state)
#     return ret


def operator_expectation(dm0:np.ndarray, op:np.ndarray, index:int|tuple[int]):
    r'''calculate the expectation value of an operator

    TODO multiple operators

    Parameters:
        dm0 (np.ndarray): the density matrix, `dm0.shape==(2**num_qubit,2**num_qubit)`
        op (np.ndarray): the operator, `op.shape==(2**N0,2**N0)`
        index (int|tuple[int]): the qubit index to apply the operator

    Returns:
        ret (np.ndarray): the expectation value
    '''
    num_state = len(dm0)
    num_qubit = hf_num_state_to_num_qubit(num_state)
    ind_map = {y:(x+num_qubit) for x,y in enumerate(index)}
    tmp0 = dm0.reshape((2,)*(2*num_qubit))
    tmp1 = [ind_map.get(x, x) for x in range(num_qubit)] + list(range(num_qubit))
    tmp2 = op.reshape((2,)*(2*len(index)))
    tmp3 = list(index) + [ind_map[x] for x in index]
    ret = opt_einsum.contract(tmp0, tmp1, tmp2, tmp3, [])
    return ret


# def apply_kraus_operator(dm0, operator_qubit_sequence):
#     ret = 0
#     for operator,qubit_sequence in operator_qubit_sequence:
#         ret = ret + apply_gate(dm0, operator, qubit_sequence)
#     return ret


# def make_measurement(dm0, kraus_op):
#     # TODO unittest
#     ret = []
#     for op,index in kraus_op:
#         tmp0 = apply_gate(dm0, op, index)
#         ret.append((np.trace(tmp0).real.item(), tmp0))
#     return ret


# def partial_trace(dm0, keep_index_sequence):
#     num_state = len(dm0)
#     num_qubit = hf_num_state_to_num_qubit(num_state)
#     ind_map = {y:(x+num_qubit) for x,y in enumerate(keep_index_sequence)}
#     tmp0 = dm0.reshape((2,)*(2*num_qubit))
#     tmp1 = list(range(num_qubit)) + [ind_map.get(x,x) for x in range(num_qubit)]
#     tmp2 = keep_index_sequence + [ind_map.get(x) for x in keep_index_sequence]
#     ret = opt_einsum.contract(tmp0, tmp1, tmp2).reshape(2**len(keep_index_sequence), 2**len(keep_index_sequence))
#     return ret

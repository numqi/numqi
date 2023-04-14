import numpy as np
import opt_einsum

from numqi.utils import hf_num_state_to_num_qubit


def new_base(num_qubit, dtype):
    ret = np.zeros((2**num_qubit,2**num_qubit), dtype=dtype)
    ret[0,0] = 1
    return ret


def apply_gate(dm, operator, qubit_sequence):
    # qubit_sequence: count from left to right |0123>
    num_state = len(dm)
    assert dm.ndim==2 and dm.shape==(num_state,num_state)
    num_qubit = hf_num_state_to_num_qubit(num_state)
    N0 = len(qubit_sequence)
    assert num_state==(2**num_qubit)
    assert all(isinstance(x,int) and (0<=x) and (x<num_qubit) for x in qubit_sequence)
    assert len(qubit_sequence)==len(set(qubit_sequence))
    assert operator.ndim==2 and operator.shape[0]==operator.shape[1]
    assert operator.shape[0]==2**N0

    tmp0 = dm.reshape([2 for _ in range(num_qubit)]+[num_state])
    tmp1 = list(range(num_qubit+1))
    tmp2 = operator.reshape([2 for _ in range(2*N0)])
    tmp3 = list(range(num_qubit+1,num_qubit+1+N0))
    tmp4 = {x:y for x,y in zip(qubit_sequence,tmp3)}
    tmp5 = [(tmp4[x] if x in tmp4 else x) for x in range(num_qubit+1)]
    dm_with_left_op = opt_einsum.contract(tmp0, tmp1, tmp2, tmp3+qubit_sequence, tmp5).reshape(num_state,num_state)

    tmp0 = dm_with_left_op.reshape([num_state] + [2 for _ in range(num_qubit)])
    tmp1 = list(range(num_qubit+1))
    tmp2 = np.conjugate(operator).reshape([2 for _ in range(2*N0)])
    tmp3 = list(range(num_qubit+1,num_qubit+1+N0))
    qubit_sequence_plus1 = [x+1 for x in qubit_sequence]
    tmp4 = {x:y for x,y in zip(qubit_sequence_plus1,tmp3)}
    tmp5 = [(tmp4[x] if x in tmp4 else x) for x in range(num_qubit+1)]
    ret = opt_einsum.contract(tmp0, tmp1, tmp2, tmp3+qubit_sequence_plus1, tmp5).reshape(num_state,num_state)
    return ret


def apply_asymmetric_gate(dm, op0, qubit_sequence0, op1, qubit_sequence1):
    num_state = len(dm)
    num_qubit = hf_num_state_to_num_qubit(num_state)
    tmp0 = apply_gate(dm.reshape(-1), op0, qubit_sequence0)
    ret = apply_gate(tmp0.reshape(-1), op1, [x+num_qubit for x in qubit_sequence1]).reshape(num_state, num_state)
    return ret


def operator_expectation(dm0, operator, qubit_sequence):
    # TODO multiple operator
    num_state = len(dm0)
    num_qubit = hf_num_state_to_num_qubit(num_state)
    ind_map = {y:(x+num_qubit) for x,y in enumerate(qubit_sequence)}
    tmp0 = dm0.reshape((2,)*(2*num_qubit))
    tmp1 = [ind_map.get(x, x) for x in range(num_qubit)] + list(range(num_qubit))
    tmp2 = operator.reshape((2,)*(2*len(qubit_sequence)))
    tmp3 = list(qubit_sequence) + [ind_map[x] for x in qubit_sequence]
    ret = opt_einsum.contract(tmp0, tmp1, tmp2, tmp3, [])
    return ret


def apply_kraus_operator(dm0, operator_qubit_sequence):
    ret = 0
    for operator,qubit_sequence in operator_qubit_sequence:
        ret = ret + apply_gate(dm0, operator, qubit_sequence)
    return ret


def make_measurement(dm0, kraus_op):
    # TODO unittest
    ret = []
    for op,index in kraus_op:
        tmp0 = apply_gate(dm0, op, index)
        ret.append((np.trace(tmp0).real.item(), tmp0))
    return ret


def partial_trace(dm0, keep_index_sequence):
    num_state = len(dm0)
    num_qubit = hf_num_state_to_num_qubit(num_state)
    ind_map = {y:(x+num_qubit) for x,y in enumerate(keep_index_sequence)}
    tmp0 = dm0.reshape((2,)*(2*num_qubit))
    tmp1 = list(range(num_qubit)) + [ind_map.get(x,x) for x in range(num_qubit)]
    tmp2 = keep_index_sequence + [ind_map.get(x) for x in keep_index_sequence]
    ret = opt_einsum.contract(tmp0, tmp1, tmp2).reshape(2**len(keep_index_sequence), 2**len(keep_index_sequence))
    return ret

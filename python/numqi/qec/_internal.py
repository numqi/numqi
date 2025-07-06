import numpy as np

# import numqi.utils
# import numqi.sim

def hf_state(x:str):
    r'''convert state string to vector, should NOT be used in performance-critical code

    Parameters:
        x (str): state string, e.g. '0000 1111', supporting sign "+-i", e.g. '0000 -i1111'

    Returns:
        ret (np.ndarray): shape=(2**n,), state vector
    '''
    tmp0 = x.strip().split(' ')
    tmp1 = []
    for x in tmp0:
        if x.startswith('-i'):
            tmp1.append((-1j, x[2:]))
        elif x.startswith('i'):
            tmp1.append((1j, x[1:]))
        elif x.startswith('-'):
            tmp1.append((-1, x[1:]))
        else:
            x = x[1:] if x.startswith('+') else x
            tmp1.append((1, x))
    tmp1 = sorted(tmp1, key=lambda x:x[1])
    num_qubit = len(tmp1[0][1])
    assert all(len(x[1])==num_qubit for x in tmp1)
    index = np.array([int(x[1],base=2) for x in tmp1], dtype=np.int64)
    coeff = np.array([x[0] for x in tmp1])
    coeff = coeff/np.linalg.norm(coeff, ord=2)
    ret = np.zeros(2**num_qubit, dtype=coeff.dtype)
    ret[index] = coeff
    return ret
_state_hf0 = hf_state


# def degeneracy(code_i):
#     # only works for d=3
#     num_qubit = numqi.utils.hf_num_state_to_num_qubit(code_i.shape[0])
#     error_list = make_error_list(num_qubit, distance=2) + [()] #identity
#     mat = np.zeros([len(error_list), len(error_list)], dtype=np.complex128)
#     for ind0 in range(len(error_list)):
#         q0 = code_i
#         for ind_op,op_i in error_list[ind0]:
#             q0 = numqi.sim.state.apply_gate(q0, op_i, ind_op)
#         for ind1 in range(len(error_list)):
#             q1 = code_i
#             for ind_op,op_i in error_list[ind1]:
#                 q1 = numqi.sim.state.apply_gate(q1, op_i, ind_op)
#             mat[ind0,ind1] = np.vdot(q0, q1)
#     EVL = np.linalg.eigvalsh(mat)
#     return EVL


def generate_code_np(circ, num_logical_dim):
    num_qubit = circ.num_qubit
    ret = []
    for ind0 in range(num_logical_dim):
        q0 = np.zeros(2**num_qubit, dtype=np.complex128)
        q0[ind0] = 1
        circ.apply_state(q0)
        ret.append(circ.apply_state(q0))
    ret = np.stack(ret, axis=0)
    return ret

import itertools
import numpy as np

import torch

import numqi.gate
import numqi.utils
import numqi.sim

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

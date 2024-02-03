import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import numqi

cp_tableau = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
np_rng = np.random.default_rng()


def get_qubit_projector_basis(num_qubit):
    assert num_qubit>=1
    hf0 = lambda x,y,z=1: np.array([[x*np.conj(x),x*np.conj(y)], [y*np.conj(x), y*np.conj(y)]])/(z*z)
    s2 = np.sqrt(2)
    tmp0 = [(1,0), (0,1), (1,1,s2), (1,-1,s2), (1,1j,s2), (1,-1j,s2)]
    np0 = np.stack([hf0(*x) for x in tmp0])
    ret = np0
    for _ in range(num_qubit-1):
        tmp0 = np.einsum(ret, [0,1,2], np0, [3,4,5], [0,3,1,4,2,5], optimize=True)
        tmp1 = [x*y for x,y in zip(ret.shape, np0.shape)]
        ret = tmp0.reshape(tmp1)
    ret = np.concatenate([np.eye(ret.shape[1])[np.newaxis], ret], axis=0)
    return ret

def demo_qubit_ud_orthonormal():
    # 2 qubits, non-orthogonal matrix, UDA!=UDP
    num_qubit = 2
    matrix_subspace = get_qubit_projector_basis(num_qubit)
    z0 = numqi.unique_determine.find_UD('udp', num_round=10, matrix_basis=matrix_subspace, indexF=[0],
                num_repeat=100, num_random_select=15, tag_reduce=True, early_stop_threshold=0.0001, num_worker=10)
    matB_list = [matrix_subspace[x] for x in z0]
    z1 = numqi.unique_determine.check_UD('udp', matB_list, num_repeat=1000, early_stop_threshold=0.0001, converge_tol=1e-7, dtype='float64')
    z2 = numqi.unique_determine.check_UD('uda', matB_list, num_repeat=1000, early_stop_threshold=0.0001, converge_tol=1e-7, dtype='float64')


def demo_EP_POVM():
    dim0 = 3
    matrix_subspace = numqi.unique_determine.get_element_probing_POVM('eq8', dim0)
    tag_ud,loss,model = numqi.unique_determine.check_UD('udp', matrix_subspace, num_repeat=3,
                early_stop_threshold=1e-11, converge_tol=1e-11, dtype='float64', return_model=True)
    matH = model.matH.numpy().copy()
    # |11> - |22>

    dim0 = 6
    matrix_subspace = numqi.unique_determine.get_element_probing_POVM('eq9', dim0)
    tag_ud,loss,model = numqi.unique_determine.check_UD('udp', matrix_subspace, num_repeat=3,
                early_stop_threshold=1e-13, converge_tol=1e-13, dtype='float64', return_model=True)
    matH = model.matH.numpy().copy()
    assert np.abs(np.trace(matrix_subspace @ matH, axis1=1, axis2=2)).max() < 1e-6


def demo_pauli_loss_function():
    num_worker = 4
    num_qubit_to_num_repeat = {2:50, 3:50, 4:400, 5:3200}
    pauli_len_dict = {2:11, 3:31, }# 4:106, 5:398
    z0 = numqi.unique_determine.load_pauli_ud_example(tag_group_by_size=True)
    z0 = {k:{k1:v1[:6] for k1,v1 in v.items()} for k,v in z0.items()} #each group only keep 6 items
    z1 = dict()
    for num_qubit,max_len in pauli_len_dict.items():
        if num_qubit not in z0:
            continue
        num_repeat = num_qubit_to_num_repeat[num_qubit]
        matrix_subspace = numqi.gate.get_pauli_group(num_qubit, use_sparse=True)

        tmp0 = [y for x,y in z0[num_qubit].items() if x<=max_len]
        index_list = [y for x in tmp0 for y in x]
        matB_list = [numqi.unique_determine.get_matrix_list_indexing(matrix_subspace, x) for x in index_list]
        uda_loss = [x[1] for x in numqi.unique_determine.check_UD('uda', matB_list, num_repeat, num_worker=num_worker)]
        udp_loss = [x[1] for x in numqi.unique_determine.check_UD('udp', matB_list, num_repeat, num_worker=num_worker)]
        tmp0 = itertools.groupby(zip(index_list,uda_loss,udp_loss), key=lambda x:len(x[0]))
        tmp1 = {x0:list(zip(*x1)) for x0,x1 in tmp0}
        for x0,x1 in tmp1.items():
            print(f'[{num_qubit},{x0},UDA]', np.sort(np.array(x1[1])))
            print(f'[{num_qubit},{x0},UDP]', np.sort(np.array(x1[2])))
        z1[num_qubit] = tmp1

# [2,11,UDA] [1.33333623 1.33333671 1.33333683 1.33334064 1.33334363 1.3333447 ]
# [2,11,UDP] [1.99999869 1.99999881 1.99999905 1.99999917 1.99999917 1.99999917]
# [3,31,UDA] [0.98112118 0.98115534 0.98131108 0.98138881 0.98143536 0.98149371]
# [3,31,UDP] [1.99999905 1.99999952 1.99999952 1.99999952 1.99999976 1.99999976]
# [4,106,UDA] [0.72969043 0.78669071 0.80056381]
# [4,106,UDP] [1.78830206 1.81233203 1.82333124]
# [5,393,UDA] [0.66714954]
# [5,393,UDP] [1.95155692]
# [5,395,UDA] [0.66707623]
# [5,395,UDP] [1.90302622]
# [5,397,UDA] [0.66705465 0.66711843]
# [5,397,UDP] [1.71428847 1.89451289]
# [5,398,UDA] [0.6671102  0.66723371]
# [5,398,UDP] [1.94959903 1.97744238]

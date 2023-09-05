import numpy as np

import numqi

def demo_search_UD_in_qutrit_projector():
    ## 1 qutrit, projector
    matrix_subspace = numqi.unique_determine.get_qutrit_projector_basis(num_qutrit=1)
    z0 = numqi.unique_determine.find_UDP_over_matrix_basis(num_round=1000, matrix_basis=matrix_subspace, indexF=[0],
            tag_reduce=True, num_repeat=10, num_random_select=1, key='1,qutrit', file='tbd00.json', num_worker=19)
    numqi.unique_determine.check_UDP_matrix_subspace(matrix_subspace[z0[0]], num_repeat=300, early_stop_threshold=1e-4)

    ## 2 qutrit, projector
    matrix_subspace = numqi.unique_determine.get_qutrit_projector_basis(num_qutrit=2)
    z0 = numqi.unique_determine.find_UDP_over_matrix_basis(num_round=1000, matrix_basis=matrix_subspace, indexF=[0],
            tag_reduce=True, early_stop_threshold=0.001, num_repeat=80, num_random_select=130, converge_tol=1e-6,
            key='2,qutrit', file='tbd00.json', num_worker=16)

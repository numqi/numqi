import numpy as np

import numqi


def demo_search_UD_in_pauli_group():
    num_qubit = 4
    num_repeat = {2:10, 3:10, 4:80, 5:80}[num_qubit]
    num_random_select = {2:0, 3:10, 4:80, 5:400}[num_qubit]
    matrix_subspace = numqi.gate.get_pauli_group(num_qubit, use_sparse=True)
    kwargs = {'num_repeat':num_repeat,  'num_random_select':num_random_select, 'indexF':[0],
                'num_worker':30, 'udp_use_vector_model':True, 'tag_reduce':False, 'key':f'{num_qubit},pauli,udp',
                'file':'tbd00.json'}
    z0 = numqi.unique_determine.find_UDP_over_matrix_basis(num_round=1, matrix_basis=matrix_subspace, **kwargs)
    # matB_list = [numqi.unique_determine.get_matrix_list_indexing(matrix_subspace, x) for x in z0]
    # z1 = numqi.unique_determine.check_UDA_matrix_subspace(matB_list, num_repeat=num_repeat*5, num_worker=19)


def demo_search_UD_in_gellmann_group():
    num_qudit = 2
    dim_qudit = 3
    num_repeat = {(2,3):80, (1,3):10}[(num_qudit,dim_qudit)]
    num_random_select = {(2,3):10, (1,3):0}[(num_qudit,dim_qudit)]

    gellmann_basis = numqi.gellmann.all_gellmann_matrix(num_qudit, tensor_n=dim_qudit, with_I=True) #last item is identity
    indexF = [len(gellmann_basis)-1]
    z0 = numqi.unique_determine.find_UDP_over_matrix_basis(num_round=10, matrix_basis=gellmann_basis, indexF=indexF, num_repeat=num_repeat,
                num_random_select=num_random_select, tag_reduce=False, key=f'{num_qudit},{dim_qudit},gellmann', file='tbd00.json', num_worker=1)
    # z0 = numqi.unique_determine.save_index_to_file('gellmann-indexB.json', '2,3,uda')
    matB_list = [gellmann_basis[x] for x in z0]
    z1 = numqi.unique_determine.check_UDA_matrix_subspace(matB_list, num_repeat=400, early_stop_threshold=1e-4, converge_tol=1e-7, dtype='float64', num_worker=19)

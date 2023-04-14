import numpy as np

import numpyqi

def test_get_sym_group_num_irrep():
    # eq-6.15 @ZhongqiMa
    data_list = [(1,1), (2,2), (3,3), (4,5), (5,7), (6,11), (7,15), (8,22), (9,30), (10,42),
            (20,627), (50,204226), (100,190569292), (200,3972999029388)]
    for x,y in data_list:
        assert numpyqi.group.get_sym_group_num_irrep(x)==y


def test_get_sym_group_young_diagram():
    data_list = [(1,1), (2,2), (3,3), (4,5), (5,7), (6,11), (7,15), (8,22), (9,30), (10,42), (20,627)]
    # (50,204226) take about 10 seconds
    for n,num_irrep in data_list:
        np0 = numpyqi.group.get_sym_group_young_diagram(n)
        assert np0.shape[0]==num_irrep
        assert np.all(np0.sum(axis=1)==n)
        assert np.all(np0[:,:-1]>=np0[:,1:])

def test_get_hook_length():
    assert numpyqi.group.get_hook_length(2,1)==2
    assert numpyqi.group.get_hook_length(4,3,1,1)==216


def test_get_all_young_tableaux():
    young_list = [(2,1), (3,1,1), (3,2), (3,3,2), (4,3,1,1), (4,3,2,1)]
    for x in young_list:
        tmp0 = numpyqi.group.get_hook_length(*x)
        z0 = numpyqi.group.get_all_young_tableaux(x)
        assert z0.shape[0]==tmp0

        mask = numpyqi.group.get_young_diagram_mask(x)
        assert np.all(np.logical_or(np.logical_not(mask[1:]), z0[:,1:]>z0[:,:-1]))
        assert np.all(np.logical_or(np.logical_not(mask[:,1:]), z0[:,:,1:]>z0[:,:,:-1]))


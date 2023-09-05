import numpy as np

import numqi

np_rng = np.random.default_rng()


def demo_qutrit_coeffB():
    coeffB_list,multiplicity_list = numqi.group.symext.get_symmetric_extension_irrep_coeff(dimB=3, kext=2)
    print([len(np.nonzero(np.abs(x)>1e-10)[0]) for x in coeffB_list]) #27,12
    print([len(np.nonzero(np.abs(x).max(axis=(2,3))>1e-10)[0]) for x in coeffB_list]) #24,9
    for ind0,coeffB_i in enumerate(coeffB_list):
        print(f'B^({ind0})')
        numqi.group.symext.print_symmetric_extension_irrep_coeffB(coeffB_i)

    coeffB_list,multiplicity_list = numqi.group.symext.get_symmetric_extension_irrep_coeff(dimB=3, kext=3)
    print([len(np.nonzero(np.abs(x)>1e-10)[0]) for x in coeffB_list]) #54,50,3
    print([len(np.nonzero(np.abs(x).max(axis=(2,3))>1e-10)[0]) for x in coeffB_list]) #46,40,1
    for ind0,coeffB_i in enumerate(coeffB_list):
        print(f'B^({ind0})')
        numqi.group.symext.print_symmetric_extension_irrep_coeffB(coeffB_i)


def demo_B3_B4_basis():
    num_young_tab = [1,2,1]
    ind_parameter = np.zeros(sum(num_young_tab), dtype=np.int64)
    ind_parameter[np.cumsum(np.array([0]+num_young_tab))[:-1]] = 1
    ind_parameter = ind_parameter.tolist()
    for dim in [2,3,4,5,6]:
        z0 = numqi.group.symext.get_B3_irrep_basis(dim)
        tmp0 = [x.shape[0] for x in z0]
        num_parameter = sum([x*x for x,y in zip(tmp0,ind_parameter) if y])
        print(dim, [x.shape[0] for x in z0], num_parameter)

    # 1,3,2,3,1
    num_young_tab = [1,3,2,3,1]
    ind_parameter = np.zeros(sum(num_young_tab), dtype=np.int64)
    ind_parameter[np.cumsum(np.array([0]+num_young_tab))[:-1]] = 1
    ind_parameter = ind_parameter.tolist()
    for dim in [2,3,4,5,6]:
        z0 = numqi.group.symext.get_B4_irrep_basis(dim)
        tmp0 = [x.shape[0] for x in z0]
        num_parameter = sum([x*x for x,y in zip(tmp0,ind_parameter) if y])
        print(dim, [x.shape[0] for x in z0], num_parameter)

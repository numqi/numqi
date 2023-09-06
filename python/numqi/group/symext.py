import math
import itertools
import functools
import numpy as np

import numqi.dicke
from ._symmetric import get_all_young_tableaux, get_young_diagram_mask, young_tableau_to_young_symmetrizer, get_sym_group_young_diagram


def _ABk_permutate(mat, ind0, ind1, dimA, dimB, kext, kind):
    assert kind in {1,2,3}
    tmp0 = [dimA] + [dimB]*kext + [dimA] + [dimB]*kext
    tmp1 = list(range(2*kext+2))
    if (kind==2) or (kind==3):
        tmp1[ind0+1],tmp1[ind1+1] = tmp1[ind1+1],tmp1[ind0+1]
    if (kind==1) or (kind==3):
        tmp1[kext+1+ind0+1],tmp1[kext+1+ind1+1] = tmp1[kext+1+ind1+1],tmp1[kext+1+ind0+1]
    ret = mat.reshape(tmp0).transpose(tmp1).reshape(mat.shape)
    return ret


def get_ABk_symmetry_index(dimA, dimB, kext, use_boson=False):
    index_to_set = np.arange((dimA*dimB**kext)**2, dtype=np.int64).reshape(dimA*dimB**kext, -1)
    tmp0 = [(x,y) for x in range(kext) for y in range(x+1,kext)]
    for ind0,ind1 in tmp0:
        if use_boson:
            index_to_set = np.minimum(index_to_set, _ABk_permutate(index_to_set, ind0, ind1, dimA, dimB, kext, kind=1))
            index_to_set = np.minimum(index_to_set, _ABk_permutate(index_to_set, ind0, ind1, dimA, dimB, kext, kind=2))
        else:
            index_to_set = np.minimum(index_to_set, _ABk_permutate(index_to_set, ind0, ind1, dimA, dimB, kext, kind=3))
    index_to_set_sym = np.minimum(index_to_set, index_to_set.T)
    tmp0 = index_to_set_sym.reshape(-1)
    tmp1 = np.unique(tmp0)
    tmp2 = -np.ones(tmp1.max()+1, dtype=np.int64)
    tmp2[tmp1] = np.arange(tmp1.shape[0])
    index_sym = tmp2[index_to_set_sym]

    tag_zero = index_to_set == index_to_set.T
    factor_skew = 2*(index_to_set < index_to_set.T) - 1
    factor_skew[tag_zero] = 0
    index_to_set_skew = np.minimum(index_to_set, index_to_set.T)+1
    index_to_set_skew[tag_zero] = 0
    tmp0 = np.abs(index_to_set_skew)
    tmp1 = np.unique(tmp0)
    tmp2 = -np.ones(tmp1.max()+1, dtype=np.int64)
    tmp2[tmp1] = np.arange(tmp1.shape[0])
    index_skew = tmp2[tmp0]
    return index_sym,index_skew,factor_skew

def get_ABk_symmetrize(np0, dimA, dimB, kext, use_boson=False):
    assert kext>=1
    assert (np0.ndim==2) and (np0.shape[0]==np0.shape[1]) and (np0.shape[0]==dimA*dimB**kext)
    if kext==1:
        ret = np0.copy()
    else:
        np0 = np0.reshape([dimA]+[dimB]*kext+[dimA]+[dimB]*kext) / math.factorial(kext)
        ret = np0 * (3 if use_boson else 1)
        for indI in list(itertools.permutations(list(range(kext))))[1:]:
            tmp0 = [[0] + [(1+x) for x in indI] + [kext+1] + [(2+kext+x) for x in indI]]
            if use_boson:
                tmp0 += [
                    [0] + [(1+x) for x in indI] + list(range(1+kext,2+kext+kext)),
                    list(range(kext+2)) + [(2+kext+x) for x in indI],
                ]
            for ind0 in tmp0:
                ret += np.transpose(np0, ind0)
        ret = ret.reshape(dimA*dimB**kext, dimA*dimB**kext)
        if use_boson:
            ret /= 3
    return ret


def get_B1B2_basis():
    basis_b = np.zeros((6,9), dtype=np.float64)
    basis_b[[0,1,2], [0,4,8]] = 1
    basis_b[[3,3,4,4,5,5], [1,3,2,6,5,7]] = 1/np.sqrt(2)
    basis_f = np.zeros((3,9), dtype=np.float64)
    basis_f[[0,0,1,1,2,2], [1,3,2,6,5,7]] = 1/np.sqrt(2) * (1-2*(np.arange(6)%2))
    tmp0 = np.concatenate([basis_b,basis_f], axis=0)
    assert np.abs(tmp0 @ tmp0.T - np.eye(9)).max() < 1e-10
    return basis_b,basis_f


def get_B1B2B3_basis():
    # hf0 = lambda x: f'{x//9}{(x%9)//3}{x%3}'
    # hf1 = lambda y: [int(x[0])*9+int(x[1])*3+int(x[2]) for x in y.split(' ')]
    basis3 = np.zeros((10,27), dtype=np.float64)
    # 000 111 222
    basis3[[0,1,2], [0,13,26]] = 1
    # 001 002 110 112 220 221
    tmp0 = [x for x in range(3,9) for _ in range(3)]
    tmp1 = [1,3,9, 2,6,18, 4,10,12, 14,16,22, 8,20,24, 17,23,25]
    basis3[tmp0,tmp1] = 1/np.sqrt(3)
    basis3[9,[5,7,11,15,19,21]] = 1/np.sqrt(6)

    basis21a = np.zeros((8,27), dtype=np.float64)
    # 001 002 112 110 220 221
    tmp0 = [0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4, 5,5,5]
    tmp1 = [1,3,9, 2,6,18, 14,16,22, 12,10,4, 24,20,8, 25,23,17]
    basis21a[tmp0,tmp1] = np.array([2,-1,-1, 2,-1,-1, 2,-1,-1, 2,-1,-1, 2,-1,-1, 2,-1,-1])/np.sqrt(6)
    basis21a[6,[7,15,19,21]] = np.array([1,-1,1,-1])/2
    basis21a[7, [5, 7, 11, 15, 19, 21]] = np.array([2,-1,2,-1,-1,-1])/np.sqrt(12)
    basis21a = basis21a[[0,1,3,7,6,4,2,5]] * np.array([1,1,-1,1,1,-1,1,-1]).reshape(-1,1) #must be in this order and phase

    basis21b = np.zeros((8,27), dtype=np.float64)
    tmp0 = [0,0, 1,1, 2,2, 3,3, 4,4, 5,5]
    tmp1 = [9,3, 18,6, 4,10, 22,16, 8,20, 17,23]
    basis21b[tmp0,tmp1] = np.array([1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1])/np.sqrt(2)
    basis21b[6, [7,15,19,21]] = np.array([1,1,-1,-1])/2
    basis21b[7, [5,7,11,15,19,21]] = np.array([2,1,-2,-1,-1,1])/np.sqrt(12)
    basis21b = basis21b[[0,1,2,6,7,4,3,5]] * np.array([-1,-1,1,1,1,1,-1,1]).reshape(-1,1) #must be in this order and phase

    basis111 = np.zeros((1,27), dtype=np.float64)
    basis111[0,[5, 15, 19, 21, 11, 7]] = np.array([1,1,1,-1,-1,-1])/np.sqrt(6)
    return basis3,basis21a,basis21b,basis111


def get_sud_symmetric_irrep_basis(dim, kext, zero_eps=1e-7):
    assert dim >= 2
    Ydiagram_list = [tuple(y for y in x if y>0) for x in get_sym_group_young_diagram(kext).tolist()]
    # TODO sparse
    eye_reshape = np.eye(dim**kext).reshape([dim]*kext+[dim**kext])

    basis_list = []
    valid_Ydiagram_list = [x for x in Ydiagram_list if len(x)<=dim]
    for Ydiagram_i in valid_Ydiagram_list:
        Ytableaux = get_all_young_tableaux(Ydiagram_i)
        Ymask = get_young_diagram_mask(Ydiagram_i).astype(np.bool_)
        Yop,coeff = young_tableau_to_young_symmetrizer(Ydiagram_i, Ytableaux[0])
        coeff = coeff * (Ytableaux.shape[0]/math.factorial(kext)) #make a normalized projector, not necessary here
        tmp0 = 0
        for x,y in zip(Yop, coeff):
            tmp0 = tmp0 + np.transpose(eye_reshape, x.tolist() + [len(x)])*y
        projector = tmp0.reshape(dim**kext, -1)
        if Ytableaux.shape[0]==1:
            EVL,EVC = np.linalg.eigh(projector)
            assert np.abs(EVL[EVL>zero_eps]-1).max() < zero_eps #should all be 1 or 0
            basis_list.append([EVC[:,EVL>zero_eps]])
        else:
            EVL,EVC = np.linalg.eigh(projector @ projector.T)
            basis_i = [EVC[:,EVL>zero_eps]]
            assert np.all(Ytableaux[0, Ymask]==np.arange(kext))
            basis_i0 = basis_i[0].reshape([dim]*kext+[-1])
            for ind_tab in range(1,len(Ytableaux)):
                tmp0 = np.argsort(Ytableaux[ind_tab, Ymask]).tolist() + [kext]
                # np.transpose(eye_reshape, tmp0).reshape(dim**kext,-1) @ basis_i[0]
                tmp1 = np.transpose(basis_i0, tmp0).reshape(dim**kext, -1)
                for x in basis_i:
                    tmp1 = tmp1 - x @ (x.T.conj() @ tmp1)
                tmp2 = np.linalg.norm(tmp1, axis=0)
                assert np.min(tmp2) > zero_eps
                basis_i.append(tmp1/tmp2)
            basis_list.append(basis_i)
    basis_list = [[y.T for y in x] for x in basis_list]
    return basis_list


def _basis_partial_trace(basis, dim):
    tmp0 = basis.reshape(basis.shape[0], dim, -1)
    ret = np.einsum(tmp0, [0,1,2], tmp0.conj(), [3,4,2], [0,3,1,4], optimize=True)
    return ret


@functools.lru_cache
def get_symmetric_extension_irrep_coeff(dimB, kext):
    dimB = int(dimB)
    kext = int(kext)
    if dimB==2:
        tmp0 = numqi.dicke.get_partial_trace_ABk_to_AB_index(kext, dim=2, return_tensor=True).transpose(2,3,0,1).copy()
        # a00,a01,a10,a11 = numqi.dicke.dicke_state_partial_trace(kext)
        # tmp0 = np.stack([np.diag(a00), np.diag(a01,1), np.diag(a10,-1), np.diag(a11)], axis=2).reshape(kext+1,kext+1,2,2)
        coeffB_list = [tmp0]
        multiplicity_list = 1, #all sym-ext are bosonic-ext, so we only use Dicke state
    else:
        basis_part = get_sud_symmetric_irrep_basis(dimB, kext)
        multiplicity_list = tuple(len(x) for x in basis_part)
        coeffB_list = [sum(_basis_partial_trace(y,dimB) for y in x) for x in basis_part]
    for x in coeffB_list:
        x.flags.writeable = False
    return coeffB_list,multiplicity_list


def print_symmetric_extension_irrep_coeffB(coeffB):
    index = np.stack(np.nonzero(coeffB), axis=1)
    index = np.array(sorted({((a,b,c,d) if a<b else (b,a,d,c)) for a,b,c,d in index}))
    tmp0 = np.stack(tuple(index.T) + (coeffB[tuple(index.T)],), axis=1)
    hf0 = lambda x: (-abs(x[1][4]),x[1][0],x[1][1],x[1][2],x[1][3])
    ind0 = [x[0] for x in sorted(enumerate(tmp0), key=hf0)]
    for x in tmp0[ind0]:
        print([int(y) for y in x[:4]]+[float(x[4])])


def print_young_tableaux(N0):
    tmp0 = [tuple(y for y in x if y>0) for x in get_sym_group_young_diagram(N0).tolist()]
    young_tableaux = {x:get_all_young_tableaux(x) for x in tmp0}
    for key,value in young_tableaux.items():
        tmp0 = '[' + ','.join(str(x) for x in key) + ']:'
        print(tmp0, f'#tableaux: {len(value)}')
        mask = get_young_diagram_mask(key)
        for ind0 in range(len(value)):
            for x,y in zip(value[ind0],mask):
                print(x[:y.sum()].tolist())
            print(('=' if ind0==(len(value)-1) else '-')*30)


# def hf_good_print(np0, zero_eps=1e-7):
#     tmp0 = np0.copy()
#     tmp0[np.abs(np0)<zero_eps] = 0
#     print(tmp0)


def get_B3_irrep_basis(dim, zero_eps=1e-7):
    assert dim >= 2
    kext = 3
    tmp0 = [tuple(y for y in x if y>0) for x in get_sym_group_young_diagram(kext).tolist()]
    young_tableaux = {x:get_all_young_tableaux(x) for x in tmp0}
    young_op = {x0:[young_tableau_to_young_symmetrizer(x0,y) for y in x1] for x0,x1 in young_tableaux.items()}

    s0 = np.eye(dim**kext)
    identity = s0.reshape([dim]*kext+[dim**kext])
    s12 = identity.transpose(0,2,1,3).reshape(s0.shape)

    op_i,coeff_i = young_op[(3,)][0]
    coeff_i = coeff_i/6
    tmp0 = 0
    for x,y in zip(op_i,coeff_i):
        tmp0 = tmp0 + np.transpose(identity, x.tolist() + [len(x)])*y
    projector3 = tmp0.reshape(dim**kext,-1)
    EVL,EVC = np.linalg.eigh(projector3)
    basis3 = EVC[:,EVL>zero_eps]

    op_i,coeff_i = young_op[(2,1)][0]
    coeff_i = coeff_i/3
    tmp0 = 0
    for x,y in zip(op_i,coeff_i):
        tmp0 = tmp0 + np.transpose(identity, x.tolist() + [len(x)])*y
    projector21a = tmp0.reshape(dim**kext,-1)
    tmp0 = projector21a @ projector21a.T.conj()
    EVL,EVC = np.linalg.eigh(tmp0)
    phi21 = EVC[:,EVL>zero_eps]
    basis21a = phi21
    tmp1 = (s0 - basis21a@basis21a.T.conj()) @ (s12 @ basis21a)
    basis21b = tmp1 / np.linalg.norm(tmp1, axis=0)
    # tmp1 = phi21.reshape([dim]*kext + [-1]).transpose(0,2,1,3).reshape(dim**kext,-1)
    # basis21a = phi21 + tmp1
    # basis21b = (phi21 - tmp1) / np.sqrt(3)

    if dim>2:
        op_i,coeff_i = young_op[(1,1,1)][0]
        coeff_i = coeff_i/6
        tmp0 = 0
        for x,y in zip(op_i,coeff_i):
            tmp0 = tmp0 + np.transpose(identity, x.tolist() + [len(x)])*y
        projector111 = tmp0.reshape(dim**kext,-1)
        EVL,EVC = np.linalg.eigh(projector111)
        basis111 = EVC[:,EVL>1e-7]
        ret = basis3.T, basis21a.T, basis21b.T, basis111.T
    else:
        ret = basis3.T, basis21a.T, basis21b.T
    return ret


def get_B4_irrep_basis(dim, zero_eps=1e-7):
    kext = 4
    assert dim>=2

    tmp0 = [tuple(y for y in x if y>0) for x in get_sym_group_young_diagram(kext).tolist()]
    young_tableaux = {x:get_all_young_tableaux(x) for x in tmp0}
    young_op = {x0:[young_tableau_to_young_symmetrizer(x0,y) for y in x1] for x0,x1 in young_tableaux.items()}

    s0 = np.eye(dim**kext)
    identity = s0.reshape([dim]*kext+[dim**kext])
    s12 = identity.transpose(0,2,1,3,4).reshape(s0.shape)
    s23 = identity.transpose(0,1,3,2,4).reshape(s0.shape)
    s132 = identity.transpose(0,2,3,1,4).reshape(s0.shape)
    s123 = identity.transpose(0,3,1,2,4).reshape(s0.shape)

    op_i,coeff_i = young_op[(4,)][0]
    coeff_i = coeff_i/24
    tmp0 = 0
    for x,y in zip(op_i,coeff_i):
        tmp0 = tmp0 + np.transpose(identity, x.tolist() + [len(x)])*y
    Y4 = tmp0.reshape(dim**kext,-1)
    EVL,EVC = np.linalg.eigh(Y4)
    basis4 = EVC[:,EVL>zero_eps] #35

    op_i,coeff_i = young_op[(3,1)][0]
    coeff_i = coeff_i/8
    tmp0 = 0
    for x,y in zip(op_i,coeff_i):
        tmp0 = tmp0 + np.transpose(identity, x.tolist() + [len(x)])*y
    Y31a = tmp0.reshape(dim**kext,-1)
    EVL,EVC = np.linalg.eigh(Y31a @ Y31a.T)
    phi31 = EVC[:,EVL>zero_eps]
    basis31a = phi31
    tmp1 = (s0 - basis31a@basis31a.T.conj()) @ (s23 @ basis31a)
    basis31b = tmp1 / np.linalg.norm(tmp1, axis=0)
    tmp1 = (s0 - basis31a@basis31a.T.conj() - basis31b@basis31b.T.conj()) @ (s123 @ basis31a)
    basis31c = tmp1 / np.linalg.norm(tmp1, axis=0)

    # basis31a = (s0 + s23 + s123) @ phi31
    # basis31b = (s0-s23) @ (s0+s123) @ phi31 * (np.sqrt(6)/4)
    # basis31c = (s0+s23) @ (s0-s123) @ phi31 * (1/np.sqrt(8))

    # op_i,coeff_i = young_op[(3,1)][1]
    # coeff_i = coeff_i/8
    # tmp0 = 0
    # for x,y in zip(op_i,coeff_i):
    #     tmp0 = tmp0 + np.transpose(identity, x.tolist() + [len(x)])*y
    # Y31b = tmp0.reshape(dim**kext,-1)
    # EVL,EVC = np.linalg.eigh(Y31b @ Y31b.T)
    # basis31b = EVC[:,EVL>zero_eps]
    # U31b = basis31b

    # op_i,coeff_i = young_op[(3,1)][2]
    # coeff_i = coeff_i/8
    # tmp0 = 0
    # for x,y in zip(op_i,coeff_i):
    #     tmp0 = tmp0 + np.transpose(identity, x.tolist() + [len(x)])*y
    # Y31c = tmp0.reshape(dim**kext,-1)
    # EVL,EVC = np.linalg.eigh(Y31c @ Y31c.T)
    # basis31c = EVC[:,EVL>zero_eps]


    op_i,coeff_i = young_op[(2,2)][0]
    coeff_i = coeff_i/12
    tmp0 = 0
    for x,y in zip(op_i,coeff_i):
        tmp0 = tmp0 + np.transpose(identity, x.tolist() + [len(x)])*y
    Y22a = tmp0.reshape(dim**kext,-1)
    EVL,EVC = np.linalg.eigh(Y22a @ Y22a.T)
    phi22 = EVC[:,EVL>zero_eps]
    basis22a = phi22
    tmp1 = (s0 - basis22a@basis22a.T.conj()) @ (s12 @ basis22a)
    basis22b = tmp1 / np.linalg.norm(tmp1, axis=0)
    # basis22a = (s0 + s12) @ phi22
    # basis22b = (s0 - s12) @ phi22 * (1/np.sqrt(3))

    # op_i,coeff_i = young_op[(2,2)][1]
    # coeff_i = coeff_i/12
    # tmp0 = 0
    # for x,y in zip(op_i,coeff_i):
    #     tmp0 = tmp0 + np.transpose(identity, x.tolist() + [len(x)])*y
    # Y22b = tmp0.reshape(dim**kext,-1)
    # EVL,EVC = np.linalg.eigh(Y22b @ Y22b.T)
    # Ub = EVC[:,EVL>zero_eps]


    if dim>=3:
        op_i,coeff_i = young_op[(2,1,1)][0]
        coeff_i = coeff_i/8
        tmp0 = 0
        for x,y in zip(op_i,coeff_i):
            tmp0 = tmp0 + np.transpose(identity, x.tolist() + [len(x)])*y
        Y211a = tmp0.reshape(dim**kext,-1)
        EVL,EVC = np.linalg.eigh(Y211a @ Y211a.T)
        phi211 = EVC[:,EVL>zero_eps]
        basis211a = phi211
        tmp1 = (s0 - basis211a@basis211a.T.conj()) @ (s12 @ basis211a)
        basis211b = tmp1 / np.linalg.norm(tmp1, axis=0)
        tmp1 = (s0 - basis211a@basis211a.T.conj() - basis211b@basis211b.T.conj()) @ (s132 @ basis211a)
        basis211c = tmp1 / np.linalg.norm(tmp1, axis=0)

        # op_i,coeff_i = young_op[(2,1,1)][1]
        # coeff_i = coeff_i/8
        # tmp0 = 0
        # for x,y in zip(op_i,coeff_i):
        #     tmp0 = tmp0 + np.transpose(identity, x.tolist() + [len(x)])*y
        # Y211b = tmp0.reshape(dim**kext,-1)
        # EVL,EVC = np.linalg.eigh(Y211b @ Y211b.T)

        # op_i,coeff_i = young_op[(2,1,1)][2]
        # coeff_i = coeff_i/8
        # tmp0 = 0
        # for x,y in zip(op_i,coeff_i):
        #     tmp0 = tmp0 + np.transpose(identity, x.tolist() + [len(x)])*y
        # Y211c = tmp0.reshape(dim**kext,-1)
        # EVL,EVC = np.linalg.eigh(Y211c @ Y211c.T)
        # phi211 = EVC[:,EVL>zero_eps]
        # basis211a = (s0-s123) @ (s0+s23) @ phi211 * (1/np.sqrt(3))
        # basis211b = (s0 + s123 - s23) @ phi211 * (1/np.sqrt(6))
        # basis211c = (s0+s123) @ (s0+s23) @ phi211


    if dim>=4:
        op_i,coeff_i = young_op[(1,1,1,1)][0]
        coeff_i = coeff_i/24
        tmp0 = 0
        for x,y in zip(op_i,coeff_i):
            tmp0 = tmp0 + np.transpose(identity, x.tolist() + [len(x)])*y
        Y1111 = tmp0.reshape(dim**kext,-1)
        EVL,EVC = np.linalg.eigh(Y1111)
        basis1111 = EVC[:,EVL>zero_eps]

    ret = basis4, basis31a, basis31b, basis31c, basis22a, basis22b
    if dim>=3:
        ret = ret + (basis211a,basis211b,basis211c)
    if dim>=4:
        ret = ret + (basis1111,)
    ret = tuple(x.T for x in ret)
    return ret

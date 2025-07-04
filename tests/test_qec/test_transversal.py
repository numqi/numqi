import numpy as np
import functools

import numqi

np_rng = np.random.default_rng()
hf_kron = lambda *x: functools.reduce(np.kron, x)

def test_transversal_group_723cyclic():
    Phi = numqi.qec.su2_finite_subgroup_gate_dict['Phi']
    Phis = numqi.qec.su2_finite_subgroup_gate_dict['Phi*']
    for sign in ['++','+-','-+','--']:
        code,info = numqi.qec.get_code_subspace('723cyclic', lambda2=np_rng.uniform(0, 7), sign=sign)
        assert np.abs(code @ numqi.qec.hf_pauli('X'*7) @ code.T - numqi.gate.X).max() < 1e-12
        F = numqi.qec.su2_finite_subgroup_gate_dict['F']
        assert np.abs(code @ hf_kron(*[F.conj()]*7) @ code.T - F).max() < 1e-12

        code,info = numqi.qec.get_code_subspace('723cyclic', lambda2=0, sign=sign)
        S = numqi.gate.rz(np.pi/2)
        assert np.abs(code @ hf_kron(*[S.conj()]*7) @ code.T - S).max() < 1e-12

        code,info = numqi.qec.get_code_subspace('723cyclic', lambda2=7, sign=sign)
        if sign[0]=='+':
            assert np.abs(code @ hf_kron(*[Phi]*7) @ code.T - Phis).max() < 1e-12
        else:
            assert np.abs(code @ hf_kron(*[Phis]*7) @ code.T - Phi).max() < 1e-12


def test_su2_finite_subgroup_gate_dict():
    for x in numqi.qec.su2_finite_subgroup_gate_dict.values():
        assert abs(x[0,0]*x[1,1] - x[0,1]*x[1,0] - 1) < 1e-10 #determinant 1
        assert np.abs(x @ x.T.conj() - np.eye(2)).max() < 1e-10


def test_su2_finite_subgroup():
    np_list = numqi.group.get_complete_group(numqi.qec.get_su2_finite_subgroup_generator('2T'))
    assert len(np_list)==24
    info = numqi.qec.get_transversal_group_info(np_list)
    assert np.all(info['dim_irrep']==np.array([1,1,1,2,2,2,3], dtype=np.int64))
    assert np.all(info['num_class']==np.array([1,1,4,4,4,4,6], dtype=np.int64))
    # numqi.group.pretty_print_character_table(info['character_table'], info['class_list'])

    # 2O, clifford
    np_list = numqi.group.get_complete_group(numqi.qec.get_su2_finite_subgroup_generator('2O'))
    assert len(np_list)==48
    info = numqi.qec.get_transversal_group_info(np_list)
    assert np.all(info['dim_irrep']==np.array([1,1,2,2,2,3,3,4], dtype=np.int64))
    assert np.all(info['num_class']==np.array([1,1,6,6,6,8,8,12], dtype=np.int64))

    # 2I (slow)
    np_list = numqi.group.get_complete_group(numqi.qec.get_su2_finite_subgroup_generator('2I'))
    assert len(np_list)==120
    # info = numqi.qec.get_transversal_group_info(np_list)
    # assert np.all(info['dim_irrep']==np.array([1,2,2,3,3,4,4,5,6], dtype=np.int64))
    # assert np.all(info['num_class']==np.array([1,1,12,12,12,12,20,20,30], dtype=np.int64))

    # BD_2n
    for n in range(1,7):
        np_list = numqi.group.get_complete_group(numqi.qec.get_su2_finite_subgroup_generator('BD'+str(2*n)))
        assert len(np_list)==4*n
        info = numqi.qec.get_transversal_group_info(np_list)
        tmp0 = np.array([1,1,1,1] + [2]*(n-1), dtype=np.int64)
        assert np.all(info['dim_irrep']==tmp0)
        tmp0 = np.array([1,1] + [2]*(n-1) + [n,n], dtype=np.int64)
        assert np.all(info['num_class']==tmp0)

    # C_2n
    # np_list = numqi.group.get_complete_group(numqi.qec.get_su2_finite_subgroup_generator('C'+str(2*n)))


def test_super_golden_gate():
    a = (1+np.sqrt(5))/2
    tau60 = np.array([[2+a, 1-1j], [1+1j, -2-a]])/np.sqrt(5*a+7)

    assert np.abs(tau60 @ tau60 - np.eye(2)).max() < 1e-12 #self-inverse
    assert np.abs(tau60 @ tau60.T.conj() - np.eye(2)).max() < 1e-12 #unitary

    rz = numqi.gate.rz
    tmp0 = 2*np.arccos((2+a)/np.sqrt(5*a+7))
    tmp1a = 1j*rz(np.pi/4) @ numqi.gate.ry(tmp0) @ rz(3*np.pi/4)
    assert np.abs(tmp1a - tau60).max() < 1e-12
    tmp1b = rz(np.pi/4) @ rz(np.pi/2) @ numqi.gate.H @ rz(tmp0) @ numqi.gate.H @ rz(-np.pi/2) @ numqi.gate.Z @ rz(-np.pi/4)
    assert np.abs(tmp1b - tau60).max() < 1e-12
    tmp1c = 1j * rz(3*np.pi/4) @ numqi.gate.H @ rz(tmp0) @ numqi.gate.H @ rz(np.pi/4)
    assert np.abs(tmp1c - tau60).max() < 1e-12
    # tmp1d = 1j * rz(3*np.pi/4) @ numqi.gate.H @ rz(np.pi*167/704) @ numqi.gate.H @ rz(np.pi/4)


def test_search_veca_C_group():
    x0 = numqi.qec.search_veca_C_group(n=6, m=5, tag_print=False)
    x1 = {(0,1,1,2,2,3),(0,1,3,3,3,4),(0,2,2,2,4,4),(1,1,1,1,1,4),(1,1,1,1,2,3),(1,1,1,2,2,2),
        (1,1,1,2,2,3),(1,1,2,2,2,3),(1,1,2,2,3,3),(1,1,2,2,3,4),(1,1,2,2,4,4),(1,1,2,3,3,4),
        (1,1,3,3,3,3),(1,1,3,3,3,4),(1,2,2,2,3,4),(1,2,2,2,4,4),(1,2,2,3,3,3),(1,2,3,3,3,4),
        (1,3,3,3,3,4),(1,3,3,3,4,4),(1,3,3,4,4,4),(2,2,2,2,2,4),(2,2,2,2,4,4),(2,2,2,3,4,4),
        (2,2,2,4,4,4),(2,2,3,4,4,4),(2,3,3,3,4,4),(2,3,3,4,4,4),(3,3,3,3,3,4),(4,4,4,4,4,4)}
    # (1,1,1,1,2,3) has KL solution
    tmp0 = {tuple(y) for y in x0}
    assert tmp0==x1

    ## empty but too slow
    #x0 = numqi.qec.search_veca_C_group(n=7, m=19)


def test_search_veca_BD_group():
    x0 = numqi.qec.search_veca_BD_group(n=7, m=17, k=2, tag_print=False)
    x1 = {(3, 3, 4, 5, 5, 6, 7), (2, 2, 3, 4, 5, 8, 9), (1, 2, 4, 4, 6, 7, 9)}
    assert {tuple(y) for y in x0}==x1

    x0 = numqi.qec.search_veca_BD_group(n=7, m=18, k=2, tag_print=False)
    x1 = {(2, 3, 4, 5, 6, 7, 8)}
    assert {tuple(y) for y in x0}==x1

    x0 = numqi.qec.search_veca_BD_group(n=6, m=5, k=None, tag_print=False)
    x1 = {(0,1,1,2,2,3),(0,1,3,3,3,4),(0,2,2,2,4,4),(1,1,1,1,1,4),(1,1,1,1,2,3),(1,1,1,2,2,2),
          (1,1,2,2,4,4),(1,1,2,3,3,4),(1,1,3,3,3,3),(1,2,2,2,3,4),(1,2,2,3,3,3),(1,3,3,4,4,4),
          (2,2,2,2,2,4),(2,2,3,4,4,4),(2,3,3,3,4,4),(3,3,3,3,3,4),(4,4,4,4,4,4)}
    assert {tuple(y) for y in x0}==x1

    x0 = numqi.qec.search_veca_BD_group(n=7, m=18, k=None, tag_print=False)
    x1 = {(2,3,4,5,6,7,8),(2,3,4,8,11,12,13),(2,3,5,7,10,12,14),(2,4,5,6,10,11,15),
          (2,7,8,12,13,14,15),(3,6,8,11,13,14,16),(4,5,8,11,12,15,16),(4,6,7,10,13,15,16)}
    assert {tuple(y) for y in x0}==x1

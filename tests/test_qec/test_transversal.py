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


def test_ReedMuller_15_1_3_transversalT():
    # import numft
    # import stim
    # code = numft.css.TetrahedralColorCode()
    # tmp0 = np.concat([code.hx,0*code.hx], axis=1)
    # tmp0 = [stim.PauliString(x) for x in (code.hx_str + code.hz_str + code.lz_str)]
    # circ = stim.Tableau.from_stabilizers(tmp0).to_circuit()
    # circ_str = circ
    circ_str = ['H 0 1', 'CX 0 1 0 4 0 5 0 6 0 14', 'H 2 3',
            ('CX 1 2 1 3 1 4 1 5 1 7 1 8 1 14 2 4 2 7 2 8 2 9 2 11 4 3 3 4 4 3 3 6 3 11 3 12 4 5 4 7 4 10 '
            '4 13 7 5 5 7 7 5 5 9 5 10 6 12 7 6 7 12 7 13 14 8 8 14 14 8 8 14 14 9 9 14 14 9 9 11 9 14 11 '
            '10 10 11 11 10 10 12 14 11 11 14 14 11 11 14 14 12 12 14 14 12 12 13 14 13 13 14 14 13')]
    hx_str = 'XXXXXXXXIIIIIII IXXIXXIIXXIXXII IIXXIXXIIXXXIXI IIIIXXXXIIIXXXX'.split(' ')
    hz_str = ('ZZZZIIIIIIIIIII ZZIIZIIZIIIIIII ZIIZIIZZIIIIIII IZZIZZIIIIIIIII IZZIIIIIZZIIIII IIZIIZIIIZIZIII '
                    'IIIIZZIIIIIZZII IIZZIIIIIZZIIII IIIZIIZIIIZIIZI IIIIZIIZIIIIZIZ').split(' ')
    lx_str = ['XXXXIIIIXXXIIII']
    lz_str = ['ZZIIIIIIZIIIIII']
    circ_state = numqi.sim.Circuit()
    n = 15
    for x0 in circ_str:
        x0 = x0.split(' ')
        x0 = x0[:1] + [int(x) for x in x0[1:]]
        if x0[0]=='H':
            for y in x0[1:]:
                circ_state.H(y)
        elif x0[0]=='CX':
            for y0,y1 in zip(x0[1::2], x0[2::2]):
                circ_state.cnot(y0, y1)
        else:
            raise NotImplementedError
    q0 = np.zeros((2**n), dtype=np.complex128)
    q0[0] = 1
    logical0 = circ_state.apply_state(q0)
    q0[0] = 0
    q0[1] = 1
    logical1 = circ_state.apply_state(q0)
    code_state = np.stack([logical0,logical1], axis=0) #logical basis for CSS code are real

    pauli_str,pauli = numqi.qec.make_pauli_error_list_sparse(n, distance=3, kind='scipy-csr01')
    z0 = code_state.conj() @ (pauli @ code_state.T).reshape(-1, 2**n, 2)
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0] - z0[:,1,1]).max() < 1e-10
    assert np.abs(z0[:,0,0].imag).max() < 1e-10
    assert np.abs(z0).max() < 1e-10 #[[15,1,3]] non-degenerate code

    for x in hx_str+hz_str:
        tmp0 = code_state.conj() @ (numqi.qec.hf_pauli(x,tag_csr=True) @ code_state.T)
        assert np.abs(tmp0-np.eye(2)).max() < 1e-10
    assert np.abs(code_state.conj() @ (numqi.qec.hf_pauli(lz_str[0],tag_csr=True) @ code_state.T) - np.array([[1,0],[0,-1]])).max() < 1e-10
    assert np.abs(code_state.conj() @ (numqi.qec.hf_pauli(lx_str[0],tag_csr=True) @ code_state.T) - np.array([[0,1],[1,0]])).max() < 1e-10

    Tdag_diag = np.array([1,np.exp(-1j*np.pi/4)], dtype=np.complex128)
    Tdag_diag15 = Tdag_diag
    for _ in range(14):
        Tdag_diag15 = (Tdag_diag15.reshape(-1,1)*Tdag_diag).reshape(-1)
    assert np.abs(code_state.conj() @ (code_state * Tdag_diag15).T - np.array([[1,0],[0,np.exp(1j*np.pi/4)]])).max() < 1e-10

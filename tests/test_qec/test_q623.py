import functools
import numpy as np
import numqi

np_rng = np.random.default_rng()
hf_kron = lambda *x: functools.reduce(np.kron, x)


def test_get_transversal_gate623():
    # 623-SO5
    a,b = np_rng.normal(size=2)
    vece = np.array([a,b,0,0,0])/np.sqrt(a*a+b*b)
    # vece = np.array([1,0,0,0,0])
    code = numqi.qec.q623.get_SO5_code(vece)
    logical_list = numqi.qec.get_transversal_group(code, num_round=20, tag_print=False)
    assert len(logical_list)==8 #BD4
    # info = numqi.qec.get_transversal_group_info([x[0] for x in logical_list])

    a,b,c = np_rng.normal(size=3)
    vece = np.array([a,b,c,0,0])/np.sqrt(a*a+b*b+c*c)
    code = numqi.qec.q623.get_SO5_code(vece)
    logical_list = numqi.qec.get_transversal_group(code, num_round=20, tag_print=False)
    assert len(logical_list)==4, f'{a},{b},{c}' #C4

    a,b,c,d = np_rng.normal(size=4)
    vece = np.array([a,b,c,d,0])/np.sqrt(a*a+b*b+c*c+d*d)
    code = numqi.qec.q623.get_SO5_code(vece)
    logical_list = numqi.qec.get_transversal_group(code, num_round=20, tag_print=True)
    assert len(logical_list)==2, f'a,b,c,d={[a,b,c,d]}' #C2

    # tmp0 = np_rng.normal(size=5)
    # vece = tmp0/np.linalg.norm(tmp0)
    # code = numqi.qec.q623.get_SO5_code(vece)
    # logical_list = numqi.qec.get_transversal_group(code, num_round=20, tag_print=True)
    # assert len(logical_list)==2 #C2

    # ((6,2,3)) stab
    code,info = numqi.qec.get_code_subspace('623stab')
    logical_list = numqi.qec.get_transversal_group(code, num_round=40, tag_print=True)
    assert len(logical_list)==8 #BD4

    # # ((6,2,3)) from ((5,2,3))
    # code523,_ = numqi.qec.get_code_subspace('523')
    # tmp0 = numqi.random.rand_haar_unitary(4)
    # tmp1 = np.stack([code523,code523*0], axis=2).reshape(2,-1)
    # code623 = np.einsum(tmp1.reshape(-1,4), [0,1], tmp0, [3,1], [0,3], optimize=True).reshape(2,-1)
    # logical_list = numqi.qec.get_transversal_group(code623, num_round=20, tag_print=True)
    # assert len(logical_list)==8 #BD4


def test_623SO5_transversal_group_BD4():
    tmp0 = np_rng.normal(size=(2))
    code,info = numqi.qec.q623.get_SO5_code_with_transversal_gate(tmp0/np.linalg.norm(tmp0))
    assert np.abs(code.conj() @ hf_kron(*info['su2X']) @ code.T - numqi.gate.X).max() < 1e-10
    assert np.abs(code.conj() @ hf_kron(*info['su2Z']) @ code.T - numqi.gate.Z).max() < 1e-10

def test_623SO5_transversal_group_C4():
    tmp0 = np_rng.normal(size=3)
    code,info = numqi.qec.q623.get_SO5_code_with_transversal_gate(tmp0/np.linalg.norm(tmp0))
    assert np.abs(code.conj() @ hf_kron(*info['su2']) @ code.T - numqi.gate.Z).max() < 1e-10


def test_623stab_code():
    op_list = numqi.qec.make_pauli_error_list_sparse(num_qubit=6, distance=3, kind='scipy-csr01')[1]
    code,_ = numqi.qec.get_code_subspace('623stab')
    z0 = code.conj() @ (op_list @ code.T).reshape(-1, 2**6, 2)
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,0,0] - z0[:,1,1]).max() < 1e-10
    tmp1 = np.zeros(153, dtype=np.float64)
    tmp1[143] = 1
    assert np.abs(z0[:,0,0]-tmp1).max() < 1e-12
    weightA, weightB = numqi.qec.get_weight_enumerator(code, use_circuit=True, tagB=True)
    weightA_ = np.array([1,0,1,0,11,16,3])
    weightB_ = np.array([1,0,1,24,35,40,27])
    assert np.abs(weightA-weightA_).max() < 1e-12
    assert np.abs(weightB-weightB_).max() < 1e-12


def test_get_SO5_code_quantum_weight_enumerator():
    for _ in range(5):
        vece = numqi.random.rand_haar_state(5, tag_complex=False)
        code,info = numqi.qec.q623.get_SO5_code(vece, return_info=True)
        error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(6, distance=3, kind='scipy-csr01')
        z0 = code.conj() @ (error_scipy @ code.T).reshape(-1,64,2)
        assert np.abs(z0[:,0,1]).max() < 1e-10
        assert np.abs(z0[:,1,0]).max() < 1e-10
        assert np.abs(z0[:,0,0]-z0[:,1,1]).max() < 1e-10
        assert np.abs(z0[:,0,0].real - info['lambda_ai']).max() < 1e-10
        weightA,weightB = numqi.qec.get_weight_enumerator(code)
        assert np.abs(weightA-info['qweA']).max() < 1e-10
        assert np.abs(weightB-info['qweB']).max() < 1e-10

def test_get_623C10():
    code,info = numqi.qec.q623.get_C10(return_info=True)
    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(6, distance=3, kind='scipy-csr01')
    z0 = code.conj() @ (error_scipy @ code.T).reshape(-1,64,2)
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0]-z0[:,1,1]).max() < 1e-10
    qweA,qweB = numqi.qec.get_weight_enumerator(code)
    assert np.abs(qweA - info['qweA']).max() < 1e-10
    assert np.abs(qweB - info['qweB']).max() < 1e-10
    assert np.abs(numqi.gate.rz(-2*np.pi/5) - code.conj() @ hf_kron(*info['su2']) @ code.T).max() < 1e-10

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
    assert len(logical_list)==8 #BD_2
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
    # assert len(logical_list)==8 #BD8


def test_623SO5_transversal_group_BD4():
    tmp0 = np_rng.normal(size=(2))
    r,s = tmp0/np.linalg.norm(tmp0)
    matO = 0.5*np.array([[-s,s,-s,s,2*r], [r,-r,r,-r,2*s], [1,1,1,1,0], [1,-1,-1,1,0], [1,1,-1,-1,0]])
    veca,vecb,vecc,vecd,vece = matO.T/2
    basis = numqi.qec.q623.get_SO5_code_basis()
    coeff0 = np.concatenate([veca+1j*vecb, vecc+1j*vecd], axis=0)
    coeff1 = np.concatenate([coeff0[5:].conj(), -coeff0[:5].conj()], axis=0)
    code = np.stack([coeff0, coeff1], axis=0) @ basis

    basisS = basis.reshape(10,2,32)[:5,0]
    tmp0 = np.array([[-s,0], [r,0], [1j,0], [0,1], [0,1j]]) @ numqi.gate.H
    q0 = (tmp0.T @ basisS).reshape(-1) / (2*np.exp(1j*np.pi/4))
    assert np.abs(q0 - code[0]).max() < 1e-10
    tmp0 = np.array([[0,-1j*s], [0,1j*r], [0,1], [-1j,0], [-1,0]]) @ numqi.gate.H
    q1 = (tmp0.T @ basisS).reshape(-1) / (2*np.exp(1j*np.pi/4))
    assert np.abs(q1 - code[1]).max() < 1e-10

    X,Y,Z,I = numqi.gate.X, numqi.gate.Y, numqi.gate.Z, numqi.gate.I
    s3 = np.sqrt(3)
    tmp0 = np.stack([2*Y, s3*X+Y, X-s3*Y, s3*X+Y, X-s3*Y, X-s3*Y], axis=0) * 0.5j
    logicalX = code.conj() @ hf_kron(*tmp0) @ code.T
    assert np.abs(logicalX - X).max() < 1e-10
    tmp0 = np.stack([1j*X, 1j*Z, 1j*Z, I, I, I], axis=0)
    logicalZ = code.conj() @ hf_kron(*tmp0) @ code.T
    assert np.abs(1j*logicalZ - Z).max() < 1e-10


def test_623SO5_transversal_group_C4():
    tmp0 = np_rng.normal(size=3)
    r,s,t = tmp0/np.linalg.norm(tmp0)
    alpha = np.arccos(t/np.sqrt(t*t+1)) + np.arctan(s/r)
    beta = -np.arccos(t/np.sqrt(t*t+1)) + np.arctan(s/r)
    a1 = np.sqrt(t*t+1)*np.cos(alpha)
    a2 = np.sqrt(t*t+1)*np.sin(alpha)
    b1 = np.sqrt(t*t+1)*np.cos(beta)
    b2 = np.sqrt(t*t+1)*np.sin(beta)
    a3 = -(a1*r + a2*s)/t
    matO = 0.5*np.array([[a1,b1,a1,b1,2*r], [a2,b2,a2,b2,2*s], [a3,a3,a3,a3,2*t], [1,-1,-1,1,0], [1,1,-1,-1,0]]).T
    assert np.abs(matO@matO.T - np.eye(5)).max() < 1e-10

    veca,vecb,vecc,vecd,vece = matO/2
    basis = numqi.qec.q623.get_SO5_code_basis()
    coeff0 = np.concatenate([veca+1j*vecb, vecc+1j*vecd], axis=0)
    coeff1 = np.concatenate([coeff0[5:].conj(), -coeff0[:5].conj()], axis=0)
    code = np.stack([coeff0, coeff1], axis=0) @ basis

    X,Y,Z,I = numqi.gate.X, numqi.gate.Y, numqi.gate.Z, numqi.gate.I
    tmp0 = np.stack([1j*X, 1j*Z, 1j*Z, I, I, I], axis=0)
    logicalZ = code.conj() @ hf_kron(*tmp0) @ code.T
    assert np.abs(1j*logicalZ - Z).max() < 1e-10


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

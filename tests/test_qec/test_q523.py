import functools
import numpy as np
import scipy.linalg
import numqi

np_rng = np.random.default_rng()
hf_kron = lambda *x: functools.reduce(np.kron, x)


def test_get_code_subspace_523():
    code,info = numqi.qec.get_code_subspace('523')
    # https://arxiv.org/abs/quant-ph/9610040
    weightA, weightB = numqi.qec.get_weight_enumerator(code, use_circuit=True)
    weightA_ = np.array([1,0,0,0,15,0])
    weightB_ = np.array([1,0,0,30,15,18])
    assert np.abs(weightA-weightA_).max() < 1e-10
    assert np.abs(weightB-weightB_).max() < 1e-10

    # https://arxiv.org/abs/2204.03560 (eq132)
    op_list = numqi.qec.make_pauli_error_list_sparse(num_qubit=6, distance=3, kind='numpy')[1]
    code623 = np.stack([code,code*0], axis=2).reshape(2,-1)
    # weightA = [1,1,0,0,15,15,0]; weightB = [1,1,0,30,45,33,18]
    matU = numqi.random.rand_haar_unitary(2)
    tmp0 = scipy.linalg.block_diag(np.eye(2), matU)
    code623a = np.einsum(code623.reshape(-1,4), [0,1], tmp0, [3,1], [0,3], optimize=True).reshape(2,-1)
    tmp1 = code623.reshape(-1, 4)
    tmp2 = np.concatenate([tmp1[:,:2], tmp1[:,2:] @ matU.T], axis=1).reshape(2, -1)
    assert np.abs(tmp2 - code623a).max() < 1e-12
    z0 = code623a.conj() @ (op_list @ code623a.T)
    assert np.abs(z0[:,0,1]).max() < 1e-12
    assert np.abs(z0[:,0,0] - z0[:,1,1]).max() < 1e-12
    assert abs(np.linalg.norm(z0[:,0,0].real)-1) < 1e-10


def test_get_transversal_group_523():
    code,info = numqi.qec.get_code_subspace('523')
    logical_list = numqi.qec.get_transversal_group(code, num_round=30) #30 should be enough
    assert len(logical_list)==24 #2T
    tmp0 = [-numqi.gate.X]*5
    np1 = code.conj() @ hf_kron(*tmp0) @ code.T
    assert np.abs(np1 - numqi.gate.X).max() < 1e-10
    tmp0 = [numqi.gate.Z]*5
    np1 = code.conj() @ hf_kron(*tmp0) @ code.T
    assert np.abs(np1 - numqi.gate.Z).max() < 1e-10
    F = numqi.gate.H @ numqi.gate.rz(-np.pi/2)
    tmp0 = [numqi.gate.Z @ numqi.gate.X @ F]*5 #F
    np1 = code.conj() @ hf_kron(*tmp0) @ code.T
    assert np.abs(np1 - F).max() < 1e-10

    code1 = np.stack([code,code*0], axis=2).reshape(2,-1)
    code2 = (code1.reshape(-1,4) @ numqi.random.rand_haar_unitary(4)).reshape(2,-1)
    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(6, distance=3, kind='scipy-csr01')
    z0 = code2.conj() @ (error_scipy @ code2.T).reshape(-1,64,2)
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0].imag).max() < 1e-10
    assert np.abs(z0[:,0,0]-z0[:,1,1]).max() < 1e-10
    logical_list = numqi.qec.get_transversal_group(code2, num_round=30) #30 should be enough
    assert len(logical_list)==8 #BD4

    ## slow
    # code,info = numqi.qec.get_code_subspace('723cyclic', lambda2=4.875)
    # logical_list = numqi.qec.get_transversal_group(code, num_round=60)
    # assert len(logical_list)==24 #2T

    # code,info = numqi.qec.get_code_subspace('steane')
    # logical_list = numqi.qec.get_transversal_group(code, num_round=60)
    # assert len(logical_list)==48 #2O


def test_523u4_623_2T():
    code523,_ = numqi.qec.get_code_subspace('523')
    theta = np_rng.uniform(0, np.pi/4)
    a = np.sqrt((3-np.sqrt(3))/12)
    b = np.sqrt((3+np.sqrt(3))/6)
    u2 = 1j*(a*numqi.gate.X + a*numqi.gate.Y - b*numqi.gate.Z)
    u4 = numqi.gate.CZ @ np.kron(u2, numqi.gate.I)
    tmp0 = np.array([np.cos(theta), np.sin(theta)]) #numqi.gate.ry(theta*2)
    code = ((code523.reshape(-1,1) * tmp0).reshape(-1, 4) @ u4.T).reshape(2,-1)
    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(6, distance=3, kind='scipy-csr01')
    z0 = code.conj() @ (error_scipy @ code.T).reshape(-1,64,2)
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0]-z0[:,1,1]).max() < 1e-10
    A1 = np.linalg.norm(z0[:18,0,0].real)**2
    A2 = np.linalg.norm(z0[18:,0,0].real)**2
    assert abs(np.cos(2*theta)**2 - A1) < 1e-10
    assert abs(np.sin(2*theta)**2 - A2) < 1e-10
    # A1=0: [1,0,1,0,11,16,3]
    # numqi.qec.get_code_subspace('623-SO5', vece=np.array([1,0,0,0,0]))[1]['qweA']

    # transversal X, Z, F
    I,X,Y,Z = numqi.gate.I, numqi.gate.X, numqi.gate.Y, numqi.gate.Z
    F = numqi.gate.H @ numqi.gate.rz(-np.pi/2)
    tmp0 = [X,I,Y,Y,I,I]
    np1 = code.conj() @ hf_kron(*tmp0) @ code.T
    assert np.abs(np1 - numqi.gate.X).max() < 1e-10
    tmp0 = [-Y,Z,Y,I,I,I]
    np1 = code.conj() @ hf_kron(*tmp0) @ code.T
    assert np.abs(np1 - numqi.gate.Z).max() < 1e-10
    tmp0 = [1j*Y@F, Y@F, Y@F, Y@F, numqi.gate.rz(2*np.pi/3), I]
    np1 = code.conj() @ hf_kron(*tmp0) @ code.T
    assert np.abs(np1 - F).max() < 1e-10


def test_523u4_623_BD4():
    code523,_ = numqi.qec.get_code_subspace('523')
    theta = np_rng.uniform(0, np.pi/4)
    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(6, distance=3, kind='scipy-csr01')
    tmp0 = np.array([np.cos(theta), np.sin(theta)])
    code = ((code523.reshape(-1,1) * tmp0).reshape(-1, 4) @ numqi.gate.CZ.T).reshape(2,-1)
    z0 = code.conj() @ (error_scipy @ code.T).reshape(-1,64,2)
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0]-z0[:,1,1]).max() < 1e-10
    A1 = np.linalg.norm(z0[:18,0,0].real)**2
    A2 = np.linalg.norm(z0[18:,0,0].real)**2
    # TODO formula for higher weight enumerator
    assert abs(np.cos(2*theta)**2 - A1) < 1e-10
    assert abs(np.sin(2*theta)**2 - A2) < 1e-10
    # logical_list = numqi.qec.get_transversal_group(code, num_round=40, tag_print=True)
    # info = numqi.qec.get_transversal_group_info([x[0] for x in logical_list])

    I,X,Y,Z = numqi.gate.I, numqi.gate.X, numqi.gate.Y, numqi.gate.Z
    tmp0 = [X,I,Y,Y,I,I]
    np1 = code.conj() @ hf_kron(*tmp0) @ code.T
    assert np.abs(np1 - numqi.gate.X).max() < 1e-10
    tmp0 = [-Y,Z,Y,I,I,I]
    np1 = code.conj() @ hf_kron(*tmp0) @ code.T
    assert np.abs(np1 - numqi.gate.Z).max() < 1e-10


def test_code523_u4_weight_enumerator():
    code523,_ = numqi.qec.get_code_subspace('523')
    z0 = []
    for _ in range(10):
        su4 = numqi.random.rand_haar_unitary(4)
        code = (np.stack([code523,code523*0], axis=2).reshape(-1,4) @ su4.T).reshape(2,-1)
        z0.append(numqi.qec.get_weight_enumerator(code, tagB=False))
    z0 = np.stack(z0)
    assert np.abs(z0[:,0]-1).max() < 1e-10
    assert np.abs(z0[:,1]+z0[:,2]-1).max() < 1e-10
    assert np.abs(z0[:,3]).max() < 1e-10
    assert np.abs(4*z0[:,1] - z0[:,4] + 11).max() < 1e-10
    assert np.abs(z0[:,1] + z0[:,5] - 16).max() < 1e-10
    assert np.abs(-3*z0[:,1] - z0[:,6] + 3).max() < 1e-10

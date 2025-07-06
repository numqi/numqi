import itertools
import functools
import numpy as np
import numqi

np_rng = np.random.default_rng()
hf_kron = lambda *x: functools.reduce(np.kron, x)

def test_723bare_code():
    code,info = numqi.qec.get_code_subspace('723bare')
    op_list = numqi.qec.make_pauli_error_list_sparse(num_qubit=7, distance=3, kind='numpy')[1]
    z0 = code.conj() @ (op_list @ code.T)
    assert np.abs(z0[:,0,1]).max() < 1e-12
    assert np.abs(z0[:,0,0] - z0[:,1,1]).max() < 1e-12
    assert abs(np.linalg.norm(z0[:,0,0].real)**2-5) < 1e-10
    # "XXIIIII XIIIXII IXIIXII IIXIIXI IIIXIIX" is 1


def test_steane_code():
    code,info = numqi.qec.get_code_subspace('steane')
    op_list = numqi.qec.make_pauli_error_list_sparse(num_qubit=7, distance=3, kind='scipy-csr01')[1]
    z0 = code.conj() @ (op_list @ code.T).reshape(-1, 2**7, 2)
    assert np.abs(z0).max() < 1e-10


def test_get_723_cyclic_code():
    error_str_list,op_list = numqi.qec.make_pauli_error_list_sparse(num_qubit=7, distance=3, kind='scipy-csr01')
    lambda2 = np_rng.uniform(0, 7)
    for sign in ['++', '+-', '-+', '--']:
        coeff,lambda_ai_dict,basis = numqi.qec.q723.get_cyclic_code(lambda2, sign=sign)
        q0 = coeff @ basis
        z0 = q0.conj() @ (op_list @ q0.T).reshape(-1, 2**7, 2)
        assert np.abs(z0.imag).max() < 1e-12
        assert np.abs(z0[:,0,1]).max() < 1e-12
        assert np.abs(z0[:,1,0]).max() < 1e-12
        assert np.abs(z0[:,0,0] - z0[:,1,1]).max() < 1e-12
        z0 = z0[:,0,0].real
        z1 = np.array([lambda_ai_dict.get(x,0) for x in error_str_list])
        assert np.abs(z0-z1).max() < 1e-12


def test_723_cyclic_code_weight_enumerator():
    wt_to_pauli_dict = {x:numqi.qec.get_pauli_with_weight_sparse(7,x)[1] for x in range(1,8)} #slow 11 seconds
    for _ in range(5):
        lambda2 = np_rng.uniform(0, 7)
        code,info = numqi.qec.get_code_subspace('723cyclic', lambda2=lambda2)
        weightA,weightB = numqi.qec.get_weight_enumerator(code, wt_to_pauli_dict=wt_to_pauli_dict)
        assert np.abs(weightA-info['qweA']).max() < 1e-10
        assert np.abs(weightB-info['qweB']).max() < 1e-10

    code523,info = numqi.qec.get_code_subspace('523')
    code723 = np.concatenate([code523.reshape(2,32,1), np.zeros((2,32,3))], axis=2).reshape(2,128)
    weightA,weightB = numqi.qec.get_weight_enumerator(code723, wt_to_pauli_dict=wt_to_pauli_dict)
    weightA_ = np.array([1,2,1,0,15,30,15,0])
    weightB_ = np.array([1,2,1,30,75,78,51,18])
    assert np.abs(weightA-weightA_).max() < 1e-10
    assert np.abs(weightB-weightB_).max() < 1e-10


def test_get_2I_lambda0():
    theta,phi = np_rng.uniform(0, 2*np.pi, size=2)
    sign = '+' if np_rng.random() > 0.5 else '-'
    code,info = numqi.qec.q723.get_2I_lambda0(theta, phi, sign, return_info=True)
    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(7, distance=3, kind='scipy-csr01')
    z0 = code.conj() @ ((error_scipy @ code.T).reshape(-1, 128, 2))
    assert np.abs(z0).max() < 1e-10 #non-degenerate
    assert np.abs(code.conj() @ numqi.qec.hf_pauli('X'*7) @ code.T - numqi.gate.X).max() < 1e-10
    assert np.abs(code.conj() @ hf_kron(*info['su2']) @ code.T - numqi.gate.rz(2*np.pi/5)).max() < 1e-10
    assert np.abs(code.conj() @ hf_kron(*info['su2R']) @ code.T - info['transR']).max() < 1e-10
    qweA,qweB = numqi.qec.get_weight_enumerator(code)
    assert np.abs(qweA - info['qweA']).max() < 1e-10
    assert np.abs(qweB - info['qweB']).max() < 1e-10

    # 2I but not KL
    x0 = np.array([0,1,1,0])/np.sqrt(2)
    subspace = np.stack([x0@info['basis0'], x0@info['basis1']], axis=0)
    assert np.abs(subspace.conj() @ hf_kron(*info['su2R']) @ subspace.T - info['transR']).max() < 1e-10
    z0 = subspace.conj() @ ((error_scipy @ subspace.T).reshape(-1, 128, 2))
    assert np.abs(z0[:,0,1]).max() > 0.1 #not satisfy KL condition


def test_get_2I_lambda075():
    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(7, distance=3, kind='scipy-csr01')
    for sign in itertools.product([-1,1], repeat=3):
        t = np_rng.uniform(-np.sqrt(5)/4, np.sqrt(5)/4)
        code,info = numqi.qec.q723.get_2I_lambda075(t, sign, return_info=True)
        z0 = code.conj() @ (error_scipy @ code.T).reshape(-1,2**7,2)
        assert np.abs(z0[:,0,1]).max() < 1e-10
        assert np.abs(z0[:,1,0]).max() < 1e-10
        assert np.abs(z0[:,0,0]-z0[:,1,1]).max() < 1e-10
        assert np.abs(z0[:,0,0].imag).max() < 1e-10
        assert np.abs(z0[:,0,0]-info['lambda_ai']).max() < 1e-10
        assert np.abs(code.conj() @ numqi.qec.hf_pauli('X'*7) @ code.T - numqi.gate.X).max() < 1e-10
        assert np.abs(code.conj() @ hf_kron(*info['su2']) @ code.T - numqi.gate.rz(-2*np.pi/5)).max() < 1e-10
        assert np.abs(code.conj() @ hf_kron(*info['su2R']) @ code.T - info['transR']).max() < 1e-10
        qweA,qweB = numqi.qec.get_weight_enumerator(code)
        assert np.abs(qweA - info['qweA']).max() < 1e-10
        assert np.abs(qweB - info['qweB']).max() < 1e-10


def test_get_BD12_veca1112222():
    sign = np_rng.integers(2, size=10)*2-1
    sign[6] = -sign[1]*sign[3]*sign[5]
    sign[8] = -sign[2]*sign[4]*sign[7]
    code,info = numqi.qec.q723.get_BD12_veca1112222(sign=sign, return_info=True)

    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(7, distance=3, kind='scipy-csr01')
    z0 = code.conj() @ ((error_scipy @ code.T).reshape(-1, 128, 2))
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0]-z0[:,1,1]).max() < 1e-10
    assert np.abs(z0[:,0,0].imag).max() < 1e-10
    qweA, qweB = numqi.qec.get_weight_enumerator(code, tagB=True)
    assert np.abs(qweA - info['qweA']).max() < 1e-10
    assert np.abs(qweB - info['qweB']).max() < 1e-10
    logicalU = code.conj() @ hf_kron(*info['su2']) @ code.T
    assert np.abs(numqi.gate.rz(-2*np.pi/6) - logicalU).max() < 1e-10


def test_get_BD12_veca0122233():
    a = np_rng.uniform(0, np.sqrt(2/3))
    sign = np_rng.integers(2, size=4)*2 - 1
    sign[3] = sign[0]*sign[1]*sign[2]

    code,info = numqi.qec.q723.get_BD12_veca0122233(a, sign=sign, return_info=True)
    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(7, distance=3, kind='scipy-csr01')
    z0 = code.conj() @ (error_scipy @ code.T).reshape(-1,128,2)
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0]-z0[:,1,1]).max() < 1e-10
    assert np.abs(z0[:,0,0].imag).max() < 1e-10
    assert np.abs(z0[:,0,0] - info['lambda_ai']).max() < 1e-10
    qweA,qweB = numqi.qec.get_weight_enumerator(code, tagB=True)
    assert np.abs(qweA - info['qweA']).max() < 1e-10
    assert np.abs(qweB - info['qweB']).max() < 1e-10
    logicalU = code.conj() @ hf_kron(*info['su2']) @ code.T
    assert np.abs(numqi.gate.rz(2*np.pi/6) - logicalU).max() < 1e-10


def test_get_BD14():
    a = np_rng.uniform(0, 1/np.sqrt(14))
    sign = np_rng.integers(2, size=5) * 2 -1
    code,info = numqi.qec.q723.get_BD14(a, sign, return_info=True)
    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(7, distance=3, kind='scipy-csr01')
    z0 = code.conj() @ ((error_scipy @ code.T).reshape(-1, 128, 2))
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0]-z0[:,1,1]).max() < 1e-10
    assert np.abs(z0[:,0,0].imag).max() < 1e-10
    assert np.abs(z0[:,0,0] - info['lambda_ai']).max() < 1e-10
    qweA, qweB = numqi.qec.get_weight_enumerator(code, tagB=True)
    assert np.abs(qweA - info['qweA']).max() < 1e-10
    assert np.abs(qweB - info['qweB']).max() < 1e-10
    logicalU = code.conj() @ hf_kron(*info['su2']) @ code.T
    assert np.abs(numqi.gate.rz(-2*np.pi/7) - logicalU).max() < 1e-10


def test_get_BD16_5theta():
    theta = np_rng.uniform(0, 2*np.pi, 5)
    code,info = numqi.qec.q723.get_BD16_5theta(theta, return_info=True)
    logicalU = code.conj() @ hf_kron(*info['su2Z']) @ code.T
    assert np.abs(logicalU-numqi.gate.rz(-np.pi/4)).max() < 1e-10
    logicalU = code.conj() @ hf_kron(*info['su2X']) @ code.T
    assert np.abs(logicalU-numqi.gate.X).max() < 1e-10

    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(7, distance=3, kind='scipy-csr01')
    z0 = code.conj() @ (error_scipy @ code.T).reshape(-1,2**7,2)
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0] - z0[:,1,1]).max() < 1e-10
    qweA,qweB = numqi.qec.get_weight_enumerator(code, tagB=True)
    assert np.abs(qweA - info['qweA']).max() < 1e-10
    assert np.abs(qweB - info['qweB']).max() < 1e-10

    ## theta=0, stabilizer
    code,info = numqi.qec.q723.get_BD16_5theta(np.zeros(5), return_info=True)
    swap = numqi.gate.Swap
    I = numqi.gate.I
    assert np.abs(code.conj() @ numqi.qec.hf_pauli('IIIIIXX') @ code.T - I).max() < 1e-10
    a = np_rng.uniform(0, 2*np.pi) #X6(a)X7(-a)
    tmp0 = hf_kron(np.eye(2**5), numqi.gate.rx(a), numqi.gate.rx(-a))
    assert np.abs(code.conj() @ tmp0 @ code.T - I).max() < 1e-10
    op = np.kron(swap, np.eye(32)) #S(1,2)
    assert np.abs(code.conj() @ op @ code.T - I).max() < 1e-10
    op = hf_kron(np.eye(2), swap, np.eye(16)) #S(2,3)
    assert np.abs(code.conj() @ op @ code.T - I).max() < 1e-10
    op = hf_kron(np.eye(8), swap, np.eye(4)) #S(4,5)
    assert np.abs(code.conj() @ op @ code.T - I).max() < 1e-10
    op = np.kron(np.eye(32), numqi.gate.Swap) #S(6,7)
    assert np.abs(code.conj() @ op @ code.T - I).max() < 1e-10


def test_get_BD16_veca1222233():
    theta = np_rng.uniform(0, 2*np.pi, size=2)
    sign = np_rng.integers(2, size=7)*2 - 1
    code,info = numqi.qec.q723.get_BD16_veca1222233(theta[0], theta[1], sign=sign, return_info=True)

    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(7, distance=3, kind='scipy-csr01')
    z0 = code.conj() @ (error_scipy @ code.T).reshape(-1,2**7,2)
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0] - z0[:,1,1]).max() < 1e-10
    qweA,qweB = numqi.qec.get_weight_enumerator(code, tagB=True)
    assert np.abs(qweA - info['qweA']).max() < 1e-10
    assert np.abs(qweB - info['qweB']).max() < 1e-10
    logicalU = code.conj() @ hf_kron(*info['su2']) @ code.T
    assert np.abs(numqi.gate.rz(-2*np.pi/8) - logicalU).max() < 1e-10


def test_get_BD16_degenerate():
    theta = np_rng.uniform(0, 2*np.pi)
    sign = np_rng.integers(2, size=8)*2-1
    sign[5] = -sign[0]*sign[1]*sign[4]
    code, info = numqi.qec.q723.get_BD16_degenerate(theta=theta, sign=sign, return_info=True)

    error_scipy = numqi.qec.make_pauli_error_list_sparse(7, distance=3, kind='scipy-csr01')[1]
    z0 = code.conj() @ (error_scipy @ code.T).reshape(-1,128,2)
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0]-z0[:,1,1]).max() < 1e-10
    qweA,qweB = numqi.qec.get_weight_enumerator(code, tagB=True)
    assert np.abs(qweA - info['qweA']).max() < 1e-10
    assert np.abs(qweB - info['qweB']).max() < 1e-10
    lambda_a = z0[:,0,0].real
    assert np.abs(lambda_a - info['lambda_a']).max() < 1e-10
    assert abs(np.linalg.norm(lambda_a)**2-2.25) < 1e-10
    lambda_ab = (np.insert(lambda_a, 0, 1)[numqi.qec.get_knill_laflamme_matrix_indexing_over_vector(num_qubit=7, distance=3)]).reshape(22,22)
    EVL = np.linalg.eigvalsh(lambda_ab)
    assert np.abs(EVL - info['lambda_ab_EVL']).max() < 1e-10
    assert abs(EVL[0]) < 1e-12

    su2 = info['su2']
    logicalU = code.conj() @ hf_kron(*su2) @ code.T
    assert np.abs(numqi.gate.rz(-2*np.pi/8) - logicalU).max() < 1e-10


def test_get_BD18():
    theta = np_rng.uniform(0, 2*np.pi)
    root = np_rng.choice([0,1])
    sign = np_rng.integers(2, size=8) * 2 -1
    code,info = numqi.qec.q723.get_BD18(theta, root, sign, return_info=True)
    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(7, distance=3, kind='scipy-csr01')
    z0 = code.conj() @ ((error_scipy @ code.T).reshape(-1, 128, 2))
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0]-z0[:,1,1]).max() < 1e-10
    assert np.abs(z0[:,0,0].imag).max() < 1e-10
    assert np.abs(z0[:,0,0].real-info['lambda_ai']).max() < 1e-10
    qweA,qweB = numqi.qec.get_weight_enumerator(code, tagB=True)
    assert np.abs(qweA-info['qweA']).max() < 1e-10
    assert np.abs(qweB-info['qweB']).max() < 1e-10
    logicalU = code.conj() @ hf_kron(*info['su2']) @ code.T
    assert np.abs(numqi.gate.rz(-2*np.pi/9) - logicalU).max() < 1e-10


def test_get_BD18_LP():
    code,info = numqi.qec.q723.get_BD18_LP(coeff2=None, sign=None, return_info=True)
    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(7, distance=3, kind='scipy-csr01')
    z0 = code.conj() @ (error_scipy @ code.T).reshape(-1,2**7,2)
    assert np.abs(z0[:,0,1]).max() < 1e-8 #cvxpy precision is not good
    assert np.abs(z0[:,1,0]).max() < 1e-8
    assert np.abs(z0[:,0,0] - z0[:,1,1]).max() < 1e-8
    logicalU = code.conj() @ hf_kron(*info['su2']) @ code.T
    assert np.abs(numqi.gate.rz(-2*np.pi/9) - logicalU).max() < 1e-10
    # qweA, qweB = numqi.qec.get_weight_enumerator(code, tagB=True)


def test_get_BD20():
    a = np_rng.uniform(0, np.sqrt(1/5))
    sign = np_rng.integers(2, size=5)*2 - 1
    code, info = numqi.qec.q723.get_BD20(a, sign, return_info=True)

    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(7, distance=3, kind='scipy-csr01')
    z0 = code.conj() @ ((error_scipy @ code.T).reshape(-1, 128, 2))
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0]-z0[:,1,1]).max() < 1e-10
    assert np.abs(z0[:,0,0].imag).max() < 1e-10
    assert np.abs(z0[:,0,0]-info['lambda_ai']).max() < 1e-10
    qweA, qweB = numqi.qec.get_weight_enumerator(code, tagB=True)
    assert np.abs(qweA - info['qweA']).max() < 1e-10
    assert np.abs(qweB - info['qweB']).max() < 1e-10
    logicalU = code.conj() @ hf_kron(*info['su2']) @ code.T
    assert np.abs(numqi.gate.rz(-2*np.pi/10) - logicalU).max() < 1e-10


def test_get_BD22():
    sign = np_rng.integers(2, size=5)*2 - 1
    a = np_rng.uniform(0, np.sqrt(3/22))
    code,info = numqi.qec.q723.get_BD22(a, sign=sign, return_info=True)

    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(7, distance=3, kind='scipy-csr01')
    z0 = code.conj() @ ((error_scipy @ code.T).reshape(-1, 128, 2))
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0]-z0[:,1,1]).max() < 1e-10
    assert np.abs(z0[:,0,0].imag).max() < 1e-10
    assert np.abs(z0[:,0,0]-info['lambda_ai']).max() < 1e-10
    qweA, qweB = numqi.qec.get_weight_enumerator(code, tagB=True)
    assert np.abs(qweA - info['qweA']).max() < 1e-10
    assert np.abs(qweB - info['qweB']).max() < 1e-10
    logicalU = code.conj() @ hf_kron(*info['su2']) @ code.T
    assert np.abs(numqi.gate.rz(-2*np.pi/11) - logicalU).max() < 1e-10


def test_get_BD24():
    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(7, distance=3, kind='scipy-csr01')
    for sign in '+-':
        code,info = numqi.qec.q723.get_BD24(sign=sign, return_info=True)
        z0 = code.conj() @ (error_scipy @ code.T).reshape(-1,128,2)
        assert np.abs(z0[:,0,1]).max() < 1e-10
        assert np.abs(z0[:,1,0]).max() < 1e-10
        assert np.abs(z0[:,0,0]-z0[:,1,1]).max() < 1e-10
        assert np.abs(z0[:,0,0].imag).max() < 1e-10
        z0 = z0[:,0,0].real
        qweA,qweB = numqi.qec.get_weight_enumerator(code, tagB=True)
        assert np.abs(qweA-info['qweA']).max() < 1e-10
        assert np.abs(qweB-info['qweB']).max() < 1e-10

        np0 = code.conj() @ hf_kron(*[numqi.gate.X]*7) @ code.T
        assert np.abs(numqi.gate.X - np0).max() < 1e-12
        np0 = code.conj() @ hf_kron(*info['su2']) @ code.T
        assert np.abs(numqi.gate.rz(np.pi/6) - np0).max() < 1e-12


def test_get_BD26():
    sign = np_rng.integers(2, size=5)*2 - 1
    a = np_rng.uniform(0, np.sqrt(2/13))
    code, info = numqi.qec.q723.get_BD26(a, sign=sign, return_info=True)
    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(7, distance=3, kind='scipy-csr01')
    z0 = code.conj() @ (error_scipy @ code.T).reshape(-1,128,2)
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0]-z0[:,1,1]).max() < 1e-10
    assert np.abs(z0[:,0,0] - info['lambda_ai']).max() < 1e-10
    qweA, qweB = numqi.qec.get_weight_enumerator(code, tagB=True)
    assert np.abs(qweA - info['qweA']).max() < 1e-10
    assert np.abs(qweB - info['qweB']).max() < 1e-10
    logicalU = code.conj() @ hf_kron(*info['su2']) @ code.T
    assert np.abs(numqi.gate.rz(-2*np.pi/13) - logicalU).max() < 1e-10


def test_get_BD28():
    a = np_rng.uniform(0, np.sqrt(1/14))
    sign = np_rng.integers(2, size=5)*2-1
    code,info = numqi.qec.q723.get_BD28(a, sign, return_info=True)

    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(7, distance=3, kind='scipy-csr01')
    z0 = code.conj() @ (error_scipy @ code.T).reshape(-1,128,2)
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0]-z0[:,1,1]).max() < 1e-10
    assert np.abs(z0[:,0,0]-info['lambda_ai']).max() < 1e-10
    qweA,qweB = numqi.qec.get_weight_enumerator(code, tagB=True)
    assert np.abs(qweA - info['qweA']).max() < 1e-10
    assert np.abs(qweB - info['qweB']).max() < 1e-10

    logicalU = code.conj() @ hf_kron(*info['su2']) @ code.T
    assert np.abs(numqi.gate.rz(-2*np.pi/14) - logicalU).max() < 1e-10


def test_get_BD30():
    a = np_rng.uniform(np.sqrt(1/15), np.sqrt(7/30))
    sign = np_rng.integers(2, size=5)*2 - 1
    code, info = numqi.qec.q723.get_BD30(a, sign=sign, return_info=True)
    error_scipy = numqi.qec.make_pauli_error_list_sparse(7, distance=3, kind='scipy-csr01')[1]
    z0 = code.conj() @ (error_scipy @ code.T).reshape(-1,128,2)
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0]-z0[:,1,1]).max() < 1e-10
    assert np.abs(z0[:,0,0]-info['lambda_ai']).max() < 1e-10
    # assert abs(np.linalg.norm(z0[:,0,0])**2 - info['A2']) < 1e-10
    qweA,qweB = numqi.qec.get_weight_enumerator(code, tagB=True)
    assert np.abs(qweA - info['qweA']).max() < 1e-10
    assert np.abs(qweB - info['qweB']).max() < 1e-10
    su2 = info['su2']
    logicalU = code.conj() @ hf_kron(*su2) @ code.T
    assert np.abs(numqi.gate.rz(-2*np.pi/15) - logicalU).max() < 1e-10


def test_get_723_BD32_code():
    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(7, distance=3, kind='scipy-csr01')
    a = np_rng.uniform(0, 1/np.sqrt(8))
    sign = np_rng.integers(2, size=9)*2 - 1
    code, info = numqi.qec.q723.get_BD32(a, sign, return_info=True)

    z0 = code.conj() @ ((error_scipy @ code.T).reshape(-1, 128, 2))
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0]-z0[:,1,1]).max() < 1e-10
    assert np.abs(z0[:,0,0].imag).max() < 1e-10
    assert np.abs(z0[:,0,0]-info['lambda_ai']).max() < 1e-10
    qweA,qweB = numqi.qec.get_weight_enumerator(code, tagB=True)
    assert np.abs(qweA - info['qweA']).max() < 1e-10
    assert np.abs(qweB - info['qweB']).max() < 1e-10

    np0 = code.conj() @ numqi.qec.hf_pauli('XXXXXXX') @ code.T
    assert np.abs(np0-numqi.gate.X).max() < 1e-10
    np0 = code.conj() @ hf_kron(*info['su2']) @ code.T
    assert np.abs(np0 - numqi.gate.rz(-2*np.pi/16)).max() < 1e-10


def test_get_BD34():
    sign = np_rng.integers(2, size=8)*2-1
    theta = np_rng.uniform(0, 2*np.pi)
    code, info = numqi.qec.q723.get_BD34(sign=sign, theta=theta, return_info=True)

    error_scipy = numqi.qec.make_pauli_error_list_sparse(7, distance=3, kind='scipy-csr01')[1]
    z0 = code.conj() @ (error_scipy @ code.T).reshape(-1,128,2)
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0]-z0[:,1,1]).max() < 1e-10
    assert abs(np.linalg.norm(z0[:,0,0].real)**2-831/289) < 1e-10

    qweA,qweB = numqi.qec.get_weight_enumerator(code, tagB=True)
    assert np.abs(qweA - info['qweA']).max() < 1e-10
    assert np.abs(qweB - info['qweB']).max() < 1e-10

    su2 = info['su2']
    logicalU = code.conj() @ hf_kron(*su2) @ code.T
    assert np.abs(numqi.gate.rz(-2*np.pi/17) - logicalU).max() < 1e-10

    tmp1 = np.stack([info['basis0'],info['basis1']], axis=0)
    z1 = np.einsum(tmp1, [0,1,2], tmp1, [3,4,5], hf_kron(*su2), [2,5], [0,3,1,4], optimize=True)
    assert np.abs(numqi.gate.rz(-2*np.pi/17).reshape(2,2,1,1) * np.eye(8) - z1).max() < 1e-10


def test_get_BD36():
    theta = np_rng.uniform(0, 2*np.pi)
    sign = np_rng.choice([1,-1], size=7, replace=True)
    code,info = numqi.qec.q723.get_BD36(sign=sign, theta=theta, return_info=True)
    error_scipy = numqi.qec.make_pauli_error_list_sparse(7, distance=3, kind='scipy-csr01')[1]
    z0 = code.conj() @ (error_scipy @ code.T).reshape(-1,128,2)
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0]-z0[:,1,1]).max() < 1e-10
    assert abs(np.linalg.norm(z0[:,0,0].real)**2-161/81) < 1e-10
    qweA,qweB = numqi.qec.get_weight_enumerator(code, tagB=True)
    assert np.abs(qweA-info['qweA']).max() < 1e-10
    assert np.abs(qweB-info['qweB']).max() < 1e-10

    su2 = info['su2']
    logicalU = code.conj() @ hf_kron(*su2) @ code.T
    assert np.abs(numqi.gate.rz(-2*np.pi/18) - logicalU).max() < 1e-10


def test_get_2O_X5():
    sign = np_rng.integers(2, size=6)*2 - 1
    sign[1] = -sign[0]
    sign[4] = -sign[0]
    sign[5] = sign[3]
    code,info = numqi.qec.q723.get_2O_X5(sign=sign, return_info=True)

    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(7, distance=3, kind='scipy-csr01')
    z0 = code.conj() @ (error_scipy @ code.T).reshape(-1,2**7,2)
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0] - z0[:,1,1]).max() < 1e-10
    qweA,qweB = numqi.qec.get_weight_enumerator(code, tagB=True)
    assert np.abs(qweA - info['qweA']).max() < 1e-10
    assert np.abs(qweB - info['qweB']).max() < 1e-10
    logicalU = code.conj() @ hf_kron(*info['su2X']) @ code.T
    assert np.abs(numqi.gate.X - logicalU).max() < 1e-10
    logicalU = code.conj() @ hf_kron(*info['su2YSY']) @ code.T
    # tmp0 = numqi.gate.ry(np.pi/4) @ numqi.gate.rz(np.pi/2) @ numqi.gate.ry(np.pi/4).T
    tmp0 = numqi.qec.get_su2_finite_subgroup_generator('2Ox')[1]
    assert np.abs(logicalU-tmp0).max() < 1e-10



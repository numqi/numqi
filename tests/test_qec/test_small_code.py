import numpy as np
import scipy.linalg

import numqi

np_rng = np.random.default_rng()


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
    tmp2 = np.concat([tmp1[:,:2], tmp1[:,2:] @ matU.T], axis=1).reshape(2, -1)
    assert np.abs(tmp2 - code623a).max() < 1e-12
    z0 = code623a.conj() @ (op_list @ code623a.T)
    assert np.abs(z0[:,0,1]).max() < 1e-12
    assert np.abs(z0[:,0,0] - z0[:,1,1]).max() < 1e-12
    assert abs(np.linalg.norm(z0[:,0,0].real)-1) < 1e-10


def test_723bare_code():
    code,info = numqi.qec.get_code_subspace('723bare')
    op_list = numqi.qec.make_pauli_error_list_sparse(num_qubit=7, distance=3, kind='numpy')[1]
    z0 = code.conj() @ (op_list @ code.T)
    assert np.abs(z0[:,0,1]).max() < 1e-12
    assert np.abs(z0[:,0,0] - z0[:,1,1]).max() < 1e-12
    assert abs(np.linalg.norm(z0[:,0,0].real)**2-5) < 1e-10
    # "XXIIIII XIIIXII IXIIXII IIXIIXI IIIXIIX" is 1


def test_723permutation_code():
    for sign in '+-':
        code,info = numqi.qec.get_code_subspace('723permutation', sign=sign)
        op_list = numqi.qec.make_pauli_error_list_sparse(num_qubit=7, distance=3, kind='scipy-csr01')[1]
        z0 = code.conj() @ (op_list @ code.T).reshape(-1, 2**7, 2)
        assert np.abs(z0[:,0,1]).max() < 1e-12
        tmp0 = np.array([2/3]*18 + [1] + [3]*3)
        ind0 = [21, 25, 29, 30, 34, 38, 39, 43, 47, 48, 52, 56, 57, 61, 65, 66, 70, 74, 75, 79, 83,
                84, 88, 92, 93, 97, 101, 102, 106, 110, 111, 115, 119, 120, 124, 128, 129, 133, 137,
                138, 142, 146, 147, 151, 155, 156, 160, 164, 165, 169, 173, 174, 178, 182, 183, 187,
                191, 192, 196, 200, 201, 205, 209]
        tmp0 = np.zeros(210, dtype=np.float64)
        tmp0[ind0] = 1/3
        assert np.abs(z0[:,0,0]-tmp0).max() < 1e-10


def test_442code():
    code,info = numqi.qec.get_code_subspace('442stab')
    op_list = numqi.qec.make_pauli_error_list_sparse(num_qubit=4, distance=2, kind='numpy')[1]
    z0 = code.conj() @ (op_list @ code.T)
    assert np.abs(z0).max() < 1e-12
    # weightA_ = np.array([1,0,0,0,3])
    # weightB_ = np.array([1,0,18,24,21])


def test_c4c6_concat_code():
    code442,_ = numqi.qec.get_code_subspace('442stab')
    code642,_ = numqi.qec.get_code_subspace('642stab')
    code_concat = np.einsum(code642.reshape(4,4,4,4), [0,1,2,3], code442, [1,4],
            code442, [2,5], code442, [3,6], [0,4,5,6], optimize=True).reshape(4,2**12)
    op_list = numqi.qec.make_pauli_error_list_sparse(num_qubit=12, distance=3, kind='scipy-csr01')[1]
    z0 = code_concat @ (op_list @ code_concat.T).reshape(-1, 2**12, 4)
    assert np.abs(z0).max() < 1e-12


def test_shor_code():
    code,_ = numqi.qec.get_code_subspace('shor')
    # code = rand_local_unitary(code)
    op_list = numqi.qec.make_pauli_error_list_sparse(num_qubit=9, distance=3, kind='scipy-csr01')[1]
    tmp0 = np.array([0]*6 + [1]*19 + [3]*3)
    tmp0 = np.zeros(9*3+9*4*9, dtype=np.float64)
    tmp0[[35, 44, 107, 224, 233, 269, 332, 341, 350]] = 1
    # ' '.join([error_str_list[x] for x in [35, 44, 107, 224, 233, 269, 332, 341, 350]])
    # ZZIIIIIII ZIZIIIIII IZZIIIIII IIIZZIIII IIIZIZIII IIIIZZIII IIIIIIZZI IIIIIIZIZ IIIIIIIZZ
    z0 = code.conj() @ (op_list @ code.T).reshape(-1, 2**9, 2)
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,0,0]-tmp0).max() < 1e-10


def test_883code():
    code,info = numqi.qec.get_code_subspace('883')
    op_list = numqi.qec.make_pauli_error_list_sparse(num_qubit=8, distance=3, kind='scipy-csr01')[1]
    z0 = code.conj() @ (op_list @ code.T).reshape(-1, 2**8, 8)
    assert np.abs(z0).max() < 1e-10


def test_steane_code():
    code,info = numqi.qec.get_code_subspace('steane')
    op_list = numqi.qec.make_pauli_error_list_sparse(num_qubit=7, distance=3, kind='scipy-csr01')[1]
    z0 = code.conj() @ (op_list @ code.T).reshape(-1, 2**7, 2)
    assert np.abs(z0).max() < 1e-10

def test_code642():
    code,_ = numqi.qec.get_code_subspace('642stab')
    op_list = numqi.qec.make_pauli_error_list_sparse(num_qubit=6, distance=2, kind='scipy-csr01')[1]
    z0 = code.conj() @ (op_list @ code.T).reshape(-1, 2**6, 4)
    assert np.abs(z0).max() < 1e-10


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


def get_MacWilliams_identity(weight, x, y):
    n = weight.shape[0]-1
    ret = sum((x**(n-i)) * (y**i) * weight[i] for i in range(n+1))
    return ret


def test_get_code_quantum_weight_enumerator():
    for key in ['523','shor','steane']:
        code,info = numqi.qec.get_code_subspace(key)
        weightA = info['qweA']
        weightB = info['qweB']
        dim = code.shape[0]
        npx = np_rng.uniform(-1, 1, 5)
        npy = np_rng.uniform(-1, 1, 5)
        z0 = get_MacWilliams_identity(weightB, npx, npy)
        z1 = dim * get_MacWilliams_identity(weightA, (npx+3*npy)/2, (npx-npy)/2)
        # https://arxiv.org/abs/2408.10323 eq(4)
        assert np.abs(z0-z1).max() < 1e-10


def test_get_623_SO5_code_quantum_weight_enumerator():
    wt_to_pauli_dict = {x:numqi.qec.get_pauli_with_weight_sparse(6,x)[1] for x in range(1,7)}
    for _ in range(5):
        vece = numqi.random.rand_haar_state(5, tag_complex=False)
        code,info = numqi.qec.get_code_subspace('623-SO5', vece=vece)
        weightA,weightB = numqi.qec.get_weight_enumerator(code, wt_to_pauli_dict=wt_to_pauli_dict)
        assert np.abs(weightA-info['qweA']).max() < 1e-12
        assert np.abs(weightB-info['qweB']).max() < 1e-12


def test_get_723_cyclic_code():
    error_str_list,op_list = numqi.qec.make_pauli_error_list_sparse(num_qubit=7, distance=3, kind='scipy-csr01')
    lambda2 = np_rng.uniform(0, 7)
    for sign in ['++', '+-', '-+', '--']:
        coeff,lambda_ai_dict,basis = numqi.qec._small_code.get_723_cyclic_code(lambda2, sign=sign)
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
    code723 = np.concat([code523.reshape(2,32,1), np.zeros((2,32,3))], axis=2).reshape(2,128)
    weightA,weightB = numqi.qec.get_weight_enumerator(code723, wt_to_pauli_dict=wt_to_pauli_dict)
    weightA_ = np.array([1,2,1,0,15,30,15,0])
    weightB_ = np.array([1,2,1,30,75,78,51,18])
    assert np.abs(weightA-weightA_).max() < 1e-10
    assert np.abs(weightB-weightB_).max() < 1e-10


def test_723permutation_code_local_unitary_equivalent():
    codep,info = numqi.qec.get_code_subspace('723permutation', sign='+')
    codem,info = numqi.qec.get_code_subspace('723permutation', sign='-')
    model = numqi.qec.QECCEqualModel(codep, codem)
    # theta0 = numqi.optimize.minimize(model, 'uniform', num_repeat=10, tol=1e-24).x
    theta0 = np.array([-0.9851071268903094, 0.9851071309835814, -0.9851071080929052, 0.8286922424445572, -0.8286922385150921,
            0.8286922425522644, 0.8286922410131242, -0.8286922420980595, 0.8286922443320521, 0.8286922422542674, -0.8286922398812964,
            0.8286922415662342, -0.9851071239082392, 0.9851071216442577, -0.9851071202221557, -0.9851071170253412, 0.9851071293328815,
            -0.9851071218120353, -0.9851071289119772, 0.9851071116144996, -0.985107125489987])
    numqi.optimize.set_model_flat_parameter(model, theta0)
    assert abs(model().item()) < 1e-12

import numpy as np

import numqi

np_rng = np.random.default_rng()


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


def test_code642():
    code,_ = numqi.qec.get_code_subspace('642stab')
    op_list = numqi.qec.make_pauli_error_list_sparse(num_qubit=6, distance=2, kind='scipy-csr01')[1]
    z0 = code.conj() @ (op_list @ code.T).reshape(-1, 2**6, 4)
    assert np.abs(z0).max() < 1e-10

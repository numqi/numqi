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


def test_get_code_subspace_color832():
    q0,info = numqi.qec.get_code_subspace('color832', return_info=True)
    for i in range(3):
        tmp0 = q0.conj() @ (numqi.qec.hf_pauli(info['lx_str'][i]) @ q0.T)
        tmp1 = ['I','I','I']
        tmp1[i] = 'X'
        assert np.abs(numqi.qec.hf_pauli(''.join(tmp1)) - tmp0).max() < 1e-10
        tmp0 = q0.conj() @ (numqi.qec.hf_pauli(info['lz_str'][i]) @ q0.T)
        tmp1[i] = 'Z'
        assert np.abs(numqi.qec.hf_pauli(''.join(tmp1)) - tmp0).max() < 1e-10
    for x in info['hz_str']+info['hx_str']:
        tmp0 = q0.conj() @ (numqi.qec.hf_pauli(x) @ q0.T)
        assert np.abs(tmp0 - np.eye(8)).max() < 1e-10

    pauli_str,pauli = numqi.qec.make_pauli_error_list_sparse(8, distance=2, kind='scipy-csr01')
    z0 = q0.conj() @ (pauli @ q0.T).reshape(-1, 2**8, q0.shape[0])
    np.abs(z0*(1-np.eye(z0.shape[1]))).max()
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0] - z0[:,1,1]).max() < 1e-10
    assert np.abs(z0[:,0,0].imag).max() < 1e-10
    assert np.abs(z0).max() < 1e-10 #[[15,1,3]] non-degenerate code

    T = np.array([1,np.exp(1j*np.pi/4)])
    Tdag = np.array([1,np.exp(-1j*np.pi/4)])
    import functools
    hf0 = lambda x,y: (x.reshape(-1,1)*y).reshape(-1)
    tmp0 = functools.reduce(hf0, [T,Tdag,Tdag,T,Tdag,T,T,Tdag])
    ccz = np.diag(np.array([1]*7+[-1]))
    assert np.abs((q0.conj() @ (q0*tmp0).T).real - ccz).max() < 1e-10

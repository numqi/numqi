import numpy as np

import numqi

np_rng = np.random.default_rng()
hf_rand = lambda *size: np_rng.normal(size=size)
hf_randc = lambda *size: np_rng.normal(size=size) + 1j*np_rng.normal(size=size)


def test_is_vector_linear_independent():
    N0 = 5
    N1 = 7
    for tag_complex in [True,False]:
        field = 'complex' if tag_complex else 'real'
        np0 = numqi.random.rand_unitary_matrix(N1, tag_complex=tag_complex)[:N0]
        assert numqi.matrix_space.is_vector_linear_independent(np0, field)

        np1 = (hf_randc if tag_complex else hf_rand)(N0+1,N0) @ np0
        assert not numqi.matrix_space.is_vector_linear_independent(np1, field)

    N0 = 4
    tmp0 = numqi.random.rand_unitary_matrix(2*N0, tag_complex=False)[:(N0+1)]
    np0 = tmp0[:,:N0] + 1j*tmp0[:,N0:]
    assert numqi.matrix_space.is_vector_linear_independent(np0, field='real')
    assert not numqi.matrix_space.is_vector_linear_independent(np0, field='complex')


def test_reduce_vector_space():
    N0 = 5
    N1 = 3
    N2 = 9
    np_rng = np.random.default_rng()

    # span_R(R^n)
    np0 = np_rng.normal(size=(N0, N1)) @ np_rng.normal(size=(N1, N2))
    np1 = np_rng.normal(size=N0) @ np0
    z0 = numqi.matrix_space.reduce_vector_space(np0, zero_eps=1e-10)
    assert np.abs(z0 @ z0.T - np.eye(z0.shape[0])).max() < 1e-10
    x,residual,rank,s = np.linalg.lstsq(z0.T, np1, rcond=None)
    assert abs(residual) < 1e-10
    assert np.abs(x.imag).max() < 1e-10

    # span_C(C^n)
    hf_randc = lambda *size: np_rng.normal(size=size) + 1j*np_rng.normal(size=size)
    np0 = hf_randc(N0, N1) @ hf_randc(N1, N2)
    np1 = hf_randc(N0) @ np0
    z0 = numqi.matrix_space.reduce_vector_space(np0, zero_eps=1e-10)
    assert np.abs(z0.conj() @ z0.T - np.eye(z0.shape[0])).max() < 1e-10
    x,residual,rank,s = np.linalg.lstsq(z0.T, np1, rcond=None)
    assert abs(residual) < 1e-10
    # x.imag generally should be not zero


def test_get_vector_orthogonal_basis():
    N0 = 5
    N1 = 3
    N2 = 9
    np_rng = np.random.default_rng()

    # span_R(R^n)
    np0 = np_rng.normal(size=(N0, N1)) @ np_rng.normal(size=(N1, N2))
    basis_orth = numqi.matrix_space.get_vector_orthogonal_basis(np0)
    tmp0 = np_rng.normal(size=(23, basis_orth.shape[0])) @ basis_orth
    assert np.abs(np0 @ tmp0.T).max() < 1e-10

    # span_C(C^n)
    hf_randc = lambda *size: np_rng.normal(size=size) + 1j*np_rng.normal(size=size)
    np0 = hf_randc(N0, N1) @ hf_randc(N1, N2)
    basis_orth = numqi.matrix_space.get_vector_orthogonal_basis(np0)
    tmp0 = hf_randc(23, basis_orth.shape[0]) @ basis_orth
    assert np.abs(np0.conj() @ tmp0.T).max() < 1e-10


def test_get_matrix_space_orthogonal_basis_R_T():
    N0 = 13
    N1 = 8
    m = 5
    # span_R(R_T^mm)
    tmp0 = np_rng.normal(size=(N1,m,m))
    tmp1 = (tmp0 + tmp0.transpose(0,2,1))/2
    matrix_space = (np_rng.normal(size=(N0,N1)) @ tmp1.reshape(N1,m*m)).reshape(N0, m, m)
    basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matrix_space, field='real')
    assert np.abs(basis-basis.transpose(0,2,1)).max() < 1e-10
    assert np.abs(basis_orth-basis_orth.transpose(0,2,1)).max() < 1e-10
    assert (basis.shape[0]+basis_orth.shape[0])==((m*m+m)//2)
    assert np.abs(matrix_space.reshape(-1,m*m) @ basis_orth.reshape(-1,m*m).T).max() < 1e-10


def test_get_matrix_space_orthogonal_basis_C_T():
    N0 = 13
    N1 = 8
    m = 5

    #span_C(R_T^mm)
    tmp0 = (np_rng.normal(size=(N0,N1)) @ np_rng.normal(size=(N1,m*m))).reshape(N0,m,m)
    matrix_space = (tmp0+tmp0.transpose(0,2,1))/2
    basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matrix_space, field='complex')
    for x in (basis,basis_orth):
        assert np.abs(x-x.transpose(0,2,1)).max() < 1e-10
    assert (basis.shape[0]+basis_orth.shape[0])==((m*m+m)//2)
    assert np.abs(matrix_space.reshape(-1,m*m) @ basis_orth.reshape(-1,m*m).T).max() < 1e-10

    # span_C(C_T^mm)
    tmp0 = (hf_randc(N0,N1) @ hf_randc(N1,m*m)).reshape(N0,m,m)
    matrix_space = (tmp0+tmp0.transpose(0,2,1))/2
    basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matrix_space, field='complex')
    for x in (basis,basis_orth):
        assert np.abs(x-x.transpose(0,2,1)).max() < 1e-10
    assert (basis.shape[0]+basis_orth.shape[0])==((m*m+m)//2)
    assert np.abs(matrix_space.reshape(-1,m*m) @ basis_orth.reshape(-1,m*m).T.conj()).max() < 1e-10


def test_get_matrix_space_orthogonal_basis_R():
    N0 = 13
    N1 = 8
    m = 5
    n = 7
    #span_R(R^mn)
    tmp0 = np_rng.normal(size=(N1,m,n))
    matrix_space = (np_rng.normal(size=(N0,N1)) @ tmp0.reshape(N1,m*n)).reshape(N0, m, n)
    basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matrix_space, field='real')
    assert (basis.shape[0]+basis_orth.shape[0])==(m*n)
    assert np.abs(matrix_space.reshape(-1,m*n) @ basis_orth.reshape(-1,m*n).T).max() < 1e-10


def test_get_matrix_space_orthogonal_basis_C():
    N0 = 13
    N1 = 8
    m = 5
    n = 7

    #span_C(R^mn)
    tmp0 = np_rng.normal(size=(N1,m,n))
    matrix_space = (hf_randc(N0,N1) @ tmp0.reshape(N1,m*n)).reshape(N0, m, n)
    basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matrix_space, field='complex')
    assert (basis.shape[0]+basis_orth.shape[0])==(m*n)
    assert np.abs(matrix_space.reshape(-1,m*n) @ basis_orth.reshape(-1,m*n).T).max() < 1e-10

    #span_C(C^mn)
    tmp0 = hf_randc(N1,m,n)
    matrix_space = (hf_randc(N0,N1) @ tmp0.reshape(N1,m*n)).reshape(N0, m, n)
    basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matrix_space, field='complex')
    assert (basis.shape[0]+basis_orth.shape[0])==(m*n)
    assert np.abs(matrix_space.reshape(-1,m*n) @ basis_orth.reshape(-1,m*n).T.conj()).max() < 1e-10


def test_get_matrix_space_orthogonal_basis_C_H():
    N0 = 13
    N1 = 8
    m = 5
    # span_R(C_H^mm)
    tmp0 = hf_randc(N1,m,m)
    tmp1 = (tmp0 + tmp0.transpose(0, 2, 1).conj())/2
    matrix_space = (np_rng.normal(size=(N0,N1)) @ tmp1.reshape(N1,m*m)).reshape(N0, m, m)
    basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matrix_space, field='real')
    assert np.abs(basis-basis.transpose(0,2,1).conj()).max() < 1e-10
    assert np.abs(basis_orth-basis_orth.transpose(0,2,1).conj()).max() < 1e-10
    assert (basis.shape[0]+basis_orth.shape[0])==(m*m)
    assert np.abs(matrix_space.reshape(-1,m*m) @ basis_orth.reshape(-1,m*m).T.conj()).max() < 1e-10


def test_get_matrix_space_orthogonal_basis_R_cT():
    N0 = 13
    N1 = 8
    m = 5
    #span_R(C_T)
    tmp0 = hf_randc(N1,m,m)
    tmp0 = (tmp0 + tmp0.transpose(0,2,1))/2
    matrix_space = (np_rng.normal(size=(N0,N1)) @ tmp0.reshape(N1,m*m)).reshape(N0, m, m)
    basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matrix_space, field='real')

    for x in [basis,basis_orth]:
        tmp0 = numqi.utils.hf_real_to_complex(x)
        tmp1 = numqi.utils.hf_complex_to_real(tmp0)
        assert np.abs(tmp0-tmp0.transpose(0,2,1)).max() < 1e-10
        assert np.abs(tmp1-x).max()<1e-10
    tmp0 = numqi.utils.hf_complex_to_real(matrix_space)
    assert np.abs(tmp0.reshape(-1,4*m*m) @ basis_orth.reshape(-1,4*m*m).T.conj()).max() < 1e-10
    assert (basis.shape[0]+basis_orth.shape[0])==((m*m+m))



def test_get_matrix_space_orthogonal_basis_R_c():
    N0 = 13
    N1 = 8
    m = 5
    n = 7
    #span_R(C)
    tmp0 = hf_randc(N1,m,n)
    matrix_space = (np_rng.normal(size=(N0,N1)) @ tmp0.reshape(N1,m*n)).reshape(N0, m, n)
    basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matrix_space, field='real')

    for x in [basis,basis_orth]:
        tmp0 = numqi.utils.hf_complex_to_real(numqi.utils.hf_real_to_complex(x))
        assert np.abs(tmp0-x).max()<1e-10
    tmp0 = numqi.utils.hf_complex_to_real(matrix_space)
    assert np.abs(tmp0.reshape(-1,4*m*n) @ basis_orth.reshape(-1,4*m*n).T.conj()).max() < 1e-10
    assert (basis.shape[0]+basis_orth.shape[0])==(2*m*n)


def test_linalg_lstsq():
    N0 = 3
    dim = 5

    # R,R -> R
    np0 = hf_rand(N0, dim)
    np1 = hf_rand(dim)
    coeff,residuals,_,_ = np.linalg.lstsq(np0.T, np1, rcond=None) #real
    tmp0 = np.sum((coeff @ np0 - np1)**2)
    assert coeff.dtype.type==np0.dtype.type
    assert abs(residuals.item()-tmp0) < 1e-10

    # C,C -> C
    np0 = hf_randc(N0, dim)
    np1 = hf_randc(dim)
    coeff,residuals,_,_ = np.linalg.lstsq(np0.T, np1, rcond=None) #complex
    tmp0 = np.sum(np.abs(coeff @ np0 - np1)**2)
    assert coeff.dtype.type==np0.dtype.type
    assert abs(residuals.item()-tmp0) < 1e-10

    # C,R -> C
    np0 = hf_randc(N0, dim)
    np1 = hf_rand(dim)
    coeff,residuals,_,_ = np.linalg.lstsq(np0.T, np1, rcond=None) #complex
    tmp0 = np.sum(np.abs(coeff @ np0 - np1)**2)
    assert coeff.dtype.type==np0.dtype.type
    assert abs(residuals.item()-tmp0) < 1e-10

    # R,C -> C
    np0 = hf_rand(N0, dim)
    np1 = hf_randc(dim)
    coeff,residuals,_,_ = np.linalg.lstsq(np0.T, np1, rcond=None) #complex
    tmp0 = np.sum(np.abs(coeff @ np0 - np1)**2)
    assert coeff.dtype.type==np1.dtype.type
    assert abs(residuals.item()-tmp0) < 1e-10


def test_complex_symmetric_matrix():
    hf_sym = lambda x: (x+x.transpose(0,2,1))/2
    hf_rand_sym = lambda N0,N1: hf_sym(np_rng.normal(size=(N0,N1,N1)))
    N0 = 23
    # not normal matrix
    for N1 in range(3, 10):
        np0 = hf_rand_sym(N0, N1) + 1j*hf_rand_sym(N0, N1)
        U,S,V = np.linalg.svd(np0)
        assert np.abs((U*S.reshape(N0,1,N1)) @ V - np0).max() < 1e-10
        assert np.abs(np.abs(U.transpose(0,2,1))-np.abs(V)).max() < 1e-10
        assert np.angle(U.transpose(0,2,1).conj()*V).std(axis=2).max() < 1e-10


def detect_relative_factor(x0, x1, zero_eps, tag_real):
    # x0(np,complex,(N0,N1))
    # x1(np,complex,(N0,N1))
    N0 = x0.shape[0]
    ind0 = np.argmax(np.abs(x1), axis=1)
    ret = x0[np.arange(N0), ind0] / x1[np.arange(N0), ind0]
    if tag_real:
        ret = ret.real
    assert np.abs(x0 - x1*ret[:,np.newaxis]).max() < zero_eps
    return ret


def test_anti_symmetric_matrix():
    # https://en.wikipedia.org/wiki/Skew-symmetric_matrix
    hf_anti = lambda x: (x-x.transpose(0,2,1))/2
    hf_rand_anti = lambda N0,N1: hf_anti(np_rng.normal(size=(N0,N1,N1)))
    N0 = 23

    for N1 in range(3,10):
        np0 = hf_rand_anti(N0, N1)

        u,s,v = np.linalg.svd(np0)
        if N1%2==1:
            assert s[:,-1].max()<1e-7
            s = s[:, :-1]
            u = u[:,:,:-1]
            v = v[:,:-1]
        assert np.abs(s[:,::2]-s[:,1::2]).max() < 1e-7

        tmp0 = detect_relative_factor(u[:,:,::2].transpose(0,2,1).reshape(-1,N1),
                    v[:,1::2].reshape(-1,N1), zero_eps=1e-7, tag_real=True).reshape(N0,-1)
        tmp1 = detect_relative_factor(u[:,:,1::2].transpose(0,2,1).reshape(-1,N1),
                    v[:,::2].reshape(-1,N1), zero_eps=1e-7, tag_real=True).reshape(N0,-1)
        u0 = u.copy()
        tmp2 = [u[:,:,1::2].transpose(0,2,1), u[:,:,::2].transpose(0,2,1)]
        v0 = np.stack(tmp2, axis=2).reshape(N0, -1, N1)
        s0 = s.copy()
        s0[:,::2] *= tmp1
        s0[:,1::2] *= tmp0
        assert np.abs((u0*s0[:,np.newaxis]) @ v0 - np0).max() < 1e-10


def test_complex_anti_symmetric_matrix():
    hf_anti = lambda x: (x-x.transpose(0,2,1))/2
    hf_rand_anti = lambda N0,N1: hf_anti(np_rng.normal(size=(N0,N1,N1)))
    N0 = 23

    for N1 in range(3,10):
        np0 = hf_rand_anti(N0, N1) + 1j*hf_rand_anti(N0,N1)

        u,s,v = np.linalg.svd(np0)
        if N1%2==1:
            assert s[:,-1].max()<1e-7
            s = s[:, :-1]
            u = u[:,:,:-1]
            v = v[:,:-1]
        assert np.abs(s[:,::2]-s[:,1::2]).max() < 1e-7

        tmp0 = detect_relative_factor(u[:,:,::2].transpose(0,2,1).reshape(-1,N1),
                    v[:,1::2].reshape(-1,N1), zero_eps=1e-7, tag_real=False).reshape(N0,-1)
        tmp1 = detect_relative_factor(u[:,:,1::2].transpose(0,2,1).reshape(-1,N1),
                    v[:,::2].reshape(-1,N1), zero_eps=1e-7, tag_real=False).reshape(N0,-1)
        u0 = u.copy()
        tmp2 = [u[:,:,1::2].transpose(0,2,1), u[:,:,::2].transpose(0,2,1)]
        v0 = np.stack(tmp2, axis=2).reshape(N0, -1, N1)
        s0 = s.copy().astype(np.complex128)
        s0[:,::2] *= tmp1.conj()
        s0[:,1::2] *= tmp0.conj()
        assert np.abs((u0*s0[:,np.newaxis]) @ v0 - np0).max() < 1e-10


def test_kraus_op_matrix_space_conversion():
    dim0 = 4
    num_term = 10
    matrix_space = numqi.random.rand_channel_matrix_space(dim0, num_term)
    kraus_op = numqi.matrix_space.matrix_subspace_to_kraus_op(matrix_space)
    matrix_space1 = numqi.matrix_space.kraus_op_to_matrix_subspace(kraus_op, reduce=True)
    assert numqi.matrix_space.is_vector_space_equivalent(matrix_space, matrix_space1, field='complex')


def test_channel_matrix_space_equivalence():
    dim0 = 4
    dim1 = 3
    num_term = 2
    matrix_space = numqi.random.rand_channel_matrix_space(dim0, num_term)
    kraus_op = numqi.matrix_space.matrix_subspace_to_kraus_op(matrix_space)

    rho0 = numqi.random.rand_density_matrix(dim1)
    hf0 = lambda rho: np.kron(numqi.channel.apply_kraus_op(kraus_op, rho), rho0)
    kraus_op1 = numqi.channel.hf_channel_to_kraus_op(hf0, dim0)
    matrix_space1 = numqi.matrix_space.kraus_op_to_matrix_subspace(kraus_op1)
    assert numqi.matrix_space.is_vector_space_equivalent(matrix_space, matrix_space1, field='complex')

    unitary = numqi.random.rand_haar_unitary(kraus_op.shape[1]*dim1)
    hf0 = lambda rho: unitary @ np.kron(numqi.channel.apply_kraus_op(kraus_op, rho), rho0) @ unitary.T.conj()
    kraus_op2 = numqi.channel.hf_channel_to_kraus_op(hf0, dim0)
    matrix_space2 = numqi.matrix_space.kraus_op_to_matrix_subspace(kraus_op2)
    assert numqi.matrix_space.is_vector_space_equivalent(matrix_space, matrix_space2, field='complex')

    ## C'(C(rho)) != C(rho)
    # dim0 = 3
    # dim1 = 4
    # dim2 = 5
    # num_term = 2
    # kraus_op = numqi.random.rand_kraus_op(num_term, dim0, dim1)
    # kraus_op1 = numqi.random.rand_kraus_op(num_term, dim1, dim2)
    # matrix_space = numqi.matrix_space.kraus_op_to_matrix_space(kraus_op)
    # hf0 = lambda rho: numqi.channel.apply_kraus_op(kraus_op1, numqi.channel.apply_kraus_op(kraus_op, rho))
    # tmp0 = numqi.channel.hf_channel_to_kraus_op(hf0, dim0)
    # matrix_space1 = numqi.matrix_space.kraus_op_to_matrix_space(tmp0)
    # print(numqi.matrix_space.is_vector_space_equivalent(matrix_space, matrix_space1, field='complex'))


def test_detect_commute_matrix():
    matrix_subspace = numqi.random.rand_reducible_matrix_subspace(11, partition=(2,3))
    matH_list = numqi.matrix_space.detect_commute_matrix(matrix_subspace, tag_real=True)
    assert len(matH_list)>0
    for matH in matH_list:
        assert np.abs(matH @ matrix_subspace - matrix_subspace @ matH).max() < 1e-10


def test_get_vector_plane():
    N0 = 23
    tmp0 = np_rng.normal(size=(2,23))
    vec0,vec1 = tmp0 / np.linalg.norm(tmp0, axis=1, keepdims=True)
    angle,hf_theta = numqi.matrix_space.get_vector_plane(vec0, vec1)
    assert np.abs(hf_theta(0) - vec0).max() < 1e-10
    assert np.abs(hf_theta(angle) - vec1).max() < 1e-10
    vec2 = hf_theta(np_rng.normal())
    assert not numqi.matrix_space.is_vector_linear_independent(np.stack([vec0,vec1,vec2]), 'real')

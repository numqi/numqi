import numpy as np
import scipy.linalg

import numqi


def test_get_angular_momentum_op():
    levi_civita = np.zeros((3,3,3), dtype=np.int64)
    levi_civita[[0,1,2],[1,2,0],[2,0,1]] = 1
    levi_civita[[0,1,2],[2,0,1],[1,2,0]] = -1

    for j2 in range(10):
        jx,jy,jz = numqi.matrix_space.get_angular_momentum_op(j2)
        jvec = jx,jy,jz
        jsquare = ((j2/2)*(j2/2+1)) * np.eye(j2+1)
        for ind0,ind1,ind2 in zip(*np.nonzero(levi_civita)):
            tmp0 = jvec[ind0] @ jvec[ind1] - jvec[ind1] @ jvec[ind0]
            tmp1 = 1j*levi_civita[ind0,ind1,ind2] * jvec[ind2]
            assert np.abs(tmp0 - tmp1).max() < 1e-10
        assert np.abs(jx @ jx + jy @ jy + jz @ jz - jsquare).max() < 1e-10


def test_get_clebsch_gordan_coeffient():
    tmp0 = [(x,y) for x in range(1,5) for y in range(1,5)]
    for j1_double,j2_double in tmp0:
        # j1_double = 1
        # j2_double = 1
        j1_vec = np.stack(numqi.matrix_space.get_angular_momentum_op(j1_double)) #jx jy jz
        j2_vec = np.stack(numqi.matrix_space.get_angular_momentum_op(j2_double)) #jx jy jz
        z0 = numqi.matrix_space.get_clebsch_gordan_coeffient(j1_double, j2_double)
        tmp0 = [numqi.matrix_space.get_angular_momentum_op(x) for x,_ in z0]
        j1_vec_plus_j2_vec_new_basis = np.stack([scipy.linalg.block_diag(*x) for x in zip(*tmp0)])
        unitary = np.concatenate([x for _,x in z0], axis=0).reshape(-1, (j1_double+1)*(j2_double+1))

        tmp0 = np.eye(j2_double+1)
        tmp0 = np.einsum(j1_vec, [0,1,2], tmp0, [3,4], [0,1,3,2,4], optimize=True).reshape(3, (j1_double+1)*(j2_double+1), -1)
        tmp1 = np.eye(j1_double+1)
        tmp1 = np.einsum(tmp1, [1,2], j2_vec, [0,3,4], [0,1,3,2,4], optimize=True).reshape(3, (j1_double+1)*(j2_double+1), -1)
        j1_vec_plus_j2_vec = tmp0 + tmp1
        assert np.abs(unitary @ j1_vec_plus_j2_vec @ unitary.T - j1_vec_plus_j2_vec_new_basis).max() < 1e-10


def test_get_irreducible_tensor_operator():
    for S_double in [1,2,3,4]:
        z0 = numqi.matrix_space.get_irreducible_tensor_operator(S_double)
        assert np.abs(z0[0][0] - np.eye(S_double+1)).max() < 1e-10

    jx,jy,jz = numqi.matrix_space.get_angular_momentum_op(1)
    T = numqi.matrix_space.get_irreducible_tensor_operator(1)
    # TODO strange sign
    T_jx =  (T[1][2] - T[1][0]) / (2*np.sqrt(2))
    T_jy =  (T[1][0] + T[1][2]) * (1j / (2*np.sqrt(2)))
    T_jz = - T[1][1] / 2 #TODO strange minus sign
    assert np.abs(T_jx-jx).max() < 1e-10
    assert np.abs(T_jy-jy).max() < 1e-10
    assert np.abs(T_jz-jz).max() < 1e-10

    # https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Supplemental_Modules_(Physical_and_Theoretical_Chemistry)/Spectroscopy/Magnetic_Resonance_Spectroscopies/Nuclear_Magnetic_Resonance/NMR_-_Theory/Rotations_and_Irreducible_Tensor_Operators
    for S_double in range(1, 5):
        # S_double = 3
        T = numqi.matrix_space.get_irreducible_tensor_operator(S_double)
        jx,jy,jz = numqi.matrix_space.get_angular_momentum_op(S_double)
        jplus = jx + 1j*jy
        jminus = jx - 1j*jy
        for k, T_i in enumerate(T):
            q = np.arange(k, -1-k, -1)
            assert np.abs(jz @ T_i - T_i @ jz - q.reshape(-1,1,1)*T_i).max() < 1e-10
            tmp0 = T_i @ jplus - jplus @ T_i
            if len(q)>1:
                assert np.abs(tmp0[1:] - np.sqrt(k*(k+1)-q[1:]*(q[1:]+1)).reshape(-1,1,1)*T_i[:-1]).max() < 1e-10
            assert np.abs(tmp0[0]).max() < 1e-10

            tmp0 = T_i @ jminus - jminus @ T_i
            if len(q)>1:
                assert np.abs(tmp0[:-1] - np.sqrt(k*(k+1)-q[:-1]*(q[:-1]-1)).reshape(-1,1,1)*T_i[1:]).max() < 1e-10
            assert np.abs(tmp0[-1]).max() < 1e-10


def test_get_irreducible_hermitian_matrix_basis():
    for S_double in [1,2,3,4,5]:
        op_list = numqi.matrix_space.get_irreducible_hermitian_matrix_basis(S_double, tag_norm=True, tag_stack=True)
        assert np.abs(op_list - op_list.transpose(0,2,1).conj()).max() < 1e-10
        assert np.abs(op_list[0] - np.eye(S_double+1)/np.sqrt(S_double+1)).max() < 1e-10
        tmp0 = op_list.reshape(-1, (S_double+1)**2)
        assert np.abs(tmp0.conj() @ tmp0.T - np.eye((S_double+1)**2)).max() < 1e-10

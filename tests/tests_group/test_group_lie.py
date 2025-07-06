import numpy as np
import scipy.special
import torch

import numqi

np_rng = np.random.default_rng()
hf_randc = lambda *size: np_rng.normal(size=size) + 1j*np_rng.normal(size=size)


def test_su2_to_so3():
    np0 = numqi.random.rand_special_orthogonal_matrix(dim=2, batch_size=23, tag_complex=True)
    np1 = numqi.group.su2_to_so3(np0)
    tmp0 = np1 @ np1.transpose(0,2,1)
    assert np.abs(tmp0-np.eye(3)).max() < 1e-7

    np2 = numqi.group.su2_to_so3(-np0)
    assert np.abs(np1-np2).max() < 1e-7


def test_so3_to_su2():
    np0 = numqi.random.rand_special_orthogonal_matrix(dim=3, batch_size=23, tag_complex=False)
    np1 = numqi.group.so3_to_su2(np0)
    tmp0 = np1 @ np1.transpose(0,2,1).conj()
    assert np.abs(tmp0-np.eye(2)).max() < 1e-7

    np2 = numqi.group.su2_to_so3(np1)
    assert np.abs(np0-np2).max() < 1e-7


def test_su2_so3():
    N0 = 233
    alpha = np_rng.uniform(0, 2*np.pi, size=N0)
    beta = np_rng.uniform(0, np.pi, size=N0)
    gamma = np_rng.uniform(0, 2*np.pi, size=N0)

    z0 = numqi.group.angle_to_so3(alpha, beta, gamma)
    z1 = numqi.group.angle_to_so3(alpha, beta, gamma+2*np.pi)
    assert np.abs(z0-z1).max() < 1e-10

    z2 = numqi.group.angle_to_su2(alpha, beta, gamma)
    z3 = numqi.group.angle_to_su2(alpha, beta, gamma+2*np.pi)
    assert np.abs(z2+z3).max() < 1e-10
    assert np.abs(numqi.group.su2_to_so3(z2)-z0).max() < 1e-10
    assert np.abs(numqi.group.su2_to_so3(z3)-z0).max() < 1e-10


def test_angle_to_su2():
    np0 = numqi.random.rand_special_orthogonal_matrix(dim=2, batch_size=23, tag_complex=True)
    z0 = numqi.group.angle_to_su2(*numqi.group.su2_to_angle(np0))
    assert np.abs(np0-z0).max() < 1e-10


def test_get_su2_irrep():
    N0 = 23
    np0 = numqi.random.rand_special_orthogonal_matrix(dim=2, batch_size=N0, tag_complex=True)
    np1 = numqi.random.rand_special_orthogonal_matrix(dim=2, batch_size=N0, tag_complex=True)

    assert np.abs(numqi.group.get_su2_irrep(1, np0)-np0).max() < 1e-10

    for j2 in range(2,10):
        # TODO irrep(j2=2) is real or not
        z0 = numqi.group.get_su2_irrep(j2, np0)
        z1 = numqi.group.get_su2_irrep(j2, np1)

        ret_ = numqi.group.get_su2_irrep(j2, np0@np1)
        ret0 = z0 @ z1
        assert np.abs(ret_-ret0).max() < 1e-8 #fail sometimes


def test_get_su2_irrep_matd():
    N0 = 233
    beta = np_rng.uniform(0, np.pi, size=N0)
    alpha = 0 * beta
    gamma = 0 * beta

    for j2 in range(2, 10):
        # eq4.74 @ZhongqiMa
        np0 = numqi.group.get_su2_irrep(j2, alpha, beta, gamma)
        tmp0 = 1 - 2*((np.arange(j2+1)[:,np.newaxis]-np.arange(j2+1))%2)
        np1 = numqi.group.get_su2_irrep(j2, alpha, -beta, gamma)
        assert np.abs(np0 - np0[:,::-1,::-1].transpose(0,2,1)).max() < 1e-10
        assert np.abs(np0 - tmp0 * np1).max() < 1e-10 #eq4.74 @ZhongqiMa
        assert np.abs(np0 - tmp0 * np0.transpose(0,2,1)).max() < 1e-10
        assert np.abs(np0 - np1.transpose(0,2,1)).max() < 1e-10
        assert np.abs(np0 - tmp0 * np0[:,::-1,::-1]).max() < 1e-10

        # eq4.75 @ZhongqiMa
        jmm = np.arange(j2+1, dtype=np.int64)
        jpm = jmm[::-1]
        tmp0 = np.sqrt(scipy.special.factorial(j2)/(scipy.special.factorial(jpm)*scipy.special.factorial(jmm)))
        tmp1 = np.cos(beta[:,np.newaxis]/2)**(jpm) * np.sin(beta[:,np.newaxis]/2)**(jmm)
        assert np.abs(np0[:,:,0]-tmp0*tmp1).max() < 1e-10
        cb = np.cos(beta/2)
        sb = np.sin(beta/2)
        if j2//2==0:
            ret0 = 0
            for r in range(j2//2+1):
                tmp1 = scipy.special.factorial(j2//2)/(scipy.special.factorial(r)*scipy.special.factorial(j2//2-r))
                ret0 = ret0 + (1-2*(r%2)) * (tmp1 * cb**(j2//2-r) * sb**r)**2
            assert np.abs(np0[:,j2//2,j2//2] - ret0).max() < 1e-10


def test_get_rational_orthogonal2_matrix():
    for m in range(-10, 10):
        for n in range(-10,10):
            if (m!=0) and (n!=0) and (abs(m)!=abs(n)):
                tmp0 = numqi.group.get_rational_orthogonal2_matrix(m, n)
                assert abs(tmp0 @ tmp0.T-np.eye(2)).max() < 1e-10


def test_su2su2_to_so4_magic():
    np0,np1 = numqi.random.rand_special_orthogonal_matrix(2, batch_size=2, tag_complex=True)
    np2 = numqi.group.su2su2_to_so4_magic(np0, np1)
    assert np.abs(numqi.group.su2su2_to_so4_magic(-np0, -np1)-np2).max() < 1e-10 #2-to-1 mapping
    assert np.abs(np2 @ np2.T - np.eye(4)).max() < 1e-10
    assert np.linalg.det(np2) > 0
    np3, np4 = numqi.group.so4_to_su2su2_magic(np2)
    tmp0 = max(np.abs(np3-np0).max(), np.abs(np4-np1).max())
    tmp1 = max(np.abs(np3+np0).max(), np.abs(np4+np1).max())
    assert min(tmp0, tmp1) < 1e-10


def test_su2_to_so3_magic():
    np0 = numqi.random.rand_special_orthogonal_matrix(2, tag_complex=True)
    np1 = numqi.group.su2_to_so3_magic(np0)
    assert np.abs(numqi.group.su2_to_so3_magic(-np0)-np1).max() < 1e-10 #2-to-1 mapping
    assert np.abs(np1 @ np1.T - np.eye(3)).max() < 1e-10
    assert np.linalg.det(np1) > 0
    np2 = numqi.group.so3_to_su2_magic(np1)
    assert np.abs(np2@np2.T.conj()-np.eye(2)).max() < 1e-10
    assert min(np.abs(np2-np0).max(), np.abs(np2+np0).max()) < 1e-10


def test_SUn_real_imag_part():
    # https://arxiv.org/abs/quant-ph/0507171v1 eq(15)
    np0 = numqi.random.rand_haar_unitary(3) #U(n) not SU(n)
    tmp0 = np0.imag.T @ np0.real
    assert np.abs(tmp0-tmp0.T).max() < 1e-10
    tmp0 = np0.imag @ np0.real.T
    assert np.abs(tmp0-tmp0.T).max() < 1e-10


def test_diagonalize_unitary_using_two_orthogonals():
    np0 = numqi.random.rand_haar_unitary(6)
    Qleft, D, Qright = numqi.group.diagonalize_unitary_using_two_orthogonals(np0)
    assert np.abs(Qleft.imag).max() < 1e-10
    assert np.abs(Qright.imag).max() < 1e-10
    assert np.abs(Qleft @ Qleft.T - np.eye(np0.shape[0])).max() < 1e-10
    assert np.abs(Qright @ Qright.T - np.eye(np0.shape[0])).max() < 1e-10
    assert np.abs(np.abs(D) - 1).max() < 1e-10
    assert np.abs(np0 - (Qleft*D) @ Qright.T).max()


def test_get_su4_kak_decomposition():
    for _ in range(100):
        np0 = numqi.random.rand_haar_unitary(4)
        A0,A1,B0,B1,vecK,diag = numqi.group.get_su4_kak_decomposition(np0)
        tmp0 = numqi.group.get_kak_kernel(vecK[1:]) * np.exp(1j*vecK[0])
        assert np.abs(np.kron(A0,A1) @ tmp0 @ np.kron(B0,B1) - np0).max() < 1e-10
        tmp0 = vecK[1:] / (np.pi/2)
        assert (1>=tmp0[0]) and (tmp0[0]>=tmp0[1]) and (tmp0[1]>=tmp0[2]) and (tmp0[2]>=0) and (tmp0[0]+tmp0[1]<=1)

    A0,A1,B0,B1,vecK,diag = numqi.group.get_su4_kak_decomposition(numqi.gate.CNOT)
    tmp0 = vecK[1:]/(np.pi/2)
    assert np.abs(tmp0-np.array([1/2,0,0])).max() < 1e-10

    A0,A1,B0,B1,vecK,diag = numqi.group.get_su4_kak_decomposition(numqi.gate.Swap)
    tmp0 = vecK[1:]/(np.pi/2)
    assert np.abs(tmp0-np.array([1/2,1/2,1/2])).max() < 1e-10


def test_get_kak_kernel():
    vecK = np_rng.normal(size=3)
    ret_ = scipy.linalg.expm(1j*(vecK[0]*numqi.group._lie._getX('XX')
            + vecK[1]*numqi.group._lie._getX('YY') + vecK[2]*numqi.group._lie._getX('ZZ')))
    np0 = numqi.group.get_kak_kernel(vecK)
    assert np.abs(np0-ret_).max() < 1e-10
    np1 = numqi.group.get_kak_kernel(torch.tensor(vecK, dtype=torch.float64)).numpy()
    assert np.abs(np1-ret_).max() < 1e-10


def test_KAK_gamma():
    # https://arxiv.org/abs/quant-ph/0507171v1 eq(32)
    XX = numqi.group._lie._getX('XX')
    YY = numqi.group._lie._getX('YY')
    ZZ = numqi.group._lie._getX('ZZ')
    magic = numqi.group._lie._getX('magic')
    assert np.abs(magic @ magic.T.conj() - np.eye(4)).max() < 1e-10
    XX = np.kron(numqi.gate.X, numqi.gate.X)
    YY = np.kron(numqi.gate.Y, numqi.gate.Y)
    ZZ = np.kron(numqi.gate.Z, numqi.gate.Z)
    gamma = np.array([[1,1,-1,1], [1,1,1,-1], [1,-1,-1,-1], [1,-1,1,1]])
    assert np.abs(gamma @ gamma.T - 4*np.eye(4)).max() < 1e-10
    vecK = np_rng.normal(size=4)
    np0 = magic.T.conj() @ scipy.linalg.expm(1j*(vecK[1]*XX + vecK[2]*YY + vecK[3]*ZZ)) * np.exp(1j*vecK[0]) @ magic
    np1 = np.diag(np.exp(1j*(gamma @ vecK)))
    assert np.abs(np0-np1).max() < 1e-10

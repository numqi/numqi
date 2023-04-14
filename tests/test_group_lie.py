import numpy as np
import scipy.special

import numqi

np_rng = np.random.default_rng()
hf_randc = lambda *size: np_rng.normal(size=size) + 1j*np_rng.normal(size=size)


def test_rand_su2():
    np0 = numqi.group.rand_su2(23)
    tmp0 = np0 @ np0.transpose(0,2,1).conj()
    assert np.abs(tmp0-np.eye(2)).max() < 1e-7


def test_rand_so3():
    np0 = numqi.group.rand_so3(23)
    tmp0 = np0 @ np0.transpose(0,2,1)
    assert np.abs(tmp0-np.eye(3)).max() < 1e-7


def test_su2_to_so3():
    np0 = numqi.group.rand_su2(23)
    np1 = numqi.group.su2_to_so3(np0)
    tmp0 = np1 @ np1.transpose(0,2,1)
    assert np.abs(tmp0-np.eye(3)).max() < 1e-7

    np2 = numqi.group.su2_to_so3(-np0)
    assert np.abs(np1-np2).max() < 1e-7


def test_so3_to_su2():
    np0 = numqi.group.rand_so3(23)
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
    N0 = 233
    np0 = numqi.group.rand_su2(N0)
    z0 = numqi.group.angle_to_su2(*numqi.group.su2_to_angle(np0))
    assert np.abs(np0-z0).max() < 1e-10


def test_get_su2_irrep():
    N0 = 233
    np0 = numqi.group.rand_su2(N0)
    np1 = numqi.group.rand_su2(N0)

    assert np.abs(numqi.group.get_su2_irrep(1, np0)-np0).max() < 1e-10

    for j2 in range(2,10):
        # TODO irrep(j2=2) is real or not
        z0 = numqi.group.get_su2_irrep(j2, np0)
        z1 = numqi.group.get_su2_irrep(j2, np1)

        ret_ = numqi.group.get_su2_irrep(j2, np0@np1)
        ret0 = z0 @ z1
        assert np.abs(ret_-ret0).max() < 1e-9 #fail sometimes


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

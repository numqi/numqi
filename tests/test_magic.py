import numpy as np

import numqi

try:
    import mosek
    USE_MOSEK = True
except ImportError:
    USE_MOSEK = False


np_rng = np.random.default_rng()
hf_randc = lambda *sz: np_rng.normal(size=sz) + 1j*np_rng.normal(size=sz)

def test_matrix_to_wigner_basis():
    batch_size = 3
    for dim in [2,3,5,7]:
        tmp0 = hf_randc(dim,dim)
        np0 = (tmp0 + tmp0.T.conj())
        ret0 = numqi.magic.matrix_to_wigner_basis(np0)
        ret1 = numqi.magic.matrix_to_wigner_basis(np0, is_hermitian=True) #real vector
        assert np.abs(ret0-ret1).max()<1e-10

        np0 = hf_randc(batch_size,dim,dim)
        tmp0 = numqi.magic.matrix_to_wigner_basis(np0)
        ret0 = numqi.magic.wigner_basis_to_matrix(tmp0)
        assert np.abs(ret0-np0).max()<1e-10


def test_get_thauma_boundary():
    # thauma is correct in xy-plane of Bloch sphere
    N0 = 23
    theta = np_rng.uniform(0, 2*np.pi, size=N0)
    tmp0 = np.stack([0*theta,np.cos(theta), np.sin(theta)], axis=1) * 0.5
    dm_list = numqi.gellmann.gellmann_basis_to_dm(tmp0)
    ret_ = numqi.magic.get_magic_state_boundary_qubit(dm_list)
    ret0 = numqi.magic.get_thauma_boundary(dm_list)
    assert np.abs(ret_-ret0).max() < 1e-7


def test_get_Heisenberg_Weyl_operator():
    for dim in [2,3,5,7,11]:
        weyl_T,weyl_A = numqi.magic.get_Heisenberg_Weyl_operator(dim)

        tmp0 = weyl_A.reshape(-1, dim, dim)
        assert np.abs(tmp0 - tmp0.conj().transpose(0,2,1)).max() < 1e-10
        assert np.abs(np.trace(tmp0, axis1=1, axis2=2)-1).max() < 1e-10
        assert np.abs(tmp0.sum(axis=0) - np.eye(dim)*dim).max() < 1e-10
        tmp1 = np.einsum(tmp0, [0,1,2], tmp0, [3,2,1], [0,3], optimize=True)
        assert np.abs(tmp1 - np.eye(dim*dim)*dim).max() < 1e-10


def test_get_qutrit_Hstate():
    matH = numqi.gate.get_quditH(3)
    EVC = np.stack([numqi.magic.get_qutrit_nonstabilizer_state(k) for k in ['Hplus','Hminus','Hi']], axis=1)
    EVL = np.array([1,-1,1j])
    assert np.abs((EVC*EVL) @ EVC.T.conj() - matH).max() < 1e-10


def test_qutrit_thauma_sdp():
    case_list = [
        ('Hplus', np.log(3-np.sqrt(3))),
        ('Strange', np.log(5/3)),
        ('Norrell', np.log(3/2)),
        ('T', np.log(1+2*np.sin(np.pi/18)))
    ]
    tmp0 = [numqi.magic.get_qutrit_nonstabilizer_state(k) for k,_ in case_list]
    dm_target_list = np.stack([x.reshape(-1,1)*x.conj() for x in tmp0])
    ret_ = np.array([v for _,v in case_list])
    ret0 = numqi.magic.get_thauma_sdp(dm_target_list, kind='max')
    ret1 = numqi.magic.get_thauma_sdp(dm_target_list, kind='min')
    ret2 = numqi.magic.get_thauma_sdp(dm_target_list, kind='entropy')
    assert np.abs(ret_-ret0).max() < (1e-6 if USE_MOSEK else 1e-4)
    assert np.abs(ret_-ret1).max() < (1e-6 if USE_MOSEK else 1e-4)
    assert np.abs(ret_-ret2).max() < (1e-6 if USE_MOSEK else 1e-4)

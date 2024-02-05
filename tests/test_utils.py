import numpy as np

import numqi

np_rng = np.random.default_rng()
hf_randc = lambda *size: np_rng.normal(size=size) + 1j*np_rng.normal(size=size)

def test_hf_complex_to_real():
    N0 = 23
    N1 = 4

    np0 = hf_randc(N0, N1, N1)
    tmp0 = numqi.utils.hf_real_to_complex(numqi.utils.hf_complex_to_real(np0))
    assert np.abs(np0-tmp0).max() < 1e-10

    np0 = hf_randc(N0, N1, N1)
    np1 = hf_randc(N0, N1, N1)
    ret_ = np0 @ np1
    tmp0 = [numqi.utils.hf_complex_to_real(x) for x in (np0,np1)]
    ret0 = numqi.utils.hf_real_to_complex(tmp0[0] @ tmp0[1])
    assert np.abs(ret_-ret0).max() < 1e-10


def test_get_purification():
    dim = 3
    for dimR in [None, 4]:
        rho = numqi.random.rand_density_matrix(dim)
        psi = numqi.utils.get_purification(rho, dimR=dimR)
        tmp0 = psi @ psi.T.conj()
        assert np.abs(tmp0 - rho).max() < 1e-10


def test_trace_distance_contraction():
    # @book-QCQI-page406/eq9.35 trace-distance is contractive under TPCP map
    din = 3
    dout = 5
    kop_term = din*dout

    rho0 = numqi.random.rand_density_matrix(din)
    rho1 = numqi.random.rand_density_matrix(din)
    kop = numqi.random.rand_kraus_op(kop_term, din, dout)

    ret0 = numqi.utils.get_trace_distance(rho0, rho1)
    tmp0 = numqi.channel.apply_kraus_op(kop, rho0)
    tmp1 = numqi.channel.apply_kraus_op(kop, rho1)
    ret1 = numqi.utils.get_trace_distance(tmp0, tmp1)
    assert ret1 < (ret0+1e-10) #epsilon is added to avoid rounding error

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

import numpy as np

import numpyqi

np_rng = np.random.default_rng()
hf_randc = lambda *size: np_rng.normal(size=size) + 1j*np_rng.normal(size=size)

def test_matrix_to_gellman_basis():
    for N0 in [3,4,5]:
        np0 = np.random.rand(N0,N0) + np.random.rand(N0,N0)*1j
        coeff = numpyqi.gellmann.matrix_to_gellmann_basis(np0, norm_I='sqrt(2/d)') #default
        tmp0 = numpyqi.gellmann.all_gellmann_matrix(N0)
        ret0 = sum(x*y for x,y in zip(coeff,tmp0))
        assert np.abs(np0-ret0).max()<1e-7


def test_all_gellmann_matrix():
    # https://arxiv.org/abs/1705.01523
    for d in [3,4,5]:
        all_term = numpyqi.gellmann.all_gellmann_matrix(d, with_I=False)
        for ind0,x in enumerate(all_term):
            for ind1,y in enumerate(all_term):
                assert abs(np.trace(x @ y)-2*(ind0==ind1)) < 1e-7


def test_matrix_to_gellmann_basis():
    N0 = 5
    for norm_I in ['1/d', 'sqrt(2/d)']:
        np0 = hf_randc(N0,N0)
        np1 = numpyqi.gellmann.matrix_to_gellmann_basis(np0, norm_I=norm_I)
        np2 = numpyqi.gellmann.gellmann_basis_to_matrix(np1, norm_I=norm_I)
        assert np.abs(np0-np2).max() < 1e-7

import numpy as np

import numpyqi


def test_rand_quantum_channel_matrix_subspace():
    zero_eps = 1e-7

    dim_in = 4
    num_sym = 6
    num_antisym = 2
    np0 = numpyqi.random.rand_quantum_channel_matrix_subspace(dim_in, num_hermite=(num_sym,num_antisym))
    assert np0.shape==(num_sym+num_antisym,dim_in,dim_in)
    tmp0 = np0.transpose(0,2,1)
    assert np.abs((np0-tmp0)[:num_sym]).max() < zero_eps
    assert np.abs((np0+tmp0)[num_sym:]).max() < zero_eps

    num_hermite = 8
    np0 = numpyqi.random.rand_quantum_channel_matrix_subspace(dim_in, num_hermite)
    assert np0.shape==(num_hermite,dim_in,dim_in)
    assert np.abs(np0-np0.transpose(0,2,1).conj()).max() < 1e-7

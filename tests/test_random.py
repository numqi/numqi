import numpy as np
import itertools

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


def test_rand_ABk_density_matrix():
    for dimA,dimB,kext in [(2,5,3),(2,3,5)]:
        np0 = numpyqi.random.rand_ABk_density_matrix(dimA,dimB,kext)
        assert abs(np.trace(np0)-1)<1e-10
        assert np.abs(np0-np0.T.conj()).max() < 1e-7
        tmp0 = np0.reshape([dimA]+[dimB]*kext+[dimA]+[dimB]*kext)
        for indI,indJ in itertools.combinations(list(range(kext)), 2):
            tmp1 = np.arange(2*kext+2, dtype=np.int64)
            tmp1[[indI+1,indJ+1]] = tmp1[[indJ+1,indI+1]]
            tmp1[[indI+2+kext,indJ+2+kext]] = tmp1[[indJ+2+kext,indI+2+kext]]
            assert np.abs(tmp0-np.transpose(tmp0,tmp1)).max() < 1e-10

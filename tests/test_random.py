import numpy as np
import itertools

import numqi

np_rng = np.random.default_rng()

def test_rand_quantum_channel_matrix_subspace():
    zero_eps = 1e-7

    dim_in = 4
    num_sym = 6
    num_antisym = 2
    np0 = numqi.random.rand_quantum_channel_matrix_subspace(dim_in, num_hermite=(num_sym,num_antisym))
    assert np0.shape==(num_sym+num_antisym,dim_in,dim_in)
    tmp0 = np0.transpose(0,2,1)
    assert np.abs((np0-tmp0)[:num_sym]).max() < zero_eps
    assert np.abs((np0+tmp0)[num_sym:]).max() < zero_eps

    num_hermite = 8
    np0 = numqi.random.rand_quantum_channel_matrix_subspace(dim_in, num_hermite)
    assert np0.shape==(num_hermite,dim_in,dim_in)
    assert np.abs(np0-np0.transpose(0,2,1).conj()).max() < 1e-7


def test_rand_ABk_density_matrix():
    for dimA,dimB,kext in [(2,5,3),(2,3,5)]:
        np0 = numqi.random.rand_ABk_density_matrix(dimA,dimB,kext)
        assert abs(np.trace(np0)-1)<1e-10
        assert np.abs(np0-np0.T.conj()).max() < 1e-7
        tmp0 = np0.reshape([dimA]+[dimB]*kext+[dimA]+[dimB]*kext)
        for indI,indJ in itertools.combinations(list(range(kext)), 2):
            tmp1 = np.arange(2*kext+2, dtype=np.int64)
            tmp1[[indI+1,indJ+1]] = tmp1[[indJ+1,indI+1]]
            tmp1[[indI+2+kext,indJ+2+kext]] = tmp1[[indJ+2+kext,indI+2+kext]]
            assert np.abs(tmp0-np.transpose(tmp0,tmp1)).max() < 1e-10


def test_rand_symmetric_inner_product():
    zero_eps = 1e-10
    for N0 in [2,3,4,5]:
        matB,matU = numqi.random.rand_symmetric_inner_product(N0)
        tmp0 = matB @ matU - matU.T @ matB
        assert np.abs(tmp0+tmp0.transpose(0,2,1)).max() < zero_eps
        vecX = np_rng.normal(size=N0)
        vecY = matU @ vecX
        assert np.abs((vecX @ matB) @ vecY - (vecY @ matB) @ vecX).max() < zero_eps


def test_rand_kraus_op():
    num_term = 4
    dim_in = 5
    dim_out = 3
    kop = numqi.random.rand_kraus_op(num_term, dim_in, dim_out)
    tmp0 = np.einsum(kop, [0,1,2], kop.conj(), [0,1,3], [2,3], optimize=True)
    assert np.abs(tmp0-np.eye(dim_in)).max() < 1e-10


def test_rand_choi_op():
    dim_in = 3
    dim_out = 5
    choi_op = numqi.random.rand_choi_op(dim_in, dim_out)
    assert np.linalg.eigvalsh(choi_op).min() >= -1e-7
    tmp0 = np.trace(choi_op.reshape(dim_in, dim_out, dim_in, dim_out), axis1=1, axis2=3)
    assert np.abs(tmp0-np.eye(dim_in)).max() < 1e-10


def test_rand_povm():
    dim = 3
    povm = numqi.random.rand_povm(dim, num_term=5)
    assert np.abs(povm - povm.transpose(0,2,1).conj()).max() < 1e-10
    assert np.abs(povm.sum(axis=0) - np.eye(dim)).max() < 1e-10
    assert np.linalg.eigvalsh(povm).min() >= -1e-7


def test_rand_Stiefel_matrix():
    dim = 7
    rank = 3

    np0 = numqi.random.rand_Stiefel_matrix(dim, rank, iscomplex=False)
    assert np0.shape==(dim,rank)
    assert np.abs(np0.T @ np0 - np.eye(rank)).max() < 1e-7

    batch_size = 5
    np0 = numqi.random.rand_Stiefel_matrix(dim, rank, iscomplex=True, batch_size=batch_size)
    assert np0.shape==(batch_size,dim,rank)
    tmp0 = np.einsum(np0, [0,1,2], np0.conj(), [0,1,3], [0,2,3], optimize=True)
    assert np.abs(tmp0-np.eye(rank)).max() < 1e-7

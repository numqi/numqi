import itertools
import numpy as np

import numqi

def test_ABkHermitian():
    dimA_list = [1,2,3]
    dimB_list = [2,3]
    kext_list = list(range(1,6))
    for dimA,dimB,kext in itertools.product(dimA_list, dimB_list, kext_list):
        layer = numqi.manifold.ABkHermitian(dimA, dimB, kext)
        mat = layer().detach().numpy().copy()
        assert np.abs(mat-mat.T.conj()).max()<1e-6
        tmp0 = [(x,y) for x in range(kext) for y in range(x+1,kext)]
        for ind0,ind1 in tmp0:
            tmp1 = numqi.manifold._ABk.ABk_permutate(mat, ind0, ind1, dimA, dimB, kext)
            assert np.all(np.abs(mat-tmp1)<1e-10)


def test_ABk2localHermitian():
    dimA_list = [1,2,3]
    dimB_list = [2,3]
    kext_list = list(range(1,6))
    for dimA,dimB,kext in itertools.product(dimA_list, dimB_list, kext_list):
        layer = numqi.manifold.ABk2localHermitian(dimA, dimB, kext)
        ret0 = layer().detach().numpy().copy()
        np0 = layer.to_AB()
        assert np.abs(ret0-ret0.T.conj()).max() < 1e-7
        tmp0 = np.kron(np0, np.eye(dimB**(kext-1)))
        ret_ = tmp0
        for x in range(1,kext):
            ret_ = ret_ + numqi.manifold._ABk.ABk_permutate(tmp0, 0, x, dimA, dimB, kext)
        assert np.abs(ret_-ret0).max() < 1e-10


def test_ABk_2local_symmetry_index():
    dimA_list = [1,2,3]
    dimB_list = [2,3]
    kext_list = list(range(1,6))
    np_rng = np.random.default_rng()
    for dimA,dimB,kext in itertools.product(dimA_list, dimB_list, kext_list):
        coeff,index = numqi.manifold._ABk.ABk_2local_symmetry_index(dimA, dimB, kext)
        tmp0 = np_rng.uniform(-1, 1, size=(dimA*dimB,dimA*dimB))
        np0 = tmp0 + tmp0.T

        parameter = np0[np.triu(np.ones((dimA*dimB,dimA*dimB),dtype=np.bool_))]
        ret0 = numqi.manifold._ABk.ABk_2local_index_to_full(parameter, coeff, index)
        tmp0 = np.kron(np0, np.eye(dimB**(kext-1)))
        ret_ = tmp0
        for x in range(1,kext):
            ret_ = ret_ + numqi.manifold._ABk.ABk_permutate(tmp0, 0, x, dimA, dimB, kext)
        assert np.abs(ret_-ret0).max() < 1e-10


def test_ABk_2local_skew_symmetry_index():
    dimA_list = [1,2,3]
    dimB_list = [2,3]
    kext_list = list(range(1,6))
    np_rng = np.random.default_rng()
    for dimA,dimB,kext in itertools.product(dimA_list, dimB_list, kext_list):
        coeff,index = numqi.manifold._ABk.ABk_2local_skew_symmetry_index(dimA, dimB, kext)
        tmp0 = np_rng.uniform(-1, 1, size=(dimA*dimB,dimA*dimB))
        np0 = tmp0 - tmp0.T

        parameter = np0[np.triu(np.ones((dimA*dimB,dimA*dimB),dtype=np.bool_),k=1)]
        ret0 = numqi.manifold._ABk.ABk_2local_index_to_full(parameter, coeff, index)
        assert np.abs(ret0+ret0.T).max() < 1e-7
        tmp0 = np.kron(np0, np.eye(dimB**(kext-1)))
        ret_ = tmp0
        for x in range(1,kext):
            ret_ = ret_ + numqi.manifold._ABk.ABk_permutate(tmp0, 0, x, dimA, dimB, kext)
        assert np.abs(ret_-ret0).max() < 1e-10

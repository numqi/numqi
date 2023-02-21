import itertools
import functools
import numpy as np
import scipy.special

import numpyqi

np_rng = np.random.default_rng()
hf_kron = lambda x: functools.reduce(np.kron, x)
hf_randc = lambda *size: np_rng.normal(size=size) + 1j*np_rng.normal(size=size)

def test_tensor2d_project_to_sym_antisym_basis():
    dimA = 3
    dimB = 3
    r = 2
    k = 2
    num_matrix = 3
    matrix_subspace = [hf_randc(dimA,dimB) for _ in range(num_matrix)]
    tmp0 = list(itertools.combinations_with_replacement(list(range(len(matrix_subspace))), r+k))
    ret_ = np.stack([numpyqi.matrix_space.naive_tensor2d_project_to_sym_antisym_basis([matrix_subspace[y] for y in x], r) for x in tmp0])
    ret0 = np.stack([numpyqi.matrix_space.tensor2d_project_to_sym_antisym_basis([matrix_subspace[y] for y in x], r) for x in tmp0])
    assert np.abs(ret_-ret0).max() < 1e-10


def test_symmetrical_is_all_permutation():
    dim = 3
    num_matrix = 4
    np_rng = np.random.default_rng()

    np_list = [np_rng.normal(size=dim) for _ in range(num_matrix)]
    z0 = numpyqi.matrix_space.get_symmetric_basis(dim, num_matrix)
    ret_ = z0.T @ (z0 @ hf_kron([x[:,np.newaxis] for x in np_list])[:,0])

    ret0 = 0
    for ind0 in itertools.permutations(list(range(num_matrix))):
        tmp0 = np_list[ind0[0]]/scipy.special.factorial(num_matrix)
        for x in ind0[1:]:
            tmp0 = (tmp0[:,np.newaxis] * np_list[x]).reshape(-1)
        ret0 = ret0 + tmp0
    assert np.abs(ret_-ret0).max() < 1e-10


def test_project_to_antisymmetric_basis():
    np_rng = np.random.default_rng()
    num_batch = 23
    for dim,repeat in [(5,2),(5,3),(5,4)]:
        np_list = [np_rng.normal(size=(dim,num_batch)) for _ in range(repeat)]

        antisym_basis = numpyqi.matrix_space.get_antisymmetric_basis(dim, repeat)
        ret_ = []
        for ind0 in range(num_batch):
            tmp0 = [x[:,ind0] for x in np_list]
            ret_.append(antisym_basis @ functools.reduce(lambda x,y: (x.reshape(-1,1)*y).reshape(-1), tmp0))
        ret_ = np.stack(ret_, axis=1)

        ret0 = numpyqi.matrix_space.project_to_antisymmetric_basis(np_list)
        assert np.abs(ret_-ret0).max() < 1e-10


def test_get_symmetric_basis():
    for dim,repeat in [(5,2),(5,3),(5,4)]:
        z0 = numpyqi.matrix_space.get_symmetric_basis(dim, repeat)
        assert np.abs(np.linalg.norm(z0, axis=1)-1).max() < 1e-7
        N0 = z0.shape[0]
        z0 = z0.reshape([N0] + [dim]*repeat)
        permutation_index = np.array(numpyqi.matrix_space.permutation_with_antisymmetric_factor(repeat)[0])
        permutation_reverse_index = np.argsort(permutation_index, axis=1)
        for r_index in permutation_reverse_index:
            z1 = z0.transpose(*([0] + (r_index+1).tolist()))
            assert np.abs(z0-z1).max() < 1e-10


def test_get_antisymmetric_basis():
    for dim,repeat in [(5,2),(5,3),(5,4)]:
        z0 = numpyqi.matrix_space.get_antisymmetric_basis(dim, repeat)
        assert np.abs(np.linalg.norm(z0, axis=1)-1).max() < 1e-7
        N0 = z0.shape[0]
        z0 = z0.reshape([N0] + [dim]*repeat)
        tmp0 = numpyqi.matrix_space.permutation_with_antisymmetric_factor(repeat)
        permutation_index = np.array(tmp0[0])
        permutation_reverse_index = np.argsort(permutation_index, axis=1)
        permutation_factor = tmp0[1]
        for r_index,factor in zip(permutation_reverse_index,permutation_factor):
            z1 = z0.transpose(*([0] + (r_index+1).tolist()))*factor
            assert np.abs(z0-z1).max() < 1e-10


def test_tensor2d_project_to_antisym_basis():
    np_rng = np.random.default_rng()
    for dim0,dim1,repeat in [(4,4,2),(4,5,2),(7,8,3),(8,8,4)]:
        np_list = [np_rng.normal(size=(dim0,dim1)) for _ in range(repeat)]

        antisym_basis0 = numpyqi.matrix_space.get_antisymmetric_basis(dim0, repeat)
        antisym_basis1 = numpyqi.matrix_space.get_antisymmetric_basis(dim1, repeat)
        ret_ = antisym_basis0 @ hf_kron(np_list) @ antisym_basis1.T

        ret0 = numpyqi.matrix_space.tensor2d_project_to_antisym_basis(np_list)
        assert np.abs(ret_-ret0).max() < 1e-10

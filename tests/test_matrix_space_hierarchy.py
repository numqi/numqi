import time
import itertools
import functools
import numpy as np
import scipy.special

import numqi

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
    ret_ = np.stack([numqi.matrix_space.naive_tensor2d_project_to_sym_antisym_basis([matrix_subspace[y] for y in x], r) for x in tmp0])
    ret0 = np.stack([numqi.matrix_space.tensor2d_project_to_sym_antisym_basis([matrix_subspace[y] for y in x], r) for x in tmp0])
    assert np.abs(ret_-ret0).max() < 1e-10


def test_symmetrical_is_all_permutation():
    rank = 4
    dim = 3
    np0 = np_rng.uniform(-1,1,size=[dim]*rank)

    sym_basis = numqi.matrix_space.get_symmetric_basis(dim, rank)
    ret_ = (sym_basis.T @ (sym_basis @ np0.reshape(-1))).reshape(np0.shape)

    ret0 = 0
    for ind0 in itertools.permutations(list(range(rank))):
        ret0 = ret0 + np0.transpose(*ind0)
    ret0 /= scipy.special.factorial(rank)
    assert np.abs(ret_-ret0).max() < 1e-10


def test_project_nd_tensor_to_antisymmetric_basis():
    rank = 3
    dim = 5
    np0 = np_rng.uniform(-1,1,size=[dim]*rank)
    ret0 = numqi.matrix_space.project_nd_tensor_to_antisymmetric_basis(np0, rank=rank)
    basis = numqi.matrix_space.get_antisymmetric_basis(dim, rank)
    ret1 = basis @ np0.reshape(-1)
    assert np.abs(ret0-ret1).max() < 1e-10

    N0 = 4
    np0 = np_rng.uniform(-1,1,size=([dim]*rank+[N0]))
    ret0 = numqi.matrix_space.project_nd_tensor_to_antisymmetric_basis(np0, rank=rank)
    basis = numqi.matrix_space.get_antisymmetric_basis(dim, rank)
    ret1 = basis @ np0.reshape(-1,N0)
    assert np.abs(ret0-ret1).max() < 1e-10


def test_project_to_antisymmetric_basis():
    num_batch = 23
    for dim,repeat in [(5,2),(5,3),(5,4)]:
        np_list = [np_rng.normal(size=(dim,num_batch)) for _ in range(repeat)]

        tmp0 = [y for i,x in enumerate(np_list) for y in (x,[i,repeat])]
        np_tensor = np.einsum(*tmp0, list(range(repeat+1)), optimize=True)
        ret_ = numqi.matrix_space.project_nd_tensor_to_antisymmetric_basis(np_tensor, rank=repeat)

        ret0 = numqi.matrix_space.project_to_antisymmetric_basis(np_list)
        assert np.abs(ret_-ret0).max() < 1e-10


def test_get_symmetric_basis():
    for dim,repeat in [(5,2),(5,3),(5,4)]:
        z0 = numqi.matrix_space.get_symmetric_basis(dim, repeat)
        assert np.abs(np.linalg.norm(z0, axis=1)-1).max() < 1e-7
        N0 = z0.shape[0]
        z0 = z0.reshape([N0] + [dim]*repeat)
        permutation_index = np.array(numqi.matrix_space.permutation_with_antisymmetric_factor(repeat)[0])
        permutation_reverse_index = np.argsort(permutation_index, axis=1)
        for r_index in permutation_reverse_index:
            z1 = z0.transpose(*([0] + (r_index+1).tolist()))
            assert np.abs(z0-z1).max() < 1e-10


def test_get_antisymmetric_basis():
    for dim,repeat in [(5,2),(5,3),(5,4)]:
        z0 = numqi.matrix_space.get_antisymmetric_basis(dim, repeat)
        assert np.abs(np.linalg.norm(z0, axis=1)-1).max() < 1e-7
        N0 = z0.shape[0]
        z0 = z0.reshape([N0] + [dim]*repeat)
        tmp0 = numqi.matrix_space.permutation_with_antisymmetric_factor(repeat)
        permutation_index = np.array(tmp0[0])
        permutation_reverse_index = np.argsort(permutation_index, axis=1)
        permutation_factor = tmp0[1]
        for r_index,factor in zip(permutation_reverse_index,permutation_factor):
            z1 = z0.transpose(*([0] + (r_index+1).tolist()))*factor
            assert np.abs(z0-z1).max() < 1e-10

def test_tensor2d_project_to_antisym_basis():
    case_list = [(4,4,2),(4,5,2),(7,8,3),(8,8,4)]
    for dim0,dim1,repeat in case_list:
        np_list = [np_rng.normal(size=(dim0,dim1)) for _ in range(repeat)]

        antisym_basis0 = numqi.matrix_space.get_antisymmetric_basis(dim0, repeat)
        antisym_basis1 = numqi.matrix_space.get_antisymmetric_basis(dim1, repeat)
        ret_ = antisym_basis0 @ hf_kron(np_list) @ antisym_basis1.T

        ret0 = numqi.matrix_space.tensor2d_project_to_antisym_basis(np_list)
        assert np.abs(ret_-ret0).max() < 1e-10

def test_has_rank_hierarchical_method():
    matrix_subspace,field = numqi.matrix_space.get_matrix_subspace_example('hierarchy-ex1')
    #if True, at least rank
    assert numqi.matrix_space.has_rank_hierarchical_method(matrix_subspace, rank=2, hierarchy_k=1)

    matrix_subspace,field = numqi.matrix_space.get_matrix_subspace_example('hierarchy-ex3')
    assert not numqi.matrix_space.has_rank_hierarchical_method(matrix_subspace, rank=2, hierarchy_k=1)
    assert not numqi.matrix_space.has_rank_hierarchical_method(matrix_subspace, rank=2, hierarchy_k=2)
    assert numqi.matrix_space.has_rank_hierarchical_method(matrix_subspace, rank=2, hierarchy_k=3)

def benchmark_has_rank_hierarchical_method():
    # table 1 https://arxiv.org/abs/2210.16389v1
    case_list = [(3,3,1), (4,8,3), (5,13,7), (6,20,12), (7,29,18), (8,39,25), (9,50), (10,63)]
    # tmp0 = [(3,3), (4,8), (5,13), (6,20), (7,29), (8,39), (9,50), (10,63)]
    # tmp1 = [(3,1), (4,3), (5,7), (6,12), (7,18)]
    # case_list = [x+(1,) for x in tmp0] + [x+(2,) for x in tmp1]
    time_list = []
    for case_i in case_list:
        dim = case_i[0]
        time_list.append([])
        for r,num_matrix in enumerate(case_i[1:], start=1):
            matrix_subspace = [hf_randc(dim,dim) for _ in range(num_matrix)]
            t0 = time.time()
            numqi.matrix_space.has_rank_hierarchical_method(matrix_subspace, rank=r+1, hierarchy_k=1)
            time_list[-1].append(time.time() - t0)
            tmp0 = time.time() - t0
        tmp0 = ', '.join([f'time(r={r})={x:.4f}s' for r,x in enumerate(time_list[-1],start=1)])
        print(f'[{dim}x{dim}]^{num_matrix} {tmp0}')

    # mac-studio 20230824
    # [3x3]^1 time(r=1)=0.0004s, time(r=2)=0.0007s
    # [4x4]^3 time(r=1)=0.0551s, time(r=2)=0.0278s
    # [5x5]^7 time(r=1)=0.1607s, time(r=2)=0.1374s
    # [6x6]^12 time(r=1)=0.4782s, time(r=2)=0.7921s
    # [7x7]^18 time(r=1)=0.5862s, time(r=2)=1.6246s
    # [8x8]^25 time(r=1)=1.0535s, time(r=2)=4.7641s
    # [9x9]^50 time(r=1)=1.8132s
    # [10x10]^63 time(r=1)=1.6689s

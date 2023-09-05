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
    r = 1
    k = 3
    num_matrix = 4
    np_list = [hf_randc(dimA,dimB).real for _ in range(num_matrix)]
    tmp0 = list(itertools.combinations_with_replacement(list(range(len(np_list))), r+k))
    ret_ = np.stack([numqi.matrix_space.naive_tensor2d_project_to_sym_antisym_basis([np_list[y] for y in x], r) for x in tmp0])

    hf0 = lambda x: np.einsum(x[0], [0,1,2], x[1], [3,2], [0,1,3], optimize=True)
    ret0 = np.stack([hf0(numqi.matrix_space.tensor2d_project_to_sym_antisym_basis(np_list, r, x)) for x in tmp0])
    tmp0 = [-1]+[x for _ in range(k-1) for x in (dimA,dimB)]
    tmp1 = [0]+[2*x+1 for x in range(k-1)]+[2*x+2 for x in range(k-1)]
    basis = numqi.matrix_space.get_symmetric_basis(dimA*dimB, k-1).reshape(tmp0).transpose(tmp1).reshape(-1,(dimA*dimB)**(k-1))
    ret1 = (ret0 @ basis).reshape(ret0.shape[:3] + (dimA**(k-1), dimB**(k-1))).transpose(0,1,3,2,4).reshape(ret0.shape[0], -1)
    assert np.abs(ret_-ret1).max() < 1e-10


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


def test_project_to_symmetric_basis():
    num_batch = 13
    for dim,repeat in [(3,4),(5,2),(5,3),(5,4)]:
        np_list = [np_rng.normal(size=(dim,num_batch)) for _ in range(repeat)]

        tmp0 = [y for i,x in enumerate(np_list) for y in (x,[i,repeat])]
        np_tensor = np.einsum(*tmp0, list(range(repeat+1)), optimize=True)
        ret_ = numqi.matrix_space.project_nd_tensor_to_symmetric_basis(np_tensor, rank=repeat)

        np_tensor = (numqi.matrix_space.get_symmetric_basis(dim, repeat).T @ ret_).reshape([dim]*repeat + [num_batch])
        ret1 = numqi.matrix_space.project_nd_tensor_to_symmetric_basis(np_tensor, rank=repeat)
        assert np.abs(ret_-ret1).max() < 1e-10

        ret0 = numqi.matrix_space.project_to_symmetric_basis(np_list)
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
    case_list = [(4,4,2),(4,5,2),(7,8,3)]
    for dim0,dim1,repeat in case_list:
        np_list = [np_rng.normal(size=(dim0,dim1)) for _ in range(repeat)]

        antisym_basis0 = numqi.matrix_space.get_antisymmetric_basis(dim0, repeat)
        antisym_basis1 = numqi.matrix_space.get_antisymmetric_basis(dim1, repeat)
        ret_ = antisym_basis0 @ hf_kron(np_list) @ antisym_basis1.T

        ret0 = numqi.matrix_space.tensor2d_project_to_antisym_basis(np_list)
        assert np.abs(ret_-ret0).max() < 1e-10

        ret1 = numqi.matrix_space.tensor2d_project_to_antisym_basis([np_list[x] for x in np_rng.permutation(repeat)])
        assert np.abs(ret_-ret1).max() < 1e-10

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
    case_list = [(3,3,1), (4,8,3), (5,13,7), (6,20,12), (7,29,18), (8,39,25), (9,50,33), (10,63,43)]
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
        tmp0 = ', '.join([f'(r={r})={x:.4f}s' for r,x in enumerate(time_list[-1],start=1)])
        print(f'[{dim}x{dim}] {tmp0}')

    # mac-studio 20230826
    # [3x3] (r=1)=0.0005s, (r=2)=0.0002s
    # [4x4] (r=1)=0.0523s, (r=2)=0.0252s
    # [5x5] (r=1)=0.1068s, (r=2)=0.0969s
    # [6x6] (r=1)=0.5067s, (r=2)=0.5393s
    # [7x7] (r=1)=0.5397s, (r=2)=0.8548s
    # [8x8] (r=1)=0.5518s, (r=2)=3.2082s
    # [9x9] (r=1)=0.6019s, (r=2)=15.1913s
    # [10x10] (r=1)=0.8134s, (r=2)=100.3873s

    # github-codespace (4GB RAM, 2CPU core, Intel(R) Xeon(R) Platinum 8272CL CPU @ 2.60GHz)
    # [3x3] (r=1)=0.0013s, (r=2)=0.0004s
    # [4x4] (r=1)=0.0091s, (r=2)=0.0058s
    # [5x5] (r=1)=0.0198s, (r=2)=0.0705s
    # [6x6] (r=1)=0.0798s, (r=2)=0.2911s
    # [7x7] (r=1)=0.1309s, (r=2)=1.4730s
    # [8x8] (r=1)=0.2642s, (r=2)=8.2653s
    # [9x9] (r=1)=0.6389s
    # [10x10] (r=1)=2.0637s

    # table 2 https://arxiv.org/abs/2210.16389v1
    case_list = [(3,4,1), (4,9,4), (5,16,9), (6,25,16), (7,36,25)]
    time_list = []
    for case_i in case_list:
        dim = case_i[0]
        time_list.append([])
        for r,num_matrix in enumerate(case_i[1:], start=1):
            matrix_subspace = [hf_randc(dim,dim) for _ in range(num_matrix)]
            t0 = time.time()
            numqi.matrix_space.has_rank_hierarchical_method(matrix_subspace, rank=r+1, hierarchy_k=2)
            time_list[-1].append(time.time() - t0)
            tmp0 = time.time() - t0
        tmp0 = ', '.join([f'(r={r})={x:.4f}s' for r,x in enumerate(time_list[-1],start=1)])
        print(f'[{dim}x{dim}] {tmp0}')

    # mac-studio 20230826
    # [3x3] (r=1)=0.0029s, (r=2)=0.0004s
    # [4x4] (r=1)=0.5479s, (r=2)=0.1798s
    # [5x5] (r=1)=0.7422s, (r=2)=0.7790s
    # [6x6] (r=1)=2.8619s, (r=2)=8.8310s
    # [7x7] (r=1)=31.4915s, (r=2)=418.1959s

    # github-codespace (4GB RAM, 2CPU core, Intel(R) Xeon(R) Platinum 8272CL CPU @ 2.60GHz)
    # [3x3] (r=1)=0.0092s, (r=2)=0.0101s
    # [4x4] (r=1)=0.0646s, (r=2)=0.0787s
    # [5x5] (r=1)=0.7503s, (r=2)=1.1243s
    # [6x6] (r=1)=12.1101s

def test_is_ABC_completely_entangled_subspace():
    case_list = [(2,2,2,2), (2,2,3,2), (2,2,4,2), (2,2,5,2), (2,2,6,2), (2,2,7,2)]
    for dimA,dimB,dimC,kmax in case_list:
        np_list = numqi.matrix_space.get_completed_entangled_subspace(dimA, dimB, dimC, tag_reduce=True)
        assert not numqi.matrix_space.is_ABC_completely_entangled_subspace(np_list, hierarchy_k=kmax-1)
        assert numqi.matrix_space.is_ABC_completely_entangled_subspace(np_list, hierarchy_k=kmax)


def benchmark_is_ABC_completely_entangled_subspace():
    case_list = [(2,2,2,2), (2,2,3,2), (2,2,4,2), (2,2,5,2), (2,2,6,2), (2,2,7,2),
                (2,2,8,2), (2,2,9,2),]# (2,3,3,3), (2,3,4,3), (2,3,5,3)]
    info_list = []
    for dimA,dimB,dimC,kmax in case_list:
        np_list = numqi.matrix_space.get_completed_entangled_subspace(dimA, dimB, dimC, tag_reduce=True)
        for k in [kmax-1,kmax]:
            t0 = time.time()
            ret = numqi.matrix_space.is_ABC_completely_entangled_subspace(np_list, hierarchy_k=k)
            info_list.append((dimA,dimB,dimC,k, ret, time.time()-t0))
        print(f'[{dimA}x{dimB}x{dimC}] {info_list[-2][-2]}@(k={kmax-1}) {info_list[-1][-2]}@(k={kmax}) time(k={kmax})={info_list[-1][-1]:.2f}s')
    # mac-studio 20230826
    # [2x2x2] False@(k=1) True@(k=2) time(k=2)=0.01s
    # [2x2x3] False@(k=1) True@(k=2) time(k=2)=0.22s
    # [2x2x4] False@(k=1) True@(k=2) time(k=2)=0.52s
    # [2x2x5] False@(k=1) True@(k=2) time(k=2)=0.55s
    # [2x2x6] False@(k=1) True@(k=2) time(k=2)=0.69s
    # [2x2x7] False@(k=1) True@(k=2) time(k=2)=0.96s
    # [2x2x8] False@(k=1) True@(k=2) time(k=2)=1.40s
    # [2x2x9] False@(k=1) True@(k=2) time(k=2)=2.18s
    # [2x3x3] False@(k=2) True@(k=3) time(k=3)=1.30s
    # [2x3x4] False@(k=2) True@(k=3) time(k=3)=8.59s
    # [2x3x5] False@(k=2) True@(k=3) time(k=3)=328.46s

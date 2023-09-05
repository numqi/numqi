import os
import json
import time
import concurrent.futures
import multiprocessing
import numpy as np
import torch
import scipy.sparse

import numqi.matrix_space
import numqi.optimize

# cannot be torch.linalg.norm()**2 nan when calculating the gradient when norm is almost zero
# see https://github.com/pytorch/pytorch/issues/99868
# hf_torch_norm_square = lambda x: torch.dot(x.conj(), x).real
hf_torch_norm_square = lambda x: torch.sum((x.conj() * x).real)


def get_matrix_list_indexing(mat_list, index):
    if isinstance(mat_list, np.ndarray):
        index = np.asarray(index)
        assert (mat_list.ndim==3) and (index.ndim==1)
        ret = mat_list[index]
    else:
        ret = [mat_list[x] for x in index]
    return ret


class DetectUDPModel(torch.nn.Module):
    def __init__(self, basis_orth, dtype='float32', device='cpu'):
        super().__init__()
        self.is_torch = isinstance(basis_orth, torch.Tensor)
        self.use_sparse = self.is_torch and basis_orth.is_sparse #use sparse only when is a torch.tensor
        assert basis_orth.ndim==3
        assert dtype in {'float32','float64'}
        self.dtype = torch.float32 if dtype=='float32' else torch.float64
        self.cdtype = torch.complex64 if dtype=='float32' else torch.complex128
        self.device = device
        self.basis_orth_conj = self._setup_basis_orth_conj(basis_orth)
        np_rng = np.random.default_rng()
        hf0 = lambda *size: torch.nn.Parameter(torch.tensor(np_rng.uniform(-1, 1, size=size), dtype=self.dtype))
        self.theta = hf0(4, basis_orth[0].shape[0])
        self.EVL = hf0(2)
        self.matH = None

    def _setup_basis_orth_conj(self, basis_orth):
        # <A,B>=tr(AB^H)=sum_ij (A_ij, conj(B_ij))
        if self.use_sparse:
            assert self.is_torch
            assert self.device=='cpu', f'sparse tensor not support device "{self.device}"'
            index = basis_orth.indices()
            shape = basis_orth.shape
            tmp0 = torch.stack([index[0], index[1]*shape[2] + index[2]])
            basis_orth_conj = torch.sparse_coo_tensor(tmp0, basis_orth.values().conj().to(self.cdtype), (shape[0], shape[1]*shape[2]))
        else:
            if self.is_torch:
                basis_orth_conj = basis_orth.conj().reshape(basis_orth.shape[0],-1).to(device=self.device, dtype=self.cdtype)
            else:
                basis_orth_conj = torch.tensor(basis_orth.conj().reshape(basis_orth.shape[0],-1), dtype=self.cdtype, device=self.device)
        return basis_orth_conj

    def forward(self):
        tmp0 = self.theta[0] + 1j*self.theta[1]
        EVC0 = tmp0 / torch.linalg.norm(tmp0)
        tmp0 = self.theta[2] + 1j*self.theta[3]
        tmp0 = tmp0 - torch.dot(EVC0.conj(), tmp0) * EVC0
        EVC1 = tmp0 / torch.linalg.norm(tmp0)
        tmp0 = torch.nn.functional.softplus(self.EVL)
        EVL = tmp0 / torch.linalg.norm(tmp0)
        matH = EVC0.reshape(-1,1)*(EVC0.conj()*EVL[0]) - EVC1.reshape(-1,1)*(EVC1.conj()*EVL[1])
        self.matH = matH
        loss = hf_torch_norm_square(self.basis_orth_conj @ matH.reshape(-1))
        return loss


def _check_UDA_UDP_matrix_subspace_one(is_uda, matB, num_repeat, converge_tol,
            early_stop_threshold, udp_use_vector_model, dtype, tag_single_thread):
    if tag_single_thread and torch.get_num_threads()!=1:
        torch.set_num_threads(1)
    if len(matB)==0:
        ret = True,np.inf
    else:
        rank = (0,matB[0].shape[0]-1,1) if is_uda else (0,1,1)
        if not isinstance(matB, np.ndarray): #sparse matrix
            index = np.concatenate([np.stack([x*np.ones(len(y.row),dtype=np.int64), y.row, y.col]) for x,y in enumerate(matB)], axis=1)
            value = np.concatenate([x.data for x in matB])
            matB = torch.sparse_coo_tensor(index, value, (len(matB), *matB[0].shape)).coalesce()
        if udp_use_vector_model:
            model = DetectUDPModel(matB, dtype)
        else:
            model = numqi.matrix_space.DetectRankModel(matB, space_char='C_H', rank=rank, dtype=dtype)
        theta_optim = numqi.optimize.minimize(model, theta0='normal', num_repeat=num_repeat,
                tol=converge_tol, early_stop_threshold=early_stop_threshold, print_every_round=0, print_freq=0)
        ret = theta_optim.fun>early_stop_threshold, theta_optim.fun
        # always assume that identity is measured, and matrix subspace A is traceless, so no need to test loss(0,n,0)
    return ret


def _check_UDA_UDP_matrix_subspace_parallel(is_uda, matB, num_repeat, converge_tol,
            early_stop_threshold, udp_use_vector_model, dtype, num_worker, tag_single_thread):
    if isinstance(matB, np.ndarray) or scipy.sparse.issparse(matB[0]):
        is_single_item = True
        if isinstance(matB, np.ndarray):
            assert (matB.ndim==3) and (matB.shape[1]==matB.shape[2])
            matB_list = [matB]
        else:
            assert all((x.shape[0]==x.shape[1]) and (x.format=='coo') for x in matB)
            matB_list = [matB]
    else:
        is_single_item = False
        if isinstance(matB[0], np.ndarray):
            assert all(((x.ndim==3) and (x.shape[1]==x.shape[2])) for x in matB)
        else:
            assert all((y.shape[0]==y.shape[1]) and (y.format=='coo') for x in matB for y in x)
        matB_list = matB
    assert len(matB_list)>0
    kwargs = {'is_uda':is_uda, 'num_repeat':num_repeat, 'converge_tol':converge_tol, 'early_stop_threshold':early_stop_threshold,
            'udp_use_vector_model':udp_use_vector_model, 'dtype':dtype, 'tag_single_thread':tag_single_thread}
    num_worker = min(num_worker, len(matB_list))
    if num_worker == 1:
        time_start = time.time()
        num_pass = 0
        ret = []
        for matB in matB_list:
            ret.append(_check_UDA_UDP_matrix_subspace_one(matB=matB, **kwargs))
            if ret[-1][0]:
                tmp0 = time.time()-time_start
                num_pass = num_pass + 1
                print(f'[{tmp0:.1f}] {num_pass}/{len(ret)}/{len(matB_list)}')
    else:
        # https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_worker, mp_context=multiprocessing.get_context('spawn')) as executor:
            job_list = [executor.submit(_check_UDA_UDP_matrix_subspace_one, matB=x, **kwargs) for x in matB_list]
            jobid_to_result = dict()
            time_start = time.time()
            num_pass = 0
            for job_i in concurrent.futures.as_completed(job_list):
                ret_i = job_i.result()
                jobid_to_result[id(job_i)] = ret_i
                if ret_i[0]:
                    tmp0 = time.time()-time_start
                    num_pass = num_pass + 1
                    print(f'[{tmp0:.1f}] {num_pass}/{len(jobid_to_result)}/{len(job_list)}')
            ret = [jobid_to_result[id(x)] for x in job_list]
    if is_single_item:
        ret = ret[0]
    return ret

def check_UDA_matrix_subspace(matB, num_repeat, converge_tol=1e-5, early_stop_threshold=1e-2, dtype='float32',
                                    udp_use_vector_model=False, num_worker=1, tag_single_thread=True):
    is_uda = True
    udp_use_vector_model = False #ignore this parameter
    ret = _check_UDA_UDP_matrix_subspace_parallel(is_uda, matB, num_repeat, converge_tol,
            early_stop_threshold, udp_use_vector_model, dtype, num_worker, tag_single_thread)
    return ret


def check_UDP_matrix_subspace(matB, num_repeat, converge_tol=1e-5, early_stop_threshold=1e-2, dtype='float32',
                                    udp_use_vector_model=False, num_worker=1, tag_single_thread=True):
    is_uda = False
    ret = _check_UDA_UDP_matrix_subspace_parallel(is_uda, matB, num_repeat, converge_tol,
            early_stop_threshold, udp_use_vector_model, dtype, num_worker, tag_single_thread)
    return ret


def save_index_to_file(file, key_str=None, index=None):
    if os.path.exists(file):
        with open(file, 'r', encoding='utf-8') as fid:
            all_data = json.load(fid)
    else:
        all_data = dict()
    if (index is not None) and len(index)>0:
        if isinstance(index[0], int): #[2,3,4]
            index_batch = [[int(x) for x in index]]
        elif isinstance(index[0], str): #["2 3 4"]
            index_batch = [[int(y) for y in x.split(' ')] for x in index]
        else: #[[2,3,4]]
            index_batch = [[int(y) for y in x] for x in index]
        data_i = [[int(y) for y in x.split(' ')] for x in all_data.get(key_str, [])] + index_batch
        hf1 = lambda x: (len(x),)+x
        tmp0 = sorted(set([tuple(sorted(set(x))) for x in data_i]), key=hf1)
        all_data[key_str] = [' '.join(str(y) for y in x) for x in tmp0]
        with open(file, 'w', encoding='utf-8') as fid:
            json.dump(all_data, fid, indent=2)
    if key_str is None:
        ret = {k:[[int(y) for y in x.split(' ')] for x in v] for k,v in all_data.items()}
    else:
        ret = [[int(y) for y in x.split(' ')] for x in all_data.get(key_str,[])]
    return ret


def remove_index_from_file(file, key_str, index):
    assert os.path.exists(file)
    assert len(index)>0
    with open(file, 'r', encoding='utf-8') as fid:
        all_data = json.load(fid)
    if isinstance(index[0], int): #[2,3,4]
        index_batch = [[int(x) for x in index]]
    elif isinstance(index[0], str): #["2 3 4"]
        index_batch = [[int(y) for y in x.split(' ')] for x in index]
    else: #[[2,3,4]]
        index_batch = [[int(y) for y in x] for x in index]
    index_set = {tuple(sorted(set(x))) for x in index_batch}
    data_i = {tuple(sorted(set([int(y) for y in x.split(' ')]))) for x in all_data.get(key_str, [])}
    hf1 = lambda x: (len(x),)+x
    tmp0 = sorted(data_i-index_set, key=hf1)
    all_data[key_str] = [' '.join(str(y) for y in x) for x in tmp0]
    with open(file, 'w', encoding='utf-8') as fid:
        json.dump(all_data, fid, indent=2)


def _find_UDA_UDP_over_matrix_basis_one(is_uda, matrix_basis, num_repeat, num_random_select, indexF, tag_reduce,
            early_stop_threshold, converge_tol, last_converge_tol, last_num_repeat,
            udp_use_vector_model, dtype, tag_single_thread, tag_print):
    if tag_single_thread:
        torch.set_num_threads(1)
    if last_converge_tol is None:
        last_converge_tol = converge_tol/10
    if last_num_repeat is None:
        last_num_repeat = num_repeat*5
    np_rng = np.random.default_rng()
    N0 = len(matrix_basis)
    if not isinstance(matrix_basis, np.ndarray): #list of sparse matrix
        assert not tag_reduce, 'tag_reduce=True is not compatible with sparse matrix'

    time_start = time.time()
    if indexF is not None:
        indexF = set([int(x) for x in indexF])
        assert all(0<=x<N0 for x in indexF)
    else:
        indexF = set()
    indexB = set(list(range(N0)))
    kwargs = {'is_uda':is_uda, 'num_repeat':num_repeat, 'converge_tol':converge_tol, 'early_stop_threshold':early_stop_threshold,
        'udp_use_vector_model':udp_use_vector_model, 'dtype':dtype, 'tag_single_thread':False}
    # tag_single_thread is already set
    index_B_minus_F = np.array(sorted(indexB - set(indexF)), dtype=np.int64)
    assert len(index_B_minus_F)>=num_random_select
    while num_random_select>0:
        selectX = set(np_rng.choice(index_B_minus_F, size=num_random_select, replace=False, shuffle=False).tolist())
        matB = get_matrix_list_indexing(matrix_basis, sorted(indexB-selectX))
        if tag_reduce:
            matB,matB_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matB, field='real', zero_eps=1e-10)
            assert space_char in {'R_T','C_H'}
        if (tag_reduce and len(matB_orth)==0) or (_check_UDA_UDP_matrix_subspace_one(matB=matB, **kwargs)[0]):
            indexB = indexB - selectX
            break
    while True:
        tmp0 = sorted(indexB - indexF)
        if len(tmp0)==0:
            break
        selectX = tmp0[np_rng.integers(len(tmp0))]
        matB = get_matrix_list_indexing(matrix_basis, sorted(indexB-{selectX}))
        if tag_reduce:
            matB,matB_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matB, field='real', zero_eps=1e-10)
            assert space_char in {'R_T','C_H'}
        if tag_reduce and (matB_orth.shape[0]==0):
            ret_hfT = True,np.inf
        else:
            ret_hfT = _check_UDA_UDP_matrix_subspace_one(matB=matB, **kwargs)
        if ret_hfT[0]:
            indexB = indexB - {selectX}
            if tag_print:
                tmp0 = time.time() - time_start
                tmp1 = 'loss(n-1,1)' if is_uda else 'loss(1,1)'
                print(f'[{tmp0:.1f}s/{len(indexB)}/{len(indexF)}] {tmp1}={ret_hfT[1]:.5f}')
        else:
            indexF = indexF | {selectX}
    matB = get_matrix_list_indexing(matrix_basis, sorted(indexB))
    kwargs['converge_tol'] = last_converge_tol
    kwargs['num_repeat'] = last_num_repeat
    ret_hfT = _check_UDA_UDP_matrix_subspace_one(matB=matB, **kwargs)
    if tag_print and ret_hfT[0]:
        tmp0 = time.time() - time_start
        tmp1 = 'loss(n-1,1)' if is_uda else 'loss(1,1)'
        print(f'[{tmp0:.1f}s/{len(indexB)}/{len(indexF)}] {tmp1}={ret_hfT[1]:.5f} [{len(indexB)}] {sorted(indexB)}')
    ret = sorted(indexB) if ret_hfT[0] else None
    return ret


def _find_UDA_UDP_over_matrix_basis(is_uda, num_round, matrix_basis, num_repeat, num_random_select, indexF, tag_reduce,
            early_stop_threshold, converge_tol, last_converge_tol, last_num_repeat, udp_use_vector_model,
            dtype, num_worker, key, file, tag_single_thread):
    num_worker = min(num_worker, num_round)
    assert num_worker>=1
    if isinstance(matrix_basis,np.ndarray):
        assert (matrix_basis.ndim==3) and (matrix_basis.shape[1]==matrix_basis.shape[2])
        assert np.abs(matrix_basis-matrix_basis.transpose(0,2,1).conj()).max() < 1e-10
    else:
        # should be scipy.sparse.coo_matrix
        assert not tag_reduce, 'tag_reduce not support sparse data'
        assert all(scipy.sparse.issparse(x) and (x.format=='coo') and (x.shape[0]==x.shape[1]) for x in matrix_basis)
        for x in matrix_basis:
            tmp0 = (x-x.T.conj()).data
            assert (len(tmp0)==0) or np.abs(tmp0).max() < 1e-10
    ret = []
    kwargs = {'is_uda':is_uda, 'matrix_basis':matrix_basis, 'num_repeat':num_repeat, 'num_random_select':num_random_select,
            'indexF':indexF, 'tag_reduce':tag_reduce, 'early_stop_threshold':early_stop_threshold, 'converge_tol':converge_tol,
            'last_converge_tol':last_converge_tol, 'last_num_repeat':last_num_repeat, 'udp_use_vector_model':udp_use_vector_model,
            'dtype':dtype, 'tag_single_thread':tag_single_thread}
    if num_worker==1:
        kwargs['tag_print'] = True
        for _ in range(num_round):
            ret_i = _find_UDA_UDP_over_matrix_basis_one(**kwargs)
            if ret_i is not None:
                ret.append(ret_i)
                if key is not None:
                    assert file is not None
                    save_index_to_file(file, key, ret_i)
    else:
        kwargs['tag_print'] = False
        kwargs['tag_single_thread'] = True
        # https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_worker, mp_context=multiprocessing.get_context('spawn')) as executor:
            job_list = [executor.submit(_find_UDA_UDP_over_matrix_basis_one, **kwargs) for _ in range(num_round)]
            time_start = time.time()
            for ind0,job_i in enumerate(concurrent.futures.as_completed(job_list)):
                ret_i = job_i.result()
                if ret_i is not None:
                    ret.append(ret_i)
                    tmp0 = time.time() - time_start
                    print(f'[round-{ind0}][{tmp0:.1f}s/{len(ret_i)}] {sorted(ret_i)}')
                    if key is not None:
                        assert file is not None
                        save_index_to_file(file, key, ret_i)
    ret = sorted(ret, key=len)
    return ret


# TODO remove indexF
def find_UDA_over_matrix_basis(num_round, matrix_basis, num_repeat, num_random_select, indexF=None, tag_reduce=True,
            early_stop_threshold=0.01, converge_tol=1e-5, last_converge_tol=None, last_num_repeat=None,
            udp_use_vector_model=False, dtype='float32', num_worker=1, key=None, file=None, tag_single_thread=True):
    is_uda = True
    udp_use_vector_model = False
    ret = _find_UDA_UDP_over_matrix_basis(is_uda, num_round, matrix_basis, num_repeat, num_random_select, indexF, tag_reduce,
            early_stop_threshold, converge_tol, last_converge_tol, last_num_repeat, udp_use_vector_model,
            dtype, num_worker, key, file, tag_single_thread)
    return ret


def find_UDP_over_matrix_basis(num_round, matrix_basis, num_repeat, num_random_select, indexF=None, tag_reduce=True,
            early_stop_threshold=0.01, converge_tol=1e-5, last_converge_tol=None, last_num_repeat=None,
            udp_use_vector_model=False, dtype='float32', num_worker=1, key=None, file=None, tag_single_thread=True):
    is_uda = False
    ret = _find_UDA_UDP_over_matrix_basis(is_uda, num_round, matrix_basis, num_repeat, num_random_select, indexF, tag_reduce,
            early_stop_threshold, converge_tol, last_converge_tol, last_num_repeat, udp_use_vector_model,
            dtype, num_worker, key, file, tag_single_thread)
    return ret


def get_UDA_theta_optim_special_EVC(matB, num_repeat=100, tol=1e-12, early_stop_threshold=1e-10, tag_single_thread=True, print_every_round=0):
    if tag_single_thread and torch.get_num_threads()!=1:
        torch.set_num_threads(1)
    if not isinstance(matB, np.ndarray): #sparse matrix
        index = np.concatenate([np.stack([x*np.ones(len(y.row),dtype=np.int64), y.row, y.col]) for x,y in enumerate(matB)], axis=1)
        value = np.concatenate([x.data for x in matB])
        matB = torch.sparse_coo_tensor(index, value, (len(matB), *matB[0].shape)).coalesce()
    model = numqi.matrix_space.DetectRankModel(matB, space_char='C_H', rank=(0, matB[0].shape[0]-1,1), dtype='float64')
    theta_optim = numqi.optimize.minimize(model, theta0='normal', num_repeat=num_repeat,
            tol=tol, early_stop_threshold=early_stop_threshold, print_every_round=print_every_round, print_freq=0)
    model()
    matH = model.matH.detach().cpu().numpy().copy()
    EVL,EVC = np.linalg.eigh(matH)
    assert (EVL[0]<=0) and (np.abs(matH @ EVC[:,0] - EVC[:,0]*EVL[0]).max() < 1e-8)
    return theta_optim, EVC[:,0]


# pauli
# num_qubit=3, num_repeat=10, num_random_select=10
# num_qubit=4, num_repeat=80, num_random_select=80
# device = 'cpu' #slow on gpu-gtx3060

# gellmann
# num_qudit=2, dim_qudit=3, num_repeat=80, num_random_select=10, early_stop_threshold=0.001

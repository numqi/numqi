import time
import concurrent.futures
import multiprocessing
import numpy as np
import torch
import scipy.sparse

import numqi.matrix_space
import numqi.random
import numqi.optimize
from ._internal import save_index_to_file, get_matrix_list_indexing

# cannot be torch.linalg.norm()**2 nan when calculating the gradient when norm is almost zero
# not using torch.dot because torch.dot(complex) might be wrong for see https://github.com/pytorch/pytorch/issues/99868
# hf_torch_norm_square = lambda x: torch.dot(x.conj(), x).real
hf_torch_norm_square = lambda x: torch.sum((x.conj() * x).real)

class _UDAEigenvalueManifold(torch.nn.Module):
    def __init__(self, dim:int, dtype:torch.dtype=torch.float32):
        super().__init__()
        assert dtype in {torch.float32, torch.float64}
        dim = int(dim)
        assert dim>=2
        tmp0 = np.random.default_rng().uniform(0, 1, size=dim-1)
        self.theta = torch.nn.Parameter(torch.tensor(tmp0, dtype=dtype))

    def forward(self):
        tmp0 = torch.nn.functional.softplus(self.theta)
        tmp1 = torch.sum(tmp0)
        tmp2 = 1/torch.sqrt(torch.dot(tmp0,tmp0) + tmp1*tmp1)
        ret = torch.concat([tmp0, -tmp1.reshape(1)], dim=0) * tmp2
        return ret


class DetectUDModel(torch.nn.Module):
    def __init__(self, dim:int, is_uda:bool, dtype:str='float32'):
        super().__init__()
        assert dtype in {'float32','float64'}
        self.dtype = torch.float32 if dtype=='float32' else torch.float64
        self.cdtype = torch.complex64 if dtype=='float32' else torch.complex128
        self.dim = dim
        if is_uda:
            self.manifold_EVC = numqi.manifold.SpecialOrthogonal(dim, method='exp', dtype=self.cdtype)
            self.manifold_EVL = _UDAEigenvalueManifold(dim, dtype=self.dtype)
        else:
            self.manifold = numqi.manifold.Stiefel(dim, 2, method='qr', dtype=self.cdtype)
            self._s12 = 1/np.sqrt(2).item()
        self.matH = None
        self.mat_list_conj = None

    def set_mat_list(self, mat_list):
        # <A,B>=tr(AB^H)=sum_ij (A_ij, conj(B_ij))
        # mat_list: 3d-nparray, list of sparse
        assert self.dim==mat_list[0].shape[0]
        dim = self.dim
        N0 = len(mat_list)
        if scipy.sparse.issparse(mat_list[0]):
            index = np.concatenate([np.stack([x*np.ones(len(y.row),dtype=np.int64), y.row, y.col]) for x,y in enumerate(mat_list)], axis=1)
            value = np.concatenate([x.data for x in mat_list])
            mat_list = torch.sparse_coo_tensor(index, value, (N0, dim, dim)).coalesce()
            # TODO skip create mat_list
            index = mat_list.indices()
            tmp0 = torch.stack([index[0], index[1]*dim + index[2]])
            mat_list_conj = torch.sparse_coo_tensor(tmp0, mat_list.values().conj().to(self.cdtype), (N0, dim*dim))
        else:
            mat_list = np.asarray(mat_list)
            assert mat_list.ndim==3
            mat_list_conj = torch.tensor(mat_list.conj().reshape(mat_list.shape[0],-1), dtype=self.cdtype)
        self.mat_list_conj = mat_list_conj

    def forward(self):
        if hasattr(self, 'manifold_EVL'): #UDA
            EVC = self.manifold_EVC()
            EVL = self.manifold_EVL()
            matH = (EVC * EVL) @ (EVC.conj().T)
        else: #UDP
            EVC0,EVC1 = self.manifold().T
            matH = EVC0.reshape(-1,1)*(EVC0.conj()*self._s12) - EVC1.reshape(-1,1)*(EVC1.conj()*self._s12)
        self.matH = matH.detach()
        loss = hf_torch_norm_square(self.mat_list_conj @ matH.reshape(-1))
        return loss

def _is_mat_list_full_rank(mat_list, zero_eps=1e-7):
    dim = mat_list[0].shape[0]
    if len(mat_list)>=dim*dim:
        if isinstance(mat_list, np.ndarray):
            tmp0 = mat_list.reshape(-1, dim*dim)
            tmp0 = tmp0 @ tmp0.T.conj()
        else: # scipy.sparse
            tmp0 = scipy.sparse.vstack([x.reshape(1,-1) for x in mat_list])
            tmp0 = (tmp0 @ tmp0.T.conj()).toarray()
        ret = np.abs(np.diag(scipy.linalg.lu(tmp0)[2])).min() > zero_eps #full rank
    else:
        ret = False
    return ret

def _check_UD_one(is_uda, mat_list, num_repeat, converge_tol, early_stop_threshold, dtype, tag_single_thread, np_rng, print_every_round):
    # always assume that identity is measured
    # mat_list: 3d-array or list of sparse matrix
    if tag_single_thread and torch.get_num_threads()!=1:
        torch.set_num_threads(1)
    if len(mat_list)==0:
        ret = False,0,None
    elif _is_mat_list_full_rank(mat_list):
        ret = True, np.inf, None #TODO, np.inf is not a good choice
    else:
        model = DetectUDModel(mat_list[0].shape[0], is_uda, dtype)
        model.set_mat_list(mat_list)
        theta_optim = numqi.optimize.minimize(model, theta0='normal', num_repeat=num_repeat, tol=converge_tol,
                early_stop_threshold=early_stop_threshold, print_every_round=print_every_round, print_freq=0, seed=np_rng)
        ret = theta_optim.fun>early_stop_threshold, theta_optim.fun, theta_optim.x
    return ret


def check_UD(kind:str, mat_list, num_repeat:int, converge_tol:float=1e-5, early_stop_threshold:float=1e-2,
                dtype:str='float32', num_worker:int=1, tag_single_thread:bool=True,
                tag_print:int=0, return_model:bool=False, seed=None):
    kind = kind.lower()
    assert kind in {'uda','udp'}
    is_uda = kind=='uda'
    np_rng = np.random.default_rng(seed)
    matS = mat_list #matrix subspace (matS)
    if isinstance(matS, np.ndarray) or scipy.sparse.issparse(matS[0]):
        is_single_item = True
        if isinstance(matS, np.ndarray):
            assert (matS.ndim==3) and (matS.shape[1]==matS.shape[2])
            matS_list = [matS]
        else:
            assert all((x.shape[0]==x.shape[1]) and (x.format=='coo') for x in matS)
            matS_list = [matS]
    else:
        is_single_item = False
        if isinstance(matS[0], np.ndarray):
            assert all(((x.ndim==3) and (x.shape[1]==x.shape[2])) for x in matS)
        else:
            assert all((y.shape[0]==y.shape[1]) and (y.format=='coo') for x in matS for y in x)
        matS_list = matS
    assert len(matS_list)>0
    kwargs = {'is_uda':is_uda, 'num_repeat':num_repeat, 'converge_tol':converge_tol, 'early_stop_threshold':early_stop_threshold,
            'dtype':dtype, 'tag_single_thread':tag_single_thread}
    kwargs['print_every_round'] = 0 if num_worker>1 else int(tag_print>1)
    num_worker = min(num_worker, len(matS_list))
    if num_worker == 1:
        time_start = time.time()
        num_pass = 0
        ret = []
        for matS in matS_list:
            ret.append(_check_UD_one(mat_list=matS, **kwargs, np_rng=np_rng))
            if ret[-1][0]:
                tmp0 = time.time()-time_start
                num_pass = num_pass + 1
                if tag_print>=1:
                    print(f'[{tmp0:.1f}] {num_pass}/{len(ret)}/{len(matS_list)}')
    else:
        # https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_worker, mp_context=multiprocessing.get_context('spawn')) as executor:
            job_list = [executor.submit(_check_UD_one, mat_list=x, **kwargs, np_rng=y) for x,y in zip(matS_list, np_rng.spawn(len(matS_list)))]
            jobid_to_result = dict()
            time_start = time.time()
            num_pass = 0
            for job_i in concurrent.futures.as_completed(job_list):
                ret_i = job_i.result()
                jobid_to_result[id(job_i)] = ret_i
                if ret_i[0]:
                    tmp0 = time.time()-time_start
                    num_pass = num_pass + 1
                    if tag_print:
                        print(f'[{tmp0:.1f}] {num_pass}/{len(jobid_to_result)}/{len(job_list)}')
            ret = [jobid_to_result[id(x)] for x in job_list]
    if return_model:
        ret_old = ret
        ret = []
        for (x0,x1,x2),matS in zip(ret_old,matS_list):
            model = DetectUDModel(matS[0].shape[0], is_uda=is_uda, dtype=dtype)
            model.set_mat_list(matS)
            numqi.optimize.set_model_flat_parameter(model, x2)
            with torch.no_grad():
                model() #set .matH
            ret.append((x0,x1,model))
    else:
        ret = [x[:2] for x in ret]
    if is_single_item:
        ret = ret[0]
    return ret

def _find_optimal_UD_one(is_uda, mat_list, num_repeat, num_init_sample, indexF, early_stop_threshold,
            converge_tol, last_converge_tol, last_num_repeat, dtype, tag_single_thread, tag_print, np_rng):
    if tag_single_thread and torch.get_num_threads()!=1:
        torch.set_num_threads(1)
    N0 = len(mat_list)

    time_start = time.time()
    if indexF is not None:
        indexF = set([int(x) for x in indexF])
        assert all(0<=x<N0 for x in indexF)
    else:
        indexF = set()
    indexA = set(list(range(N0)))
    kwargs = {'is_uda':is_uda, 'num_repeat':num_repeat, 'converge_tol':converge_tol, 'early_stop_threshold':early_stop_threshold,
        'dtype':dtype, 'tag_single_thread':False, 'print_every_round':0}
    # tag_single_thread is already set
    index_A_minus_F = np.array(sorted(indexA - set(indexF)), dtype=np.int64)
    assert len(index_A_minus_F)>=num_init_sample
    while num_init_sample>0:
        selectX = set(np_rng.choice(index_A_minus_F, size=num_init_sample, replace=False, shuffle=False).tolist())
        if _check_UD_one(mat_list=get_matrix_list_indexing(mat_list, sorted(indexA-selectX)), **kwargs, np_rng=np_rng)[0]:
            indexA = indexA - selectX
            break
    while True:
        tmp0 = sorted(indexA - indexF)
        if len(tmp0)==0:
            break
        selectX = tmp0[np_rng.integers(len(tmp0))]
        ret_hfT = _check_UD_one(mat_list=get_matrix_list_indexing(mat_list, sorted(indexA-{selectX})), **kwargs, np_rng=np_rng)
        if ret_hfT[0]:
            indexA = indexA - {selectX}
            if tag_print:
                tmp0 = time.time() - time_start
                print(f'[{tmp0:.1f}s/{len(indexA)}/{len(indexF)}] loss={ret_hfT[1]:.5f}')
        else:
            indexF = indexF | {selectX}
    kwargs['converge_tol'] = last_converge_tol
    kwargs['num_repeat'] = last_num_repeat
    ret_hfT = _check_UD_one(mat_list=get_matrix_list_indexing(mat_list, sorted(indexA)), **kwargs, np_rng=np_rng)
    if tag_print and ret_hfT[0]:
        tmp0 = time.time() - time_start
        print(f'[{tmp0:.1f}s/{len(indexA)}/{len(indexF)}] loss={ret_hfT[1]:.5f} [{len(indexA)}] {sorted(indexA)}')
    ret = sorted(indexA) if ret_hfT[0] else None
    return ret


def find_optimal_UD(kind:str, num_round:int, mat_list, num_repeat:int, num_init_sample:int=0, indexF=None,
            early_stop_threshold:float=0.01, converge_tol:float=1e-5, last_converge_tol=None, last_num_repeat=None,
            dtype:str='float32', num_worker:int=1, key:(str|None)=None, file:(str|None)=None,
            tag_single_thread:bool=True, tag_print:bool=False, seed=None):
    kind = kind.lower()
    assert kind in {'uda', 'udp'}
    is_uda = kind=='uda'
    num_worker = min(num_worker, num_round)
    np_rng = numqi.random.get_numpy_rng(seed)
    assert num_worker>=1
    if num_worker>1:
        tag_print = False
        tag_single_thread = True
    if last_converge_tol is None:
        last_converge_tol = converge_tol/10
    if last_num_repeat is None:
        last_num_repeat = num_repeat*5

    if isinstance(mat_list,np.ndarray):
        assert (mat_list.ndim==3) and (mat_list.shape[1]==mat_list.shape[2])
        assert np.abs(mat_list-mat_list.transpose(0,2,1).conj()).max() < 1e-10
    else:
        # should be scipy.sparse.coo_matrix
        assert all(scipy.sparse.issparse(x) and (x.format=='coo') and (x.shape[0]==x.shape[1]) for x in mat_list)
        for x in mat_list:
            tmp0 = (x-x.T.conj()).data
            assert (len(tmp0)==0) or np.abs(tmp0).max() < 1e-10
    ret = []
    kwargs = dict(is_uda=is_uda, mat_list=mat_list, num_repeat=num_repeat, num_init_sample=num_init_sample,
                  indexF=indexF, early_stop_threshold=early_stop_threshold, converge_tol=converge_tol,
                  last_converge_tol=last_converge_tol, last_num_repeat=last_num_repeat,
                  dtype=dtype, tag_single_thread=tag_single_thread, tag_print=tag_print)
    if num_worker==1:
        for _ in range(num_round):
            ret_i = _find_optimal_UD_one(**kwargs, np_rng=np_rng)
            if ret_i is not None:
                ret.append(ret_i)
                if key is not None:
                    assert file is not None
                    save_index_to_file(file, key, ret_i)
    else:
        # https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_worker, mp_context=multiprocessing.get_context('spawn')) as executor:

            job_list = [executor.submit(_find_optimal_UD_one, **kwargs, np_rng=x) for x in range(np_rng.spawn(num_round))]
            time_start = time.time()
            for ind0,job_i in enumerate(concurrent.futures.as_completed(job_list)):
                ret_i = job_i.result()
                if ret_i is not None:
                    ret.append(ret_i)
                    tmp0 = time.time() - time_start
                    if tag_print:
                        print(f'[round-{ind0}][{tmp0:.1f}s/{len(ret_i)}] {sorted(ret_i)}')
                    if key is not None:
                        assert file is not None
                        save_index_to_file(file, key, ret_i)
    ret = sorted(ret, key=len)
    return ret

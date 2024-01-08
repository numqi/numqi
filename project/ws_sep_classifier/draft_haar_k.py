import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import concurrent.futures
import torch

import numqi

from utils import is_positive_semi_definite

if torch.get_num_threads()!=1:
    torch.set_num_threads(1)

def _get_cha_distance_single(dm_list, model_kwargs, optim_kwargs):
    # still batch of density matrix, avoid wasting time on starting up
    model_cha = numqi.entangle.AutodiffCHAREE(**model_kwargs)
    ret = []
    for dm in dm_list:
        model_cha.set_dm_target(dm)
        ret.append(numqi.optimize.minimize(model_cha, **optim_kwargs).fun)
    ret = np.array(ret)
    return ret

def get_cha_distance(dm_list, model_kwargs, optim_kwargs, num_worker):
    batch_size = (len(dm_list)+num_worker-1)//num_worker
    tmp0 = int(np.ceil(len(dm_list)/batch_size))
    task_list = [dm_list[(x*batch_size):((x+1)*batch_size)] for x in range(tmp0)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_worker) as executor:
        job_list = [executor.submit(_get_cha_distance_single, x, model_kwargs, optim_kwargs) for x in task_list]
        ret = np.array([y for x in tqdm(job_list) for y in x.result()]) #one-by-one
    return ret

# Fundamental Limitation on the Detectability of Entanglement
# https://doi.org/10.1103/PhysRevLett.129.230503


if __name__=='__main__':
    np_rng = np.random.default_rng()
    dimA = 3
    dimB = 3
    num_sample = 5000
    haar_k_list = list(range(2, 5*dimA*dimB, 3))
    # haar_k_list = [23]
    num_worker = 12
    cha_threshold = 1e-12

    ret_ppt = []
    ret_cha = []
    dm_list = []
    model_cha = numqi.entangle.AutodiffCHAREE(dimA, dimB, distance_kind='gellmann')
    model_kwargs = dict(dim0=dimA, dim1=dimB, distance_kind='gellmann')
    kwargs = dict(theta0='uniform', tol=1e-14, num_repeat=3, print_every_round=0, early_stop_threshold=cha_threshold)
    for haar_k in haar_k_list:
        print(f"k={haar_k}")
        tmp0 = [numqi.random.rand_density_matrix(dimA*dimB, k=haar_k, seed=np_rng) for _ in range(num_sample)]
        dm_list.append(tmp0)
        ret_ppt.append([is_positive_semi_definite(x.reshape(dimA,dimB,dimA,dimB).transpose(0,3,2,1).reshape(dimA*dimB,dimA*dimB)) for x in tmp0])
        ret_cha.append(get_cha_distance(tmp0, model_kwargs, kwargs, num_worker=num_worker))
    ret_ppt = np.array(ret_ppt)
    ret_cha = np.stack(ret_cha)

    for haar_k,dm,ppt,cha in zip(haar_k_list,dm_list,ret_ppt,ret_cha):
        EVL = np.array([np.linalg.eigvalsh(z.reshape(dimA,dimB,dimA,dimB).transpose(0,3,2,1).reshape(dimA*dimB,-1))[0] for z in dm])
        tmp0 = EVL[np.logical_not(ppt)].max()
        tmp1 = EVL[ppt].min()
        tmp2 = cha[np.logical_not(ppt)].min()
        tmp3 = cha[ppt].max()
        print(f'[k={haar_k}] NPT: max(lambda)={tmp0}, min(CHA)={tmp2}')
        print(f'[k={haar_k}] PPT: min(lambda)={tmp1}, max(CHA)={tmp3}')

    fig, ax = plt.subplots()
    ax.plot(haar_k_list, np.mean(ret_ppt, axis=1), 'x', label="PPT")
    ax.plot(haar_k_list, np.mean(ret_cha<cha_threshold, axis=1), label="CHA")
    ax.set_xlabel("Haar-k measure")
    ax.set_title(rf'd={dimA}\times{dimB}$ #sample={num_sample}')
    ax.set_ylabel('Probability')
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)

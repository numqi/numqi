import os
import time
import itertools
import functools
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import torch
import pickle
import concurrent.futures

if torch.get_num_threads() != 1:
    torch.set_num_threads(1)

import numqi

def make_error_list(num_qubit, distance, op_list=None, tag_full=False):
    assert distance>1
    if op_list is None:
        op_list = [numqi.gate.X, numqi.gate.Y, numqi.gate.Z]
    ret = []
    for weight in range(1, distance):
        for index_qubit in itertools.combinations(range(num_qubit), r=weight):
            for gate in itertools.product(op_list, repeat=weight):
                ret.append([([x],y) for x,y in zip(index_qubit,gate)])
    if tag_full:
        hf_kron = lambda *x: functools.reduce(np.kron, x[1:], x[0])
        tmp0 = ret
        ret = []
        for op0 in tmp0:
            tmp1 = [numqi.gate.pauli.s0 for _ in range(num_qubit)]
            for y,z in op0:
                tmp1[y[0]] = z
            ret.append(hf_kron(*tmp1))
    return ret


class VarQECSchmidt(torch.nn.Module):
    def __init__(self, num_qubit, num_logical_dim, error_op_full, loss_type='L2'):
        super().__init__()
        self.num_logical_dim = num_logical_dim
        self.num_qubit = num_qubit
        self.error_op_torch = torch.tensor(np.stack(error_op_full, axis=0).reshape(-1,2**num_qubit), dtype=torch.complex64).to_sparse_csr()
        self.manifold = numqi.manifold.Stiefel(2**num_qubit, num_logical_dim, method='sqrtm', dtype=torch.complex64)
        self.mask = torch.triu(torch.ones(num_logical_dim, num_logical_dim, dtype=torch.complex128), diagonal=1)

    def forward(self):
        q0 = self.manifold()
        inner_product = q0.T.conj() @ (self.error_op_torch @ q0).reshape(-1, *q0.shape)
        tmp0 = (inner_product*self.mask).reshape(-1)
        tmp1 = torch.diagonal(inner_product, dim1=1, dim2=2).real
        tmp2 = (tmp1 - tmp1.mean(axis=1, keepdims=True)).reshape(-1)
        loss = torch.vdot(tmp0, tmp0).real + torch.dot(tmp2, tmp2)
        return loss


def hf_task(num_repeat):
    num_qubit = 7
    num_logical_dim = 3
    distance = 3
    error_op_full = make_error_list(num_qubit, distance, tag_full=True)
    model = VarQECSchmidt(num_qubit, num_logical_dim, error_op_full)
    kwargs = dict(theta0=('uniform',-1,1), tol=1e-5, print_freq=0, early_stop_threshold=0.01, print_every_round=0)
    theta_optim = numqi.optimize.minimize(model, num_repeat=num_repeat, **kwargs)
    return theta_optim

if __name__ == '__main__':
    num_repeat_per_worker = 30 #1 second for each repeat
    num_worker = 4
    num_total_task = num_worker*100000 #233 per hour
    best_theta_optim = None
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_worker) as executor:
        job_list = [executor.submit(hf_task, num_repeat_per_worker) for _ in range(num_total_task)]
        t0 = time.time()
        for ind0,job_i in enumerate(concurrent.futures.as_completed(job_list)):
            result = job_i.result()
            if best_theta_optim is None or result.fun < best_theta_optim.fun:
                best_theta_optim = result
            if best_theta_optim.fun < 1e-4:
                break
            total_time = time.time() - t0
            average_time = total_time / ((ind0+1)*num_repeat_per_worker)
            eta_time = total_time*num_total_task / (ind0+1) - total_time
            tmp0 = (time.time() - t0)/(ind0+1)
            tmp1 = tmp0 * num_total_task - tmp0*(ind0+1)
            print(f'[{ind0+1}/{num_total_task}][{average_time:.2g}s / {eta_time:.2f}s / {total_time:.2f}s] loss={best_theta_optim.fun:.7f}')
    with open('tbd00.pkl', 'wb') as fid:
        pickle.dump(best_theta_optim, fid)

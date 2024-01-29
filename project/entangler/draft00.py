import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import torch
import time
import scipy.optimize
import concurrent.futures

import numqi

if torch.get_num_threads()!=1:
    torch.set_num_threads(1)

np_rng = np.random.default_rng()

class DummyModel(torch.nn.Module):
    def __init__(self, matU, dim, loss='purity'):
        super().__init__()
        dimA, dimB = dim
        assert loss in {'eigen', 'purity'}
        assert dimA <= dimB
        assert (matU.ndim==2) and (matU.shape[0]==dimA*dimB) and (matU.shape[0]==matU.shape[1])
        assert np.abs(matU @ matU.T.conj() - np.eye(matU.shape[0])).max() < 1e-10
        self.loss = loss
        self.manifold_psiA = numqi.manifold.Sphere(dimA, dtype=torch.complex128)
        self.manifold_psiB = numqi.manifold.Sphere(dimB, dtype=torch.complex128)
        self.matU = torch.tensor(matU, dtype=torch.complex128)

    def forward(self):
        stateA = self.manifold_psiA()
        stateB = self.manifold_psiB()
        state_out = (self.matU.reshape(-1, stateB.shape[0]) @ stateB).reshape(-1, stateA.shape[0]) @ stateA
        tmp0 = state_out.reshape(stateA.shape[0], stateB.shape[0])
        if self.loss=='eigen':
            loss = torch.linalg.eigvalsh(tmp0 @ tmp0.T.conj())[1]
        else:
            rdmA = tmp0 @ tmp0.T.conj()
            tmp0 = (rdmA - rdmA @ rdmA).reshape(-1)
            loss = torch.vdot(tmp0, tmp0).real
        return loss


def demo00():
    dimA = 3
    dimB = 4
    kwargs_inner = dict(theta0='uniform', num_repeat=100, tol=1e-14, print_every_round=0)
    z0 = []
    for _ in range(100):
        matU = numqi.random.rand_haar_unitary(dimA*dimB)
        model = DummyModel(matU, (dimA, dimB))
        theta_optim = numqi.optimize.minimize(model, **kwargs_inner)
        print(theta_optim.fun)
        z0.append((matU,theta_optim.fun))
    z1 = max(z0, key=lambda x: x[1])
    print(z1[1])


def hf_dummy_232(theta_U, kwargs_model, kwargs_optim):
    dimA,dimB = kwargs_model['dim']
    matU = numqi.param.real_matrix_to_special_unitary(theta_U.reshape(dimA*dimB, dimA*dimB))
    model = DummyModel(matU, **kwargs_model)
    fval = -numqi.optimize.minimize(model, **kwargs_optim).fun
    return fval


def hf_dummy_233(theta_U, tag_grad, kwargs_model, kwargs_optim, zero_eps=1e-9, num_worker=18):
    ret = hf_dummy_232(theta_U, kwargs_model, kwargs_optim)
    if tag_grad:
        arg_list = []
        for ind0 in range(len(theta_U)):
            x0,x1 = theta_U.copy(), theta_U.copy()
            x0[ind0] += zero_eps
            x1[ind0] -= zero_eps
            arg_list.append(x0)
            arg_list.append(x1)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_worker) as executor:
            job_list = [executor.submit(hf_dummy_232, x, kwargs_model, kwargs_optim) for x in arg_list]
            tmp0 = [x.result() for x in job_list] #one-by-one
        tmp1 = np.array(tmp0).reshape(-1,2)
        grad = (tmp1[:,0] - tmp1[:,1]) / (2*zero_eps)
        ret = ret,grad
    return ret


if __name__=='__main__':
    from zzz233 import to_pickle, from_pickle
    dimA = 3
    dimB = 4
    kwargs_model = dict(dim=(dimA,dimB), loss='purity')
    kwargs_optim = dict(theta0='uniform', num_repeat=100, tol=1e-14, print_every_round=0)


    # model = DummyModel(np.eye(dimA*dimB), (dimA,dimB), loss='purity')
    # for _ in range(100):
    #     theta_u = np_rng.uniform(-1,1,size=dimA*dimB*dimA*dimB)
    #     tmp0 = hf_dummy_232(theta_u, kwargs_model, kwargs_optim)
    #     if tmp0 < -5e-7:
    #         print(tmp0)
    #         break
    # to_pickle(theta_u=theta_u)

    num_round = 1000
    t0 = time.time()
    loss_list = []
    for ind_round in range(num_round):
        theta_u = from_pickle('theta_optim_u')
        hf0 = lambda x, tag_grad=False: hf_dummy_233(x, tag_grad, kwargs_model, kwargs_optim, zero_eps=1e-7, num_worker=16)
        hf_callback = numqi.optimize.MinimizeCallback(print_freq=50).to_callable(hf0) #extra_key='path'
        theta_optim = scipy.optimize.minimize(hf0, theta_u, method='Nelder-Mead', callback=hf_callback)
        total_time = time.time() - t0
        average_time = total_time / (ind_round+1)
        eta_time = total_time*num_round / (ind_round+1) - total_time
        to_pickle(theta_optim_u=theta_optim.x)
        print(f'[{ind_round+1}/{num_round}][{average_time:.0f}s / {eta_time:.0f}s / {total_time:.0f}s] loss={theta_optim.fun:.6f}')
        loss_list.append(theta_optim.fun)
    for x in loss_list:
        print(x)
    to_pickle(loss_list=loss_list)

    # theta_optim_u = from_pickle('theta_optim_u')
    # matU = numqi.param.real_matrix_to_special_unitary(theta_optim_u.reshape(dimA*dimB, dimA*dimB))
    # model = DummyModel(matU, **kwargs_model)
    # fval = -numqi.optimize.minimize(model, **kwargs_optim).fun
    # model.loss = 'eigen'
    # hf_dummy_233(theta_optim_u, True, dict(dim=(dimA,dimB), loss='eigen'), kwargs_optim, zero_eps=1e-9, num_worker=16)

# https://www.sciencedirect.com/science/article/pii/S037596011401216X

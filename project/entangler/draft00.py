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
        np_rng = np.random.default_rng()
        hf0 = lambda *x: torch.nn.Parameter(torch.tensor(np_rng.uniform(-1,1,size=x), dtype=torch.float64))
        self.thetaA = hf0(2, dimA)
        self.thetaB = hf0(2, dimB)
        self.matU = torch.tensor(matU, dtype=torch.complex128)

    def forward(self):
        tmp0 = self.thetaA / torch.linalg.norm(self.thetaA)
        stateA = torch.complex(tmp0[0], tmp0[1])
        tmp0 = self.thetaB / torch.linalg.norm(self.thetaB)
        stateB = torch.complex(tmp0[0], tmp0[1])
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
    theta_u = from_pickle('theta_optim_u')

    # hf0 = lambda x, tag_grad=True: hf_dummy_233(x, tag_grad, kwargs_model, kwargs_optim, zero_eps=1e-9, num_worker=16)
    # state_dict = dict()
    # hf_callback = numqi.optimize.hf_callback_wrapper(hf0, state=state_dict, print_freq=1, tag_record_path=True)
    # theta_optim = scipy.optimize.minimize(hf0, theta_u, method='L-BFGS-B', jac=True, callback=hf_callback)

    state_dict = dict()
    hf0 = lambda x, tag_grad=False: hf_dummy_233(x, tag_grad, kwargs_model, kwargs_optim, zero_eps=1e-7, num_worker=16)
    hf_callback = numqi.optimize.hf_callback_wrapper(hf0, state=state_dict, print_freq=5, tag_record_path=True)
    theta_optim = scipy.optimize.minimize(hf0, theta_u, method='Nelder-Mead', callback=hf_callback)
    from zzz233 import to_pickle
    to_pickle(theta_optim_u=theta_optim.x)

    # theta_optim_u = from_pickle('theta_optim_u')
    # matU = numqi.param.real_matrix_to_special_unitary(theta_optim_u.reshape(dimA*dimB, dimA*dimB))
    # model = DummyModel(matU, **kwargs_model)
    # fval = -numqi.optimize.minimize(model, **kwargs_optim).fun
    # model.loss = 'eigen'

    # hf_dummy_233(theta_optim_u, True, dict(dim=(dimA,dimB), loss='eigen'), kwargs_optim, zero_eps=1e-9, num_worker=16)

# https://www.sciencedirect.com/science/article/pii/S037596011401216X

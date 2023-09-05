import time
import pickle
import itertools
import numpy as np
import torch
import opt_einsum
import concurrent.futures

if not torch.get_num_threads()!=1:
    torch.set_num_threads(1)

import numqi

def get_keep_list(num_party):
    keep_list = [tuple(x) for x in itertools.combinations(range(num_party), num_party//2)]
    if num_party % 2 == 0:
        tmp0 = set(range(num_party))
        tmp1 = []
        for x in keep_list:
            y = tuple(sorted(tmp0 - set(x)))
            tmp1.append(((x,y) if (min(x)<min(y)) else (y,x)))
        keep_list = [x[0] for x in sorted(set(tmp1))]
    return keep_list

class AbsolutelyMaximallyEntangledStateModel(torch.nn.Module):
    def __init__(self, num_party:int, dim:int, dtype='float64'):
        super().__init__()
        assert num_party>=2
        assert dim>=2
        assert dtype in {'float32','float64'}
        self.dtype = torch.float32 if (dtype=='float32') else torch.float64
        self.cdtype = torch.complex64 if (dtype=='float32') else torch.complex128
        self.num_party = int(num_party)
        self.dim = int(dim)
        tmp0 = np.random.default_rng().uniform(-1,1,size=(2,)+(dim,)*num_party)
        self.theta = torch.nn.Parameter(torch.tensor(tmp0, dtype=self.dtype))

        self.target = torch.eye(dim**(num_party//2), dtype=self.cdtype)/(dim**(num_party//2))
        self.keep_list = get_keep_list(num_party)
        tmp0 = list(range(num_party))
        self.contract_list = []
        for keep in self.keep_list:
            tmp1 = np.arange(num_party)
            tmp1[list(keep)] += num_party
            tmp2 = list(keep) + [x+num_party for x in keep]
            self.contract_list.append(opt_einsum.contract_expression([dim]*num_party, tmp0, [dim]*num_party, tmp1, tmp2))

    def forward(self):
        tmp0 = self.theta / torch.linalg.norm(self.theta)
        state = torch.complex(tmp0[0], tmp0[1])
        state_conj = state.conj().resolve_conj()
        loss = 0
        for contract_i in self.contract_list:
            tmp0 = (contract_i(state, state_conj).reshape(self.target.shape) - self.target).reshape(-1)
            loss = loss + torch.vdot(tmp0, tmp0).real
        return loss

def hf_task(num_repeat):
    num_party = 4
    dim = 6
    kwargs = dict(theta0='uniform', tol=1e-7, early_stop_threshold=0.002, print_every_round=0)
    model = AbsolutelyMaximallyEntangledStateModel(num_party, dim, dtype='float32')
    theta_optim = numqi.optimize.minimize(model, num_repeat=num_repeat, **kwargs)
    return theta_optim

if __name__ == '__main__':
    num_repeat_per_worker = 200
    num_worker = 12
    num_total_task = num_worker*10000 #100 for 1hour
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
    # num_worker=2 [8/8][0.079s / 0.00s / 31.49s] loss=0.0031225
    # num_worker=4 [16/16][0.046s / 0.00s / 36.61s] loss=0.0031843
    # num_worker=8 [32/32][0.023s / 0.00s / 37.23s] loss=0.0031226
    # num_worker=16 [64/64][0.012s / 0.00s / 38.51s] loss=0.0031297


# # hard problem (4,5) (4,6)
# num_party = 4
# dim = 6
# num_repeat = 10
# kwargs = dict(theta0='uniform', tol=1e-7, early_stop_threshold=0.002)
# model = AbsolutelyMaximallyEntangledStateModel(num_party, dim, dtype='float32')
# t0 = time.time()
# theta_optim = numqi.optimize.minimize(model, num_repeat=num_repeat, **kwargs)
# print(time.time()-t0)

# for ind_round in range(100):
#     loss = numqi.optimize.minimize_adam(model, num_step=5000, theta0='uniform', tqdm_update_freq=20, early_stop_threshold=0.001)
#     print(f'[round={ind_round}] loss={loss:.5f}')


# num_party = 4
# dim = 5
# model = AbsolutelyMaximallyEntangledStateModel(num_party, dim)
# theta_optim = numqi.optimize.minimize(model, theta0='uniform', tol=1e-10, num_repeat=10)

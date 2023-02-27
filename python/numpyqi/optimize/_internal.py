import time
import contextlib
import numpy as np
import scipy.optimize
from tqdm import tqdm

try:
    import torch
except ImportError:
    torch = None

def _get_sorted_parameter(model):
    tmp0 = sorted([(k,v) for k,v in model.named_parameters() if v.requires_grad], key=lambda x:x[0])
    ret = [x[1] for x in tmp0]
    return ret


def get_model_flat_parameter(model):
    tmp0 = _get_sorted_parameter(model)
    ret = np.concatenate([x.detach().cpu().numpy().reshape(-1) for x in tmp0])
    return ret


def get_model_flat_grad(model):
    tmp0 = _get_sorted_parameter(model)
    ret = np.concatenate([x.grad.detach().cpu().numpy().reshape(-1) for x in tmp0])
    return ret


def set_model_flat_parameter(model, theta, index01=None):
    theta = torch.tensor(theta)
    parameter_sorted = _get_sorted_parameter(model)
    if index01 is None:
        tmp0 = np.cumsum(np.array([0] + [x.numel() for x in parameter_sorted])).tolist()
        index01 = list(zip(tmp0[:-1],tmp0[1:]))
    for ind0,(x,y) in enumerate(index01):
        tmp0 = theta[x:y].reshape(*parameter_sorted[ind0].shape)
        if not parameter_sorted[ind0].is_cuda:
            tmp0 = tmp0.cpu()
        parameter_sorted[ind0].data[:] = tmp0


def hf_model_wrapper(model):
    parameter_sorted = _get_sorted_parameter(model)
    tmp0 = np.cumsum(np.array([0] + [x.numel() for x in parameter_sorted])).tolist()
    index01 = list(zip(tmp0[:-1],tmp0[1:]))
    def hf0(theta, tag_grad=True):
        set_model_flat_parameter(model, theta, index01)
        if tag_grad:
            loss = model()
            for x in parameter_sorted:
                if x.grad is not None:
                    x.grad.zero_()
            if hasattr(model, 'grad_backward'): #designed for custom automatic differentiation
                model.grad_backward(loss)
            else:
                loss.backward() #if no .grad_backward() method, it should be a normal torch.nn.Module
            # scipy.optimize.LBFGS does not support float32 @20221118
            grad = np.concatenate([x.grad.detach().cpu().numpy().reshape(-1).astype(theta.dtype) for x in parameter_sorted])
        else:
            # TODO, if tag_grad=False, maybe we should return fval only, not (fval,None)
            with torch.no_grad():
                loss = model()
            grad = None
        return loss.item(), grad
    return hf0


def hf_callback_wrapper(hf_fval, state:dict=None, print_freq:int=1):
    if state is None:
        state = dict()
    state['step'] = 0
    state['time'] = time.time()
    state['fval'] = []
    state['time_history'] = []
    def hf0(theta):
        step = state['step']
        if (print_freq>0) and (step%print_freq==0):
            t0 = state['time']
            t1 = time.time()
            fval = hf_fval(theta, tag_grad=False)[0]
            print(f'[step={step}][time={t1-t0:.3f} seconds] loss={fval}')
            state['fval'].append(fval)
            state['time'] = t1
            state['time_history'].append(t1-t0)
        state['step'] += 1
    return hf0


def check_model_gradient(model, tol=1e-5, zero_eps=1e-4, seed=None):
    np_rng = np.random.default_rng(seed)
    num_parameter = get_model_flat_parameter(model).size
    # TODO range for paramter
    theta0 = np_rng.uniform(0, 2*np.pi, size=num_parameter)

    set_model_flat_parameter(model, theta0)
    loss = model()
    for x in model.parameters():
        if x.grad is not None:
            x.grad.zero_()
    if hasattr(model, 'grad_backward'):
        model.grad_backward(loss)
    else:
        loss.backward()
    ret0 = get_model_flat_grad(model)

    def hf0(theta):
        set_model_flat_parameter(model, theta)
        ret = model()
        if hasattr(ret, 'item'):
            ret = ret.item()
        return ret
    ret_ = np.zeros(num_parameter, dtype=np.float64)
    for ind0 in range(ret_.shape[0]):
        tmp0,tmp1 = [theta0.copy() for _ in range(2)]
        tmp0[ind0] += zero_eps
        tmp1[ind0] -= zero_eps
        ret_[ind0] = (hf0(tmp0)-hf0(tmp1))/(2*zero_eps)
    assert np.abs(ret_-ret0).max()<tol


def _get_hf_theta(np_rng, key=None):
    if key is None:
        key = ('uniform', -1, 1)
    if isinstance(key, str):
        if key=='uniform':
            key = ('uniform', -1, 1)
        elif key=='normal':
            key = ('normal', 0, 1)
    if isinstance(key, np.ndarray):
        hf_theta = lambda *x: key
    elif hasattr(key, '__len__') and (len(key)>0) and isinstance(key[0], str):
        if key[0]=='uniform':
            hf_theta = lambda *x: np_rng.uniform(key[1], key[2], size=x)
        elif key[0]=='normal':
            hf_theta = lambda *x: np_rng.normal(key[1], key[2], size=x)
        else:
            assert False, f'un-recognized key "{key}"'
    elif callable(key):
        hf_theta = lambda size: key(size, np_rng)
    else:
        assert False, f'un-recognized key "{key}"'
    return hf_theta


def minimize(model, theta0=None, num_repeat=3, tol=1e-7, print_freq=-1, method='L-BFGS-B',
            print_every_round=1, maxiter=None, early_stop_threshold=None, return_all_result=False, seed=None):
    np_rng = np.random.default_rng(seed)
    hf_theta = _get_hf_theta(np_rng, theta0)
    num_parameter = len(get_model_flat_parameter(model))
    hf_model = hf_model_wrapper(model)
    theta_optim_list = []
    theta_optim_best = None
    options = dict() if maxiter is None else {'maxiter':maxiter}
    for ind0 in range(num_repeat):
        theta0 = hf_theta(num_parameter)
        hf_callback = hf_callback_wrapper(hf_model, print_freq=print_freq)
        theta_optim = scipy.optimize.minimize(hf_model, theta0, jac=True, method=method, tol=tol, callback=hf_callback, options=options)
        if return_all_result:
            theta_optim_list.append(theta_optim)
        if (theta_optim_best is None) or (theta_optim.fun<theta_optim_best.fun):
            theta_optim_best = theta_optim
        if (print_every_round>0) and (ind0%print_every_round==0):
            print(f'[round={ind0}] min(f)={theta_optim_best.fun}, current(f)={theta_optim.fun}')
        if (early_stop_threshold is not None) and (theta_optim_best.fun<=early_stop_threshold):
            break
    hf_model(theta_optim_best.x, tag_grad=False) #set theta and model.property (sometimes)
    ret = (theta_optim_best,theta_optim_list) if return_all_result else theta_optim_best
    return ret


def minimize_adam(model, num_step, theta0='no-init', optim_args=('adam',0.01), seed=None, tqdm_update_freq=20, use_tqdm=True):
    assert optim_args[0] in {'sgd', 'adam'}
    np_rng = np.random.default_rng(seed)
    num_parameter = len(get_model_flat_parameter(model))
    if theta0!='no-init':
        theta0 = _get_hf_theta(np_rng, theta0)(num_parameter)
        set_model_flat_parameter(model, theta0)
    if optim_args[0]=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=optim_args[1])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=optim_args[1])
    tmp0 = tqdm(range(num_step)) if use_tqdm else contextlib.nullcontext(range(num_step))
    with tmp0 as pbar:
        for ind0 in pbar:
            optimizer.zero_grad()
            loss = model()
            loss.backward()
            optimizer.step()
            if use_tqdm and (ind0%tqdm_update_freq==0):
                pbar.set_postfix(loss=f'{loss.item():.12f}')
    return loss.item()


def _hf_zero_grad(parameter_list):
    for x in parameter_list:
        if x.grad is not None:
            x.grad.zero_()

def get_model_hessian(model):
    parameter_sorted = _get_sorted_parameter(model)
    _hf_zero_grad(parameter_sorted)
    loss = model()
    grad_list = torch.autograd.grad(loss, parameter_sorted, create_graph=True)
    ret = []
    for grad_i in grad_list:
        shape = tuple(grad_i.shape)
        for ind0 in range(grad_i.numel()):
            ind0a = np.unravel_index(ind0, shape)
            grad_i[ind0a].backward(retain_graph=True)
            ret.append(np.concatenate([x.grad.detach().reshape(-1).numpy() for x in parameter_sorted]))
            _hf_zero_grad(parameter_sorted)
    ret = np.stack(ret)
    return ret

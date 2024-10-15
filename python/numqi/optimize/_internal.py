import time
import contextlib
import numpy as np
import scipy.optimize
from tqdm.auto import tqdm
import torch

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
        tmp0 = theta[x:y].reshape(parameter_sorted[ind0].shape)
        if not parameter_sorted[ind0].is_cuda:
            tmp0 = tmp0.cpu()
        parameter_sorted[ind0].data.copy_(tmp0)


def _get_constraint_square(x0):
    x0 = x0.reshape(-1)
    ret = torch.vdot(x0, x0).real if torch.is_complex(x0) else torch.dot(x0, x0)
    return ret

def hf_model_wrapper(model, constraint_penalty:None|float=None, constraint_p:None|float=None):
    parameter_sorted = _get_sorted_parameter(model)
    tmp0 = np.cumsum(np.array([0] + [x.numel() for x in parameter_sorted])).tolist()
    index01 = list(zip(tmp0[:-1],tmp0[1:]))
    def hf0(theta, tag_grad:bool=True, return_constraint:bool=False):
        '''
        if (tag_grad,return_constraint), return
            (False,False): loss
            (False,True): loss,constraint_squared
            (True,False): loss,grad
            (True,True): loss,grad,constraint_squared

        no matter `return_constraint` is True or False, penalty contribution is always added to the loss
        '''
        if return_constraint:
            assert constraint_penalty is not None
        set_model_flat_parameter(model, theta, index01)
        with torch.set_grad_enabled(tag_grad):
            tmp0 = model()
            if constraint_penalty is None:
                loss = tmp0
            else:
                fval,constraint = tmp0
                if isinstance(constraint, torch.Tensor):
                    constraint_square = _get_constraint_square(constraint)
                elif isinstance(constraint, list|tuple):
                    constraint_square = sum(_get_constraint_square(x) for x in constraint)
                elif isinstance(constraint, dict):
                    constraint_square = sum(_get_constraint_square(x) for x in constraint.values())
                else:
                    raise TypeError(f'constraint should be torch.Tensor, list, or dict, but got "{type(constraint)}"')
                loss = fval + constraint_penalty * constraint_square**(constraint_p/2)
        if tag_grad:
            for x in parameter_sorted:
                if x.grad is not None:
                    x.grad.zero_()
            if hasattr(model, 'grad_backward'): #designed for custom automatic differentiation
                model.grad_backward(loss)
            else:
                loss.backward() #if no .grad_backward() method, it should be a normal torch.nn.Module
            # scipy.optimize.LBFGS does not support float32 @20221118
            grad = np.concatenate([x.grad.detach().cpu().numpy().reshape(-1).astype(theta.dtype) for x in parameter_sorted])
        match (tag_grad,return_constraint):
            case (False,False):
                ret = loss.item()
            case (False,True):
                ret = loss.item(),constraint_square.item()
            case (True,False):
                ret = loss.item(),grad
            case (True,True):
                ret = loss.item(),grad,constraint_square.item()
        return ret
    return hf0


class MinimizeCallback:
    def __init__(self, print_freq:int=1, extra_key=None, tag_print:bool=True):
        if extra_key is None:
            extra_key = []
        if isinstance(extra_key, str):
            extra_key = [extra_key]
        available_key = {'grad_norm', 'path'}
        assert all((isinstance(x,str) and (x in available_key)) for x in extra_key)
        self.extra_key = extra_key
        self.tag_print = tag_print
        self.print_freq = print_freq
        self.last_time = time.time()
        self._need_grad = 'grad_norm' in extra_key #if True, the callback function will be called with tag_grad=True
        self.state = None
        self.history_state = []
        self.reset(save_history=False)

    def to_callable(self, hf_fval):
        def hf0(theta):
            if 'grad_norm' in self.extra_key:
                fval,grad = hf_fval(theta, tag_grad=True)
            else:
                fval = hf_fval(theta, tag_grad=False)
                grad = None
            self(theta, fval, grad)
        return hf0

    def __call__(self, theta, fval, grad=None):
        step = self.state['step']
        if 'path' in self.extra_key:
            self.state['path'].append(theta.copy())
        if (self.print_freq>0) and (step%self.print_freq==0):
            self.state['fval'].append(fval)
            if 'grad_norm' in self.extra_key:
                self.state['grad_norm'].append(np.linalg.norm(grad))
            t0 = self.last_time
            t1 = time.time()
            self.state['time'].append(t1-t0)
            self.last_time = t1
            if self.tag_print:
                print(f'[step={step}][time={t1-t0:.3f} seconds] loss={fval}')
        self.state['step'] += 1

    def reset(self, save_history:bool=False, theta_optim=None):
        if save_history:
            if theta_optim is not None:
                self.state['optim_x'] = theta_optim.x.copy()
                self.state['optim_fval'] = float(theta_optim.fun)
            self.history_state.append(self.state)
        tmp0 = {'step':0, 'fval':[], 'time':[]}
        if 'grad_norm' in self.extra_key:
            tmp0['grad_norm'] = []
        if 'path' in self.extra_key:
            tmp0['path'] = []
        self.state = tmp0


def finite_difference_central(hf0, x0, zero_eps=1e-4):
    # https://en.wikipedia.org/wiki/Finite_difference
    x0 = np.asarray(x0)
    assert x0.dtype.type in (np.float32, np.float64, np.complex64, np.complex128)
    is_real = x0.dtype.type in (np.float32, np.float64)
    ret = np.zeros_like(x0)
    for ind0 in range(x0.size):
        ind0 = np.unravel_index(ind0, x0.shape)
        if is_real:
            tmp0,tmp1 = [x0.copy() for _ in range(2)]
            tmp0[ind0] += zero_eps
            tmp1[ind0] -= zero_eps
            ret[ind0] = (hf0(tmp0) - hf0(tmp1)) / (2*zero_eps)
        else:
            tmp0,tmp1,tmp2,tmp3 = [x0.copy() for _ in range(4)]
            tmp0[ind0] += zero_eps
            tmp1[ind0] -= zero_eps
            tmp2[ind0] += 1j*zero_eps
            tmp3[ind0] -= 1j*zero_eps
            ret[ind0] = (hf0(tmp0) - hf0(tmp1) + 1j*(hf0(tmp2) - hf0(tmp3))) / (2*zero_eps)
    return ret


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
        ret = model().item()
        return ret
    ret_ = finite_difference_central(hf0, theta0, zero_eps=zero_eps)
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


def minimize(model, theta0=None, num_repeat=1, tol=1e-7, print_freq=0, method='L-BFGS-B',
            print_every_round=1, maxiter=None, early_stop_threshold=None,
            constraint_penalty=None, constraint_p=None, constraint_threshold=None, callback=None, seed=None):
    r'''gradient-based optimization

    Parameters:
        model (torch.nn.Module): the model to be optimized
        theta0 (None, str, np.ndarray, callable): the initial value of theta

            None: uniform(-1,1)

            'uniform': uniform(-1,1)

            'normal': normal(0,1)

            np.ndarray: return the input

            ('uniform',a,b): uniform(a,b)

            ('normal',a,b): normal(a,b)

            callable: return the output of the callable

        num_repeat (int): number of repeat
        tol (float): tolerance
        print_freq (int): print frequency, non-positive means no print, if callback is used, this parameter is ignored
        method (str): optimization method, see scipy.optimize.minimize
        print_every_round (int): print frequency for each round, non-positive means no print
        maxiter (int): maximum number of iterations, see scipy.optimize.minimize
        early_stop_threshold (float): if the loss is less than this value, the optimization will stop
        constraint_penalty (None, float): if not None, the penalty term will be added to the loss
        constraint_p (None, float): must be float when `constraint_penalty` is not None.
        constraint_threshold (None, float): must be float when `constraint_penalty` is not None.
                    The best result is selected from the results whose constraint squared is less than this value
        callback (None, MinimizeCallback): callback function, if None, MinimizeCallback(print_freq=print_freq) will be used
        seed (None, int): random seed

    Returns:
        ret (scipy.optimize.OptimizeResult,None): the result of scipy.optimize.minimize, None if no result is found (only when penalty is used)
    '''
    if callback is not None:
        assert isinstance(callback, MinimizeCallback)
        assert hasattr(callback, '__call__') and hasattr(callback, 'reset')
    if print_freq>=1:
        assert callback is None, 'print_freq and callback cannot be used at the same time'
        callback = MinimizeCallback(print_freq=print_freq)
    if constraint_penalty is not None:
        assert (constraint_penalty>0) and (constraint_p>0) and (constraint_threshold>0)
        if callback is None: #if penalty is used, callback is necessary (for handling penalty threshold)
            callback = MinimizeCallback(print_freq=print_freq)
    np_rng = np.random.default_rng(seed)
    hf_theta = _get_hf_theta(np_rng, theta0)
    num_parameter = len(get_model_flat_parameter(model))
    hf_model = hf_model_wrapper(model, constraint_penalty, constraint_p)
    theta_optim_best = None
    index_best = None
    kwargs = dict(tol=tol, method=method, jac=True)
    if maxiter is not None:
        kwargs['options'] = {'maxiter':maxiter}
    for ind0 in range(num_repeat):
        theta0 = hf_theta(num_parameter)
        hf_callback = callback.to_callable(hf_model) if (callback is not None) else None
        theta_optim = scipy.optimize.minimize(hf_model, theta0, callback=hf_callback, **kwargs)
        if (constraint_penalty is None) and ((theta_optim_best is None) or (theta_optim.fun<theta_optim_best.fun)):
            index_best = ind0
            theta_optim_best = theta_optim
        elif constraint_penalty is not None:
            tmp0 = hf_model(theta_optim.x, tag_grad=False, return_constraint=True) #loss,constraint_squared
            loss_cur,constraint_square_cur = tmp0
            if constraint_square_cur<constraint_threshold:
                if (theta_optim_best is None) or (loss_cur<theta_optim_best.fun):
                    index_best = ind0
                    theta_optim_best = theta_optim
        if (print_every_round>0) and (ind0%print_every_round==0):
            tmp0 = 'None' if theta_optim_best is None else theta_optim_best.fun
            tmp1 = '' if constraint_penalty is None else f', current(constraint^2)={constraint_square_cur}'
            print(f'[round={ind0}] min(f)={tmp0}, current(f)={theta_optim.fun}{tmp1}')
        if callback is not None:
            callback.reset(save_history=True, theta_optim=theta_optim)
        if (early_stop_threshold is not None) and (theta_optim_best is not None) and (theta_optim_best.fun<=early_stop_threshold):
            break
    if theta_optim_best is not None:
        hf_model(theta_optim_best.x, tag_grad=False) #set theta and model.property
    if (callback is not None) and (index_best is not None):
        callback.state = callback.history_state[index_best]
    return theta_optim_best


def minimize_adam(model, num_step, theta0='no-init', optim_args=('adam',0.01),
            seed=None, tqdm_update_freq=20, early_stop_threshold=None, tag_return_history=False):
    # TODO num_repeat
    assert optim_args[0] in {'sgd', 'adam'}
    use_tqdm = tqdm_update_freq>0
    np_rng = np.random.default_rng(seed)
    num_parameter = len(get_model_flat_parameter(model))
    if theta0!='no-init':
        theta0 = _get_hf_theta(np_rng, theta0)(num_parameter)
        set_model_flat_parameter(model, theta0)
    if optim_args[0]=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=optim_args[1])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=optim_args[1])
    if len(optim_args)==3:
        tmp0 = (optim_args[2]/optim_args[1])**(1/num_step)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=tmp0)
    else:
        lr_scheduler = None
    tmp0 = tqdm(range(num_step)) if use_tqdm else contextlib.nullcontext(range(num_step))
    loss_best = None
    theta_best = None
    loss_history = []
    with tmp0 as pbar:
        for ind0 in pbar:
            optimizer.zero_grad()
            loss = model()
            loss.backward()
            loss_i = loss.item()
            if tag_return_history:
                loss_history.append(loss_i)
            if (loss_best is None) or (loss_i<loss_best):
                loss_best = loss_i
                theta_best = get_model_flat_parameter(model)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            if use_tqdm and (ind0%tqdm_update_freq==0):
                pbar.set_postfix(loss=f'{loss_i:.12f}')
            if (early_stop_threshold is not None) and (loss_i<=early_stop_threshold):
                break
    # set theta and model.property (sometimes)
    set_model_flat_parameter(model, theta_best)
    with torch.no_grad():
        model()
    ret = (loss_best, loss_history) if tag_return_history else loss_best
    return ret


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

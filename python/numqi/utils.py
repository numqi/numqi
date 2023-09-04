import functools
import collections
import numpy as np
import torch


@functools.lru_cache(maxsize=128)
def hf_num_state_to_num_qubit(num_state:int, kind:str='exact'):
    assert kind in {'exact','ceil','floor'}
    if kind=='exact':
        ret = round(float(np.log2(num_state)))
        assert abs(2**ret-num_state)<1e-7
    elif kind=='ceil':
        ret = int(np.ceil(np.log2(num_state)))
    else: #floor
        ret = int(np.floor(np.log2(num_state)))
    return ret


def is_torch(x):
    ret = hasattr(x, '__torch_function__')
    return ret


def hf_tuple_of_any(x, type_=None):
    hf0 = lambda x: x if (type_ is None) else type_(x)
    if isinstance(x,collections.abc.Iterable):
        if isinstance(x, np.ndarray):
            ret = [hf0(y) for y in np.nditer(x)]
        else:
            # error when x is np.array(0)
            ret = tuple(hf0(y) for y in x)
    else:
        ret = hf0(x),
    return ret

hf_tuple_of_int = lambda x: hf_tuple_of_any(x, type_=int)


def hf_complex_to_real(x):
    dim0,dim1 = x.shape[-2:]
    shape = x.shape[:-2]
    x = x.reshape(-1, dim0, dim1)
    # ret = np.block([[x.real,-x.imag],[x.imag,x.real]])
    if isinstance(x, torch.Tensor):
        tmp0 = torch.concat([x.real, -x.imag], dim=2)
        tmp1 = torch.concat([x.imag, x.real], dim=2)
        ret = torch.concat([tmp0,tmp1], dim=1)
    else:
        ret = np.block([[x.real,-x.imag],[x.imag,x.real]])
    ret = ret.reshape(shape+(2*dim0,2*dim1))
    return ret


def hf_real_to_complex(x):
    assert (x.shape[-2]%2==0) and (x.shape[-1]%2==0)
    dim0 = x.shape[-2]//2
    dim1 = x.shape[-1]//2
    ret = x[...,:dim0,:dim1] + 1j*x[...,dim0:,:dim1]
    return ret

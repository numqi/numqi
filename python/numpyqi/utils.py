import functools
import collections
import itertools
import numpy as np

try:
    import torch
except ImportError:
    torch = None

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


def _reduce_shape_index_list_int(shape_index_list):
    if len(shape_index_list)>1:
        np0 = np.array([x[1:] for x in shape_index_list[::-1]])
        tmp0 = np.cumprod(np.array([1] + [x[0] for x in shape_index_list[::-1]]))
        tmp1 = np.sum(tmp0[:-1,np.newaxis]*np0, axis=0).tolist()
        ret = (int(tmp0[-1]),) + tuple(tmp1)
    else:
        ret = shape_index_list[0]
    return ret

def _reduce_shape_index_list_none(shape_index_list):
    tmp0 = [x[0] for x in shape_index_list]
    tmp1 = functools.reduce(lambda y0,y1: y0*y1, tmp0, 1)
    ret = (tmp1,) + (slice(None),)*(len(shape_index_list[0])-1)
    return ret

@functools.lru_cache(maxsize=128)
def reduce_shape_index_list(shape, *index_list):
    '''
    1. remove shape==1
    2. group index_list by None
    3. group index_list by integer
    '''
    assert isinstance(shape, tuple) and len(shape)>0 and all(x>1 for x in shape)
    N0 = len(shape)
    assert all(len(x)==N0 for x in index_list)
    for shape_i,tmp0 in zip(shape, zip(*index_list)):
        assert all(x==None for x in tmp0) or all((isinstance(x,int) and (0<=x<shape_i)) for x in tmp0)
    tmp0 = itertools.groupby(zip(shape,*index_list), key=lambda x: x[1]==None)
    ret = [(_reduce_shape_index_list_none(list(x1)) if x0 else _reduce_shape_index_list_int(list(x1))) for x0,x1 in tmp0]
    ret = tuple(zip(*ret))
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
    # ret = np.block([[x.real,-x.imag],[x.imag,x.real]])
    if is_torch(x):
        tmp0 = torch.concat([x.real, -x.imag], dim=1)
        tmp1 = torch.concat([x.imag, x.real], dim=1)
        ret = torch.concat([tmp0,tmp1], dim=0)
    else:
        ret = np.block([[x.real,-x.imag],[x.imag,x.real]])
    return ret


def hf_real_to_complex(x):
    assert (x.shape[0]%2==0) and (x.shape[1]%2==0)
    N0 = x.shape[0]//2
    N1 = x.shape[1]//2
    ret = x[:N0,:N1] + 1j*x[N0:,:N1]
    return ret

# symplectic group over GF(2)
import functools
import numpy as np

# canonical ordering of symplectic group elements
# @paper How to efficiently select an arbitrary clifford group element
# https://doi.org/10.1063%2F1.4903507
@functools.lru_cache
def _get_number_internal(n:int, kind:str):
    tmp0 = (1<<(2*x) for x in range(1,n+1))
    if kind=='base':
        ret = tuple(y for x in tmp0 for y in (x-1,x>>1))
    elif kind=='order':
        ret = 1
        for x in tmp0:
            ret = ret * (x-1) * (x>>1)
    else:
        assert kind=='coset'
        # TODO ai,bi
        ret = tuple((x-1)*(x>>1) for x in tmp0)
    return ret

def get_number(n:int, kind:str='base'):
    r'''Get the number of elements in the symplectic group over GF(2) of order 2n.

    Parameters:
        n (int): The order of the symplectic group.
        kind (str): 'base', 'order', or 'coset'. Default is 'base'.

    Returns:
        ret (int,tuple[int]): for `kind` being
            'base': The number of elements in the symplectic group (a1,b1,a2,b2,...,an,bn)
            'order': The order of the symplectic group. a1*b1*a2*b2*...*an*bn
            'coset': The number of cosets in the symplectic group. (a1*b1,a2*b2,...,an*bn)
    '''
    assert n>=1
    n = int(n)
    kind = str(kind).lower()
    assert kind in {'base', 'order', 'coset'}
    ret = _get_number_internal(n, kind)
    return ret


def get_inner_product(v0:np.ndarray, v1:np.ndarray):
    r'''Get symplectic inner product of two vectors GF(2).

    Parameters:
        v0 (np.ndarray): The first vector. `dtype=np.uint8`, `ndim>=1`, and `shape[-1]` is even.
        v1 (np.ndarray): The second vector. `dtype=np.uint8`, `ndim=1`

    Returns:
        ret (np.ndarray): The symplectic inner product of two vectors. `dtype=np.uint8` and `ndim=ndim(v0)-1`
    '''
    assert (v1.ndim==1) and (v0.shape[-1]==v1.size) and (v1.size%2==0)
    # assert v0.dtype.type in {np.uint8,np.uint16,np.uint32,np.uint64}
    N0 = v1.size//2
    # with warnings.catch_warnings():
    #     warnings.filterwarnings('ignore', r'RuntimeWarning: overflow encountered in scalar add')
    # overflow warning is okay here
    ret = (np.dot(v0[...,:N0], v1[N0:]) + np.dot(v0[...,N0:], v1[:N0]))%2
    return ret


def transvection(x:np.ndarray, *h_list:tuple[np.ndarray]):
    r'''Apply a transvection to a vector. see [doi-link](https://doi.org/10.1063%2F1.4903507)

    Parameters:
        x (np.ndarray): The vector. `dtype=np.uint8`, `ndim>=1`, and `shape[-1]` is even.
        h_list (tuple[np.ndarray]): The transvection. `dtype=np.uint8`, `ndim=1`, and `size` is even.

    Returns:
        ret (np.ndarray): of same shape as `x`. The result of applying the transvection.
    '''
    for h in h_list:
        tmp0 = get_inner_product(x,h)
        if x.ndim>1:
            tmp0 = tmp0[...,np.newaxis]
        x = (x + tmp0*h)%2
    return x


def int_to_bitarray(i:int, n:int):
    r'''Convert an integer to a bit array.

    `int_to_bitarray(3, 4)==[1,1,0,0]`

    Parameters:
        i (int): The integer to be converted.
        n (int): The length of the bit array.

    Returns:
        ret (np.ndarray): The bit array. `dtype=np.uint8` and `shape=(n,)`
    '''
    tmp0 = int(i).to_bytes((n + 7) // 8, 'little')
    tmp1 = np.frombuffer(tmp0, dtype=np.uint8)
    ret = np.unpackbits(tmp1, axis=0, bitorder='little')[:n]
    return ret


def bitarray_to_int(b:np.ndarray):
    r'''Convert a bit array to an integer.

    `bitarray_to_int([1,1,0,0])==3`

    Parameters:
        b (np.ndarray): The bit array. `dtype=np.uint8` and `ndim=1`

    Returns:
        ret (int): The integer.
    '''
    ret = int.from_bytes(np.packbits(b, axis=0, bitorder='little').tobytes(), byteorder='little', signed=False)
    return ret


def schmidt_orthogonalization(vec_list:list[np.ndarray]):
    r'''Schmidt orthogonalization of a list of vectors using the symplectic inner product.

    Parameters:
        vec_list (list[np.ndarray]): A list of vectors. `dtype=np.uint8`, `ndim=1`, and `size` is even.

    Returns:
        ret (list[np.ndarray]): A list of orthogonal vectors.
    '''
    assert (len(vec_list)>=2) and all(x.ndim==1 for x in vec_list) and (vec_list[0].size%2==0)
    assert all([x.size==vec_list[0].size for x in vec_list])
    assert all(x.dtype.type==np.uint8 for x in vec_list)
    vec_list = list(vec_list) #make a copy
    vec0_list = []
    vec1_list = []
    while len(vec_list)>=2:
        v0 = vec_list.pop(0)
        for ind0 in range(len(vec_list)):
            if get_inner_product(v0, vec_list[ind0])==1:
                v1 = vec_list.pop(ind0)
                tmp0 = (get_inner_product(v0,x) for x in vec_list)
                tmp1 = (get_inner_product(v1,x) for x in vec_list)
                vec_list = [(x+y*v1+z*v0)%2 for x,y,z in zip(vec_list,tmp0,tmp1)]
                vec0_list.append(v0)
                vec1_list.append(v1)
                break
    ret = vec0_list + vec1_list
    return ret


def find_transvection(v0:np.ndarray, v1:np.ndarray):
    r'''find a transvection that maps v0 to v1. see [doi-link](https://doi.org/10.1063%2F1.4903507) Lemma 2

    h0,h1 = find_transvection(v0, v1)

    v1 = transvection(transvection(v0, h0), h1) = transvection(v0, h0, h1)

    Parameters:
        v0 (np.ndarray): The first vector. `dtype=np.uint8`, `ndim=1`, and `size` is even.
        v1 (np.ndarray): The second vector. `dtype=np.uint8`, `ndim=1`, and `size` is even.

    Returns:
        ret (np.ndarray): The transvections. `dtype=np.uint8`, `ndim=2`, and `shape=(2,size)`
    '''
    assert (v0.ndim==1) and (v0.shape==v1.shape) and (v0.size%2==0)
    N0 = v0.size//2
    tmp0 = np.zeros(2*N0, dtype=np.uint8)
    assert not np.array_equal(v0, tmp0)
    assert not np.array_equal(v1, tmp0)
    if np.array_equal(v0, v1):
        ret = np.zeros((2,2*N0), dtype=np.uint8)
    elif get_inner_product(v0,v1)==1:
        ret = np.stack([(v0+v1)%2, v0*0], axis=0)
    else:
        # find a pair where they are both not 00
        indV0 = np.logical_or(v0[:N0], v0[N0:])
        indV1 = np.logical_or(v1[:N0], v1[N0:])
        tmp2 = np.nonzero(np.logical_and(indV0, indV1))[0]
        if len(tmp2):
            ind0 = tmp2[0]
            a = (v0[ind0] + v1[ind0])%2
            b = (v0[ind0+N0] + v1[ind0+N0])%2
            if (a==0) and (b==0):
                b = 1
                a = np.logical_xor(v0[ind0], v0[ind0+N0])
            v2 = np.zeros(2*N0, dtype=np.uint8)
            v2[ind0] = a
            v2[ind0+N0] = b
            ret = np.stack([(v1+v2)%2, (v0+v2)%2], axis=0)
        else:
            # look for two places where v0 has 00 and v1 doesn't, and vice versa
            tmp0 = np.nonzero(np.logical_and(indV0, np.logical_not(indV1)))[0]
            v2 = np.zeros(N0*2, dtype=np.uint8)
            if len(tmp0):
                ind0 = tmp0[0]
                if v0[ind0]==v0[ind0+N0]:
                    v2[ind0+N0] = 1
                else:
                    v2[ind0] = v0[ind0+N0]
                    v2[ind0+N0] = v0[ind0]
            tmp0 = np.nonzero(np.logical_and(np.logical_not(indV0), indV1))[0]
            if len(tmp0):
                ind0 = tmp0[0]
                if v1[ind0]==v1[ind0+N0]:
                    v2[ind0+N0] = 1
                else:
                    v2[ind0] = v1[ind0+N0]
                    v2[ind0+N0] = v1[ind0]
            ret = np.stack([(v1+v2)%2, (v0+v2)%2], axis=0)
    return ret


def from_int_tuple(int_tuple):
    r'''Convert an integer tuple to a symplectic matrix over GF(2).

    Parameters:
        int_tuple (tuple[int]): The integer tuple. `len(int_tuple)=2*N0`

    Returns:
        ret (np.ndarray): The symplectic matrix. `dtype=np.uint8`, `ndim=2`, and `shape=(2*N0,2*N0)`
    '''
    assert len(int_tuple)%2==0
    N0 = len(int_tuple)//2
    ai = int_tuple[-2] #4^n-1
    bi = int_tuple[-1] #2^(2n-1)

    f1 = int_to_bitarray(ai+1, 2*N0)
    e1 = np.array([1]+[0]*(2*N0-1), dtype=np.uint8)
    T0,T1 = find_transvection(e1, f1) #T1 T0 e1 = f1
    bits = int_to_bitarray(bi, 2*N0-1)
    tmp0 = np.concatenate([e1[:1], bits[1:N0], e1[1:2], bits[N0:]])
    h0 = transvection(tmp0, T1, T0)
    Tprime_T = (T1,T0,h0) if (bits[0]==1) else (T1,T0,h0,f1)
    g = np.eye(2*N0, dtype=np.uint8)
    if N0>1:
        tmp0 = from_int_tuple(int_tuple[:-2])
        g[1:N0,1:N0] = tmp0[:(N0-1),:(N0-1)]
        g[1:N0,(N0+1):] = tmp0[:(N0-1),(N0-1):]
        g[(N0+1):,1:N0] = tmp0[(N0-1):,:(N0-1)]
        g[(N0+1):,(N0+1):] = tmp0[(N0-1):,(N0-1):]
    ret = transvection(g, *Tprime_T)
    return ret


def to_int_tuple(mat:np.ndarray):
    r'''Convert a symplectic matrix over GF(2) to an integer tuple.

    Parameters:
        mat (np.ndarray): The symplectic matrix. `dtype=np.uint8`, `ndim=2`, and `shape=(2*N0,2*N0)`

    Returns:
        ret (tuple[int]): The integer tuple. `len(ret)=2*N0`
    '''
    assert (mat.dtype.type==np.uint8) and (mat.ndim==2) and (mat.shape[0]==mat.shape[1]) and (mat.shape[0]%2==0)
    N0 = mat.shape[0]//2
    e1 = np.array([1]+[0]*(2*N0-1), dtype=np.uint8)
    T0,T1 = find_transvection(mat[0], e1)
    tw = transvection(mat[N0], T1, T0)
    h0 = np.concatenate([e1[:1], tw[1:N0], e1[1:2], tw[(N0+1):]])
    ai = bitarray_to_int(mat[0]) - 1
    bi = bitarray_to_int(np.concatenate([tw[:N0], tw[(N0+1):]]))
    if N0==1:
        ret = (ai, bi)
    else:
        tmp0 = (T1,T0,h0,e1) if (tw[0]==0) else (T1,T0,h0)
        tmp1 = transvection(np.concatenate([mat[1:N0], mat[(N0+1):]], axis=0), *tmp0)
        tmp2 = np.concatenate([tmp1[:,1:N0], tmp1[:,(N0+1):]], axis=1)
        ret = to_int_tuple(tmp2) + (ai,bi)
    return ret


def inverse(mat:np.ndarray):
    r'''Get the inverse of a symplectic matrix over GF(2).

    Parameters:
        mat (np.ndarray): The symplectic matrix. `dtype=np.uint8`, `ndim=2`, and `shape=(2*N0,2*N0)`

    Returns:
        ret (np.ndarray): The inverse of the symplectic matrix. `dtype=np.uint8`, `ndim=2`, and `shape=(2*N0,2*N0)`
    '''
    N0 = mat.shape[0]//2
    ret = np.roll(mat.T, N0, axis=(0,1))
    return ret

import galois
import numpy as np

GF2 = galois.GF2

def str_to_gf4(x:str|list[str]):
    isone = isinstance(x, str)
    if isone:
        x = [x]
    N0 = len(x)
    N1= len(x[0])
    ret = np.zeros((N0, 2*N1), dtype=np.uint8)
    for ind0 in range(N0):
        assert isinstance(x[ind0], str) and (set(x[ind0])<=set('IXYZ'))
        ret[ind0,:N1] = np.array([(y in 'XY') for y in x[ind0]], dtype=np.uint8)
        ret[ind0,N1:] = np.array([(y in 'ZY') for y in x[ind0]], dtype=np.uint8)
    if isone:
        ret = ret[0]
    return ret


def gf4_to_str(bitXZ:np.ndarray):
    assert (bitXZ.ndim in (1,2)) and (bitXZ.shape[-1]%2==0)
    assert (bitXZ.dtype==np.uint8)
    assert bitXZ.max()<=1
    isone = bitXZ.ndim==1
    if isone:
        bitXZ = bitXZ.reshape(1,-1)
    N1 = bitXZ.shape[-1]//2
    ret = [''.join(['IZXY'[2*x0+z0] for x0,z0 in zip(x[:N1], x[N1:])]) for x in bitXZ]
    if isone:
        ret = ret[0]
    return ret


def matmul_gf4(np0:np.ndarray|GF2, np1:np.ndarray|GF2)->GF2:
    assert np0.shape[-1]%2==0
    assert np0.shape[-1] == np1.shape[(0 if (np1.ndim==1) else -2)]
    if not isinstance(np0, GF2):
        assert (np0.dtype==np.uint8) and (np1.dtype==np.uint8)
        assert (np0.max()<=1) and (np1.max()<=1)
        np0 = GF2(np0)
        np1 = GF2(np1)
    N0 = np0.shape[-1] // 2
    ind0 = slice(0,N0)
    ind1 = slice(N0,2*N0)
    np0a = np0[...,ind0]
    np0b = np0[...,ind1]
    if np1.ndim==1:
        assert np1.shape[0]
        ret = np0a @ np1[ind1] + np0b @ np1[ind0]
    else: #np1.ndim>=2
        ret = np0a @ np1[...,ind1,:] + np0b @ np1[...,ind0,:]
    return ret


def _rand_nonzero_GF2_vector(sz:int|list[int], np_rng:np.random.Generator):
    while True:
        x = np_rng.integers(0, 2, size=sz, dtype=np.uint8)
        if np.any(x!=0):
            break
    return x


def get_subspace_minus(np0:np.ndarray|GF2, np1:np.ndarray|GF2, tag_reduce:bool=True)->GF2:
    # np0 - np1
    assert (np0.ndim==2) and (np1.ndim==2)
    assert (np0.shape[1]==np1.shape[1])
    if not isinstance(np0, GF2):
        assert (np0.dtype==np.uint8) and (np0.max()==1)
        np0 = GF2(np0)
    if not isinstance(np1, GF2):
        assert (np1.dtype==np.uint8) and (np1.max()==1)
        np1 = GF2(np1)
    if tag_reduce:
        np0 = np0.row_space()
        np1 = np1.row_space()
        assert np.concatenate([np0,np1]).row_space().shape[0]==np0.shape[0]
    assert np1.shape[0]<=np0.shape[0]
    if np0.shape[0]==np1.shape[0]:
        ret = GF2.zeros((0,np0.shape[1]))
    else:
        tmp1 = np.concatenate([np1,np0], axis=0)
        tmp2 = tmp1.T.row_reduce()
        rref = tmp2[:tmp2.max(axis=1).sum()]
        pivot = (rref!=0).argmax(axis=1)
        ind0 = sorted(set(range(np1.shape[0],tmp1.shape[0])) & set(pivot.tolist()))
        ret = tmp1[ind0]
    return ret


def get_logical_from_stabilizer(stab:np.ndarray|GF2, tag_reduce:bool=True, seed=None)->tuple[GF2,GF2]:
    assert (stab.ndim==2) and (stab.shape[1]%2==0)
    if not isinstance(stab, GF2):
        assert (stab.dtype==np.uint8) and (stab.max()==1)
        stab = GF2(stab)
    if tag_reduce:
        stab = stab.row_space()
    n = stab.shape[1]//2
    k = n - stab.shape[0]
    assert np.all(matmul_gf4(stab, stab.T)==0)
    tmp0 = stab.null_space()
    logical = get_subspace_minus(np.concatenate([tmp0[:,n:], tmp0[:,:n]], axis=1), stab)
    if k==1:
        logicalX = logical[:1]
        logicalZ = logical[1:]
    else:
        np_rng = np.random.default_rng(seed)
        kernel = (logical[:,:n] @ logical[:,n:].T + logical[:,n:] @ logical[:,:n].T)
        # Every symmetric full-rank (2k,2k) matrix over GF(2) can be transformed (via congruence) into a block diagonal canonical form consisting of blocks
        vec0 = GF2(_rand_nonzero_GF2_vector((1,len(kernel)), np_rng))
        for _ in range(k-1):
            tmp0 = get_subspace_minus((vec0 @ kernel).null_space(), vec0)
            vec0 = np.concatenate([vec0, GF2(_rand_nonzero_GF2_vector((1,len(tmp0)), np_rng)) @ tmp0], axis=0)
        tmp0 = get_subspace_minus(GF2(np.eye(len(kernel), dtype=np.uint8)), vec0)
        vec1 = np.linalg.inv(tmp0 @ kernel @ vec0.T) @ tmp0
        logicalX = vec0 @ logical
        logicalZ = vec1 @ logical
    return logicalX, logicalZ

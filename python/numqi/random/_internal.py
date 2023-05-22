import math
import random
import itertools
import numpy as np
import scipy.linalg


from numqi.gellmann import gellmann_basis_to_matrix
from numqi.param import real_matrix_to_special_unitary


def get_random_rng(rng_or_seed=None):
    if rng_or_seed is None:
        ret = random.Random()
    elif isinstance(rng_or_seed, random.Random):
        ret = rng_or_seed
    else:
        ret = random.Random(int(rng_or_seed))
    return ret


def get_numpy_rng(rng_or_seed=None):
    if rng_or_seed is None:
        ret = np.random.default_rng()
    elif isinstance(rng_or_seed, np.random.Generator):
        ret = rng_or_seed
    else:
        seed = int(rng_or_seed)
        ret = np.random.default_rng(seed)
    return ret


def _random_complex(*size, seed=None):
    np_rng = get_numpy_rng(seed)
    ret = np_rng.normal(size=size + (2,)).astype(np.float64, copy=False).view(np.complex128).reshape(size)
    return ret


def rand_haar_state(dim, tag_complex=True, seed=None):
    r'''Return a random state vector from the Haar measure on the unit sphere in $\mathbb{C}^{d}$.

    $$\left\{ |\psi \rangle \in \mathbb{C} ^d\,\,: \left\| |\psi \rangle \right\| _2=1 \right\}$$

    Parameters:
        dim (int): The dimension of the Hilbert space that the state should be sampled from.
        tag_complex (bool): If True, use complex normal distribution. If False, use real normal distribution.
        seed ([None], int, numpy.RandomState): If int or RandomState, use it for RNG. If None, use default RNG.

    Returns:
        ret (numpy.ndarray): shape=(`dim`,), dtype=np.complex128
    '''
    # http://www.qetlab.com/RandomStateVector
    ret = _random_complex(dim, seed=seed)
    if tag_complex:
        ret = _random_complex(dim, seed=seed)
    else:
        np_rng = get_numpy_rng(seed)
        ret = np_rng.normal(size=dim)
    ret /= np.linalg.norm(ret)
    return ret

def rand_haar_unitary(N0, seed=None):
    # http://www.qetlab.com/RandomUnitary
    # https://pennylane.ai/qml/demos/tutorial_haar_measure.html
    ginibre_ensemble = _random_complex(N0, N0, seed=seed)
    Q,R = np.linalg.qr(ginibre_ensemble)
    tmp0 = np.sign(np.diag(R).real)
    tmp0[tmp0==0] = 1
    ret = Q * tmp0
    return ret

def rand_unitary_matrix(N0, tag_complex=True, seed=None):
    # TODO special=True/False, special unitary (orthogonal) matrix
    np_rng = get_numpy_rng(seed)
    if tag_complex:
        tmp0 = np_rng.normal(size=(N0,N0)) + 1j*np_rng.normal(size=(N0,N0))
        tmp0 = tmp0 + np.conjugate(tmp0.T)
    else:
        tmp0 = np_rng.normal(size=(N0,N0))
        tmp0 = tmp0 + tmp0.T
    ret = np.linalg.eigh(tmp0)[1]
    return ret


def rand_density_matrix(dim, k=None, kind='haar', seed=None):
    r'''Generate random density matrix

    $$\left\{ \rho \in \mathbb{C} ^{d\times d}\,: \rho \succeq 0,\mathrm{Tr}\left[ \rho \right] =1 \right\}$$

    Parameters:
        dim (int): The dimension of the Hilbert space that the state should be sampled from.
        k (int): The rank of the density matrix. If None, k=dim.
        kind (str): 'haar' or 'bures'
        seed ([None], int, numpy.RandomState): If int or RandomState, use it for RNG. If None, use default RNG.

    see also: [qetlab/RandomDensityMatrix](http://www.qetlab.com/RandomDensityMatrix)

    Returns:
        ret (numpy.ndarray): shape=(`dim`,`dim`), dtype=np.complex128
    '''
    np_rng = get_numpy_rng(seed)
    assert kind in {'haar','bures'}
    if k is None:
        k = dim
    if kind=='haar':
        ginibre_ensemble = _random_complex(dim, k, seed=np_rng)
    else:
        tmp0 = _random_complex(dim, k, seed=np_rng)
        ginibre_ensemble = (rand_haar_unitary(dim, seed=np_rng) + np.eye(dim)) @ tmp0
    ret = ginibre_ensemble @ ginibre_ensemble.T.conj()
    ret /= np.trace(ret)
    return ret


# rand_kraus_operator
def rand_kraus_op(num_term, dim0, dim1=None, tag_complex=True, seed=None):
    if dim1 is None:
        dim1 = dim0
    np_rng = get_numpy_rng(seed)
    if tag_complex:
        z0 = np_rng.normal(size=(num_term,dim1,dim0*2)).astype(np.float64, copy=False).view(np.complex128)
    else:
        z0 = np_rng.normal(size=(num_term,dim1,dim0)).astype(np.float64, copy=False)
    EVL,EVC = np.linalg.eigh(z0.reshape(-1,dim0).T.conj() @ z0.reshape(-1,dim0))
    assert all(EVL>=0)
    ret = z0 @ np.linalg.inv(EVC*np.sqrt(EVL)).T.conj()
    return ret


def rand_choi_op(dim_in, dim_out, seed=None):
    # ret(dim_in*dim_out, dim_in*dim_out)
    np_rng = get_numpy_rng(seed)
    N0 = dim_in*dim_out
    tmp0 = np_rng.normal(size=(N0,N0)) + 1j*np_rng.normal(size=(N0,N0))
    np0 = scipy.linalg.expm(tmp0 + tmp0.T.conj())
    tmp1 = np.einsum(np0.reshape(dim_in, dim_out, dim_in, dim_out), [0,1,2,1], [0,2], optimize=True)
    # matrix square root should be the same
    tmp2 = np.linalg.inv(scipy.linalg.sqrtm(tmp1).astype(tmp1.dtype))
    # TODO .astype(xxx.dtype) scipy-v1.10 bug https://github.com/scipy/scipy/issues/18250
    # tmp2 = np.linalg.inv(scipy.linalg.cholesky(tmp1))
    ret = np.einsum(np0.reshape(dim_in,dim_out,dim_in,dim_out), [0,1,2,3],
            tmp2.conj(), [0,4], tmp2, [2,5], [4,1,5,3], optimize=True).reshape(N0,N0)
    return ret


def rand_bipartite_state(dimA, dimB=None, k=None, seed=None, return_dm=False):
    r'''Generate random bipartite pure state

    $$\left\{ |\psi \rangle \in \mathbb{C} ^{d_1d_2}\,\,: \left\| |\psi \rangle \right\| _2=1 \right\}$$

    see also [qetlab/RandomStateVector](http://www.qetlab.com/RandomStateVector)

    Parameters:
        dimA (int): dimension of subsystem A
        dimB (int,None): dimension of subsystem B, if None, `dimB=dimA`
        k (int): rank of the state
        seed (int,None,numpy.RandomState): random seed
        return_dm (bool): if True, return density matrix, otherwise return state vector (default)

    Returns:
        ret (numpy.ndarray):
            if `return_dm=True`, density matrix, shape=(dimA*dimB, dimA*dimB), dtype=complex128
            if `return_dm=False`, state vector, shape=(dimA*dimB,), dtype=complex128

    '''
    np_rng = get_numpy_rng(seed)
    if dimB is None:
        dimB = dimA
    if k is None:
        ret = rand_haar_state(dimA*dimB, np_rng)
    else:
        assert (0<k) and (k<=dimA) and (k<=dimB)
        tmp0 = np.linalg.qr(_random_complex(dimA, dimA, seed=np_rng), mode='complete')[0][:,:k]
        tmp1 = np.linalg.qr(_random_complex(dimB, dimB, seed=np_rng), mode='complete')[0][:,:k]
        tmp2 = _random_complex(k, seed=np_rng)
        tmp2 /= np.linalg.norm(tmp2)
        ret = ((tmp0*tmp2) @ tmp1.T).reshape(-1)
    if return_dm:
        ret = ret[:,np.newaxis] * ret.conj()
    return ret


def rand_separable_dm(dimA, dimB=None, k=2, seed=None):
    r'''Generate random separable density matrix

    $$\left\{ \rho \in \mathbb{C} ^{d_1d_2\times d_1d_2}\,\,: \rho =\sum_k{p_i\rho _{i}^{\left( A \right)}\otimes \rho _{i}^{\left( B \right)}} \right\}$$

    Parameters:
        dimA (int): dimension of subsystem A
        dimB (int,None): dimension of subsystem B, if None, `dimB=dimA`
        k (int): number of terms in the separable state
        seed (int,None,numpy.RandomState): random seed

    Returns:
        ret (numpy.ndarray): density matrix, shape=(dimA*dimB, dimA*dimB), dtype=complex128
    '''
    if dimB is None:
        dimB = dimA
    np_rng = get_numpy_rng(seed)
    probability = np_rng.uniform(0, 1, size=k)
    probability /= probability.sum()
    ret = 0
    for ind0 in range(k):
        tmp0 = rand_density_matrix(dimA, kind='haar', seed=np_rng)
        tmp1 = rand_density_matrix(dimB, kind='haar', seed=np_rng)
        ret = ret + probability[ind0] * np.kron(tmp0, tmp1)
    return ret


def rand_hermite_matrix(N0, eig=None, tag_complex=True, seed=None):
    np_rng = get_numpy_rng(seed)
    if eig is None:
        if tag_complex:
            tmp0 = np_rng.normal(size=(N0,N0)) + 1j*np_rng.normal(size=(N0,N0))
            ret = tmp0 + tmp0.T.conj()
        else:
            tmp0 = np_rng.normal(size=(N0,N0))
            ret = tmp0 + tmp0.T
    else:
        min_eig,max_eig = eig
        EVL = np_rng.uniform(min_eig, max_eig, size=(N0,))
        EVC = rand_unitary_matrix(N0, tag_complex, seed=np_rng)
        tmp0 = EVC.T.conj() if tag_complex else EVC.T
        ret = (EVC * EVL) @ tmp0
    return ret


def rand_channel_matrix_space(dim_in, num_term, seed=None):
    np_rng = get_numpy_rng(seed)
    ret = [np.eye(dim_in)]
    for _ in range(num_term-1):
        tmp0 = np_rng.normal(size=(dim_in,dim_in))+1j*np_rng.normal(size=(dim_in,dim_in))
        ret.append(tmp0 + tmp0.T.conj())
    ret = np.stack(ret)
    return ret


def rand_quantum_channel_matrix_subspace(dim_in, num_hermite, seed=None):
    if hasattr(num_hermite, '__len__'):
        assert len(num_hermite)==2
        tag_real = True
        num_sym,num_antisym = [int(x) for x in num_hermite]
    else:
        tag_real = False
        num_hermite = int(num_hermite)
    np_rng = get_numpy_rng(seed)
    N0 = dim_in
    ret = [np.eye(N0)[np.newaxis]]
    if tag_real:
        N1 = (N0*(N0-1)) // 2
        assert (1<=num_sym) and (num_sym<=(N0*N0-N1))
        if num_sym>1: #aS,aA,aD,aI
            tmp0 = rand_unitary_matrix(N1+N0-1, tag_complex=False, seed=np_rng)[:(num_sym-1)]
            tmp1 = np.zeros((num_sym-1, N0*N0), dtype=np.float64)
            tmp1[:,:N1] = tmp0[:,:N1]
            tmp1[:,(2*N1):-1] = tmp0[:,N1:]
            ret.append(gellmann_basis_to_matrix(tmp1).real)
        if num_antisym>0:
            tmp0 = np.zeros((num_antisym, N0*N0), dtype=np.float64)
            tmp0[:,N1:(2*N1)] = rand_unitary_matrix(N1, tag_complex=False, seed=np_rng)[:num_antisym]
            ret.append(gellmann_basis_to_matrix(tmp0).imag)
    else:
        assert num_hermite>=1
        if num_hermite>1:
            tmp0 = rand_unitary_matrix(N0*N0-1, tag_complex=False, seed=np_rng)[:(num_hermite-1)]
            tmp1 = gellmann_basis_to_matrix(np.concatenate([tmp0,np.zeros([num_hermite-1,1])], axis=1))
            ret.append(tmp1)
    ret = np.concatenate(ret, axis=0)
    return ret


def rand_ABk_density_matrix(dimA, dimB, kext, seed=None):
    assert kext>=1
    np_rng = get_numpy_rng(seed)
    tmp0 = np_rng.normal(size=(dimA*dimB**kext, dimA*dimB**kext)) + 1j*np_rng.normal(size=(dimA*dimB**kext, dimA*dimB**kext))
    tmp0 = tmp0 @ tmp0.T.conj()
    if kext==1:
        ret = tmp0.reshape([dimA]+[dimB]*kext+[dimA]+[dimB]*kext) / np.trace(tmp0)
    else:
        np0 = tmp0.reshape([dimA]+[dimB]*kext+[dimA]+[dimB]*kext) / (np.trace(tmp0)*math.factorial(kext))
        ret = np0.copy()
        for indI in list(itertools.permutations(list(range(kext))))[1:]:
            tmp0 = [0] + [(1+x) for x in indI] + [kext+1] + [(2+kext+x) for x in indI]
            ret += np.transpose(np0, tmp0)
    ret = ret.reshape(dimA*dimB**kext, dimA*dimB**kext)
    return ret


def rand_reducible_matrix_subspace(num_matrix, partition, return_unitary=False, seed=None):
    np_rng = get_numpy_rng(seed)
    partition = [int(x) for x in partition]
    assert len(partition)>1 and all(x>0 for x in partition)
    N0 = sum(partition)
    unitary = rand_unitary_matrix(N0, tag_complex=False, seed=np_rng)
    tmp0 = np.cumsum(np.array([0]+partition))
    unitary_part_list = [unitary[x:y] for x,y in zip(tmp0[:-1],tmp0[1:])]
    matrix_subspace = 0
    for unitary_i in unitary_part_list:
        dim = unitary_i.shape[0]
        tmp0 = np_rng.normal(size=(num_matrix,dim,dim))
        matrix_subspace = matrix_subspace + unitary_i.T @ tmp0 @ unitary_i
    ret = (matrix_subspace,unitary) if return_unitary else matrix_subspace
    return ret


def rand_symmetric_inner_product(N0, zero_eps=1e-10, seed=None):
    # (ret0)matB(np,float,(num_matrix,N0,N0))
    # (ret1)matU(np,float,(N0,N0))
    # for all x, x^T B U x = x^T U^T B x
    # TODO complex?
    np_rng = get_numpy_rng(seed)
    assert N0>=2
    matU = np_rng.normal(size=(N0,N0))
    tmp0 = np.eye(N0)
    tmp1 = np.einsum(matU, [3,1], tmp0, [0,2], [0,1,2,3], optimize=True)
    tmp2 = np.einsum(matU, [2,0], tmp0, [1,3], [0,1,2,3], optimize=True)
    tmp3 = np.einsum(matU, [2,1], tmp0, [0,3], [0,1,2,3], optimize=True)
    tmp4 = np.einsum(matU, [3,0], tmp0, [1,2], [0,1,2,3], optimize=True)
    z0 = (tmp1-tmp2-tmp3+tmp4).reshape(-1, N0*N0)
    EVL,EVC = np.linalg.eigh(z0.T @ z0)
    num_zero = (EVL<zero_eps).sum()
    assert num_zero>0
    matB = (EVC[:,:num_zero].T).reshape(-1, N0, N0)
    return matB, matU


def rand_orthonormal_matrix_basis(num_orthonormal, dim_qudit, num_qudit=1, num_sample=None, with_I=False, seed=None):
    np_rng = get_numpy_rng(seed)
    tag_one = num_sample is None
    num_sample = 1 if tag_one else num_sample
    assert num_sample>=1
    # we can always select the first orthonormal as the computational basis
    povm_basis = np.zeros((dim_qudit,dim_qudit,dim_qudit), dtype=np.float64)
    ind1 = np.arange(dim_qudit, dtype=np.int64)
    povm_basis[ind1,ind1,ind1] = 1
    ret = []
    for _ in range(num_sample):
        tmp0 = np_rng.normal(size=(num_qudit*(num_orthonormal-1),dim_qudit,dim_qudit))
        unitary = real_matrix_to_special_unitary(tmp0).reshape(num_qudit, num_orthonormal-1, dim_qudit, dim_qudit)
        tmp1 = [[(y[:,:,np.newaxis]*y[:,np.newaxis].conj()) for y in x] for x in unitary]
        tmp1 = [np.stack([povm_basis]+x, axis=0) for x in tmp1]
        tmp2 = tmp1[0]
        for ind1 in range(1, num_qudit):
            tmp3 = np.einsum(tmp2, [0,1,2,3], tmp1[ind1], [0,4,5,6], [0,1,4,2,5,3,6], optimize=True)
            tmp2 = tmp3.reshape(num_orthonormal, dim_qudit**(ind1+1), dim_qudit**(ind1+1), dim_qudit**(ind1+1))
        tmp2 = tmp2.reshape(num_orthonormal*(dim_qudit**num_qudit), dim_qudit**num_qudit, dim_qudit**num_qudit)
        if with_I:
            tmp2 = np.concatenate([np.eye(tmp2.shape[1])[np.newaxis], tmp2], axis=0)
        ret.append(tmp2)
    if tag_one:
        ret = ret[0]
    return ret

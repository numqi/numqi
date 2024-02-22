import math
import random
import itertools
import numpy as np
import scipy.linalg


from numqi.gellmann import gellmann_basis_to_matrix
from numqi.manifold import to_special_orthogonal_exp

from ._public import get_random_rng, get_numpy_rng

# TODO batch_size
# TODO refactor to use numqi.manifold

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


def rand_haar_unitary(dim:int, seed:None|int|np.random.RandomState=None):
    r'''Return a random unitary matrix from the Haar measure on the unitary group $U(N)$.

    also see
        http://www.qetlab.com/RandomUnitary
        https://pennylane.ai/qml/demos/tutorial_haar_measure.html

    Parameters:
        dim (int): the column (row) of matrix
        seed ([None], int, numpy.RandomState): If int or RandomState, use it for RNG. If None, use default RNG.

    Returns:
        ret (numpy.ndarray): shape=(`dim`,`dim`), dtype=np.complex128
    '''
    ginibre_ensemble = _random_complex(dim, dim, seed=seed)
    Q,R = np.linalg.qr(ginibre_ensemble)
    tmp0 = np.sign(np.diag(R).real)
    tmp0[tmp0==0] = 1
    ret = Q * tmp0
    return ret


def rand_special_orthogonal_matrix(dim:int, batch_size:None|int=None,
        tag_complex:bool=False, seed:None|int|np.random.RandomState=None):
    r'''generate random special orthogonal matrix

    Parameters:
        dim (int): the column (row) of matrix
        batch_size (int,None): If None, return a single matrix. If int, return a batch of matrices.
        tag_complex (bool): If True, `ret` is a special unitary matrix. If False, `ret` is a special orthogonal matrix.
        seed ([None], int, numpy.RandomState): If int or RandomState, use it for RNG. If None, use default RNG.

    Returns:
        ret (numpy.ndarray): shape=(`batch_size`,`dim`,`dim`), dtype=np.complex128 if `tag_complex=True` else np.float64
    '''
    assert dim>=2
    np_rng = get_numpy_rng(seed)
    is_single = (batch_size is None)
    if is_single:
        batch_size = 1
    tmp0 = (dim*dim-1) if tag_complex else ((dim*dim-dim)//2)
    ret = to_special_orthogonal_exp(np_rng.normal(size=(batch_size,tmp0)), dim)
    if is_single:
        ret = ret[0]
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


def rand_kraus_op(num_term:int, dim_in:int, dim_out:int, tag_complex:bool=True, seed=None):
    r'''Generate random Kraus operator

    $$ \lbrace K\in \mathbb{C} ^{k\times d_o\times d_i}\,\,: \sum_s{K_{s}^{\dagger}K_s}=I_{d_i} \rbrace $$

    Parameters:
        num_term (int): number of terms in the Kraus operator
        dim_in (int): dimension of input space
        dim_out (int): dimension of output space
        tag_complex (bool): If True, `ret` is a complex Kraus operator. If False, `ret` is a real Kraus operator.
        seed (int,None,numpy.RandomState): random seed

    Returns:
        ret (numpy.ndarray): shape=(`num_term`,`dim_out`,`dim_in`), dtype=np.complex128 if `tag_complex=True` else `np.float64`
    '''
    np_rng = get_numpy_rng(seed)
    if tag_complex:
        z0 = np_rng.normal(size=(num_term,dim_out,dim_in*2)).astype(np.float64, copy=False).view(np.complex128)
    else:
        z0 = np_rng.normal(size=(num_term,dim_out,dim_in)).astype(np.float64, copy=False)
    EVL,EVC = np.linalg.eigh(z0.reshape(-1,dim_in).T.conj() @ z0.reshape(-1,dim_in))
    assert all(EVL>=0)
    ret = z0 @ np.linalg.inv(EVC*np.sqrt(EVL)).T.conj()
    return ret


# TODO change to choi_op(dim_out,dim_in,dim_out_dim_in)
def rand_choi_op(dim_in:int, dim_out:int, rank:int=None, seed=None):
    r'''Generate random Choi operator

    $$ \lbrace C\in \mathbb{C} ^{d_id_o\times d_id_o}\,\,:C\succeq 0,\mathrm{Tr}_{d_o}\left[ C \right] =I_{d_i} \rbrace $$

    Parameters:
        dim_in (int): dimension of input space
        dim_out (int): dimension of output space
        seed (int,None,numpy.RandomState): random seed

    Returns:
        ret (numpy.ndarray): shape=(`dim_in*dim_out`,`dim_in*dim_out`), dtype=np.complex128
            A reasonable reshape is `ret.reshape(dim_in,dim_out,dim_in,dim_out)`
    '''
    # ret(dim_in*dim_out, dim_in*dim_out)
    np_rng = get_numpy_rng(seed)
    N0 = dim_in*dim_out
    if rank is None:
        rank = N0
    tmp0 = np_rng.normal(size=(N0,rank)) + 1j*np_rng.normal(size=(N0,rank))
    np0 = tmp0 @ tmp0.T.conj()
    tmp1 = np.einsum(np0.reshape(dim_in, dim_out, dim_in, dim_out), [0,1,2,1], [0,2], optimize=True)
    # matrix square root should be the same
    tmp2 = np.linalg.inv(scipy.linalg.sqrtm(tmp1).astype(tmp1.dtype))
    # TODO .astype(xxx.dtype) scipy-v1.10 bug https://github.com/scipy/scipy/issues/18250
    # tmp2 = np.linalg.inv(scipy.linalg.cholesky(tmp1))
    ret = np.einsum(np0.reshape(dim_in,dim_out,dim_in,dim_out), [0,1,2,3],
            tmp2.conj(), [0,4], tmp2, [2,5], [4,1,5,3], optimize=True).reshape(N0,N0)
    return ret


def rand_povm(dim:int, num_term:int, seed=None):
    r'''generate random positive operator-valued measure (POVM)

    Parameters:
        dim (int): dimension of the Hilbert space
        num_term (int): number of terms in the POVM
        seed (int,None,numpy.RandomState): random seed

    Returns:
        ret (numpy.ndarray): shape=(`num_term`,`dim`,`dim`), dtype=np.float64
    '''
    np_rng = get_numpy_rng(seed)
    tmp0 = np_rng.normal(size=(num_term,dim,dim)) + 1j*np_rng.normal(size=(num_term,dim,dim))
    tmp1 = tmp0 @ tmp0.transpose(0,2,1).conj()
    tmp2 = np.linalg.inv(scipy.linalg.sqrtm(tmp1.sum(axis=0)).astype(tmp1.dtype))
    ret = tmp2 @ tmp1 @ tmp2
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


def rand_separable_dm(dimA, dimB=None, k=2, seed=None, pure_term=False):
    r'''Generate random separable density matrix

    $$\left\{ \rho \in \mathbb{C} ^{d_1d_2\times d_1d_2}\,\,: \rho =\sum_k{p_i\rho _{i}^{\left( A \right)}\otimes \rho _{i}^{\left( B \right)}} \right\}$$

    Parameters:
        dimA (int): dimension of subsystem A
        dimB (int,None): dimension of subsystem B, if None, `dimB=dimA`
        k (int): number of terms in the separable state
        seed (int,None,numpy.RandomState): random seed
        pure_term (bool): if True, each term is a pure state, otherwise each term is a density matrix

    Returns:
        ret (numpy.ndarray): density matrix, shape=(dimA*dimB, dimA*dimB), dtype=complex128
    '''
    if dimB is None:
        dimB = dimA
    np_rng = get_numpy_rng(seed)
    probability = np_rng.uniform(0, 1, size=k)
    probability /= probability.sum()
    ret = 0
    if pure_term:
        for ind0 in range(k):
            tmp0 = rand_haar_state(dimA, seed=np_rng)
            tmp1 = rand_haar_state(dimB, seed=np_rng)
            tmp = np.kron(tmp0, tmp1)
            ret = ret + probability[ind0] * tmp[:,np.newaxis] * tmp.conj()
    else:
        for ind0 in range(k):
            tmp0 = rand_density_matrix(dimA, kind='haar', seed=np_rng)
            tmp1 = rand_density_matrix(dimB, kind='haar', seed=np_rng)
            ret = ret + probability[ind0] * np.kron(tmp0, tmp1)
    return ret


def rand_hermitian_matrix(d:int, eig=None, tag_complex:bool=True, seed=None):
    r'''Generate random Hermitian matrix

    $$\left\{ H\in \mathbb{C} ^{d\times d}\,\,: H=H^{\dagger } \right\}$$

    Parameters:
        d (int): dimension of matrix
        eig (None, tuple): eigenvalue range (min_eig,max_eig), if None, eigenvalue is not constrained
        tag_complex (bool): If True, `ret` is a complex Hermitian matrix. If False, `ret` is a real symmetric matrix.
        seed (None, int, numpy.RandomState): If int or RandomState, use it for RNG. If None, use default RNG.

    Returns:
        ret (numpy.ndarray): shape=(`d`,`d`), dtype=np.complex128 if `tag_complex=True` else np.float64
    '''
    np_rng = get_numpy_rng(seed)
    if eig is None:
        if tag_complex:
            tmp0 = np_rng.normal(size=(d,d)) + 1j*np_rng.normal(size=(d,d))
            ret = tmp0 + tmp0.T.conj()
        else:
            tmp0 = np_rng.normal(size=(d,d))
            ret = tmp0 + tmp0.T
    else:
        min_eig,max_eig = eig
        EVL = np_rng.uniform(min_eig, max_eig, size=(d,))
        EVC = rand_special_orthogonal_matrix(d, tag_complex=tag_complex, seed=np_rng)
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
            tmp0 = rand_special_orthogonal_matrix(N1+N0-1, tag_complex=False, seed=np_rng)[:(num_sym-1)]
            tmp1 = np.zeros((num_sym-1, N0*N0), dtype=np.float64)
            tmp1[:,:N1] = tmp0[:,:N1]
            tmp1[:,(2*N1):-1] = tmp0[:,N1:]
            ret.append(gellmann_basis_to_matrix(tmp1).real)
        if num_antisym>0:
            tmp0 = np.zeros((num_antisym, N0*N0), dtype=np.float64)
            tmp0[:,N1:(2*N1)] = rand_special_orthogonal_matrix(N1, tag_complex=False, seed=np_rng)[:num_antisym]
            ret.append(gellmann_basis_to_matrix(tmp0).imag)
    else:
        assert num_hermite>=1
        if num_hermite>1:
            tmp0 = rand_special_orthogonal_matrix(N0*N0-1, tag_complex=False, seed=np_rng)[:(num_hermite-1)]
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
    unitary = rand_special_orthogonal_matrix(N0, tag_complex=False, seed=np_rng)
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
        tmp0 = np_rng.normal(size=(num_qudit*(num_orthonormal-1),dim_qudit*dim_qudit-1))
        unitary = to_special_orthogonal_exp(tmp0, dim_qudit).reshape(num_qudit, num_orthonormal-1, dim_qudit, dim_qudit)
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


def rand_adjacent_matrix(dim:int, seed=None):
    r'''Generate random adjacent matrix of undirected graph

    Parameters:
        dim (int): number of verteces
        seed (int,None,numpy.RandomState): random seed

    Returns:
        ret (numpy.ndarray): adjacent matrix, shape=(dim,dim), dtype=np.uint8
    '''
    np_rng = get_numpy_rng(seed)
    assert dim>=2
    tmp0 = np.triu(np_rng.integers(0, 2, size=(dim,dim)), 1)
    ret = (tmp0 + tmp0.T).astype(np.uint8)
    return ret


def rand_n_sphere(dim:int, size=None, seed=None):
    r'''Generate random vector from n-sphere

    wiki-link: https://en.wikipedia.org/wiki/N-sphere

    Parameters:
        dim (int): dimension of the vector
        size (None, int, tuple): size of the output array
        seed (int,None,numpy.RandomState): random seed

    Returns:
        ret (numpy.ndarray): shape=`size`+(dim,), dtype=np.float64
    '''
    assert dim>=1
    np_rng = get_numpy_rng(seed)
    is_single = (size is None)
    if is_single:
        size = ()
    elif not hasattr(size, '__len__'):
        size = int(size),
    N0 = 1 if (len(size)==0) else np.prod(size)
    tmp0 = np_rng.normal(size=(N0,dim))
    tmp0 = tmp0 / np.linalg.norm(tmp0, axis=-1, keepdims=True)
    if is_single:
        ret = tmp0[0]
    else:
        ret = tmp0.reshape(size+(dim,))
    return ret


# TODO Lp norm ball https://mathoverflow.net/q/9185

def rand_n_ball(dim:int, size=None, seed=None):
    r'''Generate random vector from n-ball

    wiki-link: https://en.wikipedia.org/wiki/Ball_(mathematics)

    Box-Muller transform: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform

    stackexchange-link: https://stats.stackexchange.com/a/481716

    Parameters:
        dim (int): dimension of the vector
        size (None, int, tuple): size of the output array
        seed (int,None,numpy.RandomState): random seed

    Returns:
        ret (numpy.ndarray): shape=`size`+(dim,), dtype=np.float64
    '''
    assert dim>=1
    np_rng = get_numpy_rng(seed)
    is_single = (size is None)
    if is_single:
        size = ()
    elif not hasattr(size, '__len__'):
        size = int(size),
    N0 = 1 if (len(size)==0) else np.prod(size)
    tmp0 = np_rng.normal(size=(N0,dim))
    tmp0 /= np.linalg.norm(tmp0, axis=-1, keepdims=True)
    tmp1 = np_rng.uniform(0, 1, size=N0) ** (1/dim)
    tmp2 = tmp0 * tmp1[:,np.newaxis]
    if is_single:
        ret = tmp2[0]
    else:
        ret = tmp2.reshape(size+(dim,))
    return ret

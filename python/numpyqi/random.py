import numpy as np


# TODO merge with pyqet

def get_numpy_rng(np_rng_or_seed_or_none):
    if np_rng_or_seed_or_none is None:
        ret = np.random.default_rng()
    elif isinstance(np_rng_or_seed_or_none, np.random.Generator):
        ret = np_rng_or_seed_or_none
    else:
        seed = int(np_rng_or_seed_or_none)
        ret = np.random.default_rng(seed)
    return ret


def _random_complex(*size, seed=None):
    np_rng = get_numpy_rng(seed)
    ret = np_rng.normal(size=size + (2,)).astype(np.float64, copy=False).view(np.complex128).reshape(size)
    return ret


def rand_haar_state(N0, seed=None):
    # http://www.qetlab.com/RandomStateVector
    ret = _random_complex(N0, seed=seed)
    ret /= np.linalg.norm(ret)
    return ret

def rand_state(num_qubit, tag_complex=True, seed=None):
    np_rng = get_numpy_rng(seed)
    ret = np_rng.normal(size=2**num_qubit)
    if tag_complex:
        ret = ret + np_rng.normal(size=2**num_qubit)*1j
    ret = ret / np.sqrt(np.vdot(ret, ret))
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
    np_rng = get_numpy_rng(seed)
    if tag_complex:
        tmp0 = np_rng.normal(size=(N0,N0)) + 1j*np_rng.normal(size=(N0,N0))
        tmp0 = tmp0 + np.conjugate(tmp0.T)
    else:
        tmp0 = np_rng.normal(size=(N0,N0))
        tmp0 = tmp0 + tmp0.T
    ret = np.linalg.eigh(tmp0)[1]
    return ret


def rand_density_matrix(N0, k=None, kind='haar', seed=None):
    # http://www.qetlab.com/RandomDensityMatrix
    np_rng = get_numpy_rng(seed)
    assert kind in {'haar','bures'}
    if k is None:
        k = N0
    if kind=='haar':
        ginibre_ensemble = _random_complex(N0, k, seed=np_rng)
    else:
        tmp0 = _random_complex(N0, k, seed=np_rng)
        ginibre_ensemble = (rand_haar_unitary(N0, seed=np_rng) + np.eye(N0)) @ tmp0
    ret = ginibre_ensemble @ ginibre_ensemble.T.conj()
    ret /= np.trace(ret)
    return ret


def rand_kraus_operator(num_term, dim0, dim1=None, tag_complex=True, seed=None):
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

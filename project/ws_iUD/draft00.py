import itertools
import functools
import scipy.special
import numpy as np

import numqi

def get_nlocal_measurement_set(num_party, num_local):
    if not hasattr(num_local, '__len__'):
        num_local_list = int(num_local),
    else:
        num_local_list = tuple(sorted({int(x) for x in num_local}))
    assert all(0<=x<=num_party for x in num_local_list)
    ret = []
    pauli = [numqi.gate.X, numqi.gate.Y, numqi.gate.Z]
    for num_local in num_local_list:
        if num_local==0:
            ret.append(np.eye(2**num_party))
        else:
            for ind0 in itertools.combinations(range(num_party), num_local):
                tmp0 = [tuple(range(3))]*num_local
                for ind1 in itertools.product(*tmp0):
                    tmp1 = [numqi.gate.I for _ in range(num_party)]
                    for x,y in zip(ind0,ind1):
                        tmp1[x] = pauli[y]
                    for x0,x1 in zip(ind0, ind1):
                        tmp1[x0] = pauli[x1]
                    ret.append(functools.reduce(np.kron, tmp1))
    ret = np.stack(ret)
    return ret

def test_get_nlocal_measurement_set():
    for N0 in [2,3]:
        for num_local in range(1,N0+1):
            ret0 = get_nlocal_measurement_set(N0, num_local)
            tmp0 = int(scipy.special.binom(N0, num_local)) * (3**num_local)
            assert ret0.shape==(tmp0, 2**N0, 2**N0)
            assert np.abs(ret0-ret0.transpose(0,2,1).conj()).max() < 1e-10
            assert np.trace(ret0, axis1=1, axis2=2).max() < 1e-10
            tmp0 = ret0.reshape(ret0.shape[0], 4**N0)
            EVL = np.linalg.eigvalsh(tmp0 @ tmp0.T.conj())
            assert EVL[0] > 1e-7


def get_GHZ_state(N0):
    state = np.zeros(2**N0, dtype=np.float64)
    state[[0,-1]] = 1/np.sqrt(2)
    dm = state[:,np.newaxis]*state
    return state,dm


def get_hessian_mat(state, op_measure, zero_eps=1e-10):
    assert (state.ndim==1) and (op_measure.shape[1]==state.shape[0]) and (op_measure.shape[2]==state.shape[0])
    assert abs(np.linalg.norm(state)-1) < zero_eps
    assert np.abs(op_measure.conj().transpose(0,2,1)-op_measure).max() < zero_eps
    op_measure_result = ((op_measure @ state) @ state.conj()).real
    tmp0 = 2*(op_measure @ state - op_measure_result[:,np.newaxis]*state)
    tmp1 = np.concatenate([tmp0.real, tmp0.imag], axis=1)
    hessian = tmp1.T @ tmp1

    vec0 = np.concatenate([state.real, state.imag])
    vec1 = np.concatenate([-state.imag, state.real])
    assert abs(((hessian @ vec0) @ vec0)) < zero_eps
    assert abs(((hessian @ vec1) @ vec1)) < zero_eps
    EVL,EVC = np.linalg.eigh(hessian)
    rank_zero = (EVL<zero_eps).sum()
    if rank_zero > 2:
        tmp0 = (np.eye(2*state.shape[0]) - vec0[:,np.newaxis]*vec0 - vec1[:,np.newaxis]*vec1) @ EVC[:,:rank_zero]
        _,EVC0 = np.linalg.eigh(tmp0 @ tmp0.T)
        tmp1 = EVC0[:,(2-rank_zero):]
        vec2 = tmp1[:state.shape[0]] + 1j*tmp1[state.shape[0]:]
    else:
        vec2 = None
    return hessian,EVL,vec2

def get_Wtype_state(np0):
    np0 = np0 / np.linalg.norm(np0)
    N0 = np0.shape[0]
    ret = np.zeros(2**N0, dtype=np0.dtype)
    ret[2**np.arange(N0)] = np0
    return ret

np_rng = np.random.default_rng()
hf_randc = lambda *x: np_rng.normal(size=x) + 1j*np_rng.normal(size=x)

num_party = 3
state_ghz, dm_ghz = get_GHZ_state(num_party)

op_measure = get_nlocal_measurement_set(num_party, (0,1,2))
# basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(op_measure, field='real')
hessian,EVL,vec2 = get_hessian_mat(state_ghz, op_measure)

state = np.array([1,1])/np.sqrt(2)
op_measure = np.stack([numqi.gate.X, numqi.gate.Y])
hessian,EVL,vec2 = get_hessian_mat(state, op_measure)


num_party = 4
tmp0 = hf_randc(num_party)
np0 = hf_randc(num_party)/np.linalg.norm(tmp0)
state = get_Wtype_state(np0)
op_measure = get_nlocal_measurement_set(num_party, (0,1,2))
hessian,EVL,vec2 = get_hessian_mat(state, op_measure)

# basis = np.eye(2**num_party)[:-1]
# state = basis @ state
# op_measure = basis @ op_measure @ basis.T.conj()
# hessian,EVL,vec2 = get_hessian_mat(state, op_measure)

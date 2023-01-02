import numpy as np
import scipy.linalg

import numpyqi.gellmann

def build_matrix_with_index_value(dim0, dim1, index_value):
    value = np.array([x[2] for x in index_value])
    ret = np.zeros((dim0,dim1), dtype=value.dtype)
    ind0 = [x[0] for x in index_value]
    ind1 = [x[1] for x in index_value]
    ret[ind0, ind1] = value
    return ret


def is_linear_independent(np0, zero_eps=1e-7):
    if np0.shape[0]>np0.shape[1]:
        ret = False
    else:
        U = scipy.linalg.lu(np0.conj() @ np0.T)[2]
        ret = np.abs(U[-1,-1]).max()>zero_eps
    return ret


def is_space_equivalent(space0, space1, zero_eps=1e-10):
    assert (space0.ndim>=2) and (space1.ndim>=2)
    if space0.ndim>2:
        space0 = space0.reshape(space0.shape[0], -1)
    if space1.ndim>2:
        space1 = space1.reshape(space1.shape[0], -1)
    assert space0.shape[1]==space1.shape[1]
    x = np.linalg.lstsq(space0.T, space1.T, rcond=None)[0]
    ret = np.abs(x.T @ space0 - space1).max() < zero_eps
    if ret:
        x = np.linalg.lstsq(space1.T, space0.T, rcond=None)[0]
        ret = np.abs(x.T @ space1 - space0).max() < zero_eps
    return ret


def find_closed_vector_in_space(space, vec):
    assert space.ndim>=2
    if space.ndim>2:
        space = space.reshape(space.shape[0], -1)
    vec = vec.reshape(-1)
    assert space.shape[1]==vec.shape[0]
    coeff,residuals,_,_ = np.linalg.lstsq(space.T, vec, rcond=None)
    return coeff, residuals


## can be used to determine whether linear-dependent, but bad performance
# def get_PSD_min_eig(np0, hermite_eps=1e-5):
#     try:
#         tmp2 = np.linalg.inv(np0)
#     except np.linalg.LinAlgError:
#         ret = 0
#     else:
#         if np.abs(tmp2-tmp2.T.conj()).max()>hermite_eps:
#             ret = 0 #must be numerical instable caused by invertible
#         else:
#             ret = 1/scipy.sparse.linalg.eigsh(tmp2, k=1, which='LA', return_eigenvectors=False)[0]
#     return ret


def get_hs_orthogonal_basis(matrix_space, hermite=True):
    assert (matrix_space.ndim==3) and (matrix_space.shape[1]==matrix_space.shape[2])
    N0 = matrix_space.shape[0]
    dim = matrix_space.shape[1]
    assert N0 < dim*dim
    assert is_linear_independent(matrix_space.reshape(N0,dim*dim))
    if hermite:
        tmp0 = numpyqi.gellmann.matrix_to_gellmann_basis(matrix_space, norm_I='sqrt(2/d)')
        assert np.abs(tmp0.imag).max() < 1e-7
        matrix_space_q = scipy.linalg.qr(tmp0.real.T, mode='economic')[0].T
        EVC = np.linalg.eigh(np.eye(matrix_space_q.shape[1]) - matrix_space_q.T @ matrix_space_q)[1]
        matrix_space_orth_q = EVC[:,(matrix_space_q.shape[0]-matrix_space_q.shape[1]):].T
        ret = numpyqi.gellmann.gellmann_basis_to_matrix(matrix_space_orth_q, norm_I='sqrt(2/d)')
    else:
        matrix_space_q = scipy.linalg.qr(matrix_space.reshape(N0,dim*dim).T, mode='economic')[0].T
        EVC = np.linalg.eigh(np.eye(matrix_space_q.shape[1]) - matrix_space_q.T @ matrix_space_q.conj())[1]
        ret = EVC[:,(matrix_space_q.shape[0]-matrix_space_q.shape[1]):].T.reshape(-1, dim, dim)
    return ret


def reduce_matrix_space(matrix_space, zero_eps=1e-10):
    N0,dim,_ = matrix_space.shape
    _,S,V = np.linalg.svd(matrix_space.reshape(N0,-1), full_matrices=False)
    ret = V[:(S>zero_eps).sum()].reshape(-1, dim, dim)
    return ret


def get_hermite_channel_matrix_space(matrix_space, zero_eps=1e-10):
    matrix_space = reduce_matrix_space(matrix_space, zero_eps)
    N0,dim,_ = matrix_space.shape
    # assert is_linear_independent(matrix_space.reshape(N0,-1))
    tmp0 = matrix_space.reshape(N0, -1).T
    tmp1 = np.eye(dim).reshape(-1)
    z0 = np.linalg.lstsq(tmp0, tmp1, rcond=None)[0]
    assert np.abs(tmp0@z0-tmp1).max() < zero_eps
    tmp1 = matrix_space.transpose(0,2,1).conj().reshape(N0,-1).T
    z0 = np.linalg.lstsq(tmp0, tmp1, rcond=None)[0]
    assert np.abs(tmp0 @ z0 - tmp1).max() < zero_eps

    tmp0 = matrix_space + matrix_space.transpose(0,2,1).conj()
    tmp1 = 1j*(matrix_space - matrix_space.transpose(0,2,1).conj())
    tmp2 = numpyqi.gellmann.matrix_to_gellmann_basis(np.concatenate([tmp0,tmp1], axis=0))
    matrix_space_q = scipy.linalg.qr(tmp2.real.T, pivoting=True, mode='economic')[0].T[:N0]
    matrix_space_hermite = numpyqi.gellmann.gellmann_basis_to_matrix(matrix_space_q)
    return matrix_space_hermite


def matrix_space_to_kraus_op(matrix_space, is_hermite=False, zero_eps=1e-10):
    if not is_hermite:
        matrix_space = get_hermite_channel_matrix_space(matrix_space, zero_eps)
    N0,dim,_ = matrix_space.shape
    EVL = np.linalg.eigvalsh(matrix_space)
    EVL_max = EVL.max()
    s = -1/EVL_max if (EVL_max>0) else (-1/EVL.min())
    z0 = matrix_space*s + np.eye(dim)
    tmp0 = z0.sum(axis=0)
    t = 1/np.linalg.eigvalsh(tmp0).max()
    matrix_space_PSD_sum1 = np.concatenate([(np.eye(dim) - t*tmp0)[np.newaxis], t*z0], axis=0)
    EVL,EVC = np.linalg.eigh(matrix_space_PSD_sum1)
    tmp0 = np.sqrt(np.maximum(0,EVL))
    tmp1 = np.arange(N0+1)
    tmp2 = np.zeros((N0+1,N0+1,dim,dim), dtype=matrix_space_PSD_sum1.dtype)
    tmp2[tmp1,tmp1] = (EVC*tmp0[:,np.newaxis]) @ EVC.transpose(0,2,1).conj()
    kraus_op = tmp2.reshape(N0+1,(N0+1)*dim,dim)
    return kraus_op


def kraus_op_to_matrix_space(op, reduce=True, zero_eps=1e-10):
    assert op.ndim==3
    dim_in = op.shape[2]
    ret = np.einsum(op.conj(), [0,1,2], op, [3,1,4], [0,3,2,4], optimize=True).reshape(-1, dim_in, dim_in)
    if reduce:
        tmp0 = reduce_matrix_space(ret, zero_eps)
        # _,S,V = np.linalg.svd(ret.reshape(-1, dim_in*dim_in), full_matrices=False)
        # tmp0 = V[:(S>zero_eps).sum()].reshape(-1, dim_in, dim_in)
        ret = get_hermite_channel_matrix_space(tmp0, zero_eps)
    return ret

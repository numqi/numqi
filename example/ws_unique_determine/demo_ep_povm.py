import numpy as np

import numqi


def get_EP_POVM00(dim:int):
    # https://journals.aps.org/pra/abstract/10.1103/PhysRevA.93.052105 eq8
    dim = int(dim)
    assert dim>=2
    ret = np.zeros((2*dim, dim, dim), dtype=np.complex128)
    ret[0] = np.eye(dim)
    ret[1,0,0] = 1
    ind0 = np.arange(1,dim)
    ret[ind0+1, 0, ind0] = 1
    ret[ind0+1, ind0, 0] = 1
    ret[ind0+dim, 0, ind0] = -1j
    ret[ind0+dim, ind0, 0] = 1j
    return ret


def get_EP_POVM01(dim:int):
    # https://journals.aps.org/pra/abstract/10.1103/PhysRevA.93.052105 eq9
    # faiure set
    #       |00> + |02> + |20> + |22> - (|11> + |13> + |31> + |33>)
    #       (I+sx) \otimes sz
    assert dim>=4
    assert (dim%2)==0
    s12 = 1/np.sqrt(2)
    ind0 = np.arange(dim, dtype=np.int64)
    tmp0 = np.zeros((dim,dim), dtype=np.complex128) #B1
    tmp0[ind0,(ind0//2)*2] = s12
    tmp0[ind0,(ind0//2)*2+1] = s12 * (1-2*(ind0%2))
    tmp1 = np.zeros((dim,dim), dtype=np.complex128) #B2
    tmp1[ind0,(ind0//2)*2+1] = s12
    tmp1[ind0,((ind0//2)*2+2)%dim] = s12 * (1-2*(ind0%2))
    tmp2 = np.zeros((dim,dim), dtype=np.complex128) #B3
    tmp2[ind0,(ind0//2)*2] = s12
    tmp2[ind0,(ind0//2)*2+1] = (1j*s12) * (1-2*(ind0%2))
    tmp3 = np.zeros((dim,dim), dtype=np.complex128) #B4
    tmp3[ind0,(ind0//2)*2+1] = s12
    tmp3[ind0,((ind0//2)*2+2)%dim] = (1j*s12) * (1-2*(ind0%2))
    tmp4 = np.concatenate([tmp0,tmp1,tmp2,tmp3], axis=0)
    ret = tmp4[:,:,np.newaxis] * tmp4[:,np.newaxis].conj()
    return ret


dim0 = 3
matrix_subspace = get_EP_POVM00(dim0)
basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matrix_subspace, field='real')
numqi.unique_determine.check_UDP_matrix_subspace(basis, num_repeat=80, early_stop_threshold=1e-4, converge_tol=1e-7, dtype='float64', num_worker=1)

model = numqi.matrix_space.DetectRankModel(basis, space_char='C_H', rank=(0,1,1), dtype='float64')
theta_optim200 = numqi.optimize.minimize(model, theta0='normal', num_repeat=80,
        tol=1e-7, early_stop_threshold=1e-4, print_every_round=0)
matH,coeff,residual = model.get_matrix(theta_optim200.x, basis_orth)
# |11> - |22>

dim0 = 6
matrix_subspace = get_EP_POVM01(dim0)
basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matrix_subspace, field='real')
numqi.unique_determine.check_UDP_matrix_subspace(basis, num_repeat=80, early_stop_threshold=1e-4, converge_tol=1e-7, dtype='float64', num_worker=1)

model = numqi.matrix_space.DetectRankModel(basis, space_char='C_H', rank=(0,1,1), dtype='float64')
theta_optim200 = numqi.optimize.minimize(model, theta0='normal', num_repeat=80,
        tol=1e-14, early_stop_threshold=1e-4, print_every_round=0)
matH,coeff,residual = model.get_matrix(theta_optim200.x, basis_orth)
# matH = np.kron(np.ones((2,2)), np.diag([1,0,-1]))
assert np.abs(np.trace(matrix_subspace @ matH, axis1=1, axis2=2)).max() < 1e-6
tmp0 = np.array([0,1,1,0,0,0])
tmp1 = np.array([0,0,0,0,1,1])
z0 = tmp0[:,np.newaxis] * tmp0 - tmp1[:,np.newaxis] * tmp1
np.abs(np.trace(z0 @ matrix_subspace, axis1=1, axis2=2)).max()

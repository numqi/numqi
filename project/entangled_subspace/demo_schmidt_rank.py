import numpy as np
import time
import sympy

import numqi

np_rng = np.random.default_rng()

rho_bes = numqi.entangle.load_upb('tiles', return_bes=True)[1]
EVL,EVC = np.linalg.eigh(rho_bes)
mask = EVL>1e-4
EVL = EVL[mask]
EVC = EVC[:,mask]
assert np.abs((EVC*EVL) @ EVC.T.conj() - rho_bes).max() < 1e-7
dimA = 3
dimB = 3
matrix_subspace = EVC.reshape(dimA,dimB,EVC.shape[1]).transpose(2,0,1)
#Example4 in http://arxiv.org/abs/2210.16389v1
assert not numqi.matrix_space.has_rank_hierarchical_method(matrix_subspace, rank=2, hierarchy_k=1)
assert numqi.matrix_space.has_rank_hierarchical_method(matrix_subspace, rank=2, hierarchy_k=2)


#Example5 in http://arxiv.org/abs/2210.16389v1
dimA = 4
dimB = 4
tmp0 = [
    [(0,0,1), (1,1,1), (2,2,1), (3,3,1)],
    [(0,1,1), (1,2,1), (2,3,1), (3,0,1)],
    [(0,2,1), (1,3,1), (2,0,1), (3,1,-1)],
]
matrix_subspace = np.stack([numqi.matrix_space.build_matrix_with_index_value(dimA, dimB, x) for x in tmp0])
tmp0 = matrix_subspace.reshape(-1,dimA*dimB)
rho = np.einsum(tmp0, [0,1], tmp0.conj(), [0,2], [1,2], optimize=True) / 12
assert np.linalg.eigvalsh(rho)[0] > -1e-7
assert abs(np.trace(rho)-1) < 1e-7
assert not numqi.entangle.is_ppt(rho, [dimA, dimB]) #entangled
assert numqi.matrix_space.has_rank_hierarchical_method(matrix_subspace, rank=3, hierarchy_k=1) #SN(rho)>=3

sympy.sieve.extend(50)
tmp0 = [(x+1)//2 for x in sympy.sieve._list[1:] if x%4==1]
# [3, 7, 9, 15, 19, 21]
dim = tmp0[np_rng.integers(len(tmp0))]
dim = 3
# dim=3: loss(1)=0.038 loss(2)=1.6e-15
# dim=7: loss(1)=3.6e-4 loss(2)=8.0e-14
# dim=9: loss(1)=1.7e-5 loss(2)=2.2e-13
# dim=15: loss(1)=2.7e-9 loss(2)=1.3e-13
# dim=19: loss(1)=1.1e-8 loss(2)=5.7e-13
rho = numqi.entangle.load_upb('quadres', dim, return_bes=True)[1]
assert numqi.entangle.is_ppt(rho, [dim,dim])
EVL,EVC = np.linalg.eigh(rho)
mask = EVL>1e-4
EVL = EVL[mask]
EVC = EVC[:,mask]
assert np.abs((EVC*EVL) @ EVC.T.conj() - rho).max() < 1e-7
matrix_subspace = EVC.reshape(dim,dim,EVC.shape[1]).transpose(2,0,1)
# numqi.matrix_space.has_rank_hierarchical_method(matrix_subspace, rank=2, hierarchy_k=1) #False
# numqi.matrix_space.has_rank_hierarchical_method(matrix_subspace, rank=2, hierarchy_k=2) #False

basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matrix_subspace, field='complex')
model = numqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=1)
theta_optim1 = numqi.optimize.minimize(model, 'normal', num_repeat=10, tol=1e-13)
model = numqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=2)
theta_optim2 = numqi.optimize.minimize(model, 'normal', num_repeat=3, tol=1e-13)
print(theta_optim1.fun, theta_optim2.fun)


# TODO how to re-construct the SN(2) pure state

ret_k1 = numqi.matrix_space.has_rank_hierarchical_method(matrix_subspace, rank=2, hierarchy_k=1)
ret_k2 = numqi.matrix_space.has_rank_hierarchical_method(matrix_subspace, rank=2, hierarchy_k=2)

basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(matrix_subspace, field='complex')
model = numqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=1)
theta_optim1 = numqi.optimize.minimize(model, 'normal', num_repeat=10, tol=1e-13)
model = numqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=2)
theta_optim2 = numqi.optimize.minimize(model, 'normal', num_repeat=3, tol=1e-13)
print(f'{dim=}, {ret_k1=}, {ret_k2=}, loss1={theta_optim1.fun}, loss2={theta_optim2.fun}')


dim = 10
upb,rho = numqi.entangle.load_upb('GenTiles1', dim, return_bes=True)
matrix_subspace = np.einsum(upb[0], [0,1], upb[1], [0,2], [0,1,2], optimize=True).reshape(upb[0].shape[0], -1)
EVL,EVC = np.linalg.eigh(rho)
mask = EVL>1e-4
EVL = EVL[mask]
EVC = EVC[:,mask]
assert np.abs(EVC.T.conj() @ matrix_subspace.T).max() < 1e-10

dim_list = [4,8,16,32,64,128]
time_list = []
loss_list = []
for dim in dim_list:
    upb = numqi.entangle.load_upb('GenTiles1', dim)
    basis_orth = np.einsum(upb[0], [0,1], upb[1], [0,2], [0,1,2], optimize=True)

    time0 = time.time()
    model = numqi.matrix_space.DetectRankModel(basis_orth, space_char='C', rank=1)
    theta_optim1 = numqi.optimize.minimize(model, 'normal', num_repeat=10, tol=1e-13)
    del model
    loss1 = float(theta_optim1.fun)
    del theta_optim1
    model = numqi.matrix_space.DetectRankModel(basis_orth, space_char='C', rank=2)
    theta_optim2 = numqi.optimize.minimize(model, 'normal', num_repeat=3, tol=1e-13)
    loss2 = float(theta_optim2.fun)
    del model
    del theta_optim2
    time_list.append(time.time()-time0)
    loss_list.append((loss1, loss2))
    print(f'[{dim=}][{time_list[-1]:.3f}s] loss1={loss1}, loss2={loss2}')

# dim_list=[4, 8, 16, 32, 64, 128]
# time_list=[0.9039320945739746,
#   2.1091482639312744,
#   4.291492700576782,
#   16.946312189102173,
#   299.2143979072571,
#   6959.400391340256],
# loss_list=[(0.02972175201999017, 3.3870567765274723e-13),
#   (0.016270529296662322, 5.619214496457762e-13),
#   (0.007914641763062885, 2.601934748752519e-12),
#   (0.0037891011526235453, 5.458186615740737e-12),
#   (0.0018140563424678888, 7.454183217036884e-12),
#   (0.0008705427783195833, 1.7801017907290265e-11)]



# import numpy as np
# import torch
# torch.set_num_threads(1)
# import matplotlib.pyplot as plt
# plt.ion()
# import numqi

# dm_tiles = numqi.entangle.load_upb('tiles', return_bes=True)[1]
# dm_pyramid = numqi.entangle.load_upb('pyramid', return_bes=True)[1]

# hf0 = lambda x: x*dm_pyramid+(1-x)*dm_tiles
# dm0 = hf0(0.01)

# dm0 = numqi.entangle.hf_interpolate_dm(dm_tiles, alpha=0.98)
# model = numqi.entangle.PureBosonicExt(3, 3, 32, distance_kind='ree')
# model.set_dm_target(dm0)
# loss = numqi.optimize.minimize(model, num_repeat=3)

# dimA = 3
# dimB = 3

# EVL,EVC = np.linalg.eigh(dm0)
# mask = EVL>1e-7
# EVL = EVL[mask]
# EVC = EVC[:,mask]
# matrix_subspace = EVC.reshape(dimA,dimB,EVC.shape[1]).transpose(2,0,1)
# numqi.matrix_space.has_rank_hierarchical_method(matrix_subspace, rank=2, hierarchy_k=3) #SN(rho)>=3

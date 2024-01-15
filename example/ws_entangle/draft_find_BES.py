import torch
import pickle
import numpy as np
from tqdm import tqdm

import numqi

np_rng = np.random.default_rng()

torch.set_num_threads(1)

def quick_beta_boundary(dm0, dimA, dimB, kext=16):
    beta_l,beta_u = numqi.entangle.get_density_matrix_boundary(dm0)
    dm_u = numqi.entangle.hf_interpolate_dm(dm0, beta=beta_u)
    dm_l = numqi.entangle.hf_interpolate_dm(dm0, beta=beta_l)
    beta_ppt_l,beta_ppt_u = numqi.entangle.get_ppt_boundary(dm0, (dimA, dimB))
    model_svqc = numqi.pureb.PureBosonicExt(dimA, dimB, kext=kext)
    beta_svqc_u = model_svqc.get_boundary(dm_u, xtol=1e-4, threshold=1e-7, num_repeat=1, use_tqdm=True)
    beta_svqc_l = model_svqc.get_boundary(dm_l, xtol=1e-4, threshold=1e-7, num_repeat=1, use_tqdm=True)
    ret_u = [beta_u,beta_ppt_u,beta_svqc_u, f'PPT-SVQC={beta_ppt_u-beta_svqc_u:.4f},dm-PPT={beta_u-beta_ppt_u:.4f}']
    ret_l = [-beta_l,beta_ppt_l,beta_svqc_l, f'PPT-SVQC={beta_ppt_l-beta_svqc_l:.4f},dm-PPT={-beta_l-beta_ppt_l:.4f}']
    return ret_u,ret_l

def quick_plot_dm0_dm1_plane(rho0, rho1, dim, num_point, pureb_kext, tag_cha, tag_gppt=False,
                            savepath='tbd00.png', num_eig0=0, label0=None, label1=None):
    dimA,dimB = dim
    theta1,hf_theta = numqi.entangle.get_density_matrix_plane(rho0, rho1)
    theta_list = np.linspace(-np.pi, np.pi, num_point)
    pureb_kext = [int(x) for x in pureb_kext] if hasattr(pureb_kext, '__len__') else [int(pureb_kext)]

    beta_pureb = dict()
    kwargs = dict(xtol=1e-4, converge_tol=1e-10, threshold=1e-7, num_repeat=3, use_tqdm=False)
    for key in pureb_kext:
        model = numqi.entangle.PureBosonicExt(dimA, dimB, kext=key)
        beta_pureb[key] = [model.get_boundary(hf_theta(x), **kwargs) for x in tqdm(theta_list, desc=f'PureB({key})')]

    if tag_cha:
        model_cha = numqi.entangle.CHABoundaryBagging(dim)
        beta_cha = np.array([model_cha.solve(hf_theta(x),use_tqdm=False) for x in tqdm(theta_list, desc='CHA')])
    else:
        beta_cha = None
    fig,ax,all_data = numqi.entangle.plot_bloch_vector_cross_section(dm_tiles, dm_pyramid, (3,3), num_point,
            beta_pureb, beta_cha, num_eig0=num_eig0, tag_gppt=tag_gppt, label0=label0, label1=label1, savepath=savepath)
    return fig,ax

dimA = 3
dimB = 3

dm_tiles = numqi.entangle.load_upb('tiles', return_bes=True)[1]
dm_pyramid = numqi.entangle.load_upb('pyramid', return_bes=True)[1]
dm_quadres = numqi.entangle.load_upb('quadres', 3, return_bes=True)[1]
dm_sixparam = numqi.entangle.load_upb('sixparam', return_bes=True)[1]
dm_sixparam1 = numqi.entangle.load_upb('sixparam', return_bes=True)[1]
dm_genshift3 = numqi.entangle.load_upb('genshifts', 3, return_bes=True)[1]

fig,ax,all_data = quick_plot_dm0_dm1_plane(dm_tiles, dm_pyramid, (3,3), num_point=201,
            pureb_kext=16, tag_cha=False, label0=r'$\rho_{tiles}$', label1=r'$\rho_{pyramid}$')

# fig,ax,all_data = quick_plot_dm0_dm1_plane(dm_tiles, dm_pyramid, (3,3), num_point=201,
#             pureb_kext=[8,32], tag_cha=True, label0=r'Tiles UPB', label1=r'Pyramid UPB')
# fig.savefig('data/20220726_3x3_upb_cha_ppt_gellman_distance_polar.png', dpi=200)
# with open('data/20220726_3x3_upb_cha_ppt_gellman_distance_polar.pkl', 'wb') as fid:
#     pickle.dump(all_data, fid)

# model_eig.rho_vec.data[:] = torch.tensor(numqi.gellmann.dm_to_gellmann_basis(dm_tiles))
# tmp0 = torch.tensor(numqi.gellmann.dm_to_gellmann_basis(dm_tiles))
# model_eig.rho_vec.data[:] = tmp0 + np.random.uniform(-1,1,size=tmp0.shape)*1e-5
# model_eig = numqi.entangle.BESNumEigenModel(dimA, dimB, N0=5, N1=4, with_ppt=True)
# hf_model = numqi.optimize.hf_model_wrapper(model_eig)
# theta0 = numqi.optimize.get_model_flat_parameter(model_eig)
# theta_optim = scipy.optimize.minimize(hf_model, theta0, method='L-BFGS-B', jac=True, tol=1e-20)
# print(theta_optim.fun)

dimA = 3
dimB = 3
rank0 = 4
model_eig = numqi.entangle.BESNumEigenModel(dimA, dimB, rank0, with_ppt=True)


optimizer = torch.optim.Adam(model_eig.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
best_loss = 1
best_theta = None
with tqdm(range(30000)) as pbar:
    for ind0 in pbar:
        optimizer.zero_grad()
        loss = model_eig()
        if loss.item()<best_loss:
            best_loss = loss.item()
            best_theta = model_eig.rho_vec.detach().numpy().copy()
        loss.backward()
        optimizer.step()
        if ind0%300==0:
            pbar.set_postfix(loss=f'{loss.item():.6g}')
            lr_scheduler.step()
print(best_loss)
# model_eig.rho_vec.data[:] = torch.tensor(best_theta)
print(quick_beta_boundary(best_theta, dimA, dimB, kext=16))

dm0 = numqi.gellmann.gellmann_basis_to_dm(best_theta)
dm1 = numqi.gellmann.gellmann_basis_to_dm(best_theta)
fig,ax,all_data = quick_plot_dm0_dm1_plane(dm0, dm1, (dimA,dimB),
            num_point=201, pureb_kext=None, tag_cha=False, num_eig0=5)
# fig.savefig('data/20220806_4x3_plane01.png', dpi=200)
# with open('data/20220806_4x3_plane01.pkl', 'wb') as fid:
#     pickle.dump(all_data, fid)


model_eig = numqi.entangle.BESNumEigen3qubitModel(4)
optimizer = torch.optim.Adam(model_eig.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
best_loss = 1
best_theta = None
with tqdm(range(30000)) as pbar:
    for ind0 in pbar:
        optimizer.zero_grad()
        loss = model_eig()
        if loss.item()<best_loss:
            best_loss = loss.item()
            best_theta = model_eig.rho_vec.detach().numpy().copy()
        loss.backward()
        optimizer.step()
        if ind0%300==0:
            pbar.set_postfix(loss=f'{loss.item():.6g}')
            lr_scheduler.step()
print(best_loss)
# model_eig.rho_vec.data[:] = torch.tensor(best_theta)
print(quick_beta_boundary(best_theta, 4, 2, kext=16))

dm0 = numqi.gellmann.gellmann_basis_to_dm(best_theta)
fig,ax,all_data = quick_plot_dm0_dm1_plane(dm0, dm1, (4,2),
            num_point=201, pureb_kext=None, tag_cha=True, num_eig0=4)
# with open('data/20220806_2x2x2_plane.pkl', 'wb') as fid:
#     pickle.dump(all_data, fid)
# fig.savefig('data/20220806_2x2x2_plane.png', dpi=200)

# for x in [dm_tiles,dm_pyramid,dm_quadres]:
#     tmp0 = numqi.gellmann.dm_to_gellmann_basis(x)
#     print(np.dot(rho_vec_norm, tmp0 / np.linalg.norm(tmp0)))

# model = numqi.pureb.PureBosonicExt(dimA, dimB, kext=32)
# beta_pureb = model.get_boundary(dm_target, xtol=1e-4, threshold=1e-7, num_repeat=1)

fig,ax,all_data = quick_plot_dm0_dm1_plane(dm_tiles, dm1, (dimA,dimB), num_point=201, pureb_kext=None, tag_cha=False)
# fig.savefig('data/20220806_4x3_plane00.png', dpi=200)
# with open('data/20220806_4x3_plane00.pkl', 'wb') as fid:
#     pickle.dump(all_data, fid)


# import time
# dm_target = np.diag(np.array([1,1,1,1,0,0,0,0,0])/4)
# model = numqi.entangle.AutodiffCHAREE(dim0=3, dim1=3, num_state=18, distance_kind='gellmann')
# model.set_dm_target(dm_target)
# t0 = time.time()
# loss = numqi.optimize.minimize(model, theta0='uniform', tol=1e-12, num_repeat=3, print_every_round=0).fun
# print(time.time()-t0, loss)


dimA = 3
dimB = 3
dm0 = numqi.random.rand_density_matrix(9)
dm1 = numqi.random.rand_density_matrix(9)

dm_tiles = numqi.entangle.load_upb('tiles', return_bes=True)[1]
dm_pyramid = numqi.entangle.load_upb('pyramid', return_bes=True)[1]
dm_quadres = numqi.entangle.load_upb('quadres', 3, return_bes=True)[1]
dm_sixparam = numqi.entangle.load_upb('sixparam', return_bes=True)[1]
dm_sixparam1 = numqi.entangle.load_upb('sixparam', return_bes=True)[1]
dm_genshift3 = numqi.entangle.load_upb('genshifts', 3, return_bes=True)[1]


def demo_misc00():
    dm_tiles = numqi.entangle.load_upb('tiles', return_bes=True)[1]
    dm_pyramid = numqi.entangle.load_upb('pyramid', return_bes=True)[1]

    fig,ax,all_data = quick_plot_dm0_dm1_plane(dm_tiles, dm_pyramid, (3,3), num_point=201,
                pureb_kext=None, tag_cha=False, tag_gppt=True)
    # fig.savefig('data/20220905_realign_upb_bes.png', dpi=200)


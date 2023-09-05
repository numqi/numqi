import numpy as np
import matplotlib.pyplot as plt

import numqi

np_rng = np.random.default_rng()
tableau = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
hf_trace0 = lambda x: x-np.trace(x)/x.shape[0]*np.eye(x.shape[0])


dim = 3
dimA = dim
dimB = dim
kext = 3
use_ppt = True
use_boson = True
dm0 = numqi.entangle.get_werner_state(dim, 1)
# dm0 = numqi.entangle.load_upb('pyramid', return_bes=True)[1]
# dm1 = numqi.entangle.get_isotropic_state(dim, 1)
dm1 = numqi.entangle.load_upb('tiles', return_bes=True)[1]
# dm0 = numqi.random.rand_density_matrix(dim*dim)
# dm1 = numqi.random.rand_density_matrix(dim*dim)
theta_dm1,hf_theta = numqi.entangle.get_density_matrix_plane(dm0, dm1)
theta_list = np.linspace(0, 2*np.pi, 201)

kwargs = dict(dim=(dim,dim), kext=kext, use_ppt=use_ppt, use_boson=use_boson, return_info=True, use_tqdm=True)
tmp0 = np.stack([hf_theta(x) for x in theta_list])
tmp1 = numqi.entangle.get_ABk_symmetric_extension_boundary(tmp0, **kwargs)
beta_kext_list = np.array([x[0] for x in tmp1])
vecA_list = np.stack([x[1]['vecA'] for x in tmp1], axis=0)
vecN_list = np.stack([x[1]['vecN'] for x in tmp1], axis=0)

basis0 = numqi.gellmann.dm_to_gellmann_basis(dm0)
basis1 = numqi.gellmann.dm_to_gellmann_basis(dm1)
vec_proj_A,vec_proj_N = numqi.maximum_entropy.get_supporting_plane_2d_projection(vecA_list, vecN_list, basis0, basis1, theta_list)
beta_dm = np.array([numqi.entangle.get_density_matrix_boundary(hf_theta(x))[1] for x in theta_list])
fig,ax = plt.subplots()
tmp0 = slice(None,None,1)
# numqi.maximum_entropy.draw_line_list(ax, vec_proj_A[tmp0], vec_proj_N[tmp0], kind='tangent', color='#CCCCCC', radius=0.1, label='SDP dual')
numqi.maximum_entropy.draw_line_list(ax, vec_proj_A[tmp0], vec_proj_N[tmp0], kind='norm', color='#CCCCCC', radius=0.03, label='SDP dual')
tmp0 = beta_dm*np.cos(theta_list)
tmp1 = beta_dm*np.sin(theta_list)
ax.plot(tmp0, tmp1, linestyle='solid', color=tableau[4], label='dm boundary')
tmp0 = beta_kext_list*np.cos(theta_list)
tmp1 = beta_kext_list*np.sin(theta_list)
ax.plot(tmp0, tmp1, linestyle='dashed', color='r', label='k-ext boundary')
hf0 = lambda theta,r: (r*np.cos(theta), r*np.sin(theta))
for theta, label in [(0,'Werner'),(theta_dm1,'tiles UPB/BES')]:
    radius = 0.4
    ax.plot([0, radius*np.cos(theta)], [0, radius*np.sin(theta)], linestyle=':', label=label)
ax.legend()
tmp0 = (', PPT' if use_ppt else '') + (', Boson' if use_boson else '')
ax.set_title(rf'$d_A={dimA}, d_B={dimB}, k={kext}$'+tmp0)
fig.tight_layout()
fig.savefig('tbd01.png', dpi=200)
# fig.savefig('data/3x3_k3_werner_tiles.png', dpi=200)
# fig.savefig('data/3x3_k5_ppt_pyramid_tiles.png', dpi=200)

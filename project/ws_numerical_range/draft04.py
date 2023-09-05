import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm

import pyqet

np_rng = np.random.default_rng()
tableau = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
hf_trace0 = lambda x: x-np.trace(x)/x.shape[0]*np.eye(x.shape[0])


dim = 2
kext = 3

kind = 'boson' if dim==2 else 'symmetric'
matGext = pyqet.maximum_entropy.get_ABk_gellmann_preimage_op(dim, dim, kext, kind=kind)

alpha_boundary = (kext+dim*dim-dim)/(kext*dim+dim-1)
tstar = pyqet.gellmann.dm_to_gellmann_norm(pyqet.entangle.get_werner_state(dim, alpha_boundary))

model = pyqet.maximum_entropy.MaximumEntropyTangentModel(matGext, factor=0.5)
vecB = pyqet.gellmann.dm_to_gellmann_basis(pyqet.entangle.get_werner_state(dim, 1))
model.set_target_vec(vecB)

# theta_optim = pyqet.optimize.minimize(model, theta0='uniform', num_repeat=3, tol=1e-10,
#                 print_every_round=1, print_freq=10)
# print(abs(tstar-theta_optim.fun)) #1e-2

loss = pyqet.optimize.minimize_adam(model, num_step=2000, theta0='uniform',
                    optim_args=('adam',0.03,0.003), tqdm_update_freq=50)
abs(tstar-loss) #1e-3

vecA,vecN = model.get_vector()


dim = 3
dimA = dim
dimB = dim
kext = 3
dm0 = pyqet.entangle.get_werner_state(dim, 1)
dm1 = pyqet.entangle.get_isotropic_state(dim, 1)
dm1 = pyqet.entangle.load_upb('tiles', return_bes=True)[1]
theta_dm1,hf_theta = pyqet.entangle.get_density_matrix_plane(dm0, dm1)
theta_list = np.linspace(0, 2*np.pi, 201)

beta_dm = np.array([pyqet.entangle.get_density_matrix_boundary(hf_theta(x))[1] for x in theta_list])

kwargs = dict(dim=(dim,dim), kext=kext, xtol=1e-5, use_ppt=False, use_boson=True, return_info=False, use_tqdm=False)
tmp0 = np.stack([hf_theta(x) for x in theta_list])
beta_kext_list = pyqet.entangle.get_ABk_symmetric_extension_boundary(tmp0, **kwargs)

matGext = pyqet.maximum_entropy.get_ABk_gellmann_preimage_op(dimA, dimB, kext, kind='boson')
model = pyqet.maximum_entropy.MaximumEntropyTangentModel(matGext, factor=0.5)
vecAN_list = []
for theta_i in tqdm(theta_list):
    model.set_target_vec(pyqet.gellmann.dm_to_gellmann_basis(hf_theta(theta_i)))
    # theta_optim = pyqet.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0)
    pyqet.optimize.minimize_adam(model, num_step=5000, theta0='uniform', optim_args=('adam',0.03,1e-4), tqdm_update_freq=0)
    vecAN_list.append(model.get_vector())
vecA_list = np.stack([x[0] for x in vecAN_list])
vecN_list = np.stack([x[1] for x in vecAN_list])

tmp0 = pyqet.gellmann.dm_to_gellmann_basis(dm0)
basis0 = tmp0/np.linalg.norm(tmp0)
tmp0 = pyqet.gellmann.dm_to_gellmann_basis(dm1)
tmp0 = tmp0 - np.dot(tmp0, basis0)*basis0
basis1 = tmp0/np.linalg.norm(tmp0)
tmp0 = vecN_list @ basis0
tmp1 = vecN_list @ basis1
vec_proj_N = np.angle(tmp0 + 1j*tmp1)
tmp2 = np.einsum(vecA_list,[0,1],vecN_list,[0,1],[0],optimize=True)
tmp3 = tmp2 / (np.cos(theta_list)*tmp0 + np.sin(theta_list)*tmp1)
vec_proj_A = np.stack([tmp3*np.cos(theta_list), tmp3*np.sin(theta_list)], axis=1)


fig,ax = plt.subplots()
tmp0 = slice(None,None,1)
# pyqet.maximum_entropy.draw_line_list(ax, vec_proj_A[tmp0], vec_proj_N[tmp0], kind='tangent', color='#CCCCCC', radius=0.15, label='maxent tangent')
pyqet.maximum_entropy.draw_line_list(ax, vec_proj_A[tmp0], vec_proj_N[tmp0], kind='norm', color='#CCCCCC', radius=0.03, label='maxent tangent')
tmp0 = beta_dm*np.cos(theta_list)
tmp1 = beta_dm*np.sin(theta_list)
ax.plot(tmp0, tmp1, linestyle='solid', color=tableau[4], label='dm boundary')
tmp0 = beta_kext_list*np.cos(theta_list)
tmp1 = beta_kext_list*np.sin(theta_list)
# color: '#CCCCCC' tableau[2]
ax.plot(tmp0, tmp1, linestyle='dashed', color='r', label='k-boson-ext boundary')
hf0 = lambda theta,r: (r*np.cos(theta), r*np.sin(theta))
for theta, label in [(0,'werner'),(theta_dm1,'tiles UPB/BES')]:
    radius = 0.3
    ax.plot([0, radius*np.cos(theta)], [0, radius*np.sin(theta)], linestyle=':', label=label)
ax.legend()
# ax.set_xlim(-0.26,0.38)
# ax.set_ylim(-0.26, 0.65)
ax.set_title(rf'$d_A={dimA}, d_B={dimB}, k={kext}$')
# ax.axis('off')
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)
# fig.savefig('3x3_k4_werner_isotropic.png', dpi=200)



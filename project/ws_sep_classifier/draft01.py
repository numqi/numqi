import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import numqi

from utils import DummyFCModel, plot_bloch_vector_plane

def generate_dm_boundary_data(batch_size, ratio_list, dim, seed=None):
    # warning, the generated data here are always paired (x,-x)
    tmp0 = np.array(ratio_list)
    ratio_list = tmp0 / np.sum(tmp0)
    num_list = np.around(ratio_list*(batch_size//2+1)).astype(np.int64)

    np_rng = numqi.random.get_numpy_rng(seed)
    # random density matrix
    tmp0 = np_rng.normal(size=(num_list[0], dim*dim-1))
    tmp0 /= np.linalg.norm(tmp0, axis=1, keepdims=True)
    dm_list = [numqi.gellmann.gellmann_basis_to_dm(tmp0)]
    # random pure state and its neighbors
    tmp0 = np_rng.normal(size=(num_list[1], 2*dim)).astype(np.float32, copy=False).view(np.complex64)
    tmp0 /= np.linalg.norm(tmp0, axis=1, keepdims=True)
    tmp1 = numqi.gellmann.dm_to_gellmann_basis(np.einsum(tmp0, [0,1], tmp0.conj(), [0,2], [0,1,2], optimize=True))
    tmp1 += np_rng.normal(scale=0.01, size=tmp1.shape)
    tmp1 /= np.linalg.norm(tmp1, axis=1, keepdims=True)
    dm_list.append(numqi.gellmann.gellmann_basis_to_dm(tmp1))
    dm_list = np.concatenate(dm_list, axis=0)
    vec = numqi.gellmann.dm_to_gellmann_basis(dm_list)
    beta_dm_l, beta_dm_u = numqi.entangle.get_density_matrix_boundary(dm_list)
    xdata = np.concatenate([-vec,vec], axis=0)[:batch_size]
    ydata = np.concatenate([-beta_dm_l, beta_dm_u], axis=0)[:batch_size]
    return xdata, ydata


def generate_ppt_boundary_data(batch_size, dim, seed=None):
    # warning, the generated data here are always paired (x,-x)
    assert hasattr(dim, '__len__') and (len(dim)==2)
    dimA,dimB = dim
    dim = dimA*dimB
    np_rng = numqi.random.get_numpy_rng(seed)
    vec = np_rng.normal(size=(batch_size//2+1, dim*dim-1))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    dm_list = numqi.gellmann.gellmann_basis_to_dm(vec)
    beta_ppt_l,beta_ppt_u = numqi.entangle.get_ppt_boundary(dm_list, (dimA,dimB))
    xdata = np.concatenate([-vec,vec], axis=0)[:batch_size]
    ydata = np.concatenate([-beta_ppt_l, beta_ppt_u], axis=0)[:batch_size]
    return xdata, ydata



def get_lr_scheduler(optimizer, lr_start_end, total_step, every_step=1):
    gamma = (lr_start_end[1]/lr_start_end[0])**(1/(total_step//every_step))
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    return lr_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dim = 8
batch_size = 4096
lr_start_end = 0.01, 1e-5
num_step = 30
step_lr_scheduler = 10

ratio_list = 7, 2

model = DummyFCModel(dim, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_start_end[0])
lr_scheduler = get_lr_scheduler(optimizer, lr_start_end, num_step, step_lr_scheduler)
loss_history = []
model.train()
with tqdm(range(num_step)) as pbar:
    for ind_step in pbar:
        xdata_i, ydata_i = generate_dm_boundary_data(batch_size, ratio_list, dim)
        xdata_i = torch.tensor(xdata_i, dtype=torch.float32, device=device)
        ydata_i = torch.tensor(ydata_i, dtype=torch.float32, device=device)
        xdata_i,ydata_i = xdata_i.to(device), ydata_i.to(device)
        optimizer.zero_grad()
        loss = torch.mean((model(xdata_i) - ydata_i)**2)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if ind_step%10==0:
            pbar.set_postfix(train_loss='{:.5}'.format(sum(loss_history[-10:])/10))
        if (ind_step>0) and (ind_step%step_lr_scheduler==0):
            lr_scheduler.step()


fig,ax = plt.subplots()
ax.plot(np.arange(len(loss_history)), loss_history, '.', markersize=1)
ax.set_xlabel('step')
ax.set_ylabel('loss')
ax.set_yscale('log')
ax.grid()
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)


op0 = numqi.random.rand_density_matrix(dim)
op1 = numqi.random.rand_density_matrix(dim)
plot_bloch_vector_plane(model, op0, op1, dim, filename='tbd01.png')
if dim==(3,3):
    op0 = numqi.state.Werner(d=3, alpha=1)
    op1 = numqi.entangle.load_upb('tiles', return_bes=True)[1]
    plot_bloch_vector_plane(model, op0, op1, dim, filename='tbd01.png')

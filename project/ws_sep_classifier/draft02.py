import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import scipy.optimize

import numqi

# from utils import plot_bloch_vector_plane

class PositiveWeightLinear(torch.nn.Module):
    def __init__(self, in_features:int, out_features:int, device=None, dtype=torch.float32):
        super().__init__()
        assert dtype in [torch.float32, torch.float64]
        self.theta0 = torch.nn.Parameter(torch.rand(out_features, in_features, dtype=dtype, device=device)-0.5)
        self.theta1 = torch.nn.Parameter(torch.rand(out_features, dtype=dtype, device=device)-np.log10(in_features)-0.5)
        # self.theta = torch.nn.Parameter(torch.rand(out_features, in_features, dtype=dtype, device=device)-np.log10(in_features)-0.5)
        self.bias = torch.nn.Parameter((torch.rand(out_features, dtype=dtype, device=device)-0.5)/np.sqrt(in_features))

    def forward(self, x):
        tmp0 = torch.nn.functional.softmax(self.theta0, dim=1)
        tmp1 = torch.nn.functional.softplus(self.theta1)
        ret = torch.nn.functional.linear(x, tmp0) * tmp1 + self.bias
        # tmp0 = torch.nn.functional.softplus(self.theta)
        # ret = torch.nn.functional.linear(x, tmp0, self.bias)
        return ret

def generate_ppt_boundary_data(batch_size, dim, tag_symmetry=True, seed=None):
    # warning, the generated data here are always paired (x,-x)
    assert hasattr(dim, '__len__') and (len(dim)==2)
    dimA,dimB = dim
    dim = dimA*dimB
    np_rng = numqi.random.get_numpy_rng(seed)
    if tag_symmetry:
        vec = np_rng.normal(size=(batch_size//2+1, dim*dim-1))
    else:
        vec = np_rng.normal(size=(batch_size, dim*dim-1))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    dm_list = numqi.gellmann.gellmann_basis_to_dm(vec)
    beta_ppt_l,beta_ppt_u = numqi.entangle.get_ppt_boundary(dm_list, (dimA,dimB))
    if tag_symmetry:
        xdata = np.concatenate([-vec,vec], axis=0)[:batch_size]
        ydata = np.concatenate([-beta_ppt_l, beta_ppt_u], axis=0)[:batch_size]
    else:
        xdata = vec
        ydata = beta_ppt_u
    ret = xdata*ydata[:,np.newaxis]
    return ret

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
    ret = xdata*ydata[:,np.newaxis]
    return ret


class DummyConvexModel(torch.nn.Module):
    def __init__(self, dim, dim_fc_list=None, device:(torch.device|None)=None):
        super().__init__()
        dtype = torch.float32
        if dim_fc_list is None:
            dim_fc_list = [dim**2-1, 128, 128, 128, 128, 1]
        else:
            assert dim_fc_list[0] == dim**2-1
        if device is None:
            device = torch.device('cpu')
        tmp0 = [torch.nn.Linear(dim_fc_list[0], dim_fc_list[1], device=device, dtype=dtype)]
        tmp0 += [PositiveWeightLinear(dim_fc_list[x], dim_fc_list[x+1], device=device, dtype=dtype) for x in range(1, len(dim_fc_list)-1)]
        self.fc_list = torch.nn.ModuleList(tmp0)
        # tmp0 = [torch.nn.Parameter(0.1*torch.randn(x, dtype=torch.float32, device=device)-1) for x in dim_fc_list[1:]]
        # self.theta_prelu = torch.nn.ParameterList(tmp0)
        # self.theta_prelu = torch.nn.Parameter(0.1*torch.randn(len(dim_fc_list)-1, dtype=torch.float32, device=device))

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1,shape[-1])
        for ind0 in range(len(self.fc_list)):
            y = self.fc_list[ind0](x)
            if ind0<(len(self.fc_list)-1):
                y = torch.nn.functional.leaky_relu(y)
                # y = torch.nn.functional.prelu(y, torch.nn.functional.softplus(self.theta_prelu[ind0]))
                # y = torch.nn.functional.selu(y)
            if x.shape[-1]==y.shape[-1]: #TODO resnet may not convex
                x = y + x
            else:
                x = y
        if len(shape)==1:
            x = x[0]
        else:
            x = x.reshape(shape[:-1])
        return x

def get_lr_scheduler(optimizer, lr_start_end, total_step, every_step=1):
    gamma = (lr_start_end[1]/lr_start_end[0])**(1/(total_step//every_step))
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    return lr_scheduler


np_rng = np.random.default_rng()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dimA = 2
dimB = 2
batch_size = 512
weight_rho0 = int(0.1*batch_size)
lr_start_end = 0.01, 0.001
num_step = 10000
step_lr_scheduler = 10

tmp0 = (weight_rho0/(weight_rho0+batch_size))
worst_loss = (1-tmp0)**2*tmp0 + (1-tmp0)
print('worst_loss:', worst_loss)

dim = dimA*dimB

model = DummyConvexModel(dim, device=device)
loss_history = []
optimizer = torch.optim.Adam(model.parameters(), lr=lr_start_end[0])
# optimizer = torch.optim.SGD(model.parameters(), lr=lr_start_end[0], momentum=0.9)
lr_scheduler = get_lr_scheduler(optimizer, lr_start_end, num_step, step_lr_scheduler)
model.train() #doesn't matter
torch_rho0 = torch.zeros((1, dim*dim-1), dtype=torch.float32, device=device)
ydata_i = torch.tensor([-1]+[0]*(batch_size), dtype=torch.float32, device=device)
ydata_weight = torch.tensor([weight_rho0]+[1]*(batch_size), dtype=torch.float32, device=device)/(weight_rho0+batch_size)
with tqdm(range(num_step)) as pbar:
    for ind_step in pbar:
        optimizer.zero_grad()
        # tmp0 = torch.tensor(generate_dm_boundary_data(batch_size, ratio_list=[2,7], dim=dimA*dimB), dtype=torch.float32, device=device)
        tmp0 = torch.tensor(generate_ppt_boundary_data(batch_size, (dimA,dimB)), dtype=torch.float32, device=device)
        xdata_i = torch.concat([torch_rho0, tmp0], axis=0)
        loss = torch.dot(ydata_weight, (model(xdata_i) - ydata_i)**2)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if ind_step%10==0:
            pbar.set_postfix(train_loss='{:.5}'.format(sum(loss_history[-10:])/10))
        if (ind_step>0) and (ind_step%step_lr_scheduler==0):
            lr_scheduler.step()


op0 = numqi.random.rand_density_matrix(dim)
op1 = numqi.random.rand_density_matrix(dim)
num_point = 201
_, hf_plane = numqi.entangle.get_density_matrix_plane(op0, op1)
theta_list = np.linspace(-np.pi, np.pi, num_point)
model.eval()
beta_model = []
r_outer = np.sqrt((dim-1)/(2*dim)+0.01)
with torch.no_grad():
    for theta_i in tqdm(theta_list):
        tmp0 = torch.tensor(numqi.gellmann.dm_to_gellmann_basis(hf_plane(theta_i)), dtype=torch.float32, device=device)
        hf0 = lambda x: (model(tmp0*x).item())
        if (hf0(r_outer)<0) or hf0(0)>0:
            beta_model.append(np.nan)
        else:
            beta_model.append(scipy.optimize.root_scalar(hf0, bracket=[0,r_outer]).root)
beta_model = np.array(beta_model)
print('#nan', np.isnan(beta_model).sum())
fig,ax,_ = numqi.entangle.plot_bloch_vector_cross_section(op0, op1, (dimA,dimB), beta_cha=beta_model, num_point=201)
ax.set_aspect('equal')
fig.savefig('tbd00.png')


np0 = np.linspace(-1, 1, 100)*np.sqrt((dim-1)/(2*dim))
with torch.no_grad():
    # theta_i = theta_list[0]
    theta_i = np.pi*2/3
    tmp0 = torch.tensor(numqi.gellmann.dm_to_gellmann_basis(hf_plane(theta_i)), dtype=torch.float32, device=device)
    hf0 = lambda x: model(tmp0*x)
    z0 = np.array([hf0(x).item() for x in np0])
fig,(ax0,ax1) = plt.subplots(1, 2, figsize=(8,4))
ax0.plot(np0, z0)
ax1.plot(loss_history)
ax0.set_xlabel('beta')
ax0.set_ylabel('model(rho)')
ax1.set_yscale('log')
ax1.set_xlabel('step')
ax1.set_ylabel('loss')
fig.tight_layout()
fig.savefig('tbd01.png')

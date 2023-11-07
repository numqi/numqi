import pickle
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import numqi

def is_positive_semi_definite(np0):
    # https://math.stackexchange.com/a/13311
    # https://math.stackexchange.com/a/87538
    # Sylvester's criterion
    try:
        np.linalg.cholesky(np0)
        ret = True
    except np.linalg.LinAlgError:
        ret = False
    return ret

def rand_n_sphere(dim, size=None, seed=None):
    np_rng = numqi.random.get_numpy_rng(seed)
    is_single = (size is None)
    if is_single:
        size = ()
    elif not hasattr(size, '__len__'):
        size = int(size),
    N0 = 1 if (len(size)==0) else np.prod(size)
    tmp0 = np_rng.normal(size=(N0,dim))
    tmp0 = tmp0 / np.linalg.norm(tmp0, axis=-1, keepdims=True)
    if is_single:
        ret = tmp0[0]
    else:
        ret = tmp0.reshape(size+(dim,))
    return ret

def save_data(xdata, ydata, filepath='data_boundary.pkl'):
    with open(filepath, 'wb') as fid:
        pickle.dump(dict(xdata=xdata, ydata=ydata), fid)

def load_data(filepath='data_boundary.pkl'):
    with open(filepath, 'rb') as fid:
        tmp0 = pickle.load(fid)
        ret = tmp0['xdata'], tmp0['ydata']
    return ret

class DummyFCModel(torch.nn.Module):
    def __init__(self, scale_factor=10):
        super().__init__()
        tmp0 = [80, 128, 512, 512, 512, 128, 128, 128, 2]
        self.fc_list = torch.nn.ModuleList([torch.nn.Linear(tmp0[x], tmp0[x+1]) for x in range(len(tmp0)-1)])
        self.bn_list = torch.nn.ModuleList([torch.nn.BatchNorm1d(tmp0[x+1]) for x in range(len(tmp0)-1)])
        self.scale_factor = scale_factor

    def forward(self, x):
        x = x/torch.linalg.norm(x)
        is_single = (x.ndim==1)
        if is_single:
            x = x.reshape(1,-1)
        for ind0 in range(len(self.fc_list)):
            y = self.fc_list[ind0](x)
            y = self.bn_list[ind0](y)
            if ind0<(len(self.fc_list)-1):
                y = torch.nn.functional.leaky_relu(y)
            if x.shape[-1]==y.shape[-1]:
                x = y + x
            else:
                x = y
        x = torch.nn.functional.softplus(x)
        if is_single:
            x = x[0]
        return x

np_rng = np.random.default_rng()
dimA = 3
dimB = 3
num_sample = 1000000
batch_size = 256
SCALE_FACTOR = 10

# vec_list = rand_n_sphere(dimA*dimB*dimA*dimB-1, size=num_sample//2, seed=np_rng)
# xdata = []
# ydata = []
# for vec in tqdm(vec_list):
#     rho = numqi.gellmann.gellmann_basis_to_dm(vec)
#     beta_dm_l,beta_dm_u = numqi.entangle.get_density_matrix_boundary(rho)
#     beta_ppt_l,beta_ppt_u = numqi.entangle.get_ppt_boundary(rho, (dimA,dimB))
#     xdata.append(vec)
#     ydata.append((beta_dm_u,beta_ppt_u))
#     xdata.append(-vec)
#     ydata.append((-beta_dm_l,-beta_ppt_l))
# xdata = np.stack(xdata)
# ydata = np.array(ydata)
# save_data(xdata, ydata)
xdata,ydata = load_data()

tmp0 = torch.tensor(xdata, dtype=torch.float32)
tmp1 = torch.tensor(ydata, dtype=torch.float32)
trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tmp0, tmp1), batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tmp0, tmp1), batch_size=batch_size)

model = DummyFCModel()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
loss_history = []
for epoch in range(6):
    model.train()
    with tqdm(trainloader) as pbar:
        for data_i,label_i in pbar:
            optimizer.zero_grad()
            loss = torch.mean((model(data_i)-label_i*SCALE_FACTOR)[:,1]**2)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            pbar.set_postfix(train_loss='{:.5}'.format(sum(loss_history[-10:])/10))
    lr_scheduler.step()
with torch.no_grad():
    model.eval()
    tmp0 = [(model(x).numpy()/SCALE_FACTOR,y.numpy()) for x,y in testloader]
    tmp1 = np.concatenate([x[0] for x in tmp0])
    tmp2 = np.concatenate([x[1] for x in tmp0])
    print('mae=', np.abs(tmp1-tmp2).mean(axis=0))

fig,ax = plt.subplots()
ax.plot(np.arange(len(loss_history)), loss_history, '.')
ax.set_xlabel('iteration')
ax.set_ylabel('loss')
ax.grid()
fig.tight_layout()
fig.savefig('tbd01.png', dpi=200)


# dm0 = numqi.random.rand_density_matrix(dimA*dimB, k=2*dimA*dimB)
# dm1 = numqi.random.rand_density_matrix(dimA*dimB, k=2*dimA*dimB)
dm0 = numqi.state.Werner(dimA, 1)
dm1 = numqi.state.Isotropic(dimA, 1)
theta1, hf_theta = numqi.entangle.get_density_matrix_plane(dm0, dm1)

model.eval()
theta_list = np.linspace(-np.pi, np.pi, 200)
beta_ppt = np.zeros_like(theta_list)
beta_dm = np.zeros_like(theta_list)
beta_clf = np.zeros((len(theta_list),2))
for ind0,x in enumerate(theta_list):
    dm_target = hf_theta(x)
    beta_ppt[ind0] = numqi.entangle.get_ppt_boundary(dm_target, (dimA, dimB))[1]
    beta_dm[ind0] = numqi.entangle.get_density_matrix_boundary(dm_target)[1]
    tmp0 = torch.tensor(numqi.gellmann.dm_to_gellmann_basis(dm_target), dtype=torch.float32)
    with torch.no_grad():
        beta_clf[ind0] = model(tmp0).numpy()/SCALE_FACTOR

label0 = 'werner'
label1 = 'isotropic'
fig,ax = plt.subplots()
hf0 = lambda theta,r: (r*np.cos(theta), r*np.sin(theta))
for theta, label in [(0,label0),(theta1,label1)]:
    radius = 0.3
    ax.plot([0, radius*np.cos(theta)], [0, radius*np.sin(theta)], linestyle=':', label=label)
ax.plot(*hf0(theta_list, beta_dm), label='DM')
ax.plot(*hf0(theta_list, beta_ppt), linestyle='--', label='PPT')
# ax.plot(*hf0(theta_list, beta_clf[:,0]), linestyle='--', label='classifier-dm')
ax.plot(*hf0(theta_list, beta_clf[:,1]), linestyle='--', label='classifier-ppt')
ax.set_aspect('equal')
ax.legend()
fig.tight_layout()
fig.savefig('tbd01.png', dpi=200)

import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import sklearn.metrics
import scipy.stats
import scipy.special

import numqi

tableau = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']


def get_witness(dimA, dimB, num_sample, kext=3, seed=None):
    np_rng = numqi.random.get_numpy_rng(seed)
    kwargs = dict(dim=(dimA,dimB), kext=kext, use_ppt=True, use_boson=True, return_info=True, use_tqdm=True)
    dm0 = np.stack([numqi.random.rand_density_matrix(dimA*dimB, seed=np_rng) for _ in range(num_sample)])
    beta,vecA,vecN = numqi.entangle.get_ABk_symmetric_extension_boundary(dm0, **kwargs)
    tmp0 = np.eye(dimA*dimB) / (dimA*dimB)
    ret = [(numqi.gellmann.gellmann_basis_to_dm(y)-tmp0, 2*np.dot(x,y)) for x,y in zip(vecA,vecN)]
    # list of (op, delta)
    # Tr[rho op] <= delta for all SEP
    return ret


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


def rand_npt_entangle_state(dim, num_sample, haar_k=None, witness_list=None, seed=None):
    np_rng = numqi.random.get_numpy_rng(seed)
    dimA,dimB = dim
    if haar_k is None:
        haar_k = 2*dimA*dimB
    ent_list = []
    dm_list = []
    ppt_tag = [] if witness_list is not None else None
    num_total = 0
    while len(ent_list)<num_sample:
        num_total = num_total + 1
        rho = numqi.random.rand_density_matrix(dimA*dimB, k=haar_k, seed=np_rng)
        dm_list.append(rho)
        rho_pt = rho.reshape(dimA,dimB,dimA,dimB).transpose(0,3,2,1).reshape(dimA*dimB,dimA*dimB)
        if not is_positive_semi_definite(rho_pt):
            ent_list.append(rho)
            if ppt_tag is not None:
                ppt_tag.append(False)
        elif ppt_tag is not None:
            tmp0 = any(np.vdot(rho.reshape(-1), x.reshape(-1)).real>y for x,y in witness_list)
            ppt_tag.append(tmp0)
            if tmp0:
                ent_list.append(rho)
    ret = (ent_list,ppt_tag) if (ppt_tag is not None) else ent_list
    return ret

class DensityMatrixBoundary(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        np_rng = numqi.random.get_numpy_rng()
        self.theta = torch.nn.Parameter(torch.tensor(np_rng.uniform(-1,1,size=(2,dim,dim)), dtype=torch.float64))
        self.rho0 = torch.eye(dim, dtype=torch.complex128) / dim
        self.rho = None

    def forward(self):
        tmp0 = torch.complex(self.theta[0], self.theta[1])
        tmp1 = tmp0 @ tmp0.T.conj()
        rho = tmp1 / torch.trace(tmp1)
        self.rho = rho.detach()
        tmp0 = (rho - self.rho0).reshape(-1)
        loss = -torch.vdot(tmp0, tmp0).real/2
        return loss


def rand_n_ball(dim, size=None, radius=1, seed=None):
    # https://en.wikipedia.org/wiki/Unit_sphere
    dim = int(dim)
    assert dim>=1
    np_rng = numqi.random.get_numpy_rng(seed)
    is_single = (size is None)
    if is_single:
        size = 1
    size = tuple(int(x) for x in size) if hasattr(size,'__len__') else (int(size),)
    tmp0 = np_rng.normal(size=size+(dim,))
    tmp0 /= np.linalg.norm(tmp0, axis=-1, keepdims=True)
    tmp1 = np_rng.uniform(0,1,size=size+(1,))**(1.0/dim)
    ret = tmp0 * tmp1 * radius
    if is_single:
        ret = ret[0]
    return ret


def get_density_matrix_boundary(dim):
    tmp0 = dict([(2,1/4), (3,1/3), (4,3/8), (5,0.4), (6,5/12), (7,3/7), (8,7/16), (9,4/9)])
    if dim in tmp0:
        ret = tmp0[dim]
    else:
        model = DensityMatrixBoundary(dim)
        theta_optim = numqi.optimize.minimize(model, theta0='uniform', tol=1e-12, num_repeat=3)
        ret = -theta_optim.fun
    ret = np.sqrt(max(ret, 0))
    return ret


def rand_sixparam_ent_state(num_sample, num_per_upb=10, kext=3, seed=None):
    np_rng = numqi.random.get_numpy_rng(seed)
    tmp0 = [np_rng.uniform(0, 2*np.pi, size=6) for _ in range(num_sample)]
    dm_list = [numqi.entangle.load_upb('sixparam',x,return_bes=True)[1] for x in tmp0]
    kwargs = dict(dim=(3,3), kext=kext, use_ppt=True, use_boson=True, return_info=True, use_tqdm=True)
    beta,vecA,vecN = numqi.entangle.get_ABk_symmetric_extension_boundary(dm_list, **kwargs)
    alpha = np_rng.uniform(0,1,size=num_sample)
    tmp0 = numqi.gellmann.gellmann_basis_to_dm(vecA)
    ret = [x*z+y*(1-z) for x,y in zip(dm_list,tmp0) for z in np_rng.uniform(0,1,size=num_per_upb)]
    return ret


def save_data(sep_list, npt_list, bes_list, filepath='data_sep_ent_bes.pkl'):
    vec_sep = numqi.gellmann.dm_to_gellmann_basis(np.stack(sep_list))
    vec_npt = numqi.gellmann.dm_to_gellmann_basis(np.stack(npt_list))
    vec_bes = numqi.gellmann.dm_to_gellmann_basis(np.stack(bes_list))
    with open(filepath, 'wb') as fid:
        tmp0 = dict(sep=vec_sep, npt=vec_npt, bes=vec_bes)
        pickle.dump(tmp0, fid)

def load_data(kind='vec', filepath='data_sep_ent_bes.pkl'):
    assert kind in {'vec','dm'}
    with open(filepath, 'rb') as fid:
        tmp0 = pickle.load(fid)
        ret = tmp0['sep'], tmp0['npt'], tmp0['bes']
    if kind=='dm':
        ret = tuple(numqi.gellmann.gellmann_basis_to_dm(x) for x in ret)
    return ret


class DummyFCModel(torch.nn.Module):
    def __init__(self, scale_factor=10):
        super().__init__()
        tmp0 = [80, 128, 128, 512, 512, 128, 128, 1]
        self.fc_list = torch.nn.ModuleList([torch.nn.Linear(tmp0[x], tmp0[x+1]) for x in range(len(tmp0)-1)])
        self.bn_list = torch.nn.ModuleList([torch.nn.BatchNorm1d(tmp0[x+1]) for x in range(len(tmp0)-1)])
        self.scale_factor = scale_factor

    def forward(self, x):
        x = x * self.scale_factor
        is_single = (len(x.shape)==1)
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
        x = x[:,0]
        if is_single:
            x = x[0]
        return x

    def predict(self, dm_list):
        if isinstance(dm_list, np.ndarray) and dm_list.ndim==2:
            is_single = True
            dm_list = dm_list[np.newaxis]
        else:
            is_single = False
            dm_list = np.stack(dm_list)
            assert dm_list.ndim==3
        vec = torch.tensor(numqi.gellmann.dm_to_gellmann_basis(dm_list), dtype=torch.float32)
        with torch.no_grad():
            model.eval()
            predict = model(vec).numpy()
        if is_single:
            predict = predict[0]
        # ENT(negative), SEP(positive)
        return predict


def rand_separable_dm(dimA, dimB, k, alpha=1/2, seed=None):
    np_rng = numqi.random.get_numpy_rng(seed)
    probability = scipy.stats.dirichlet.rvs(alpha*np.ones(k), random_state=np_rng)[0]
    ret = 0
    for ind0 in range(k):
        tmp0 = numqi.random.rand_haar_state(dimA, seed=np_rng)
        tmp1 = numqi.random.rand_haar_state(dimB, seed=np_rng)
        tmp2 = (tmp0.reshape(-1,1)*tmp1).reshape(-1)
        ret = ret + probability[ind0] * tmp2.reshape(-1,1)*tmp2.conj()
    return ret


np_rng = np.random.default_rng()
dimA = 3
dimB = 3
num_total_data = 1000000
sep_ent_bes_ratio = [0.3, 0.6, 0.1]
haar_k_sep = dimA*dimB
haar_k_ent = 2*dimA*dimB
batch_size = 256

tmp0 = int(num_total_data*sep_ent_bes_ratio[0])
data_sep_list = [rand_separable_dm(dimA, dimB, k=haar_k_sep, alpha=1/2, seed=np_rng) for _ in range(tmp0//2)]
data_sep_list += [rand_separable_dm(dimA, dimB, k=haar_k_sep*2, alpha=1/2, seed=np_rng) for _ in range(tmp0//2)]
# data_npt_list = rand_npt_entangle_state((dimA,dimB), int(num_total_data*sep_ent_bes_ratio[1]), haar_k=haar_k_ent, seed=np_rng)
# data_bes_list = rand_sixparam_ent_state(int(num_total_data*sep_ent_bes_ratio[2])//10, num_per_upb=10, kext=3, seed=np_rng)
# save_data(data_sep_list, data_npt_list, data_bes_list)
vec_sep_list,vec_npt_list,vec_bes_list = load_data(kind='vec')

# SEP:1 ENT:0
data = np.concatenate([vec_sep_list, vec_npt_list, vec_bes_list], axis=0)
label = np.concatenate([np.ones(len(vec_sep_list)), np.zeros(len(vec_npt_list)), np.zeros(len(vec_bes_list))], axis=0)

tmp0 = torch.tensor(data, dtype=torch.float32)
tmp1 = torch.tensor(label, dtype=torch.int64)
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
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(data_i), label_i.to(torch.float32), reduction='mean')
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            pbar.set_postfix(train_loss='{:.5}'.format(sum(loss_history[-10:])/10))
    lr_scheduler.step()
with torch.no_grad():
    model.eval()
    tmp0 = [(model(x), y) for x,y in testloader]
    tmp1 = np.concatenate([x[0].numpy() for x in tmp0], axis=0)
    tmp2 = np.concatenate([x[1].numpy() for x in tmp0], axis=0)
    acc = ((tmp1>0).astype(int)==tmp2).mean()
    auc = sklearn.metrics.roc_auc_score(tmp2, scipy.special.expit(tmp1))

dm_npt = rand_npt_entangle_state((dimA,dimB), 1000, haar_k=haar_k_ent, seed=np_rng)
predict_npt = model.predict(dm_npt)
dm_sep = [rand_separable_dm(dimA, dimB, k=haar_k_sep, seed=np_rng) for _ in range(1000)]
predict_sep = model.predict(dm_sep)
# dm_bes = rand_sixparam_ent_state(100, num_per_upb=10, kext=3, seed=np_rng)
# predict_bes = model.predict(dm_bes)

print((predict_npt<0).mean(), (predict_sep>0).mean()) #, (predict_bes<0).mean())

fig,ax = plt.subplots()
ax.hist(predict_sep, bins=20, density=True, label='SEP', color=tableau[0], alpha=0.3)
ax.hist(predict_npt, bins=20, density=True, label='NPT', color=tableau[1], alpha=0.3)
# ax.hist(predict_bes, bins=20, density=True, label='BES', color=tableau[2], alpha=0.3)
ax.legend()
ax.set_xlabel('logits')
ax.set_ylabel('probablity')
fig.tight_layout()
fig.savefig('tbd01.png', dpi=200)


dm0 = numqi.random.rand_density_matrix(dimA*dimB, k=2*dimA*dimB)
dm1 = numqi.random.rand_density_matrix(dimA*dimB, k=2*dimA*dimB)
dm0 = numqi.state.Werner(dimA, 1)
dm1 = numqi.state.Isotropic(dimA, 1)
dm1 = numqi.entangle.load_upb('tiles', return_bes=True)[1]
theta1, hf_theta = numqi.entangle.get_density_matrix_plane(dm0, dm1)

theta_list = np.linspace(-np.pi, np.pi, 200)
beta_ppt = np.zeros_like(theta_list)
beta_dm = np.zeros_like(theta_list)
for ind0,x in enumerate(theta_list):
    dm_target = hf_theta(x)
    beta_ppt[ind0] = numqi.entangle.get_ppt_boundary(dm_target, (dimA, dimB))[1]
    beta_dm[ind0] = numqi.entangle.get_density_matrix_boundary(dm_target)[1]

tmp0 = beta_dm*np.cos(theta_list)
tmp1 = beta_dm*np.sin(theta_list)
xdata = np.linspace(tmp0.min()*1.1, tmp0.max()*1.1, 40)
ydata = np.linspace(tmp1.min()*1.1, tmp1.max()*1.1, 40)
tmp0 = hf_theta((xdata.reshape(-1,1,1), ydata.reshape(-1,1)))
with torch.no_grad():
    model.eval()
    predict = model(torch.tensor(tmp0.reshape(-1,tmp0.shape[-1]), dtype=torch.float32)).numpy().reshape(tmp0.shape[0],-1)

tmp1 = numqi.gellmann.gellmann_basis_to_dm(tmp0.reshape(-1,tmp0.shape[-1]))
tag_dm = np.array([is_positive_semi_definite(x) for x in tmp1]).reshape(tmp0.shape[0],-1)
tmp2 = tmp1.reshape(-1,dimA,dimB,dimA,dimB).transpose(0,1,4,3,2).reshape(-1,dimA*dimB,dimA*dimB)
tag_ppt = np.array([is_positive_semi_definite(x) for x in tmp2]).reshape(tmp0.shape[0],-1)

label0 = 'werner'
label1 = 'tiles/BES'
fig,ax = plt.subplots()
tmp0 = predict.T.copy()
# tmp0[tmp0<-6] = np.nan
hcontour = ax.contourf(xdata, ydata, tmp0)
fig.colorbar(hcontour)
hf0 = lambda theta,r: (r*np.cos(theta), r*np.sin(theta))
for theta, label in [(0,label0),(theta1,label1)]:
    radius = 0.3
    ax.plot([0, radius*np.cos(theta)], [0, radius*np.sin(theta)], linestyle=':', label=label)
ax.plot(*hf0(theta_list, beta_dm), label='DM')
ax.plot(*hf0(theta_list, beta_ppt), linestyle='--', label='PPT')
ax.legend()
ax.set_aspect('equal')
fig.savefig('tbd02.png', dpi=200)

import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import sklearn.metrics
import scipy.special

import numqi

from utils import rand_npt_entangle_state, rand_separable_dm, rand_sixparam_ent_state

tableau = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']


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

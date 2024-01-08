import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.stats

import numqi

cp_tableau = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3', '#937860', '#da8bc3', '#8c8c8c', '#ccb974', '#64b5cd']

class DummyFCModel(torch.nn.Module):
    def __init__(self, dim, dim_fc_list=None, device:(torch.device|None)=None, scale=0.05):
        super().__init__()
        if dim_fc_list is None:
            dim_fc_list = [dim**2-1, 128, 256, 256, 512, 512, 512, 256, 256, 128, 1]
        else:
            assert dim_fc_list[0] == dim**2-1
        if device is None:
            device = torch.device('cpu')
        self.fc_list = torch.nn.ModuleList([torch.nn.Linear(dim_fc_list[x], dim_fc_list[x+1], device=device) for x in range(len(dim_fc_list)-1)])
        self.bn_list = torch.nn.ModuleList([torch.nn.BatchNorm1d(dim_fc_list[x+1], device=device) for x in range(len(dim_fc_list)-1)])
        assert scale>=0
        self.a = torch.tensor(np.sqrt(1/(2*dim*(dim-1))) * (1-scale), dtype=torch.float32, device=device)
        self.b = torch.tensor(np.sqrt((dim-1)/(2*dim)) * (1+scale), dtype=torch.float32, device=device)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1,shape[-1])
        x = x/torch.linalg.norm(x, dim=-1, keepdim=True)
        for ind0 in range(len(self.fc_list)):
            y = self.fc_list[ind0](x)
            y = self.bn_list[ind0](y)
            if ind0<(len(self.fc_list)-1):
                y = torch.nn.functional.leaky_relu(y)
            if x.shape[-1]==y.shape[-1]:
                x = y + x
            else:
                x = y
        x = (self.b - self.a) * torch.sigmoid(x[:,0]) + self.a
        if len(shape)==1:
            x = x[0]
        else:
            x = x.reshape(shape[:-1])
        return x


def plot_bloch_vector_plane(model, op0, op1, dim, num_theta=500, tag_ppt=True, filename=None):
    if hasattr(dim, '__len__'):
        assert len(dim)==2
        dimA,dimB = dim
        dim = dimA*dimB
    else:
        dimA,dimB = None,None
    tag = model.training
    model.eval()
    device = next(iter(model.parameters())).device
    theta_list = np.linspace(0, 2*np.pi, num_theta)

    _, hf_plane = numqi.entangle.get_density_matrix_plane(op0, op1)
    beta_dm_list = np.array([numqi.entangle.get_density_matrix_boundary(hf_plane(x))[1] for x in theta_list])

    if (dimA is not None) and tag_ppt:
        beta_ppt_list = np.array([numqi.entangle.get_ppt_boundary(hf_plane(x), (dimA, dimB))[1] for x in theta_list])
    else:
        beta_ppt_list = None

    tmp0 = np.stack([numqi.gellmann.dm_to_gellmann_basis(hf_plane(x)) for x in theta_list]) #already normalized
    model.eval()
    with torch.no_grad():
        beta_clf_list = model(torch.tensor(tmp0, dtype=torch.float32, device=device)).numpy().copy()

    r_inscribed = np.sqrt(1/(2*dim*(dim-1)))
    r_bounding = np.sqrt((dim-1)/(2*dim))

    fig, ax = plt.subplots()
    tmp0 = np.cos(theta_list)
    tmp1 = np.sin(theta_list)
    ax.plot(beta_dm_list*tmp0, beta_dm_list*tmp1, color=cp_tableau[0], label='DM')
    if beta_ppt_list is not None:
        ax.plot(beta_ppt_list*tmp0, beta_ppt_list*tmp1, color=cp_tableau[3], label='PPT')
    ax.plot(beta_clf_list*tmp0, beta_clf_list*tmp1, color=cp_tableau[1], linestyle='dashed', label='Classifier')
    ax.plot(r_inscribed*tmp0, r_inscribed*tmp1, color=cp_tableau[2], linestyle='dashed')
    ax.plot(r_bounding*tmp0, r_bounding*tmp1, color=cp_tableau[2], linestyle='dashed')
    ax.axis('equal')
    ax.legend()
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, dpi=200)
    if tag:
        model.train()
    return fig,ax


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


def rand_witness(dimA, dimB, num_sample, kext=3, seed=None):
    np_rng = numqi.random.get_numpy_rng(seed)
    kwargs = dict(dim=(dimA,dimB), kext=kext, use_ppt=True, use_boson=True, return_info=True, use_tqdm=True)
    dm0 = np.stack([numqi.random.rand_density_matrix(dimA*dimB, seed=np_rng) for _ in range(num_sample)])
    beta,vecA,vecN = numqi.entangle.get_ABk_symmetric_extension_boundary(dm0, **kwargs)
    tmp0 = np.eye(dimA*dimB) / (dimA*dimB)
    ret = [(numqi.gellmann.gellmann_basis_to_dm(y)-tmp0, 2*np.dot(x,y)) for x,y in zip(vecA,vecN)]
    # list of (op, delta)
    # Tr[rho op] <= delta for all SEP
    return ret


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

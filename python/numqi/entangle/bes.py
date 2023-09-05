import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import numqi.param
import numqi.gellmann

from ._misc import get_density_matrix_boundary, get_density_matrix_plane, hf_interpolate_dm
from .cha import CHABoundaryBagging
from .ppt import get_generalized_ppt_boundary, get_ppt_boundary
from .pureb import PureBosonicExt

def plot_dm0_dm1_plane(dm0, dm1, dimA, dimB, num_point=201, pureb_kext=None, tag_cha=False,
        num_eig0=None, label0=None, label1=None, with_gppt=False):
    if label0 is None:
        label0 = r'$\rho_a$'
    if label1 is None:
        label1 = r'$\rho_b$'
    theta1,hf_theta = get_density_matrix_plane(dm0, dm1)
    if num_eig0 is None:
        tmp0 = np.linalg.eigvalsh(hf_interpolate_dm(hf_theta(theta1),
                beta=get_density_matrix_boundary(dm1)[1]))
        num_eig0 = (tmp0<1e-7).sum()

    theta_list = np.linspace(-np.pi, np.pi, num_point)
    beta_ppt = np.zeros_like(theta_list)
    beta_dm = np.zeros_like(theta_list)
    eig_dm = np.zeros((dimA*dimB, len(theta_list)), dtype=np.float64)
    for ind0,x in enumerate(tqdm(theta_list)):
        dm_target = hf_theta(x)
        beta_ppt[ind0] = get_ppt_boundary(dm_target, (dimA, dimB))[1]
        beta_dm[ind0] = get_density_matrix_boundary(dm_target)[1]
        eig_dm[:,ind0] = np.linalg.eigvalsh(hf_interpolate_dm(dm_target, beta=beta_dm[ind0]))

    if tag_cha:
        dm_target_list = [hf_theta(x) for x in theta_list]
        model_cha = CHABoundaryBagging((dimA,dimB))
        beta_cha = np.array([model_cha.solve(x,use_tqdm=False) for x in tqdm(dm_target_list)])
    else:
        beta_cha = None
    if pureb_kext is not None:
        if not hasattr(pureb_kext, '__len__'):
            pureb_kext = [int(pureb_kext)]
        beta_pureb = np.zeros((len(pureb_kext), len(theta_list)), dtype=np.float64)
        for ind0 in range(len(pureb_kext)):
            model = PureBosonicExt(dimA, dimB, kext=pureb_kext[ind0])
            for ind1,x in enumerate(tqdm(theta_list, desc=f'PureB-{pureb_kext[ind0]}')):
                beta_pureb[ind0,ind1] = model.get_boundary(hf_theta(x), xtol=1e-4, converge_tol=1e-10, threshold=1e-7, num_repeat=1, use_tqdm=False)
    else:
        beta_pureb = None

    if with_gppt:
        beta_gppt = np.zeros_like(theta_list)
        for ind0,x in enumerate(tqdm(theta_list, desc='generalized-ppt')):
            dm_target = hf_theta(x)
            beta_gppt[ind0] = get_generalized_ppt_boundary(dm_target, (dimA,dimB), xtol=1e-5)
    else:
        beta_gppt = None

    fig,ax = plt.subplots()
    hf0 = lambda theta,r: (r*np.cos(theta), r*np.sin(theta))
    for theta, label in [(0,label0),(theta1,label1)]:
        radius = 0.3
        ax.plot([0, radius*np.cos(theta)], [0, radius*np.sin(theta)], linestyle=':', label=label)
    ax.plot(*hf0(theta_list, beta_dm), label='DM')
    ax.plot(*hf0(theta_list, beta_ppt), linestyle='--', label='PPT')
    if beta_pureb is not None:
        for ind0 in range(len(pureb_kext)):
            ax.plot(*hf0(theta_list, beta_pureb[ind0]), label=f'PureB({pureb_kext[ind0]})')
    if beta_cha is not None:
        ax.plot(*hf0(theta_list, beta_cha), label='CHA')
    for ind0 in range(1, num_eig0):
        ax.plot(*hf0(theta_list, eig_dm[ind0]), label=rf'$\lambda_{ind0+1}$ dm')
    if beta_gppt is not None:
        ax.plot(*hf0(theta_list, beta_gppt), label='realignment')
    ax.legend(fontsize='small')
    # ax.legend(fontsize=11, ncol=2, loc='lower right')
    # ax.tick_params(axis='both', which='major', labelsize=11)
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)
    all_data = dict(dm0=dm0, dm1=dm1, theta_list=theta_list, eig_dm=eig_dm, beta_dm=beta_dm,
            beta_ppt=beta_ppt, beta_cha=beta_cha, pureb_kext=pureb_kext, beta_pureb=beta_pureb, num_eig0=num_eig0, beta_gppt=beta_gppt)
    return fig,ax,all_data


class DensityMatrixLocalUnitaryEquivalentModel(torch.nn.Module):
    def __init__(self, dimA, dimB, num_term=1):
        super().__init__()
        np_rng = np.random.default_rng()
        self.theta0 = torch.nn.Parameter(torch.tensor(np_rng.uniform(-1, 1, size=(num_term,dimA,dimA)), dtype=torch.float64))
        self.theta1 = torch.nn.Parameter(torch.tensor(np_rng.uniform(-1, 1, size=(num_term,dimB,dimB)), dtype=torch.float64))
        tmp0 = np_rng.uniform(0, 1, size=num_term)
        self.probability = torch.nn.Parameter(torch.tensor(tmp0/tmp0.sum(), dtype=torch.float64))
        self.dimA = dimA
        self.dimB = dimB
        self.dm0 = None
        self.dm1 = None

    def set_density_matrix(self, dm0=None, dm1=None):
        if dm0 is not None:
            assert (dm0.shape[0]==(self.dimA*self.dimB)) and (dm0.shape[1]==(self.dimA*self.dimB))
            self.dm0 = torch.tensor(dm0, dtype=torch.complex128)
        if dm1 is not None:
            assert (dm1.shape[0]==(self.dimA*self.dimB)) and (dm1.shape[1]==(self.dimA*self.dimB))
            self.dm1 = torch.tensor(dm1, dtype=torch.complex128)

    def forward(self):
        probability = torch.nn.functional.softmax(self.probability, dim=0)
        u0 = numqi.param.real_matrix_to_special_unitary(self.theta0)
        u0H = u0.conj()
        u1 = numqi.param.real_matrix_to_special_unitary(self.theta1)
        u1H = u1.conj()
        dm0 = self.dm0.reshape(self.dimA,self.dimB,self.dimA,self.dimB)
        ret = torch.einsum(dm0, [0,1,2,3], u0, [8,4,0], u0H, [8,5,2], u1, [8,6,1], u1H, [8,7,3], probability, [8], [4,6,5,7]).reshape(self.dm1.shape)
        # loss = numqi.utils.get_relative_entropy(ret, self.dm1)
        tmp0 = (ret - self.dm1).reshape(-1)
        loss = torch.dot(tmp0, tmp0.conj()).real
        return loss


class BESNumEigenModel(torch.nn.Module):
    def __init__(self, dimA, dimB, rank0, rank1=None, with_ppt=True, with_ppt1=True):
        # TODO what is the loss for the genshfits UPB/BES
        super().__init__()
        np_rng = np.random.default_rng()
        tmp0 = np_rng.uniform(-1, 1, size=(dimA*dimB)**2-1)
        self.rho_vec = torch.nn.Parameter(torch.tensor(tmp0 / np.linalg.norm(tmp0), dtype=torch.float64))
        self.dimA = dimA
        self.dimB = dimB
        self.rank0 = rank0
        self.rank1 = (dimA*dimB-rank0) if (rank1 is None) else rank1
        self.with_ppt = with_ppt
        self.with_ppt1 = with_ppt1

    def forward(self):
        rho_vec_norm = self.rho_vec / torch.linalg.norm(self.rho_vec)
        rho_norm = numqi.gellmann.gellmann_basis_to_dm(rho_vec_norm)
        tmp0 = torch.linalg.eigvalsh(rho_norm)
        beta0 = 1/(1-rho_norm.shape[0]*tmp0[0])
        beta1 = 1/(1-rho_norm.shape[0]*tmp0[-1])
        dm0 = numqi.gellmann.gellmann_basis_to_dm(beta0*rho_vec_norm)
        dm1 = numqi.gellmann.gellmann_basis_to_dm(beta1*rho_vec_norm)
        loss0 = torch.linalg.eigvalsh(dm0)[:(dm0.shape[0]-self.rank0)].sum()
        loss1 = torch.linalg.eigvalsh(dm1)[:(dm0.shape[0]-self.rank1)].sum()
        loss = (loss0 + loss1)**2
        if self.with_ppt:
            tmp1 = dm0.reshape(self.dimA,self.dimB,self.dimA,self.dimB).transpose(1,3).reshape(self.dimA*self.dimB,-1)
            loss2 = torch.linalg.eigvalsh(tmp1)[0]**2
            loss = loss + loss2
            # without this constraint, it will not converge to BES
            tmp1 = dm1.reshape(self.dimA,self.dimB,self.dimA,self.dimB).transpose(1,3).reshape(self.dimA*self.dimB,-1)
            loss3 = torch.linalg.eigvalsh(tmp1)[0]**2
            loss = loss + loss3
        return loss


class BESNumEigen3qubitModel(torch.nn.Module):
    def __init__(self, rank0, rank1=None, with_ppt=True):
        # TODO what is the loss for the genshfits UPB/BES
        super().__init__()
        np_rng = np.random.default_rng()
        dimA,dimB,dimC = 2,2,2
        tmp0 = np_rng.uniform(-1, 1, size=(dimA*dimB*dimC)**2-1)
        self.rho_vec = torch.nn.Parameter(torch.tensor(tmp0 / np.linalg.norm(tmp0), dtype=torch.float64))
        self.dimA = dimA
        self.dimB = dimB
        self.dimC = dimC
        self.rank0 = rank0
        self.rank1 = (dimA*dimB*dimC-rank0) if (rank1 is None) else rank1
        self.with_ppt = with_ppt

    def forward(self):
        rho_vec_norm = self.rho_vec / torch.linalg.norm(self.rho_vec)
        rho_norm = numqi.gellmann.gellmann_basis_to_dm(rho_vec_norm)
        tmp0 = torch.linalg.eigvalsh(rho_norm)
        beta0 = 1/(1-rho_norm.shape[0]*tmp0[0])
        beta1 = 1/(1-rho_norm.shape[0]*tmp0[-1])
        dm0 = numqi.gellmann.gellmann_basis_to_dm(beta0*rho_vec_norm)
        dm1 = numqi.gellmann.gellmann_basis_to_dm(beta1*rho_vec_norm)
        loss0 = torch.linalg.eigvalsh(dm0)[:(dm0.shape[0]-self.rank0)].sum()
        loss1 = torch.linalg.eigvalsh(dm1)[:(dm0.shape[0]-self.rank1)].sum()
        loss = (loss0 + loss1)**2
        if self.with_ppt:
            for dm_i in [dm0,dm1]:
                tmp1 = dm_i.reshape(self.dimA,self.dimB*self.dimC,self.dimA,-1).transpose(1,3).reshape(self.dimA*self.dimB*self.dimC,-1)
                loss = loss + torch.linalg.eigvalsh(tmp1)[0]**2
                tmp1 = dm_i.reshape(-1,self.dimC,self.dimA*self.dimB,self.dimC).transpose(1,3).reshape(self.dimA*self.dimB*self.dimC,-1)
                loss = loss + torch.linalg.eigvalsh(tmp1)[0]**2
        return loss

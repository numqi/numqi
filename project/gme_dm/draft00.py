import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import numqi

np_rng = np.random.default_rng()
tableau = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']

hf_data = lambda *x: os.path.join('data', *x)
if not os.path.exists(hf_data()):
    os.makedirs(hf_data())

def demo_werner_gme():
    alpha_list = np.linspace(0,1,100)
    dim = 3
    datapath = hf_data('fig_werner_gme.pkl')
    if not os.path.exists(datapath):
        model = numqi.entangle.DensityMatrixGMEModel([dim,dim], num_ensemble=18, rank=9)
        ret = []
        for alpha_i in tqdm(alpha_list):
            werner_rho = numqi.state.Werner(dim, alpha=alpha_i)
            model.set_density_matrix(werner_rho)
            ret.append(numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
        ret = np.array(ret)
        ret_analytical = numqi.state.get_Werner_GME(dim, alpha_list)
        with open(datapath, 'wb') as fid:
            tmp0 = dict(alpha_list=alpha_list, ret=ret, ret_analytical=ret_analytical)
            pickle.dump(tmp0, fid)
    else:
        with open(datapath, 'rb') as fid:
            tmp0 = pickle.load(fid)
            alpha_list = tmp0['alpha_list']
            ret = tmp0['ret']
            ret_analytical = tmp0['ret_analytical']

    fig,ax = plt.subplots()
    ax.plot(alpha_list, ret, label='numerical result', color=tableau[0])
    ax.plot(alpha_list, ret_analytical, 'o', markerfacecolor='none', label='analytical result', color=tableau[0])
    ax.axvline(1/dim, color='k', linestyle='--', label='sep-ent boundary')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$E_G(\alpha)$')
    ax.legend()
    fig.savefig(hf_data('fig_werner_gme.pdf'))
    fig.savefig(hf_data('fig_werner_gme.png'), dpi=200)


def demo_bound_entangled_state():
    datapath = hf_data('fig_bes_gme.pkl')
    if not os.path.exists(datapath):
        rho_bes = numqi.entangle.load_upb('tiles', return_bes=True)[1]
        alpha_list = np.linspace(0, 1, 100)
        ret = []
        model = numqi.entangle.DensityMatrixGMEModel(dim_list=[3,3], num_ensemble=18, rank=9)
        for alpha_i in tqdm(alpha_list):
            rho = (1-alpha_i) * np.eye(9) / 9 + alpha_i * rho_bes
            model.set_density_matrix(rho)
            ret.append(numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
        ret = np.array(ret)
        with open(datapath, 'wb') as fid:
            pickle.dump(dict(alpha_list=alpha_list, ret=ret), fid)
    else:
        with open(datapath, 'rb') as fid:
            tmp0 = pickle.load(fid)
            alpha_list = tmp0['alpha_list']
            ret = tmp0['ret']

    fig,ax = plt.subplots()
    ax.plot(alpha_list, ret, label='numerical result')
    ax.axvline(0.8649, color='k', linestyle='--', label='sep-ent boundary')
    # 0.8649 is from https://journals.aps.org/pra/abstract/10.1103/PhysRevA.98.012315
    ax.set_yscale('log')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$E_G(\alpha)$')
    ax.tick_params(axis='y', which='minor', bottom=False, top=False, left=False, right=False)
    ax.legend()
    fig.savefig(hf_data('fig_bes_gme.pdf'))
    fig.savefig(hf_data('fig_bes_gme.png'), dpi=200)


def demo_higher_entangled_state():
    # https://doi.org/10.1103/PhysRevA.106.062443
    datapath = hf_data('fig_higher_entangled.pkl')
    dimA = 4
    dimB = 4
    tmp0 = [
        [(0,0,1), (1,1,1), (2,2,1), (3,3,1)],
        [(0,1,1), (1,2,1), (2,3,1), (3,0,1)],
        [(0,2,1), (1,3,1), (2,0,1), (3,1,-1)],
    ]
    matrix_subspace = np.stack([numqi.matrix_space.build_matrix_with_index_value(dimA, dimB, x) for x in tmp0])
    rho = np.einsum(matrix_subspace, [0,1,2], matrix_subspace.conj(), [0,3,4], [1,2,3,4], optimize=True).reshape(dimA*dimB,dimA*dimB) / 12

    if not os.path.exists(datapath):
        alpha_list = np.linspace(0, 1, 200)
        ret1 = []
        ret2 = []
        model1 = numqi.entangle.DensityMatrixGMEModel([dimA,dimB], num_ensemble=32, rank=16, CPrank=1)
        model2 = numqi.entangle.DensityMatrixGMEModel([dimA,dimB], num_ensemble=64, rank=16, CPrank=2)
        for alpha_i in tqdm(alpha_list):
            tmp0 = (1-alpha_i) * np.eye(dimA*dimB) / (dimA*dimB) + alpha_i * rho
            model1.set_density_matrix(tmp0)
            ret1.append(numqi.optimize.minimize(model1, num_repeat=3, tol=1e-10, print_every_round=0).fun)
            model2.set_density_matrix(tmp0)
            ret2.append(numqi.optimize.minimize(model2, num_repeat=3, tol=1e-10, print_every_round=0).fun)
        ret1 = np.array(ret1)
        ret2 = np.array(ret2)
        with open(datapath, 'wb') as fid:
            tmp0 = dict(alpha_list=alpha_list, ret1=ret1, ret2=ret2)
            pickle.dump(tmp0, fid)
    else:
        with open(datapath, 'rb') as fid:
            tmp0 = pickle.load(fid)
            alpha_list = tmp0['alpha_list']
            ret1 = tmp0['ret1']
            ret2 = tmp0['ret2']

    fig,ax = plt.subplots()
    ax.plot(alpha_list, ret1, label='r=2')
    ax.plot(alpha_list, ret2, label='r=3')
    ax.set_xlim(0, 1)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$E_r(\alpha)$')
    ax.set_yscale('log')
    ax.grid()
    ax.legend()
    fig.tight_layout()
    fig.savefig(hf_data('fig_higher_entangled.pdf'))
    fig.savefig(hf_data('fig_higher_entangled.png'), dpi=200)


def demo_multipartite():
    datapath = hf_data('fig_multipartite.pkl')
    # (0000 + 0011 + 1100 - 1111)/2
    psi_cluster = np.zeros(16, dtype=np.float64)
    psi_cluster[[0, 3, 12, 15]] = np.array([1,1,1,-1])/2
    rho_cluster = psi_cluster.reshape(-1,1) * psi_cluster.conj()

    # (0000 + 1111)/sqrt(2)
    psi_ghz = numqi.state.GHZ(4)
    rho_ghz = psi_ghz.reshape(-1,1) * psi_ghz.conj()

    psi_W = numqi.state.W(4)
    rho_W = psi_W.reshape(-1,1) * psi_W.conj()

    psi_dicke = numqi.state.Dicke(2, 2)
    rho_dicke = psi_dicke.reshape(-1,1) * psi_dicke.conj()

    tlist = np.linspace(0, 1, 32)
    x = np.exp(-tlist)
    ret_cluster = (3/8)*(1 + x - np.sqrt(1+(2-3*x)*x))
    ret_ghz = (1/2)*(1-np.sqrt(1-x*x))
    tmp0 = x>(2183/2667)
    ret_W = tmp0 * (37*(81*x-37)/2816) + (1-tmp0) * (3/8)*(1+x-np.sqrt(1+(2-3*x)*x))
    tmp0 = (x > 5/7)
    ret_dicke = tmp0 * (5*(3*x-1)/16) + (1-tmp0) * (5/18)*(1+2*x-np.sqrt(1+(4-5*x)*x))

    if not os.path.exists(datapath):
        model = numqi.entangle.DensityMatrixGMEModel(dim_list=[2,2,2,2], num_ensemble=32, rank=16)
        mask_diag = np.eye(rho_cluster.shape[0], dtype=np.float64)
        mask_offdiag = 1-mask_diag
        ret_model = []
        for rho in [rho_cluster, rho_ghz, rho_W, rho_dicke]:
            for t in tqdm(tlist):
                model.set_density_matrix(rho*mask_diag + rho*mask_offdiag*np.exp(-t))
                ret_model.append(numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
        ret_model = np.array(ret_model).reshape(4,-1)
        with open(datapath, 'wb') as fid:
            tmp0 = dict(tlist=tlist, ret_model=ret_model, ret_cluster=ret_cluster, ret_ghz=ret_ghz, ret_W=ret_W, ret_dicke=ret_dicke)
            pickle.dump(tmp0, fid)
    else:
        with open(datapath, 'rb') as fid:
            tmp0 = pickle.load(fid)
            ret_model = tmp0['ret_model']

    fig,ax = plt.subplots()
    ax.plot(tlist, ret_model[0], '-', label='Cluster state', color=tableau[0])
    ax.plot(tlist, ret_cluster, 'o', markerfacecolor='none', color=tableau[0])
    ax.plot(tlist, ret_model[1], '-', label='GHZ state', color=tableau[1])
    ax.plot(tlist, ret_ghz, 'o', markerfacecolor='none', color=tableau[1])
    ax.plot(tlist, ret_model[2], '-', label='W state', color=tableau[2])
    ax.plot(tlist, ret_W, 'o', markerfacecolor='none', color=tableau[2])
    ax.plot(tlist, ret_model[3], '-', label='Dicke state', color=tableau[3])
    ax.plot(tlist, ret_dicke, 'o', markerfacecolor='none', color=tableau[3])
    ax.set_xlabel('$t$')
    ax.set_ylabel(r'$E_G(t)$')
    ax.legend()
    fig.savefig(hf_data('fig_multipartite.pdf'))
    fig.savefig(hf_data('fig_multipartite.png'), dpi=200)


if __name__=='__main__':
    demo_werner_gme()
    demo_bound_entangled_state()
    demo_higher_entangled_state()
    demo_multipartite()

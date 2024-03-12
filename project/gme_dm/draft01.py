import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import numqi

np_rng = np.random.default_rng()

# linear entropy http://dx.doi.org/10.1103/PhysRevLett.114.160501


def demo_werner_convex_concave():
    alpha_list = np.linspace(0, 1, 100)
    dim = 3

    model = numqi.entangle.DensityMatrixLinearEntropyModel([dim,dim], num_ensemble=27, kind='convex')
    ret0 = []
    for alpha_i in tqdm(alpha_list):
        model.set_density_matrix(numqi.state.Werner(dim, alpha=alpha_i))
        ret0.append(numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
    ret0 = np.array(ret0)

    model = numqi.entangle.DensityMatrixLinearEntropyModel([dim,dim], num_ensemble=27, kind='concave')
    ret1 = []
    for alpha_i in tqdm(alpha_list):
        model.set_density_matrix(numqi.state.Werner(dim, alpha=alpha_i))
        ret1.append(-numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
    ret1 = np.array(ret1)


    fig,ax = plt.subplots()
    ax.axvline(1/dim, color='r')
    ax.plot(alpha_list, ret0, label='convex')
    ax.plot(alpha_list, ret1, label='concave')
    ax.legend()
    # ax.set_yscale('log')
    ax.set_xlabel('alpha')
    ax.set_ylabel('linear entropy')
    ax.set_title(f'Werner({dim})')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)


def demo_Horodecki1997_3x3_contourf():
    alist = np.linspace(0, 1, 30)
    plist = np.linspace(0.92, 1, 30)

    ret = []
    model = numqi.entangle.DensityMatrixLinearEntropyModel([3,3], num_ensemble=27, kind='convex')
    tmp0 = [(a,p) for a in alist for p in plist]
    for a,p in tqdm(tmp0):
        rho = numqi.state.get_bes3x3_Horodecki1997(a)
        model.set_density_matrix(numqi.entangle.hf_interpolate_dm(rho, alpha=p))
        ret.append(numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
    ret = np.array(ret).reshape(len(alist), len(plist))


    fig,ax = plt.subplots()
    tmp0 = np.log(np.maximum(1e-7, ret))
    hcontourf = ax.contourf(alist, plist, tmp0.T, levels=10)
    cbar = fig.colorbar(hcontourf)
    ax.set_xlabel('a')
    ax.set_ylabel('p')
    ax.set_title('manifold-opt')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)


def demo_werner3():
    alpha_list = np.linspace(0, 1, 100)
    dim = 3

    tmp0 = np.stack([numqi.state.Werner(dim, alpha=alpha_i) for alpha_i in alpha_list])
    ret = numqi.entangle.get_linear_entropy_entanglement_ppt(tmp0, (dim,dim), use_tqdm=True)

    model = numqi.entangle.DensityMatrixLinearEntropyModel([dim,dim], num_ensemble=27, kind='convex')
    ret0 = []
    for alpha_i in tqdm(alpha_list):
        model.set_density_matrix(numqi.state.Werner(dim, alpha=alpha_i))
        ret0.append(numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
    ret0 = np.array(ret0)

    fig,ax = plt.subplots()
    ax.plot(plist, ret, label='manifold-opt')
    ax.plot(plist, ret0, 'x', label='PPT')
    ax.legend()
    ax.set_xlabel('p')
    ax.set_ylabel('linear entropy')
    ax.set_yscale('log')
    ax.set_title('Werner(3)')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)


def demo_Horodecki1997_3x3():
    rho = numqi.state.get_bes3x3_Horodecki1997(0.23)
    plist = np.linspace(0.92, 1, 30)

    tmp0 = np.stack([numqi.entangle.hf_interpolate_dm(rho,alpha=p) for p in plist])
    ret = numqi.entangle.get_linear_entropy_entanglement_ppt(tmp0, (3,3), use_tqdm=True)
    # 0.0017232268486448987

    ret0 = []
    model = numqi.entangle.DensityMatrixLinearEntropyModel([3,3], num_ensemble=27, kind='convex')
    for p in tqdm(plist):
        model.set_density_matrix(numqi.entangle.hf_interpolate_dm(rho, alpha=p))
        ret0.append(numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
    ret0 = np.array(ret0)

    fig,ax = plt.subplots()
    ax.plot(plist, ret, label='manifold-opt')
    ax.plot(plist, ret0, 'x', label='PPT')
    ax.legend()
    ax.set_xlabel('p')
    ax.set_ylabel('linear entropy')
    ax.set_yscale('log')
    ax.set_title('Horodecki1997-2qutrit(0.23)')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)

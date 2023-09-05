import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import numqi


def demo_check_ABk_symmetric_extension_irrep():
    # SDP dA,dB=2,bosonic extension
    # k=8: 0.2s
    # k=16: 1.7s
    # k=24: 7.5s
    dimA = 2
    dimB = 2
    kext = 8
    for _ in tqdm(range(10)):
        # if kext<=5:
        #     rho_ABk = rand_ABk_density_matrix(dimA, dimB, kext)
        #     rho_AB = np.trace(rho_ABk.reshape(dimA*dimB,dimB**(kext-1),dimA*dimB,dimB**(kext-1)), axis1=1, axis2=3)
        # else:
        #     rho_AB = numqi.random.rand_separable_dm(dimA, dimB)
        rho_AB = numqi.random.rand_separable_dm(dimA, dimB)
        has_kext = numqi.entangle.check_ABk_symmetric_extension(rho_AB, (dimA,dimB), kext)
        assert has_kext


def demo_SymmetricExtABkIrrepModel():
    # gradient descent dA,dB=2,bosonic extension
    # k=128: 2.8s
    # k=512: 140s
    dimA = 2
    dimB = 2
    kext = 128
    ret = []
    threshold = 1e-7
    for _ in tqdm(range(30)):
        rho_AB = numqi.random.rand_separable_dm(dimA, dimB)
        model = numqi.entangle.SymmetricExtABkIrrepModel(dimA, dimB, kext, use_cholesky=True)
        model.set_dm_target(rho_AB)
        theta_optim = numqi.optimize.minimize(model, theta0='uniform', num_repeat=3, tol=threshold/100, early_stop_threshold=threshold, print_every_round=0, print_freq=-1)
        if theta_optim.fun > threshold:
            numqi.optimize.minimize_adam(model, theta0='uniform', num_step=10000, early_stop_threshold=threshold)
        with torch.no_grad():
            ret.append(model())
    assert all(x<1e-7 for x in ret)


def demo_get_ABk_symmetric_extension_ree_werner():
    dim = 3
    kext = 6
    num_point = 30
    alpha_kext_boundary = (kext+dim**2-dim)/(kext*dim+dim-1)
    alpha_list = np.linspace(0, 1, num_point, endpoint=False) #alpha=1 is not stable

    ret_analytical = []
    dm_kext_boundary = numqi.entangle.get_werner_state(dim,alpha_kext_boundary)
    for alpha_i in alpha_list:
        if alpha_i <= alpha_kext_boundary:
            ret_analytical.append(0)
        else:
            dm0 = numqi.entangle.get_werner_state(dim, alpha_i)
            ret_analytical.append(numqi.utils.get_relative_entropy(dm0, dm_kext_boundary))
    ret_analytical = np.array(ret_analytical)

    dm_list = [numqi.entangle.get_werner_state(dim, x) for x in alpha_list]
    ret0 = numqi.entangle.get_ABk_symmetric_extension_ree(dm_list, (dim,dim), kext, use_ppt=False, use_tqdm=True)

    fig,ax = plt.subplots()
    ax.plot(alpha_list, ret_analytical, label='analytical')
    ax.plot(alpha_list, ret0, 'x', label='irrep SDP')
    ax.legend()
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'relative entropy with respect to k-ext')
    ax.set_title(f'Werner({dim}) kext={kext}')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)

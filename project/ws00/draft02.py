import numpy as np
import matplotlib.pyplot as plt
import torch

import numqi

def demo_renyi_entropy():
    dim = 4
    rho = numqi.random.rand_density_matrix(dim)

    ret_von = numqi.utils.get_von_neumann_entropy(rho)
    # alpha_list = 1 + np.linspace(0, 1, 30)[1:]**3
    alpha01_list = np.linspace(0, 1, 30)[1:-1]
    ret01_renyi = np.array([numqi.utils.get_Renyi_entropy(rho,x) for x in alpha01_list])
    alpha12_list = np.linspace(1, 10, 50)[1:]
    ret12_renyi = np.array([numqi.utils.get_Renyi_entropy(rho,x) for x in alpha12_list])

    fig,ax = plt.subplots()
    ax.plot([0], [np.log(dim)], 'x', color='red')
    ax.plot(alpha01_list, ret01_renyi, '.')
    ax.plot([1], ret_von, 'x', color='red')
    ax.plot(alpha12_list, ret12_renyi, '.')
    ax.axhline(-np.log(np.linalg.eigvalsh(rho)[-1]), color='red', linestyle='-')
    ax.grid()
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)


def demo_alpha_entropic_inequality():
    # https://arxiv.org/abs/2103.07712 (eq23)
    alpha = 1.2
    dimA = 3
    dimB = 4
    rhoAB = numqi.random.rand_separable_dm(dimA, dimB, k=2)
    tmp0 = rhoAB.reshape(dimA, dimB, dimA, dimB)
    rdmA = np.trace(tmp0, axis1=1, axis2=3)
    rdmB = np.trace(tmp0, axis1=0, axis2=2)

    entropy_AB = numqi.utils.get_Renyi_entropy(rhoAB, alpha)
    entropy_A = numqi.utils.get_Renyi_entropy(rdmA, alpha)
    entropy_B = numqi.utils.get_Renyi_entropy(rdmB, alpha)
    print(entropy_AB, entropy_A, entropy_B)
    assert entropy_AB>=entropy_A
    assert entropy_AB>=entropy_B

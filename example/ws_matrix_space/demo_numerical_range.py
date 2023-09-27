import numpy as np
import matplotlib.pyplot as plt

import numqi

np_rng = np.random.default_rng()
hf_randc = lambda *x: np_rng.normal(size=x) + 1j*np_rng.normal(size=x)


def demo_matrix_numerical_range():
    N0 = 5
    matA = np_rng.normal(size=(N0,N0)) + 1j*np_rng.normal(size=(N0,N0))
    alpha_list = np_rng.uniform(0, 2*np.pi, size=5)

    boundary = numqi.matrix_space.get_matrix_numerical_range(matA, num_point=200)
    max_list = []
    min_list = []
    for alpha_i in alpha_list:
        min_list.append(numqi.matrix_space.get_matrix_numerical_range_along_direction(matA, alpha_i, kind='min')[0])
        max_list.append(numqi.matrix_space.get_matrix_numerical_range_along_direction(matA, alpha_i, kind='max')[0])
    min_list = np.array(min_list)
    max_list = np.array(max_list)

    fig,ax = plt.subplots()
    ax.plot(boundary.real, boundary.imag)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for ind0,alpha_i in enumerate(alpha_list):
        tmp0 = np.sqrt(max(abs(xlim[0]), abs(xlim[1]))**2 + max(abs(ylim[0]), abs(ylim[1]))**2)
        ax.plot([0,tmp0*np.cos(alpha_i)], [0, tmp0*np.sin(alpha_i)], 'k')
        ax.plot([0,tmp0*np.cos(alpha_i+np.pi)], [0, tmp0*np.sin(alpha_i+np.pi)], 'k:')
    tmp0 = max_list*np.exp(1j*alpha_list)
    ax.plot(tmp0.real, tmp0.imag, 'rx', markersize=10)
    tmp0 = min_list*np.exp(1j*alpha_list)
    ax.plot(tmp0.real, tmp0.imag, 'r+', markersize=10)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig('tbd00.png')
    # fig.savefig('data/demo_matrix_space_numerical_range.png')

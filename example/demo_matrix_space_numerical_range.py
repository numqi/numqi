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


def demo_normal_matrix_numerical_range():
    # NR(normal matrix) is the convex hull of eigenvalues
    N0 = 4
    matU = numqi.random.rand_unitary_matrix(N0)
    EVL = hf_randc(N0)
    matA = (matU * EVL) @ matU.T.conj()
    boundary = numqi.matrix_space.get_matrix_numerical_range(matA, num_point=200)
    fig,ax = plt.subplots()
    ax.plot(boundary.real, boundary.imag)
    ind0 = slice(None,None,5)
    ax.plot(boundary.real[ind0], boundary.imag[ind0], '.', markersize=10)
    ax.plot(EVL.real, EVL.imag, '+')
    for ind0,x in enumerate(EVL):
        ax.text(x.real, x.imag, str(ind0))
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)
    plt.close(fig)

def demo_abnormal_matrix_numerical_range():
    # NR(abnormal matrix) is some mixture of the normal part and abnormal part
    seed = np_rng.integers(0, 2**32-1)
    np_rng = np.random.default_rng(seed=seed) #1457230309
    hf_randc = lambda *x: np_rng.normal(size=x) + 1j*np_rng.normal(size=x)
    N0 = 5
    N1 = 2
    matU = numqi.random.rand_unitary_matrix(N0, seed=np_rng)
    EVL = hf_randc(N1)
    matIN = hf_randc(N0-N1, N0-N1) #abnormal matrix
    tmp0 = (matU[:,:N1]*EVL) @ matU[:,:N1].T.conj()
    tmp1 = matU[:,N1:] @ matIN @ matU[:,N1:].T.conj()
    matA0 = tmp0*0 + tmp1
    matA1 = tmp0 + tmp1
    boundary0 = numqi.matrix_space.get_matrix_numerical_range(matA0, num_point=200)
    boundary1 = numqi.matrix_space.get_matrix_numerical_range(matA1, num_point=200)
    # tmp0 = np.linalg.eigvalsh((matIN @ matIN.T.conj() - matIN.T.conj() @ matIN))

    fig,ax = plt.subplots()
    ax.plot(boundary0.real, boundary0.imag, label='NR(abnormal-part of $A$)')
    ax.plot(boundary1.real, boundary1.imag, label='NR($A$)')
    ind0 = slice(None,None)
    ax.plot(boundary1.real[ind0], boundary1.imag[ind0], '.', markersize=10)
    ax.plot(EVL.real, EVL.imag, '+', label='eigenvalue($A$)')
    for ind0,x in enumerate(EVL):
        ax.text(x.real, x.imag, str(ind0))
    ax.legend()
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)
    plt.close(fig)

# TODO demo joint algebraic numerical range

import itertools
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

import numqi

cp_tableau = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3', '#937860', '#da8bc3', '#8c8c8c', '#ccb974', '#64b5cd']

def hf_cos_sin_wrapper(order):
    assert order>=1
    def hf0(theta):
        theta = np.asarray(theta)
        assert theta.ndim<=1
        if theta.ndim==1:
            tmp0 = np.arange(order+1)[:,np.newaxis]*theta
            ret = np.concatenate([np.cos(tmp0), np.sin(tmp0[1:])], axis=1)
        else:
            tmp0 = np.arange(order+1)*theta
            ret = np.concatenate([np.cos(tmp0), np.sin(tmp0[1:])], axis=0)
        return ret
    return hf0


def hf_is_similar(x, y, zero_eps=0.001):
    assert (x.ndim==1) and (x.shape==y.shape) and (x.shape[0]%2==1)
    xc = x[:(x.shape[0]//2 + 1)].copy().astype(np.complex128)
    xc[1:] += 1j*x[(x.shape[0]//2 + 1):]
    yc = y[:(y.shape[0]//2 + 1)].copy().astype(np.complex128)
    yc[1:] += 1j*y[(y.shape[0]//2 + 1):]
    if np.abs(np.abs(xc)-np.abs(yc)).max() > zero_eps:
        return False
    if (len(xc)==1) or (np.abs(xc[1:]).max() < zero_eps):
        return True
    ind0 = np.argmax(np.abs(xc[1:])) + 1
    phase = np.angle(yc[ind0]/xc[ind0])/ind0
    tmp0 = np.exp(1j*np.arange(len(xc))*phase)
    if np.abs(xc*tmp0-yc).max() < zero_eps:
        return True
    yc = yc.conj()
    phase = np.angle(yc[ind0]/xc[ind0])/ind0
    tmp0 = np.exp(1j*np.arange(len(xc))*phase)
    if np.abs(xc*tmp0-yc).max() < zero_eps:
        return True
    return False

def get_square(N0:int):
    assert N0>=2
    tmp0 = int(np.ceil((np.sqrt(4*N0+1) - 1)/2))
    tmp1 = int(np.ceil(np.sqrt(N0)))
    if tmp0==tmp1:
        ret = tmp1,tmp1
    else:
        ret = tmp0,tmp0+1
    return ret

def demo_classify_all_cross_section():
    # https://arxiv.org/abs/quant-ph/0301152
    cos_sin_order = 4
    dim = 4
    gm_list = numqi.gellmann.all_gellmann_matrix(dim)
    index01_set = list(itertools.combinations(list(range(dim*dim-1)), 2))

    theta_list = np.linspace(0, 2*np.pi, 301)
    hf_cos_sin = hf_cos_sin_wrapper(cos_sin_order)
    index_to_moment = dict()
    index_to_data = dict()
    for ind0,ind1 in index01_set:
        print(ind0,ind1)
        tmp0 = np.eye(dim)/dim + gm_list[ind0]
        tmp1 = np.eye(dim)/dim + gm_list[ind1]
        hf_plane = numqi.entangle.get_density_matrix_plane(tmp0, tmp1)[1]
        index_to_data[(ind0,ind1)] = np.array([2*numqi.entangle.get_density_matrix_boundary(hf_plane(x))[1] for x in theta_list])
        hf0 = lambda x: 2*numqi.entangle.get_density_matrix_boundary(hf_plane(x))[1] * hf_cos_sin(x)
        index_to_moment[(ind0,ind1)] = scipy.integrate.quad_vec(hf0, 0, 2*np.pi, epsrel=1e-8)[0]
    group_list = []
    for k0,v0 in index_to_moment.items():
        for group in group_list:
            if hf_is_similar(index_to_moment[group[0]], v0):
                group.append(k0)
                break
        else:
            group_list.append([k0])

    r_inner = np.sqrt(2/(dim*dim-dim))
    r_outter = np.sqrt(2*(dim-1)/dim)
    fig,tmp0 = plt.subplots(*get_square(len(group_list)), figsize=(9,9))
    ax_list = [y for x in tmp0 for y in x]
    for ax,group in zip(ax_list, group_list):
        for ind0,ind1 in group:
            ax.plot(index_to_data[(ind0,ind1)]*np.cos(theta_list), index_to_data[(ind0,ind1)]*np.sin(theta_list), color=cp_tableau[0])
        ax.plot(r_inner*np.cos(theta_list), r_inner*np.sin(theta_list), color=cp_tableau[1], linestyle='dashed')
        ax.plot(r_outter*np.cos(theta_list), r_outter*np.sin(theta_list), color=cp_tableau[1], linestyle='dashed')
        ax.set_aspect('equal')
    for x in ax_list[len(group_list):]:
        x.axis('off')
    fig.suptitle(f'Cross Section of {dim}-Level Systems')
    fig.tight_layout()
    fig.savefig('tbd01.png', dpi=200)
    # fig.savefig('qudit5_cross.png', dpi=200)

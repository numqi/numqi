import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.special

import numqi

tableau = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
# import seaborn as sns
# sns.palplot(sns.color_palette(tableau))


def demo_multipartite_Maciej2019_gm():
    num_party = 3
    bipartition_list = numqi.matrix_space.get_bipartition_list(num_party)
    # [tuple(range(x)) for x in range(1,num_party)] is not correct since B^k is not symmetric
    dimB_list = list(range(3,6))
    theta_list = np.linspace(0, np.pi, 20)

    ret_analytical = np.stack([numqi.matrix_space.get_GM_Maciej2019(x, theta_list) for x in dimB_list])
    ret_ppt = []
    for dimB in dimB_list:
        dim_list = [2]+[dimB]*(num_party-1)
        for theta in tqdm(theta_list, desc=f'[ppt][d={dimB}]'):
            matrix_subspace = numqi.matrix_space.get_GES_Maciej2019(dimB, theta=theta, num_party=num_party)
            ret_ppt.append(numqi.matrix_space.get_generalized_geometric_measure_ppt(matrix_subspace, dim_list, bipartition_list))
    ret_ppt = np.array(ret_ppt).reshape(len(dimB_list), len(theta_list))

    ret_gd = []
    gd_kwargs = dict(theta0='uniform', num_repeat=3, tol=1e-12, print_every_round=0)
    for dimB in dimB_list:
        dim_list = [2]+[dimB]*(num_party-1)
        model_list = [numqi.matrix_space.DetectCanonicalPolyadicRankModel(dim_list, rank=1, bipartition=x) for x in bipartition_list]
        for theta in tqdm(theta_list, desc=f'[gd][d={dimB}]'):
            matrix_subspace = numqi.matrix_space.get_GES_Maciej2019(dimB, theta=theta, num_party=num_party)
            for model in model_list:
                model.set_target(matrix_subspace)
                ret_gd.append(numqi.optimize.minimize(model, **gd_kwargs).fun)
    ret_gd = np.array(ret_gd).reshape(len(dimB_list), len(theta_list), len(bipartition_list))

    fig,ax = plt.subplots()
    for ind0 in range(len(dimB_list)):
        ax.plot(theta_list, ret_analytical[ind0], '-', label=f'd={dimB_list[ind0]}', color=tableau[ind0])
        ax.plot(theta_list, ret_gd[ind0].min(axis=1), 'o', markersize=6, markerfacecolor='none', markeredgecolor=tableau[ind0])
        ax.plot(theta_list, ret_ppt[ind0], 'x', markersize=6, color=tableau[ind0])
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$E_2(\mathcal{S}_{2\times d\times d}^{\theta}$)')
    ax.legend()
    fig.tight_layout()
    # fig.savefig('data/Maciej2019_multipartite_gm.pdf')


def demo_bipartite_Maciej2019_gm():
    dimB_list = list(range(3,8))
    theta_list = np.linspace(0, np.pi, 20)

    ret_analytical = np.stack([numqi.matrix_space.get_GM_Maciej2019(x, theta_list) for x in dimB_list])
    ret_ppt = []
    ret_gd = []
    gd_kwargs = dict(theta0='uniform', num_repeat=3, tol=1e-12, print_every_round=0)
    for dimB in tqdm(dimB_list):
        model = numqi.matrix_space.DetectCanonicalPolyadicRankModel((2,dimB), rank=1)
        for theta in theta_list:
            matrix_subspace = numqi.matrix_space.get_GES_Maciej2019(dimB, theta=theta)
            ret_ppt.append(numqi.matrix_space.get_generalized_geometric_measure_ppt(matrix_subspace, [2,dimB]))
            model.set_target(matrix_subspace)
            ret_gd.append(numqi.optimize.minimize(model, **gd_kwargs).fun)
    ret_gd = np.array(ret_gd).reshape(len(dimB_list), len(theta_list))
    ret_ppt = np.array(ret_ppt).reshape(len(dimB_list), len(theta_list))

    fig,ax = plt.subplots()
    for ind0 in range(len(dimB_list)):
        ax.plot(theta_list, ret_analytical[ind0], '-', label=f'd={dimB_list[ind0]}', color=tableau[ind0])
        ax.plot(theta_list, ret_gd[ind0], 'o', markersize=6, markerfacecolor='none', markeredgecolor=tableau[ind0])
        ax.plot(theta_list, ret_ppt[ind0], 'x', markersize=6, color=tableau[ind0])
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$E_2(\mathcal{S}_{2\times d}^{\theta}$)')
    ax.legend()
    fig.tight_layout()
    # fig.savefig('data/Maciej2019_bipartite_gm.pdf')


def demo_Wstate_border_rank():
    num_qubit = 5
    dim_list = [2]*num_qubit
    Wstate = numqi.state.W(num_qubit).reshape(dim_list)
    model = numqi.matrix_space.DetectCanonicalPolyadicRankModel(dim_list, rank=2)
    model.set_target(Wstate)
    kwargs = dict(theta0='uniform', tol=1e-14, num_repeat=3, print_every_round=1, early_stop_threshold=1e-14)
    theta_optim = numqi.optimize.minimize(model, **kwargs)
    # [round=0] min(f)=4.065757175375495e-09, current(f)=4.065757175375495e-09
    # [round=1] min(f)=4.065757175375495e-09, current(f)=9.057507299736756e-09
    # [round=2] min(f)=4.065757175375495e-09, current(f)=6.180069234140717e-09


def demo_qubit_dicke_state_gm():
    num_qubit = 7
    dim_list = [2]*num_qubit
    dicke_basis = numqi.dicke.get_dicke_basis(num_qubit, dim=2)
    model = numqi.matrix_space.DetectCanonicalPolyadicRankModel(dim_list, rank=1)
    kwargs = dict(theta0='uniform', tol=1e-14, num_repeat=3, print_every_round=0, early_stop_threshold=1e-14)
    ret0 = []
    dicke_k_list = np.arange(num_qubit+1)
    for dicke_k in dicke_k_list:
        model.set_target(dicke_basis[dicke_k].reshape(dim_list))
        ret0.append(numqi.optimize.minimize(model, **kwargs).fun)
    ret0 = np.array(ret0)
    ret_ = numqi.state.get_qubit_dicke_state_gm(num_qubit, dicke_k_list)
    print(ret0, np.abs(ret0-ret_).max())


def demo_qubit_dicke_state_border_rank():
    num_qubit = 5
    dicke_k = 2
    dim_list = [2]*num_qubit
    dicke_basis = numqi.dicke.get_dicke_basis(num_qubit, dim=2)
    kwargs = dict(theta0='uniform', tol=1e-14, num_repeat=3, print_every_round=0, early_stop_threshold=1e-14)
    for rank in range(1, dicke_k+3):
        model = numqi.matrix_space.DetectCanonicalPolyadicRankModel(dim_list, rank)
        model.set_target(dicke_basis[dicke_k].reshape(dim_list))
        theta_optim = numqi.optimize.minimize(model, **kwargs)
        print(f'E_{rank+1}: {theta_optim.fun}')
    # E_2: 0.6544
    # E_3: 0.3077088939166458
    # E_4: 1.1018637668946951e-08
    # E_5: 1.4988010832439613e-14

    all_data = dict()
    kwargs = dict(theta0='uniform', tol=1e-14, num_repeat=3, print_every_round=0, early_stop_threshold=1e-14)
    for num_qubit in range(2, 7):
        dicke_basis = numqi.dicke.get_dicke_basis(num_qubit, dim=2)
        dim_list = [2]*num_qubit
        for dicke_k in tqdm(range(1, num_qubit//2+1), desc=f'[n={num_qubit}]'):
            tmp0 = []
            rlist = np.arange(2, dicke_k+3)
            for r in rlist:
                model = numqi.matrix_space.DetectCanonicalPolyadicRankModel(dim_list, r-1)
                model.set_target(dicke_basis[dicke_k].reshape(dim_list))
                tmp0.append(numqi.optimize.minimize(model, **kwargs).fun)
            all_data[(num_qubit, dicke_k)] = rlist, np.array(tmp0)

    fig,ax = plt.subplots()
    n_to_alpha = dict(zip([2,3,4,5,6,7], np.linspace(0.4, 1, 5)))
    k_to_color = dict(zip([1,2,3], [tableau[0],tableau[1],tableau[3]]))
    tmp0 = sorted(all_data.items(), key=lambda x: x[0])
    for (n,k),(rlist,Er) in tmp0:
        ax.plot(rlist, Er, 'o-', label=f'n={n}, k={k}', color=k_to_color[k], alpha=n_to_alpha[n])
    ax.set_ylim([-0.03, 0.72])
    ax.set_xticks(np.arange(2, 6))
    ax.set_xlabel('$r$')
    ax.set_ylabel(r'$E_r(|D_n^k \rangle)$')
    ax.legend()
    fig.tight_layout()
    # fig.savefig('data/qubit_dicke_state_border_rank.pdf')

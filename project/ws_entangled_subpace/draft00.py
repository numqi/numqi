import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.linalg
import plotly
import plotly.subplots

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


def demo_ces():
    ## gradient descent
    case_list = [(2,2,2), (2,2,4),(2,2,6),(2,3,4),(2,3,6),(2,3,8),(3,3,6),(3,3,8),(3,4,7),(4,4,7),(4,5,10)]
    kwargs = dict(theta0='uniform', num_repeat=3, tol=1e-12, print_every_round=0)
    for dimA,dimB,dimC in case_list:
        np_list = numqi.matrix_space.get_completed_entangled_subspace((dimA, dimB, dimC), kind='quant-ph/0409032')[0]
        t0 = time.time()
        model = numqi.matrix_space.DetectCanonicalPolyadicRankModel((dimA, dimB, dimC), rank=1)
        model.set_target(np_list)
        theta_optim1 = numqi.optimize.minimize(model, **kwargs)
        tmp0 = time.time()-t0
        print(f'[{dimA}x{dimB}x{dimC}][{tmp0:.2f}s], loss(r=1)= {theta_optim1.fun:.2e}')
    # [2x2x2][0.07s], loss(r=1)= 2.50e-01
    # [2x2x4][0.08s], loss(r=1)= 4.50e-02
    # [2x2x6][0.10s], loss(r=1)= 1.23e-02
    # [2x3x4][0.08s], loss(r=1)= 1.41e-02
    # [2x3x6][0.15s], loss(r=1)= 2.86e-03
    # [2x3x8][0.21s], loss(r=1)= 7.62e-04
    # [3x3x6][0.18s], loss(r=1)= 7.20e-04
    # [3x3x8][0.32s], loss(r=1)= 1.57e-04
    # [3x4x7][0.40s], loss(r=1)= 7.98e-05
    # [4x5x10][2.92s], loss(r=1)= 3.54e-07

    ## PPT
    case_list = [(2,2,2), (2,2,4),(2,2,6),(2,3,4),(2,3,6),(2,3,8),(3,3,6),(3,3,8),(3,4,7)]
    for dimA,dimB,dimC in case_list:
        np_list = numqi.matrix_space.get_completed_entangled_subspace((dimA, dimB, dimC), kind='quant-ph/0409032')[0]
        t0 = time.time()
        E2_ppt = numqi.matrix_space.get_geometric_measure_ppt(np_list, [dimA,dimB,dimC])
        tmp0 = time.time()-t0
        print(f'[{dimA}x{dimB}x{dimC}][{tmp0:.3f}s], E2_ppt= {E2_ppt:.2e}')
    # [2x2x2][0.092s], loss(r=1)= 2.00e-01
    # [2x2x4][0.164s], loss(r=1)= 4.49e-02
    # [2x2x6][0.331s], loss(r=1)= 1.23e-02
    # [2x3x4][0.318s], loss(r=1)= 1.41e-02
    # [2x3x6][1.071s], loss(r=1)= 2.86e-03
    # [2x3x8][2.103s], loss(r=1)= 7.62e-04
    # [3x3x6][2.709s], loss(r=1)= 7.20e-04
    # [3x3x8][8.554s], loss(r=1)= 1.57e-04
    # [3x4x7][18.273s], loss(r=1)= 7.98e-05

    ## hierarchy method
    case_list = [(2,2,2,2), (2,2,4,2), (2,2,6,2), (2,3,4,3)]
    info_list = []
    for dimA,dimB,dimC,kmax in case_list:
        np_list = numqi.matrix_space.get_completed_entangled_subspace((dimA, dimB, dimC), kind='quant-ph/0409032')[0]
        for k in range(1, kmax+1):
            t0 = time.time()
            ret = numqi.matrix_space.is_ABC_completely_entangled_subspace(np_list, hierarchy_k=k)
            info_list.append((dimA,dimB,dimC,k, ret, time.time()-t0))
        print(f'[{dimA}x{dimB}x{dimC}] {info_list[-2][-2]}@(k={kmax-1}) {info_list[-1][-2]}@(k={kmax}) time(k={kmax})={info_list[-1][-1]:.2f}s')
    ## mac-studio 20231107
    # [2x2x2] False@(k=1) True@(k=2) time(k=2)=0.03s
    # [2x2x4] False@(k=1) True@(k=2) time(k=2)=0.63s
    # [2x2x6] False@(k=1) True@(k=2) time(k=2)=0.76s
    # [2x3x4] False@(k=2) True@(k=3) time(k=3)=9.16s

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
    ret_ = numqi.state.get_qubit_dicke_state_GME(num_qubit, dicke_k_list)
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


def _rand_hermtian_matrix_with_trace_norm(d:int, norm:float, seed=None):
    np_rng = numqi.random.get_numpy_rng(seed)
    tmp0 = np_rng.normal(size=(d, d)) + 1j*np_rng.normal(size=(d, d))
    ret = tmp0 + tmp0.T.conj()
    EVL = np.linalg.eigvalsh(ret)
    ret *= norm / np.sum(np.abs(EVL))
    return ret


def demo_robustness():
    kwargs = dict(theta0='uniform', num_repeat=3, tol=1e-12, print_every_round=0)
    norm_list = np.linspace(0.1, 5, 20)
    num_sample = 1000
    theta_list = [np.pi/2, np.pi/4, np.pi/6]
    dimB = 3

    model = numqi.matrix_space.DetectCanonicalPolyadicRankModel((2,dimB), rank=1)
    ret_list = []
    for theta_i in theta_list:
        matrix_subspace = numqi.matrix_space.get_GES_Maciej2019(dimB, theta=theta_i)
        for norm_i in tqdm(norm_list, desc=f'[theta={theta_i:.2f}]'):
            for _ in range(num_sample):
                matU = scipy.linalg.expm(-1j*_rand_hermtian_matrix_with_trace_norm(2*dimB, norm_i))
                tmp0 = np.einsum(matU, [0,1], matrix_subspace.reshape(-1,2*dimB), [2,1], [2,0], optimize=True).reshape(-1,2,dimB)
                model.set_target(tmp0)
                ret_list.append(numqi.optimize.minimize(model, **kwargs).fun)
    ret_list = np.array(ret_list).reshape(len(theta_list), len(norm_list), num_sample)

    # with open('data/robustness.pkl', 'wb') as fid:
    #     pickle.dump(dict(norm_list=norm_list, ret_list=ret_list), fid)

    fig,ax = plt.subplots()
    ax.plot(norm_list, ret_list[0].min(axis=1), 'o-', label=r'$\theta=\pi/2$', color=tableau[0])
    ax.plot(norm_list, ret_list[1].min(axis=1), 'o-', label=r'$\theta=\pi/4$', color=tableau[1])
    ax.plot(norm_list, ret_list[2].min(axis=1), 'o-', label=r'$\theta=\pi/6$', color=tableau[4])
    ax.set_xlabel(r'$\|H\|_{tr}$')
    ax.set_ylabel(r'Minimum value of $E_2$')
    ax.legend()
    fig.tight_layout()
    fig.savefig('data/robustness.pdf')


def demo_Wlike_state_gme():
    phi_list = np.linspace(0, np.pi, 20)
    theta_list = np.linspace(0, 2 * np.pi, 40)

    ret_analytical = []
    ret_gd = []
    abc_list = np.array([[(np.sin(p)*np.cos(t), np.sin(p)*np.sin(t), np.cos(p)) for p in phi_list] for t in theta_list])
    kwargs = dict(theta0='uniform', tol=1e-14, num_repeat=3, print_every_round=0, early_stop_threshold=1e-14)
    model = numqi.matrix_space.DetectCanonicalPolyadicRankModel([2,2,2], rank=1)
    for x in tqdm(abc_list.reshape(-1,3)):
        ret_analytical.append(numqi.state.get_Wtype_state_GME(*x))
        model.set_target(numqi.state.Wtype(x).reshape(2,2,2))
        ret_gd.append(numqi.optimize.minimize(model, **kwargs).fun)
    ret_analytical = np.array(ret_analytical).reshape(len(theta_list), -1)
    ret_gd = np.array(ret_gd).reshape(len(theta_list), -1)

    error = np.abs(ret_analytical-ret_gd)
    print('maximum absolute error:', error.max())

    fig = plotly.subplots.make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'surface'}]])
    fig.add_trace(plotly.graph_objects.Surface(
        x=abc_list[:,:,0],  y=abc_list[:,:,1],  z=abc_list[:,:,2],  surfacecolor=ret_analytical,
        colorscale='balance', colorbar=dict(x=0.45, xpad=0),  showscale=True,  name='analytical_result'), row=1, col=1)
    fig.add_trace(plotly.graph_objects.Surface(
        x=abc_list[:,:,0], y=abc_list[:,:,1], z=abc_list[:,:,2], surfacecolor=error,
        colorscale='Viridis', colorbar=dict(x=1, xpad=0), showscale=True, name='absolute_error'), row=1, col=2)
    fig.update_scenes(
        xaxis=dict(title_text="a", tickfont=dict(size=10)),
        yaxis=dict(title_text="b", tickfont=dict(size=10)),
        zaxis=dict(title_text="c", tickfont=dict(size=10), range=[-0.95, 1]),
        camera_eye=dict(x=1.5, y=1.5, z=1.5)
    )
    tmp0 = dict(l=50, r=50, b=100, t=100, pad=10) # left margin, right margin, bottom margin, top margin, padding
    fig.update_layout(autosize=False, width=900, height=500, margin=tmp0)
    # fig.show() # show in jupyter
    fig.write_image("tbd00.png") # pip install -U kaleido



if __name__=='__main__':
    demo_bipartite_Maciej2019_gm()
    # demo_Wstate_border_rank()
    # demo_qubit_dicke_state_border_rank()
    # demo_qubit_dicke_state_border_rank()
    # demo_robustness()

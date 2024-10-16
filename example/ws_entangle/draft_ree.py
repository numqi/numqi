import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import numqi

tableau = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']

np_rng = np.random.default_rng()


def demo_ree_gellmann_random_dm():
    dimA = 4
    dimB = 4
    kext = 16

    dm_target = numqi.random.rand_density_matrix(dimA*dimB, kind='haar')
    dm_norm = numqi.gellmann.dm_to_gellmann_norm(dm_target)
    beta_u = numqi.entangle.get_density_matrix_boundary(dm_target)[1]
    beta_list = np.linspace(0, beta_u, 50)
    dm_target_list = [numqi.utils.hf_interpolate_dm(dm_target,beta=x,dm_norm=dm_norm) for x in beta_list]

    beta_ppt = numqi.entangle.get_ppt_boundary(dm_target, (dimA, dimB))[1]

    kwargs = dict(tol=1e-12, num_repeat=1, print_every_round=0)
    model = numqi.entangle.AutodiffCHAREE((dimA, dimB), distance_kind='ree') #None meanns 2*dimA*dimB
    ree_cha = []
    for dm_target_i in tqdm(dm_target_list):
        model.set_dm_target(dm_target_i)
        ree_cha.append(numqi.optimize.minimize(model, **kwargs).fun)
    ree_cha = np.array(ree_cha)

    model = numqi.entangle.AutodiffCHAREE((dimA, dimB), distance_kind='gellmann')
    gellmann_cha = []
    for dm_target_i in tqdm(dm_target_list):
        model.set_dm_target(dm_target_i)
        ree_cha.append(numqi.optimize.minimize(model, **kwargs).fun)
        model.set_dm_target(dm_target_i)
        gellmann_cha.append(numqi.optimize.minimize(model, **kwargs).fun)
    gellmann_cha = np.array(gellmann_cha)

    ree_pureb = []
    model = numqi.entangle.PureBosonicExt(dimA, dimB, kext, distance_kind='ree')
    for dm_target_i in tqdm(dm_target_list):
        model.set_dm_target(dm_target_i)
        ree_pureb.append(numqi.optimize.minimize(model, **kwargs).fun)
    ree_pureb = np.array(ree_pureb)

    gellmann_pureb = []
    model = numqi.entangle.PureBosonicExt(dimA, dimB, kext, distance_kind='gellmann')
    for dm_target_i in tqdm(dm_target_list):
        model.set_dm_target(dm_target_i)
        gellmann_pureb.append(numqi.optimize.minimize(model, **kwargs).fun)
    gellmann_pureb = np.array(gellmann_pureb)

    fig,ax = plt.subplots()
    ax.plot(beta_list, ree_cha, color=tableau[0], label='CHA REE')
    ax.plot(beta_list, gellmann_cha, linestyle='dashed', color=tableau[0], label='CHA Gell-Mann')
    ax.plot(beta_list, ree_pureb, color=tableau[1], label=f'PureB(k={kext}) REE')
    ax.plot(beta_list, gellmann_pureb, linestyle='dashed', color=tableau[1], label=f'PureB(k={kext}) Gell-Mann')
    ax.axvline(beta_ppt, label=r'$\beta_{PPT}$', color=tableau[2])
    ax.legend()
    ax.set_ylim(1e-13, None)
    ax.set_yscale('log')
    ax.set_xlabel(r'$\beta$')
    ax.set_title(f'random density matrix {dimA}x{dimB}')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)


def demo_werner_gellmann():
    dim = 2
    kext_list = np.arange(1, 9)
    dm_norm = numqi.gellmann.dm_to_gellmann_norm(numqi.state.Werner(dim, 1))
    alpha_list = np.linspace(0, 1, 100, endpoint=False) # alpha=1 is unstable for matrix logarithm
    beta_list = alpha_list * dm_norm
    dm_target_list = [numqi.state.Werner(dim, x) for x in alpha_list]

    dm_target = numqi.random.rand_density_matrix(dim*dim, kind='haar')
    dm_target = np.diag([1,0,0,1])/2
    dm_target = numqi.state.Isotropic(dim, 1)
    beta_u = numqi.entangle.get_density_matrix_boundary(dm_target)[1]
    beta_list = np.linspace(0, beta_u, 100)
    # z0 = numqi.utils.hf_interpolate_dm(dm_target, beta=beta_u)
    numqi.entangle.is_ppt(numqi.utils.hf_interpolate_dm(dm_target, beta=beta_u), (dim,dim))
    dm_target_list = [numqi.utils.hf_interpolate_dm(dm_target, beta=x) for x in beta_list]

    z0 = []
    for kext in kext_list:
        model = numqi.entangle.PureBosonicExt(dim, dim, kext, distance_kind='gellmann')
        for dm_target_i in tqdm(dm_target_list):
            model.set_dm_target(dm_target_i)
            z0.append(numqi.optimize.minimize(model, tol=1e-12, num_repeat=1, print_every_round=0).fun)
    z0 = np.array(z0).reshape(len(kext_list), len(beta_list))

    # beta_kext_boundary = (kext_list+dim**2-dim)/(kext_list*dim+dim-1)*dm_norm
    fig,ax = plt.subplots()
    for ind0 in range(len(kext_list)):
        ax.plot(beta_list, z0[ind0], label=f'k={kext_list[ind0]}')
        # ax.axvline(beta_kext_boundary[ind0], color='k', linestyle='dashed')
    ax.legend()
    ax.set_ylim(1e-13, None)
    # ax.set_yscale('log')
    ax.set_title('werner(2)')
    ax.set_xlabel('beta')
    ax.set_ylabel('Gell-Mann distance')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)


def demo_cha_gellmann():
    dimA = 3
    dimB = 3
    num_state_list = [12,13,14,2*dimA*dimB,15]

    alpha_list = np.linspace(0, 1, 50, endpoint=False) # alpha=1 is unstable for matrix logarithm
    dm_target_list = [numqi.state.Werner(dimA, x) for x in alpha_list]
    beta_list = alpha_list * numqi.gellmann.dm_to_gellmann_norm(dm_target_list[-1])

    dimA = 3
    dimB = 4
    num_state_list = [12,13,14,2*dimA*dimB,15]
    dm0 = numqi.random.rand_density_matrix(dimA*dimB)
    beta_u = numqi.entangle.get_density_matrix_boundary(dm0)[1]
    beta_list = np.linspace(0, beta_u, 50)
    dm_target_list = [numqi.utils.hf_interpolate_dm(dm0, beta=x) for x in beta_list]

    z0 = []
    for num_state in num_state_list:
        model = numqi.entangle.AutodiffCHAREE((dimA, dimB), num_state=num_state, distance_kind='gellmann')
        for dm_target_i in tqdm(dm_target_list):
            model.set_dm_target(dm_target_i)
            tmp0 = numqi.optimize.minimize(model, theta0='uniform', tol=1e-9, num_repeat=1, print_every_round=0).fun
            z0.append(tmp0)
    z0 = np.array(z0).reshape(len(num_state_list), len(beta_list))

    fig, ax = plt.subplots()
    for ind0 in range(len(num_state_list)):
        ax.plot(beta_list, z0[ind0], label=f"CHA({num_state_list[ind0]})")
    ax.set_xlim(min(beta_list), max(beta_list))
    # ax.set_ylim(1e-13, 1)
    ax.set_yscale('log')
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("REE")
    ax.legend()
    ax.set_title(f"Werner({dimA})")
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)


def demo_werner_ree():
    # about 3 minutes
    dim = 3
    kext_list = [8,32]
    alpha_list = np.linspace(0, 1, 100, endpoint=False) # alpha=1 is unstable for matrix logarithm

    dm_target_list = [numqi.state.Werner(dim, x) for x in alpha_list]
    beta_list = np.array([numqi.gellmann.dm_to_gellmann_norm(x) for x in dm_target_list]) #Euclidean norm

    ree_analytical = np.array([numqi.state.get_Werner_ree(dim, x) for x in alpha_list])

    ree_ppt = numqi.entangle.get_ppt_ree(dm_target_list, dim, dim)

    model = numqi.entangle.AutodiffCHAREE((dim,dim), distance_kind='ree')
    ree_cha = []
    for dm_target_i in tqdm(dm_target_list):
        model.set_dm_target(dm_target_i)
        tmp0 = numqi.optimize.minimize(model, theta0='uniform', tol=1e-12, num_repeat=1, print_every_round=0).fun
        # tmp0 = numqi.optimize.minimize_adam(model, num_step=100, theta0='uniform', tqdm_update_freq=0)
        ree_cha.append(tmp0)
    ree_cha = np.array(ree_cha)

    ree_pureb = []
    for kext in kext_list:
        model = numqi.entangle.PureBosonicExt(dim, dim, kext, distance_kind='ree')
        for dm_target_i in tqdm(dm_target_list):
            model.set_dm_target(dm_target_i)
            ree_pureb.append(numqi.optimize.minimize(model, tol=1e-12, num_repeat=1, print_every_round=0).fun)
    ree_pureb = np.array(ree_pureb).reshape(-1, len(dm_target_list))

    fig, ax = plt.subplots()
    ax.plot(beta_list, ree_analytical, "x", label="analytical")
    ax.plot(beta_list, ree_ppt, label="PPT")
    ax.plot(beta_list, ree_cha, label="CHA")
    for ind0 in range(len(kext_list)):
        ax.plot(beta_list, ree_pureb[ind0], label=f"PureB({kext_list[ind0]})")
    ax.set_xlim(min(beta_list), max(beta_list))
    ax.set_ylim(1e-13, 1)
    ax.set_yscale('log')
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("REE")
    ax.legend()
    ax.set_title(f"Werner({dim})")
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)


def demo_tiles_upb_bes():
    dimA = 3
    dimB = 3
    kext_list = [8,32]
    dm_target = numqi.entangle.load_upb('tiles', return_bes=True)[1]
    dm_norm = numqi.gellmann.dm_to_gellmann_norm(dm_target)
    beta_dm = numqi.entangle.get_density_matrix_boundary(dm_target)[1]
    beta_list = np.linspace(0, beta_dm, 100, endpoint=False)

    dm_target_list = [numqi.utils.hf_interpolate_dm(dm_target, beta=x, dm_norm=dm_norm) for x in beta_list]
    beta_ppt = numqi.entangle.get_ppt_boundary(dm_target, (dimA, dimB))[1]

    model = numqi.entangle.AutodiffCHAREE((dimA, dimB), distance_kind='ree')
    ree_cha = []
    for dm_target_i in tqdm(dm_target_list):
        model.set_dm_target(dm_target_i)
        tmp0 = numqi.optimize.minimize(model, theta0='uniform', tol=1e-10, num_repeat=1, print_every_round=0).fun
        ree_cha.append(tmp0)
    ree_cha = np.array(ree_cha)
    beta_cha = model.get_boundary(dm_target, xtol=1e-4, threshold=1e-7, converge_tol=1e-10, num_repeat=1)
    # beta=0.8649*rho_norm=0.2279211623566359 https://arxiv.org/abs/1705.01523

    ree_pureb = []
    for kext in kext_list:
        model = numqi.entangle.PureBosonicExt(dimA, dimB, kext, distance_kind='ree')
        for dm_target_i in tqdm(dm_target_list):
            model.set_dm_target(dm_target_i)
            ree_pureb.append(numqi.optimize.minimize(model, tol=1e-10, num_repeat=1, print_every_round=0).fun)
    ree_pureb = np.array(ree_pureb).reshape(-1, len(dm_target_list))

    fig,ax = plt.subplots()
    ax.plot(beta_list, ree_cha, label='cha')
    for ind0 in range(len(kext_list)):
        ax.plot(beta_list, ree_pureb[ind0], label=f"PureB({kext_list[ind0]})")
    ax.axvline(beta_cha, linestyle=':', color='r', label='cha')
    ax.axvline(beta_ppt, linestyle='-.', color='r', label='PPT')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel('REE')
    ax.set_ylim(1e-12, 0.1)
    ax.set_yscale('log')
    ax.legend()
    ax.set_title('tiles UPB/BES')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)



def demo_tiles_upb_pureb_ree():
    # about 4 minutes
    dimA = 3
    dimB = 3
    dm_target = numqi.entangle.load_upb('tiles', return_bes=True)[1]
    # dm_target = numqi.random.rand_density_matrix(dimA*dimB, kind='haar')
    beta_ppt = numqi.entangle.get_ppt_boundary(dm_target, (dimA, dimB))[1]
    beta_dm = numqi.entangle.get_density_matrix_boundary(dm_target)[1]
    beta_cha = numqi.entangle.CHABoundaryBagging((dimA,dimB)).solve(dm_target, maxiter=150)
    beta_list = np.linspace(beta_cha, beta_dm, 30)
    kext_list = [24,32,40]
    ree_pureb = []
    beta_pureb = []
    dm_target_list = [numqi.utils.hf_interpolate_dm(dm_target,beta=x) for x in beta_list]
    kwargs = dict(num_repeat=1, print_every_round=0, tol=1e-10)
    for kext in kext_list:
        model = numqi.entangle.PureBosonicExt(dimA, dimB, kext=kext, distance_kind='ree')
        for dm_i in tqdm(dm_target_list, desc=f'PureB(k={kext})'):
            model.set_dm_target(dm_i)
            ree_pureb.append(numqi.optimize.minimize(model, **kwargs).fun)
        beta_pureb.append(model.get_boundary(dm_target_list[-1]))
    ree_pureb = np.array(ree_pureb).reshape(len(kext_list), -1)
    fig,ax = plt.subplots()
    for ind0 in range(len(kext_list)):
        ax.plot(beta_list, ree_pureb[ind0], color=tableau[ind0], label=f'PureB(k={kext_list[ind0]})')
        ax.axvline(beta_pureb[ind0], linestyle=':', color=tableau[ind0])
    ax.axvline(beta_cha, color='red', label=r'$\beta_{CHA}$')
    ax.axvline(beta_ppt, linestyle=':', color='red', label=r'$\beta_{PPT}$')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel('REE')
    ax.set_title('tiles UPB/BES')
    ax.legend()
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)
    # fig.savefig('data/demo_tiles_upb_pureb_ree.png', dpi=200)

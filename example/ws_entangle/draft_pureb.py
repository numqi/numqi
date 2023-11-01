import torch
import numpy as np
import scipy.special
from tqdm import tqdm
import matplotlib.pyplot as plt
tableau = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']

import numqi

torch.set_num_threads(1)

np_rng = np.random.default_rng()


def demo_pureb_random_separable():
    # about 1 minutes
    dimA = 3
    dimB = 3
    kext_list = [16,24,32]
    num_sample = 100
    dm_target_list = [numqi.random.rand_separable_dm(dimA, dimB) for _ in range(num_sample)]

    kwargs = dict(num_repeat=3, print_every_round=0, tol=1e-10)
    ret = []
    for kext_i in kext_list:
        model = numqi.entangle.PureBosonicExt(dimA, dimB, kext=kext_i, distance_kind='ree')
        for dm_i in tqdm(dm_target_list):
            model.set_dm_target(dm_i)
            ret.append(numqi.optimize.minimize(model, **kwargs).fun)
    ret = np.array(ret).reshape(len(kext_list), -1)
    # most are zero

    fig,ax = plt.subplots()
    ax.plot(kext_list, ret.mean(axis=1), color=tableau[0])
    ax.fill_between(kext_list, ret.min(axis=1), ret.max(axis=1), alpha=0.3, color=tableau[0])
    ax.set_xlabel('k-ext')
    ax.set_title('REE over 1000 separable states $d_A=d_B=3$ (min,mean,max)')
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)


def demo_pureb_werner_ree():
    # TODO compare with irrep SDP
    dim = 2
    kext = 5
    alpha_list = np.linspace(0, 1, 50, endpoint=False)

    # (1,k)-ext boundary
    alpha_kext_boundary = (kext+dim**2-dim)/(kext*dim+dim-1)
    dm_kext_boundary = numqi.state.Werner(dim, alpha_kext_boundary)
    ret_ = []
    for alpha_i in alpha_list:
        if alpha_i<=alpha_kext_boundary:
            ret_.append(0)
        else:
            tmp0 = numqi.state.Werner(dim, alpha_i)
            ret_.append(numqi.utils.get_relative_entropy(tmp0, dm_kext_boundary))
    ret_ = np.array(ret_)

    model = numqi.entangle.PureBosonicExt(dim, dim, kext=kext, distance_kind='ree')
    ree_pureb = []
    kwargs = dict(num_repeat=3, print_every_round=0, tol=1e-10)
    for alpha_i in tqdm(alpha_list):
        model.set_dm_target(numqi.state.Werner(dim, alpha_i))
        ree_pureb.append(numqi.optimize.minimize(model, **kwargs).fun)
    ree_pureb = np.array(ree_pureb)

    fig,ax = plt.subplots()
    ax.plot(alpha_list, ret_, 'x', color=tableau[0], label='RE with respect to k-ext boundary')
    ax.plot(alpha_list, ree_pureb, color=tableau[1], label=f'PureB(k={kext})')
    ax.set_xlim(min(alpha_list), max(alpha_list))
    ax.axvline(alpha_kext_boundary, color=tableau[0], linestyle=':')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('REE')
    ax.set_ylim(1e-13, 1)
    ax.set_yscale('log')
    ax.legend()
    ax.set_title(f'Werner({dim})')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)


def demo_pureb_boundary_tiles():
    kext_list = [8,16,32,64]
    dm_tiles = numqi.entangle.load_upb('tiles', return_bes=True)[1]
    for kext in kext_list:
        model = numqi.entangle.PureBosonicExt(3, 3, kext=kext, distance_kind='ree')
        beta = model.get_boundary(dm_tiles, xtol=1e-4, threshold=1e-7, num_repeat=3, use_tqdm=True)
        print(kext, beta)
    # kext beta time(seconds)
    # 8 0.24145564897892074 30
    # 16 0.23489330520171586 48
    # 32 0.231290449794623 63
    # 64 0.22936034868368044 67
    # 128 0.22858830823930337 171
    # 256 0.22813795131341688
    # 512 0.22788060449862454


def demo_pureb_ree_2x2_inv_decay():
    dimA = 2
    dimB = 2
    dm_target = np.diag([1,0,1,0])/2
    kext_list = np.arange(4, 128, 4)
    ree_pureb = []
    for kext in tqdm(kext_list):
        model = numqi.entangle.PureBosonicExt(dimA, dimB, kext=kext, distance_kind='ree')
        model.set_dm_target(dm_target)
        ree_pureb.append(numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
    ree_pureb = np.array(ree_pureb)

    # y = 1/(ax+b)
    coeffb,coeffa = np.polynomial.polynomial.polyfit(kext_list, 1/ree_pureb, deg=1)
    yfit = 1/(coeffa*kext_list+coeffb)

    fig,ax = plt.subplots()
    ax.plot(kext_list, ree_pureb, 'x', label=f'PureB(k)')
    tmp0 = r'fit: $y=\frac{1}{' + f'{coeffa:.2f}x{coeffb:+.2f}' + '}$'
    ax.plot(kext_list, yfit, label=tmp0)
    ax.legend()
    ax.set_xlabel('k-ext')
    ax.set_ylabel('REE')
    ax.set_title(r'$diag(0.5,0,0.5,0)$')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)


def demo_pureb_ree_2x2_exp_decay():
    hf0 = lambda x,y,a: a*x + (1-a)*y
    dm0 = np.diag([0,1,0,0])
    dm1 = np.kron(np.diag([0,1]), np.ones((2,2))/2)
    dimA = 2
    dimB = 2
    lambda_ = 0.8
    dm_target = hf0(dm0, dm1, lambda_)
    kext_list = np.arange(4, 64, 4)

    ree_pureb = []
    ree_analytical = []
    for kext in tqdm(kext_list):
        model = numqi.entangle.PureBosonicExt(dimA, dimB, kext=kext)
        model.set_dm_target(dm_target)
        ree_pureb.append(numqi.optimize.minimize(model, num_repeat=3, tol=1e-12, print_every_round=0).fun)

        model.pj_imag.data[:] = 0
        model.pj_real.data[0,:] = 0
        model.pj_real.data[0,0] = np.sqrt(lambda_)
        model.pj_real.data[1] = torch.tensor((np.sqrt(1-lambda_)/2**(kext/2))*np.sqrt(scipy.special.binom(kext, np.arange(kext+1))))
        ree_analytical.append(model().item())
    ree_pureb = np.array(ree_pureb)
    ree_analytical = np.array(ree_analytical)

    tmp0 = np.log2(ree_analytical[:10])
    tmp1 = np.polynomial.polynomial.polyfit(kext_list[:len(tmp0)], tmp0, deg=1)
    # 2^(ax+b)=k c^x
    coeffK = 2**tmp1[0]
    coeffC = 2**tmp1[1]
    yfit = coeffK * (coeffC**kext_list)

    fig,ax = plt.subplots()
    ax.plot(kext_list, ree_pureb, 'x', label='PureB(k)')
    ax.plot(kext_list, ree_analytical, label=r'$\sqrt{\lambda}|01^k\rangle+\sqrt{1-\lambda}|1+^k\rangle$')
    tmp0 = f'fit: $y={coeffK:.2f}' + r'\times' + f'{coeffC:.2f}^k$'
    ax.plot(kext_list, yfit, ':', label=tmp0)
    ax.set_xlabel('k-ext')
    ax.set_ylabel('REE')
    ax.set_ylim(1e-13, None)
    ax.set_title(r'$\lambda |01\rangle \langle 01| + (1-\lambda)|1+\rangle\langle 1+|$' + rf'  $\lambda={lambda_}$')
    ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)


def demo_pureb_ree_2x2_mix_decay():
    dm0 = np.diag([0,1,0,1])/2
    dm1 = np.diag([1,0,1,0])/2
    hf0 = lambda x,y,a: (1-a)*x + a*y
    dimA = 2
    dimB = 2

    lambda_list = np.linspace(0, 1, 101)
    kext_list = np.array([4, 8, 16, 32, 64, 128])
    ree_pureb = []
    for kext in kext_list:
        model = numqi.entangle.PureBosonicExt(dimA, dimB, kext=kext, distance_kind='ree')
        for lambda_i in tqdm(lambda_list):
            model.set_dm_target(hf0(dm0, dm1, lambda_i))
            ree_pureb.append(numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
    ree_pureb = np.array(ree_pureb).reshape(-1, len(lambda_list))
    fig,ax = plt.subplots()
    for ind0 in range(len(kext_list)):
        ax.plot(lambda_list, ree_pureb[ind0], label=f'k={kext_list[ind0]}')
    ax.legend()
    ax.set_xlabel(r'$\lambda$')
    ax.set_yscale('log')
    ax.set_ylabel('REE')
    ax.set_xlim(0, 1)
    ax.set_ylim(1e-14, None)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.set_title(r'$diag(\lambda,1-\lambda,\lambda,1-\lambda)/2$')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)

    kext_list = list(range(4, 64, 2))
    lambda_list = [0, 0.01, 0.03, 0.05, 0.1, 0.5]
    ree_pureb = []
    for kext in tqdm(kext_list):
        model = numqi.entangle.PureBosonicExt(dimA, dimB, kext=kext, distance_kind='ree')
        for lambda_i in lambda_list:
            model.set_dm_target(hf0(dm0, dm1, lambda_i))
            ree_pureb.append(numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
    ree_pureb = np.array(ree_pureb).reshape(len(kext_list), -1)
    fig,ax = plt.subplots()
    for ind0 in range(len(lambda_list)):
        ax.plot(kext_list, ree_pureb[:,ind0], '-x', label=rf'$\lambda={lambda_list[ind0]}$')
    ax.set_xlabel('k-ext')
    ax.set_ylabel('REE')
    ax.set_title(r'$diag(\lambda,1-\lambda,\lambda,1-\lambda)/2$')
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    fig.tight_layout()
    ax.tick_params(axis='both', which='major', labelsize=11)
    fig.savefig('tbd01.png', dpi=200)


def demo_pureb_ree_maximally_mixed_state():
    ## about 150 seconds
    # para_list = [(x,2,y) for x in range(4, 12, 3) for y in range(2*x+5, 32, 2)]
    # para_list += [(10,2,x) for x in range(40, 128, 8)] + [(7,2,x) for x in range(40, 80, 8)]
    # para_list += [(4,2,x) for x in range(40, 128, 8)]
    # para_list += [(7,2,x) for x in range(80, 128, 8)]
    # para_list += [(x,3,y) for x in range(3, 12, 3) for y in range(2*x//3+5, 32, 2)]
    # para_list += [(9,3,x) for x in range(40, 80, 8)]
    # para_list += [(6,3,x) for x in range(40, 80, 8)]
    # para_list += [(3,3,x) for x in range(40, 80, 8)]
    para_list = [(x,2,y) for x in range(4, 24, 3) for y in range(2*x+5, max(3*x, 40), 4)]
    svqc_ree_data = dict()
    with tqdm(para_list) as pbar:
        for dimA,dimB,kext in pbar:
            model = numqi.entangle.PureBosonicExt(dimA, dimB, kext=kext, distance_kind='ree')
            model.set_dm_target(np.eye(dimA*dimB)/(dimA*dimB))
            tmp0 = numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun
            svqc_ree_data[(dimA,dimB,kext)] = tmp0
            pbar.set_postfix(key=f'{dimA}x{dimB},{kext}', ree=f'{tmp0:.5g}')

    plot_data = dict()
    for key,value in svqc_ree_data.items():
        tmp0 = key[0],key[1]
        if tmp0 in plot_data:
            plot_data[tmp0].append((key[2], value))
        else:
            plot_data[tmp0] = [(key[2], value)]
    plot_data = sorted(plot_data.items(), key=lambda x:x[0][::-1])
    fig,ax = plt.subplots()
    for ind0,((dimA,dimB),x1) in enumerate(plot_data):
        tmp0 = np.array(sorted([(x[0],x[1]) for x in x1], key=lambda x: x[0]))
        ax.plot(tmp0[:,0], tmp0[:,1], '-x', color=tableau[ind0], label=f'$d_A={dimA},d_B={dimB}$')
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel('$k$-ext')
    ax.set_ylabel('REE')
    ax.set_title(r'REE of $\rho_0$ via SVQC-pure-k')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)
    # fig.savefig('pureb_ree_rho0.png', dpi=200)


def demo_pureb_quantum_werner():
    # about 3 minutes
    dim = 2
    kext = 8
    num_layer = 9
    alpha_list = np.linspace(0, 1, 50, endpoint=False)

    kwargs = dict(tol=1e-12, num_repeat=1, print_every_round=0)

    ree_pureb_q = []
    model = numqi.entangle.QuantumPureBosonicExt(dim, dim, kext, num_layer)
    for alpha_i in tqdm(alpha_list):
        model.set_dm_target(numqi.state.Werner(dim, alpha_i))
        ree_pureb_q.append(numqi.optimize.minimize(model, **kwargs).fun)
    ree_pureb_q = np.array(ree_pureb_q)

    ree_pureb = []
    model = numqi.entangle.PureBosonicExt(dim, dim, kext, distance_kind='ree')
    for alpha_i in tqdm(alpha_list):
        model.set_dm_target(numqi.state.Werner(dim, alpha_i))
        ree_pureb.append(numqi.optimize.minimize(model, **kwargs).fun)
    ree_pureb = np.array(ree_pureb)

    ree_analytical = np.array([numqi.state.get_Werner_ree(dim, x) for x in alpha_list])

    fig,ax = plt.subplots()
    ax.plot(alpha_list, ree_analytical, label='analytical')
    ax.plot(alpha_list, ree_pureb_q, label=f'PureB-Q({kext}) #layer={num_layer}')
    ax.plot(alpha_list, ree_pureb, '+', label=f'PureB({kext})')
    ax.set_xlim(min(alpha_list), max(alpha_list))
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylim(1e-15, None)
    ax.set_yscale('log')
    ax.set_ylabel('REE')
    ax.legend()
    ax.set_title(f'Werner-{dim}')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)

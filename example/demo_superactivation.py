import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import numqi

np_rng = np.random.default_rng()
hf_trace = lambda x,y: np.dot(x.T.reshape(-1), y.reshape(-1))


def rand_bisurface_uniform(N0, np_rng):
    assert N0%2==0
    tmp0 = np_rng.normal(size=N0).reshape(2,-1)
    ret = (tmp0 / np.linalg.norm(tmp0, axis=1, keepdims=True)).reshape(-1)
    return ret


def demo_loss_landscape():
    theta = np_rng.uniform(0, np.pi/2)
    npA,npB,npAB,npL,npR,field = numqi.matrix_space.get_matrix_subspace_example('0error-eq537', theta)

    tmp0 = np.concatenate([npL,npR])/np.sqrt(2)
    tmp1 = np.concatenate([npR,npL])/np.sqrt(2)
    _,hf_angle = numqi.matrix_space.get_vector_plane(tmp0, tmp1)
    angle_list = np.linspace(0, 2*np.pi, 200)
    ret = []
    for angle_i in angle_list:
        tmp0 = hf_angle(angle_i)
        tmp1 = tmp0[:(len(tmp0)//2)]
        tmp2 = tmp0[(len(tmp0)//2):]
        tmp1 /= np.linalg.norm(tmp1)
        tmp2 /= np.linalg.norm(tmp2)
        ret.append(np.linalg.norm((npAB @ tmp1) @ tmp2)**2)
    ret = np.array(ret)

    fig, ax = plt.subplots()
    ax.plot(ret*np.cos(angle_list), ret*np.sin(angle_list))
    ax.plot([0, 0.3], [0,0], ':', color='red')
    ax.text(0.45, 0, r'$L(x^*,y^*,\theta)$', color='red', horizontalalignment='center', verticalalignment='center')
    ax.plot([0, 0], [0,0.36], ':', color='red')
    ax.text(0, 0.45, r'$L(y^*,x^*,\theta)$', color='red', horizontalalignment='center', verticalalignment='center')
    ax.plot([0, -0.3], [0,0], ':', color='red')
    ax.text(-0.45, 0, r'$L(-x^*,-y^*,\theta)$', color='red', horizontalalignment='center', verticalalignment='center')
    ax.plot([0, 0], [0,-0.36], ':', color='red')
    ax.text(0, -0.45, r'$L(-y^*,-x^*,\theta)$', color='red', horizontalalignment='center', verticalalignment='center')
    ax.set_aspect('equal')
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.5, 0.5)
    fig.tight_layout()
    fig.savefig('tbd00.png')
    # fig.savefig('superactivation_landscope_xystar.png', dpi=200)


def demo_loss_landscape_random():
    theta = np_rng.uniform(0, np.pi/2)
    theta = np.pi/4
    npA,npB,npAB,npL,npR,field = numqi.matrix_space.get_matrix_subspace_example('0error-eq537', theta)

    # tmp0 = np.concatenate([npL,npR])/np.sqrt(2)
    tmp0 = np_rng.normal(size=2*len(npL))
    tmp1 = np_rng.normal(size=2*len(npL))
    _,hf_angle = numqi.matrix_space.get_vector_plane(tmp0, tmp1)
    angle_list = np.linspace(0, 2*np.pi, 200)
    ret = []
    for angle_i in angle_list:
        tmp0 = hf_angle(angle_i)
        tmp1 = tmp0[:(len(tmp0)//2)]
        tmp2 = tmp0[(len(tmp0)//2):]
        tmp1 /= np.linalg.norm(tmp1)
        tmp2 /= np.linalg.norm(tmp2)
        ret.append(np.linalg.norm((npAB @ tmp1) @ tmp2)**2)
    ret = np.array(ret)

    fig, ax = plt.subplots()
    ax.plot(ret*np.cos(angle_list), ret*np.sin(angle_list))
    fig.tight_layout()
    fig.savefig('tbd00.png')
    # fig.savefig('superactivation_landscope_random.png', dpi=200)


def demo_special_matrix_D():
    hf_matD = lambda m,n: np.einsum(np.eye(m), [0,3], np.diag(1-2*(np.arange(n)%2)), [1,2],
                    [0,1,2,3], optimize=True).reshape(m*n, m*n)

    for N0,N1 in [(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8)]:
        matD = hf_matD(N0,N1)
        EVL,EVC = np.linalg.eig(matD)
        ind_1 = np.abs(EVL-1) < 1e-10
        ind_m1 = np.abs(EVL+1) < 1e-10
        ind_I = np.abs(EVL-1j) < 1e-10
        ind_mI = np.abs(EVL+1j) < 1e-10
        assert np.all(ind_1+ind_m1+ind_I+ind_mI)
        print(f'({N0},{N1}): (1,-1,i,-i)=', ind_1.sum(), ind_m1.sum(), ind_I.sum(), ind_mI.sum())
    # (2,2): (1,-1,i,-i)= 1 1 1 1
    # (3,3): (1,-1,i,-i)= 3 2 2 2
    # (4,4): (1,-1,i,-i)= 4 4 4 4
    # (5,5): (1,-1,i,-i)= 7 6 6 6
    # (6,6): (1,-1,i,-i)= 9 9 9 9
    # (7,7): (1,-1,i,-i)= 13 12 12 12
    # (8,8): (1,-1,i,-i)= 16 16 16 16

    N0_N1_list = sorted([(x,y) for x in range(2,10) for y in range(2,10)], key=lambda x:x[0]*x[1])
    # N0_N1_list = [(2,2), (2,3), (3,2), (2,4), (4,2), (3,3), (2,5), (5,2),(2,6),(6,2),(3,4),(4,3)]
    for N0,N1 in N0_N1_list:
        matA = np.eye(N0)
        matB = np.diag(1-2*(np.arange(N1)%2))
        matD = np.einsum(matA, [0,3], matB, [1,2], [0,1,2,3], optimize=True).reshape(N0*N1,N0*N1)
        EVL,EVC = np.linalg.eig(matD)
        assert np.abs(np.abs(EVL)-1).max() < 1e-10
        EVL_pi = np.sort(np.angle(EVL))/np.pi
        print(F'{N0}x{N1}', EVL_pi)



def demo_multipartite_quantum_channel_0error():
    # doi.org/10.1007/978-3-319-42794-2 quantum error-error information theory eq-5.24
    matrix_space,field = numqi.matrix_space.get_matrix_subspace_example('0error-eq524')

    tmp0 = matrix_space.reshape(-1, 16)
    tmp0 = tmp0 / np.linalg.norm(tmp0, axis=1, keepdims=True)
    projector = tmp0.T @ tmp0.conj()
    projector_orth = np.eye(16) - projector
    hf_channel = lambda rho: np.diag([hf_trace(projector, rho), hf_trace(np.eye(projector.shape[0])-projector, rho)])
    kraus_op = numqi.channel.hf_channel_to_kraus_op(hf_channel, dim_in=16)

    psi0 = np.eye(4)/2
    psi1 = np.diag([1,-1,1,-1])/2
    bit = 1
    psiB = psi0
    for bit in [0,1]:
        psiA = psi0 if bit==0 else psi1

        tmp0 = kraus_op.reshape(-1, 2, 4, 4)
        ret = np.einsum(tmp0, [0,1,2,3], tmp0.conj(), [0,4,5,6], tmp0, [7,8,9,10], tmp0.conj(), [7,11,12,13],
                psiA, [2,9], psiA.conj(), [5,12], psiB, [3,10], psiB.conj(), [6,13], [1,8,4,11], optimize=True).reshape(4,4)
        print(bit, ret)
        # bit=0 np.diag([0.5,0,0,0.5]) #quantum error-error information theory eq-5.26
        # bit=1 np.diag([0,0.5,0.5,0]) #quantum error-error information theory eq-5.27

        # choi_op = numqi.channel.kraus_op_to_choi_op(kraus_op)
        # choi2_op = np.kron(choi_op,choi_op).reshape(4,4,2, 4,4,2, 4,4,2, 4,4,2).transpose(
        #             0,3,1,4,2,5, 6,9,7,10,8,11).reshape(256,4,256,4).reshape(1024,1024)
        # kraus2_op = numqi.channel.choi_op_to_kraus_op(choi2_op, dim_in=256, zero_eps=1e-10)
        # tmp0 = (psiA.reshape(-1,1) * psiB.reshape(-1)).reshape(-1)
        # tmp1 = kraus2_op @ tmp0
        # ret = tmp1.T @ tmp1.conj()


def demo_typical_fail_converged_loss():
    theta_list = np.linspace(0, np.pi/2, 50)
    loss_list = []
    kwargs = dict(theta0=rand_bisurface_uniform, num_repeat=50, tol=1e-8,
            early_stop_threshold=1e-5, print_every_round=0)
    for theta_i in tqdm(theta_list):
        npA,npB,npAB,npL,npR,field = numqi.matrix_space.get_matrix_subspace_example('0error-eq537', theta_i)

        model = numqi.matrix_space.DetectOrthogonalRankOneModel(npA, dtype='float64')
        theta_optim_A = numqi.optimize.minimize(model, **kwargs)

        model = numqi.matrix_space.DetectOrthogonalRankOneModel(npB, dtype='float64')
        theta_optim_B = numqi.optimize.minimize(model, **kwargs)

        model = numqi.matrix_space.DetectOrthogonalRankOneModel(npAB, dtype='float64')
        theta_optim_AB = numqi.optimize.minimize(model, **kwargs)
        loss_list.append((theta_optim_A.fun, theta_optim_B.fun, theta_optim_AB.fun))
    loss_list = np.array(loss_list).T

    fig,ax = plt.subplots()
    ax.plot(theta_list, loss_list[0], label=r'$B^{(a,\alpha)}$')
    ax.plot(theta_list, loss_list[1], label=r'$B^{(b,\alpha)}$')
    ax.plot(theta_list, loss_list[2], label=r'$B^{(\alpha)}$')
    ax.legend()
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel('convergeed loss')
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)
    # fig.savefig('superactivation-loss.png', dpi=200)


def demo_relation_between_kraus_op_and_matrix_space():
    num_term = 4
    dim_in = 4
    dim_out = 4
    kwargs = dict(theta0=rand_bisurface_uniform, num_repeat=50, tol=1e-8,
            early_stop_threshold=1e-5, print_every_round=0)
    for ind_round in range(10):
        kraus_op0 = numqi.random.rand_kraus_op(num_term, dim_in, dim_out)
        kraus_op1 = numqi.random.rand_kraus_op(num_term, dim_in, dim_out)

        matrix_space0 = np.stack([(x.T.conj() @ y) for x in kraus_op0 for y in kraus_op0])
        matrix_space1 = np.stack([(x.T.conj() @ y) for x in kraus_op1 for y in kraus_op1])
        matrix_space2 = np.stack([np.kron(x,y) for x in matrix_space0 for y in matrix_space1], axis=0)
        # not support complex yet
        # theta_optim_list = []
        # for space in [matrix_space0, matrix_space1, matrix_space2]:
        #     model = numqi.matrix_space.DetectOrthogonalRankOneModel(space, dtype='float64')
        #     theta_optim_list.append(numqi.optimize.minimize(model, **kwargs))
        # print(ind_round, [x.fun for x in theta_optim_list])


def demo_superactivation_converge_probability():
    theta = np_rng.uniform(0, np.pi/2)
    npA,npB,npAB,npL,npR,field = numqi.matrix_space.get_matrix_subspace_example('0error-eq537', theta)
    model = numqi.matrix_space.DetectOrthogonalRankOneModel(npAB, dtype='float64')
    kwargs = dict(theta0=rand_bisurface_uniform, num_repeat=1, tol=1e-9, print_every_round=0)
    z0 = np.array([numqi.optimize.minimize(model,**kwargs).fun for _ in tqdm(range(50000))])
    (z0 < 1e-8).mean() #1e-8 should be tuned to get a clear gap
    # bisphere-surface real xBy repeat=50k
    # theta=8pi/32  p=0.00164
    # theta=6pi/32  p=0.00162
    # theta=4pi/32 p=0.00132
    # theta=3pi/32 p=0.0012
    # theta=2pi/32  p=0.00084
    # theta=pi/32  p=0.0004


def demo_superactivation_various_model():
    theta = np_rng.uniform(0, np.pi/2)
    npA,npB,npAB,psiL,psiR,field = numqi.matrix_space.get_matrix_subspace_example('0error-eq537', theta)
    _,npAB_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(npAB, field='real')

    # (best) real-space rank1
    model = numqi.matrix_space.DetectOrthogonalRankOneModel(npAB, dtype='float64')
    theta_optim = numqi.optimize.minimize(model, 'normal', num_repeat=2000, tol=1e-8, early_stop_threshold=1e-5, print_every_round=10)
    if theta_optim.fun < 1e-5:
        theta_optim = numqi.optimize.minimize(model, theta_optim.x, num_repeat=1, tol=1e-14)
        vecX,vecY,loss_abs_max = model.get_vecX_vecY(reshape=(4,4), tag_print=True)

    # (bad) (a_i A_i) is rank1
    model = numqi.matrix_space.DetectOrthogonalRankOneModel(npAB_orth, dtype='float64')
    theta_optim0 = numqi.optimize.minimize(model, 'normal', method='L-BFGS-B', num_repeat=40, tol=1e-9)
    model = numqi.matrix_space.DetectOrthogonalRankOneModel(npAB_orth, dtype='complex128')
    theta_optim0 = numqi.optimize.minimize(model, 'normal', method='L-BFGS-B', num_repeat=40, tol=1e-9)

    # space(x B) not full rank
    # (bad) reduce the parameter dimension, at the cost of evaluating the eigevalues
    model = numqi.matrix_space.DetectOrthogonalRankOneEigenModel(npAB)
    theta_optim0 = numqi.optimize.minimize(model, 'normal', method='L-BFGS-B', num_repeat=200, early_stop_threshold=1e-6, tol=1e-9)

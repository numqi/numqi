import numpy as np
import matplotlib.pyplot as plt
import torch

import numqi

def get_trace_distance(rho0:np.ndarray, rho1:np.ndarray):
    tmp0 = rho0 - rho1
    assert (tmp0.ndim==2) and (np.abs(tmp0-tmp0.T.conj()).max()<1e-10)
    ret = np.abs(np.linalg.eigvalsh(tmp0)).sum() / 2
    # abs is not a good choice for loss function, so no torch-version
    return ret

def test_trace_distance_contraction():
    # @book-QCQI-page406/eq9.35 trace-distance is contractive under TPCP map
    din = 3
    dout = 5
    kop_term = din*dout

    rho0 = numqi.random.rand_density_matrix(din)
    rho1 = numqi.random.rand_density_matrix(din)
    kop = numqi.random.rand_kraus_op(kop_term, din, dout)

    ret0 = get_trace_distance(rho0, rho1)
    tmp0 = numqi.channel.apply_kraus_op(kop, rho0)
    tmp1 = numqi.channel.apply_kraus_op(kop, rho1)
    ret1 = get_trace_distance(tmp0, tmp1)
    assert ret1 < (ret0+1e-10) #epsilon is added to avoid rounding error

def choi_op_to_bloch_map(op):
    assert (op.ndim==4) #(in,out,in,out)
    assert (op.shape[0]==op.shape[2]) and (op.shape[1]==op.shape[3])
    din,dout = op.shape[:2]
    tmp0 = op.transpose(1,3,2,0).reshape(dout*dout, din, din)
    tmp1 = numqi.gellmann.matrix_to_gellmann_basis(tmp0)
    op_gm = numqi.gellmann.matrix_to_gellmann_basis(tmp1.T.reshape(-1,dout,dout)).real.T
    matA = op_gm[:-1,:-1] * 2
    vecb = op_gm[:-1,-1] * np.sqrt(2/din)
    return matA, vecb


def test_choi_op_to_bloch_map():
    din = 3
    dout = 5
    choi = numqi.random.rand_choi_op(din, dout)

    rho = numqi.random.rand_density_matrix(din)
    ret_ = numqi.channel.apply_choi_op(choi, rho)

    matA, vecb = choi_op_to_bloch_map(choi.reshape(din, dout, din, dout))
    tmp0 = matA @ numqi.gellmann.dm_to_gellmann_basis(rho) + vecb
    ret0 = numqi.gellmann.gellmann_basis_to_dm(tmp0)
    assert np.abs(ret_-ret0).max() < 1e-10


def test_channel_fix_point():
    # @book-QCQI-page408/Ex9.9 Schauder's fixed point theorem
    dim = 3
    num_term = dim*dim
    kop = numqi.random.rand_kraus_op(num_term, dim, dim)
    choi = numqi.channel.kraus_op_to_choi_op(kop)

    matA,vecb = choi_op_to_bloch_map(choi.reshape(dim, dim, dim, dim))
    tmp0 = np.linalg.solve(matA - np.eye(dim*dim-1), -vecb) #geneircally, it is not invertible
    rho_fix = numqi.gellmann.gellmann_basis_to_dm(tmp0)

    ret0 = numqi.channel.apply_choi_op(choi, rho_fix)
    assert np.abs(rho_fix - ret0).max() < 1e-10


def get_Renyi_entropy(rho:np.ndarray, alpha:float):
    assert alpha!=1
    assert (rho.ndim==2) and (np.abs(rho-rho.T.conj()).max()<1e-10)
    EVL = np.linalg.eigvalsh(rho)
    ret = np.log((EVL**alpha).sum()) / (1-alpha)
    return ret

def demo_renyi_entropy():
    dim = 4
    rho = numqi.random.rand_density_matrix(dim)

    ret_von = numqi.entangle.get_von_neumann_entropy(rho, eps=1e-10)
    # alpha_list = 1 + np.linspace(0, 1, 30)[1:]**3
    alpha01_list = np.linspace(0, 1, 30)[1:-1]
    ret01_renyi = np.array([get_Renyi_entropy(rho,x) for x in alpha01_list])
    alpha12_list = np.linspace(1, 10, 50)[1:]
    ret12_renyi = np.array([get_Renyi_entropy(rho,x) for x in alpha12_list])

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

    entropy_AB = get_Renyi_entropy(rhoAB, alpha)
    entropy_A = get_Renyi_entropy(rdmA, alpha)
    entropy_B = get_Renyi_entropy(rdmB, alpha)
    print(entropy_AB, entropy_A, entropy_B)
    assert entropy_AB>=entropy_A
    assert entropy_AB>=entropy_B


def get_purification(rho, dimR=None, seed=None):
    # all purification are connected by Stiefel manifold
    assert (rho.ndim==2) and (np.abs(rho-rho.T.conj()).max()<1e-10)
    assert abs(np.trace(rho)-1).max() < 1e-10
    ret = np.linalg.cholesky(rho)
    if dimR is not None:
        dim = rho.shape[0]
        assert dimR>=dim
        np_rng = np.random.default_rng(seed)
        tmp0 = np_rng.normal(size=(dimR*dim*2*dim))
        tmp1 = numqi.manifold.to_stiefel_qr(tmp0, dim=dimR*dim, rank=dim)
        ret = ret @ tmp1.T
    return ret

def test_get_purification():
    dim = 3
    for dimR in [None, 4]:
        rho = numqi.random.rand_density_matrix(dim)
        psi = get_purification(rho, dimR=dimR)
        tmp0 = psi @ psi.T.conj()
        assert np.abs(tmp0 - rho).max() < 1e-10

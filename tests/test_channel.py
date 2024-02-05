import numpy as np

import numqi

def test_choi_op_to_kraus_op():
    dim_in = 5
    dim_out = 3
    choi_op = numqi.random.rand_choi_op(dim_in, dim_out)
    kraus_op = numqi.channel.choi_op_to_kraus_op(choi_op, dim_in, zero_eps=1e-10)
    super_op = numqi.channel.kraus_op_to_super_op(kraus_op)

    ret0 = numqi.channel.kraus_op_to_choi_op(kraus_op)
    assert np.abs(choi_op-ret0).max() < 1e-10

    ret0 = numqi.channel.super_op_to_choi_op(super_op)
    assert np.abs(choi_op-ret0).max() < 1e-10


def test_hf_channel_to_kraus_op():
    dim0 = 4
    dim1 = 3
    choi_op = numqi.random.rand_choi_op(dim0, dim1)

    hf_channel = lambda rho: numqi.channel.apply_choi_op(choi_op, rho)
    kraus_op = numqi.channel.hf_channel_to_kraus_op(hf_channel, dim0)
    choi_op1 = numqi.channel.kraus_op_to_choi_op(kraus_op)
    assert np.abs(choi_op - choi_op1).max() < 1e-7


def test_choi_op_to_bloch_map():
    din = 3
    dout = 5
    choi = numqi.random.rand_choi_op(din, dout)

    rho = numqi.random.rand_density_matrix(din)
    ret_ = numqi.channel.apply_choi_op(choi, rho)

    matA, vecb = numqi.channel.choi_op_to_bloch_map(choi.reshape(din, dout, din, dout))
    tmp0 = matA @ numqi.gellmann.dm_to_gellmann_basis(rho) + vecb
    ret0 = numqi.gellmann.gellmann_basis_to_dm(tmp0)
    assert np.abs(ret_-ret0).max() < 1e-10


def test_channel_fix_point():
    # @book-QCQI-page408/Ex9.9 Schauder's fixed point theorem
    dim = 3
    num_term = dim*dim
    kop = numqi.random.rand_kraus_op(num_term, dim, dim)
    choi = numqi.channel.kraus_op_to_choi_op(kop)

    matA,vecb = numqi.channel.choi_op_to_bloch_map(choi.reshape(dim, dim, dim, dim))
    tmp0 = np.linalg.solve(matA - np.eye(dim*dim-1), -vecb) #geneircally, it is not invertible
    rho_fix = numqi.gellmann.gellmann_basis_to_dm(tmp0)

    ret0 = numqi.channel.apply_choi_op(choi, rho_fix)
    assert np.abs(rho_fix - ret0).max() < 1e-10

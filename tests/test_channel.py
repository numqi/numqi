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

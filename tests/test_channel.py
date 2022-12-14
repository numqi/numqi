import numpy as np

import numpyqi

def test_choi_op_to_kraus_op():
    dim_in = 5
    dim_out = 3
    choi_op = numpyqi.random.rand_choi_op(dim_in, dim_out)
    kraus_op = numpyqi.channel.choi_op_to_kraus_op(choi_op, dim_in, zero_eps=1e-10)
    super_op = numpyqi.channel.kraus_op_to_super_op(kraus_op)

    ret0 = numpyqi.channel.kraus_op_to_choi_op(kraus_op)
    assert np.abs(choi_op-ret0).max() < 1e-10

    ret0 = numpyqi.channel.super_op_to_choi_op(super_op)
    assert np.abs(choi_op-ret0).max() < 1e-10

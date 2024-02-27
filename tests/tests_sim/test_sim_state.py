import numpy as np

import numqi

np_rng = np.random.default_rng()

def test_reduce_shape_index():
    snone = slice(None)
    case_list = [
        ((2,2,2,2), (None,1,0,None), (2,4,2), (snone,2,snone)),
        ((2,2,2,2), (None,None,1,1), (4,4), (snone,3)),
    ]
    for shape,index,shape1,index1 in case_list:
        ret = numqi.sim.state.reduce_shape_index(shape, index)
        assert ret == (shape1, index1)


def test_inner_product_psi0_O_psi1():
    N0 = 2
    num_term = 4
    q0 = numqi.random.rand_haar_state(2**N0)
    q1 = numqi.random.rand_haar_state(2**N0)
    coeff = np_rng.normal(size=num_term)
    op_list = np.stack([numqi.random.rand_hermitian_matrix(2**N0) for _ in range(num_term)])
    tmp0 = tuple(range(N0))
    operator_list = [[(x,*tmp0)] for x in op_list]

    tmp0 = np.einsum(op_list, [0,1,2], coeff, [0], [1,2], optimize=True)
    ret_ = np.vdot(q0, tmp0@q1)

    tmp0 = numqi.sim.state.inner_product_psi0_O_psi1(q0, q1, operator_list)
    ret0 = np.dot(tmp0, coeff)
    assert np.abs(ret_-ret0).max() < 1e-7

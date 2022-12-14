import itertools
import numpy as np

import numpyqi

hfe = lambda x, y, eps=1e-5: np.max(np.abs(x - y) / (np.abs(x) + np.abs(y) + eps))

np_rng = np.random.default_rng()
hf_randc = lambda *size: np_rng.uniform(-1,1,size=size)+1j*np_rng.uniform(-1,1,size=size)

def test_qec_inner_product_grad():
    num_qubit = 3
    num_logical_qubit = 1
    num_logical_dim = 2**num_logical_qubit

    np0 = hf_randc(2**num_logical_qubit, 2**num_qubit)
    op_list = [[([x],hf_randc(2,2))] for x in range(num_qubit)]
    np1 = hf_randc(len(op_list),num_logical_dim,num_logical_dim)
    zero_eps = 1e-7

    def hf0(q0):
        tmp0 = numpyqi.qec.knill_laflamme_inner_product(q0.reshape(-1,2**num_qubit), op_list)
        ret = (tmp0*np1).real.sum()
        return ret
    ret_grad_ = np.zeros_like(np0)
    for indI in itertools.product(*[list(range(x)) for x in np0.shape]):
        x0,x1,x2,x3 = [np0.copy() for _ in range(4)]
        x0[indI] += zero_eps
        x1[indI] -= zero_eps
        x2[indI] += 1j*zero_eps
        x3[indI] -= 1j*zero_eps
        ret_grad_[indI] = (hf0(x0)-hf0(x1))/(2*zero_eps) + 1j*(hf0(x2)-hf0(x3))/(2*zero_eps)

    ret_grad = numpyqi.qec.knill_laflamme_inner_product_grad(np0.reshape(-1,2**num_qubit), op_list, np1.conj())
    assert np.abs(ret_grad-ret_grad_).max() < 1e-7


def test_code523():
    code = numpyqi.qec.generate_code523()
    code_np = numpyqi.qec.generate_code_np(code['encode'], code['num_logical_dim'])
    assert np.abs(numpyqi.qec.degeneracy(code_np[0]) - np.ones(16)).max() < 1e-7
    assert np.abs(numpyqi.qec.degeneracy(code_np[1]) - np.ones(16)).max() < 1e-7
    qweA,qweB = numpyqi.qec.quantum_weight_enumerator(code_np)
    assert np.abs(qweA - np.array([0,0,0,15,0])).max() < 1e-7
    assert np.abs(qweB - np.array([0,0,30,15,18])).max() < 1e-7

    z0 = numpyqi.qec.check_stabilizer(code['stabilizer'], code_np)
    assert np.abs(z0-1).max() < 1e-7


def test_knill_laflamme_loss():
    str_list = ['523', '422', '442', '642', '883', '8_64_2', '10_4_4', '11_2_5'] #10_16_3
    for str_i in str_list: #15 seconds
        code = getattr(numpyqi.qec, 'generate_code'+str_i)()
        code_np = numpyqi.qec.generate_code_np(code['encode'], code['num_logical_dim'])
        error_list = numpyqi.qec.make_error_list(code['num_qubit'], code['distance'])
        tmp0 = numpyqi.qec.knill_laflamme_inner_product(code_np, error_list)
        loss = numpyqi.qec.knill_laflamme_loss(tmp0, kind='L2')
        assert loss < 1e-7

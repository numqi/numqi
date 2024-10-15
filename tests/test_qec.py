import itertools
import numpy as np
import torch

import numqi

np_rng = np.random.default_rng()
hf_randc = lambda *size: np_rng.uniform(-1,1,size=size)+1j*np_rng.uniform(-1,1,size=size)

def test_knill_laflamme_inner_product():
    num_qubit = 3
    num_logical_qubit = 1
    num_logical_dim = 2**num_logical_qubit

    np0 = hf_randc(2**num_logical_qubit, 2**num_qubit)
    op_list = [[([x],hf_randc(2,2))] for x in range(num_qubit)]
    np1 = hf_randc(len(op_list),num_logical_dim,num_logical_dim)
    zero_eps = 1e-7

    def hf0(q0):
        tmp0 = numqi.qec.knill_laflamme_inner_product(q0.reshape(-1,2**num_qubit), op_list)
        ret = (tmp0*np1).real.sum()
        return ret
    ret_ = np.zeros_like(np0)
    for indI in itertools.product(*[list(range(x)) for x in np0.shape]):
        x0,x1,x2,x3 = [np0.copy() for _ in range(4)]
        x0[indI] += zero_eps
        x1[indI] -= zero_eps
        x2[indI] += 1j*zero_eps
        x3[indI] -= 1j*zero_eps
        ret_[indI] = (hf0(x0)-hf0(x1))/(2*zero_eps) + 1j*(hf0(x2)-hf0(x3))/(2*zero_eps)

    q0_torch = torch.tensor(np0, dtype=torch.complex128, requires_grad=True)
    tmp0 = torch.tensor(np1, dtype=torch.complex128)
    loss = (numqi.qec.knill_laflamme_inner_product(q0_torch, op_list)*tmp0).sum().real
    loss.backward()
    ret0 = q0_torch.grad.detach().numpy()
    assert np.abs(ret0-ret_).max() < 1e-7


def test_code523():
    code = numqi.qec.generate_code523()
    code_np = numqi.qec.generate_code_np(code['encode'], code['num_logical_dim'])
    assert np.abs(numqi.qec.degeneracy(code_np[0]) - np.ones(16)).max() < 1e-7
    assert np.abs(numqi.qec.degeneracy(code_np[1]) - np.ones(16)).max() < 1e-7
    qweA,qweB = numqi.qec.quantum_weight_enumerator(code_np)
    assert np.abs(qweA - np.array([1,0,0,0,15,0])).max() < 1e-7
    assert np.abs(qweB - np.array([1,0,0,30,15,18])).max() < 1e-7

    z0 = numqi.qec.check_stabilizer(code['stabilizer'], code_np)
    assert np.abs(z0-1).max() < 1e-7


def test_knill_laflamme_loss():
    str_list = ['523', '422', '442', '642', '883', '8_64_2', '10_4_4'] #11_2_5 10_16_3
    for str_i in str_list: #15 seconds
        code = getattr(numqi.qec, 'generate_code'+str_i)()
        code_np = numqi.qec.generate_code_np(code['encode'], code['num_logical_dim'])
        error_list = numqi.qec.make_error_list(code['num_qubit'], code['distance'])
        tmp0 = numqi.qec.knill_laflamme_inner_product(code_np, error_list)
        loss = numqi.qec.knill_laflamme_loss(tmp0, kind='L2')
        assert loss < 1e-7


def build_circuit(num_depth, num_qubit):
    prime = [x for x in range(3, num_qubit) if np.gcd(x,num_qubit)==1]
    assert len(prime)>=1
    prime = prime[0]
    circ = numqi.sim.Circuit(default_requires_grad=True)
    for _ in range(num_depth):
        for x in range(num_qubit):
            circ.u3(x)
        tmp0 = np.mod(-np.arange(num_qubit+1), num_qubit)
        for x,y in zip(tmp0[:-1],tmp0[1:]):
            circ.cu3(x, y)
        for x in range(num_qubit):
            circ.u3(x)
        tmp0 = np.mod(-np.arange(num_qubit+1)*prime, num_qubit)
        for x,y in zip(tmp0[:-1],tmp0[1:]):
            circ.cu3(x, y)
    return circ


## unittest too slow, remove VarQEC in future version
# def test_varqec():
#     # about 20 seconds
#     str_qecc = '((5,2,3))'
#     tmp0 = numqi.qec.parse_str_qecc(str_qecc)
#     num_qubit = tmp0['num_qubit']
#     num_logical_dim = tmp0['num_logical_dim']
#     distance = tmp0['distance']
#     num_layer = 2
#     kwargs = dict(theta0=('uniform',0,2*np.pi), num_repeat=1, tol=1e-12, print_freq=0, print_every_round=1, early_stop_threshold=1e-8)

#     error_list = numqi.qec.make_error_list(num_qubit, distance)

#     circuit = build_circuit(num_layer, num_qubit)
#     model = numqi.qec.VarQEC(circuit, num_logical_dim, error_list, loss_type='L2')
#     theta_optim = numqi.optimize.minimize(model, seed=233, **kwargs)
#     assert theta_optim.fun < 1e-8
#     code0 = model.get_code()

#     theta_optim = numqi.optimize.minimize(model, seed=236, **kwargs)
#     assert theta_optim.fun < 1e-8
#     code1 = model.get_code()

#     # equivalent QECC
#     model = numqi.qec.QECCEqualModel(code0, code1)
#     theta_optim = numqi.optimize.minimize(model, seed=237, **kwargs)
#     assert theta_optim.fun < 1e-8

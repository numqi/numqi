import itertools
import numpy as np
import torch

import numqi

try:
    import mosek
    USE_MOSEK = True
except ImportError:
    USE_MOSEK = False

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


def _check_stabilizer(stabilizer_circ_list, code):
    # code (list,np)
    ret = []
    for q0 in code:
        ret.append([np.vdot(q0, x.apply_state(q0)) for x in stabilizer_circ_list])
    ret = np.array(ret)
    return ret

def test_code523():
    code = numqi.qec.generate_code523()
    code_np = numqi.qec.generate_code_np(code['encode'], code['num_logical_dim'])
    qweA,qweB = numqi.qec.get_weight_enumerator(code_np, use_circuit=True, tagB=True)
    assert np.abs(qweA - np.array([1,0,0,0,15,0])).max() < 1e-7
    assert np.abs(qweB - np.array([1,0,0,30,15,18])).max() < 1e-7
    z0 = _check_stabilizer(code['stabilizer'], code_np)
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


def test_is_code_feasible():
    assert numqi.qec.is_code_feasible(7, 3, 3, solver='CLARABEL') #True
    assert not numqi.qec.is_code_feasible(7, 1, 4, solver='CLARABEL') #infeasible
    if USE_MOSEK: #fail if using "CLARABEL" solver
        assert not numqi.qec.is_code_feasible(8, 9, 3, solver='MOSEK', drop_constraint=[2])
        assert not numqi.qec.is_code_feasible(8, 9, 3, solver='MOSEK', drop_constraint=[12, 14])
        tmp2 = {16, 18, 20, 22, 26, 28, 30, 32, 34, 36, 38, 40, 42, 46, 48, 50, 52, 54, 56, 58, 62, 64, 66, 68, 70, 74, 76, 78, 82}
        drop_constraint = [10,11,12,13] + sorted(set(range(15,86))-tmp2)
        assert not numqi.qec.is_code_feasible(10, 5, 4, solver='MOSEK', drop_constraint=drop_constraint)


def test_get_Krawtchouk_polynomial():
    # https://en.wikipedia.org/wiki/Kravchuk_polynomials
    case_dict = dict()
    case_dict[(2,0)] = np.array([[1]])
    case_dict[(2,1)] = np.array([[0,1],[-2,0]])
    case_dict[(2,2)] = np.array([[0,-0.5,0.5],[0,-2,0],[2,0,0]])
    case_dict[(2,3)] = np.array([[0,1/3,-1/2,1/6],[-2/3,1,-1,0],[0,2,0,0],[-4/3,0,0,0]])
    for (q,k),v in case_dict.items():
        assert np.abs(numqi.qec.get_Krawtchouk_polynomial(q,k)-v).max() < 1e-10


def test_code_get_Krawtchouk_polynomial():
    for key in ['523','shor','steane']:
        code,info = numqi.qec.get_code_subspace(key)
        weightA = info['qweA']
        weightB = info['qweB']
        dim = code.shape[0]
        num_qubit = len(weightA) - 1
        # https://arxiv.org/abs/2408.10323 eq(16)
        assert abs(weightA.sum()*dim - 2**num_qubit) < 1e-10
        for k in range(num_qubit+1):
            np0 = numqi.qec.get_Krawtchouk_polynomial(q=4, k=k)
            npx = np.arange(num_qubit+1)
            tmp0 = npx.reshape(-1,1)**np.arange(k+1)
            RHS = weightA @ (tmp0 @ np0 @ num_qubit**np.arange(k+1)) * (dim*dim/2**num_qubit)
            LHS = weightB[k] * (dim)
            # https://arxiv.org/abs/2408.10323 eq(15)
            assert abs(LHS-RHS) < 1e-6



def test_is_code_feasible_linear_programming():
    pass_list = [(5,2,3), (6,2,3), (7,3,3), (8,9,3)]
    fail_list = [(5,3,3), (6,3,3), (7,4,3), (8,10,3)]
    for x in pass_list:
        assert numqi.qec.is_code_feasible_linear_programming(*x)[0]
    for x in fail_list:
        assert not numqi.qec.is_code_feasible_linear_programming(*x)[0]

    # K=1
    pass_list = [(5,3), (6,4), (7,4), (8,4), (9,4)]
    fail_list = [(5,4), (6,5), (7,5), (8,5), (9,5)]
    for n,d in pass_list:
        assert numqi.qec.is_code_feasible_linear_programming(n,1,d)[0]
    for n,d in fail_list:
        assert not numqi.qec.is_code_feasible_linear_programming(n,1,d)[0]

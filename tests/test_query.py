import numpy as np
import torch

import numqi

try:
    import mosek
    USE_MOSEK = True
except ImportError:
    USE_MOSEK = False

if torch.get_num_threads()!=1:
    torch.set_num_threads(1)

# 24 seconds
def test_QueryGroverModel():
    # arxiv2205.07449 table-1
    case_list = [
        (3, 1, 0.21875), #1s
        (3, 2, 0), #1s
        (4, 2, 0.09155), #3s
        (4, 3, 0), #2s
        # (5, 3, 0.1031), #30s
        # (5, 4, 0), #13s
    ]
    for num_qubit,num_query,ret_ in case_list:
        model = numqi.query.QueryGroverModel(num_qubit, num_query, use_fractional=False, dtype='float64')
        theta_optim = numqi.optimize.minimize(model, theta0=('uniform', -1, 1), tol=1e-12, num_repeat=3, early_stop_threshold=1e-4, print_every_round=0)
        assert abs(ret_-model.error_rate) < 1e-4


def _QueryGroverQuantumModel_build_circuit(num_qubit, num_layer, num_query, use_fractional):
    def hf_one_block():
        for _ in range(num_layer):
            tmp0 = list(range(0, num_qubit-1, 2)) + list(range(1, num_qubit-1, 2))
            for ind1 in tmp0:
                circ.ry(ind1)
                circ.rx(ind1)
                circ.ry(ind1+1)
                circ.rx(ind1+1)
                circ.cnot(ind1, ind1+1)
    circ = numqi.sim.Circuit(default_requires_grad=True)
    if use_fractional:
        circ.register_custom_gate('oracle', numqi.query.FractionalGroverOracle)
    else:
        circ.register_custom_gate('oracle', numqi.query.GroverOracle)
    for _ in range(num_query):
        hf_one_block()
        circ.oracle(num_qubit)
    hf_one_block()
    return circ


## TOO slow
# def test_QueryGroverQuantumModel():
#     case_list = [
#         (3, 2, 8, 0, 233), #about 20 seconds (1 round)
#         # (4, 3, 24, 0, None), #about 400 steps
#     ]
#     kwargs = dict(theta0=('uniform', 0, 2*np.pi), tol=1e-9, num_repeat=1, print_every_round=0, print_freq=50, early_stop_threshold=1e-7)
#     for num_qubit,num_query,num_layer,ret_,seed in case_list:
#         circuit = _QueryGroverQuantumModel_build_circuit(num_qubit, num_layer, num_query, use_fractional=False)
#         model = numqi.query.QueryGroverQuantumModel(circuit)
#         theta_optim = numqi.optimize.minimize(model, seed=seed, **kwargs)
#         assert abs(model.error_rate-ret_) < 1e-4


def test_grover_sdp():
    case_list = [
        (3, 1, 0.2187), #2s
        (3, 2, 0), #2s
        # (4, 2, 0.091468), #30s, fail for use_limit=True
        # (4, 3, 0), #50s
    ]
    for num_qubit,num_query,ret_ in case_list:
        ret0 = numqi.query.grover_sdp(num_qubit, num_query, use_limit=False)
        assert abs(ret_-ret0) < (1e-4 if USE_MOSEK else 1e-3)
        ret1 = numqi.query.grover_sdp(num_qubit, num_query, use_limit=True)
        assert abs(ret_-ret1) < (1e-4 if USE_MOSEK else 1e-3)


# about 30 seconds
def test_HammingQueryQuditModel_hamming_modulo():
    # arxiv2205.07449
    num_bit = 5
    num_modulo = 5
    num_query = 4
    dim_query = num_bit + 1 #why?
    partition = [2,2,3,3,2]
    num_XZ = None #100 None

    bitmap = numqi.query.get_hamming_modulo_map(num_bit, num_modulo)
    model = numqi.query.HammingQueryQuditModel(num_query, dim_query, partition, bitmap, num_XZ, use_fractional=False, alpha_upper_bound=None)
    theta_optim = numqi.optimize.minimize(model, 'uniform', num_repeat=1, tol=1e-12, print_freq=0, early_stop_threshold=1e-6, seed=236)
    # 5500 steps to converge
    assert model.error_rate < 1e-5
    # strange, `pytest -n auto` always give a large error_rate

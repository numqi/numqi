import numpy as np
import opt_einsum

import numqi

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

def test_np_operator_expectation_dm(num_qubit=3):
    assert num_qubit>=3
    ind0,ind1 = np.sort(np.random.permutation(num_qubit)[:2]).tolist()
    np0 = numqi.random.rand_density_matrix(2**num_qubit)
    operator = np.random.randn(4,4) + 1j*np.random.randn(4,4)

    tmp0 = operator.reshape(2,2,2,2)
    tmp1 = [ind0,ind1,ind0+num_qubit,ind1+num_qubit]
    tmp2 = [(np.eye(2),[x,x+num_qubit]) for x in sorted(set(range(num_qubit))-{ind0,ind1})]
    tmp3 = list(range(2*num_qubit))
    operator_pad = opt_einsum.contract(tmp0, tmp1, *(y for x in tmp2 for y in x), tmp3).reshape(2**num_qubit,2**num_qubit)
    ret_ = np.trace(operator_pad @ np0)

    ret0 = numqi.sim.dm.operator_expectation(np0, operator, [ind0,ind1])
    assert hfe(ret_, ret0) < 1e-7

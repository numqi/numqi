import numpy as np
import cvxpy

def _hf_grover_oracle(num_qubit):
    N0 = 2**num_qubit
    matE = [np.ones((N0,N0), dtype=np.int64)]
    for x in range(N0):
        tmp0 = np.ones((N0,N0), dtype=np.int64)
        tmp0[x] = -1
        tmp0[:,x] = -1
        tmp0[x,x] = 1
        matE.append(tmp0)
    matF = np.zeros((N0,N0,N0))
    tmp0 = np.arange(N0)
    matF[tmp0,tmp0,tmp0] = 1
    return matE,matF


def grover_sdp(num_qubit, num_query, use_limit=False, return_matrix=False):
    matE,matF = _hf_grover_oracle(num_qubit)
    # matE (Gamma in the paper), matF (Delta in the paper)
    epsilon = cvxpy.Variable()
    N0 = 2**num_qubit
    not_use_limit = not use_limit

    matM = [[cvxpy.Variable((N0,N0), symmetric=True) for _ in range(N0 + not_use_limit)] for _ in range(num_query)]
    matG = [cvxpy.Variable((N0,N0),symmetric=True) for _ in range(N0)]
    constrains = [x>>0 for x in matG]
    constrains += [y>>0 for x in matM for y in x]
    constrains += [sum(matM[0]) == np.ones((N0,N0))]
    constrains += [(sum(cvxpy.multiply(matE[y+1],matM[x-1][y]) for y in range(N0))==sum(matM[x])) for x in range(1,num_query)]
    constrains += [sum(cvxpy.multiply(matE[x+use_limit],matM[-1][x]) for x in range(N0+not_use_limit))==sum(matG)]
    constrains += [cvxpy.multiply(cvxpy.diag(matG[x]),cvxpy.diag(matF[x]))>=(1-epsilon)*cvxpy.diag(matF[x]) for x in range(N0)]
    # constrains += [cvxpy.diag(matG[x])>=(1-epsilon)*cvxpy.diag(matF[x]) for x in range(N0)]

    prob = cvxpy.Problem(cvxpy.Minimize(epsilon), constrains)
    prob.solve()
    if return_matrix:
        tmp0 = {
            'matG': [x.value for x in matG],
            'matM': [[y.value for y in x[:(num_qubit+1)]] for x in matM],
        }
        ret = epsilon.value, tmp0
    else:
        ret = epsilon.value
    return ret

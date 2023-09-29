import numpy as np
import scipy.stats

import numqi

np_rng = np.random.default_rng()

def is_positive_semi_definite(np0):
    # https://math.stackexchange.com/a/13311
    # https://math.stackexchange.com/a/87538
    # Sylvester's criterion
    try:
        np.linalg.cholesky(np0)
        ret = True
    except np.linalg.LinAlgError:
        ret = False
    return ret


dimA = 3
dimB = 3
num_sample = 1000

alpha = np.ones(dimA*dimB)/2

dm_list = []
ret = []
sampler = scipy.stats.dirichlet(alpha)
for _ in range(num_sample):
    matU = numqi.random.rand_haar_unitary(dimA*dimB)
    tmp0 = sampler.rvs()
    rho = (matU*tmp0) @ matU.T.conj()
    dm_list.append(rho)
    rho_pt = rho.reshape(dimA,dimB,dimA,dimB).transpose(0,3,2,1).reshape(dimA*dimB,-1)
    ret.append(is_positive_semi_definite(rho_pt))
ret = np.array(ret)

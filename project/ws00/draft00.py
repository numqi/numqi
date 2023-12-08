import numpy as np
import numqi

def rand_symplectic_matrix(N0):
    # https://the-walrus.readthedocs.io/en/latest/code/random.html
    np_rng = numqi.random.get_numpy_rng()
    tmp0 = numqi.random.rand_haar_unitary(N0)
    matO = np.block([[tmp0.real, -tmp0.imag], [tmp0.imag, tmp0.real]])
    tmp1 = numqi.random.rand_haar_unitary(N0)
    matP = np.block([[tmp1.real, -tmp1.imag], [tmp1.imag, tmp1.real]])
    tmp2 = np.exp(np.abs(np_rng.normal(size=N0) + np_rng.normal(size=N0)*1j))
    tmp3 = np.concatenate([tmp2, 1/tmp2])
    ret = (matO * tmp3) @ matP
    return ret

N0 = 5
np0 = rand_symplectic_matrix(N0)
omega = np.kron(np.array([[0,-1],[1,0]]), np.eye(N0))
print(np.abs(np0.T @ omega @ np0 - omega).max())

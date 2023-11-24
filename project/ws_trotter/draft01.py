import numpy as np

# diamond norm
def get_kraus_operator_diamond_norm(kop):
    # https://qetlab.com/DiamondNorm
    # kop(N0, dim_out, dim_in)
    assert (kop.ndim == 3)
    tmp0 = np.einsum(kop, [0,1,2], kop.conj(), [0,1,3], [2,3], optimize=True)
    ret = np.linalg.norm(tmp0, ord=2) #the largest eigenvalue
    return ret

np0 = np.array([[1,2,3,4], [0,1,2,0], [1,1,-1,3]]).reshape(-1,2,2) #(N0, dim_out, dim_in)
get_kraus_operator_diamond_norm(np0) #37.65097169808497


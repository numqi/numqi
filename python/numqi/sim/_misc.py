import numpy as np

from .circuit import Circuit
from .state import new_base

def build_graph_state(adjacent_mat, return_stabilizer_circ=False):
    assert (adjacent_mat.ndim==2) and (adjacent_mat.shape[0]==adjacent_mat.shape[1])
    assert (adjacent_mat.dtype==np.uint8) and (adjacent_mat.max()<=1)
    N0 = adjacent_mat.shape[0]
    tmp0,tmp1 = np.nonzero(adjacent_mat)
    edge_list = sorted({((x,y) if x<y else (y,x)) for x,y in zip(tmp0.tolist(),tmp1.tolist())})
    circ = Circuit()
    for ind0 in range(N0):
        circ.H(ind0)
    for ind0,ind1 in edge_list:
        circ.cz(ind0,ind1)
    q0 = circ.apply_state(new_base(N0)).real
    if return_stabilizer_circ:
        stabilizer_circ_list = []
        for ind0 in range(N0):
            circ = Circuit()
            circ.X(ind0)
            for x in np.nonzero(adjacent_mat[ind0])[0]:
                circ.Z(x)
            stabilizer_circ_list.append(circ)
        ret = q0, stabilizer_circ_list
    else:
        ret = q0
    return ret

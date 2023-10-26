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


def get_all_non_isomorphic_graph(num_node:int):
    # https://www.graphclasses.org/smallgraphs.html
    assert num_node in {2,3,4,5}
    if num_node==2:
        graph = [[], [(0,1)]]
    elif num_node==3:
        graph = [
            [], [(0,1)], [(0,2), (1,2)],
            [(0,1)], [(0,1),(1,2)],
        ]
    elif num_node==4:
        graph = [
            [], [(0,1),(1,2),(0,2),(0,3),(1,3),(2,3)],
            [(0,1)], [(0,1),(0,2),(1,2),(3,1),(3,2)],
            [(0,1),(1,2)], [(0,1),(0,2),(1,2),(0,3)],
            [(0,1),(2,3)], [(0,1),(1,2),(2,3),(0,3)],
            [(0,1),(0,2),(0,3)], [(0,1),(1,2),(0,2)],
            [(0,1),(1,2),(2,3)],
        ]
    elif num_node==5:
        graph = [
            [], [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)],
            [(0,1)], [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4)],
            [(0,1),(1,2)], [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3),(0,4),(1,4)],
            [(0,1),(2,3)], [(0,1),(1,2),(2,3),(0,3),(0,4),(1,4),(0,4),(1,4)],
            [(0,1),(0,2),(0,3)], [(0,1),(1,2),(2,3),(0,3),(0,4),(1,4),(0,4)],
            [(0,1),(1,2),(3,4)], [(0,1),(1,2),(2,3),(0,3),(0,4),(1,4),(2,4)],
            [(0,1),(1,2),(2,3)], [(0,1),(1,2),(2,3),(0,4),(1,4),(2,4),(3,4)],
            [(0,1),(1,2),(0,2)], [(0,1),(1,2),(0,2),(3,0),(3,1),(4,0),(4,1)],
            [(0,1),(0,2),(0,3),(0,4)], [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)],
            [(0,1),(1,2),(2,3),(3,0)], [(0,1),(0,2),(0,3),(0,4),(1,2),(3,4)],
            [(0,1),(0,2),(0,3),(3,4)], [(0,1),(1,2),(2,3),(3,0),(1,3),(0,4)],
            [(0,1),(1,2),(2,0),(0,3)], [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3)],
            [(0,1),(1,2),(2,3),(3,4)], [(0,1),(1,2),(2,3),(3,0),(0,4),(1,4)],
            [(0,1),(1,2),(2,0),(3,4)], [(0,1),(1,2),(2,3),(3,0),(0,4),(1,4)],
            [(0,1),(1,2),(2,3),(3,0),(0,4)], [(0,1),(0,2),(0,3),(1,2),(3,4)],
            [(0,1),(1,2),(2,0),(0,3),(1,4)],
            [(0,1),(0,2),(0,3),(0,4),(1,2)], [(0,1),(1,2),(2,3),(3,0),(0,2)],
            [(0,1),(1,2),(2,3),(3,4),(4,0)],
        ]
    ret = []
    for edge_list in graph:
        tmp0 = np.zeros((num_node,num_node), dtype=np.uint8)
        if len(edge_list):
            tmp1 = np.array(edge_list)
            tmp0[tmp1[:,0], tmp1[:,1]] = 1
            tmp0[tmp1[:,1], tmp1[:,0]] = 1
        ret.append(tmp0)
    ret = np.stack(ret, axis=0)
    return graph,ret

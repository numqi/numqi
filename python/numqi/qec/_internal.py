import numpy as np

def hf_state(x:str):
    r'''convert state string to vector, should NOT be used in performance-critical code

    Parameters:
        x (str): state string, e.g. '0000 1111', supporting sign "+-i", e.g. '0000 -i1111'

    Returns:
        ret (np.ndarray): shape=(2**n,), state vector
    '''
    tmp0 = x.strip().split(' ')
    tmp1 = []
    for x in tmp0:
        if x.startswith('-i'):
            tmp1.append((-1j, x[2:]))
        elif x.startswith('i'):
            tmp1.append((1j, x[1:]))
        elif x.startswith('-'):
            tmp1.append((-1, x[1:]))
        else:
            x = x[1:] if x.startswith('+') else x
            tmp1.append((1, x))
    tmp1 = sorted(tmp1, key=lambda x:x[1])
    num_qubit = len(tmp1[0][1])
    assert all(len(x[1])==num_qubit for x in tmp1)
    index = np.array([int(x[1],base=2) for x in tmp1], dtype=np.int64)
    coeff = np.array([x[0] for x in tmp1])
    coeff = coeff/np.linalg.norm(coeff, ord=2)
    ret = np.zeros(2**num_qubit, dtype=coeff.dtype)
    ret[index] = coeff
    return ret
_state_hf0 = hf_state


def generate_code_np(circ, num_logical_dim):
    num_qubit = circ.num_qubit
    ret = []
    for ind0 in range(num_logical_dim):
        q0 = np.zeros(2**num_qubit, dtype=np.complex128)
        q0[ind0] = 1
        circ.apply_state(q0)
        ret.append(circ.apply_state(q0))
    ret = np.stack(ret, axis=0)
    return ret


def build_graph_state(adjacent:np.ndarray):
    # http://arxiv.org/abs/0704.2122v1
    # Nonadditive quantum error-correcting code eq(1)
    assert isinstance(adjacent, np.ndarray) and (adjacent.ndim==2) and (adjacent.shape[0]==adjacent.shape[1])
    assert (adjacent.dtype==np.uint8) and (adjacent.max()<=1)
    N0 = adjacent.shape[0]
    bit = (((np.arange(1<<N0)[:, None]) >> np.arange(N0-1,-1,-1)) & 1).astype(np.uint8)
    sign = 1-2*((np.einsum(adjacent.astype(np.uint32), [0,1], bit, [2,0], bit, [2,1], [2], optimize=True) // 2) % 2).astype(np.int64)
    psi = sign * ((1/2)**(N0/2))
    return psi


def build_CWS_code(adjacent:np.ndarray, codeword:np.ndarray):
    psi = build_graph_state(adjacent)
    tmp0 = np.array([1,1], dtype=np.int8)
    tmp1 = np.array([0,-2], dtype=np.int8)
    N0,N1 = codeword.shape
    sign = tmp0 + codeword[:,:1]*tmp1
    for ind2 in range(1,N1):
        tmp2 = tmp0 + codeword[:,ind2].reshape(N0,1,1)*tmp1
        sign = (sign.reshape(N0,-1,1) * tmp2).reshape(N0, -1)
    code = sign * psi
    return code


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

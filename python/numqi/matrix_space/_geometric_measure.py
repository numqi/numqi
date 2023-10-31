import numpy as np
import cvxpy

from ._misc import get_vector_orthogonal_basis, get_bipartition_list

def get_geometric_measure_ppt(state):
    assert (state.ndim==2) or (state.ndim==3)
    dimA = state.shape[-2]
    dimB = state.shape[-1]
    if state.ndim==2:
        tmp0 = state.reshape(-1) / np.linalg.norm(state.reshape(-1))
        projector_orth = np.eye(dimA*dimB) - tmp0[:, np.newaxis] * tmp0.conj()
    else:
        tmp0 = state.reshape(state.shape[0], -1)
        basis_orth = get_vector_orthogonal_basis(tmp0)
        projector_orth = basis_orth.T @ basis_orth.conj()
    rho = cvxpy.Variable(projector_orth.shape, hermitian=True)
    constraints = [
        rho>>0,
        cvxpy.trace(rho)==1,
        cvxpy.partial_transpose(rho, [dimA,dimB], axis=0)>>0,
    ]
    obj = cvxpy.Minimize(cvxpy.real(cvxpy.trace(projector_orth @ rho)))
    prob = cvxpy.Problem(obj, constraints)
    prob.solve()
    return prob.value


def get_generalized_geometric_measure_ppt(state, dim_list, bipartition_list=None):
    if len(dim_list)==2:
        ret = get_geometric_measure_ppt(state)
    else:
        if bipartition_list is None:
            bipartition_list = get_bipartition_list(len(dim_list))
        else:
            bipartition_list = sorted({tuple(sorted({int(y) for y in x})) for x in bipartition_list})
        dim_list = tuple(int(x) for x in dim_list)
        N0 = np.prod(dim_list)
        assert (len(dim_list)>=2) and all(x>1 for x in dim_list)
        assert (state.ndim==len(dim_list)) or (state.ndim==len(dim_list)+1)
        if state.ndim==len(dim_list):
            assert state.shape==dim_list
            tmp0 = state.reshape(-1) / np.linalg.norm(state.reshape(-1))
            projector_orth = np.eye(N0) - tmp0[:, np.newaxis] * tmp0.conj()
        else:
            assert state.shape[1:]==dim_list #matrix subspace
            basis_orth = get_vector_orthogonal_basis(state.reshape(state.shape[0], -1))
            projector_orth = basis_orth.T @ basis_orth.conj()
        projector_orth = projector_orth.reshape(dim_list+dim_list)
        rho = cvxpy.Variable((N0,N0), hermitian=True)
        ret = []
        for bipartition in bipartition_list:
            permutation = tuple(bipartition) + tuple(sorted(set(range(len(dim_list))) - set(bipartition)))
            projector_orth_i = np.transpose(projector_orth, permutation + tuple(x+len(dim_list) for x in permutation)).reshape(N0, -1)
            tmp0 = np.prod([dim_list[x] for x in bipartition]).item()
            constraints = [
                rho>>0,
                cvxpy.trace(rho)==1,
                cvxpy.partial_transpose(rho, [tmp0,N0//tmp0], axis=0)>>0,
            ]
            obj = cvxpy.Minimize(cvxpy.real(cvxpy.trace(projector_orth_i @ rho)))
            prob = cvxpy.Problem(obj, constraints)
            prob.solve()
            ret.append(prob.value)
        ret = min(ret)
    return ret


def get_GES_Maciej2019(d:int, num_party:int=2, theta:float=np.pi/2, xi:float=0):
    assert d>=2
    assert num_party>=2
    dim = (d-1)**(num_party-1)
    tmp0 = np.arange(d-1, dtype=np.int64)
    ind0 = tmp0
    ind1 = tmp0 + 1
    for _ in range(num_party-2):
        ind0 = ((ind0*d).reshape(-1,1) + tmp0).reshape(-1)
        ind1 = ((ind1*d).reshape(-1,1) + (tmp0 + 1)).reshape(-1)
    tmp0 = np.arange(dim, dtype=np.int64)
    ret = np.zeros([dim, 2, d**(num_party-1)], dtype=np.complex128)
    ret[tmp0, 0, ind0] = np.cos(theta/2)
    ret[tmp0, 1, ind1] = np.exp(1j*xi)*np.sin(theta/2)
    ret = ret.reshape([dim, 2] + [d]*(num_party-1))
    return ret


def get_GM_Maciej2019(d:int, theta:float):
    ret = 0.5 - 0.5*np.sqrt(np.maximum(1 - np.sin(theta)**2 * np.sin(np.pi/d)**2, 0))
    return ret

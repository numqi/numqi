import numpy as np
import scipy.sparse.linalg
from tqdm.auto import tqdm

import numqi.gellmann
import numqi.matrix_space
import numqi.utils

def _ree_bisection_solve(hf0, x0, x1, xtol, threshold, use_tqdm):
    # function must be increasing, otherwise the result is not correct
    # the REE function is okay
    # do not evaluate hf0 on x0 or x1, the REE might be nan
    assert x0<x1
    maxiter = int(np.ceil(np.log2(max(2, (x1-x0)/xtol))))
    tmp0 = tqdm(range(maxiter)) if use_tqdm else range(maxiter)
    history_info = []
    for _ in tmp0:
        xi = (x0 + x1)/2
        yi = hf0(xi)
        history_info.append((xi,yi))
        if yi>=threshold:
            x1 = xi
        else:
            x0 = xi
    history_info = np.array(sorted(history_info, key=lambda x: x[0]))
    return xi,history_info


def _sdp_ree_solve(rho, use_tqdm, cvx_rho, cvxP, prob, obj, return_info, is_single_item):
    ret = []
    hf0 = lambda x: np.ascontiguousarray(x.value)
    for rho_i in (tqdm(rho) if use_tqdm else rho):
        cvx_rho.value = rho_i
        prob.solve()
        ree = np.trace(rho_i @ scipy.linalg.logm(rho_i)).real + obj.value
        assert ree > -1e-4, str(ree) #for zero value, the prob.value will be around -1e-6
        # assert ree>-1e-5 #fail with solver=SCS
        ree = max(ree, 0)
        if return_info:
            info = {
                'X': hf0(cvxP['X']),
                'Xpow': np.stack([hf0(x) for x in cvxP['Xpow']], axis=0),
                'T': np.stack([hf0(x) for x in cvxP['T']], axis=0),
                'mlogX': hf0(cvxP['mlogX']),
            }
            ret.append((ree,info))
        else:
            ret.append(ree)
    if is_single_item:
        ret = ret[0]
    else:
        if not return_info:
            ret = np.array(ret)
    return ret


def hf_interpolate_dm(dm0, alpha=None, beta=None, dm_norm=None):
    assert (alpha is not None) or (beta is not None)
    N0 = dm0.shape[0]
    rhoI = np.eye(N0)/N0
    if beta is not None:
        if dm_norm is None:
            dm_norm = numqi.gellmann.dm_to_gellmann_norm(dm0)
        alpha = beta / dm_norm
    ret = alpha*dm0 + (1-alpha)*rhoI
    return ret


def get_density_matrix_boundary(dm, dm_norm=None):
    r'''return the boundary of the density matrix

    Args:
        dm (np.ndarray): density matrix, 2d array
        dm_norm (float): norm of the density matrix. Defaults to None.
        return_alpha (bool, optional): whether to return `alpha`. Defaults to False.
        return_both (bool, optional): whether to return both the upper and lower boundary. Defaults to False.

    Returns:
        beta_l (float): minimum value of beta
        beta_u (float): maximum value of beta
    '''
    # TODO remove return_alpha, return_both
    # return: alpha, beta, dm
    assert (dm.ndim==2) and (dm.shape[0]==dm.shape[1])
    N0 = dm.shape[0]
    if dm_norm is None:
        dm_norm = numqi.gellmann.dm_to_gellmann_norm(dm)
    tmp0 = (np.linalg.eigvalsh(dm) - 1/N0)/dm_norm
    beta_l = -1/(N0*tmp0[-1])
    beta_u = -1/(N0*tmp0[0])
    return beta_l,beta_u


def get_density_matrix_plane(dm0, dm1):
    vec0 = numqi.gellmann.dm_to_gellmann_basis(dm0)
    vec1 = numqi.gellmann.dm_to_gellmann_basis(dm1)
    norm0 = np.linalg.norm(vec0)
    basis0 = vec0 / norm0
    norm1 = np.linalg.norm(vec1)
    tmp0 = vec1 - np.dot(basis0, vec1) * basis0
    basis0 = vec0 / norm0
    basis1 = tmp0 / np.linalg.norm(tmp0)
    theta1 = np.arccos(np.dot(vec0, vec1)/(norm0*norm1))
    def hf0(theta_or_xy, norm=1):
        if hasattr(theta_or_xy, '__len__'):
            x,y = theta_or_xy
            ret = x * basis0 + y * basis1
        else:
            theta = theta_or_xy
            tmp0 = norm*(basis0*np.cos(theta) + basis1*np.sin(theta))
            ret = numqi.gellmann.gellmann_basis_to_dm(tmp0)
        return ret
    return theta1,hf0


def check_swap_witness(rho, eps=-1e-7, return_info=False):
    assert (rho.ndim==2) and (rho.shape[0]==rho.shape[1])
    dimA = int(np.sqrt(rho.shape[0]))
    assert dimA*dimA==rho.shape[0]
    rho.reshape(dimA,dimA,dimA,dimA).transpose(0,1,3,2)
    tmp0 = np.einsum(rho.reshape(dimA,dimA,dimA,dimA), [0,1,1,0], [], optimize=True).real.item()
    tmp1 = tmp0 > eps
    ret = (tmp1,tmp0) if return_info else tmp1
    return ret

# TODO, can positive linear map be parameterized
def check_reduction_witness(rho, dim=None, eps=-1e-7):
    # https://www.quantiki.org/wiki/reduction-criterion
    # weaker than PPT
    N0 = rho.shape[0]
    if dim is None:
        tmp0 = int(np.sqrt(N0))
        assert tmp0*tmp0==N0
        dim = [tmp0,tmp0]
    assert (len(dim)>1) and (rho.shape[1]==N0) and (np.prod(dim)==N0) and all(x>1 for x in dim)
    def hf0(i):
        tmp0 = np.prod(dim[:i]) if i>0 else 1
        tmp1 = np.prod(dim[(i+1):]) if (i+1)<len(dim) else 1
        tmp2 = rho.reshape(tmp0,dim[i],tmp1,tmp0,dim[i],tmp1)
        tmp3 = np.einsum(tmp2, [0,1,2,0,4,2], [1,4], optimize=True)
        if tmp0!=1:
            tmp3 = np.kron(np.eye(tmp0), tmp3)
        if tmp1!=1:
            tmp3 = np.kron(tmp3, np.eye(tmp1))
        EVL = scipy.sparse.linalg.eigsh(tmp3-rho, k=1, sigma=None, which='SA', return_eigenvectors=False)[0]
        return EVL
    ret = all(hf0(i)>eps for i in range(len(dim)))
    return ret


def get_werner_state(d, alpha):
    # https://en.wikipedia.org/wiki/Werner_state
    # https://www.quantiki.org/wiki/werner-state
    # alpha = ((1-2*p)*d+1) / (1-2*p+d)
    # alpha: [-1,1]
    # SEP: [-1,1/d]
    # (1,k)-ext: [-1, (k+d^2-d)/(kd+d-1)]
    # https://journals.aps.org/pra/abstract/10.1103/PhysRevA.88.032323
    assert d>1
    assert (-1<=alpha) and (alpha<=1)
    pmat = np.eye(d**2).reshape(d,d,d,d).transpose(0,1,3,2).reshape(d**2,d**2)
    ret = (np.eye(d**2)-alpha*pmat) / (d**2-d*alpha)
    return ret


def get_werner_state_ree(d, alpha):
    # REE(relative entropy of entangement)
    if alpha<=1/d:
        ret = 0
    else:
        rho0 = get_werner_state(d, alpha)
        rho1 = get_werner_state(d, 1/d)
        ret = numqi.utils.get_relative_entropy(rho0, rho1, kind='infinity')
    return ret


def get_isotropic_state(d, alpha):
    # https://www.quantiki.org/wiki/isotropic-state
    # alpha: [-1/(d^2-1), 1]
    # SEP: [-1/(d^2-1), 1/(d+1)]
    # (1,k)-ext: [-1/(d^2-1),(kd+d^2-d-k)/(k(d^2-1))]
    # https://journals.aps.org/pra/abstract/10.1103/PhysRevA.88.032323
    assert d>1
    assert ((-1/(d**2-1))<=alpha) and (alpha<=1) #beyond this range, the density matrix is not SDP
    tmp0 = np.eye(d).reshape(-1)
    ret = ((1-alpha)/d**2) * np.eye(d**2) + (alpha/d) * (tmp0[:,np.newaxis]*tmp0)
    return ret


def get_isotropic_state_ree(d, alpha):
    if alpha<=1/(d+1):
        ret = 0
    else:
        rho0 = get_isotropic_state(d, alpha)
        rho1 = get_isotropic_state(d, 1/(d+1))
        ret = numqi.utils.get_relative_entropy(rho0, rho1, kind='infinity')
    return ret


# no idea why this function is here @zhangc20230221
# def copy_numpy_to_cp(np0, cp0):
#     assert (np0.size==cp0.size) and (np0.itemsize==cp0.itemsize)
#     cp0.data.copy_from_host(np0.__array_interface__['data'][0], np0.size*np0.itemsize)


def product_state_to_dm(ketA, ketB, probability):
    tmp0 = (ketA[:,:,np.newaxis]*ketB[:,np.newaxis]).reshape(ketA.shape[0],-1)
    ret = np.sum((tmp0[:,:,np.newaxis]*tmp0[:,np.newaxis].conj())*probability[:,np.newaxis,np.newaxis], axis=0)
    return ret


def get_max_entangled_state(dim):
    ret = np.diag(np.ones(dim)*1/np.sqrt(dim))
    ret = ret.reshape(-1)
    return ret


def get_quantum_negativity(rho, dimA):
    dimB = rho.shape[0]//dimA
    tmp0 = rho.reshape(dimA, dimB, dimA, dimB).transpose(0,3,2,1).reshape(dimA*dimB,dimA*dimB)
    ret = (np.abs(np.linalg.eigvals(tmp0)).sum()-1) / 2
    return ret


def get_density_matrix_numerical_range(op0, op1, num_point=100):
    assert (op0.ndim==2) and (op0.shape[0]==op0.shape[1]) and op0.shape==op1.shape
    assert np.abs(op0-op0.T.conj()).max() < 1e-10
    assert np.abs(op1-op1.T.conj()).max() < 1e-10
    tmp0 = numqi.matrix_space.get_matrix_numerical_range(op0, op1, num_point)
    ret = np.stack([tmp0.real, tmp0.imag], axis=1)
    return ret


def _check_input_rho_SDP(rho, dim, use_tqdm=False, eps0=1e-10, eps1=1e-6, tag_positive=True):
    # if rho is 2-dim, then expand to 3-dim
    rho = np.asarray(rho)
    assert len(dim)==2
    dimA = int(dim[0])
    dimB = int(dim[1])
    assert rho.ndim in {2,3}
    is_single_item = rho.ndim==2
    if is_single_item:
        rho = rho[np.newaxis]
    assert (rho.shape[1]==rho.shape[2]) and (rho.shape[1]==dimA*dimB)
    assert abs(np.trace(rho,axis1=1,axis2=2)-1).max() < eps0
    assert np.abs(rho-rho.transpose(0,2,1).conj()).max() < eps0
    if tag_positive:
        assert np.linalg.eigvalsh(rho).min() > -eps1
    use_tqdm = use_tqdm and (rho.shape[0]>1)
    return rho,is_single_item,dimA,dimB,use_tqdm

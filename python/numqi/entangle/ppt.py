import functools
import itertools
import numpy as np
from tqdm import tqdm
import scipy.optimize
import scipy.sparse.linalg
import cvxpy

import numqi.gellmann

from ._misc import get_density_matrix_boundary, _sdp_ree_solve, _check_input_rho_SDP

def get_ppt_numerical_range(op0, op1, dim, num_theta=None, theta_list=None, use_tqdm=True):
    # num_theta=400 TODO default to None
    assert (num_theta is None) != (theta_list is None)
    if num_theta is not None:
        theta_list = np.linspace(0, 2*np.pi, int(num_theta))
    else:
        theta_list = np.asarray(theta_list)
        assert theta_list.ndim==1
    assert len(dim)==2
    dimA,dimB = dim
    N0 = dimA*dimB
    assert (op0.shape==op1.shape) and (op0.shape==(N0,N0))

    cvx_rho = cvxpy.Variable((N0,N0), hermitian=True)
    cvx_op = cvxpy.Parameter((N0,N0), hermitian=True)
    obj = cvxpy.Maximize(cvxpy.real(cvxpy.trace(cvx_op @ cvx_rho)))
    constraints = [
        cvx_rho>>0,
        cvxpy.trace(cvx_rho)==1,
        cvxpy.partial_transpose(cvx_rho, [dimA,dimB], 0)>>0,
    ]
    prob = cvxpy.Problem(obj, constraints)
    ret = []
    for theta_i in (tqdm(theta_list) if use_tqdm else theta_list):
        cvx_op.value = np.cos(theta_i)*op0 + np.sin(theta_i)*op1
        prob.solve()
        tmp0 = cvx_rho.value.copy()
        ret.append([np.trace(x @ tmp0).real for x in [op0,op1]])
    ret = np.array(ret)
    return ret


def get_ppt_boundary(dm, dim, dm_norm=None, within_dm=True):
    assert len(dim)==2
    dimA,dimB = dim
    assert dm.shape==(dimA*dimB,dimA*dimB)
    if dm_norm is None:
        dm_norm = numqi.gellmann.dm_to_gellmann_norm(dm)
    tmp0 = dm.reshape(dimA,dimB,dimA,dimB).transpose(0,3,2,1).reshape(dimA*dimB,dimA*dimB)
    beta_pt_l,beta_pt_u = get_density_matrix_boundary(tmp0, dm_norm=dm_norm)
    if within_dm:
        beta_l,beta_u = get_density_matrix_boundary(dm, dm_norm=dm_norm)
        beta_pt_l = max(beta_l, beta_pt_l)
        beta_pt_u = min(beta_u, beta_pt_u)
    return beta_pt_l,beta_pt_u


def is_ppt(rho, dim=None, eps=-1e-7, return_info=False):
    # Positive Partial Transpose (PPT)
    # https://en.wikipedia.org/wiki/Entanglement_witness
    # https://en.wikipedia.org/wiki/Peres%E2%80%93Horodecki_criterion
    N0 = rho.shape[0]
    if dim is None:
        tmp0 = int(np.sqrt(N0))
        assert tmp0*tmp0==N0
        dim = [tmp0,tmp0]
    assert (len(dim)>1) and (rho.shape[1]==N0) and (np.prod(dim)==N0) and all(x>1 for x in dim)
    def hf0(i):
        tmp0 = np.prod(dim[:i]) if i>0 else 1
        tmp1 = np.prod(dim[(i+1):]) if (i+1)<len(dim) else 1
        rhoT = rho.reshape(tmp0,dim[i],tmp1,tmp0,dim[i],tmp1).transpose(0,4,2,3,1,5).reshape(N0,N0)
        EVL = scipy.sparse.linalg.eigsh(rhoT, k=1, sigma=None, which='SA', return_eigenvectors=False)[0]
        return EVL
    tmp0 = (hf0(i) for i in range(len(dim)))
    if return_info:
        tmp1 = list(tmp0)
        ret = all(x>eps for x in tmp1), tmp1
    else:
        ret = all(x>eps for x in tmp0)
    return ret


@functools.lru_cache
def _is_generalized_ppt_dim_list(num_partite):
    tmp0 = set(range(2*num_partite))
    z0 = [(y, tuple(sorted(tmp0-set(y)))) for x in range(1, num_partite) for y in itertools.combinations(range(2*num_partite), x)]
    z1 = sorted(set([tuple(sorted((x,tuple(sorted(tmp0-set(x)))))) for x in itertools.combinations(tmp0, num_partite)]))
    ret = tuple([((), tuple(range(2*num_partite)))] + z0 + z1)
    return ret


def is_generalized_ppt(rho, dim, return_info=False):
    '''Generalized Positive Partial Transpose (PPT)

    Parameters:
        rho (np.ndarray): density matrix
        dim (tuple[int]): tuple of integers
        return_info (bool): whether to return the list of nuclear norms

    Returns:
        tag (bool): whether rho is generalized PPT (superset of SEP)
        info (list[float]): list of nuclear norms
    '''
    # doi.org/10.1016/S0375-9601(02)01538-4
    assert (rho.ndim==2) and rho.shape[0]==rho.shape[1]
    N0 = rho.shape[0]
    dim = tuple(int(x) for x in dim)
    assert (len(dim)>1) and (np.prod(dim)==N0) and all(x>1 for x in dim)

    dim_list = _is_generalized_ppt_dim_list(len(dim))
    shape = dim + dim
    rho = rho.reshape(shape)
    ret = []
    for dim0,dim1 in dim_list:
        tmp0 = 1 if len(dim0)==0 else np.prod([1]+[shape[x] for x in dim0])
        tmp1 = rho.transpose(*dim0, *dim1).reshape(tmp0, -1)
        # nuclear norm: sum of singular values
        ret.append((dim0, dim1, np.linalg.norm(tmp1, ord='nuc')))
        if (not return_info) and (ret[-1][2]>1):
            break
    tag = all(x[2]<=1 for x in ret)
    ret = (tag,ret) if return_info else tag
    return ret


def get_generalized_ppt_boundary(dm, dim, threshold=1e-10, xtol=1e-5):
    '''get boundary according to Generalized Positive Partial Transpose (PPT) criterion

    Parameters:
        dm (np.ndarray): density matrix
        dim (tuple[int]): tuple of integers
        threshold (float): threshold for the nuclear norm
        xtol (float): tolerance for the root finding

    Returns:
        beta (float): boundary in Gell-Mann space
    '''
    def hf0(x):
        tmp0 = rho0 + x*dm_unit_vec
        tmp1 = is_generalized_ppt(tmp0, dim, return_info=True)[1]
        ret = max([x[2] for x in tmp1]) - (1 + threshold)
        return ret
    dm_norm = numqi.gellmann.dm_to_gellmann_norm(dm)
    rho0 = np.eye(dm.shape[0])/dm.shape[0]
    dm_unit_vec = (dm - rho0) / dm_norm
    beta_dm = get_density_matrix_boundary(dm, dm_norm=dm_norm)[1]
    if hf0(beta_dm)<0:
        ret = beta_dm
    else:
        ret = scipy.optimize.root_scalar(hf0, bracket=[0, beta_dm], xtol=xtol).root
    return ret


# see cvxquad https://github.com/hfawzi/cvxquad
# TODO add docs for this part
def cvx_matrix_xlogx(cvxX, sqrt_order=3, pade_order=3):
    # user's duty to make sure cvxX is SDP
    is_complex = cvxX.is_complex()
    assert (len(cvxX.shape)==2) and (cvxX.shape[0]==cvxX.shape[1])
    assert (sqrt_order>=2) and (pade_order>=2)
    n = cvxX.shape[0]

    # initially in range [-1,1], convert to range [0,1]
    tmp0 = np.polynomial.legendre.leggauss(pade_order)
    node = (tmp0[0]+1)/2
    weight = tmp0[1]/2
    assert node.min()>1e-6

    if is_complex:
        hf0 = lambda: cvxpy.Variable((n,n), hermitian=True)
    else:
        hf0 = lambda: cvxpy.Variable((n,n), symmetric=True)
    cvxP = {
        'X': cvxX,
        'XlogX': hf0(),
        'T': [hf0() for _ in range(pade_order)],
        'Xpow': [hf0() for _ in range(sqrt_order)],
    }
    tmp0 = cvxP['Xpow'][1:] + [np.eye(n)]
    constraint = [(cvxpy.bmat([[cvxX, x], [x, y]]) >> 0) for x,y in zip(cvxP['Xpow'], tmp0)]
    tmp0 = [cvxX-(s/w)*t for s,w,t in zip(node,weight,cvxP['T'])]
    tmp1 = [(1-s)*cvxX+s*cvxP['Xpow'][0] for s in node]
    constraint += [(cvxpy.bmat([[x, cvxX], [cvxX, y]]) >> 0) for x,y in zip(tmp0,tmp1)]
    constraint += [(2**sqrt_order)*sum(cvxP['T']) + cvxP['XlogX'] >> 0]
    return cvxP, constraint


# see cvxquad https://github.com/hfawzi/cvxquad
# TODO add docs for this part
def cvx_matrix_mlogx(cvxX, sqrt_order=3, pade_order=3):
    # user's duty to make sure cvxX is SDP
    is_complex = cvxX.is_complex()
    assert (len(cvxX.shape)==2) and (cvxX.shape[0]==cvxX.shape[1])
    assert (sqrt_order>=2) and (pade_order>=2)
    dim = cvxX.shape[0]

    # initially in range [-1,1], convert to range [0,1]
    tmp0 = np.polynomial.legendre.leggauss(pade_order)
    node = (tmp0[0]+1)/2
    weight = tmp0[1]/2
    assert node.min()>1e-6

    if is_complex:
        hf0 = lambda: cvxpy.Variable((dim,dim), hermitian=True)
    else:
        hf0 = lambda: cvxpy.Variable((dim,dim), symmetric=True)
    cvxP = {
        'X': cvxX,
        'Xpow': [hf0() for _ in range(sqrt_order)],
        'T': [hf0() for _ in range(pade_order)],
        'mlogX': hf0(), #-log(X)
    }
    eye = np.eye(dim)
    tmp0 = cvxP['Xpow'][1:] + [cvxP['X']]
    constraint = [(cvxpy.bmat([[eye,x], [x,y]])>>0) for x,y in zip(cvxP['Xpow'], tmp0)]
    tmp0 = [(eye-(s/w)*x) for s,w,x in zip(node, weight, cvxP['T'])]
    tmp1 = [((1-s)*eye+s*cvxP['Xpow'][0]) for s in node]
    constraint += [(cvxpy.bmat([[x,eye], [eye,y]])>>0) for x,y in zip(tmp0,tmp1)]
    constraint += [(2**sqrt_order)*sum(cvxP['T']) + cvxP['mlogX'] >> 0]
    return cvxP, constraint


def get_ppt_ree(rho, dimA, dimB, return_info=False, sqrt_order=3, pade_order=3, use_tqdm=False):
    rho,is_single_item,dimA,dimB,use_tqdm = _check_input_rho_SDP(rho, (dimA,dimB), use_tqdm)
    dim = dimA * dimB
    cvxX = cvxpy.Variable((dim,dim), hermitian=True)
    cvxP, constraint = cvx_matrix_mlogx(cvxX, sqrt_order=sqrt_order, pade_order=pade_order)
    # cvxP['X'] is cvxX
    constraint += [
        cvxX>>0,
        cvxpy.trace(cvxX)==1,
        cvxpy.partial_transpose(cvxX, [dimA,dimB], 1)>>0,
    ]
    cvx_rho = cvxpy.Parameter((dimA*dimB,dimA*dimB), hermitian=True)
    obj = cvxpy.Minimize(cvxpy.real(cvxpy.trace(cvx_rho @ cvxP['mlogX'])))
    prob = cvxpy.Problem(obj, constraint)
    ret = _sdp_ree_solve(rho, use_tqdm, cvx_rho, cvxP, prob, obj, return_info, is_single_item)
    return ret
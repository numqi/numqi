import numpy as np
import scipy.sparse.linalg
import scipy.linalg
import scipy.integrate
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import numqi.gellmann
import numqi.matrix_space
import numqi.utils

# TODO docs/api
# TODO rename _density_matrix to _dm

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


def hf_interpolate_dm(rho:np.ndarray, alpha:float|None=None, beta:float|None=None, dm_norm:float|None=None):
    r'''interpolate the density matrix between `rho` and the maximally mixed state

    Parameters:
        rho (np.ndarray): density matrix, `ndim=2`
        alpha (float,None): `rho0*(1-alpha) + rho*alpha`, return `rho` itself if `alpha=1`, and maximally mixed state if `alpha=0`,
                     if None, then use `beta`
        beta (float,None): `rho0 + beta*unit(rho0)`, `beta` reflects the vector length in Gell-Mann basis, if None, then use `alpha`
        dm_norm (float,None): norm of the density matrix. if None, then calculate it internally

    Returns:
        ret (np.ndarray): interpolated density matrix, `ndim=2`
    '''
    assert (alpha is not None) or (beta is not None)
    N0 = rho.shape[0]
    rhoI = np.eye(N0)/N0
    if beta is not None:
        if dm_norm is None:
            dm_norm = numqi.gellmann.dm_to_gellmann_norm(rho)
        alpha = beta / dm_norm
    ret = alpha*rho + (1-alpha)*rhoI
    return ret


def get_density_matrix_boundary(dm, dm_norm=None):
    r'''return the boundary of the density matrix

    Parameters:
        dm (np.ndarray): density matrix, 2d array (support batch)
        dm_norm (float,NoneType): norm of the density matrix. if None, then calculate it internally

    Returns:
        beta_l (float): minimum value of beta
        beta_u (float): maximum value of beta
    '''
    assert (dm.ndim>=2) and (dm.shape[-2]==dm.shape[-1])
    shape = dm.shape
    N0 = shape[-1]
    dm = dm.reshape(-1, N0, N0)
    if dm_norm is None:
        dm_norm = numqi.gellmann.dm_to_gellmann_norm(dm)
    else:
        dm_norm = np.asarray(dm_norm)
        if dm_norm.size==1:
            dm_norm = dm_norm.reshape(1)
        else:
            assert dm_norm.shape==shape[:-2]
            dm_norm = dm_norm.reshape(-1)
    assert np.abs(dm-dm.transpose(0,2,1).conj()).max() < 1e-10, 'dm must be Hermitian'
    tmp0 = (np.linalg.eigvalsh(dm) - 1/N0)/dm_norm.reshape(-1,1)
    beta_l = -1/(N0*tmp0[:,-1])
    beta_u = -1/(N0*tmp0[:,0])
    if len(shape)==2:
        beta_l,beta_u = beta_l[0],beta_u[0]
    else:
        beta_l,beta_u = beta_l.reshape(shape[:-2]),beta_u.reshape(shape[:-2])
    return beta_l,beta_u


def get_density_matrix_plane(op0:np.ndarray, op1:np.ndarray):
    r'''return the plane spanned by two Hermitian operators

    Parameters:
        op0 (np.ndarray): Hermitian operator, `ndim=2`, no need to be unit trace
        op1 (np.ndarray): Hermitian operator, `ndim=2`, no need to be unit trace

    Returns:
        theta1 (float): angle between op0 and op1 in the Gell-Mann basis
        hf0 (function): function to generate the density matrix on the plane.
            function signature: `hf0(theta_or_xy:float|tuple[float], norm:float=1) -> np.ndarray`.
            if `theta_or_xy` is a float, then return the density matrix with the given angle and `norm`.
            if `theta_or_xy` is a tuple, then return the density matrix with the given x,y in the plane.
    '''
    vec0 = numqi.gellmann.dm_to_gellmann_basis(op0)
    vec1 = numqi.gellmann.dm_to_gellmann_basis(op1)
    norm0 = np.linalg.norm(vec0)
    basis0 = vec0 / norm0
    norm1 = np.linalg.norm(vec1)
    tmp0 = vec1 - np.dot(basis0, vec1) * basis0
    basis0 = vec0 / norm0
    basis1 = tmp0 / np.linalg.norm(tmp0)
    theta1 = np.arccos(np.dot(vec0, vec1)/(norm0*norm1))
    def hf0(theta_or_xy:float|tuple[float], norm:float=1):
        if hasattr(theta_or_xy, '__len__'):
            x,y = theta_or_xy
            ret = x * basis0 + y * basis1
        else:
            theta = theta_or_xy
            tmp0 = norm*(basis0*np.cos(theta) + basis1*np.sin(theta))
            ret = numqi.gellmann.gellmann_basis_to_dm(tmp0)
        return ret
    return theta1,hf0


def check_swap_witness(rho:np.ndarray, eps:float=-1e-7):
    r'''return whether the density matrix passes the swap witness criterion

    Parameters:
        rho (np.ndarray): density matrix, `ndim=2`
        eps (float): threshold for the swap witness

    Returns:
        ret (bool): whether the density matrix passes the swap witness criterion
    '''
    assert (rho.ndim==2) and (rho.shape[0]==rho.shape[1])
    dimA = int(np.sqrt(rho.shape[0]))
    assert dimA*dimA==rho.shape[0]
    rho.reshape(dimA,dimA,dimA,dimA).transpose(0,1,3,2)
    tmp0 = np.einsum(rho.reshape(dimA,dimA,dimA,dimA), [0,1,1,0], [], optimize=True).real.item()
    ret = tmp0 > eps
    return ret


def check_reduction_witness(rho:np.ndarray, dim:tuple[int], eps:float=-1e-7):
    r'''return whether the density matrix passes the reduction criterion
    [quantiki-link](https://www.quantiki.org/wiki/reduction-criterion)

    weaker than PPT

    TODO, can positive linear map be parameterized

    Parameters:
        rho (np.ndarray): density matrix, `ndim=2`
        dim (tuple[int]): dimension of the density matrix, `(dimA,dimB,dimC,...)`
        eps (float): threshold for the reduction witness

    Returns:
        ret (bool): whether the density matrix passes the reduction criterion
    '''
    N0 = rho.shape[0]
    assert (rho.ndim==2) and (rho.shape[0]==rho.shape[1])
    dim = numqi.utils.hf_tuple_of_int(dim)
    assert (len(dim)>1) and (np.prod(dim)==rho.shape[0]) and all(x>1 for x in dim)
    def hf0(i):
        tmp0 = np.prod(dim[:i]) if i>0 else 1
        tmp1 = np.prod(dim[(i+1):]) if (i+1)<len(dim) else 1
        tmp2 = rho.reshape(tmp0,dim[i],tmp1,tmp0,dim[i],tmp1)
        tmp3 = np.einsum(tmp2, [0,1,2,0,4,2], [1,4], optimize=True)
        if tmp0!=1:
            tmp3 = np.kron(np.eye(tmp0), tmp3)
        if tmp1!=1:
            tmp3 = np.kron(tmp3, np.eye(tmp1))
        if N0>=5: #5 is chosen intuitively
            EVL = scipy.sparse.linalg.eigsh(tmp3-rho, k=1, sigma=None, which='SA', return_eigenvectors=False)[0]
        else:
            EVL = np.linalg.eigvalsh(tmp3-rho).min()
        return EVL
    ret = all(hf0(i)>eps for i in range(len(dim)))
    return ret


# def product_state_to_dm(ketA:np.ndarray, ketB:np.ndarray, probability):
#     tmp0 = (ketA[:,:,np.newaxis]*ketB[:,np.newaxis]).reshape(ketA.shape[0],-1)
#     ret = np.sum((tmp0[:,:,np.newaxis]*tmp0[:,np.newaxis].conj())*probability[:,np.newaxis,np.newaxis], axis=0)
#     return ret


def get_negativity(rho:np.ndarray, dim:tuple[int]):
    r'''return the negativity of the density matrix
    [wiki-link](https://en.wikipedia.org/wiki/Negativity_(quantum_mechanics))

    Parameters:
        rho (np.ndarray): density matrix, `ndim=2`
        dim (tuple[int]): dimension of the density matrix, `(dimA,dimB)`

    Returns:
        ret (float): negativity of the density matrix
    '''
    assert len(dim)==2
    dimA = int(dim)
    dimB = int(dim)
    assert (rho.ndim==2) and (rho.shape[0]==dimA*dimB) and (rho.shape[0]==rho.shape[1])
    assert np.abs(rho-rho.T.conj()).max() < 1e-10
    tmp0 = rho.reshape(dimA, dimB, dimA, dimB).transpose(0,3,2,1).reshape(dimA*dimB,dimA*dimB)
    ret = (np.abs(np.linalg.eigvals(tmp0)).sum()-1) / 2
    return ret


def get_dm_numerical_range(op0:np.ndarray, op1:np.ndarray, num_point:int=100):
    r'''return the joint algebra numerical range of two Hermitian operators (projection)

    Parameters:
        op0 (np.ndarray): Hermitian operator, `ndim=2`
        op1 (np.ndarray): Hermitian operator, `ndim=2`
        num_point (int): number of points to sample

    Returns:
        ret (np.ndarray): joint algebra numerical range, `ndim=2`, `shape=(num_point,2)`
    '''
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


def _cos_sin_wrapper(theta, order:int):
    theta = np.asarray(theta)
    assert theta.ndim<=1
    if theta.ndim==1:
        tmp0 = np.arange(order+1)[:,np.newaxis]*theta
        ret = np.concatenate([np.cos(tmp0), np.sin(tmp0[1:])], axis=1)
    else:
        tmp0 = np.arange(order+1)*theta
        ret = np.concatenate([np.cos(tmp0), np.sin(tmp0[1:])], axis=0)
    return ret


def get_dm_cross_section_moment(op0:np.ndarray, op1:np.ndarray, order:int=1,
            quad_epsrel:float=1e-8, dim:None|tuple[int]=None, kind:str='dm'):
    r'''return the moment of the cross section of the density matrix or PPT boundary spanned by two Hermitian operators

    Parameters:
        op0 (np.ndarray): traceless Hermitian operators, `ndim=2`
        op1 (np.ndarray): traceless Hermitian operators, `ndim=2`
        order (int): order of the moment, for dimension=3/4/5, order=1 is enough. larger order might be needed for larger dimension
        quad_epsrel (float): relative tolerance for the quadrature
        dim (None|tuple[int]): dimension of the density matrix, `(dimA,dimB)`, only used when `kind='ppt'`
        kind (str): 'dm' for density matrix, 'ppt' for PPT boundary

    Returns:
        ret (np.ndarray): moment of the cross section of the density matrix or PPT boundary, `ndim=1`
    '''
    assert (op0.ndim==2) and (op0.shape[0]==op0.shape[1]) and (op0.shape==op1.shape)
    for x in [op0,op1]:
        assert abs(np.trace(x)) < 1e-10
        assert np.abs(x.conj().T - x).max() < 1e-10
    assert kind in {'dm','ppt'}
    if kind=='ppt':
        assert (dim is not None) and (len(dim)==2)
        dim = int(dim[0]), int(dim[1])
        assert op0.shape[0]==dim[0]*dim[1]
    order = int(order)
    assert order>=1
    hf_plane = numqi.entangle.get_density_matrix_plane(op0, op1)[1]
    if kind=='dm':
        hf0 = lambda x: numqi.entangle.get_density_matrix_boundary(hf_plane(x))[1] * _cos_sin_wrapper(x, order)
    else: #ppt
        hf0 = lambda x: numqi.entangle.get_ppt_boundary(hf_plane(x), dim)[1] * _cos_sin_wrapper(x, order)
    ret = scipy.integrate.quad_vec(hf0, 0, 2*np.pi, epsrel=quad_epsrel)[0]
    return ret


def is_dm_cross_section_similar(moment0:np.ndarray, moment1:np.ndarray, zero_eps:float=0.001):
    r'''return whether two moments are similar

    Parameters:
        moment0 (np.ndarray): moment of the cross section of the density matrix, `ndim=1`
        moment1 (np.ndarray): moment of the cross section of the density matrix, `ndim=1`
        zero_eps (float): threshold for zero

    Returns:
        ret (bool): whether two moments are similar
    '''
    assert (moment0.ndim==1) and (moment0.shape==moment1.shape) and (moment0.shape[0]%2==1)
    xc = moment0[:(moment0.shape[0]//2 + 1)].copy().astype(np.complex128)
    xc[1:] += 1j*moment0[(moment0.shape[0]//2 + 1):]
    yc = moment1[:(moment1.shape[0]//2 + 1)].copy().astype(np.complex128)
    yc[1:] += 1j*moment1[(moment1.shape[0]//2 + 1):]
    if np.abs(np.abs(xc)-np.abs(yc)).max() > zero_eps:
        return False
    if (len(xc)==1) or (np.abs(xc[1:]).max() < zero_eps):
        return True
    ind0 = np.argmax(np.abs(xc[1:])) + 1
    phase = np.angle(yc[ind0]/xc[ind0])/ind0
    tmp0 = np.exp(1j*np.arange(len(xc))*phase)
    if np.abs(xc*tmp0-yc).max() < zero_eps:
        return True
    yc = yc.conj()
    phase = np.angle(yc[ind0]/xc[ind0])/ind0
    tmp0 = np.exp(1j*np.arange(len(xc))*phase)
    if np.abs(xc*tmp0-yc).max() < zero_eps:
        return True
    return False


def group_dm_cross_section_moment(moment:np.ndarray, zero_eps:float=0.0001):
    r'''group the moments of the cross section of the density matrix

    Parameters:
        moment (np.ndarray): moments of the cross section of the density matrix, `ndim=2`
        zero_eps (float): threshold for zero

    Returns:
        group_list (list[list[int]]): list of groups of indices
    '''
    moment = np.asarray(moment, dtype=np.float64)
    assert moment.ndim==2
    group_list = []
    for ind0,moment_i in enumerate(moment):
        for group in group_list:
            if is_dm_cross_section_similar(moment[group[0]], moment_i, zero_eps):
                group.append(ind0)
                break
        else:
            group_list.append([ind0])
    return group_list

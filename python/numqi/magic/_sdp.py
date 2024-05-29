import numpy as np
import cvxpy
from tqdm.auto import tqdm

from ..entangle.ppt import cvx_matrix_mlogx
from ..gellmann import matrix_to_gellmann_basis

from ._internal import get_Heisenberg_Weyl_operator


def _get_thauma_sdp_min(rho, cvx_sigma, wigner_trace_norm, use_tqdm):
    dim = rho.shape[-1]
    cvx_rho_conj_vec = cvxpy.Parameter(dim*dim, complex=True)
    tmp0 = cvxpy.reshape(cvx_sigma, dim*dim, order='C')
    obj = cvxpy.Maximize(cvxpy.sum(cvxpy.real(cvxpy.multiply(cvx_rho_conj_vec, tmp0))))
    constraint = [wigner_trace_norm<=1, cvx_sigma>>0]
    prob = cvxpy.Problem(obj, constraint)
    ret = []
    rho_conj_vec_list = np.conj(rho.reshape(-1,dim*dim))
    for rho_i in (tqdm(rho_conj_vec_list) if use_tqdm else rho_conj_vec_list):
        cvx_rho_conj_vec.value = rho_i
        prob.solve()
        ret.append(-np.log(prob.value))
    return ret

def _get_thauma_sdp_max(rho, cvx_sigma, wigner_trace_norm, use_tqdm):
    dim = rho.shape[-1]
    cvx_eta = cvxpy.Variable()
    cvx_rho = cvxpy.Parameter((dim,dim), complex=True)
    constraint = [wigner_trace_norm<=1, cvx_sigma>>(cvx_eta*cvx_rho)]
    obj = cvxpy.Maximize(cvx_eta)
    prob = cvxpy.Problem(obj, constraint)
    ret = []
    for rho_i in (tqdm(rho) if use_tqdm else rho):
        cvx_rho.value = rho_i
        prob.solve()
        ret.append(-np.log(prob.value))
    return ret


def _get_thauma_sdp_entropy(rho, cvx_sigma, wigner_trace_norm, use_tqdm, sqrt_order, pade_order):
    dim = rho.shape[-1]
    cvx_rho_conj_vec = cvxpy.Parameter(dim*dim, complex=True)
    cvxP, constraint = cvx_matrix_mlogx(cvx_sigma, sqrt_order=sqrt_order, pade_order=pade_order)
    constraint += [cvx_sigma>>0, wigner_trace_norm<=1]
    tmp0 = cvxpy.reshape(cvxP['mlogX'], dim*dim, order='C')
    obj = cvxpy.Minimize(cvxpy.sum(cvxpy.real(cvxpy.multiply(cvx_rho_conj_vec, tmp0))))
    prob = cvxpy.Problem(obj, constraint)

    ret = []
    rho_conj_vec_list = np.conj(rho.reshape(-1,dim*dim))
    for rho_i in (tqdm(rho_conj_vec_list) if use_tqdm else rho_conj_vec_list):
        cvx_rho_conj_vec.value = rho_i
        prob.solve()
        EVL = np.linalg.eigvalsh(rho_i.reshape(dim,dim))
        tmp0 = np.dot(EVL, np.log(np.maximum(EVL, 1e-10))) + obj.value
        assert tmp0 > -1e-4, str(tmp0) #for zero value, the prob.value will be around -1e-6
        # assert tmp0>-1e-5 #fail with solver=SCS
        ret.append(max(tmp0, 0))
    return ret


def get_thauma_sdp(rho, kind:str, use_tqdm:bool=False, sqrt_order:int=3, pade_order:int=3):
    r'''get the thauma of quantum state using semi-definite programming (SDP)

    reference: Efficiently Computable Bounds for Magic State Distillation
    [doi-link](https://doi.org/10.1103%2FPhysRevLett.124.090505)

    Parameters:
        rho (np.ndarray): the quantum state(s) of shape (dim, dim) or (batch_size, dim, dim)
        kind (str): the kind of thauma, can be {'min', 'max', 'entropy'}
        use_tqdm (bool): whether to use tqdm for progress bar
        sqrt_order (int): the order of square root approximation
        pade_order (int): the order of Pade approximation

    Returns:
        ret (np.ndarray): the thauma of quantum state(s)
    '''
    assert kind in {'min','max','entropy'}
    rho = np.asarray(rho)
    assert rho.ndim in {2,3}
    dim = rho.shape[-1]
    is_single_item = rho.ndim==2
    if is_single_item:
        rho = rho[np.newaxis]
    use_tqdm = use_tqdm and (rho.shape[0]>1)
    _,weyl_A = get_Heisenberg_Weyl_operator(dim)

    cvx_sigma = cvxpy.Variable((dim,dim), hermitian=True) #order=C on ubuntu with Mosek solver
    tmp0 = cvxpy.reshape(cvx_sigma, dim*dim, order='C')
    wigner_trace_norm = cvxpy.sum(cvxpy.abs(cvxpy.real(weyl_A.conj().reshape(dim*dim, dim*dim) @ tmp0))) / dim

    if kind=='min':
        ret = _get_thauma_sdp_min(rho, cvx_sigma, wigner_trace_norm, use_tqdm)
    elif kind=='max':
        ret = _get_thauma_sdp_max(rho, cvx_sigma, wigner_trace_norm, use_tqdm)
    elif kind=='entropy':
        ret = _get_thauma_sdp_entropy(rho, cvx_sigma, wigner_trace_norm, use_tqdm, sqrt_order, pade_order)
    ret = np.array(ret)
    if is_single_item:
        ret = ret[0]
    return ret


def get_thauma_boundary(rho:np.ndarray, use_tqdm:bool=False):
    r'''get the thauma boundary of quantum state using semi-definite programming (SDP)

    Parameters:
        rho (np.ndarray): the quantum state(s) of shape (dim, dim) or (batch_size, dim, dim)
        use_tqdm (bool): whether to use tqdm for progress bar

    Returns:
        ret (np.ndarray): the thauma boundary of quantum state(s)
    '''
    rho = np.asarray(rho)
    assert rho.ndim in {2,3}
    is_single_item = rho.ndim==2
    if is_single_item:
        rho = rho[np.newaxis]
    assert np.abs(rho-rho.transpose(0,2,1).conj()).max() < 1e-10
    use_tqdm = use_tqdm and (rho.shape[0]>1)
    dim = rho.shape[-1]
    _,weyl_A = get_Heisenberg_Weyl_operator(dim)

    cvx_beta = cvxpy.Variable()
    cvx_rho_vec_norm = cvxpy.Parameter((dim,dim), hermitian=True)
    cvx_rho = np.eye(dim)/dim + cvx_beta * cvx_rho_vec_norm
    tmp0 = cvxpy.reshape(cvx_rho, dim*dim, order='C')
    wigner_trace_norm = cvxpy.sum(cvxpy.abs(cvxpy.real(weyl_A.conj().reshape(dim*dim, dim*dim) @ tmp0))) / dim
    constraint = [cvx_rho >> 0, wigner_trace_norm<=1]
    obj = cvxpy.Maximize(cvx_beta)
    prob = cvxpy.Problem(obj, constraint)

    ret = []
    for rho_i in (tqdm(rho) if use_tqdm else rho):
        tmp0 = rho_i-np.eye(dim)*(np.trace(rho_i)/dim)
        cvx_rho_vec_norm.value =  tmp0 * (np.sqrt(2)/np.linalg.norm(tmp0, ord='fro'))
        prob.solve()
        ret.append(prob.value)
    ret = np.array(ret)
    return ret


def plot_qubit_thauma_set_3d():
    r'''plot the thauma set (tetrahedron) of qubit

    Parameters:

    Returns:
        fig (matplotlib.figure.Figure): the figure object
        ax (mpl_toolkits.mplot3d.Axes3D): the axis object
    '''
    import scipy.spatial
    import itertools
    import mpl_toolkits.mplot3d
    import matplotlib.pyplot as plt
    np_rng = np.random.default_rng()
    zero_eps = 1e-10
    tmp0 = np.array([(a,b,c,d) for a in [1,-1] for b in [1,-1] for c in [1,-1] for d in [1,-1]])
    tmp1 = matrix_to_gellmann_basis(get_Heisenberg_Weyl_operator(2)[1].reshape(-1,2,2)).real
    matAb = np.stack([x@tmp1 for x in tmp0]) * np.array([1,1,1,0.5]) - np.array([0,0,0,1])

    ind_list = np.array(list(itertools.combinations(range(16), 3)))
    tmp0 = np.linalg.eigvalsh(np.stack([matAb[x,:3] @ matAb[x,:3].T for x in ind_list]))
    ind_list = ind_list[tmp0[:,0] > zero_eps]
    vertex_list = np.stack([np.linalg.solve(matAb[x,:3], -matAb[x,3]) for x in ind_list])
    tmp0 = np.concatenate([vertex_list, np.ones([vertex_list.shape[0],1])],axis=1)
    vertex_list = vertex_list[np.einsum(matAb, [0,1], tmp0, [2,1], [0,2], optimize=True).max(axis=0) < zero_eps]

    fig = plt.figure()
    ax = mpl_toolkits.mplot3d.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    point_list = 2*vertex_list[scipy.spatial.ConvexHull(vertex_list).simplices]
    for x in point_list:
        ax.add_collection3d(mpl_toolkits.mplot3d.art3d.Poly3DCollection([x], facecolors=np_rng.uniform(0,1,3), alpha=0.3, edgecolors='none'))
    tmp0 = np.array([[0,0,1],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[0,0,-1]])
    point_list = tmp0[np.array([[0,1,2],[0,2,3],[0,3,4],[0,4,1],[1,2,5],[2,3,5],[3,4,5],[4,1,5]])]
    for x in point_list:
        ax.add_collection3d(mpl_toolkits.mplot3d.art3d.Poly3DCollection([x], facecolors=np_rng.uniform(0,1,3), edgecolors='none'))
    for x in 'xyz':
        getattr(ax, f'set_{x}label')(f'{x} axis')
        getattr(ax, f'set_{x}lim')([-1.2, 1.2])
        getattr(ax, f'set_{x}ticks')([-1, 0, 1])
    ax.set_aspect('equal')
    return fig,ax

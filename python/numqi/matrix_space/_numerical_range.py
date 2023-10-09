import numpy as np
import scipy.sparse.linalg
import scipy.optimize
import cvxpy
from tqdm.auto import tqdm

from ._misc import get_matrix_orthogonal_basis

def get_matrix_numerical_range(matA, num_point=100):
    r'''get the numerical range of a square matrix A, A could be complex

    $$W(A)=\left\{ x^{\dagger}Ax:x\in\mathbb{C}^{n},\lVert x\rVert_{2}=1\right\}$$

    see https://arxiv.org/abs/2212.12811 appendix A

    Parameters:
        matA (np.ndarray): a square matrix, could be complex
        num_point (int): number of points to sample the numerical range

    Returns:
        ret (np.ndarray): a 1d array of length num_point, dtype is complex, the numerical range of matA
    '''
    assert (matA.ndim==2) and (matA.shape[0]==matA.shape[1])
    assert num_point>=1
    theta_list = np.linspace(0, 2*np.pi, num_point)
    matA_conj = matA.T.conj()
    ret = []
    for theta_i in theta_list:
        tmp0 = np.exp(1j*theta_i)/2
        tmp1 = tmp0 * matA + tmp0.conj() * matA_conj
        EVC = scipy.sparse.linalg.eigsh(tmp1, k=1, which='LA', return_eigenvectors=True)[1][:,0]
        ret.append(np.vdot(EVC, matA @ EVC))
    ret = np.array(ret)
    return ret


def get_matrix_numerical_range_along_direction(matA, alpha, kind='max'):
    r'''get the numerical range of a square matrix A along a direction alpha

    $$W^{\alpha}\left(A\right)=\left\{ x\in\mathbb{R}:xe^{i\alpha}\in W\left(A\right)\right\}$$

    This function might give wrong results especially when the numerical range is not smooth,
    In that case, a warning will be printed.

    see https://arxiv.org/abs/2212.12811 appendix A

    Parameters:
        matA (np.ndarray): a square matrix, could be complex
        alpha (float): the direction to sample the numerical range, in radian
        kind (str): 'max' or 'min', the maximum or minimum value along the direction

    Returns:
        maximum (float): the maximum value along the direction $e^{i\alpha}$
        EVC (np.ndarray): the eigenvector corresponding to the maximum value
    '''
    assert (matA.ndim==2) and (matA.shape[0]==matA.shape[1])
    alpha = np.mod(alpha, 2*np.pi)
    matA_conj = matA.T.conj()
    def hf0(theta, return_info=False):
        tmp0 = np.exp(1j*theta)/2
        matB = tmp0 * matA + tmp0.conj() * matA_conj
        EVC = scipy.sparse.linalg.eigsh(matB, k=1, which='LA', return_eigenvectors=True)[1][:,0]
        tmp2 = np.vdot(EVC, matA @ EVC) / np.exp(1j*alpha)
        tmp3 = np.angle(tmp2) #(-pi, pi]
        ret = np.abs(np.mod(tmp3-np.pi/2, 2*np.pi)-np.pi)-np.pi/2 #(-pi/2, pi/2)
        if return_info:
            ret = ret, matB, EVC, tmp2
        return ret
    tmp0 = [((-alpha) if kind=='max' else (np.pi-alpha))+x for x in (-np.pi/2,0,np.pi/2)]
    tmp1 = [hf0(x) for x in tmp0]
    # tmp0 = (-alpha-np.pi/2, -alpha, -alpha+np.pi/2) if kind=='max' else (np.pi/2-alpha, np.pi-alpha, 3*np.pi/2-alpha)
    # tmp0 = hf0(-alpha-np.pi/2), hf0(-alpha), hf0(-alpha+np.pi/2)
    if tmp1[0]*tmp1[1]<0:
        bracket = tmp0[0],tmp0[1]
    else:
        assert tmp1[1]*tmp1[2]<0
        bracket = tmp0[1],tmp0[2]
    theta_optim = scipy.optimize.root_scalar(hf0, bracket=bracket)
    _,matB,EVC,value = hf0(theta_optim.root, return_info=True)
    if abs(value.imag)>1e-10:
        # degeneracy
        # EVL,EVC = np.linalg.eigh(matB)
        # index_degeneracy = abs(EVL-EVL[-1])<1e-7
        # EVC = EVC[:,index_degeneracy]
        # EVC.T.conj() @ matA @ EVC
        # TODO equivalent to a biquadratic equation problem. hard to solve
        # usually due to the numerical range is not smooth, see https://arxiv.org/abs/2212.12811 example 4
        print('[get_matrix_numerical_range_along_direction()] WARNING: value.imag =', value.imag)
    return value.real, EVC


def get_real_bipartite_numerical_range(mat, kind='min', method='eigen'):
    r'''get the real bipartite numerical range of a square matrix A

    $$B\in\mathbb{R}^{mn\times mn},W^{x+iy}\left(B\right)=W^{\arctan y/x}\left(\frac{xB+iyB^{\Gamma}}{\sqrt{x^{2}+y^{2}}}\right)$$

    where $W^{\arctan y/x}(A)$ is the numerical range of $A$ along the direction $e^{i\arctan y/x}$, see
    `numqi.matrix_space.get_matrix_numerical_range_along_direction()`

    see https://arxiv.org/abs/2212.12811

    Parameters:
        mat (np.ndarray): 4-dimensional numpy array of shape `(dimA,dimB,dimA,dimB)`, must be real
        kind (str): 'min' or 'max', the boundary of numerical range to compute
        method (str): 'rotation' or 'eigen'.
            If 'rotation': use `numqi.matrix_space.get_matrix_numerical_range_along_direction()` to compute the numerical range.
            'rotation' method might give wrong results, especially when numerical range is not smooth.

            If 'eigen': (see theorem 2 in https://arxiv.org/abs/2212.12811) use the eigenvalue decomposition to compute the numerical range.
            the matrix `mat` must be symmetric.

            Two methods **usually** give the same output and one might be faster (TODO benchmark)

    Returns:
        ret (float): the boundary of the real bipartite numerical range
    '''
    assert (mat.ndim==4) and (mat.shape[0]==mat.shape[2]) and (mat.shape[1]==mat.shape[3])
    assert not np.iscomplexobj(mat)
    assert kind in {'min', 'max'}
    assert method in {'rotation', 'eigen'}
    dimA = mat.shape[0]
    dimB = mat.shape[1]
    mat_pt = mat.transpose(0,3,2,1).reshape(dimA*dimB, -1)
    mat = mat.reshape(dimA*dimB, -1)
    if method=='rotation':
        ret = get_matrix_numerical_range_along_direction(mat+1j*mat_pt, np.pi/4, kind)[0]/np.sqrt(2)
    elif method=='eigen':
        assert np.abs(mat - mat.T).max() < 1e-10, 'matrix must be symmetric when method="eigen"'
        if kind=='min':
            hf0 = lambda p: -scipy.sparse.linalg.eigsh(p*mat+(1-p)*mat_pt, k=1, which='SA', return_eigenvectors=False)[0]
            theta_optim = scipy.optimize.minimize_scalar(hf0)
            ret = -theta_optim.fun
        if kind=='max':
            hf0 = lambda p: scipy.sparse.linalg.eigsh(p*mat+(1-p)*mat_pt, k=1, which='LA', return_eigenvectors=False)[0]
            theta_optim = scipy.optimize.minimize_scalar(hf0)
            ret = theta_optim.fun
    return ret


def detect_real_matrix_subspace_rank_one(matrix_subspace):
    r'''detect whether a real matrix subspace has nonzero rank-one element

    Given a series of real matrices $A_i$, the matrix subspace spanned over real field is defined as

    $$\langle A\rangle_\mathbb{R}=\left\{\sum_{i=1}^{n}x_{i}A_{i}: x_{i}\in\mathbb{R}\right\}$$

    Then this function can detect whether $\langle A\rangle_\mathbb{R}$ has nonzero rank-one element.

    see https://arxiv.org/abs/2212.12811 theorem 3

    Parameters:
        matrix_subspace (np.ndarray): 3-dimensional numpy array of shape `(N0,N1,N1)`, must be real

    Returns:
        tag_rank_one (bool): if `False`, then all nonzero elements in $\langle A\rangle_\mathbb{R}$ are of rank at least two.
            If `True`, most likely there exists nonzero element in $\langle A\rangle_\mathbb{R}$ of rank one (might be not).
    '''
    assert (matrix_subspace.ndim==3) and (not np.iscomplexobj(matrix_subspace))
    dimA = matrix_subspace.shape[1]
    dimB = matrix_subspace.shape[2]
    basis = get_matrix_orthogonal_basis(matrix_subspace, field='real')[0]
    tmp0 = basis.reshape(basis.shape[0], dimA*dimB)
    projector = tmp0.T @ tmp0
    upper_bound = get_real_bipartite_numerical_range(projector.reshape(dimA,dimB,dimA,dimB), kind='max')
    if upper_bound < 1:
        tag_rank_one = False
    else:
        tag_rank_one = True #could be wrong
    return tag_rank_one, upper_bound


def get_joint_algebraic_numerical_range(op_list, direction, return_info=False, use_tqdm=True):
    r'''get the joint algebraic numerical range (JANR) of a list of operators along a direction

    $$ L(A_{1},A_{2},\cdots,A_{r})=\left\{ a\in\mathbb{C}^{r}:\rho\in\mathbb{C}^{d\times d},\rho\succeq0,\mathrm{Tr}[\rho]=1,a_{i}=\mathrm{Tr}[A_{i}\rho]\right\} $$

    $$ \max\;\beta $$

    $$ s.t.\;\begin{cases}
\rho\succeq 0\\
\mathrm{Tr}[\rho]=1\\
\mathrm{Tr}[\rho A_{i}]=\beta\hat{n}_{i} & i=1,\cdots,m
\end{cases} $$

    Parameters:
        op_list (list): a list of operators, each operator is a 2d numpy array
        direction (np.ndarrray): the boundary along the direction will be calculated, if 2d, then each row is a direction
        return_info (bool): if `True`, then return the boundary and the boundary's normal vector
        use_tqdm (bool): if `True`, then use tqdm to show the progress

    Returns:
        beta (np.ndarray): the distance from the origin to the boundary along the direction.
            If `direction` is 2d, then `beta` is 1d array.
        boundary (np.ndarray): the boundary along the direction. only returned if `return_info` is `True`
        normal_vector (np.ndarray): the normal vector of the boundary. only returned if `return_info` is `True`
    '''
    op_list = np.stack(op_list, axis=0)
    num_op,dim,_ = op_list.shape
    assert np.abs(op_list-op_list.transpose(0,2,1).conj()).max() < 1e-10, 'op_list must be Hermitian'
    direction = np.asarray(direction)
    assert (direction.ndim==1) or (direction.ndim==2)
    assert direction.shape[-1]==num_op
    is_single = (direction.ndim==1)
    direction = direction.reshape(-1,num_op)
    if direction.shape[0]==1:
        use_tqdm = False
    cvx_rho = cvxpy.Variable((dim,dim), hermitian=True)
    cvx_vec = cvxpy.Parameter(num_op)
    cvx_beta = cvxpy.Variable()
    cvx_op = cvxpy.real(op_list.transpose(0,2,1).reshape(-1, dim*dim, order='C') @ cvxpy.reshape(cvx_rho, dim*dim, order='F'))
    constraints = [
        cvx_rho>>0,
        cvxpy.real(cvxpy.trace(cvx_rho))==1,
        cvx_beta*cvx_vec==cvx_op,
    ]
    cvx_obj = cvxpy.Maximize(cvx_beta)
    prob = cvxpy.Problem(cvx_obj, constraints)
    obj_list = []
    boundary_list = []
    norm_vec_list = []
    ret = []
    for vec_i in (tqdm(direction) if use_tqdm else direction):
        cvx_vec.value = vec_i
        prob.solve()
        obj_list.append(cvx_obj.value)
        if return_info:
            boundary_list.append(cvx_op.value.copy())
            norm_vec_list.append(constraints[-1].dual_value.copy())
    if is_single:
        if return_info:
            ret = (obj_list[0], boundary_list[0], norm_vec_list[0])
        else:
            ret = obj_list[0]
    else:
        obj_list = np.array(obj_list)
        if return_info:
            ret = obj_list, np.stack(boundary_list, axis=0), np.stack(norm_vec_list, axis=0)
        else:
            ret = obj_list
    return ret


def draw_line_list(ax, xydata, norm_theta_list, kind='norm', color='#ABABAB', radius=2.5, label=None):
    assert kind in {'norm', 'tangent'}
    N0 = len(norm_theta_list)
    assert xydata.shape==(N0,2)
    xdata = np.zeros((N0,3), dtype=np.float64)
    ydata = np.zeros((N0,3), dtype=np.float64)
    xdata[:,2] = np.nan
    ydata[:,2] = np.nan
    if kind=='tangent':
        xdata[:,0] = xydata[:,0] + radius*np.cos(norm_theta_list-np.pi/2)
        xdata[:,1] = xydata[:,0] + radius*np.cos(norm_theta_list+np.pi/2)
        ydata[:,0] = xydata[:,1] + radius*np.sin(norm_theta_list-np.pi/2)
        ydata[:,1] = xydata[:,1] + radius*np.sin(norm_theta_list+np.pi/2)
    else:
        xdata[:,0] = xydata[:,0]
        xdata[:,1] = xydata[:,0] + radius*np.cos(norm_theta_list)
        ydata[:,0] = xydata[:,1]
        ydata[:,1] = xydata[:,1] + radius*np.sin(norm_theta_list)
    ax.plot(xdata.reshape(-1), ydata.reshape(-1), color=color, label=label)

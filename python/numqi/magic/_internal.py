import functools
import numpy as np
import mpl_toolkits
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
import sympy.ntheory


def get_Weyl_H(dim:int):
    r'''get the Weyl-Heisenberg operator H, same as `same as numqi.gate.get_quditH`

    Weyl-Heisenberg matrices [wiki-link](https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices)

    Parameters:
        dim (int): the dimension

    Returns:
        ret (np.ndarray): the Weyl-Heisenberg operator H of shape (dim,dim)
    '''
    tmp0 = np.exp(2j*np.pi*np.arange(dim)/dim)
    ret = np.vander(tmp0, dim, increasing=True) / np.sqrt(dim)
    return ret


def get_Weyl_Z(dim:int):
    r'''get the Weyl-Heisenberg operator Z, same as `same as numqi.gate.get_quditZ`

    Weyl-Heisenberg matrices [wiki-link](https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices)

    Parameters:
        dim (int): the dimension

    Returns:
        ret (np.ndarray): the Weyl-Heisenberg operator Z of shape (dim,dim)
    '''
    ret = np.diag(np.exp(2j*np.pi*np.arange(dim)/dim))
    return ret


def get_Weyl_X(dim:int):
    r'''get the Weyl-Heisenberg operator X

    Weyl-Heisenberg matrices [wiki-link](https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices)

    Parameters:
        dim (int): the dimension

    Returns:
        ret (np.ndarray): the Weyl-Heisenberg operator X of shape (dim,dim)
    '''
    ret = np.diag(np.ones(dim-1), -1)
    ret[0,-1] = 1
    return ret


@functools.lru_cache
def _get_Heisenberg_Weyl_operator_hf0(dim:int, _kind:str='normal'):
    assert (dim>=2) and sympy.ntheory.isprime(dim)
    assert _kind in {'normal', 'dagger'}
    z_diag = np.diag(get_Weyl_Z(dim))
    tmp0 = np.arange(dim, dtype=np.int64).reshape(-1,1)
    z_power = z_diag**tmp0
    matH = get_Weyl_H(dim)
    x_power = np.einsum(matH.conj(), [0,1], z_power, [3,1], matH, [2,1], [3,0,2], optimize=True)
    tmp0 = np.exp(-(1j*(dim+1)*np.pi/dim) * np.arange(dim).reshape(-1,1) * np.arange(dim))
    T_list = np.einsum(tmp0, [0,1], z_power, [0, 2], x_power, [1,2,3], [0,1,2,3], optimize=True)
    tmp0 = T_list.sum(axis=(0,1))/dim
    A_list = np.einsum(T_list, [0,1,2,3], tmp0, [3,4], T_list.conj(), [0,1,5,4], [0,1,2,5], optimize=True)
    if _kind=='normal':
        ret = T_list,A_list
    elif _kind=='dagger':
        ret = T_list, A_list.reshape(dim*dim, dim*dim).conj().T.copy()
    return ret

def get_Heisenberg_Weyl_operator(dim:int):
    r'''get the Heisenberg-Weyl operator of prime dimension

    Parameters:
        dim (int): the prime dimension

    Returns:
        T_list (np.ndarray): the list of Heisenberg-Weyl operator $T_u$ of shape (dim,dim,dim,dim)
        A_list (np.ndarray): the list of Heisenberg-Weyl operator $A_u$ in matrix form of shape (dim,dim,dim,dim)
    '''
    ret = _get_Heisenberg_Weyl_operator_hf0(int(dim), 'normal')
    return ret


def get_qutrit_nonstabilizer_state(key:str):
    r'''get the non-stabilizer state of qutrit

    reference: The resource theory of stabilizer quantum computation
    [doi-link](https://doi.org/10.1088/1367-2630/16/1/013009)

    Parameters:
        key (str): the key of the non-stabilizer state, can be {'Hplus', 'Hminus', 'Hi', 'T', 'Strange', 'Norrell'}

    Returns:
        ret (np.ndarray): the non-stabilizer state of shape (3,)
    '''
    assert key in {'Hplus', 'Hminus', 'Hi', 'T', 'Strange', 'Norrell'}
    if key=='Hplus':
        tmp0 = np.array([np.sqrt(3)+1, 1, 1])
        ret = tmp0 / np.linalg.norm(tmp0)
    elif key=='Hminus':
        tmp0 = np.array([1-np.sqrt(3), 1, 1])
        ret = tmp0 / np.linalg.norm(tmp0)
    elif key in {'Hi', 'Strange'}:
        # https://doi.org/10.1088/1367-2630/16/1/013009
        ret = np.array([0, 1, -1]) / np.sqrt(2)
    elif key=='Norrell':
        # https://doi.org/10.1088/1367-2630/16/1/013009
        ret = np.array([-1,2,-1]) / np.sqrt(6)
    elif key=='T':
        tmp0 = np.exp(2j*np.pi/9)
        ret = np.array([tmp0, 1, 1/tmp0]) / np.sqrt(3)
    return ret

def plot_qubit_magic_state_3d():
    r'''plot the magic state set (octahedron) of qubit

    Parameters:

    Returns:
        fig (matplotlib.figure.Figure): the figure object
        ax (mpl_toolkits.mplot3d.Axes3D): the axis object
    '''
    np_rng = np.random.default_rng()
    fig = plt.figure()
    ax = mpl_toolkits.mplot3d.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
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


def matrix_to_wigner_basis(np0:np.ndarray, is_hermitian:bool=False):
    r'''convert the matrix to Wigner basis

    Parameters:
        np0 (np.ndarray): the matrix of shape (...,dim,dim)
        is_hermitian (bool): whether the matrix is hermitian

    Returns:
        ret (np.ndarray): the Wigner basis of shape (...,dim*dim)
    '''
    assert (np0.ndim>=2) and (np0.shape[-1]==np0.shape[-2])
    shape = np0.shape
    dim = np0.shape[-1]
    weyl_A = _get_Heisenberg_Weyl_operator_hf0(dim, _kind='dagger')[1]
    ret = (np0.reshape(-1, dim*dim) @ weyl_A).reshape(shape[:-2]+(dim*dim,)) / dim
    if is_hermitian:
        ret = ret.real
    return ret


def wigner_basis_to_matrix(np0:np.ndarray):
    r'''convert the Wigner basis to matrix

    Parameters:
        np0 (np.ndarray): the Wigner basis of shape (...,dim*dim)

    Returns:
        ret (np.ndarray): the matrix of shape (...,dim,dim)
    '''
    shape = np0.shape
    dim = int(np.sqrt(shape[-1]))
    assert dim*dim==shape[-1]
    weyl_A = _get_Heisenberg_Weyl_operator_hf0(dim, _kind='normal')[1]
    ret = (np0.reshape(-1, dim*dim) @ weyl_A.reshape(dim*dim, dim*dim)).reshape(shape[:-1]+(dim,dim))
    return ret


def get_wigner_trace_norm(np0:np.ndarray):
    r'''get the Wigner trace norm of matrix

    Parameters:
        np0 (np.ndarray): the matrix of shape (...,dim,dim)

    Returns:
        ret (np.ndarray): the Wigner trace norm of shape (...)
    '''
    tmp0 = matrix_to_wigner_basis(np0, is_hermitian=True)
    ret = np.abs(tmp0).sum(dim=-1)
    return ret

def get_magic_state_boundary_qubit(rho:np.ndarray):
    r'''get the magic state boundary of qubit

    Parameters:
        rho (np.ndarray): density matrix of shape (2,2) or (batch_size, 2, 2)

    Returns:
        ret (np.ndarray): the magic state boundary of shape (batch_size,)
    '''
    assert (rho.ndim in {2,3}) and (rho.shape[-1]==2) and (rho.shape[-2]==2)
    is_single_item = rho.ndim==2
    if is_single_item:
        rho = rho.reshape(1,2,2)
    sx = (rho[:,0,1] + rho[:,1,0]).real
    sy = (rho[:,1,0] - rho[:,0,1]).imag
    sz = (rho[:,0,0] - rho[:,1,1]).real
    tmp0 = np.stack([sx,sy,sz], axis=1)
    tmp0 /= np.linalg.norm(tmp0, ord=2, axis=1, keepdims=True)
    ret = 0.5 / np.abs(tmp0).sum(axis=1) #factor 0.5 is for Gell-Mann normalization (used in numqi)
    if is_single_item:
        ret = ret[0]
    return ret

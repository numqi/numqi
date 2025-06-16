import functools
import numpy as np
import scipy.linalg
import scipy.special
import torch

def angle_to_su2(alpha:float|np.ndarray, beta:float|np.ndarray, gamma:float|np.ndarray):
    r'''Convert Euler angles to SU(2) matrix

    Parameters:
        alpha (np.ndarray,float): alpha angle
        beta (np.ndarray,float): beta angle
        gamma (np.ndarray,float): gamma angle

    Returns:
        ret (np.ndarray): SU(2) matrix
    '''
    alpha,beta,gamma = [np.asarray(x) for x in (alpha,beta,gamma)]
    shape = np.broadcast_shapes(alpha.shape, beta.shape, gamma.shape)
    N0 = np.prod(np.array(shape, dtype=np.int64))
    alpha,beta,gamma = [np.broadcast_to(x, shape).reshape(N0) for x in (alpha,beta,gamma)]
    exp_apg = np.exp(0.5j*(alpha+gamma))
    exp_amg = np.exp(0.5j*(alpha-gamma))
    cb = np.cos(beta/2)
    sb = np.sin(beta/2)
    ret = np.stack([
        cb*exp_apg.conj(), -sb*exp_amg.conj(), sb*exp_amg, cb*exp_apg
    ], axis=1).reshape(shape + (2,2))
    # ret, -ret
    return ret


def angle_to_so3(alpha:float|np.ndarray, beta:float|np.ndarray, gamma:float|np.ndarray):
    r'''Convert Euler angles to SO(3) matrix

    Parameters:
        alpha (np.ndarray,float): alpha angle
        beta (np.ndarray,float): beta angle
        gamma (np.ndarray,float): gamma angle

    Returns:
        ret (np.ndarray): SO(3) matrix
    '''
    alpha,beta,gamma = [np.asarray(x) for x in (alpha,beta,gamma)]
    shape = np.broadcast_shapes(alpha.shape, beta.shape, gamma.shape)
    N0 = np.prod(np.array(shape, dtype=np.int64))
    alpha,beta,gamma = [np.broadcast_to(x, shape).reshape(N0) for x in (alpha,beta,gamma)]
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)
    cg = np.cos(gamma)
    sg = np.sin(gamma)
    ret = np.stack([
        ca*cb*cg-sa*sg,-ca*cb*sg-sa*cg,ca*sb,
        sa*cb*cg+ca*sg,-sa*cb*sg+ca*cg,sa*sb,
        -sb*cg,sb*sg,cb,
    ], axis=1).reshape(shape + (3,3))
    return ret


def _so3_to_angle_hf0(x00, x02, x12, x20, x21, x22, zero_eps):
    x00, x02, x12, x20, x21, x22 = [x.real for x in (x00, x02, x12, x20, x21, x22)] #drop imag part
    beta = np.arccos(x22) #(0,pi)
    alpha = np.zeros_like(beta)
    gamma = np.zeros_like(beta)
    ind0 = beta<zero_eps
    ind1 = beta>(np.pi-zero_eps)
    ind2 = np.logical_not(np.logical_or(ind0, ind1))
    if np.any(ind0):
        tmp0 = np.arccos(x00) #(0,pi) alpha+gamma
        alpha[ind0] = tmp0/2
        gamma[ind0] = alpha
    if np.any(ind1):
        tmp0 = np.arccos(-x00) #(0,pi) alpha-gamma
        alpha[ind1] = tmp0
        gamma[ind1] = 0*alpha
    if np.any(ind2):
        tmp0 = 1/np.sin(beta[ind2])
        tmp1 = np.arccos(-x20*tmp0) #(0,pi)
        tmp2 = (x21*tmp0)<0
        gamma[ind2] = tmp1*np.logical_not(tmp2) + (2*np.pi-tmp1)*tmp2 #(0,2*pi)
        tmp1 = np.arccos(x02*tmp0) #(0,pi)
        tmp2 = (x12*tmp0)<0
        alpha[ind2] = tmp1*np.logical_not(tmp2) + (2*np.pi-tmp1)*tmp2 #(0,2*pi)
    return alpha,beta,gamma


def so3_to_angle(np0:np.ndarray, zero_eps:float=1e-7):
    r'''Convert SO(3) matrix to Euler angles

    Parameters:
        np0 (np.ndarray): SO(3) matrix, shape (...,3,3)
        zero_eps (float): zero threshold

    Returns:
        alpha (np.ndarray): angle alpha, shape (...), (0,2*pi)
        beta (np.ndarray): angle beta, shape (...), (0,pi)
        gamma (np.ndarray): angle gamma, shape (...), (0,2*pi)
    '''
    assert (np0.ndim>=2) and (np0.shape[-2:]==(3,3))
    shape = np0.shape[:-2]
    np0 = np0.real.reshape(-1, 3, 3) #drop imag if complex
    alpha,beta,gamma = _so3_to_angle_hf0(np0[:,0,0], np0[:,0,2], np0[:,1,2], np0[:,2,0], np0[:,2,1], np0[:,2,2], zero_eps)
    if len(shape)==0:
        ret = alpha[0],beta[0],gamma[0]
    else:
        ret = tuple(x.reshape(shape) for x in [alpha,beta,gamma])
    return ret


def su2_to_angle(np0:np.ndarray, zero_eps:float=1e-7):
    r'''Convert SU(2) matrix to Euler angles

    Parameters:
        np0 (np.ndarray): SU(2) matrix, shape (...,2,2)
        zero_eps (float): zero threshold

    Returns:
        alpha (np.ndarray): angle alpha, shape (...), (0,2*pi)
        beta (np.ndarray): angle beta, shape (...), (0,pi)
        gamma (np.ndarray): angle gamma, shape (...), (0,4*pi)
    '''
    # alpha(0,2*pi) beta(0,pi) gamma(0,4*pi)
    assert (np0.ndim>=2) and (np0.shape[-2:]==(2,2))
    shape = np0.shape[:-2]
    np0 = np0.reshape(-1, 2, 2)
    assert np.abs(np0[:,0,0]-np0[:,1,1].conj()).max() < zero_eps
    assert np.abs(np0[:,1,0]+np0[:,0,1].conj()).max() < zero_eps
    a = np0[:,0,0]
    aH = a.conj()
    b = np0[:,0,1]
    bH = b.conj()
    # ret = np.stack([
    #     0.5*(a*a+aH*aH-b*b-bH*bH), -0.5j*(a*a-aH*aH+b*b-bH*bH), -a*b-aH*bH,
    #     0.5j*(a*a-aH*aH-b*b+bH*bH), 0.5*(a*a+aH*aH+b*b+bH*bH), 1j*(aH*bH-a*b),
    #     aH*b+a*bH, 1j*(aH*b-a*bH), a*aH-b*bH,
    # ], axis=1).real.reshape(*shape, 3, 3)
    tmp0 = [
        0.5*(a*a+aH*aH-b*b-bH*bH), #x00
        -a*b-aH*bH, #x02
        1j*(aH*bH-a*b), #x12
        aH*b+a*bH, #x20
        1j*(aH*b-a*bH), #x21
        a*aH-b*bH, #x22
    ]
    alpha,beta,gamma = _so3_to_angle_hf0(*tmp0, zero_eps)
    tmp1 = (np.exp(0.5j*(alpha+gamma)) * a).real < 0
    gamma = gamma + tmp1*(2*np.pi)
    if len(shape)==0:
        ret = alpha[0],beta[0],gamma[0]
    else:
        ret = tuple(x.reshape(shape) for x in [alpha,beta,gamma])
    return ret


def so3_to_su2(np0:np.ndarray, zero_eps:float=1e-7):
    r'''Convert SO(3) matrix to SU(2) matrix

    Parameters:
        np0 (np.ndarray): SO(3) matrix, shape (...,3,3)
        zero_eps (float): zero threshold

    Returns:
        ret (np.ndarray): SU(2) matrix, shape (...,2,2)
    '''
    alpha,beta,gamma = so3_to_angle(np0, zero_eps)
    ret = angle_to_su2(alpha, beta, gamma)
    # ret, -ret
    return ret


def su2_to_so3(np0:np.ndarray, zero_eps:float=1e-7):
    r'''Convert SU(2) matrix to SO(3) matrix

    Parameters:
        np0 (np.ndarray): SU(2) matrix, shape (...,2,2)
        zero_eps (float): zero threshold

    Returns:
        ret (np.ndarray): SO(3) matrix, shape (...,3,3)
    '''
    assert (np0.ndim>=2) and (np0.shape[-2:]==(2,2))
    shape = np0.shape[:-2] + (3,3)
    np0 = np0.reshape(-1, 2, 2)
    assert np.abs(np0[:,0,0]-np0[:,1,1].conj()).max() < zero_eps
    assert np.abs(np0[:,1,0]+np0[:,0,1].conj()).max() < zero_eps
    a = np0[:,0,0]
    aH = a.conj()
    b = np0[:,0,1]
    bH = b.conj()
    ret = np.stack([
        0.5*(a*a+aH*aH-b*b-bH*bH), -0.5j*(a*a-aH*aH+b*b-bH*bH), -a*b-aH*bH,
        0.5j*(a*a-aH*aH-b*b+bH*bH), 0.5*(a*a+aH*aH+b*b+bH*bH), 1j*(aH*bH-a*b),
        aH*b+a*bH, 1j*(aH*b-a*bH), a*aH-b*bH,
    ], axis=1).real.reshape(shape)
    return ret


# TODO numerical stability issue
def _round_factorial(np0:int):
    tmp0 = np.round(np0).astype(np.int64)
    assert np.abs(tmp0-np0).max() < 1e-10
    assert np.all(tmp0>=0)
    ret = scipy.special.factorial(np0)
    return ret


@functools.lru_cache
def _get_su2_irrep_get_coeff(j2:int):
    # ret(list,(tuple,2),j2//2+1)
    #    coeff0: (np,float64,(N1,N1))
    #    coeff1: (np,int,(N1,N1))
    assert j2>=1
    npM = np.arange(j2+1)[::-1] - j2/2
    tmp0 = scipy.special.factorial(j2/2+npM) * scipy.special.factorial(j2/2)
    ret = []
    coeff_full = np.triu(scipy.linalg.circulant(np.array([0]+list(range(j2,0,-1)),dtype=np.int64)))
    for ind0 in range(j2//2+1):
        R = coeff_full[:(j2+1-2*ind0),:(j2+1-2*ind0)]+ind0
        M = npM[ind0:(j2+1-ind0)].reshape(-1, 1)
        N = M.reshape(1, -1)
        tmp0 = 1-2*(R%2) #(-1)**r
        tmp1 = np.sqrt(_round_factorial(j2/2+M)*_round_factorial(j2/2-M)*_round_factorial(j2/2+N)*_round_factorial(j2/2-N))
        tmp2 = _round_factorial(j2/2+M-R)*_round_factorial(j2/2-N-R)*_round_factorial(R)*_round_factorial(R-M+N)
        coeff0 = tmp0*tmp1/tmp2
        coeff1 = np.round(2*R - M + N).astype(np.int64)
        ret.append((coeff0,coeff1))
    return ret


def get_su2_irrep(j2:int, *mat_or_angle:tuple, return_matd:bool=False):
    r'''Get irrep of SU(2)

    Parameters:
        j2 (int): `j =0, 0.5, 1, 1.5, ...`, `j2=0, 1, 2, 3, ...`
        mat_or_angle (tuple): SU(2) matrix (tuple of length 1) or Euler angles (tuple of length 3)
        return_matd (bool): return matd

    Returns:
        ret (np.ndarray): irrep of SU(2), shape (...,j2+1,j2+1)
        matd (np.ndarray): matd, shape (...,j2+1,j2+1)
    '''
    j2 = int(j2)
    assert j2>=0
    assert len(mat_or_angle) in {1,3}
    if len(mat_or_angle)==1:
        alpha,beta,gamma = su2_to_angle(mat_or_angle[0])
        shape = alpha.shape
    else:
        alpha,beta,gamma = [np.asarray(x) for x in mat_or_angle]
        shape = np.broadcast_shapes(alpha.shape, beta.shape, gamma.shape)
        alpha,beta,gamma = [np.broadcast_to(x, shape) for x in (alpha,beta,gamma)]
    N0 = np.prod(np.array(shape, dtype=np.int64))
    alpha,beta,gamma = [np.broadcast_to(x, shape).reshape(N0) for x in (alpha,beta,gamma)]
    if j2==0:
        ret = np.ones(shape+(1,1))
        if return_matd:
            ret = ret, ret.copy()
    else:
        coeff_list = _get_su2_irrep_get_coeff(j2)
        matd = np.zeros((N0,j2+1,j2+1), dtype=np.float64)
        tmp0 = [slice(x,j2+1-x,1) for x in range(len(coeff_list))]
        cb = np.vander(np.cos(beta/2), N=j2+1, increasing=True)
        sb = np.vander(np.sin(beta/2), N=j2+1, increasing=True)
        for ind0,(coeff0,coeff1) in zip(tmp0,coeff_list):
            matd[:,ind0,ind0] += coeff0 * cb[:,j2-coeff1] * sb[:,coeff1]
        tmp0 = np.arange(j2+1)[::-1] - j2/2
        tmp1 = np.exp(-1j*tmp0*alpha[:,np.newaxis])
        tmp2 = np.exp(-1j*tmp0*gamma[:,np.newaxis])
        ret = (tmp1[:,:,np.newaxis] * matd * tmp2[:,np.newaxis,:]).reshape(shape+(j2+1,j2+1))
        if return_matd:
            ret = ret, matd.reshape(ret.shape)
    return ret


def get_rational_orthogonal2_matrix(m:int, n:int):
    r'''Get rational orthogonal 2x2 matrix
    [wiki-link](https://en.wikipedia.org/wiki/Pythagorean_triple)

    Parameters:
        m (int): m
        n (int): n

    Returns:
        ret (np.ndarray): rational orthogonal 2x2 matrix
    '''
    m = int(m)
    n = int(n)
    assert (m!=0) and (n!=0) and (abs(m)!=abs(n))
    a = m*m - n*n
    b = 2*m*n
    c = m*m + n*n
    # print(a,b,c)
    st = a/c
    ct = b/c
    ret = np.array([[ct,st],[-st,ct]])
    return ret


_local_constant = dict()
def _getX(key:str):
    if len(_local_constant) == 0:
        X = np.array([[0,1], [1,0]], dtype=np.float64)
        Y = np.array([[0,-1j], [1j,0]], dtype=np.complex128)
        Z = np.array([[1,0], [0,-1]], dtype=np.float64)
        S = np.array([[1,0], [0,1j]])
        H = np.array([[1,1], [1,-1]])/np.sqrt(2)
        _local_constant['I'] = np.eye(2, dtype=np.float64)
        _local_constant['X'] = X
        _local_constant['Y'] = Y
        _local_constant['Z'] = Z
        _local_constant['S'] = S
        _local_constant['H'] = H
        _local_constant['XX'] = np.kron(X, X)
        _local_constant['YY'] = np.kron(Y, Y)
        _local_constant['ZZ'] = np.kron(Z, Z)
        _local_constant['magic'] = np.array([[1,0,0,1j], [0,1j,1,0], [0,1j,-1,0], [1,0,0,-1j]])/np.sqrt(2)
        _local_constant['gamma'] = np.array([[1,1,-1,1], [1,1,1,-1], [1,-1,-1,-1], [1,-1,1,1]], dtype=np.float64)
        _local_constant['gammaT'] = _local_constant['gamma'].T.copy()
    return _local_constant[key]


def su2su2_to_so4_magic(np0:np.ndarray, np1:np.ndarray):
    # https://arxiv.org/abs/quant-ph/0507171v1 eq(12)
    magic = _getX('magic')
    ret = (magic.T.conj() @ np.kron(np0, np1.conj()) @ magic).real
    return ret


def su2_to_so3_magic(np0:np.ndarray):
    # https://arxiv.org/abs/quant-ph/0507171v1 eq(11)
    magic = _getX('magic')
    ret = (magic.T.conj() @ np.kron(np0, np0.conj()) @ magic)[1:, 1:].real
    return ret


def so3_to_su2_magic(np0:np.ndarray, zero_eps:float=1e-10):
    # https://arxiv.org/abs/quant-ph/0507171v1 eq(11)
    tmp0 = np.concatenate([np.array([[1,0,0,0]]), np.concatenate([np.zeros((3,1)), np0], axis=1)], axis=0)
    magic = _getX('magic')
    tmp1 = magic @ tmp0 @ magic.T.conj()
    U,S,V = np.linalg.svd(tmp1.reshape(2,2,2,2).transpose(0,2,1,3).reshape(4,4))
    assert (abs(S[0]-2) < zero_eps) and (S[1] < zero_eps)
    abcd = U[:,0].reshape(2,2) * np.sqrt(2)
    det = abcd[0,0]*abcd[1,1] - abcd[0,1]*abcd[1,0]
    assert (abs(det)-1) < zero_eps
    abcd *= np.exp(-1j*np.angle(det)/2)
    return abcd


def _sort_kx_ky_kz(A0,A1,B0,B1,vecK):
    ind0 = np.argsort(vecK[1:])[::-1]
    if np.any(ind0!=np.array([0,1,2])):
        tmp0 = _getX('I') if (ind0[0]==0) else (_getX('S') if (ind0[0]==1) else _getX('H'))
        (ind0[0]!=2) and (ind0[2]!=2)
        if (ind0[1]==2) or (ind0[2]==1):
            tmp0 = tmp0 @ _getX('H') @ _getX('S') @ _getX('H')
        A0,A1,B0,B1 = A0 @ tmp0, A1 @ tmp0, tmp0.T.conj()@B0, tmp0.T.conj()@B1
        vecK[1:] = vecK[1:][ind0]
    return A0,A1,B0,B1,vecK


def get_su4_kak_decomposition(np0:np.ndarray):
    magic = _getX('magic')
    gamma = _getX('gamma')
    Qleft, diag, Qright = diagonalize_unitary_using_two_orthogonals(magic.T.conj() @ np0 @ magic)
    A0,A1 = so4_to_su2su2_magic(Qleft)
    A1 = A1.conj()
    B0,B1 = so4_to_su2su2_magic(Qright.T)
    B1 = B1.conj()
    vecK = gamma.T @ np.angle(diag) / 4
    ## reduce vecK to range [0,pi/2)
    I,X,Y,Z = _getX('I'), _getX('X'), _getX('Y'), _getX('Z')
    ind0 = np.floor(vecK[1:]/(np.pi/2)).astype(np.int64)
    tmp0 = (X if (ind0[0]%2==1) else I) @ (Y if (ind0[1]%2==1) else I) @ (Z if (ind0[2]%2==1) else I)
    A0,A1 = A0 @ tmp0, A1 @ tmp0
    A0 *= (1j)**(ind0.sum())
    vecK[1:] -= ind0 * np.pi/2
    ## sort vecK[1:] in descending order
    A0,A1,B0,B1,vecK = _sort_kx_ky_kz(A0,A1,B0,B1,vecK)
    ## make sure kx+ky<=pi/2
    if vecK[1]+vecK[2]>np.pi/2:
        tmp0 = _getX('S') # swap X,Y
        vecK[[1,2]] = vecK[[2,1]].copy()
        # A0,A1,B0,B1 = A0 @ tmp0, A1 @ tmp0, tmp0.T.conj()@B0, tmp0.T.conj()@B1
        A0,B0 = A0@_getX('Z'), _getX('Z') @ B0 #reverse X,Y
        vecK[[1,2]] = -vecK[[2,1]]
        ind0 = np.floor(vecK[1:3]/(np.pi/2)).astype(np.int64) #shift X,Y
        tmp0 = (X if (ind0[0]%2==1) else I) @ (Y if (ind0[1]%2==1) else I)
        A0,A1 = A0 @ tmp0, A1 @ tmp0
        A0 *= (1j)**(ind0.sum())
        vecK[1:3] -= ind0 * np.pi/2
        A0,A1,B0,B1,vecK = _sort_kx_ky_kz(A0,A1,B0,B1,vecK)
    # if vecK[3]==0: #skip, determining zero is numerically unstable
    return A0, A1, B0, B1, vecK, diag


@functools.lru_cache
def _get_kak_kernel_torch(dtype):
    gamma = torch.tensor(_getX('gamma')[:,1:], dtype=dtype)
    if dtype==torch.float32:
        magic = torch.tensor(_getX('magic'), dtype=torch.complex64)
    else:
        magic = torch.tensor(_getX('magic'), dtype=torch.complex128)
    return gamma, magic

def get_kak_kernel(vecK:np.ndarray|torch.Tensor):
    istorch = isinstance(vecK, torch.Tensor)
    assert vecK.shape==(3,)
    if istorch:
        gamma,magic = _get_kak_kernel_torch(vecK.dtype)
        tmp0 = torch.exp(1j*(gamma @ vecK))
        ret = (magic * tmp0) @ magic.T.conj()
    else:
        tmp0 = np.exp(1j*(_getX('gamma')[:,1:] @ vecK))
        magic = _getX('magic')
        ret = (magic*tmp0) @ magic.T.conj()
        # tmp0 = scipy.linalg.expm(1j*(vecK[1]*XX + vecK[2]*YY + vecK[3]*ZZ)) * np.exp(1j*vecK[0])
    return ret


def diagonalize_unitary_using_two_orthogonals(u:np.ndarray):
    """Decomposes u into L @ np.diag(D) @ R.T where L and R are real orthogonal.
    """
    # https://scicomp.stackexchange.com/a/33400
    diag_r, diag_i, left, right = scipy.linalg.qz(u.real, u.imag)
    diag = np.diagonal(diag_r) + np.diagonal(diag_i) * 1j
    if np.linalg.det(left) < 0: #make sure SO
        left[:,0] *= -1
        diag[0] *= -1
    if np.linalg.det(right) < 0: #make sure SO
        right[:,0] *= -1
        diag[0] *= -1
    return left, diag, right


# def diagonalize_two_matrix_Eckart_Young():
#     # TODO may scipy.linalg.qz
#     pass


def so4_to_su2su2_magic(np0:np.ndarray, zero_eps:float=1e-10):
    # https://arxiv.org/abs/quant-ph/0507171v1 eq(12)
    magic = _getX('magic')
    tmp1 = magic @ np0 @ magic.T.conj()
    U,S,V = np.linalg.svd(tmp1.reshape(2,2,2,2).transpose(0,2,1,3).reshape(4,4))
    assert (abs(S[0]-2) < zero_eps) and (S[1] < zero_eps)

    abcd0 = U[:,0].reshape(2,2) * np.sqrt(2)
    det = abcd0[0,0]*abcd0[1,1] - abcd0[0,1]*abcd0[1,0]
    assert (abs(det)-1) < zero_eps
    abcd0 *= np.exp(-1j*np.angle(det)/2)

    abcd1 = V[0].conj().reshape(2,2) * np.sqrt(2)
    det = abcd1[0,0]*abcd1[1,1] - abcd1[0,1]*abcd1[1,0]
    assert (abs(det)-1) < zero_eps
    abcd1 *= np.exp(-1j*np.angle(det)/2)
    return abcd0, abcd1

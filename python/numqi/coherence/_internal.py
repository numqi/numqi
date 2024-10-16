import numpy as np
import cvxpy

def get_coherence_of_formation_pure(psi:np.ndarray) -> float:
    r'''get the coherence of formation of a pure state

    reference: Quantum Coherence and Intrinsic Randomness
    [arxiv-link](https://arxiv.org/abs/1605.07818)

    Parameters:
        psi (np.ndarray): a pure state vector

    Returns:
        cof (float): the coherence of formation of the input state
    '''
    assert psi.ndim==1
    tmp0 = psi.real*psi.real + psi.imag*psi.imag
    eps = np.finfo(tmp0.dtype).eps
    tmp1 = np.log(np.maximum(tmp0, eps))
    ret = max(0, -np.dot(tmp0, tmp1))
    return ret


def get_coherence_of_formation_1qubit(rho:np.ndarray) -> float:
    r'''get the coherence of formation of a 1-qubit state

    TODO reference

    Parameters:
        rho (np.ndarray): a 2x2 density matrix

    Returns:
        cof (float): the coherence of formation of the 1-qubit mixed state
    '''
    assert rho.shape == (2,2)
    tmp0 = 0.5 + 0.5*np.sqrt(max(0, 1-4*abs(rho[0,1])**2))
    ret = max(0, float(-tmp0*np.log(tmp0) - (1-tmp0)*np.log(1-tmp0)))
    return ret


def get_geometric_coherence_pure(psi:np.ndarray) -> float:
    r'''get the geometric measure of coherence for a pure state

    reference: Numerical and analytical results for geometric measure of coherence and geometric measure of entanglemen
    [doi-link](https://doi.org/10.1038/s41598-020-68979-z)

    Parameters:
        psi (np.ndarray): a pure state vector

    Returns:
        gmc (float): the geometric measure of coherence of the input state
    '''
    assert psi.ndim==1
    tmp0 = psi.real**2 + psi.imag**2
    ret = max(0, 1 - tmp0.max())
    return ret


def get_geometric_coherence_1qubit(rho:np.ndarray) -> float:
    r'''get the geometric measure of coherence for a 1-qubit state

    TODO-reference

    Parameters:
        rho (np.ndarray): a 2x2 density matrix

    Returns:
        gmc (float): the geometric measure of coherence of the 1-qubit mixed state
    '''
    assert rho.shape == (2,2)
    ret = 0.5 - 0.5*np.sqrt(max(0, 1-4*abs(rho[0,1])**2))
    return ret


def get_geometric_coherence_sdp(rho:np.ndarray):
    r'''Get the geometric measure of coherence using semi-definite programming (SDP)

    reference: Numerical and analytical results for geometric measure of coherence and geometric measure of entanglement
    [arxiv-link](https://arxiv.org/abs/1903.10944)

    Parameters:
        rho (np.ndarray): the density matrix, shape=(dim,dim), support batch input (batch,dim,dim)

    Returns:
        ret (float,np.ndarray): the coherence
    '''
    assert rho.ndim in {2,3}
    isone = (rho.ndim == 2)
    if isone:
        rho = rho[None]
    assert np.abs(rho-rho.transpose(0,2,1).conj()).max() < 1e-10
    dim = rho.shape[-1]
    cvxD = cvxpy.Variable((dim,dim), diag=True)
    cvxX = cvxpy.Variable((dim,dim), complex=True)
    cvxrho = cvxpy.Parameter((dim,dim), PSD=True) #hermitian=True
    constraint = [
        cvxpy.bmat([[cvxrho, cvxX], [cvxpy.conj(cvxX.T), cvxD]]) >> 0,
        cvxD >> 0,
        cvxpy.trace(cvxD) == 1
    ]
    obj = cvxpy.Maximize(cvxpy.real(cvxpy.trace(cvxX)))
    prob = cvxpy.Problem(obj, constraint)
    ret = []
    for x in rho:
        cvxrho.value = x
        prob.solve()
        ret.append(1 - obj.value**2)
    ret = ret[0] if isone else np.array(ret)
    return ret

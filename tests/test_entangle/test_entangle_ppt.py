import numpy as np
import cvxpy
import scipy.linalg

import numqi

try:
    import mosek
    USE_MOSEK = True
except ImportError:
    USE_MOSEK = False


def test_is_ppt():
    # false positive example from https://qetlab.com/Main_Page
    # tiles UPB/BES
    dm_tiles = numqi.entangle.load_upb('tiles', return_bes=True)[1]
    assert numqi.entangle.is_ppt(dm_tiles)
    # Determined to be entangled via the realignment criterion. Reference:
    # K. Chen and L.-A. Wu. A matrix realignment method for recognizing entanglement. Quantum Inf. Comput., 3:193-202, 2003

    dim = [2,3] #this is correct for PPT
    for _ in range(100):
        tmp0 = numqi.random.rand_bipartite_state(*dim, k=1, return_dm=True)
        assert numqi.entangle.is_ppt(tmp0, dim)==True
        tmp0 = numqi.random.rand_bipartite_state(*dim, k=2, return_dm=True)
        assert numqi.entangle.is_ppt(tmp0, dim)==False


def _test_cvx_matrix_xlogx_hf0(Y, sqrt_order, pade_order):
    # maximize_X tr(X) - tr(XlogX) + tr(XlogY)
    # solution: tr(Y)
    # https://arxiv.org/abs/1705.00812 Semidefinite approximations of the matrix logarithm
    is_complex = np.iscomplexobj(Y)
    n = Y.shape[0]
    if is_complex:
        cvxX = cvxpy.Variable((n,n), hermitian=True)
    else:
        cvxX = cvxpy.Variable((n,n), symmetric=True)
    cvxP,constraint = numqi.entangle.cvx_matrix_xlogx(cvxX, sqrt_order, pade_order)
    tmp0 = cvxpy.trace(cvxX) - (cvxpy.trace(cvxP['XlogX']) - cvxpy.trace(cvxX@scipy.linalg.logm(Y)))
    obj = cvxpy.Maximize(cvxpy.real(tmp0) if is_complex else tmp0)
    prob = cvxpy.Problem(obj, constraint)
    prob.solve()

    assert abs(obj.value - np.trace(Y).real) < 1e-3 #1e-4 will fail for solver=SCS
    hf0 = lambda x: np.ascontiguousarray(x.value)
    z0 = {
        'X': hf0(cvxX),
        'XlogX': hf0(cvxP['XlogX']),
        # 'T': np.stack([hf0(x) for x in cvxP['T']], axis=0),
        'Xpow': np.stack([hf0(x) for x in cvxP['Xpow']], axis=0),
    }
    tmp0 = z0['X'] @ scipy.linalg.logm(z0['X'])
    assert np.abs(z0['XlogX']-tmp0).max() < 1e-3
    EVL,EVC = np.linalg.eigh(z0['X'])
    tmp0 = EVL**(1-2.0**np.arange(-pade_order, 0).reshape(-1,1,1)) # 1-2**(-k)
    assert np.abs((EVC*tmp0) @ EVC.T.conj() - z0['Xpow']).max() < 1e-3


def test_cvx_matrix_xlogx():
    np_rng = np.random.default_rng()
    n = 4

    tmp0 = np_rng.normal(size=(n,n))
    Y = (tmp0 @ tmp0.T.conj())/(n*n)
    _test_cvx_matrix_xlogx_hf0(Y, sqrt_order=3, pade_order=3)

    tmp0 = np_rng.normal(size=(n,n)) + 1j*np_rng.normal(size=(n,n))
    Y = (tmp0 @ tmp0.T.conj())/(n*n)
    _test_cvx_matrix_xlogx_hf0(Y, sqrt_order=3, pade_order=3)


def test_cvx_relative_entropy_entanglement_random():
    dimA = 2
    dimB = 2
    sqrt_order = 3
    assert min(dimA,dimB)>1, 'need to be entangled'
    rho = numqi.random.rand_bipartite_state(dimA, dimB, k=min(dimA,dimB), return_dm=True) #entangled
    ree,z0 = numqi.entangle.get_ppt_ree(rho, dimA, dimB, return_info=True, sqrt_order=sqrt_order, pade_order=3)

    tau = z0['X']
    assert np.abs(tau - tau.T.conj()).max() < 1e-6
    assert abs(np.trace(tau)-1) < 1e-6
    assert np.linalg.eigvalsh(tau.reshape(dimA,dimB,dimA,dimB).transpose([0,3,2,1]).reshape(dimA*dimB,-1))[0] > -1e-4
    tmp0 = numqi.utils.get_relative_entropy(rho, tau)
    assert abs(tmp0 - ree) < 1e-4

    Xpow = z0['Xpow']
    assert np.abs(tau @ Xpow - Xpow @ tau).max() < 1e-4
    EVL0 = np.linalg.eigvalsh(tau)
    mask = EVL0>1e-4
    EVL0[mask]
    EVL1 = np.linalg.eigvalsh(Xpow)
    tmp0 = EVL0[mask] ** (2.0**np.arange(-sqrt_order, 0).reshape(-1,1))
    assert np.abs(tmp0 - EVL1[:,mask]).max() < 1e-3

    # if min(eig(tau)) is too small, then the log(X) is numerical instable
    # for random pure state, it's almost impossible to get nonzero min(eig(tau))
    # so we check with werner state (see _werner00 below)
    if EVL0.min() > 1e-3:
        mlogX = z0['mlogX']
        assert np.abs(tau @ mlogX - mlogX @ tau).max() < 1e-4
        assert np.abs(scipy.linalg.logm(tau) + mlogX).max() < 1e-4

def test_cvx_relative_entropy_entanglement_werner00():
    dimA = 2
    dimB = 2
    sqrt_order = 3
    rho = numqi.entangle.get_werner_state(dimA, alpha=0.8)
    ree,z0 = numqi.entangle.get_ppt_ree(rho, dimA, dimB, return_info=True, sqrt_order=sqrt_order, pade_order=3)

    tau = z0['X']
    EVL0 = np.linalg.eigvalsh(tau)
    assert EVL0.min() > 1e-4
    mlogX = z0['mlogX']
    assert np.abs(tau @ mlogX - mlogX @ tau).max() < 1e-4
    assert np.abs(scipy.linalg.logm(tau) + mlogX).max() < 1e-4


def test_cvx_relative_entropy_entanglement_werner01():
    # about 6 seconds
    dim = 3
    alpha_list = np.linspace(0, 1, 10, endpoint=False) #alpha=1 is unstable
    ret_ = np.array([numqi.entangle.get_werner_state_ree(dim, x) for x in alpha_list])
    dm_list = [numqi.entangle.get_werner_state(dim,x) for x in alpha_list]
    ret0 = numqi.entangle.get_ppt_ree(dm_list, dim, dim, sqrt_order=3, pade_order=3)
    assert np.abs(ret_-ret0).max() < 1e-4 #1e-5 fail for solver=SCS

# TODO rename all rho to dm

def test_get_ppt_boundary():
    dimA = 3
    dimB = 4
    dm0 = numqi.random.rand_density_matrix(dimA*dimB)
    dm0_norm = numqi.gellmann.dm_to_gellmann_norm(dm0)
    for x0 in [None,dm0_norm]:
        beta_l,beta_u = numqi.entangle.get_ppt_boundary(dm0, (dimA,dimB), dm_norm=x0, within_dm=False)
        assert beta_l<0
        assert beta_u>0
        for beta in [beta_l,beta_u]:
            tmp0 = numqi.entangle.hf_interpolate_dm(dm0, beta=beta, dm_norm=x0)
            tmp0 = tmp0.reshape(dimA,dimB,dimA,dimB).transpose([0,3,2,1]).reshape(dimA*dimB,-1)
            assert np.linalg.eigvalsh(tmp0)[0] > -1e-7


def test_get_ppt_boundary_werner():
    for dim in [2,3,4,5]:
        dm0 = numqi.entangle.get_werner_state(dim, alpha=1)
        beta_l,beta_u = numqi.entangle.get_ppt_boundary(dm0, (dim,dim), within_dm=True)

        ret_ = numqi.entangle.get_werner_state(dim, alpha=-1)
        ret0 = numqi.entangle.hf_interpolate_dm(dm0, beta=beta_l)
        assert np.abs(ret_-ret0).max() < 1e-10

        ret_ = numqi.entangle.get_werner_state(dim, alpha=1/dim)
        ret0 = numqi.entangle.hf_interpolate_dm(dm0, beta=beta_u)
        assert np.abs(ret_-ret0).max() < 1e-10


def test_get_ppt_boundary_isotropic():
    for dim in [2,3,4,5]:
        dm0 = numqi.entangle.get_isotropic_state(dim, alpha=1)
        beta_l,beta_u = numqi.entangle.get_ppt_boundary(dm0, (dim,dim), within_dm=True)

        ret_ = numqi.entangle.get_isotropic_state(dim, alpha=-1/(dim*dim-1))
        ret0 = numqi.entangle.hf_interpolate_dm(dm0, beta=beta_l)
        assert np.abs(ret_-ret0).max() < 1e-10

        ret_ = numqi.entangle.get_isotropic_state(dim, alpha=1/(dim+1))
        ret0 = numqi.entangle.hf_interpolate_dm(dm0, beta=beta_u)
        assert np.abs(ret_-ret0).max() < 1e-10


def test_get_ppt_numerical_range():
    # # https://doi.org/10.1103/PhysRevA.98.012315 Fig 4
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    op0 = np.kron(np.array([[1/np.sqrt(2),0],[0,0]]), sz)
    op1 = (np.kron(sy, sx) - np.kron(sx, sy)) / 2
    theta_list = np.linspace(0, 2*np.pi, 4, endpoint=False)
    z0 = numqi.entangle.get_ppt_numerical_range(op0, op1, dim=(2,2), theta_list=theta_list, use_tqdm=False)
    s12 = 1/np.sqrt(2)
    ret_ = np.array([[s12,0],[0,1/2], [-s12,0],[0,-1/2]])
    assert np.abs(ret_-z0).max() < (1e-6 if USE_MOSEK else 1e-4)


def test_get_generalized_ppt_boundary():
    rho = numqi.entangle.load_upb('tiles', return_bes=True)[1]
    beta_gppt = numqi.entangle.get_generalized_ppt_boundary(rho, dim=(3,3))
    ret_ = 0.23445029690864194 #obtained from previous running
    assert abs(beta_gppt-ret_) < 3e-4
    # rho_norm = numqi.gellmann.dm_to_gellmann_norm(rho)
    # beta=0.8649*rho_norm=0.2279211623566359 https://arxiv.org/abs/1705.01523

import itertools
import numpy as np

import numqi

np_rng = np.random.default_rng()


try:
    import mosek
    USE_MOSEK = True
except ImportError:
    USE_MOSEK = False


def test_get_symmetric_extension_index_list():
    dimA = 2
    dimB = 3
    kext = 3
    ind_1d_list = numqi.entangle.symext.get_symmetric_extension_index_list(dimA, dimB, kext, kind='1d')
    ind_2d_list = numqi.entangle.symext.get_symmetric_extension_index_list(dimA, dimB, kext, kind='2d')
    np0 = np_rng.normal(size=(dimA*dimB**kext,dimA*dimB**kext))
    for ind_1d,ind_2d in zip(ind_1d_list,ind_2d_list):
        z0 = np0[ind_2d[:,np.newaxis],ind_2d]
        z1 = np.reshape(np.reshape(np0, -1, order='F')[ind_1d], (dimA*dimB**kext,dimA*dimB**kext), order='F')
        assert np.abs(z0-z1).max() < 1e-12


# TODO test sep in 2ext
# TODO test dB=2 sym-ext is always bosonic-ext
def test_check_ABk_symmetric_extension():
    dimA = 2
    dimB = 3
    kext = 3

    np0 = numqi.random.rand_ABk_density_matrix(dimA, dimB, kext)
    np1 = np.trace(np0.reshape(dimA*dimB,dimB**(kext-1),dimA*dimB,dimB**(kext-1)), axis1=1, axis2=3)
    has_kext,np2 = numqi.entangle.symext.check_ABk_symmetric_extension_naive(np1, (dimA,dimB), kext)
    assert has_kext
    tmp0 = np.trace(np2.reshape(dimA*dimB,dimB**(kext-1), dimA*dimB,dimB**(kext-1)), axis1=1, axis2=3)
    assert np.abs(tmp0-np1).max() < 1e-7
    tmp0 = np2.reshape([dimA]+[dimB]*kext+[dimA]+[dimB]*kext)
    for indI,indJ in itertools.combinations(list(range(kext)), 2):
        tmp1 = np.arange(2*kext+2, dtype=np.int64)
        tmp1[[indI+1,indJ+1]] = tmp1[[indJ+1,indI+1]]
        tmp1[[indI+2+kext,indJ+2+kext]] = tmp1[[indJ+2+kext,indI+2+kext]]
        assert np.abs(tmp0-np.transpose(tmp0,tmp1)).max() < 1e-7


def test_get_cvxpy_transpose0213_indexing():
    N0,N1,N2,N3 = 2,3,5,7
    np0 = np_rng.normal(size=(N0*N1,N2*N3))
    ret_ = np0.reshape(N0,N1,N2,N3).transpose(0,2,1,3).reshape(N0*N2,N1*N3)
    ind0 = numqi.entangle.symext.get_cvxpy_transpose0213_indexing(N0, N1, N2, N3)
    ret0 = np.reshape(np.reshape(np0, -1, order='F')[ind0], (N0*N2,N1*N3), order='F')
    assert np.abs(ret_-ret0).max() < 1e-10


def test_werner_state_kext():
    dim = 3
    kext = 3
    boundary = (kext+dim*dim-dim)/(kext*dim+dim-1)
    # kext=2 is simply density-matrix-boundary

    alpha_yes_list = np.linspace(-1, boundary, 5)
    dm_list = [numqi.entangle.get_werner_state(dim,x) for x in alpha_yes_list]
    ret0 = numqi.entangle.check_ABk_symmetric_extension(dm_list, (dim,dim), kext, use_ppt=False)
    assert all(ret0)

    # if use mosek, the alpha_no_list can be np.linspace(boundary+1e-4, 1, 5)
    # if use SCS, the alpha_no_list can be np.linspace(boundary+5e-2, 1, 5)
    tmp0 = 1e-4 if USE_MOSEK else 5e-2
    alpha_no_list = np.linspace(boundary+tmp0, 1, 5)
    dm_list = [numqi.entangle.get_werner_state(dim,x) for x in alpha_no_list]
    ret0 = numqi.entangle.check_ABk_symmetric_extension(dm_list, (dim,dim), kext, use_ppt=False)
    assert all((not x) for x in ret0)

    rho = numqi.entangle.get_werner_state(dim, boundary)
    ret_ = numqi.gellmann.dm_to_gellmann_norm(rho)
    ret0 = numqi.entangle.get_ABk_symmetric_extension_boundary(rho, (dim,dim), kext, use_ppt=False, use_boson=False)
    assert abs(ret_-ret0) < (1e-7 if USE_MOSEK else 1e-4)
    # SCS about 1e-6
    # mosek about 1e-10


def test_werner_state_kext_ppt():
    dim = 3
    kext = 3
    boundary = 1/dim

    alpha_yes_list = np.linspace(-1, boundary, 5)
    dm_list = [numqi.entangle.get_werner_state(dim,x) for x in alpha_yes_list]
    ret0 = numqi.entangle.check_ABk_symmetric_extension(dm_list, (dim,dim), kext, use_ppt=True)
    assert all(ret0)

    # if use mosek, the alpha_no_list can be np.linspace(boundary+1e-4, 1, 5)
    # if use SCS, the alpha_no_list can be np.linspace(boundary+5e-2, 1, 5)
    tmp0 = 1e-4 if USE_MOSEK else 5e-2
    alpha_no_list = np.linspace(boundary+tmp0, 1, 5)
    dm_list = [numqi.entangle.get_werner_state(dim,x) for x in alpha_no_list]
    ret0 = numqi.entangle.check_ABk_symmetric_extension(dm_list, (dim,dim), kext, use_ppt=True)
    assert all((not x) for x in ret0)

    rho = numqi.entangle.get_werner_state(dim, boundary)
    ret_ = numqi.gellmann.dm_to_gellmann_norm(rho)
    ret0 = numqi.entangle.get_ABk_symmetric_extension_boundary(rho, (dim,dim), kext, use_ppt=True, use_boson=False)
    assert abs(ret_-ret0) < (1e-7 if USE_MOSEK else 1e-4)


def test_isotropic_state_kext():
    dim = 3
    for kext in [2,3]:
        boundary = (kext*dim+dim*dim-dim-kext)/(kext*(dim*dim-1))

        alpha_yes_list = np.linspace(-1/(dim*dim-1), boundary, 5)
        dm_list = [numqi.entangle.get_isotropic_state(dim,x) for x in alpha_yes_list]
        ret0 = numqi.entangle.check_ABk_symmetric_extension(dm_list, (dim,dim), kext, use_ppt=False)
        assert all(ret0)

        tmp0 = 1e-4 if USE_MOSEK else 5e-2
        alpha_no_list = np.linspace(boundary+tmp0, 1, 5)
        dm_list = [numqi.entangle.get_isotropic_state(dim,x) for x in alpha_no_list]
        ret0 = numqi.entangle.check_ABk_symmetric_extension(dm_list, (dim,dim), kext, use_ppt=True)
        assert all((not x) for x in ret0)

        rho = numqi.entangle.get_isotropic_state(dim, boundary)
        ret_ = numqi.gellmann.dm_to_gellmann_norm(rho)
        ret0 = numqi.entangle.get_ABk_symmetric_extension_boundary(rho, (dim,dim), kext, use_ppt=False, use_boson=False)
        assert abs(ret_-ret0) < (1e-7 if USE_MOSEK else 1e-4)


# dimA=2 dimB=3 kext=2: time=0.1s
# dimA=2 dimB=3 kext=3: time=0.5s
# dimA=3 dimB=3 kext=3: time=1.69s
# dimA=4 dimB=3 kext=3: time=4.8s 2GB
# dimA=5 dimB=3 kext=3: time=11s 3GB
# dimA=6 dimB=3 kext=3: time=23s 7GB
def test_check_ABk_symmetric_extension_irrep():
    dimA = 2
    dimB = 3
    kext = 3
    dm_list = []
    for _ in range(5):
        rho_ABk = numqi.random.rand_ABk_density_matrix(dimA, dimB, kext)
        tmp0 = np.trace(rho_ABk.reshape(dimA*dimB,dimB**(kext-1),dimA*dimB,dimB**(kext-1)), axis1=1, axis2=3)
        # tmp0 = numqi.random.rand_separable_dm(dimA, dimB)
        dm_list.append(tmp0)
    ret0 = numqi.entangle.check_ABk_symmetric_extension(dm_list, (dimA,dimB), kext)
    assert all(ret0)


def test_SymmetricExtABkIrrepModel():
    dimA = 2
    dimB = 3
    kext = 3
    model = numqi.entangle.symext.SymmetricExtABkIrrepModel(dimA, dimB, kext)
    model.set_dm_target(numqi.random.rand_separable_dm(dimA, dimB))
    model()
    rhoAB = model.rhoAB_transpose.detach().numpy().copy().reshape(dimA,dimA,dimB,dimB).transpose(0,2,1,3).reshape(dimA*dimB,-1)
    assert np.abs(rhoAB-rhoAB.T.conj()).max() < 1e-10
    assert abs(np.trace(rhoAB)-1) < 1e-10
    assert (np.linalg.eigvalsh(rhoAB)[0] + 1e-10) > 0


def test_get_ABk_symmetric_extension_ree_werner():
    # time: mosek, num_point=4
    num_point = 4
    para_list = [
        (2,2), #1.5s
        (2,3), #1.8s
        (2,4), #2.1s
        (2,5), #2.3s
        (2,6), #2.6s
        (2,7), #3.0s
        (3,2), #7.6s
        (3,3), #14.6s
        # (3,4), #46s
    ]
    for dim,kext in para_list:
        alpha_kext_boundary = (kext+dim**2-dim)/(kext*dim+dim-1)
        # when alpha=1, logm issue

        if alpha_kext_boundary>(1-1e-4):
            alpha_list = np.linspace(0, alpha_kext_boundary, num_point, endpoint=False)
        else:
            alpha_list = np.linspace(0, alpha_kext_boundary, num_point)
        dm_list = [numqi.entangle.get_werner_state(dim, x) for x in alpha_list]
        ret0 = numqi.entangle.get_ABk_symmetric_extension_ree(dm_list, (dim,dim), kext, use_ppt=False)
        assert abs(np.abs(ret0).max()) < (1e-5 if USE_MOSEK else 1e-4)

        if alpha_kext_boundary<(1-1e-4):
            alpha_list = np.linspace(alpha_kext_boundary, 1, num_point, endpoint=False)
            dm_kext_boundary = numqi.entangle.get_werner_state(dim,alpha_kext_boundary)
            dm_list = [numqi.entangle.get_werner_state(dim, x) for x in alpha_list]
            ret0 = numqi.entangle.get_ABk_symmetric_extension_ree(dm_list, (dim,dim), kext, use_ppt=False)
            ret_ = np.array([numqi.utils.get_relative_entropy(x, dm_kext_boundary) for x in dm_list])
            assert np.abs(ret0-ret_).max() < (1e-5 if USE_MOSEK else 1e-4)


def test_witness():
    dimA = 3
    dimB = 3
    kext = 3
    kwargs = dict(dim=(dimA,dimB), kext=kext, use_ppt=True, use_boson=True, return_info=True, use_tqdm=True)
    dm0 = numqi.random.rand_density_matrix(dimA*dimB)
    beta,vecA,vecN = numqi.entangle.get_ABk_symmetric_extension_boundary(dm0, **kwargs)
    op_witness = numqi.gellmann.gellmann_basis_to_dm(vecN) - np.eye(dimA*dimB)/(dimA*dimB)
    delta = np.dot(vecA, vecN)*2

    for _ in range(1000):
        tmp0 = numqi.random.rand_separable_dm(dimA,dimB,k=dimA*dimB)
        assert np.trace(tmp0@op_witness).real < delta + 1e-7

    model = numqi.entangle.AutodiffCHAREE(dimA, dimB, distance_kind='gellmann')
    model.set_expectation_op(-op_witness)
    tmp0 = -numqi.optimize.minimize(model, theta0='uniform', tol=1e-9, num_repeat=1, print_every_round=0).fun
    assert tmp0 < delta + 1e-7

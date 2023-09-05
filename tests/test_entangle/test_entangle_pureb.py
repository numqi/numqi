import torch
import numpy as np

import numqi

if torch.get_num_threads()!=1:
    torch.set_num_threads(1)

np_rng = np.random.default_rng()

def test_pureb_maximally_mixed_state():
    dimAB_list = [(3,3), (4,4), (5,5)]
    kext_list = [8,10,12,14]
    kwargs = dict(num_repeat=3, print_every_round=0, tol=1e-10)
    for dimA,dimB in dimAB_list:
        dm_target = np.eye(dimA*dimB)/(dimA*dimB)
        for kext_i in kext_list:
            model = numqi.entangle.PureBosonicExt(dimA=dimA, dimB=dimB, kext=kext_i, distance_kind='ree')
            model.set_dm_target(dm_target)
            assert numqi.optimize.minimize(model, **kwargs).fun < 1e-9


def test_pureb_werner2_ree():
    # about 11 seconds
    dim = 2 #only when dim=2, bonsonic-ext boundary matchs with k-ext boundary
    num_point = 8
    kext_list = [5,16,32,64]
    kwargs = dict(num_repeat=3, print_every_round=0, tol=1e-12)
    for kext in kext_list:
        # http://dx.doi.org/10.1103/PhysRevA.88.032323
        alpha_kext_boundary = (kext+dim**2-dim)/(kext*dim+dim-1)
        model = numqi.entangle.PureBosonicExt(dim, dim, kext=kext, distance_kind='ree')

        alpha_list = np.linspace(0, alpha_kext_boundary, num_point)
        for alpha_i in alpha_list:
            model.set_dm_target(numqi.entangle.get_werner_state(dim, alpha_i))
            ret0 = numqi.optimize.minimize(model, **kwargs).fun
            assert ret0 < 1e-8

        alpha_list = np.linspace(alpha_kext_boundary, 1, num_point, endpoint=False)
        dm_kext_boundary = numqi.entangle.get_werner_state(dim,alpha_kext_boundary)
        for alpha_i in alpha_list:
            tmp0 = numqi.entangle.get_werner_state(dim, alpha_i)
            ret_ = numqi.utils.get_relative_entropy(tmp0, dm_kext_boundary)
            model.set_dm_target(tmp0)
            ret0 = numqi.optimize.minimize(model, **kwargs).fun
            assert abs(ret0-ret_) < 1e-8


def test_pureb_boundary_werner2():
    # about 10 seconds
    dim = 2
    for kext in [5,16,32]:
        dm0 = numqi.entangle.get_werner_state(2, 1)
        alpha_kext_boundary = (kext+dim**2-dim)/(kext*dim+dim-1)
        model = numqi.entangle.PureBosonicExt(dim, dim, kext=kext, distance_kind='ree')
        beta0 = model.get_boundary(dm0, xtol=1e-4, converge_tol=1e-10, threshold=1e-7, num_repeat=3, use_tqdm=False)
        beta_ = numqi.gellmann.dm_to_gellmann_norm(numqi.entangle.get_werner_state(dim, alpha_kext_boundary))
        assert abs(beta0-beta_) < 5e-4


def test_pureb_isotropic2_ree():
    # about 11 seconds
    dim = 2 #only when dim=2, bonsonic-ext boundary matchs with k-ext boundary
    num_point = 8
    kext_list = [5,16,32,64]
    kwargs = dict(num_repeat=3, print_every_round=0, tol=1e-12)
    for kext in kext_list:
        # http://dx.doi.org/10.1103/PhysRevA.88.032323
        alpha_kext_boundary = (kext*dim+dim*dim-dim-kext)/(kext*(dim*dim-1))
        model = numqi.entangle.PureBosonicExt(dim, dim, kext=kext, distance_kind='ree')

        alpha_list = np.linspace(-1/(dim*dim-1), alpha_kext_boundary, num_point)
        for alpha_i in alpha_list:
            model.set_dm_target(numqi.entangle.get_isotropic_state(dim, alpha_i))
            ret0 = numqi.optimize.minimize(model, **kwargs).fun
            assert ret0 < 1e-8

        alpha_list = np.linspace(alpha_kext_boundary, 1, num_point, endpoint=False)
        dm_kext_boundary = numqi.entangle.get_isotropic_state(dim,alpha_kext_boundary)
        for alpha_i in alpha_list:
            tmp0 = numqi.entangle.get_isotropic_state(dim, alpha_i)
            ret_ = numqi.utils.get_relative_entropy(tmp0, dm_kext_boundary)
            model.set_dm_target(tmp0)
            ret0 = numqi.optimize.minimize(model, **kwargs).fun
            assert abs(ret0-ret_) < 1e-8


def test_pureb_boundary_isotropic2():
    # about 10 seconds
    dim = 2
    for kext in [5,16,32]:
        dm0 = numqi.entangle.get_isotropic_state(2, 1)
        alpha_kext_boundary = (kext*dim+dim*dim-dim-kext)/(kext*(dim*dim-1))
        model = numqi.entangle.PureBosonicExt(dim, dim, kext=kext, distance_kind='ree')
        beta0 = model.get_boundary(dm0, xtol=1e-4, converge_tol=1e-10, threshold=1e-7, num_repeat=3, use_tqdm=False)
        beta_ = numqi.gellmann.dm_to_gellmann_norm(numqi.entangle.get_isotropic_state(dim, alpha_kext_boundary))
        assert abs(beta0-beta_) < 5e-4


def test_pureb_ree_seperable():
    # about 20 seconds
    # (3,3,8) fail with a relatively low probability
    # (5,3,8) fail with large probability
    para_list = [(3,3,16),(3,3,32),(3,5,8),(3,5,16),(5,5,8),(5,5,16)]
    for dimA,dimB,kext in para_list:
        model = numqi.entangle.PureBosonicExt(dimA, dimB, kext=kext, distance_kind='ree')
        for _ in range(10):
            seed = np_rng.integers(0, int(1e18))
            np_rng_i = np.random.default_rng(seed) #for reproducible
            dm_target = numqi.random.rand_separable_dm(dimA, dimB, seed=np_rng_i)
            model.set_dm_target(dm_target)
            ret = numqi.optimize.minimize(model, num_repeat=3, print_every_round=0, tol=1e-10, early_stop_threshold=1e-8).fun
            assert ret<1e-8, f'seed={seed} dimA={dimA} dimB={dimB} kext={kext}'


def _pureb_boundary_tiles_upb_bes_hf0(kext, ret_):
    dm0 = numqi.entangle.load_upb('tiles', return_bes=True)[1]
    # beta=0.8649*rho_norm=0.2279211623566359 https://arxiv.org/abs/1705.01523
    kwargs = dict(xtol=1e-4, converge_tol=1e-10, threshold=1e-7, num_repeat=3, use_tqdm=False)
    model = numqi.entangle.PureBosonicExt(dimA=3, dimB=3, kext=kext, distance_kind='ree')
    beta0 = model.get_boundary(dm0, **kwargs)
    assert abs(beta0-ret_) < 3e-4


# split as three tests to parallelize
def test_pureb_boundary_tiles_upb_bes_k8():
    # obtained from previous running
    _pureb_boundary_tiles_upb_bes_hf0(kext=8, ret_=0.24145564897892074)


def test_pureb_boundary_tiles_upb_bes_k16():
    # obtained from previous running
    _pureb_boundary_tiles_upb_bes_hf0(kext=16, ret_=0.23489330520171586)


## about 80s
# def test_pureb_boundary_tiles_upb_bes_k32():
#     # obtained from previous running
#     _pureb_boundary_tiles_upb_bes_hf0(kext=32, ret_=0.231290449794623)

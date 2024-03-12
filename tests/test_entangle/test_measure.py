import numpy as np
import torch

import numqi

try:
    import mosek
    use_MOSEK = True
except ImportError:
    use_MOSEK = False

np_rng = np.random.default_rng()

def test_werner_gme():
    alpha_list = np_rng.uniform(0, 1, 10)
    dim = 3

    model = numqi.entangle.DensityMatrixGMEModel([dim,dim], num_ensemble=27)
    ret = []
    for alpha_i in alpha_list:
        model.set_density_matrix(numqi.state.Werner(dim, alpha=alpha_i))
        ret.append(numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
    ret = np.array(ret)
    ret_analytical = numqi.state.get_Werner_GME(dim, alpha_list)
    assert np.abs(ret-ret_analytical).max() < 1e-7


def test_isotropic_gme():
    alpha_list = np_rng.uniform(0, 1, 10)
    dim = 3

    model = numqi.entangle.DensityMatrixGMEModel([dim,dim], num_ensemble=27)
    ret = []
    for alpha_i in alpha_list:
        model.set_density_matrix(numqi.state.Isotropic(dim, alpha=alpha_i))
        ret.append(numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
    ret = np.array(ret)
    ret_analytical = numqi.state.get_Isotropic_GME(dim, alpha_list)
    assert np.abs(ret-ret_analytical).max() < 1e-7


def test_2qubit_gme():
    model = numqi.entangle.DensityMatrixGMEModel(dim_list=[2,2], num_ensemble=12)
    for _ in range(10):
        rho = numqi.random.rand_density_matrix(4)
        model.set_density_matrix(rho)
        ret = numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun
        ret_ = numqi.entangle.get_gme_2qubit(rho)
        assert abs(ret-ret_) < 1e-7


def test_gme_4qubit():
    # https://doi.org/10.1103/PhysRevA.78.060301
    # (0000 + 0011 + 1100 - 1111)/2
    psi_cluster = np.zeros(16, dtype=np.float64)
    psi_cluster[[0, 3, 12, 15]] = np.array([1,1,1,-1])/2
    rho_cluster = psi_cluster.reshape(-1,1) * psi_cluster.conj()

    # (0000 + 1111)/sqrt(2)
    psi_ghz = numqi.state.GHZ(4)
    rho_ghz = psi_ghz.reshape(-1,1) * psi_ghz.conj()

    psi_W = numqi.state.W(4)
    rho_W = psi_W.reshape(-1,1) * psi_W.conj()

    psi_dicke = numqi.state.Dicke(2, 2)
    rho_dicke = psi_dicke.reshape(-1,1) * psi_dicke.conj()

    xlist = np_rng.uniform(0, 1, size=5)
    ret_cluster = (3/8)*(1 + xlist - np.sqrt(1+(2-3*xlist)*xlist))
    ret_ghz = (1/2)*(1-np.sqrt(1-xlist*xlist))
    tmp0 = xlist>(2183/2667)
    ret_W = tmp0 * (37*(81*xlist-37)/2816) + (1-tmp0) * (3/8)*(1+xlist-np.sqrt(1+(2-3*xlist)*xlist))
    tmp0 = (xlist > 5/7)
    ret_dicke = tmp0 * (5*(3*xlist-1)/16) + (1-tmp0) * (5/18)*(1+2*xlist-np.sqrt(1+(4-5*xlist)*xlist))
    ret_ = np.stack([ret_cluster, ret_ghz, ret_W, ret_dicke])

    model = numqi.entangle.DensityMatrixGMEModel(dim_list=[2,2,2,2], num_ensemble=32)
    mask_diag = np.eye(rho_cluster.shape[0], dtype=np.float64)
    mask_offdiag = 1-mask_diag
    ret_model = []
    for rho in [rho_cluster, rho_ghz, rho_W, rho_dicke]:
        for x in xlist:
            model.set_density_matrix(rho*mask_diag + rho*mask_offdiag*x)
            ret_model.append(numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
    ret_model = np.array(ret_model).reshape(4,-1)

    assert np.abs(ret_-ret_model).max() < 1e-7


def test_flip_op():
    dimA = 3
    dimB = 4
    psi_AB = numqi.random.rand_haar_state(dimA*dimB).reshape(dimA, dimB)

    rdm = psi_AB @ psi_AB.conj().T
    purity = np.vdot(rdm.reshape(-1), rdm.reshape(-1)).real

    flip_op = np.eye(dimA*dimA).reshape(dimA,dimA,dimA,dimA).transpose(0,1,3,2)
    z0 = np.einsum(psi_AB, [0,1], psi_AB.conj(), [2,1], psi_AB, [3,4], psi_AB.conj(), [5,4], flip_op, [2,5,0,3], [], optimize=True).real
    assert abs(purity-z0)<1e-10

def test_get_linear_entropy_entanglement_ppt():
    dimA = 3
    dimB = 3
    rho = numqi.state.get_bes3x3_Horodecki1997(np_rng.uniform(0,1))
    ret,matW = numqi.entangle.get_linear_entropy_entanglement_ppt(rho, (3,3), return_info=True)

    eps = 1e-7 if use_MOSEK else 1e-4 # solver=SCS is less accurate
    assert np.abs(matW - matW.T.conj()).max() < eps
    assert abs(np.trace(matW) - 1) < eps
    assert np.linalg.eigvalsh(matW)[0] > -eps
    tmp0 = matW.reshape(dimA*dimB,dimA*dimB,-1).transpose(1,0,2).reshape(dimA*dimB*dimA*dimB,-1)
    assert np.abs(tmp0 - matW).max() < eps
    tmp0 = matW.reshape(dimA*dimB,dimA*dimB,dimA*dimB,dimA*dimB)
    tmp1 = np.einsum(tmp0,[0,1,2,1],[0,2],optimize=True)
    assert np.abs(tmp1-rho).max() < eps
    tmp1 = np.einsum(tmp0,[0,1,0,2],[1,2],optimize=True)
    assert np.abs(tmp1-rho).max() < eps
    assert np.linalg.eigvalsh(tmp0.transpose(0,3,2,1).reshape(dimA*dimB*dimA*dimB,-1))[0]>-eps
    tmp0 = matW.reshape(dimA,dimB,dimA,dimB,dimA,dimB,dimA,dimB)
    tmp1 = np.einsum(tmp0, [0,1,2,3,4,1,6,3], [0,2,4,6], optimize=True).reshape(dimA*dimA,-1)
    flip_op = np.eye(dimA*dimA).reshape(dimA,dimA,dimA,dimA).transpose(0,1,3,2)
    tmp2 = 1-np.trace((flip_op.reshape(dimA*dimA,-1) @ tmp1)).real
    assert abs(tmp2-ret) < eps


def test_linear_entropy_tensor_network():
    dim0 = 3
    dim1 = 4

    rho = numqi.random.rand_density_matrix(dim0*dim1)
    EVL,EVC = np.linalg.eigh(rho)
    assert np.abs(rho - (EVC*EVL) @ EVC.T.conj()).max() < 1e-10

    manifold = numqi.manifold.Stiefel(2*dim0*dim1, rank=dim0*dim1, dtype=torch.complex128)
    mat_st = manifold().detach().numpy()

    z0 = (EVC*np.sqrt(EVL)) @ mat_st.T
    plist = np.linalg.norm(z0, ord=2, axis=0)**2
    psilist = (z0 / np.sqrt(plist)).T
    z1 = np.einsum(plist, [0], psilist, [0,1], psilist.conj(), [0,2], [1,2], optimize=True)
    assert np.abs(z1 - rho).max() < 1e-10
    tmp0 = psilist.reshape(-1, dim0, dim1)
    rdm = np.einsum(tmp0, [0,1,2], tmp0.conj(), [0,3,2], [0,1,3], optimize=True)
    tmp1 = np.linalg.norm(rdm, ord='fro', axis=(1,2))**2
    ret_ = np.dot(plist, tmp1)

    tmp0 = (EVC * np.sqrt(EVL)).reshape(dim0, dim1, -1)
    tmp1 = np.einsum(tmp0, [0,3,4], tmp0.conj(), [1,3,5], mat_st, [2,4], mat_st.conj(), [2,5], [2,0,1], optimize=True)
    rdm1 = tmp1 / np.trace(tmp1, axis1=1, axis2=2).reshape(-1,1,1)
    assert np.abs(rdm1 - rdm).max() < 1e-10
    tmp2 = np.linalg.norm(tmp1, ord='fro', axis=(1,2))**2
    ret0 = (tmp2 / np.trace(tmp1, axis1=1, axis2=2)).sum()
    # ret0 = np.dot(plist, tmp2)
    assert abs(ret_ - ret0) < 1e-10

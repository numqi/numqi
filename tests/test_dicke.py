import functools
import itertools
import numpy as np
import scipy.linalg
import scipy.special
import torch

import numqi

np_rng = np.random.default_rng()
hf_kron = lambda *x: functools.reduce(np.kron, x)

def test_qudit_partial_trace_AC_to_AB():
    hf_randc = lambda *x: np.random.randn(*x) + 1j*np.random.randn(*x)
    hf_norm = lambda x: x/np.linalg.norm(x)
    for dimA in range(2,5):
        for dimB in range(2, 5):
            for k in range(2, 5):
                Bij = numqi.dicke.get_partial_trace_ABk_to_AB_index(k, dimB)
                num_klist = numqi.dicke.get_dicke_number(k, dimB)
                np0 = hf_norm(hf_randc(dimA*num_klist)).reshape(dimA,num_klist)
                ret0 = numqi.dicke.partial_trace_ABk_to_AB(np0, Bij)
                assert abs(np.trace(ret0)-1)<1e-7
                assert np.all(np.linalg.eigvalsh(ret0)+1e-7>0) #almost PSD (ignoring rounding error)

                Bij_torch = [[torch.tensor(y) for y in x] for x in Bij]
                ret0 = numqi.dicke.partial_trace_ABk_to_AB(torch.tensor(np0), Bij_torch).numpy()
                assert abs(np.trace(ret0)-1)<1e-7
                assert np.all(np.linalg.eigvalsh(ret0)+1e-7>0) #almost PSD (ignoring rounding error)


def test_get_dicke_klist():
    para_list = [(2,2), (2,3), (2,4), (2,5), (3,2), (3,3), (3,4), (3,5)]
    for n,d in para_list:
        tmp0 = np.array(numqi.dicke.get_dicke_klist(n, d))
        assert tmp0.shape==(scipy.special.binom(n+d-1, d-1), d)
        assert np.all(tmp0.sum(axis=1)==n)


def test_get_dicke_basis():
    para_list = [(2,2), (2,3), (2,4), (2,5), (3,2), (3,3), (3,4), (3,5)]
    for n,d in para_list:
        basis = numqi.dicke.get_dicke_basis(n, d)
        assert not np.iscomplexobj(basis)
        N0 = basis.shape[0]
        assert np.abs(basis @ basis.T - np.eye(N0)).max() < 1e-10
        for indI in itertools.permutations(list(range(1,n+1))):
            tmp0 = [0] + list(indI)
            tmp1 = basis.reshape([N0]+[d]*n).transpose(*tmp0).reshape(N0, -1)
            assert np.abs(basis-tmp1).max() < 1e-10


def test_get_partial_trace_ABk_to_AB_index():
    para_list = [(2,2), (2,3), (2,4), (2,5), (3,2), (3,3), (3,4), (3,5)]
    for num_qudit,dim_qudit in para_list:
        Brsab = numqi.dicke.get_partial_trace_ABk_to_AB_index(num_qudit, dim_qudit, return_tensor=True)
        basis = numqi.dicke.get_dicke_basis(num_qudit, dim_qudit)
        N0 = basis.shape[0]
        tmp0 = basis.reshape(N0, dim_qudit, -1)
        ret_ = np.einsum(tmp0, [0,1,2], tmp0, [3,4,2], [1,4,0,3], optimize=True)
        assert np.abs(Brsab - ret_).max() < 1e-10


def test_get_qubit_dicke_partial_trace():
    for num_qubit in [2,3,4,5,6]:
        a00,a01,a11 = numqi.dicke.get_qubit_dicke_partial_trace(num_qubit)
        ret0 = np.zeros((2,2,num_qubit+1,num_qubit+1), dtype=np.float64)
        ret0[0,0] = np.diag(a00)
        ret0[0,1] = np.diag(a01, -1)
        ret0[1,0] = np.diag(a01, 1)
        ret0[1,1] = np.diag(a11)
        ret_ = numqi.dicke.get_partial_trace_ABk_to_AB_index(num_qubit, dim=2, return_tensor=True)
        assert np.abs(ret0-ret_).max() < 1e-10


def test_get_qubit_dicke_rdm_tensor():
    case_list = [(1,x) for x in range(2,8)] + [(2,x) for x in range(3,8)]
    for rdm,n in case_list:
        Tabrs = numqi.dicke.get_qubit_dicke_rdm_tensor(n, rdm)
        basis = numqi.dicke.get_dicke_basis(n, 2)[::-1]
        coeff = numqi.random.rand_haar_state(n+1)
        tmp0 = (coeff @ basis).reshape(2**rdm, -1)
        tmp1 = numqi.dicke.get_dicke_basis(rdm, 2)[::-1]
        ret_ = tmp1 @ np.einsum(tmp0, [0,1], tmp0.conj(), [2,1], [0,2], optimize=True) @ tmp1.T
        ret0 = np.einsum(Tabrs, [0,1,2,3], coeff, [0], coeff.conj(), [1], [2,3], optimize=True)
        assert np.abs(ret_-ret0).max() < 1e-10


# def test_get_qubit_dicke_rdm_pauli_tensor():
#     for n,rdm in [(7,2), (7,4), (8,3)]:
#         Tuab_list,factor_list,pauli_str_list,weight_count = numqi.dicke.get_qubit_dicke_rdm_pauli_tensor(n, rdm)
#         basis = numqi.dicke.get_dicke_basis(n, 2)[::-1]
#         coeff = numqi.random.rand_haar_state(n+1)
#         tmp0 = np.cumsum([0] + [weight_count[x] for x in range(1,rdm)])
#         ind0_list = {(i+1):slice(x,y) for i,(x,y) in enumerate(zip(tmp0,tmp0[1:]))}
#         for wt,ind0 in ind0_list.items():
#             Tuab = Tuab_list[ind0]
#             pauli_str = pauli_str_list[ind0]
#             ret0 = np.einsum(Tuab, [0,1,2], coeff, [1], coeff.conj(), [2], [0], optimize=True).real
#             tmp0 = (coeff @ basis).reshape(2**wt, -1)
#             rho_rdm = np.einsum(tmp0, [0,1], tmp0.conj(), [2,1], [0,2], optimize=True)
#             ret_ = np.array([np.trace(numqi.qec.hf_pauli(x)@rho_rdm) for x in pauli_str])
#             assert np.abs(ret_-ret0).max() < 1e-10


def test_u2_to_dicke():
    for ncopy in [1,2,3,4]:
        np0 = numqi.random.rand_haar_unitary(2)
        basis = numqi.dicke.get_dicke_basis(ncopy, 2)[::-1]
        np1 = np0
        for _ in range(ncopy-1):
            np1 = np.kron(np1, np0)
        ret_ = basis @ np1 @ basis.T
        ret0 = numqi.dicke.u2_to_dicke(np0, ncopy)
        assert np.abs(ret_-ret0).max() < 1e-12
        tmp0 = torch.tensor(np0,dtype=torch.complex128,requires_grad=True)
        ret1 = numqi.dicke.u2_to_dicke(tmp0, ncopy).detach().numpy()
        assert np.abs(ret_-ret1).max() < 1e-12


def dicke_to_u2(R, jxyz=None, zero_eps=1e-12):
    assert R.ndim==2 and R.shape[0]==R.shape[1]
    n = R.shape[-1] - 1 #2j
    if jxyz is None:
        jxyz = np.stack(numqi.matrix_space.get_angular_momentum_op(n), axis=0)
    R3 = np.einsum(R, [0,1], jxyz, [4,1,2], R.T.conj(), [2,3], jxyz, [5,3,0], [4,5], optimize=True).real / (n/2*(n/2+1)*(n+1)/3)

    theta = np.arccos((np.trace(R3)-1)/2)
    if theta < zero_eps:
        nx,ny,nz = 0,0,1 #arbitrary
    else:
        nx,ny,nz = np.array([(R3[2,1]-R3[1,2]), (R3[0,2]-R3[2,0]), (R3[1,0]-R3[0,1])])/(2*np.sin(theta))
    U = np.cos(theta/2)*np.eye(2) + 1j*np.sin(theta/2)*np.array([[nz,nx-1j*ny], [nx+1j*ny, -nz]])
    ret = U/np.sqrt(U[0,0]*U[1,1] - U[0,1]*U[1,0])
    return ret


def test_su2_irrep():
    pxyz = np.stack([numqi.gate.X, numqi.gate.Y, numqi.gate.Z], axis=0)
    for n0 in range(1, 6):
        dicke = numqi.dicke.get_dicke_basis(n0, dim=2)[::-1]
        jxyz = np.stack(numqi.matrix_space.get_angular_momentum_op(n0), axis=0)
        for _ in range(10):
            abc = np_rng.uniform(-np.pi, np.pi, size=3)
            abc = np.array([1,0,0])
            np0 = scipy.linalg.expm(1j*(abc[0]*pxyz[0] + abc[1]*pxyz[1] + abc[2]*pxyz[2])/2)
            np1 = dicke @ hf_kron(*[np0]*n0) @ dicke.T
            np2 = scipy.linalg.expm(1j*(abc[0]*jxyz[0] + abc[1]*jxyz[1] + abc[2]*jxyz[2]))
            assert np.abs(np1-np2).max() < 1e-10

            np3 = numqi.dicke.u2_to_dicke(np0, n0)
            assert np.abs(np1-np3).max() < 1e-10

            np4 = dicke_to_u2(np2)
            assert min(np.abs(np4 - np0).max(), np.abs(np4+np0).max()) < 1e-8

    # n0 = 3
    # pxyz = np.stack([numqi.gate.X, numqi.gate.Y, numqi.gate.Z], axis=0)
    # dicke = numqi.dicke.get_dicke_basis(n0, dim=2)[::-1]
    # jxyz = np.stack(numqi.matrix_space.get_angular_momentum_op(n0), axis=0)
    # x0 = numqi.random.rand_n_sphere(4)

    # x1 = np.eye(jxyz.shape[1])*x0[0] + 1j*(x0[1]*jxyz[0] + x0[2]*jxyz[1] + x0[3]*jxyz[2])*2

    # abc = np_rng.uniform(-0.5, 0.5, size=3)
    # np0 = scipy.linalg.expm(1j*(abc[0]*pxyz[0] + abc[1]*pxyz[1] + abc[2]*pxyz[2]))
    # np1 = dicke @ hf_kron(*[np0]*n0) @ dicke.T
    # np2 = scipy.linalg.expm(1j*(abc[0]*jxyz[0] + abc[1]*jxyz[1] + abc[2]*jxyz[2])*2)
    # assert np.abs(np1-np2).max() < 1e-10

    # tmp0 = np.linalg.norm(abc)
    # tmp1 = np.concat([np.cos(tmp0).reshape(1), np.sin(tmp0)/tmp0 * abc])
    # x0 = tmp1[0]*np.eye(2) + 1j*(tmp1[1]*pxyz[0] + tmp1[2]*pxyz[1] + tmp1[3]*pxyz[2])


def test_get_qubit_dicke_Tabi():
    for n in [1,2,3,4]:
        Tabi = numqi.dicke.get_qubit_dicke_Tabi(n)
        dicke0 = numqi.dicke.get_dicke_basis(n, dim=2)[::-1]
        dicke1 = numqi.dicke.get_dicke_basis(n+1, dim=2)[::-1]
        ret_ = np.einsum(dicke1.reshape(-1, 2**n, 2), [0,1,2], dicke0, [3,1], [0,3,2], optimize=True)
        assert np.abs(Tabi - ret_).max() < 1e-10


def test_get_local_operator_symmetry_projection():
    hf_randc = lambda *x: np_rng.normal(size=x) + 1j*np_rng.normal(size=x)
    hf_kron = lambda *x: functools.reduce(np.kron, x)
    for n0 in [1,2,3,4,5]:
        matA  = hf_randc(n0,2,2)
        ret0 = numqi.dicke.get_local_operator_symmetry_projection(matA)
        basis = numqi.dicke.get_dicke_basis(matA.shape[0], dim=2)[::-1]
        ret_ = basis @ hf_kron(*matA) @ basis.T
        assert np.abs(ret0 - ret_).max() < 1e-10

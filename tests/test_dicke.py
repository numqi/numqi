import itertools
import numpy as np
import scipy.special
import torch

import numqi


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


def test_get_qubit_dicke_rdm_pauli_tensor():
    for n,rdm in [(7,2), (7,4), (8,3)]:
        Tuab_list,factor_list,pauli_str_list,weight_count = numqi.dicke.get_qubit_dicke_rdm_pauli_tensor(n, rdm)
        basis = numqi.dicke.get_dicke_basis(n, 2)[::-1]
        coeff = numqi.random.rand_haar_state(n+1)
        tmp0 = np.cumsum([0] + [weight_count[x] for x in range(1,rdm)])
        ind0_list = {(i+1):slice(x,y) for i,(x,y) in enumerate(zip(tmp0,tmp0[1:]))}
        for wt,ind0 in ind0_list.items():
            Tuab = Tuab_list[ind0]
            pauli_str = pauli_str_list[ind0]
            ret0 = np.einsum(Tuab, [0,1,2], coeff, [1], coeff.conj(), [2], [0], optimize=True).real
            tmp0 = (coeff @ basis).reshape(2**wt, -1)
            rho_rdm = np.einsum(tmp0, [0,1], tmp0.conj(), [2,1], [0,2], optimize=True)
            ret_ = np.array([np.trace(numqi.qec.hf_pauli(x)@rho_rdm) for x in pauli_str])
            assert np.abs(ret_-ret0).max() < 1e-10


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

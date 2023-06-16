import itertools
import numpy as np
import scipy.special

try:
    import torch
except ImportError:
    torch = None

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

                if torch is not None:
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

import itertools
import numpy as np

try:
    import torch
except ImportError:
    torch = None

import numpyqi

def test_mixed_dicke_state():
    np_rng = np.random.default_rng()
    for num_qubitA in range(3, 6):
        for num_qubitB in range(1, 3):
            prob = np_rng.uniform(0, 1, size=(2**num_qubitA, num_qubitB+1))
            tmp0 = numpyqi.dicke.mixed_dicke_state(prob)
            assert abs(np.trace(tmp0, axis1=1, axis2=2).sum()-1) < 1e-7

            if torch is not None:
                prob_torch = torch.tensor(prob, dtype=torch.float64)
                tmp1 = numpyqi.dicke.mixed_dicke_state(prob_torch).numpy()
                assert np.abs(tmp0-tmp1).max() < 1e-7


def test_dicke_state_partial_trace():
    for n in range(2,8):
        z0 = np.zeros((n+1,n+1,2,2), dtype=np.float64)
        tmp0 = np.arange(n+1)
        a00,a01,a10,a11 = numpyqi.dicke.dicke_state_partial_trace(n)
        z0[tmp0,tmp0,0,0] = a00
        z0[tmp0[:-1],tmp0[1:],0,1] = a01
        z0[tmp0[1:],tmp0[:-1],1,0] = a10
        z0[tmp0,tmp0,1,1] = a11

        for x,y in itertools.product(range(n+1),range(n+1)):
            hf0 = lambda x: np.trace(x.reshape(2, x.shape[0]//2, 2, -1), axis1=1, axis2=3)
            tmp0 = numpyqi.dicke.dicke_state(n, x, return_dm=False)
            tmp1 = numpyqi.dicke.dicke_state(n, y, return_dm=False)
            ret_ = hf0(tmp0[:,np.newaxis] * tmp1.conj())
            assert np.abs(ret_-z0[x,y]).max() < 1e-7


def test_partial_trace_AC_to_AB():
    hf_randc = lambda *x: np.random.randn(*x) + 1j*np.random.randn(*x)
    hf_norm = lambda x: x/np.linalg.norm(x)
    for dimA in range(2,5):
        for k in range(5, 25, 3):
            np0 = hf_norm(hf_randc(dimA*k)).reshape(dimA,k)
            ret0 = numpyqi.dicke.partial_trace_AC_to_AB(np0)
            assert abs(np.trace(ret0)-1)<1e-7
            assert np.all(np.linalg.eigvalsh(ret0)+1e-7>0) #almost PSD (ignoring rounding error)

            if torch is not None:
                ret0 = numpyqi.dicke.partial_trace_AC_to_AB(torch.tensor(np0)).numpy()
                assert abs(np.trace(ret0)-1)<1e-7
                assert np.all(np.linalg.eigvalsh(ret0)+1e-7>0) #almost PSD (ignoring rounding error)


def test_qudit_dicke_state_partial_trace():
    for n,d in itertools.product(range(2, 5), range(2, 5)):
        Bij,klist,klist_to_ij = numpyqi.dicke.qudit_dicke_state_partial_trace(d, n)
        hf0 = lambda x: np.trace(x.reshape(d, x.shape[0]//d, d, x.shape[1]//d), axis1=1, axis2=3)
        for ki,kj in itertools.product(klist, klist):
            tmp0 = [(Bij[x][2][klist_to_ij[ki,kj,x]] if ((ki,kj,x) in klist_to_ij) else 0) for x in range(d**2)]
            ret_ = np.array(tmp0).reshape(d,d)
            tmp0 = numpyqi.dicke.qudit_dicke_state(*ki)
            tmp1 = numpyqi.dicke.qudit_dicke_state(*kj)
            ret0 = hf0(tmp0[:,np.newaxis] * tmp1.conj())
            assert np.abs(ret_-ret0).max()<1e-7


def test_qudit_partial_trace_AC_to_AB():
    hf_randc = lambda *x: np.random.randn(*x) + 1j*np.random.randn(*x)
    hf_norm = lambda x: x/np.linalg.norm(x)
    for dimA in range(2,5):
        for dimB in range(2, 5):
            for k in range(2, 5):
                Bij,klist,klist_to_ij = numpyqi.dicke.qudit_dicke_state_partial_trace(dimB, k)
                np0 = hf_norm(hf_randc(dimA*len(klist))).reshape(dimA,len(klist))
                ret0 = numpyqi.dicke.qudit_partial_trace_AC_to_AB(np0, Bij)
                assert abs(np.trace(ret0)-1)<1e-7
                assert np.all(np.linalg.eigvalsh(ret0)+1e-7>0) #almost PSD (ignoring rounding error)

                if torch is not None:
                    Bij_torch = [[torch.tensor(y) for y in x] for x in Bij]
                    ret0 = numpyqi.dicke.qudit_partial_trace_AC_to_AB(torch.tensor(np0), Bij_torch).numpy()
                    assert abs(np.trace(ret0)-1)<1e-7
                    assert np.all(np.linalg.eigvalsh(ret0)+1e-7>0) #almost PSD (ignoring rounding error)

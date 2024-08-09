import numpy as np
import torch

import numqi


def test_get_hyperdeterminant():
    N0 = 3
    np0 = np.random.randn(N0, 8) + 1j*np.random.randn(N0, 8)
    ret0 = numqi.entangle.get_hyperdeterminant(np0, mode='tensor')
    ret1 = numqi.entangle.get_hyperdeterminant(np0, mode='index')
    assert np.abs(ret0-ret1).max()<1e-10
    ret2 = numqi.entangle.get_hyperdeterminant(torch.tensor(np0, dtype=torch.complex128), mode='tensor').numpy()
    ret3 = numqi.entangle.get_hyperdeterminant(torch.tensor(np0, dtype=torch.complex128), mode='index').numpy()
    assert np.abs(ret0-ret2).max()<1e-10
    assert np.abs(ret0-ret3).max()<1e-10

    np0 = np.random.randn(N0, 8)
    ret0 = numqi.entangle.get_hyperdeterminant(np0, mode='tensor')
    ret1 = numqi.entangle.get_hyperdeterminant(np0, mode='index')
    assert np.abs(ret0-ret1).max()<1e-10
    ret2 = numqi.entangle.get_hyperdeterminant(torch.tensor(np0, dtype=torch.float64), mode='tensor').numpy()
    ret3 = numqi.entangle.get_hyperdeterminant(torch.tensor(np0, dtype=torch.float64), mode='index').numpy()
    assert np.abs(ret0-ret2).max()<1e-10
    assert np.abs(ret0-ret3).max()<1e-10


def _get_3tangle_GHZ_W_hf0(abcdf, p):
    a, b, c, d, f = abcdf
    tmp0 = 1/np.sqrt(abs(a)**2+abs(b)**2)
    a,b = a*tmp0, b*tmp0
    tmp0 = 1/np.sqrt(abs(c)**2 + abs(d)**2 + abs(f)**2)
    c,d,f = c*tmp0, d*tmp0, f*tmp0
    assert 0<=p<=1
    return a,b,c,d,f,p

def get_3tangle_GHZ_W_pure(abcdf, p, phi):
    # https://doi.org/10.1088/1367-2630/10/4/043014
    a,b,c,d,f,p = _get_3tangle_GHZ_W_hf0(abcdf, p)
    tmp0 = np.sqrt(np.maximum(p*((1-p)**3), 0))
    ret = 4 * np.abs(p*p*a*a*b*b - 4*tmp0*np.exp(3j*phi)*b*c*d*f)
    return ret

def get_GHZ_W_state_pure(abcdf, p, phi):
    # https://doi.org/10.1088/1367-2630/10/4/043014
    a,b,c,d,f,p = _get_3tangle_GHZ_W_hf0(abcdf, p)
    ret = np.zeros(8, dtype=np.complex128)
    ret[0] = a*np.sqrt(p)
    ret[7] = b*np.sqrt(p)
    tmp0 = -np.sqrt(max(0, 1-p)) * np.exp(1j*phi)
    ret[1] = c*tmp0
    ret[2] = d*tmp0
    ret[4] = f*tmp0
    return ret

def test_get_3tangle_GHZ_W_pure():
    # https://github.com/numqi/dm-stiefel/blob/main/draft_3tangle.py
    theta_list = np.linspace(0, np.pi, 10)
    model = numqi.entangle.ThreeTangleModel(num_term=4*8)

    ret_analytical = []
    ret_opt = []
    kwargs = dict(theta0='uniform', num_repeat=3, tol=1e-10, print_every_round=0)
    for theta_i in theta_list:
        abcdf = np.cos(theta_i), np.sin(theta_i), 1, 0, 0
        p = 1
        phi = 0
        ret_analytical.append(get_3tangle_GHZ_W_pure(abcdf, p, phi))
        psi = get_GHZ_W_state_pure(abcdf, p, phi)
        model.set_density_matrix(psi.reshape(8,1) * psi.conj())
        ret_opt.append(numqi.optimize.minimize(model, **kwargs).fun)
    ret_analytical = np.array(ret_analytical)
    ret_opt = np.array(ret_opt)
    assert np.abs(ret_analytical-ret_opt).max()<1e-10

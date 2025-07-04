import functools
import numpy as np
import numqi

np_rng = np.random.default_rng()
hf_kron = lambda *x: functools.reduce(np.kron, x)

def test_get_BD38_LP():
    code,info = numqi.qec.q823.get_BD38_LP(return_info=True)
    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(8, distance=3, kind='scipy-csr01')
    z0 = code.conj() @ (error_scipy @ code.T).reshape(-1,2**8,2)
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0] - z0[:,1,1]).max() < 1e-10
    logicalU = code.conj() @ hf_kron(*info['su2']) @ code.T
    assert np.abs(numqi.gate.rz(-2*np.pi/19) - logicalU).max() < 1e-10


def test_get_BD64():
    sign = np_rng.integers(2, size=9)*2 - 1
    theta = np_rng.uniform(0, 2*np.pi, size=3)
    code,info = numqi.qec.q823.get_BD64(theta, sign, return_info=True)
    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(8, distance=3, kind='scipy-csr01')
    z0 = code.conj() @ (error_scipy @ code.T).reshape(-1,2**8,2)
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0] - z0[:,1,1]).max() < 1e-10
    logicalU = code.conj() @ hf_kron(*info['su2']) @ code.T
    assert np.abs(numqi.gate.rz(-2*np.pi/32) - logicalU).max() < 1e-10


def test_get_BD72():
    theta = np_rng.uniform(0, 2*np.pi, size=5)
    sign = np_rng.integers(2, size=3)*2-1
    code,info = numqi.qec.q823.get_BD72(theta=theta, sign=sign, return_info=True)
    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(8, distance=3, kind='scipy-csr01')
    z0 = code.conj() @ (error_scipy @ code.T).reshape(-1,2**8,2)
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0] - z0[:,1,1]).max() < 1e-10
    logicalU = code.conj() @ hf_kron(*info['su2']) @ code.T
    assert np.abs(numqi.gate.rz(-2*np.pi/36) - logicalU).max() < 1e-10


def test_get_BD74():
    sign = np_rng.integers(2, size=4)*2-1
    theta = np_rng.uniform(0, 2*np.pi, size=4)
    code, info = numqi.qec.q823.get_BD74(theta=theta, sign=sign)
    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(8, distance=3, kind='scipy-csr01')
    z0 = code.conj() @ (error_scipy @ code.T).reshape(-1,2**8,2)
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0] - z0[:,1,1]).max() < 1e-10
    logicalU = code.conj() @ hf_kron(*info['su2']) @ code.T
    assert np.abs(numqi.gate.rz(-2*np.pi/37) - logicalU).max() < 1e-10


def test_get_BD76():
    theta = np_rng.uniform(0, 2*np.pi, size=6)
    sign = np_rng.integers(2, size=2)*2-1
    code, info = numqi.qec.q823.get_BD76(theta=theta, sign=sign, return_info=True)
    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(8, distance=3, kind='scipy-csr01')
    z0 = code.conj() @ (error_scipy @ code.T).reshape(-1,2**8,2)
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0] - z0[:,1,1]).max() < 1e-10
    logicalU = code.conj() @ hf_kron(*info['su2']) @ code.T
    assert np.abs(numqi.gate.rz(-2*np.pi/38) - logicalU).max() < 1e-10


def test_get_BD78():
    theta = np_rng.uniform(0, 2*np.pi, size=6)
    sign = np_rng.integers(2, size=2)*2-1
    code,info = numqi.qec.q823.get_BD78(theta=theta, sign=sign, return_info=True)
    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(8, distance=3, kind='scipy-csr01')
    z0 = code.conj() @ (error_scipy @ code.T).reshape(-1,2**8,2)
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0] - z0[:,1,1]).max() < 1e-10
    logicalU = code.conj() @ hf_kron(*info['su2']) @ code.T
    assert np.abs(numqi.gate.rz(-2*np.pi/39) - logicalU).max() < 1e-10


def test_get_BD80():
    theta = np_rng.uniform(0, 2*np.pi, size=6)
    sign = np_rng.integers(2, size=2)*2-1
    code, info = numqi.qec.q823.get_BD80(theta=theta, sign=sign, return_info=True)
    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(8, distance=3, kind='scipy-csr01')
    z0 = code.conj() @ (error_scipy @ code.T).reshape(-1,2**8,2)
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0] - z0[:,1,1]).max() < 1e-10
    logicalU = code.conj() @ hf_kron(*info['su2']) @ code.T
    assert np.abs(numqi.gate.rz(-2*np.pi/40) - logicalU).max() < 1e-10


def test_get_BD84():
    theta = np_rng.uniform(0, 2*np.pi, size=7)
    sign = np_rng.integers(2)*2-1
    code, info = numqi.qec.q823.get_BD84(theta=theta, sign=sign, return_info=True)
    error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(8, distance=3, kind='scipy-csr01')
    z0 = code.conj() @ (error_scipy @ code.T).reshape(-1,2**8,2)
    assert np.abs(z0[:,0,1]).max() < 1e-10
    assert np.abs(z0[:,1,0]).max() < 1e-10
    assert np.abs(z0[:,0,0] - z0[:,1,1]).max() < 1e-10
    logicalU = code.conj() @ hf_kron(*info['su2']) @ code.T
    assert np.abs(numqi.gate.rz(-2*np.pi/42) - logicalU).max() < 1e-10

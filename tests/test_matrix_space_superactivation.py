import numpy as np
import pytest

import numqi

try:
    import torch
except ImportError:
    torch = None

np_rng = np.random.default_rng()

def test_superactivation_matrix_subspace():
    theta = np_rng.uniform(0, np.pi/2)
    npA,npB,npAB,npL,npR,field = numqi.matrix_space.get_matrix_subspace_example('0error-eq537', theta)

    assert np.abs((npAB @ npR) @ npL).max() < 1e-12
    assert np.abs((npAB @ npL) @ npR).max() < 1e-12

    unitary0 = numqi.random.rand_unitary_matrix(4, tag_complex=False)
    unitary1 = numqi.random.rand_unitary_matrix(4, tag_complex=False)
    npA1 = unitary0 @ npA @ unitary0.T
    npB1 = unitary1 @ npB @ unitary1.T
    npAB1 = np.stack([np.kron(x,y) for x in npA1 for y in npB1])
    npL1 = unitary0 @ npL.reshape(4,4) @ unitary1.T
    npR1 = unitary0 @ npR.reshape(4,4) @ unitary1.T
    assert np.abs((npL1.reshape(-1) @ npAB1) @ npR1.reshape(-1)).max() < 1e-12
    assert np.abs((npR1.reshape(-1) @ npAB1) @ npL1.reshape(-1)).max() < 1e-12
    # assert np.abs(np.einsum(npA1, [0,1,2], npB1, [3,4,5], npL1, [1,4], npR1, [2,5], [0,3], optimize=True)).max() < 1e-12

    theta_list = np_rng.uniform(0, np.pi/2, size=9)
    alpha_list = np_rng.uniform(-1, 2, size=11)
    ret = []
    for theta_i in theta_list:
        npA,npB,npAB,npL,npR,field = numqi.matrix_space.get_matrix_subspace_example('0error-eq537', theta_i)
        ret.append([])
        for alpha_i in alpha_list:
            tmp0 = alpha_i*npL + (1-alpha_i)*npR
            tmp1 = alpha_i*npR + (1-alpha_i)*npL
            tmp0 /= np.linalg.norm(tmp0)
            tmp1 /= np.linalg.norm(tmp1)
            ret[-1].append(np.linalg.norm((npAB @ tmp0) @ tmp1)**2)
    ret = np.array(ret)
    assert np.std(ret, axis=0).max() < 1e-10

    EVL,_ = numqi.matrix_space.detect_antisym_y_Ux(npAB, a=1)
    assert abs(EVL[0]) < 1e-10
    assert abs(EVL[1]) > 0.01


@pytest.mark.skipif(torch is None, reason='pytorch is not installed')
def test_superactivation_matrix_subspace01():
    npA,npB,npAB,npL,npR,field = numqi.matrix_space.get_matrix_subspace_example('0error-eq537')
    # not a public api yet
    model = numqi.matrix_space._gradient.DetectOrthogonalRankOneModel(npAB, dtype='float64')
    model.theta0.data[:] = torch.tensor(np.eye(4).reshape(-1)/2)
    model.theta1.data[:] = torch.tensor(np.diag([1,-1,1,-1]).reshape(-1)/2)
    assert abs(model().item()) <1e-10
    hess = numqi.optimize.get_model_hessian(model)
    EVL = np.linalg.eigvalsh(hess)
    assert np.abs(EVL[:2]).max() < 1e-10


def test_matrix_subspace_to_biquadratic_form():
    hf_randc = lambda *x: np_rng.normal(size=x) + 1j*np_rng.normal(size=x)
    num_matrix = 4
    dimA = 5
    dimB = 6
    matrix_space = hf_randc(num_matrix, dimA, dimB)
    matrix_space_biq = numqi.matrix_space.matrix_subspace_to_biquadratic_form(matrix_space)
    tmp0 = hf_randc(dimA)
    tmp1 = hf_randc(dimB)
    z0 = np.sum(np.abs((tmp0 @ matrix_space) @ tmp1)**2)
    tmp2 = (tmp0[:,np.newaxis]*tmp1).reshape(-1)
    z1 = tmp2.conj() @ (matrix_space_biq.reshape(tmp2.shape[0], -1) @ tmp2)
    assert np.abs(z0-z1)<1e-10


def test_misc00():
    ## doi.org/10.1007/978-3-319-42794-2 quantum error-error information theory eq-5.9
    hf_HS_norm = lambda x,y: np.dot(x.reshape(-1), y.reshape(-1).conj())

    tmp0 = np.array([[0.5,0,0,0,np.sqrt(49902)/620], [0.5,-0.5,0,0,0], [0,0.5,-0.5,0,0],
            [0,0,0.5,-np.sqrt(457)/50,np.sqrt(457)/50], [0,0,0,-0.62,-289/1550]])
    tmp1 = np.array([[0.5,0,0,0,-np.sqrt(49902)/620], [0.5,0.5,0,0,0],[0,0.5,0.5,0,0],
            [0,0,0.5,np.sqrt(457)/50,-np.sqrt(457)/50],[0,0,0,0.5,0.5]])
    tmp2 = np.diag([0,0,0,0,0.3])
    kraus_op = np.stack([tmp0,tmp1,tmp2])
    assert np.abs(kraus_op.reshape(15,5).T @ kraus_op.reshape(15,5).conj() - np.eye(5)).max() < 1e-10

    state_list = np.array([
        [1,0,0,0,0],
        [0,1,0,0,0],
        [0,0,1,0,0],
        [0,0,0,1,0],
        [0,0,0,1/np.sqrt(2),1/np.sqrt(2)],
    ])
    hf_state_to_dm = lambda x: x.reshape(-1,1)*x.conj().reshape(-1)
    z0 = [numqi.channel.apply_kraus_op(kraus_op, hf_state_to_dm(x)) for x in state_list]
    z1 = np.array([[hf_HS_norm(x,y) for y in z0] for x in z0])
    mask = np.array([[0,0,1,1,0], [0,0,0,1,1], [1,0,0,0,1], [1,1,0,0,0], [0,1,1,0,0]], dtype=np.bool_)
    assert np.abs(z1*mask).max() < 1e-10

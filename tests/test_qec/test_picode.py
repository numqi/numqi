import numpy as np

import numqi

def test_permutation713code_beth():
    s = np.sqrt
    tmp0 = np.array([s(15), 0, -s(7), 0, s(21), 0, s(21), 0])/8
    # tmp0 = np.array([s(15), s(7), s(21), -s(21)])/8
    coeff = np.stack([tmp0, tmp0[::-1]], axis=1)
    Taij,factor_list,pauli_str_list,weight_count = numqi.dicke.get_qubit_dicke_rdm_pauli_tensor(7, 2, kind='scipy-csr01')
    lambda_aij = coeff.T.conj() @ (Taij @ coeff).reshape(-1, coeff.shape[0], coeff.shape[1])
    assert np.abs(lambda_aij[:,0,1]).max() < 1e-12
    assert np.abs(lambda_aij[:,1,0]).max() < 1e-12
    assert np.abs(lambda_aij[:,0,0].imag).max() < 1e-12
    assert np.abs(lambda_aij[:,0,0] - lambda_aij[:,1,1]).max() < 1e-12
    assert abs(np.dot(lambda_aij[:,0,0].real**2, factor_list) - 7) < 1e-10

    np1 = coeff.T.conj() @ numqi.dicke.u2_to_dicke(numqi.gate.X, 7) @ coeff
    assert np.abs(np1-numqi.gate.X).max() < 1e-10
    np1 = coeff.T.conj() @ numqi.dicke.u2_to_dicke(numqi.gate.Z, 7) @ coeff
    assert np.abs(np1-numqi.gate.Z).max() < 1e-10
    np0 = numqi.gate.H @ numqi.gate.rz(-np.pi/2)
    np1 = coeff.T.conj() @ numqi.dicke.u2_to_dicke(np0, 7) @ coeff
    assert np.abs(np1-np0.conj()).max() < 1e-10
    a = (1+np.sqrt(5))/2
    b = (1-np.sqrt(5))/2 #a*b=-1
    np0 = np.array([[b + 1j/b, 1], [-1, b - 1j/b]])/2
    np1 = coeff.T.conj() @ numqi.dicke.u2_to_dicke(np0, 7) @ coeff
    logicalU = np.array([[a - 1j/a, 1], [-1, a + 1j/a]])/2
    assert np.abs(np1 - logicalU).max() < 1e-10

    a = (1+np.sqrt(5))/2
    b = (1-np.sqrt(5))/2 #a*b=-1
    np0 = np.array([[b - 1j/b, 1], [-1, b + 1j/b]])/2
    np1 = coeff.T.conj() @ numqi.dicke.u2_to_dicke(np0, 7) @ coeff
    logicalU = np.array([[a + 1j/a, 1], [-1, a - 1j/a]])/2
    assert np.abs(np1 - logicalU).max() < 1e-10

    # tmp0 = np.array([s(15), 0, -s(7), 0, s(21), 0, s(21), 0])/8
    tmp0 = np.array([s(15), 0, s(7), 0, s(21), 0, -s(21), 0])/8
    # tmp0 = np.array([s(15), s(7), s(21), -s(21)])/8
    coeff = np.stack([tmp0, tmp0[::-1]], axis=1)
    np0 = np.array([[a + 1j/a, 1], [-1, a - 1j/a]])/2
    np1 = coeff.T.conj() @ numqi.dicke.u2_to_dicke(np0, 7) @ coeff
    logicalU = np.array([[b - 1j/b, 1], [-1, b + 1j/b]])/2
    assert np.abs(np1 - logicalU).max() < 1e-10

    code,info = numqi.qec.get_code_subspace('723permutation', sign='-')
    basis = numqi.dicke.get_dicke_basis(7, dim=2)
    coeff = code @ basis.T.conj()
    assert np.abs(coeff.conj() @ coeff.T - np.eye(2)).max() < 1e-10
    # X Z F Phi
    np1 = coeff.conj() @ numqi.dicke.u2_to_dicke(numqi.gate.X, 7) @ coeff.T
    assert np.abs(np1-numqi.gate.X).max() < 1e-10
    np1 = -coeff.conj() @ numqi.dicke.u2_to_dicke(numqi.gate.Z, 7) @ coeff.T
    assert np.abs(np1-numqi.gate.Z).max() < 1e-10
    np0 = numqi.qec.su2_finite_subgroup_gate_dict['F']
    np1 = 1j * (coeff.conj() @ numqi.dicke.u2_to_dicke(numqi.gate.Z @ np0.conj(), 7) @ coeff.T)
    assert np.abs(np0-np1).max() < 1e-10
    np0 = numqi.qec.su2_finite_subgroup_gate_dict['Phi*']
    np1 = coeff.conj() @ numqi.dicke.u2_to_dicke(numqi.qec.su2_finite_subgroup_gate_dict['Phi'].T.conj(), 7) @ coeff.T
    assert np.abs(np0-np1).max() < 1e-10


def test_permutation713code_q212():
    # https://arxiv.org/pdf/2310.05358 eq(14) even-odd code
    # transversal group 2I
    s = np.sqrt
    tmp0 = np.array([s(3/10),0,0,0,0,1j*s(7/10),0,0])
    tmp1 = np.array([0,0,1j*s(7/10),0,0,0,0,s(3/10)])
    coeff = np.stack([tmp0, tmp1], axis=1)
    Taij = numqi.dicke.get_qubit_dicke_rdm_pauli_tensor(7, 2, kind='scipy-csr01')[0]
    lambda_aij = coeff.T.conj() @ (Taij @ coeff).reshape(-1, coeff.shape[0], coeff.shape[1])
    assert np.abs(lambda_aij[:,0,1]).max() < 1e-12
    assert np.abs(lambda_aij[:,1,0]).max() < 1e-12
    assert np.abs(lambda_aij[:,0,0].imag).max() < 1e-12
    assert np.abs(lambda_aij[:,0,0] - lambda_aij[:,1,1]).max() < 1e-12

    np1 = coeff.T.conj() @ numqi.dicke.u2_to_dicke(numqi.gate.X, 7) @ coeff
    assert np.abs(np1 - numqi.gate.X).max() < 1e-10
    np0 = numqi.gate.rz(6*np.pi/5)
    np1 = coeff.T.conj() @ numqi.dicke.u2_to_dicke(np0, 7) @ coeff
    assert np.abs(np1 - numqi.gate.rz(2*np.pi/5)).max() < 1e-10
    I,X,Y,Z = numqi.gate.I, numqi.gate.X, numqi.gate.Y, numqi.gate.Z
    hfR = lambda a,b,t=1: I*np.cos(t*np.pi/5) + 1j*np.sin(t*np.pi/5)/np.sqrt(5) * (a*Y + b*Z)
    np0 = coeff.T.conj() @ numqi.dicke.u2_to_dicke(hfR(-2,-1,3), 7) @ coeff
    assert np.abs(hfR(-2,1) - np0).max() < 1e-10


def test_permutation_11_1_3code():
    s = np.sqrt
    # https://arxiv.org/abs/2411.13142 eq(2)
    tmp0 = np.array([s(5),0,0,0,0,0,0,0,s(11),0,0,0])/4
    tmp1 = np.array([0,0,0,s(11),0,0,0,0,0,0,0,s(5)])/4
    coeff = np.stack([tmp0, tmp1], axis=1)
    Taij = numqi.dicke.get_qubit_dicke_rdm_pauli_tensor(11, 2, kind='scipy-csr01')[0]
    lambda_aij = coeff.T.conj() @ (Taij @ coeff).reshape(-1, coeff.shape[0], coeff.shape[1])
    assert np.abs(lambda_aij[:,0,1]).max() < 1e-12
    assert np.abs(lambda_aij[:,1,0]).max() < 1e-12
    assert np.abs(lambda_aij[:,0,0].imag).max() < 1e-12
    assert np.abs(lambda_aij[:,0,0] - lambda_aij[:,1,1]).max() < 1e-12

    # transversal gate
    np0 = numqi.gate.rz(3*np.pi/4)
    np1 = coeff.T.conj() @ numqi.dicke.u2_to_dicke(np0, 11) @ coeff
    assert np.abs(numqi.gate.rz(np.pi/4) - np1).max() < 1e-12
    np0 = numqi.gate.rz(np.pi/4)
    np1 = coeff.T.conj() @ numqi.dicke.u2_to_dicke(np0, 11) @ coeff
    assert np.abs(numqi.gate.rz(3*np.pi/4) + np1).max() < 1e-12
    np1 = coeff.T.conj() @ numqi.dicke.u2_to_dicke(numqi.gate.X, 11) @ coeff
    assert np.abs(np1-numqi.gate.X).max() < 1e-10
    np1 = coeff.T.conj() @ numqi.dicke.u2_to_dicke(numqi.gate.Z, 11) @ coeff
    assert np.abs(np1-numqi.gate.Z).max() < 1e-10


def test_get_bg_picode():
    for b2,g in [(6,3),(7,3),(8,3),(9,3),(10,3)]:
        # if b2>=g+3 and g>=3, then at least d>=3
        assert (b2>=g+3) and (g>=3)
        coeff = numqi.qec.get_bg_picode(b2,g)
        Taij = numqi.dicke.get_qubit_dicke_rdm_pauli_tensor(coeff.shape[0]-1, 2, kind='scipy-csr01')[0]
        lambda_aij = coeff.T.conj() @ (Taij @ coeff).reshape(-1, coeff.shape[0], coeff.shape[1])
        assert np.abs(lambda_aij[:,0,1]).max() < 1e-12
        assert np.abs(lambda_aij[:,1,0]).max() < 1e-12
        assert np.abs(lambda_aij[:,0,0].imag).max() < 1e-12
        assert np.abs(lambda_aij[:,0,0] - lambda_aij[:,1,1]).max() < 1e-12

        if b2%2==0:
            np0 = coeff.T.conj() @ numqi.dicke.u2_to_dicke(numqi.gate.rz(2*np.pi/b2), coeff.shape[0]-1) @ coeff
            np1 = numqi.gate.rz(2*np.pi*g/b2)
            assert np.abs(np0+np1).max() < 1e-10


def test_723permutation_code():
    for sign in '+-':
        code,info = numqi.qec.get_code_subspace('723permutation', sign=sign)
        op_list = numqi.qec.make_pauli_error_list_sparse(num_qubit=7, distance=3, kind='scipy-csr01')[1]
        z0 = code.conj() @ (op_list @ code.T).reshape(-1, 2**7, 2)
        assert np.abs(z0[:,0,1]).max() < 1e-12
        tmp0 = np.array([2/3]*18 + [1] + [3]*3)
        ind0 = [21, 25, 29, 30, 34, 38, 39, 43, 47, 48, 52, 56, 57, 61, 65, 66, 70, 74, 75, 79, 83,
                84, 88, 92, 93, 97, 101, 102, 106, 110, 111, 115, 119, 120, 124, 128, 129, 133, 137,
                138, 142, 146, 147, 151, 155, 156, 160, 164, 165, 169, 173, 174, 178, 182, 183, 187,
                191, 192, 196, 200, 201, 205, 209]
        tmp0 = np.zeros(210, dtype=np.float64)
        tmp0[ind0] = 1/3
        assert np.abs(z0[:,0,0]-tmp0).max() < 1e-10


def test_723permutation_code_local_unitary_equivalent():
    codep,info = numqi.qec.get_code_subspace('723permutation', sign='+')
    codem,info = numqi.qec.get_code_subspace('723permutation', sign='-')
    model = numqi.qec.QECCEqualModel(codep, codem)
    # theta0 = numqi.optimize.minimize(model, 'uniform', num_repeat=10, tol=1e-24).x
    theta0 = np.array([-0.9851071268903094, 0.9851071309835814, -0.9851071080929052, 0.8286922424445572, -0.8286922385150921,
            0.8286922425522644, 0.8286922410131242, -0.8286922420980595, 0.8286922443320521, 0.8286922422542674, -0.8286922398812964,
            0.8286922415662342, -0.9851071239082392, 0.9851071216442577, -0.9851071202221557, -0.9851071170253412, 0.9851071293328815,
            -0.9851071218120353, -0.9851071289119772, 0.9851071116144996, -0.985107125489987])
    numqi.optimize.set_model_flat_parameter(model, theta0)
    assert abs(model().item()) < 1e-12

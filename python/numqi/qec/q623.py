import itertools
import numpy as np

import numqi.gate
import numqi.random

from ._pauli import make_pauli_error_list_sparse

def get_C10(return_info=False):
    a = np.sqrt(1/10) #0.3162
    b = np.sqrt(1/15) #0.2582
    p3 = np.exp(1j*np.pi/3)
    p6 = p3*p3
    code = np.zeros((2,64), dtype=np.complex128)
    code[0, [0,3,13,21,25,30,37,41,46,49,54,58]] = np.array([a,-a,b,b*p6,b/p6,a,b*p3,b/p3,-a,-b,-a,-a])
    code[1, [5,9,14,17,22,26,33,38,42,50,60,63]] = np.array([a,a,b,a,b/p6,b*p6,-a,b/p3,b*p3,-b,-a,a])
    ret = code
    if return_info:
        qweA = np.array([1, 0, 0.84, 0, 11.64, 15.36, 3.16])
        qweB = np.array([1, 0, 0.84, 23.36, 36.6, 39.36, 26.84])
        su2 = numqi.gate.rz(np.array([1, 1, 1, 1, 2, 3])*2*np.pi/5) #rz(-2*np.pi/5)
        # TODO lambda_ai
        info = dict(qweA=qweA, qweB=qweB, su2=su2)
        ret = code,info
    return ret


def get_SO5_code_basis(phase=0, tag_kron=True, seed=None):
    np_rng = numqi.random.get_numpy_rng(seed)
    if phase is None:
        expia = np.exp(1j*np_rng.uniform(0, 2*np.pi, size=5))
    else:
        phase = np.asarray(phase).reshape(-1)
        assert len(phase) in {1,5}
        if len(phase)==1:
            expia = np.exp(1j*phase*np.ones(5))
        else:
            expia = np.exp(1j*phase)
    # hf0 = lambda x: [int(y,base=2) for y in x.split(' ')]
    # ind0 = hf0('00001 00010 00100 01000 10000')
    # ind1 = hf0('11110 11101 11011 10111 01111')
    ind0,ind1 = [1,2,4,8,16], [30,29,27,23,15]
    basisA = np.zeros((5,32), dtype=np.float64)
    basisA[np.arange(5), ind0] = 1/np.sqrt(2)
    basisB = np.zeros((5,32), dtype=np.float64)
    basisB[np.arange(5), ind1] = 1/np.sqrt(2)
    basis = basisA + expia.reshape(5,1)*basisB
    if tag_kron:
        basis = np.kron(np.eye(2), basis)
    return basis


def get_SO5_code(vece_or_abcde, phase=0, return_info=False, seed=None, zero_eps=1e-10):
    # https://arxiv.org/abs/2410.07983
    np_rng = numqi.random.get_numpy_rng(seed)
    vece_or_abcde = np.asarray(vece_or_abcde)
    assert vece_or_abcde.shape in {(5,), (5,5)}
    if vece_or_abcde.shape==(5,):
        vece = vece_or_abcde
        assert abs(np.linalg.norm(vece)-1) < zero_eps, 'unit vector required'
        tmp0 = np.linalg.eigh(np.eye(5) - vece.reshape(-1,1)*vece)[1][:,1:].T
        tmp1 = numqi.random.rand_special_orthogonal_matrix(4, seed=np_rng) @ tmp0
        matO = np.concatenate([tmp1, vece.reshape(1,-1)], axis=0).T
    else:
        assert np.abs(vece_or_abcde @ vece_or_abcde.T - np.eye(5)).max() < zero_eps, 'orthogonal matrix required'
        matO = vece_or_abcde
    veca,vecb,vecc,vecd,vece = matO.T
    coeff0 = np.concatenate([veca+1j*vecb, vecc+1j*vecd], axis=0)/2
    coeff1 = np.concatenate([coeff0[5:].conj(), -coeff0[:5].conj()], axis=0)
    coeff = np.stack([coeff0, coeff1], axis=0)
    basis = get_SO5_code_basis(phase, seed=np_rng)
    code = coeff @ basis
    ret = code
    if return_info:
        lambda_ai_dict = dict()
        for ind0,ind1 in itertools.combinations(range(5), 2):
            tmp0 = ['I']*6
            tmp0[5-ind0] = 'X'
            tmp0[5-ind1] = 'X'
            lambda_ai_dict[''.join(tmp0)] = -vece[ind0]*vece[ind1]/2
            tmp0[5-ind0] = 'Y'
            tmp0[5-ind1] = 'Y'
            lambda_ai_dict[''.join(tmp0)] = -vece[ind0]*vece[ind1]/2
            tmp0[5-ind0] = 'Z'
            tmp0[5-ind1] = 'Z'
            lambda_ai_dict[''.join(tmp0)] = vece[ind0]*vece[ind0]/2 + vece[ind1]*vece[ind1]/2
        error_str_list = make_pauli_error_list_sparse(num_qubit=6, distance=3, kind='scipy-csr01')[0]
        lambda_ai = np.array([lambda_ai_dict.get(x,0) for x in error_str_list], dtype=np.float64)
        tmp0 = (vece**4).sum()
        qweA = np.array([1, 0, 0.5+0.5*tmp0, 0.5-0.5*tmp0, 11.5-0.5*tmp0, 15.5+0.5*tmp0, 3]) #by numerical fitting
        qweB = np.array([1, 0, 0.5+0.5*tmp0, 23+tmp0, 37-2*tmp0, 41-tmp0, 25.5+1.5*tmp0])
        info = dict(coeff=coeff, basis=basis, phase=phase, vece=vece, matO=matO, lambda_ai=lambda_ai, qweA=qweA, qweB=qweB)
        ret = code, info
    return ret


def get_SO5_code_with_transversal_gate(vece:np.ndarray):
    assert (vece.ndim==1) and (vece.shape[0] in (2,3,4,5))
    assert abs(np.dot(vece,vece)-1) < 1e-10, 'unit vector required'
    X,Y,Z,I = numqi.gate.X, numqi.gate.Y, numqi.gate.Z, numqi.gate.I
    if vece.shape[0]==2: #BD4
        r,s = vece
        matO = 0.5*np.array([[-s,s,-s,s,2*r], [r,-r,r,-r,2*s], [1,1,1,1,0], [1,-1,-1,1,0], [1,1,-1,-1,0]])
        code,info = get_SO5_code(matO, return_info=True)
        s3 = np.sqrt(3)
        info['su2X'] = np.stack([2*Y, s3*X+Y, X-s3*Y, s3*X+Y, X-s3*Y, X-s3*Y], axis=0) * 0.5j
        info['su2Z'] = np.stack([1j*X, 1j*Z, -Z, I, I, I], axis=0)
    elif vece.shape[0]==3: #C4
        r,s,t = vece
        alpha = np.arccos(t/np.sqrt(t*t+1)) + np.arctan(s/r)
        beta = -np.arccos(t/np.sqrt(t*t+1)) + np.arctan(s/r)
        a1 = np.sqrt(t*t+1)*np.cos(alpha)
        a2 = np.sqrt(t*t+1)*np.sin(alpha)
        b1 = np.sqrt(t*t+1)*np.cos(beta)
        b2 = np.sqrt(t*t+1)*np.sin(beta)
        a3 = -(a1*r + a2*s)/t
        matO = 0.5*np.array([[a1,b1,a1,b1,2*r], [a2,b2,a2,b2,2*s], [a3,a3,a3,a3,2*t], [1,-1,-1,1,0], [1,1,-1,-1,0]])
        code,info = get_SO5_code(matO, return_info=True)
        info['su2'] = np.stack([1j*X, 1j*Z, -Z, I, I, I], axis=0)
    else:
        tmp0 = np.concat([vece, np.zeros(5-vece.shape[0])])
        code,info = get_SO5_code(tmp0, return_info=True) #no transversal gate
    return code, info

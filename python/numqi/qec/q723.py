import itertools
import numpy as np
import scipy.linalg

import numqi.gate

from ._pauli import hf_pauli, make_pauli_error_list_sparse

def get_cyclic_code(lambda2:float, sign:str='++', return_info=False):
    assert (lambda2>=0) and (lambda2<=7)
    assert sign in {'++', '+-', '-+', '--'} #sign of c1 and c3
    x = np.sqrt(lambda2).item()
    s7 = np.sqrt(7)
    assert np.all(0<=x) and np.all(x<=s7)
    c0 = np.sqrt(s7*x+8)/8
    c1 = (7**(1/4))*np.sqrt(x)/8 * (1 if sign[0]=='+' else -1)
    tmp0 = 0.4*np.sqrt(max(7*c0*c0-15*s7*x/64,0)) * (1 if sign[1]=='+' else -1)
    c3 = 0.4*s7*c0 + tmp0
    c2 = -2*c3 + s7*c0
    c4 = -np.sqrt(3)*c1
    coeff = np.array([c0,c1,c2,c3,c4])
    basis = get_723_cyclic_code_basis()
    code = coeff @ basis
    if return_info:
        qweA = np.array([1, 0, lambda2, 0, 21-2*lambda2, 0, 42+lambda2, 0])
        qweB = np.array([1, 0, lambda2, 21+3*lambda2, 21-2*lambda2, 126-6*lambda2, 42+lambda2, 45+3*lambda2])
        lambda_ai = dict()
        for ind0,ind1 in itertools.combinations(range(7), 2):
            tmp0 = ['I']*7
            tmp0[ind0] = 'X'
            tmp0[ind1] = 'X'
            lambda_ai[''.join(tmp0)] = x/(3*np.sqrt(7))
            tmp0[ind0] = 'Y'
            tmp0[ind1] = 'Y'
            lambda_ai[''.join(tmp0)] = x/(3*np.sqrt(7))
            tmp0[ind0] = 'Z'
            tmp0[ind1] = 'Z'
            lambda_ai[''.join(tmp0)] = x/(3*np.sqrt(7))
        error_str_list = make_pauli_error_list_sparse(num_qubit=7, distance=3, kind='scipy-csr01')[0]
        lambda_ai = np.array([lambda_ai.get(x,0) for x in error_str_list], dtype=np.float64)
        info = dict(logicalX='X'*7, logicalZ='Z'*7, lambda2=lambda2, sign=sign,
                    basis=basis, coeff=coeff, qweA=qweA, qweB=qweB, lambda_ai=lambda_ai)
    return code,info


def get_723_cyclic_code_basis():
    # equivalent construction but not efficient
    # tmp0 = '''0000000
    #     0000011 0000110 0001100 0011000 0110000 1100000 1000001
    #     0000101 0001010 0010100 0101000 1010000 0100001 1000010
    #     0001001 0010001 0010010 0100010 0100100 1000100 1001000
    #     0001111 0011110 0111100 1111000 1110001 1100011 1000111
    #     0011011 0110110 1101100 1011001 0110011 1100110 1001101
    #     0011101 0111010 1110100 1101001 1010011 0100111 1001110
    #     0101011 1010110 0101101 1011010 0110101 1101010 1010101
    #     0010111 0101110 1011100 0111001 1110010 1100101 1001011
    #     1111110 1111101 1111011 1110111 1101111 1011111 0111111'''
    # tmp1 = [numqi.qec._internal.hf_state(x.strip()) for x in tmp0.strip().split('\n')]
    # basis = np.stack([tmp1[0], (tmp1[1]+tmp1[2]+tmp1[3])/np.sqrt(3), tmp1[8],
    #                     (tmp1[4]+tmp1[5]+tmp1[6]+tmp1[7])/2, tmp1[9]], axis=0)
    # Xseven = hf_pauli('X'*7)
    # basis_not = basis @ Xseven
    ret = np.zeros((2,5,128), dtype=np.float64)
    ret[0,0,0] = 1
    ret[0,1,[3,5,6,9,10,12,17,18,20,24,33,34,36,40,48,65,66,68,72,80,96]] = 1/np.sqrt(21)
    ret[0,2,[23,46,57,75,92,101,114]] = 1/np.sqrt(7)
    ret[0,3,[15,27,29,30,39,43,45,51,53,54,58,60,71,77,78,83,85,86,89,90,99,102,105,106,108,113,116,120]] = 1/np.sqrt(28)
    ret[0,4,[63,95,111,119,123,125,126]] = 1/np.sqrt(7)
    ret[1,0,127] = 1
    ret[1,1,[31,47,55,59,61,62,79,87,91,93,94,103,107,109,110,115,117,118,121,122,124]] = 1/np.sqrt(21)
    ret[1,2,[13,26,35,52,70,81,104]] = 1/np.sqrt(7)
    ret[1,3,[7,11,14,19,21,22,25,28,37,38,41,42,44,49,50,56,67,69,73,74,76,82,84,88,97,98,100,112]] = 1/np.sqrt(28)
    ret[1,4,[1,2,4,8,16,32,64]] = 1/np.sqrt(7)
    return ret


def get_2I_lambda0(theta:float, phi:float, sign:str, return_info:bool=False):
    assert sign in '+-'
    s = np.sqrt
    basis0 = np.zeros((4,128), dtype=np.float64)
    basis0[0, [0,63,95,111,113,114,116,120]] = np.array([1,1,-1,1,-1,1,1,1])/s(8)
    basis0[1, [19,28,37,42,70,73]] = np.array([-1,1,1,-1,1,-1])/s(6)
    basis0[2, [21,26,38,41,67,76]] = np.array([1,-1,1,-1,1,-1])/s(6)
    basis0[3, [22,25,35,44,69,74]] = np.array([-1,1,-1,1,1,-1])/s(6)
    basis1 = (hf_pauli('X'*7) @ basis0.T).T.copy()
    ct = np.cos(theta)
    st = np.sin(theta)
    s3 = s(1/3)
    a = -np.cos(phi)*st*s3
    b = (np.cos(phi-np.pi/3) + np.cos(phi+np.pi/3)) * ct*s3 + s(1/30)
    c = -np.cos(phi-np.pi/3) * st*s3
    d = np.cos(phi-np.pi/3) * ct*s3 - s(1/30)
    e = (np.cos(phi-np.pi/3) - np.cos(phi)) * st*s3
    f = np.cos(phi+np.pi/3) * ct*s3 - s(1/30)
    coeff = np.array([s(2/5)*(1 if sign=='+' else -1), a+1j*b, c+1j*d, e+1j*f])
    code = np.stack([coeff@basis0, coeff@basis1], axis=0)
    ret = code
    if return_info:
        qweA = np.array([1,0,0,0,21,0,42,0]) + 8*st*st*np.array([0,0,0,0,-1,3,-3,1])
        qweB = np.array([1,0,0,21,21,126,42,45]) + 8*st*st*np.array([0,0,0,-1,4,-6,4,-1])
        su2 = numqi.gate.rz(np.array([2,2,2,4,4,4,4])*np.pi/5) #rz(2*np.pi/5)
        I,X,Y,Z = numqi.gate.I, numqi.gate.X, numqi.gate.Y, numqi.gate.Z
        hfR = lambda a,b,t=1: I*np.cos(t*np.pi/5) + 1j*np.sin(t*np.pi/5)/np.sqrt(5) * (a*Y + b*Z)
        if sign=='+':
            transR = hfR(-2,1)
            tmp0 = [(2, 1, 1), (-2, 1, 1), (2, 1, 1), (2, -1, 2), (2, -1, 2), (2, -1, 2), (-2, -1, 2)]
        else:
            transR = hfR(2,1)
            tmp0 = [(-2, 1, 1), (2, 1, 1), (-2, 1, 1), (-2, -1, 2), (-2, -1, 2), (-2, -1, 2), (2, -1, 2)]
        su2R = [hfR(a,b,t) for a,b,t in tmp0]
        info = dict(basis0=basis0, basis1=basis1, theta=theta, phi=phi, sign=sign, su2=su2, su2R=su2R, transR=transR, qweA=qweA, qweB=qweB)
        ret = code, info
    return ret


def get_2I_lambda075(t:float, sign=None, return_info:bool=False):
    assert abs(t) <= np.sqrt(5/16)
    if sign is None:
        sign = np.ones(3, dtype=np.float64)
    else:
        sign = tuple(int(x) for x in sign)
        assert (len(sign) == 3) and all(x in [1,-1] for x in sign)
        sign = np.asarray(sign)
    basis0 = np.zeros((4,128), dtype=np.float64)
    basis0[0, 0] = 1
    basis0[1, [7, 11, 19, 35, 67, 124]] = np.array([1,-1,1,1,1,-1])/np.sqrt(6)
    basis0[2, [29, 45, 54, 58, 78, 85, 90, 102, 105, 113]] = np.array([1,1,-1,1,1,-1,1,-1,1,-1])/np.sqrt(10)
    basis0[3, [30, 46, 53, 57, 77, 86, 89, 101, 106, 114]] = np.array([1,1,-1,1,1,-1,1,-1,1,-1])/np.sqrt(10)
    basis1 = (hf_pauli('X'*7) @ basis0.T).T
    b = sign[2]*np.sqrt(max(0,5/16-t*t))
    coeff = np.array([np.sqrt(1/10), 1j*sign[0]*np.sqrt(3/20), sign[1]/4-t-1j*b, sign[1]/4+t+1j*b])
    code = np.stack([coeff@basis0, coeff@basis1], axis=0)
    ret = code
    if return_info:
        su2 = numqi.gate.rz(np.array([1,1,1,1,1,2,2])*2*np.pi/5)
        t2 = t*t
        qweA = np.array([1,0,3/4,0,12+24*t2,45/2-72*t2,81/4+72*t2,15/2-24*t2])
        qweB = np.array([1,0,3/4,63/4+24*t2,99/2-96*t2,153/2+144*t2,291/4-96*t2,159/4+24*t2])
        lambda_ai = np.zeros(210, dtype=np.float64)
        lambda_ai[[201, 205, 209]] = -0.5 #IIIIIXX IIIIIYY IIIIIZZ
        I,X,Y,Z = numqi.gate.I, numqi.gate.X, numqi.gate.Y, numqi.gate.Z
        hfR = lambda a,b,t=1: I*np.cos(t*np.pi/5) + 1j*np.sin(t*np.pi/5)/np.sqrt(5) * (a*Y + b*Z)
        tmp0 = int(sign[0]),int(sign[1])
        if tmp0==(-1,-1):
            transR = hfR(2,1)
            tmp1 = [(-2, -1, 1), (-2, -1, 1), (-2, -1, 1), (2, -1, 1), (-2, -1, 1), (-2, 1, 2), (-2, 1, 2)]
        elif tmp0==(1, -1):
            transR = hfR(-2,1)
            tmp1 = [(2, -1, 1), (2, -1, 1), (2, -1, 1), (-2, -1, 1), (2, -1, 1), (2, 1, 2), (2, 1, 2)]
        elif tmp0==(-1, 1):
            transR = hfR(2,1)
            tmp1 = [(-2, -1, 1), (-2, -1, 1), (-2, -1, 1), (2, -1, 1), (-2, -1, 1), (2, 1, 2), (2, 1, 2)]
        elif tmp0==(1, 1):
            transR = hfR(-2,1)
            tmp1 = [(2, -1, 1), (2, -1, 1), (2, -1, 1), (-2, -1, 1), (2, -1, 1), (-2, 1, 2), (-2, 1, 2)]
        su2R = [hfR(a,b,t) for a,b,t in tmp1]
        info = dict(basis0=basis0, basis1=basis1, coeff=coeff, t=t, sign=sign, su2=su2,
                qweA=qweA, qweB=qweB, lambda_ai=lambda_ai, transR=transR, su2R=su2R)
        ret = code, info
    return ret


def get_BD12_veca1112222(sign=None, return_info=False):
    if sign is None:
        sign = np.array([1,1,1,1,1,1,-1,1,-1,1])
    else:
        sign = tuple(int(x) for x in sign)
        assert (len(sign)==10) and all(x in [-1,1] for x in sign)
    assert (sign[2]*sign[4]*sign[7]*sign[8])==-1
    assert (sign[1]*sign[3]*sign[5]*sign[6])==-1
    basis0 = np.zeros((10, 128), dtype=np.complex128)
    basis0[np.arange(9), [0,53,54,58,60,85,90,99,105]] = 1
    basis0[9, 14] = 1j
    basis1 = (hf_pauli('X'*7) @ basis0.T).T
    coeff = np.sqrt(np.array([10,12,5,8,5,18,12,15,15,20])/120) * sign
    code = np.stack([coeff@basis0, coeff@basis1], axis=0)
    ret = code
    if return_info:
        su2 = numqi.gate.rz(np.array([1,1,1,2,2,2,2])*2*np.pi/6)
        A2 = 269/150
        qweA = np.array([1,0,A2,0,21-2*A2,0,42+A2,0])
        qweB = np.array([1,0,A2,21+3*A2,21-2*A2,126-6*A2,42+A2,45+3*A2])
        info = dict(sign=sign, basis0=basis0, basis1=basis1, coeff=coeff, su2=su2, qweA=qweA, qweB=qweB)
        ret = code, info
    return ret


def get_BD12_veca0122233(a:float, sign=None, return_info=False):
    ## TODO
    # [0,3,4,5,6,1,2]
    # [0,2,2,2,4,6,6]
    # [2,2,2,2,4,4,6]
    # [0,2,2,4,4,4,6]
    # [1,1,2,3,4,5,6]
    # [2,2,2,4,4,4,4]
    s = np.sqrt
    assert 0 <= a <= s(2/3)
    if sign is None:
        sign = np.array([1,1,1,1], dtype=np.int64)
    else:
        sign = tuple(int(x) for x in sign)
        assert len(sign) == 4
        assert all(x in [1,-1] for x in sign)
        sign = np.asarray(sign, dtype=np.int64)
    assert (sign[0]*sign[1]*sign[2]*sign[3])==1
    basis0 = np.zeros((4,128), dtype=np.complex128)
    basis0[0, [0,101,105,113, 3,102,106,114]] = np.array([1,1,1,1,-1,-1,-1,-1])/s(8)
    basis0[1, [37,38,49,41,42,50,64,67]] = 1j/np.sqrt(8)
    basis0[2, [28,31]] = 1j/np.sqrt(2)
    basis0[3, [92,95]] = np.array([1,-1]) / np.sqrt(2)
    basis1 = (hf_pauli('X'*7) @ basis0.T).T
    tmp0 = s(max(0,2/3-a*a))
    coeff = np.array([a,tmp0,s(1/2)*a,s(1/2)*tmp0])*sign
    code = np.stack([coeff@basis0,coeff@basis1], axis=0)
    ret = code
    if return_info:
        su2 = numqi.gate.rz(np.array([0,1,2,2,2,3,3])*2*np.pi/6)
        lambda_ai = np.zeros(210, dtype=np.float64)
        lambda_ai[[83,92,101,120,124,128,129,133,137,156,160,164]] = np.array([-1,-1,-1,1,1,1,1,1,1,1,1,1])/3
        lambda_ai[[38,47,56,201]] = 1/3 - a*a
        lambda_ai[29] = 3*a*a - 1
        lambda_ai[62] = -2*a*s(max(0,2/3-a*a)) * sign[0]*sign[1]
        lambda_ai[71] = a*s(max(0,2/3-a*a)) * sign[0] * sign[1]
        lambda_ai[205] = -2*a*a + 2/3
        A2 = 29/9 - 8*a*a + 12*a**4
        # A2 range [17/9, 29/9]
        qweA = np.array([1,0,A2,0,21-2*A2,0,42+A2,0])
        qweB = np.array([1,0,A2,21+3*A2,21-2*A2,126-6*A2,42+A2,45+3*A2])
        info = dict(a=a, sign=sign, coeff=coeff, basis0=basis0, basis1=basis1, su2=su2, lambda_ai=lambda_ai, qweA=qweA, qweB=qweB)
        ret = (code, info)
    return ret


def get_BD14(a:float, sign=None, return_info=False):
    assert 0 <= a <= np.sqrt(1/14)
    if sign is None:
        sign = np.ones(5, dtype=np.float64)
    else:
        sign = tuple(int(x) for x in sign)
        assert len(sign) == 5
        assert all(x in [1,-1] for x in sign)
        sign = np.asarray(sign)
    s = np.sqrt
    basis0 = np.zeros((5,128), dtype=np.complex128)
    basis0[0, [0,14,86,89]] = np.array([1,s(2)*1j,1,s(2)])/s(6)
    basis0[[1,2], [19,21]] = 1j
    basis0[3, [60,101]] = np.array([s(3),2])/s(7)
    basis0[4, [58,99]] = np.array([s(3),-2])/s(7)
    basis1 = (hf_pauli('X'*7) @ basis0.T).T
    coeff = np.array([s(3/7), a, s(max(0,1/14-a*a)), s(3/14+a*a), s(2/7-a*a)])*sign
    code = np.stack([coeff@basis0, coeff@basis1], axis=0)
    ret = code
    if return_info:
        # TODO BD14 [2,2,4,4,4,4,6]
        su2 = numqi.gate.rz(np.array([1,2,3,4,5,5,6])*np.pi/7)
        lambda_ai = np.zeros(210, dtype=np.float64)
        lambda_ai[[29,38,83,92,119,155,191]] = np.array([1,-1,-1,-1,1,-1,-1])/7
        lambda_ai[[47,128,182]] = np.array([-1,1,-1])*3/7
        lambda_ai[74] = 5/7
        lambda_ai[[101,110]] = np.array([-1,1])*(-4*a*a + 1/7)
        lambda_ai[[56,146]] = 16/7*a*a - 11/49
        lambda_ai[[65,137]] = -16/7*a*a - 3/49
        lambda_ai[209] = 12/7*a*a - 17/49
        lambda_ai[[164,173]] = np.array([-1,1])*(-12/7*a*a + 3/49)
        lambda_ai[200] = -12/7*a*a - 11/49
        lambda_ai[[183,187]] = sign[1]*sign[2]*2*a*s(max(0,1/14-a*a)) - sign[3]*sign[4]*2/7*s(3/14+a*a)*s(2/7-a*a)
        # 2*coeff[1]*coeff[2] - 2/7*coeff[3]*coeff[4]
        A2 = 3701/2401 -1384/343*a*a + 2768/49*a**4 - np.prod(sign[1:])*16/7*a*s(max(0,1/14-a*a))*s(3/14+a*a)*s(2/7-a*a)
        # A2 range [1.44? 1.54?]
        qweA = np.array([1,0,A2,0,21-2*A2,0,42+A2,0])
        qweB = np.array([1,0,A2,21+3*A2,21-2*A2,126-6*A2,42+A2,45+3*A2])
        info = dict(a=a, sign=sign, coeff=coeff, basis0=basis0, basis1=basis1, su2=su2, lambda_ai=lambda_ai, qweA=qweA, qweB=qweB)
        ret = (code, info)
    return ret


def get_BD16_5theta(theta, return_info=False):
    theta = np.asarray(theta)
    assert len(theta)==5
    ind0 = [0,3,61,62,93,94,109,110,117,118,121,122]
    # [127, 124, 66, 65, 34, 33, 18, 17, 10, 9, 6, 5]
    basis0 = np.zeros((len(ind0), 128), dtype=np.complex128)
    basis0[np.arange(len(ind0)), ind0] = 1
    basis1 = (hf_pauli('XXXXXII') @ basis0.T).T
    phase = np.exp(1j*np.array([np.pi/2, np.pi/2] + [y for x in theta for y in [x,-x]]))
    coeff = np.array([np.sqrt(3)/4]*2 + [1/4]*10) * phase
    code = np.stack([coeff@basis0,coeff@basis1], axis=0)
    ret = code
    if return_info:
        su2X = [numqi.gate.X]*5 + [numqi.gate.I]*2
        su2Z = numqi.gate.rz(np.array([3,3,3,3,3,4,-4])*np.pi/4)
        A2 = np.cos(2*(theta.reshape(-1,1)-theta)).sum()/16 + 53/16
        # A2 range [53/16, 78/16]
        qweA = np.array([1,0,A2,0,21-2*A2,0,42+A2,0])
        qweB = np.array([1,0,A2,21+3*A2,21-2*A2,126-6*A2,42+A2,45+3*A2])
        info = dict(theta=theta, coeff=coeff, basis0=basis0, basis1=basis1, su2X=su2X, su2Z=su2Z, qweA=qweA, qweB=qweB)
        ret = code, info
    return ret


def get_BD16_veca1222233(theta0:float, theta1:float, sign=None, return_info:bool=False):
    if sign is None:
        sign = np.array([1,1,1,1,1,1,1], dtype=np.int64)
    else:
        sign = tuple(int(x) for x in sign)
        assert len(sign) == 7
        assert all(x in [1,-1] for x in sign)
        sign = np.asarray(sign, dtype=np.int64)
    s = np.sqrt
    basis0 = np.zeros((7, 128), dtype=np.complex128)
    basis0[np.arange(7), [0,60,78,85,89,114,35]] = np.array([1,1,1,1,1,1,1j])
    basis1 = (hf_pauli('X'*7) @ basis0.T).T
    coeff = np.array([np.exp(1j*theta0), s(3), s(3), s(2), s(2), np.exp(1j*theta1), 2])/4 * sign
    code = np.stack([coeff@basis0,coeff@basis1], axis=0)
    ret = code
    if return_info:
        su2 = numqi.gate.rz(np.array([1,2,2,2,2,3,3])*2*np.pi/8)
        a = np.cos(2*theta0) + np.cos(2*theta1)
        b = np.cos(2*theta0-2*theta1)
        qweA = np.array([1, 0, 21/8, 1/8*(2-a), 1/16*(215+18*a+b), 3/16*(29-14*a-b),
                        1/16*(635+38*a+3*b), 1/16*(25-12*a-b)])
        qweB = np.array([1, 0, 21/8, 1/16*(441+10*a+b), 1/16*(352-48*a-4*b), 1/16*(1590+84*a+6*b),
                        1/16*(846-64*a-4*b), 1/16*(809+18*a+b)])
        info = dict(basis0=basis0, basis1=basis1, coeff=coeff, su2=su2,
                    theta0=theta0, theta1=theta1, sign=sign, qweA=qweA, qweB=qweB)
        ret = code, info
    return ret


def get_BD16_degenerate(theta:float|None=None, sign=None, return_info=True, seed=None):
    np_rng = np.random.default_rng(seed)
    if theta is None:
        theta = np_rng.uniform(0, 2*np.pi)
    if sign is None:
        sign = np_rng.integers(2, size=8)*2-1
        sign[5] = -sign[0]*sign[1]*sign[4]
    assert (sign.shape==(8,)) and all((x==1 or x==-1) for x in sign)
    assert sign[0]*sign[1]*sign[4]*sign[5]==-1
    basis0 = np.zeros((8, 128), dtype=np.float64)
    basis0[np.arange(8), [0,36,92,113,11,47,114,87]] = 1
    basis1 = (hf_pauli('X'*7) @ basis0.T).T
    s = np.sqrt
    tmp0 = np.array([s(3/32), s(3/32), s(3/16), s(1/8), s(5/32), s(5/32), s(1/8), 1/4])
    tmp1 = np.exp(1j*theta)
    coeff = np.array([1j, 1j, tmp1, tmp1, 1j*tmp1, 1j*tmp1, tmp1, 1]) * tmp0 * sign
    code = np.stack([coeff@basis0, coeff@basis1], axis=0)
    ret = code
    if return_info:
        lambda_a = np.zeros(210, dtype=np.float64)
        lambda_a[38] = 1 #ZIZIIII
        lambda_a[[47,65,74,128,146,155]] = -1/4 #ZIIZIII ZIIIIZI ZIIIIIZ IIZZIII IIZIIZI IIZIIIZ
        lambda_a[92] = -3/8 #IZIZIII
        lambda_a[[110,119]] = 1/8 #IZIIIZI IZIIIIZ
        lambda_a[164] = 3/8 #IIIZZII
        lambda_a[[173,182]] = 1/4 #IIIZIZI IIIZIIZ
        lambda_a[[191,200]] = -1/8 #IIIIZZI IIIIZIZ
        lambda_a[209] = 1/2 #IIIIIZZ
        lambda_a[93] = -1/8*sign[0]*sign[1] #IXIIXII
        lambda_a[97] = 1/8*sign[0]*sign[1] #IYIIYII
        lambda_a[[201,205]] = 1/4*sign[3]*sign[6] #IIIIIXX IIIIIYY
        tmp0 = np.sort(np.roots([32, -128, 133, -30]).real) # -30 + 133x - 128xx + 32xxx=0
        lambda_ab_EVL = np.array([0, tmp0[0], 1/2, 3/4,3/4, 7/8,7/8, 1,1,1,1,1,1,1,1, 9/8,9/8, tmp0[1], 5/4,5/4, 3/2, tmp0[2]])
        c2t = np.cos(2*theta)
        qweA = np.array([1,0,9/4,7/32,209/16,9,571/16,89/32]) + c2t*np.array([0,0,0,7/32,-55/16,9,-137/16,89/32])
        qweB = np.array([1,0,9/4,403/16,221/8,189/2,457/8,773/16]) + c2t*np.array([0,0,0,-41/16,89/8,-18,103/8,-55/16])
        su2 = numqi.gate.rz(np.array([3,3,4,4,5,6,6])*2*np.pi/8)
        ret = code, dict(theta=theta, sign=sign, coeff=coeff, basis0=basis0, basis1=basis1, su2=su2,
                    qweA=qweA, qweB=qweB, lambda_a=lambda_a, lambda_ab_EVL=lambda_ab_EVL)
    return ret


def get_BD18(theta:float, root:int=0, sign=None, return_info=True):
    assert root in {0,1}
    # import sympy, sympy.abc
    # x = sympy.abc.x
    # [x.n(24) for x in sympy.real_roots(-25+3222*x*x-49329*x**4-58320*x**6+209952*x**8)][3:-1]
    # np.polynomial.Polynomial([-25,0,3222,0,-49329,0,-58320,0,209952]).roots()[5:7]
    b = [0.0949564154092439210937958, 0.230377064728512746573312][root]
    if sign is None:
        sign = np.ones(8, dtype=np.int64)
    else:
        sign = np.array([int(x) for x in sign], dtype=np.int64)
        assert np.all((sign*sign)==1)
    # sign[[1,2,3,6]] can be set {0,1} arbitrarily
    sign[0] = 1 #sign[a] is controlled by theta
    sign[4] = -sign[1]*sign[2]*sign[3] #sign[e]=-sign[bcd]
    sign[5] = -sign[3] #sign[d]==-sign[f]
    sign[7] = sign[1]*sign[3]*sign[6]
    s = np.sqrt
    d = 1/s(36*b*b+14).item()
    coeff = np.array([1/s(18)*np.exp(1j*theta), b, s(max(0, 1/18-b*b)),
            d, s(1/9-b*b), s(b*b+d*d+1/18), s(1/3-2*d*d), s(7/18)])*sign
    basis0 = np.zeros((8,128), dtype=np.complex128)
    basis0[[0,1,2,3,4,5], [0,58,60,90,99,101]] = 1
    basis0[6, [86,105]] = 1/s(2)
    basis0[7, [54,14,25]] = np.array([s(2),-s(2)*1j,s(3)*1j])/s(7)
    basis1 = (hf_pauli('X'*7) @ basis0.T).T
    code = np.stack([coeff@basis0, coeff@basis1], axis=0)
    ret = code
    if return_info:
        su2 = numqi.gate.rz(np.array([2,2,4,6,6,6,8])*np.pi/9)
        lambda_ai = np.zeros(210, dtype=np.float64)
        lambda_ai[110] = -1/9
        lambda_ai[[ 29,  38,  47,  74,  83, 119, 155]] = np.array([1,-1,-1,1,-1,1,-1])/3
        lambda_ai[[56,65]] = (4*b*b - 1/9)*np.array([1,-1])
        lambda_ai[146] = 4*b*b + 1/9
        lambda_ai[164] = -4*b*b - 1/3
        lambda_ai[209] = -4*b*b - 5/9
        lambda_ai[92] = -4*d*d -1/9
        lambda_ai[182] = -4*d*d + 1/3
        lambda_ai[191] = -4*d*d + 5/9
        lambda_ai[[101, 128]] = 4*d*d - 1/9
        lambda_ai[137] = - 4*b*b - 4*d*d + 1/3
        lambda_ai[173] = 4*b*b + 4*d*d - 5/9
        lambda_ai[200] = 4*b*b + 4*d*d - 7/9
        lambda_ai[[21,25]] = sign[1]*sign[3]*2*(b*d + s(1/18)*s(1/3-2*d*d)) #2*(b*d + s(1/7)*g*h)
        lambda_ai[[165,169]] = sign[2]*sign[7]*(2/3*s(1/18-b*b) - s(2)*s((1/9-b*b)*(1/3-2*d*d))) #s(8/7)*c*h - s(2)*e*g)
        lambda_ai[[183,187]] = sign[1]*sign[2]*2*(b*s(1/18-b*b) + s((1/9-b*b)*(b*b+d*d+1/18))) #2*b*c + 2*e*f
        lambda_ai[[156,160]] = sign[1]*sign[7] * (2/3*b + s(2)*s(1/3-2*d*d)*(d-s(b*b+d*d+1/18))) # s(8/7)*b*h + s(2)*g*(d-f)
        tmp0 = s((1 - 27*b*b + 162*b**4)*(1-6*d*d))
        tmp2 = -1 + 9*b*b + 27*d*d - 54*d**4
        A2 = (-257/6 + 112*b*b + 1802*d*d/3 + 462*d**4 - 8/(27*s(3))*tmp0 + 4/9*d*(-1 + 6*d*d)* s(2+36*b*b+36*d*d)
            + 8/27*b*(6*d*s(6-36*d*d) + s((1 - 27*b*b + 162*b**4)*(1 + 18*b*b + 18*d*d)) - s(6)*s(tmp2)))
        # A2 be either 2.548? or 2.159?
        st2 = np.sin(theta)**2
        tmp0 = np.array([1,0,A2,0,21-2*A2,0,42+A2,0])
        tmp1 = np.array([0,0,0,20,-176,408,-368,116])/81
        qweA = tmp0 + st2*tmp1
        tmp0 = np.array([1,0,A2,21+3*A2,21-2*A2,126-6*A2,42+A2,45+3*A2])
        tmp1 = np.array([0, 0, 0, -96, 464, -816, 624, -176])/81
        qweB = tmp0 + st2*tmp1
        info = dict(b=b, d=d, root=root, sign=sign, su2=su2, theta=theta, coeff=coeff, basis0=basis0, basis1=basis1, lambda_ai=lambda_ai, qweA=qweA, qweB=qweB)
        ret = code, info
    return ret


def get_BD18_LP(coeff2=None, sign=None, return_info=True, seed=None):
    matA = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                     [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                     [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]], dtype=np.int64)
    vecb = np.array([1/6, 1/6, 1/6, 1/6, 2/9, 2/9])
    if sign is None:
        sign = np.array([1]*14, dtype=np.int64)
    else:
        sign = tuple(int(x) for x in sign)
        assert len(sign) == 14
        assert all(x in [1,-1] for x in sign)
        sign = np.asarray(sign, dtype=np.int64)
    if coeff2 is None:
        np_rng = np.random.default_rng(seed)
        matB = scipy.linalg.null_space(matA) #(12,6)
        tmp0 = np_rng.normal(size=6)
        tmp1 = matB @ (tmp0/np.linalg.norm(tmp0))
        tmp2 = np_rng.uniform(-1/18/tmp1.max(),-1/18/tmp1.min())
        tmp3 = 1/18 + tmp2*tmp1
        coeff = np.sqrt(np.array([1/18,5/18] + tmp3.tolist())) * sign
    else:
        assert (coeff2.shape==(14,)) and (coeff2.min()>=0)
        coeff2 = np.asarray(coeff2).copy()
        assert abs(coeff2[0] - 1/18) < 1e-10
        assert abs(coeff2[1] - 5/18) < 1e-10
        assert np.abs(matA @ coeff2[2:] - vecb).max() < 1e-10
        coeff2[0] = 1/8
        coeff2[1] = 5/8
        coeff = np.sqrt(coeff2) * sign
    basis0 = np.zeros((14, 128), dtype=np.complex128)
    basis0[[0,1], [0,7]] = 1
    basis0[np.arange(2,14), [57,58,60,89,90,92,105,106,108,113,114,116]] = 1j
    basis1 = (hf_pauli('X'*7) @ basis0.T).T
    code = np.stack([coeff@basis0,coeff@basis1], axis=0)
    ret = code
    if return_info:
        su2 = numqi.gate.rz(np.array([4,4,4,4,6,6,6])*np.pi/9)
        info = dict(su2=su2, coeff=coeff, basis0=basis0, basis1=basis1)
        ret = code, info
    return ret


def get_BD20(a:float, sign=None, return_info=False):
    assert 0 <= a <= np.sqrt(1/5)
    if sign is None:
        sign = np.ones(5, dtype=np.float64)
    else:
        sign = tuple(int(x) for x in sign)
        assert len(sign) == 5
        assert all(x in [1,-1] for x in sign)
        sign = np.asarray(sign)
    s = np.sqrt
    si = lambda x: np.sqrt(x)*1j
    basis0 = np.zeros((5,128), dtype=np.complex128)
    basis0[0, [0,13,60,102]] = np.array([1,s(2)*1j,s(3),1])/s(7)
    basis0[[1,2], [19,35]] = 1
    basis0[3, [85,90]] = np.array([2,s(5)])/3
    basis0[4, [101,106]] = np.array([2,-s(5)])/3
    basis1 = (hf_pauli('X'*7) @ basis0.T).T
    coeff = np.array([s(7/20), a*1j, si(max(0,1/5-a*a)), s(7/20-a*a), s(1/10+a*a)]) * sign
    code = np.stack([coeff@basis0, coeff@basis1], axis=0)
    ret = code
    if return_info:
        su2 = numqi.gate.rz(np.array([2,4,4,6,6,8,8])*np.pi/10)
        lambda_ai = np.zeros(210, dtype=np.float64)
        lambda_ai[[65,74,200,209]] = np.array([1,-1,1,-1])/5
        lambda_ai[[83,182,191]] = np.array([-2,-3,-4])/5
        lambda_ai[[110,137,146]] = np.array([1,1,-1]) * (2/9 - 16/9*a*a)
        lambda_ai[101] = -2/90+16/9*a*a
        lambda_ai[119] = -1/45 - 20/9*a*a
        lambda_ai[[128,155]] = np.array([1,-1]) * (17/45 - 20/9*a*a)
        lambda_ai[92] = -8/45 + 20/9*a*a
        lambda_ai[[29,38]] = np.array([1,-1])*(-0.4 + 4*a*a)
        lambda_ai[[75,79]] = 2*(-1/9*coeff[3]*coeff[4] - coeff[1]*coeff[2]).real
        lambda_ai[[156,160]] = -2*np.sqrt(5/63)*(coeff[0]*coeff[4]).real
        lambda_ai[[201,205]] = 2*np.sqrt(4/63)*(coeff[0]*coeff[4]).real
        A2 = 1451/675 - 1468/135*a*a + 1520/27*a**4 - 16/9*np.prod(sign[1:])*a*np.sqrt((1/5-a*a)*(7/20-a*a)*(1/10+a*a))
        # A2 range [1.58?, 2.22?]
        qweA = np.array([1,0,A2,0,21-2*A2,0,42+A2,0])
        qweB = np.array([1,0,A2,21+3*A2,21-2*A2,126-6*A2,42+A2,45+3*A2])
        ret = code, dict(su2=su2, coeff=coeff, basis0=basis0, basis1=basis1, a=a, sign=sign, lambda_ai=lambda_ai, qweA=qweA, qweB=qweB)
    return ret


def get_BD22(a:float, sign=None, return_info=False):
    assert 0 <= a <= np.sqrt(3/22)
    if sign is None:
        sign = np.ones(5, dtype=np.float64)
    else:
        sign = tuple(int(x) for x in sign)
        assert len(sign) == 5
        assert all(x in [1,-1] for x in sign)
        sign = np.asarray(sign)
    s = np.sqrt
    si = lambda x: np.sqrt(x)*1j
    basis0 = np.zeros((5, 128), dtype=np.complex128)
    basis0[0, [0,90,60,35]] = np.array([1,s(2),s(3),2j])/s(10)
    basis0[[1,2], [13,21]] = 1
    basis0[3, [78,105]] = np.array([s(5),2])/3
    basis0[4, [86,113]] = np.array([s(5),-2])/3
    basis1 = (hf_pauli('X'*7) @ basis0.T).T
    coeff = np.array([s(5/11), a*1j, si(3/22-a*a), s(3/11-a*a), s(3/22+a*a)]) * sign
    code = np.stack([coeff@basis0, coeff@basis1], axis=0)
    ret = code
    if return_info:
        su2 = numqi.gate.rz(np.array([2,4,6,6,6,8,10])*np.pi/11)
        lambda_ai = np.zeros(210, dtype=np.float64)
        lambda_ai[[29,65,74,110,209]] = np.array([-1,1,-1,-1,-1])*3/11
        lambda_ai[[101,119,200]] = np.array([-1,1,-1])*5/11
        lambda_ai[[56,128,191]] = -1/11
        lambda_ai[[120,124]] = 2*(1/9*coeff[3]*coeff[4] - coeff[1]*coeff[2]).real
        lambda_ai[[156,160]] = 2/3*(coeff[0]*coeff[4]).real
        lambda_ai[[129,133]] = 2/3*(coeff[0]*coeff[3]).real
        lambda_ai[155] = -7/33 - 20/9*a*a
        lambda_ai[173] = -1/33 - 20/9*a*a
        lambda_ai[182] = -17/33 + 20/9*a*a
        lambda_ai[146] = -1/3 + 20/9*a*a
        lambda_ai[83] = -7/33 + 16/9*a*a
        lambda_ai[164] = 5/33 + 16/9*a*a
        lambda_ai[92] = 1/33 - 16/9*a*a
        lambda_ai[137] = 13/33 - 16/9*a*a
        lambda_ai[38] = -1/11 + 4*a*a
        lambda_ai[47] = 5/11 - 4*a*a
        A2 = 743/363 - 760/99*a*a + 1520/27*a**4 + 16*np.prod(sign[1:])/9*a*np.sqrt((3/11-a*a)*(3/22+a*a)*(3/22-a*a))
        # A2 range [1.76? 2.047?]
        qweA = np.array([1,0,A2,0,21-2*A2,0,42+A2,0])
        qweB = np.array([1,0,A2,21+3*A2,21-2*A2,126-6*A2,42+A2,45+3*A2])
        ret = code, dict(su2=su2, coeff=coeff, basis0=basis0, basis1=basis1, a=a, sign=sign, lambda_ai=lambda_ai, qweA=qweA, qweB=qweB)
    return ret


def get_BD24(sign:str, return_info:bool=False):
    assert sign in '+-'
    sign = 1 if sign=='+' else -1
    basis0 = np.zeros((2,128), dtype=np.float64)
    basis0[0, [0,58,92,102]] = np.array([1,1,1,-1])/2
    basis0[1, [3,5,63,95,105,113]] = np.array([1,-1,1,-1,1,-1])/np.sqrt(6)
    basis1 = (hf_pauli('X'*7) @ basis0.T).T
    coeff = np.array([np.sqrt(1/2),sign*np.sqrt(1/2)])
    code = np.stack([coeff @ basis0, coeff @ basis1], axis=0)
    ret = code
    if return_info:
        qweA = np.array([1, 0, 7/6, 0, 56/3, 0, 259/6, 0])
        qweB = np.array([1, 0, 7/6, 49/2, 56/3, 119, 259/6, 97/2])
        su2 = numqi.gate.rz(np.array([1,1,3,3,5,5,7])*2*np.pi/12) #logical rz(pi/6)
        info = dict(basis0=basis0, basis1=basis1, qweA=qweA, qweB=qweB, su2=su2)
        ret = code, info
    return ret


def get_BD26(a:float, sign=None, return_info=False):
    assert 0 <= a <= np.sqrt(2/13)
    if sign is None:
        sign = np.ones(5, dtype=np.float64)
    else:
        sign = tuple(int(x) for x in sign)
        assert len(sign) == 5
        assert all(x in [1,-1] for x in sign)
        sign = np.asarray(sign)
    s = np.sqrt
    si = lambda x: np.sqrt(x)*1j
    basis0 = np.zeros((5,128), dtype=np.complex128)
    basis0[0, [0,58,90,102,67]] = np.array([1,2,s(3),s(5),1j])/s(14)
    basis0[[1,2,3,4], [105,113,13,21]] = np.array([1,1,1,1])
    basis1 = (hf_pauli('X'*7) @ basis0.T).T
    coeff = np.array([s(7/13), a, s(max(0,2/13-a*a)), si(3/13-a*a), si(1/13+a*a)]) * sign
    code = np.stack([coeff@basis0, coeff@basis1], axis=0)
    ret = code
    if return_info:
        su2 = numqi.gate.rz(np.array([4,4,6,6,8,10,12])*np.pi/13)
        lambda_ai = np.zeros(210, dtype=np.float64)
        lambda_ai[[29,65,110,119]] = np.array([1,1,1,-1])*5/13
        lambda_ai[[56,74,101,191,200]] = np.array([-1,-1,-1,-1,1])*3/13
        lambda_ai[[128,146,155,173,182]] = np.array([1,1,-1,1,-1])/13
        lambda_ai[209] = -11/13
        lambda_ai[[21,25]] = np.sqrt(12)/13
        lambda_ai[[120,124]] = 2*(coeff[1]*coeff[2] - coeff[3]*coeff[4]).real
        lambda_ai[38] = 1/13 - 4*a*a
        lambda_ai[83] = 3/13 - 4*a*a
        lambda_ai[92] = -5/13 + 4*a*a
        lambda_ai[47] = -7/13 + 4*a*a
        lambda_ai[137] = -9/13 + 4*a*a
        lambda_ai[164] = -1/13 - 4*a*a
        A2 = 485/169 - 160/13*a*a + 80*a**4 + 16*np.prod(sign[1:])*a*np.sqrt((2/13-a*a)*(3/13-a*a)*(1/13+a*a))
        # A2 range [2.20?, 2.88?]
        qweA = np.array([1,0,A2,0,21-2*A2,0,42+A2,0])
        qweB = np.array([1,0,A2,21+3*A2,21-2*A2,126-6*A2,42+A2,45+3*A2])
        ret = code, dict(su2=su2, coeff=coeff, basis0=basis0, basis1=basis1, a=a, sign=sign, lambda_ai=lambda_ai, qweA=qweA, qweB=qweB)
    return ret


def get_BD28(a:float, sign=None, return_info=False):
    assert 0 <= a <= np.sqrt(1/14)
    if sign is None:
        sign = np.ones(5, dtype=np.float64)
    else:
        sign = tuple(int(x) for x in sign)
        assert len(sign) == 5
        assert all(x in [1,-1] for x in sign)
        sign = np.asarray(sign)
    s = np.sqrt
    si = lambda x: np.sqrt(x)*1j
    basis0 = np.zeros((5, 128), dtype=np.complex128)
    basis0[0, [0,60,90,113,13]] = np.array([1,s(3),s(5),2,s(6)*1j])/s(19)
    basis0[[1,2,3,4], [86,102,19,35]] = 1
    basis1 = (hf_pauli('X'*7) @ basis0.T).T
    coeff = np.array([s(19/28), a, s(5/28-a*a), si(max(0,1/14-a*a)), si(1/14+a*a)]) * sign
    code = np.stack([coeff@basis0, coeff@basis1], axis=0)
    ret = code
    if return_info:
        su2 = numqi.gate.rz(np.array([4,6,6,8,8,10,12])*np.pi/14)
        lambda_ai = np.zeros(210, dtype=np.float64)
        lambda_ai[[47,  56, 164, 173, 191]] = np.array([-1,-1,1,-1,-1])*2/7
        lambda_ai[[65,  74, 209]] = np.array([3,-3,-3])/7
        lambda_ai[[128, 182, 200]] = np.array([1,-1,-1])/7
        lambda_ai[92] = -4/7
        lambda_ai[[101,119]] = (1/7 - 4*a*a)*np.array([1,-1])
        lambda_ai[[156,160]] = 2*np.sqrt(5/19)*(coeff[0]*coeff[1]).real
        lambda_ai[29] = 2/7 - 4*a*a
        lambda_ai[38] = 2/7 + 4*a*a
        lambda_ai[137] = 4*a*a - 4/7
        lambda_ai[155] = -1/7 - 4*a*a
        lambda_ai[[75,79]] = 2*(coeff[1]*coeff[2] - coeff[3]*coeff[4]).real
        A2 = 95/49 - 20/7*a*a + 80*a**4 + 16*np.prod(sign[1:])*a*np.sqrt((5/28-a*a)*(1/196-a**4))
        # A2 range [1.85?, 2.15?]
        qweA = np.array([1,0,A2,0, 21-2*A2, 0, 42+A2, 0])
        qweB = np.array([1,0,A2, 21+3*A2, 21-2*A2, 126-6*A2, 42+A2, 45+3*A2])
        ret = code, dict(su2=su2, coeff=coeff, basis0=basis0, basis1=basis1, a=a, sign=sign, lambda_ai=lambda_ai, qweA=qweA, qweB=qweB)
    return ret


def get_BD30(a:float, sign=None, return_info=False):
    assert 0 <= a <= np.sqrt(7/30)
    if sign is None:
        sign = np.ones(5, dtype=np.float64)
    else:
        sign = tuple(int(x) for x in sign)
        assert len(sign) == 5
        assert all(x in [1,-1] for x in sign)
        sign = np.asarray(sign)
    basis0 = np.zeros((5,128), dtype=np.float64)
    basis0[0, [0,3,58,90,102]] = np.array([1,1,2,2,np.sqrt(6)])/4
    basis0[[1,2,3,4], [13,21,105,113]] = 1
    basis1 = (hf_pauli('X'*7) @ basis0.T).T
    s = np.sqrt
    si = lambda x: np.sqrt(x)*1j
    coeff = np.array([s(8/15), a*1j, si(max(0,3/10-a*a)), s(max(0,7/30-a*a)), s(max(0,a*a-1/15))]) * sign
    code = np.stack([coeff@basis0, coeff@basis1], axis=0)
    ret = code
    if return_info:
        su2 = numqi.gate.rz(np.array([4,4,6,6,8,14,16])*np.pi/15)
        lambda_ai = np.zeros(210, dtype=np.float64)
        lambda_ai[[21,25]] = 4/15
        lambda_ai[29] = 7/15
        lambda_ai[[56,101,191,200]] = np.array([-1,-1,-1,1])/5
        lambda_ai[[65,74,110,119]] = np.array([1,-1,1,-1])/3
        lambda_ai[[128, 146, 155, 173, 182, 201, 205]] = np.array([1,1,-1,1,-1,1,-1])/15
        lambda_ai[209] = -13/15
        lambda_ai[[38,83]] = 4*a*a - 11/15
        lambda_ai[[47,92]] = 7/15 - 4*a*a
        lambda_ai[137] = 1/5 - 4*a*a
        lambda_ai[164] = 4*a*a - 1
        lambda_ai[[120,124]] = (-2*coeff[1]*coeff[2] + 2*coeff[3]*coeff[4]).real
        A2 = 313/75 - 24*a*a + 80*a**4 + 16*np.prod(sign[1:])*np.sqrt(a*a*(3/10-a*a)*(7/30-a*a)*(a*a-1/15))
        # A2 range [2.173? 2.94?]
        qweA = np.array([1,0,A2,0, 21-2*A2, 0, A2+42, 0])
        qweB = np.array([1,0,A2, 3*A2+21, 21-2*A2, 126-6*A2, A2+42, 3*A2+45])
        info = dict(basis0=basis0, basis1=basis1, coeff=coeff, su2=su2, sign=sign, a=a, lambda_ai=lambda_ai, qweA=qweA, qweB=qweB)
        ret = code, info
    return ret


def get_BD32(a:float, sign=None, return_info=False):
    # sqrt(T)
    if sign is None:
        sign = np.ones(9, dtype=np.float64)
    else:
        sign = tuple(int(x) for x in sign)
        assert len(sign) == 9
        assert all(x in [1,-1] for x in sign)
        sign = np.asarray(sign)
    assert 0<= a <= 1/np.sqrt(8)
    s = np.sqrt
    si = lambda x: np.sqrt(x)*1j
    coeff = np.array([s(1/32), si(a*a+1/16), si(3/16-a*a), si(4/32), s(3/32), s(7/32), s(5/32), s(max(0,1/8-a*a)), a]) * sign
    basis0 = np.zeros((9, 128), dtype=np.float64)
    basis0[[0,1,2,3,4,5,6,7,8], [0,13,21,35,60,90,102,105,113]] = 1
    basis1 = (hf_pauli('X'*7) @ basis0.T).T
    code = np.stack([coeff@basis0, coeff@basis1], axis=0)
    if return_info:
        su2 = numqi.gate.rz(np.array([2,3,4,4,5,6,7])*2*np.pi/16)
        lambda_ai = np.zeros(210, dtype=np.float64)
        lambda_ai[[29,110,146,173]] = np.array([1,1,-1,-1])/8
        lambda_ai[[65,74,209]] = np.array([1,-1,-1])/2
        lambda_ai[[56,191]] = -3/8
        lambda_ai[[128, 155, 182]] = np.array([1,-1,-1])/4
        lambda_ai[[47, 164]] = np.array([1,-1]) * (3/8 - 4*a*a)
        lambda_ai[[38, 137]] = np.array([-1,1]) * (1/8 - 4*a*a)
        lambda_ai[92] = -1/8 - 4*a*a
        lambda_ai[83] = -5/8 + 4*a*a
        lambda_ai[[120,124]] = 2*(coeff[7]*coeff[8] - coeff[1]*coeff[2]).real
        A2 = 67/32 -10*a*a + 80*a**4 + 16*np.prod(sign[[1,2,7,8]])*a*np.sqrt((a*a+1/16)*(3/16-a*a)*(1/8-a*a))
        # A2 range [53/32, 2.1032206?]
        qweA = np.array([1, 0, A2, 0, 21-2*A2, 0, 42+A2, 0]) #same as 723-cyclic
        qweB = np.array([1,0,A2,21+3*A2,21-2*A2,126-6*A2,42+A2,45+3*A2])
        info = dict(su2=su2, basis0=basis0, basis1=basis1, coeff=coeff, a=a, sign=sign, lambda_ai=lambda_ai, A2=A2, qweA=qweA, qweB=qweB)
        ret = code,info
    else:
        ret = code
    return ret


def get_BD34(sign=None, theta=0, return_info=False):
    if sign is None:
        sign = np.ones(8, dtype=np.float64)
    else:
        sign = tuple(int(x) for x in sign)
        assert (len(sign)==8) and all([x in [-1, 1] for x in sign])
    basis0 = np.zeros((8,128), dtype=np.float64)
    basis0[np.arange(8), [0, 3, 58, 90, 102, 105, 14, 21]] = 1
    basis1 = (hf_pauli('X'*7) @ basis0.T).T
    s = np.sqrt
    si = lambda x: np.sqrt(x)*1j
    phase = np.array([np.exp(1j*theta)]*2 + [1]*6)
    coeff = np.array([1,1,2,2,s(6),s(7),si(2),3j])/s(34) * phase * np.asarray(sign)
    code = np.stack([coeff@basis0, coeff@basis1], axis=0)
    ret = code
    if return_info:
        c2t = np.cos(2*theta)
        tmp0 = np.array([289,0,831,44,4063,768,12289,212])/289
        tmp1 = np.array([0,0,0,-44,344,-768,680,-212])/289
        qweA = tmp0 + tmp1*c2t
        tmp0 = np.array([289,0,831,8394,5255,29892,14169,15154])/289
        tmp1 = np.array([0,0,0,168,-848,1536,-1200,344])/289
        qweB = tmp0 + tmp1*c2t
        su2 = numqi.gate.rz(np.array([2,2,3,4,5,8,9])*2*np.pi/17)
        ret = code, dict(basis0=basis0, basis1=basis1, coeff=coeff, su2=su2, qweA=qweA, qweB=qweB, sign=sign, theta=theta)
    return ret


def get_BD36(sign=None, theta:float=0, return_info=False):
    if sign is None:
        sign = np.ones(7, dtype=np.float64)
    else:
        assert len(sign)==7
        sign = [int(x) for x in sign]
        assert all((x==1 or x==-1) for x in sign)
        sign = np.array([np.exp(1j*theta)] + sign)
    basis0 = np.zeros((8,128), dtype=np.float64)
    basis0[np.arange(8),[0,14,60,35,102,105,90,21]] = np.array([1]*8)
    basis1 = (hf_pauli('X'*7) @ basis0.T).T
    s = np.sqrt
    si = lambda x: np.sqrt(x)*1j
    coeff = np.array([s(1),s(2),si(3),s(4),si(5),si(6),si(7),s(8)])/6 * sign
    code = np.stack([coeff@basis0, coeff@basis1], axis=0)
    ret = code
    if return_info:
        c2t = np.cos(2*theta)
        tmp0 = np.array([1,0,161/81,7/81,1330/81,35/27,3472/81,28/81])
        tmp1 = np.array([0,0,0,7/81,-49/81,35/27,-91/81,28/81])*c2t
        qweA = tmp0 + tmp1
        tmp0 = np.array([1,0,161/81,721/27,497/27,3010/27,3731/81,4079/81])
        tmp1 = np.array([0,0,0,-7/27,112/81,-70/27,56/27,-49/81])*c2t
        qweB = tmp0 + tmp1
        su2 = numqi.gate.rz(np.array([40,60,80,100,120,140,160])*np.pi/180)
        ret = code, dict(basis0=basis0, basis1=basis1, coeff=coeff, su2=su2, qweA=qweA, qweB=qweB, sign=sign, theta=theta)
    return ret


def get_2O_X5(sign=None, return_info:bool=False):
    if sign is None:
        sign = np.array([1, -1, 1, 1, -1, 1], dtype=np.int64)
    else:
        sign = tuple(int(x) for x in sign)
        assert len(sign) == 6
        assert all(x in [1,-1] for x in sign)
        assert (sign[1]==-sign[0]) and (sign[4]==-sign[0]) and (sign[5]==sign[3])
        sign = np.asarray(sign, dtype=np.int64)
    s = np.sqrt
    basis0 = np.zeros((6,128), dtype=np.float64)
    basis0[0, [0,15,20,99,108,119,3,12,23,96,111,116]] = np.array([1]*6 + [-1]*6)/np.sqrt(12)
    basis0[1, [36,40,51,60,68,72,83,92,39,43,48,63,71,75,80,95]] = np.array([1]*8 + [-1]*8)/4
    basis0[2, [32,35,84,87,52,55,64,67]] = np.array([1]*4 + [-1]*4)/np.sqrt(8)
    basis0[3, [33,53,66,86,34,54,65,85]] = np.array([1]*4 + [-1]*4)/np.sqrt(8)
    basis0[4, [24,123,27,120]] = np.array([1,1,-1,-1]) / 2
    basis0[5, [45,78,46,77]] = np.array([1,1,-1,-1]) / 2
    basis1 = (hf_pauli('XXXXXII') @ basis0.T).T
    coeff = np.array([1/4, 1/2, 1/2, s(1/12), s(3)/4, s(1/6)]) * sign
    code = np.stack([coeff@basis0,coeff@basis1], axis=0)
    ret = code
    if return_info:
        su2X = np.stack([numqi.gate.X]*5 + [numqi.gate.I]*2, axis=0)
        axis = np.array([[-1,0,-s(3)], [-1,0,-s(3)], [-1,0,1], [1,0,1], [-1,0,-1]])
        I,X,Y,Z = numqi.gate.I, numqi.gate.X, numqi.gate.Y, numqi.gate.Z
        theta = np.array([np.pi, np.pi, np.pi/2, np.pi/2, np.pi/2])
        nx,ny,nz = (axis.T/np.linalg.norm(axis, axis=1)).reshape(3,-1,1,1)
        tmp0 = np.cos(theta/2).reshape(-1,1,1)*I - 1j*np.sin(theta/2).reshape(-1,1,1)*(X*nx + Y*ny + Z*nz)
        su2YSY = np.stack(tmp0.tolist() + [I,I], axis=0)
        qweA = np.array([1,0,1,0,19,0,43,0])
        qweB = np.array([1,0,1,24,19,120,43,48])
        info = dict(basis0=basis0, basis1=basis1, coeff=coeff, sign=sign, su2X=su2X, su2YSY=su2YSY, qweA=qweA, qweB=qweB)
        ret = code, info
    return ret



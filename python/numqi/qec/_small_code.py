import numpy as np

import numqi.dicke

from ._pauli import hf_pauli
from .q623 import get_SO5_code
from .q723 import get_cyclic_code
from ._internal import hf_state


def stabilizer_to_code(stab_list:list[str], logicalZ_list:list[str], tag_print:bool=True):
    '''convert stabilizer to code subspace

    TODO: add logicalX_list for fix the phase of code subspace

    Parameters:
        stab_list (list[str]): list of Pauli string for stabilizer generator, e.g. ['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ']
        logicalZ_list (list[str]): logical Z operator list, e.g. ['ZZZZZ']
        tag_print (bool): whether to print the code subspace

    Returns:
        code_list (np.ndarray): shape=(2**k,2**n), code subspace, code_list[i] is the code subspace for logical i
    '''
    n = len(stab_list[0])
    assert all([len(x)==n for x in stab_list])
    k = len(logicalZ_list)
    assert all([len(x)==n for x in logicalZ_list])
    eye = np.eye(2**n)
    tmp0 = [hf_pauli(x)+eye for x in stab_list]
    qecc_projector = tmp0[0]
    for x in tmp0[1:]:
        qecc_projector = qecc_projector @ x
    logicalZ_np_list = [hf_pauli(x) for x in logicalZ_list]
    code_list = []
    for ind0 in range(2**k):
        bit = bin(ind0)[2:].rjust(k,'0')
        tmp0 = qecc_projector
        for x,y in zip(bit,logicalZ_np_list):
            tmp0 = tmp0 @ (eye + (1 if (x=='0') else -1)*y)
        EVL,EVC = np.linalg.eigh(tmp0)
        assert (EVL[-1]-2**n) < 1e-10
        code = EVC[:,-1]
        if tag_print:
            ind0 = np.nonzero(np.abs(code)>1e-4)[0]
            print(f'[{bit}]', ' '.join([bin(x)[2:].rjust(n,'0') for x in ind0]), code[ind0])
        code_list.append(code)
    return np.stack(code_list)


def get_code_subspace(key:str|None=None, **kwargs):
    r'''get code subspace for some well-known quantum error correction codes

    Parameters:
        key (str|None): key of the code, if None, print available key
        kwargs (dict): additional parameters for some codes

    Returns:
        ret (np.ndarray): shape=(2,2**n), code subspace, ret[i] is the code subspace for logical i
        info (dict): additional information
    '''
    # TODO carbon code https://errorcorrectionzoo.org/c/carbon
    if key is None:
        tmp0 = '442stab 523 623stab 623-SO5 642stab steane 723bare 723permutation 723cyclic 883 shor surface17'
        # 723graph
        print('Available key:', tmp0)
    info = dict()
    if key=='442stab':
        stab_list = ['XXXX', 'ZZZZ']
        logicalX0 = 'XXII'
        logicalZ0 = 'ZIZI'
        logicalX1 = 'IXIX'
        logicalZ1 = 'IIZZ'
        # code = stabilizer_to_code(stab_list, [logicalZ0,logicalZ1])
        q00 = '0000 1111'
        q01 = '0101 1010'
        q10 = '0011 1100'
        q11 = '0110 1001'
        ret = np.stack([hf_state(x) for x in [q00,q01,q10,q11]], axis=0)
        info = {
            'stab': stab_list,
            'logicalZ': [logicalZ0, logicalZ1],
            'logicalX': [logicalX0, logicalX1],
            'qweA': np.array([1,0,0,0,3]),
            'qweB': np.array([1,0,18,24,21]),
        }
    elif key=='523':
        # https://errorcorrectionzoo.org/c/braunstein
        stab_list = ['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ']
        logicalX = 'XXXXX'
        logicalZ = 'ZZZZZ'
        # code523 = stabilizer_to_code(stab_list, [logicalZ])
        code0 = hf_state('00000 -00011 00101 -00110 01001 01010 -01100 -01111 -10001 10010 10100 -10111 -11000 -11011 -11101 -11110')
        code1 = hf_state('00001 00010 00100 00111 01000 -01011 -01101 01110 10000 10011 -10101 -10110 11001 -11010 11100 -11111')
        ret = np.stack([code0, code1], axis=0)
        info = {
            'stab': stab_list,
            'logicalZ': logicalZ,
            'logicalX': logicalX,
            'qweA': np.array([1,0,0,0,15,0]),
            'qweB': np.array([1,0,0,30,15,18]),
        }
        # circ = numqi.qec.generate_code523()['encode']
        # ret = np.stack([circ.apply_state(x) for x in np.eye(2, 32).astype(np.complex128)], axis=0)
    elif key=='623stab':
        # https://errorcorrectionzoo.org/c/stab_6_1_3
        # https://arxiv.org/abs/0803.1495
        stab_list = ['YIZXXY', 'ZXIIXZ', 'IZXXXX', 'ZZZIZI', 'IIIZIZ']
        # gauge subgroup generator: IIIXII IIIZIZ
        logicalX = 'ZIXIXI'
        logicalZ = 'IZIIZZ'
        code0 = hf_state('000000 001111 010010 -011101 -100111 -101000 -110101 111010')
        code1 = hf_state('000101 001010 -010111 011000 100010 101101 -110000 111111')
        ## different from the paper, such that "ZXIIXZ" is the stabilizer instead of "-ZXIIXZ"
        # code0 = hf_state('000000 -100111 001111 -101000 -010010 110101 011101 -111010')
        # code1 = hf_state('001010 101101 000101 100010 -011000 -111111 010111 110000')
        ret = np.stack([code0, code1], axis=0)
        info = {
            'stab': stab_list,
            'logicalZ': logicalZ,
            'logicalX': logicalX,
            'qweA': np.array([1,0,1,0,11,16,3]),
            'qweB': np.array([1,0,1,24,35,40,27]),
        }
    elif key=='642stab':
        # https://errorcorrectionzoo.org/c/stab_6_2_2
        # https://www.nature.com/articles/nature03350
        stab_list = ['XIIXXX', 'XXXIIX', 'ZIIZZZ', 'ZZZIIZ']
        logicalX0 = 'IXXIII'
        logicalZ0 = 'IIZZIZ'
        logicalX1 = 'XIXXII'
        logicalZ1 = 'IIIZZI'
        # code642 = stabilizer_to_code(stab_list, [logicalZ0, logicalZ1])
        code00 = hf_state('000000 011110 100111 111001')
        code01 = hf_state('001011 010101 101100 110010')
        code10 = hf_state('000110 011000 100001 111111')
        code11 = hf_state('001101 010011 101010 110100')
        ret = np.stack([code00, code01, code10, code11], axis=0)
        info = {
            'stab': stab_list,
            'logicalZ': [logicalZ0, logicalZ1],
            'logicalX': [logicalX0, logicalX1],
            'qweA': np.array([1,0,0,0,9,0,6]),
            'qweB': np.array([1,0,9,24,99,72,51]),
        }
    elif key=='623-SO5':
        # https://arxiv.org/abs/2410.07983
        vece = kwargs.get('vece', np.array([1,1,1,1,1])/np.sqrt(5))
        coeff,info = get_SO5_code(vece, return_info=True)
    elif key=='steane':
        tag_cyclic = kwargs.get('cyclic', True)
        if tag_cyclic: #[0,1,3,2,5,6,4] cyclic symmetry
            code0 = hf_state('0000000 1100101 0101110 0010111 1001011 1110010 0111001 1011100')
            code1 = hf_state('1111111 0011010 1010001 1101000 0110100 0001101 1000110 0100011')
            stab_list = ['IIXIXXX', 'IXIXXXI', 'XIIXIXX', 'IIZIZZZ', 'IZIZZZI', 'ZIIZIZZ']
        else:
            code0 = hf_state('0000000 1010101 0110011 1100110 0001111 1011010 0111100 1101001')
            code1 = hf_state('1111111 0101010 1001100 0011001 1110000 0100101 1000011 0010110')
            stab_list = ['IIIXXXX', 'IXXIIXX', 'XIXIXIX', 'IIIZZZZ', 'IZZIIZZ', 'ZIZIZIZ']
        logicalX = 'XXXXXXX'
        logicalZ = 'ZZZZZZZ'
        ret = np.stack([code0, code1], axis=0)
        info = {
            'stab': stab_list,
            'logicalZ': logicalZ,
            'logicalX': logicalX,
            'qweA': np.array([1,0,0,0,21,0,42,0]),
            'qweB': np.array([1,0,0,21,21,126,42,45]),
        }
    elif key=='723bare':
        # bare((7,2,3)) https://arxiv.org/abs/1702.01155
        stab_list = 'XIIIXII IXIIXII IIXIIXI IIIXIIX IIZZIYY ZZZXZZI'.split(' ')
        logicalX = 'IXXXIII'
        logicalZ = 'ZZIIZII'
        # code = stabilizer_to_code(stab_list, [logicalZ])
        ret = np.zeros((2, 128), dtype=np.float64)
        ret[0,[0,1,2,8,9,11,16,18,19,25,26,27,36,37,38,44,45,47,52,54,55,61,62,63,68,69,70,76,77,
               79,84,86,87,93,94,95,96,97,98,104,105,107,112,114,115,121,122,123]] = 1/8
        ret[0,[3,10,17,24,39,46,53,60,71,78,85,92,99,106,113,120]] = -1/8
        ret[1,[4,13,22,31,32,41,50,59,64,73,82,91,100,109,118,127]] = 1/8
        ret[1,[5,6,7,12,14,15,20,21,23,28,29,30,33,34,35,40,42,43,48,49,51,56,57,58,65,66,67,72,
               74,75,80,81,83,88,89,90,101,102,103,108,110,111,116,117,119,124,125,126]] = -1/8
        info = {
            'stab': stab_list,
            'logicalZ': logicalZ,
            'logicalX': logicalX,
            'qweA': np.array([1,0,5,0,11,0,47,0]),
            'qweB': np.array([1,0,5,36,11,96,47,60]),
        }
    # elif key=='723graph':
    #     np0 = np.zeros((7,7), dtype=np.uint8)
    #     tmp0 = [[0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [6,0]]
    #     tmp0 += [[y,x] for x,y in tmp0]
    #     tmp0 = np.array(tmp0).T
    #     np0[tmp0[0], tmp0[1]] = 1
    #     code0 = numqi.sim.build_graph_state(np0, return_stabilizer_circ=False)
    #     circ = numqi.sim.Circuit()
    #     for x in range(7):
    #         circ.Z(x)
    #     code1 = circ.apply_state(code0)
    #     ret = np.stack([code0, code1], axis=0)
    #     info = {
    #         'qweA': np.array([1,0,0,0,21,0,42,0]),
    #         'qweB': np.array([1,0,0,21,21,126,42,45]),
    #     }
    elif key=='723permutation':
        # http://arxiv.org/abs/quant-ph/0304153v3
        # Permutationally Invariant Codes for Quantum Error Correction eq(62)
        # non-cws code
        sign = kwargs.get('sign', '+')
        assert sign in '+-'
        dicke_basis = numqi.dicke.get_dicke_basis(7, 2)[::-1]
        s = np.sqrt
        if sign=='+':
            tmp0 = np.array([s(15), -s(7), s(21), s(21)])/8
        else: #sign == '-'
            tmp0 = np.array([s(15), s(7), s(21), -s(21)])/8
        code0 = tmp0 @ dicke_basis[[0,2,4,6]]
        code1 = tmp0 @ dicke_basis[::-1][[0,2,4,6]] #apply X to all qubits
        ret = np.stack([code0,code1], axis=0)
        info = {
            'logicalX': 'XXXXXXX',
            'logicalZ': 'ZZZZZZZ',
            'qweA': np.array([1, 0, 7, 0, 7, 0, 49, 0]),
            'qweB': np.array([1, 0, 7, 42, 7, 84, 49, 66])
        }
    elif key=='723cyclic':
        lambda2 = kwargs.get('lambda2', 6)
        sign = kwargs.get('sign', '++')
        coeff, _, basis = get_cyclic_code(lambda2, sign)
        ret = coeff @ basis
        info = {
            'logicalX': 'XXXXXXX',
            'logicalZ': 'ZZZZZZZ',
            'lambda2': lambda2,
            'sign': sign,
            'basis': basis,
            'coeff': coeff,
            'qweA': np.array([1, 0, lambda2, 0, 21-2*lambda2, 0, 42+lambda2, 0]),
            'qweB': np.array([1, 0, lambda2, 21+3*lambda2, 21-2*lambda2, 126-6*lambda2, 42+lambda2, 45+3*lambda2]),
        }
    elif key=='883':
        # https://errorcorrectionzoo.org/c/stab_8_3_3
        # https://arxiv.org/abs/quant-ph/9605021 eq(25)
        q0 = '00000000 01010101 00110011 01100110 00001111 01011010 00111100 01101001 11111111 10101010 11001100 10011001 11110000 10100101 11000011 10010110'
        q1 = '-11000000 -10010101 11110011 10100110 -11001111 -10011010 11111100 10101001 -00111111 -01101010 00001100 01011001 -00110000 -01100101 00000011 01010110'
        q2 = '-10100000 -11110101 -10010011 -11000110 10101111 11111010 10011100 11001001 -01011111 -00001010 -01101100 -00111001 01010000 00000101 01100011 00110110'
        q3 = '01100000 00110101 -01010011 -00000110 -01101111 -00111010 01011100 00001001 10011111 11001010 -10101100 -11111001 -10010000 -11000101 10100011 11110110'
        q4 = '10001000 -11011101 -10111011 11101110 10000111 -11010010 -10110100 11100001 01110111 -00100010 -01000100 00010001 01111000 -00101101 -01001011 00011110'
        q5 = '-01001000 00011101 -01111011 00101110 -01000111 00010010 -01110100 00100001 -10110111 11100010 -10000100 11010001 -10111000 11101101 -10001011 11011110'
        q6 = '-00101000 01111101 00011011 -01001110 00100111 -01110010 -00010100 01000001 -11010111 10000010 11100100 -10110001 11011000 -10001101 -11101011 10111110'
        q7 = '11101000 -10111101 11011011 -10001110 -11100111 10110010 -11010100 10000001 00010111 -01000010 00100100 -01110001 -00011000 01001101 -00101011 01111110'
        ret = np.stack([hf_state(x) for x in [q0,q1,q2,q3,q4,q5,q6,q7]], axis=0)
        info = {
            'qweA': np.array([1,0,0,0,0,0,28,0,3]),
            'qweB': np.array([1,0,0,56,210,336,728,504,213])
        }
    elif key=='shor':
        stab_list = ['ZZIIIIIII', 'IZZIIIIII', 'IIIZZIIII', 'IIIIZZIII', 'IIIIIIZZI', 'IIIIIIIZZ', 'XXXXXXIII', 'IIIXXXXXX']
        logicalZ = 'XXXIIIIII'
        logicalX = 'ZIIZIIZII'
        # code = stabilizer_to_code(stab_list, [logicalZ])
        tmp0 = hf_state('000 111')
        code0 = np.kron(np.kron(tmp0, tmp0), tmp0)
        tmp0 = hf_state('000 -111')
        code1 = np.kron(np.kron(tmp0, tmp0), tmp0)
        ret = np.stack([code0, code1], axis=0)
        info = {
            'stab': stab_list,
            'logicalZ': logicalZ,
            'logicalX': logicalX,
            'qweA': np.array([1,0,9,0,27,0,75,0,144,0]),
            'qweB': np.array([1,0,9,39,27,207,75,333,144,189]),
        }
    elif key=='surface17':
        # https://arxiv.org/abs/1608.05053
        # https://errorcorrectionzoo.org/c/surface-17
        stab_list = ['ZZIIIIIII', 'XXIXXIIII', 'IZZIZZIII', 'IIXIIXIII', 'IIIXIIXII', 'IIIZZIZZI', 'IIIIXXIXX', 'IIIIIIIZZ']
        logicalZ = 'XXXIIIIII'
        logicalX = 'ZIIZIIZII'
        # code = stabilizer_to_code(stab_list, [logicalZ])
        code0 = hf_state('000000000 000000111 000011011 000011100 000100011 000100100 000111000 000111111 001001000 001001111 001010011 001010100 001101011 '
                        '001101100 001110000 001110111 110001000 110001111 110010011 110010100 110101011 110101100 110110000 110110111 111000000 111000111 '
                        '111011011 111011100 111100011 111100100 111111000 111111111')
        code1 = hf_state('000000000 -000000111 000011011 -000011100 -000100011 000100100 -000111000 000111111 001001000 -001001111 001010011 -001010100 '
                        '-001101011 001101100 -001110000 001110111 -110001000 110001111 -110010011 110010100 110101011 -110101100 110110000 -110110111 '
                        '-111000000 111000111 -111011011 111011100 111100011 -111100100 111111000 -111111111')
        ret = np.stack([code0,code1])
        info = {
            'stab': stab_list,
            'logicalZ': logicalZ,
            'logicalX': logicalX,
            'qweA': np.array([1, 0, 4, 0, 22, 0, 100, 0, 129, 0]),
            'qweB': np.array([1, 0, 4, 24, 22, 192, 100, 408, 129, 144]),
        }
    else:
        raise ValueError
    return ret, info

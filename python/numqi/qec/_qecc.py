import re

import numqi.sim
import numqi.gate

def parse_str_qecc(str_qecc:str):
    # ((n,K,d)) ((5,2,3))
    # ((n,K,de(w)=d)) ((5,2,de(2)=3))
    assert str_qecc.startswith('((') and str_qecc.endswith('))')
    tmp0 = str_qecc[2:-2].split(',',2)
    num_qubit = int(tmp0[0])
    num_logical_dim = int(tmp0[1])
    if '=' in tmp0[2]:
        weight_z = float(tmp0[2].split('(',1)[1].split(')',1)[0])
        distance = int(tmp0[2].split('=',1)[1])
    else:
        weight_z = None
        distance = int(tmp0[2])
    ret = dict(num_qubit=num_qubit, num_logical_dim=num_logical_dim,
            weight_z=weight_z, distance=distance)
    return ret


def parse_simple_pauli(str0, tag_circuit=True):
    if re.search('[0-9]', str0) is None: #XIYXX
        assert set(str0)<=set('XYZI')
        tmp0 = [(y,x) for x,y in enumerate(str0) if y!='I']
    else: #X0Y2X3X4
        tmp0 = re.match('([XYZI][0-9]+)+', str0)
        assert (tmp0 is not None) and len(str0)==tmp0.span(0)[1]
        tmp0 = [(x[0],int(x[1:])) for x in re.findall('[XYZI][0-9]+', str0)]
    if tag_circuit:
        ret = numqi.sim.Circuit()
        for x,y in tmp0:
            if x=='X':
                ret.rx(y)
            elif x=='Y':
                ret.ry(y)
            elif x=='Z':
                ret.rz(y)
    else:
        tmp1 = {'X':numqi.gate.X, 'Y':numqi.gate.Y, 'Z':numqi.gate.Z}
        ret = [(tmp1[x],y) for x,y in tmp0]
    return ret


# https://markus-grassl.de/QECC/circuits/index.html
def generate_code523():
    name = '((5,2,3))'
    ret = dict(name=name, **parse_str_qecc(name))
    circ = numqi.sim.Circuit()
    for x in range(4):
        circ.H(x)
    circ.cz(3, 4)
    circ.cy(2, 3)
    circ.cz(2, 4)
    circ.cx(1, 2)
    circ.cz(1, 3)
    circ.cx(1, 4)
    circ.cy(0, 2)
    circ.cx(0, 3)
    circ.cx(0, 4)
    tmp0 = ['XIYXX','IXXZX','ZIXYZ','IZZXZ']
    ret['encode'] = circ
    ret['stabilizer'] = [parse_simple_pauli(x,tag_circuit=True) for x in tmp0]
    return ret


def generate_code422():
    name = '((4,2,2))'
    ret = dict(name=name, **parse_str_qecc(name))
    circ = numqi.sim.Circuit()
    for x in range(3):
        circ.H(x)
    circ.cx(2, 3)
    circ.cz(1, 2)
    circ.cz(1, 3)
    circ.cz(0, 1)
    tmp0 = ['XZII','ZXZZ','IIXX']
    ret['encode'] = circ
    ret['stabilizer'] = [parse_simple_pauli(x,tag_circuit=True) for x in tmp0]
    return ret


def generate_code442():
    name = '((4,4,2))'
    ret = dict(name=name, **parse_str_qecc(name))
    circ = numqi.sim.Circuit()
    for x in range(2):
        circ.H(x)
    circ.cx(1, 2)
    circ.cz(1, 3)
    circ.cz(0, 1)
    circ.cz(0, 2)
    circ.cx(0, 3)
    tmp0 = ['XZZX', 'ZXXZ']
    ret['encode'] = circ
    ret['stabilizer'] = [parse_simple_pauli(x,tag_circuit=True) for x in tmp0]
    return ret


def generate_code642():
    name = '((6,4,2))'
    ret = dict(name=name, **parse_str_qecc(name))
    circ = numqi.sim.Circuit()
    circ.cz(4, 2)
    circ.cz(2, 4)
    circ.cx(2, 5)
    circ.H(0)
    circ.H(1)
    circ.cz(1, 2)
    circ.cx(1, 3)
    circ.cx(1, 4)
    circ.cz(1, 5)
    circ.cz(0, 1)
    circ.cx(0, 2)
    circ.cz(0, 3)
    circ.cz(0, 4)
    circ.cx(0, 5)
    tmp0 = ['XZXZZX', 'ZXZXXZ']
    ret['encode'] = circ
    ret['stabilizer'] = [parse_simple_pauli(x,tag_circuit=True) for x in tmp0]
    return ret


def generate_code883():
    name = '((8,8,3))'
    ret = dict(name=name, **parse_str_qecc(name))
    circ = numqi.sim.Circuit()
    for x in range(5):
        circ.H(x)
    circ.cz(7, 5)
    circ.cz(7, 6)
    circ.cz(6, 5)
    circ.cy(6, 7)
    circ.cx(5, 6)
    circ.cx(5, 7)
    circ.cz(4, 5)
    circ.cz(4, 6)
    circ.cx(4, 7)
    circ.cy(3, 5)
    circ.cy(3, 6)
    circ.cz(3, 7)
    circ.cy(2, 3)
    circ.cz(2, 4)
    circ.cz(2, 5)
    circ.cy(2, 6)
    circ.cx(2, 7)
    circ.cz(1, 2)
    circ.cz(1, 3)
    circ.cy(1, 4)
    circ.cx(1, 5)
    circ.cy(1, 6)
    circ.cz(0, 2)
    circ.cy(0, 4)
    circ.cz(0, 5)
    circ.cx(0, 6)
    circ.cy(0, 7)
    tmp0 = ['XIZIYZXY', 'IXZZYXYI', 'IIXYZZYX', 'ZZZZXZZX']
    ret['encode'] = circ
    ret['stabilizer'] = [parse_simple_pauli(x,tag_circuit=True) for x in tmp0]
    return ret


def generate_code8_64_2():
    name = '((8,64,2))'
    ret = dict(name=name, **parse_str_qecc(name))
    circ = numqi.sim.Circuit()
    circ.H(0)
    circ.H(1)
    circ.cz(6, 2)
    circ.cz(5, 2)
    circ.cz(5, 3)
    circ.cx(5, 6)
    circ.cz(4, 2)
    circ.cz(4, 3)
    circ.cx(4, 5)
    circ.cx(4, 6)
    circ.cz(3, 5)
    circ.cx(3, 7)
    circ.cx(2, 3)
    circ.cz(2, 5)
    circ.cz(2, 6)
    circ.cx(2, 7)
    circ.cz(1, 2)
    circ.cz(1, 3)
    circ.cx(1, 4)
    circ.cx(1, 5)
    circ.cx(1, 6)
    circ.cz(1, 7)
    circ.cz(0, 1)
    circ.cx(0, 2)
    circ.cx(0, 3)
    circ.cz(0, 4)
    circ.cz(0, 5)
    circ.cz(0, 6)
    circ.cx(0, 7)
    tmp0 = ['XZXXZZZX', 'ZXZZXXXZ']
    ret['encode'] = circ
    ret['stabilizer'] = [parse_simple_pauli(x,tag_circuit=True) for x in tmp0]
    return ret


def generate_code10_4_4():
    name = '((10,4,4))'
    ret = dict(name=name, **parse_str_qecc(name))
    circ = numqi.sim.Circuit()
    circ.cnot(8, 9)
    for x in range(8):
        circ.H(x)
    circ.cy(7, 8)
    circ.cy(7, 9)
    circ.cz(6, 7)
    circ.cz(6, 8)
    circ.cx(6, 9)
    circ.cz(5, 6)
    circ.cx(5, 7)
    circ.cy(5, 8)
    circ.cy(5, 9)
    circ.cy(4, 5)
    circ.cy(4, 7)
    circ.cx(4, 8)
    circ.cx(4, 9)
    circ.cy(3, 4)
    circ.cx(3, 5)
    circ.cz(3, 6)
    circ.cy(3, 7)
    circ.cx(3, 8)
    circ.cz(2, 5)
    circ.cz(2, 6)
    circ.cz(2, 7)
    circ.cx(2, 8)
    circ.cx(2, 9)
    circ.cy(1, 4)
    circ.cx(1, 5)
    circ.cy(1, 6)
    circ.cz(1, 8)
    circ.cz(1, 9)
    circ.cx(0, 4)
    circ.cy(0, 6)
    circ.cx(0, 7)
    circ.cz(0, 8)
    circ.cy(0, 9)
    tmp0 = ['XIIIXIYXZY', 'IXIIYXYIZZ', 'IIXIIZZZXX', 'IIIXYXZYXI', 'ZIIIXYIYXX', 'IZIIIXZXYY', 'ZIIZIIXZZX', 'ZIZZZIZXYY']
    ret['encode'] = circ
    ret['stabilizer'] = [parse_simple_pauli(x,tag_circuit=True) for x in tmp0]
    return ret


# def generate_code10_16_3():
#     # TODO wrong circuit below
#     name = '((10,16,3))'
#     ret = dict(name=name, **parse_str_qecc(name))
#     circ = numqi.sim.Circuit()
#     for x in range(5):
#         circ.H(x)
#     circ.cz(9, 8)
#     circ.cy(8, 9)
#     circ.cx(6, 5)
#     circ.cz(6, 8)
#     circ.cy(6, 9)
#     circ.H(5)
#     circ.cy(5, 7)
#     circ.cx(5, 8)
#     circ.cy(5, 9)
#     circ.cy(4, 7)
#     circ.cy(4, 8)
#     circ.cz(3, 4)
#     circ.cz(3, 5)
#     circ.cx(3, 6)
#     circ.cz(3, 7)
#     circ.cx(3, 8)
#     circ.cy(3, 9)
#     circ.cz(2, 3)
#     circ.cz(2, 4)
#     circ.cx(2, 5)
#     circ.cz(2, 6)
#     circ.cz(2, 7)
#     circ.cx(2, 8)
#     circ.cz(2, 9)
#     circ.cy(1, 3)
#     circ.cx(1, 4)
#     circ.cx(1, 5)
#     circ.cz(1, 6)
#     circ.cy(1, 7)
#     circ.cy(1, 8)
#     circ.cz(1, 9)
#     circ.cy(0, 3)
#     circ.cx(0, 4)
#     circ.cz(0, 5)
#     circ.cz(0, 6)
#     circ.cz(0, 8)
#     tmp0 = ['XIIYXZZIZI', 'IXIYXXZYYZ', 'IIXZZXZZXZ', 'IZIXZZXZXY', 'IZZZXIIYYI', 'ZZIIIXIZZY']
#     ret['encode'] = circ
#     ret['stabilizer'] = [parse_simple_pauli(x,tag_circuit=True) for x in tmp0]
#     return ret


def generate_code11_2_5():
    name = '((11,2,5))'
    ret = dict(name=name, **parse_str_qecc(name))
    circ = numqi.sim.Circuit()
    for x in range(10):
        circ.H(x)
    circ.cx(9, 10)
    circ.cz(8, 9)
    circ.cx(8, 10)
    circ.cx(7, 8)
    circ.cz(7, 10)
    circ.cz(6, 7)
    circ.cx(6, 9)
    circ.cz(6, 10)
    circ.cy(5, 6)
    circ.cy(5, 8)
    circ.cz(5, 9)
    circ.cz(5, 10)
    circ.cz(4, 5)
    circ.cx(4, 6)
    circ.cz(4, 8)
    circ.cz(4, 9)
    circ.cx(4, 10)
    circ.cz(3, 5)
    circ.cy(3, 6)
    circ.cx(3, 8)
    circ.cx(3, 9)
    circ.cy(3, 10)
    circ.cz(2, 5)
    circ.cz(2, 6)
    circ.cx(2, 7)
    circ.cx(2, 8)
    circ.cz(2, 9)
    circ.cz(1, 5)
    circ.cz(1, 6)
    circ.cy(1, 7)
    circ.cz(1, 8)
    circ.cy(1, 10)
    circ.cz(0, 5)
    circ.cz(0, 6)
    circ.cz(0, 7)
    circ.cx(0, 9)
    circ.cx(0, 10)
    tmp0 = ['XIIIIZZZIXX', 'IXIIIZZYZIY', 'IIXIIZZXXZI', 'IIIXIZYIXXY', 'IIIIXZXIZZX',
            'IIZIIXYIYZZ', 'IZZIIIXZIXZ', 'IZIIZIZXXIZ', 'ZZZIZIIZXZX', 'IZIZIIZIZXX']
    ret['encode'] = circ
    ret['stabilizer'] = [parse_simple_pauli(x,tag_circuit=True) for x in tmp0]
    return ret

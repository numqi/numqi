import re
import functools
import itertools
import numpy as np
from tqdm import tqdm
import scipy.special

import numpyqi.utils
import numpyqi.state
import numpyqi.gate

try:
    import torch
    from ._torch_only import QECCEqualModel, VarQECUnitary, VarQEC
    from .._torch_op import KnillLaflammeInnerProduct
except ImportError:
    torch = None

def make_error_list(num_qubit, distance, op_list=None, tag_full=False):
    assert distance>1
    if op_list is None:
        op_list = [numpyqi.gate.X, numpyqi.gate.Y, numpyqi.gate.Z]
    ret = []
    for weight in range(1, distance):
        for index_qubit in itertools.combinations(range(num_qubit), r=weight):
            for gate in itertools.product(op_list, repeat=weight):
                ret.append([([x],y) for x,y in zip(index_qubit,gate)])
    if tag_full:
        hf_kron = lambda *x: functools.reduce(np.kron, x[1:], x[0])
        tmp0 = ret
        ret = []
        for op0 in tmp0:
            tmp1 = [numpyqi.gate.pauli.s0 for _ in range(num_qubit)]
            for y,z in op0:
                tmp1[y[0]] = z
            ret.append(hf_kron(*tmp1))
    return ret


def hf_split_element(np0, num_of_each):
    # split np0 (len(np0)=N) into len(num_of_each) sets, elements in each set is unordered
    assert len(num_of_each) > 0
    assert sum(num_of_each) <= len(np0)
    tmp0 = list(range(np0.shape[0]))
    tmp1 = np.ones(np0.shape[0], dtype=np.bool_)
    if num_of_each[0]==0:
        if len(num_of_each)==1:
            yield (),
        else:
            tmp0 = (),
            for x in hf_split_element(np0, num_of_each[1:]):
                yield tmp0+x
    else:
        for ind0 in itertools.combinations(tmp0, num_of_each[0]):
            ind0 = list(ind0)
            if len(num_of_each)==1:
                yield tuple(np0[ind0].tolist()),
            else:
                tmp1[ind0] = False
                tmp2 = np0[tmp1]
                tmp1[ind0] = True
                tmp3 = tuple(np0[ind0].tolist()),
                for x in hf_split_element(tmp2, num_of_each[1:]):
                    yield tmp3 + x
# z0 = hf_split_element(np.arange(num_qubit), [0,1,0])

def make_asymmetric_error_set(num_qubit, distance, weight_z=1):
    # 0<=nx+ny+nz<=num_qubit
    # 0<=nx,ny,nz
    # nx+ny+cz*nz<distance
    assert weight_z>0
    ret = []
    for nxy in range(min(num_qubit,distance)):
        tmp0 = int(np.ceil((distance-nxy)/weight_z))
        for nz in range(min(num_qubit-nxy+1, tmp0)):
            if (nxy==0) and (nz==0):
                continue
            for nx in range(nxy+1):
                ny = nxy - nx
                for indx,indy,indz in hf_split_element(np.arange(num_qubit), [nx,ny,nz]):
                    tmp0 = [([x],numpyqi.gate.X) for x in indx]
                    tmp1 = [([x],numpyqi.gate.Y) for x in indy]
                    tmp2 = [([x],numpyqi.gate.Z) for x in indz]
                    ret.append(tmp0+tmp1+tmp2)
    return ret


def knill_laflamme_inner_product(q0, op_list):
    if numpyqi.utils.is_torch(q0):
        ret = KnillLaflammeInnerProduct.apply(q0, op_list)
    else:
        assert q0.ndim==2 #np.ndarray
        num_logical_dim = q0.shape[0]
        num_logical_qubit = numpyqi.utils.hf_num_state_to_num_qubit(num_logical_dim, kind='exact')
        ret = []
        q0_conj = q0.conj()
        for op_sequence in op_list:
            q1 = q0.reshape(-1)
            for (ind1,op_i) in op_sequence:
                q1 = numpyqi.state.apply_gate(q1, op_i, [x+num_logical_qubit for x in ind1])
            ret.append(q0_conj @ q1.reshape(num_logical_dim,-1).T)
        ret = np.stack(ret)
    return ret


# TODO numpyqi._torch_only.KnillLaflammeInnerProduct

def knill_laflamme_inner_product_grad(q0, op_list, term_grad):
    assert q0.ndim==2
    num_logical_dim = q0.shape[0]
    num_logical_qubit = numpyqi.utils.hf_num_state_to_num_qubit(num_logical_dim, kind='exact')
    q0_grad = np.zeros_like(q0)
    hf0 = lambda x: x.reshape(num_logical_dim, -1)
    for ind0 in range(len(op_list)):
        q1 = q0.reshape(-1)
        for ind1,op_i in op_list[ind0]:
            q1 = numpyqi.state.apply_gate(q1, op_i, [x+num_logical_qubit for x in ind1])
        q0_grad += term_grad[ind0].conj() @ hf0(q1)

        q1 = q0.reshape(-1)
        for ind1,op_i in reversed(op_list[ind0]):
            q1 = numpyqi.state.apply_gate(q1, op_i.T.conj(), [x+num_logical_qubit for x in ind1])
        q0_grad += term_grad[ind0].T @ hf0(q1)
    return q0_grad


def knill_laflamme_loss(inner_product, kind='L2'):
    assert kind in {'L1','L2'}
    assert inner_product.ndim==3
    num_logical_dim = inner_product.shape[1]
    if numpyqi.utils.is_torch(inner_product):
        mask = torch.triu(torch.ones(num_logical_dim, num_logical_dim,
                 dtype=torch.complex128, device=inner_product.device), diagonal=1)
        hf0 = lambda x: x if (kind=='L1') else torch.square(x)
        tmp0 = hf0(torch.abs(inner_product*mask)).sum()
        tmp1 = torch.diagonal(inner_product, dim1=1, dim2=2)
        tmp2 = hf0(torch.abs(tmp1 - tmp1.mean(dim=1, keepdim=True))).sum()
        loss = tmp0 + tmp2
    else:
        mask = np.triu(np.ones((num_logical_dim,num_logical_dim)), k=1)
        hf0 = lambda x: x if (kind=='L1') else np.square(x)
        tmp0 = hf0(np.abs(inner_product*mask)).sum()
        tmp1 = np.diagonal(inner_product, axis1=1, axis2=2)
        tmp2 = hf0(np.abs(tmp1 - tmp1.mean(axis=1, keepdims=True))).sum()
        loss = tmp0 + tmp2
    return loss


def degeneracy(code_i):
    # only works for d=3
    num_qubit = numpyqi.utils.hf_num_state_to_num_qubit(code_i.shape[0])
    error_list = make_error_list(num_qubit, distance=2) + [()] #identity
    mat = np.zeros([len(error_list), len(error_list)], dtype=np.complex128)
    for ind0 in range(len(error_list)):
        q0 = code_i
        for ind_op,op_i in error_list[ind0]:
            q0 = numpyqi.state.apply_gate(q0, op_i, ind_op)
        for ind1 in range(len(error_list)):
            q1 = code_i
            for ind_op,op_i in error_list[ind1]:
                q1 = numpyqi.state.apply_gate(q1, op_i, ind_op)
            mat[ind0,ind1] = np.vdot(q0, q1)
    EVL = np.linalg.eigvalsh(mat)
    return EVL


def quantum_weight_enumerator(code, use_tqdm=False):
    assert code.ndim==2
    num_qubit = numpyqi.utils.hf_num_state_to_num_qubit(code.shape[1], kind='exact')
    num_logical_dim = code.shape[0]
    num_logical_qubit = numpyqi.utils.hf_num_state_to_num_qubit(num_logical_dim, kind='ceil')
    if 2**num_logical_qubit > num_logical_dim:
        code = np.pad(code, [(0,2**num_logical_qubit-num_logical_dim),(0,0)], mode='constant', constant_values=0)
    code_conj = code.conj()
    op_list = [numpyqi.gate.X, numpyqi.gate.Y, numpyqi.gate.Z]
    retA = np.zeros(num_qubit, dtype=np.float64)
    retB = np.zeros(num_qubit, dtype=np.float64)
    for ind0 in range(num_qubit):
        weight = ind0 + 1
        tmp0 = itertools.combinations(range(num_qubit), r=weight)
        index_gate_generator = ((x,y) for x in tmp0 for y in itertools.product(op_list, repeat=weight))
        if use_tqdm:
            total = len(op_list)**weight * int(round(scipy.special.binom(num_qubit,weight)))
            index_gate_generator = tqdm(index_gate_generator, desc=f'weight={weight}', total=total)
        for index_qubit,gate in index_gate_generator:
            gate_list = [([x+num_logical_qubit],y) for x,y in zip(index_qubit,gate)]
            q0 = code.reshape(-1)
            for ind_op,op_i in gate_list:
                q0 = numpyqi.state.apply_gate(q0, op_i, ind_op)
            tmp0 = code_conj.reshape(-1, 2**num_qubit) @ q0.reshape(-1, 2**num_qubit).T
            retA[ind0] += abs(np.trace(tmp0).item())**2
            retB[ind0] += np.vdot(tmp0.reshape(-1), tmp0.reshape(-1)).real.item()
    retA /= num_logical_dim**2
    retB /= num_logical_dim
    return retA, retB


def check_stabilizer(stabilizer_circ_list, code):
    # code (list,np)
    ret = []
    for q0 in code:
        ret.append([np.vdot(q0, x.apply_state(q0)) for x in stabilizer_circ_list])
    ret = np.array(ret)
    return ret

def generate_code_np(circ, num_logical_dim):
    num_qubit = circ.num_qubit
    ret = []
    for ind0 in range(num_logical_dim):
        q0 = np.zeros(2**num_qubit, dtype=np.complex128)
        q0[ind0] = 1
        circ.apply_state(q0)
        ret.append(circ.apply_state(q0))
    ret = np.stack(ret, axis=0)
    return ret


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
        ret = numpyqi.circuit.Circuit()
        for x,y in tmp0:
            if x=='X':
                ret.rx(y)
            elif x=='Y':
                ret.ry(y)
            elif x=='Z':
                ret.rz(y)
    else:
        tmp1 = {'X':numpyqi.gate.X, 'Y':numpyqi.gate.Y, 'Z':numpyqi.gate.Z}
        ret = [(tmp1[x],y) for x,y in tmp0]
    return ret


# https://markus-grassl.de/QECC/circuits/index.html
def generate_code523():
    name = '((5,2,3))'
    ret = dict(name=name, **parse_str_qecc(name))
    circ = numpyqi.circuit.Circuit()
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
    circ = numpyqi.circuit.Circuit()
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
    circ = numpyqi.circuit.Circuit()
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
    circ = numpyqi.circuit.Circuit()
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
    circ = numpyqi.circuit.Circuit()
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
    circ = numpyqi.circuit.Circuit()
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
    circ = numpyqi.circuit.Circuit()
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
#     circ = numpyqi.circuit.Circuit()
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
    circ = numpyqi.circuit.Circuit()
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

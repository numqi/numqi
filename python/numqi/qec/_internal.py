import functools
import itertools
import numpy as np
from tqdm.auto import tqdm
import scipy.special
import torch

import numqi.gate
import numqi.utils
import numqi.sim

def make_error_list(num_qubit, distance, op_list=None, tag_full=False):
    assert distance>1
    if op_list is None:
        op_list = [numqi.gate.X, numqi.gate.Y, numqi.gate.Z]
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
            tmp1 = [numqi.gate.pauli.s0 for _ in range(num_qubit)]
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
                    tmp0 = [([x],numqi.gate.X) for x in indx]
                    tmp1 = [([x],numqi.gate.Y) for x in indy]
                    tmp2 = [([x],numqi.gate.Z) for x in indz]
                    ret.append(tmp0+tmp1+tmp2)
    return ret


def degeneracy(code_i):
    # only works for d=3
    num_qubit = numqi.utils.hf_num_state_to_num_qubit(code_i.shape[0])
    error_list = make_error_list(num_qubit, distance=2) + [()] #identity
    mat = np.zeros([len(error_list), len(error_list)], dtype=np.complex128)
    for ind0 in range(len(error_list)):
        q0 = code_i
        for ind_op,op_i in error_list[ind0]:
            q0 = numqi.sim.state.apply_gate(q0, op_i, ind_op)
        for ind1 in range(len(error_list)):
            q1 = code_i
            for ind_op,op_i in error_list[ind1]:
                q1 = numqi.sim.state.apply_gate(q1, op_i, ind_op)
            mat[ind0,ind1] = np.vdot(q0, q1)
    EVL = np.linalg.eigvalsh(mat)
    return EVL


def quantum_weight_enumerator(code, use_tqdm=False):
    assert code.ndim==2
    num_qubit = numqi.utils.hf_num_state_to_num_qubit(code.shape[1], kind='exact')
    num_logical_dim = code.shape[0]
    num_logical_qubit = numqi.utils.hf_num_state_to_num_qubit(num_logical_dim, kind='ceil')
    if 2**num_logical_qubit > num_logical_dim:
        code = np.pad(code, [(0,2**num_logical_qubit-num_logical_dim),(0,0)], mode='constant', constant_values=0)
    code_conj = code.conj()
    op_list = [numqi.gate.X, numqi.gate.Y, numqi.gate.Z]
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
                q0 = numqi.sim.state.apply_gate(q0, op_i, ind_op)
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


class _KnillLaflammeInnerProductTorchOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q0_torch, op_list):
        q0 = q0_torch.detach().numpy()
        num_logical_dim = q0.shape[0]
        num_logical_qubit = numqi.utils.hf_num_state_to_num_qubit(num_logical_dim, kind='exact')
        ret = []
        q0_conj = q0.conj()
        for op_sequence in op_list:
            q1 = q0.reshape(-1)
            for (ind1,op_i) in op_sequence:
                q1 = numqi.sim.state.apply_gate(q1, op_i, [x+num_logical_qubit for x in ind1])
            ret.append(q0_conj @ q1.reshape(num_logical_dim,-1).T)
        ret = torch.from_numpy(np.stack(ret, axis=0))
        ctx.save_for_backward(q0_torch)
        ctx._pyqet_data = dict(op_list=op_list)
        return ret

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        q0 = ctx.saved_tensors[0].detach().numpy()
        grad_output = grad_output.detach().numpy()
        op_list = ctx._pyqet_data['op_list']
        num_logical_dim = q0.shape[0]
        num_logical_qubit = numqi.utils.hf_num_state_to_num_qubit(num_logical_dim, kind='exact')
        q0_grad = np.zeros_like(q0)
        hf0 = lambda x: x.reshape(num_logical_dim, -1)
        for ind0 in range(len(op_list)):
            q1 = q0.reshape(-1)
            for ind1,op_i in op_list[ind0]:
                q1 = numqi.sim.state.apply_gate(q1, op_i, [x+num_logical_qubit for x in ind1])
            q0_grad += grad_output[ind0].conj() @ hf0(q1)

            q1 = q0.reshape(-1)
            for ind1,op_i in reversed(op_list[ind0]):
                q1 = numqi.sim.state.apply_gate(q1, op_i.T.conj(), [x+num_logical_qubit for x in ind1])
            q0_grad += grad_output[ind0].T @ hf0(q1)
        q0_grad = torch.from_numpy(q0_grad)
        return q0_grad,None


def knill_laflamme_inner_product(q0, op_list):
    if isinstance(q0, torch.Tensor):
        ret = _KnillLaflammeInnerProductTorchOp.apply(q0, op_list)
    else:
        assert q0.ndim==2 #np.ndarray
        num_logical_dim = q0.shape[0]
        num_logical_qubit = numqi.utils.hf_num_state_to_num_qubit(num_logical_dim, kind='exact')
        ret = []
        q0_conj = q0.conj()
        for op_sequence in op_list:
            q1 = q0.reshape(-1)
            for (ind1,op_i) in op_sequence:
                q1 = numqi.sim.state.apply_gate(q1, op_i, [x+num_logical_qubit for x in ind1])
            ret.append(q0_conj @ q1.reshape(num_logical_dim,-1).T)
        ret = np.stack(ret)
    return ret

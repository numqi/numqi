import functools
import math
import numpy as np

import numqi.gate
import numqi.dicke

def get_bg_picode(b2:int, g:int):
    # https://arxiv.org/abs/2411.13142 eq(7)
    assert (b2>=1) and (b2>=g)
    coeff = np.zeros((b2+g+1, 2), dtype=np.float64)
    a = np.sqrt((b2-g) / (2*b2))
    b = np.sqrt((b2+g) / (2*b2))
    coeff[0,0] = a
    coeff[b2,0] = b
    coeff[b2+g,1] = a
    coeff[g,1]  = b
    return coeff

# def get_Aydin_picode():
#     # https://doi.org/10.22331/q-2024-04-30-1321
#     # https://arxiv.org/abs/2411.13142 appendix A
#     pass


# def get_picode_ABBp():
#     pass #TODO


def _get_picode_weight_enumerator_hf0(code, wt_to_pauli_dict, index, tagB):
    dimK = code.shape[0]
    num_qubit = code.shape[1] - 1
    if wt_to_pauli_dict is None:
        wt_to_pauli_dict = numqi.dicke.get_pauli_symmetrical_projection(num_qubit, index=index)
    retA = []
    retB = []
    for x in index:
        if x==0:
            retA.append(1)
            if tagB:
                retB.append(1)
        else:
            pauli = wt_to_pauli_dict[x]['xyz']
            # WARNING: multiplicity might has round-off error
            multiplicity = np.array(wt_to_pauli_dict[x]['multiplicity'], dtype=np.float64)
            tmp0 = code.conj() @ (pauli @ code.T)
            tmp1 = np.diagonal(tmp0, axis1=1, axis2=2).real.sum(axis=1)
            retA.append((np.dot(tmp1*tmp1, multiplicity)/(dimK*dimK)))
            if tagB:
                tmp1 = tmp0.reshape(tmp0.shape[0], -1)*multiplicity.reshape(-1,1)
                retB.append(np.vdot(tmp1.reshape(-1), tmp0.reshape(-1)).real / dimK)
    return retA, retB

def get_picode_weight_enumerator(code, wt_to_pauli_dict:dict|None=None, index:int|list[int]|None=None, tagB:bool=True):
    assert (code.ndim==2) and (code.shape[0]<=code.shape[1])
    num_qubit = code.shape[1] - 1
    if index is None:
        index = tuple(range(num_qubit+1))
    isone = not hasattr(index, '__len__')
    if isone:
        index = (index,)
    index = tuple(int(x) for x in index)
    retA,retB = _get_picode_weight_enumerator_hf0(code, wt_to_pauli_dict, index, tagB)
    if tagB:
        ret = np.array(retA), np.array(retB)
    else:
        ret = np.array(retA)
    if isone:
        ret = (ret[0].item(), ret[1].item()) if tagB else ret.item()
    return ret


def get_code723_beth():
    s = np.sqrt
    tmp0 = np.array([s(15), 0, -s(7), 0, s(21), 0, s(21), 0])/8
    coeff = np.stack([tmp0, tmp0[::-1]], axis=0)

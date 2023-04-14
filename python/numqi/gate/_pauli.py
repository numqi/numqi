import functools
import itertools
import numpy as np

from ._internal import pauli

hf_kron = lambda x: functools.reduce(np.kron, x)

_one_pauli_str_to_np = dict(zip('IXYZ', [pauli.s0, pauli.sx, pauli.sy, pauli.sz]))

def pauli_str_to_matrix(pauli_str, return_orth=False):
    #'XX YZ IZ'
    pauli_str = sorted(set(pauli_str.split()))
    num_qubit = len(pauli_str[0])
    assert all(len(x)==num_qubit for x in pauli_str)
    matrix_space = np.stack([hf_kron([_one_pauli_str_to_np[y] for y in x]) for x in pauli_str])
    if return_orth:
        pauli_str_orth = sorted(set(get_pauli_group(num_qubit, kind='str')) - set(pauli_str))
        matrix_space_orth = np.stack([hf_kron([_one_pauli_str_to_np[y] for y in x]) for x in pauli_str_orth])
        ret = matrix_space,matrix_space_orth
    else:
        ret = matrix_space
    return ret


@functools.lru_cache
def get_pauli_group(num_qubit, kind='numpy'):
    assert kind in {'numpy','str','str_to_index'}
    if kind=='numpy':
        ret = tuple(hf_kron([_one_pauli_str_to_np[y] for y in x]) for x in itertools.product(*['IXYZ']*num_qubit))
    else:
        tmp0 = tuple(''.join(x) for x in itertools.product(*['IXYZ']*num_qubit))
        if kind=='str':
            ret = tmp0
        else: #str_to_index
            ret = {y:x for x,y in enumerate(tmp0)}
    return ret


def pauli_index_str_convert(terms, num_qubit=None):
    if isinstance(terms[0], str):
        num_qubit = len(terms[0])
        tmp0 = get_pauli_group(num_qubit, kind='str_to_index')
        ret = [tmp0[x] for x in terms]
    else:
        assert num_qubit is not None
        tmp0 = get_pauli_group(num_qubit, kind='str')
        ret = [tmp0[x] for x in terms]
    return ret

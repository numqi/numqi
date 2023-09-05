import math
import numpy as np

def get_xbit(m, n):
    assert (0<m) and (m<=n) and (m<=64)
    tmp0 = np.arange(2**m, dtype=np.uint64).view(np.uint8).reshape(-1, 8)
    ret = np.unpackbits(tmp0, axis=1, count=n, bitorder='little')
    return ret


def get_measure_matrix(bitmap, partition):
    partition = np.asarray(partition)
    ret = np.zeros((bitmap.shape[0], partition.sum()), dtype=np.int64)
    tmp0 = np.cumsum(np.pad(partition, [(1,0)], mode='constant'))
    for ind0 in range(len(partition)):
        ret[bitmap==ind0,tmp0[ind0]:tmp0[ind0+1]] = 1
    return ret


def get_hamming_weight(x):
    assert x>=0
    ret = sum((x>>i)&1 for i in range(math.ceil(math.log2(max(x+1,2)))))
    return ret


def get_hamming_modulo_map(num_bit, num_modulo):
    ret = np.array([get_hamming_weight(x)%num_modulo for x in range(2**num_bit)])
    return ret


def get_exact_map(num_bit, ind_list):
    assert num_bit>0
    assert (min(ind_list)>=0) and (max(ind_list)<=num_bit)
    ind_list = {int(x) for x in ind_list}
    ret = np.array([(get_hamming_weight(x) in ind_list) for x in range(2**num_bit)], dtype=np.int64)
    return ret

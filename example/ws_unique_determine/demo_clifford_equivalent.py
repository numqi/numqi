import sys
import time
import itertools
import collections
import numpy as np
from tqdm import tqdm

import numqi

endianness_map = {
    '>': 'big',
    '<': 'little',
    '=': sys.byteorder,
    '|': 'not applicable',
}

def pauli_index_to_GF4(index, num_qubit:int):
    assert num_qubit <= 32
    index = np.array(index, dtype=np.uint64)
    assert index.ndim==1
    if endianness_map[index.dtype.byteorder]=='little':
        index = index.byteswap()
    index_z2 = np.unpackbits(index.view(np.uint8).reshape(index.shape[0],-1), axis=1, bitorder='big')[:,-(2*num_qubit):].reshape(-1,num_qubit,2)
    ret = np.zeros((len(index_z2)*num_qubit, 2), dtype=np.uint8)
    for x,y in [((0,1),(1,0)), ((1,0),(1,1)), ((1,1),(0,1))]:
        tmp0 = np.all(index_z2==np.array(x, dtype=np.uint8), axis=2).reshape(-1)
        ret[tmp0] = np.array(y, dtype=np.uint8)
    ret = ret.reshape(-1, num_qubit, 2).transpose(0,2,1).reshape(-1,2*num_qubit)
    return ret


def GF4_to_pauli_index(np0):
    assert (np0.dtype.type==np.uint8) and (np0.ndim==2)
    assert np0.shape[1]%2==0
    num_qubit = np0.shape[1]//2
    N0 = np0.shape[0]
    np0 = np0.reshape(N0,2,num_qubit).transpose(0,2,1).reshape(N0*num_qubit,2)
    np1 = np.zeros_like(np0)
    for x,y in [((0,1),(1,0)), ((1,0),(1,1)), ((1,1),(0,1))]:
        ind0 = np.all(np0==np.array(y,dtype=np.uint8), axis=1)
        np1[ind0] = np.array(x, dtype=np.uint8)
    tmp0 = 1<<np.arange(2*num_qubit)[::-1]
    ret = np1.reshape(N0, num_qubit*2) @ tmp0
    return ret


def get_pauli_subset_equivalent(subset, num_qubit, print_every_N=10000):
    first_element = tuple(subset)
    equivalent_set = {first_element}
    first_element_GF4 = pauli_index_to_GF4(first_element, num_qubit)
    tmp0 = [((1<<(2*x))-1) for x in range(1,num_qubit+1)]
    tmp1 = [(1<<(2*x-1)) for x in range(1,num_qubit+1)]
    order_a_b_list = [y for x in zip(tmp0,tmp1) for y in x]
    t0 = time.time()
    last_print_N0 = 0
    for ind_sp2n in itertools.product(*[range(x) for x in order_a_b_list]):
        tmp0 = numqi.group.spf2.from_int_tuple(ind_sp2n)
        tmp1 = np.sort(GF4_to_pauli_index((first_element_GF4 @ tmp0) % 2))
        equivalent_set.add(tuple(tmp1.tolist()))
        N0 = len(equivalent_set)
        if (N0%print_every_N==0) and (N0>last_print_N0):
            print(f'[{time.time()-t0:.1f}s] {N0}')
            last_print_N0 = N0
    return equivalent_set


def get_pauli_subset_stabilizer(subset, num_qubit, print_every_N=10000):
    first_element = tuple(sorted(subset))
    first_element_GF4 = pauli_index_to_GF4(first_element, num_qubit)
    tmp0 = [((1<<(2*x))-1) for x in range(1,num_qubit+1)]
    tmp1 = [(1<<(2*x-1)) for x in range(1,num_qubit+1)]
    order_a_b_list = [y for x in zip(tmp0,tmp1) for y in x]
    t0 = time.time()
    last_print_N0 = 0
    stabilizer_list = []
    for ind_sp2n in itertools.product(*[range(x) for x in order_a_b_list]):
        tmp0 = numqi.group.spf2.from_int_tuple(ind_sp2n)
        tmp1 = np.sort(GF4_to_pauli_index((first_element_GF4 @ tmp0) % 2))
        if tuple(tmp1.tolist())==first_element:
            stabilizer_list.append(ind_sp2n)
        N0 = len(stabilizer_list)
        if (N0%print_every_N==0) and (N0>last_print_N0):
            print(f'[{time.time()-t0:.1f}s] {N0}')
            last_print_N0 = N0
    return stabilizer_list


def test_GF4_to_pauli_index():
    np_rng = np.random.default_rng()
    for num_qubit in [3,4,5]:
        N0 = 4**num_qubit
        np0 = np_rng.integers(0, N0, size=23).astype(np.uint64)
        np1 = pauli_index_to_GF4(np0, num_qubit)
        np2 = GF4_to_pauli_index(np1)
        assert np.array_equal(np0, np2)


num_qubit = 2
ud_list = numqi.unique_determine.save_index_to_file('pauli-indexB-core.json', f'{num_qubit},udp')

equivalent_set = get_pauli_subset_equivalent(ud_list[0], num_qubit)
z0 = [tuple(x) for x in ud_list if len(x)==len(ud_list[0])]
tmp0 = [x for x in z0 if (x not in equivalent_set)]
if len(tmp0)==0:
    print('all equivalent')
else:
    print(f'#not equivalent={len(tmp0)}')


def get_pauli_all_subset_equivalent(num_qubit, order_start=1, order_end=None):
    num_pauli = 4**num_qubit
    assert num_qubit in {2,3} #>=3 is too slow
    if order_end is None:
        order_end = num_pauli
    all_equivalent_set = dict()
    for subset_order in range(max(1,order_start), min(order_end, num_pauli)):
        subset_list = list(itertools.combinations(list(range(num_pauli)), subset_order))
        z0 = []
        for subset_i in tqdm(subset_list):
            for x in z0:
                if subset_i in x:
                    break
            else:
                z0.append(get_pauli_subset_equivalent(subset_i, num_qubit))
        tmp0 = collections.Counter([len(x) for x in z0])
        tmp0 = sorted(tmp0.items(),key=lambda x:x[0])
        tmp0 = '+'.join([f'{v}x{k}' for k,v in tmp0])
        print(f'#[len={subset_order}]={len(z0)}, {tmp0}')
        all_equivalent_set[subset_order] = z0
    return all_equivalent_set


def print_all_equivalent_set(all_equivalent_set):
    z0 = {'origin':all_equivalent_set, 'withI':dict(), 'noI':dict()}
    tmp0 = dict()
    for key,value in all_equivalent_set.items():
        z0['withI'][key] = [x for x in value if (0 in sorted(x)[0])]
    for key,value in all_equivalent_set.items():
        z0['noI'][key] = [x for x in value if (0 not in sorted(x)[0])]

    for key,value in z0.items():
        print(key)
        for subset_order,z0 in value.items():
            tmp0 = collections.Counter([len(x) for x in z0])
            tmp0 = sorted(tmp0.items(),key=lambda x:x[0])
            tmp0 = '+'.join([f'{v}x{k}' for k,v in tmp0])
            print(f'#[len={subset_order}]={len(z0)}, {tmp0}')
        print()

all_equivalent_set = get_pauli_all_subset_equivalent(num_qubit=2)
print_all_equivalent_set(all_equivalent_set)
'''
origin
#[len=1]=2, 1x1+1x15
#[len=2]=3, 1x15+1x45+1x60
#[len=3]=7, 1x15+1x20+1x45+2x60+2x180
#[len=4]=14, 1x15+1x20+1x30+1x45+3x60+1x90+4x180+2x360
#[len=5]=24, 1x6+1x30+2x45+3x60+1x72+3x90+1x120+4x180+8x360
#[len=6]=36, 1x6+1x10+1x15+1x45+4x60+1x72+4x90+2x120+5x180+15x360+1x720
#[len=7]=45, 1x10+2x15+4x60+4x90+3x120+12x180+15x360+4x720
#[len=8]=48, 2x15+2x60+4x90+4x120+18x180+12x360+6x720
#[len=9]=45, 1x10+2x15+4x60+4x90+3x120+12x180+15x360+4x720
#[len=10]=36, 1x6+1x10+1x15+1x45+4x60+1x72+4x90+2x120+5x180+15x360+1x720
#[len=11]=24, 1x6+1x30+2x45+3x60+1x72+3x90+1x120+4x180+8x360
#[len=12]=14, 1x15+1x20+1x30+1x45+3x60+1x90+4x180+2x360
#[len=13]=7, 1x15+1x20+1x45+2x60+2x180
#[len=14]=3, 1x15+1x45+1x60
#[len=15]=2, 1x1+1x15

withI
#[len=1]=1, 1x1
#[len=2]=1, 1x15
#[len=3]=2, 1x45+1x60
#[len=4]=5, 1x15+1x20+1x60+2x180
#[len=5]=9, 1x30+1x45+2x60+1x90+2x180+2x360
#[len=6]=15, 1x6+1x45+1x60+1x72+2x90+1x120+2x180+6x360
#[len=7]=21, 1x10+1x15+3x60+2x90+1x120+3x180+9x360+1x720
#[len=8]=24, 1x15+1x60+2x90+2x120+9x180+6x360+3x720
#[len=9]=24, 1x15+1x60+2x90+2x120+9x180+6x360+3x720
#[len=10]=21, 1x10+1x15+3x60+2x90+1x120+3x180+9x360+1x720
#[len=11]=15, 1x6+1x45+1x60+1x72+2x90+1x120+2x180+6x360
#[len=12]=9, 1x30+1x45+2x60+1x90+2x180+2x360
#[len=13]=5, 1x15+1x20+1x60+2x180
#[len=14]=2, 1x45+1x60
#[len=15]=1, 1x15

noI
#[len=1]=1, 1x15
#[len=2]=2, 1x45+1x60
#[len=3]=5, 1x15+1x20+1x60+2x180
#[len=4]=9, 1x30+1x45+2x60+1x90+2x180+2x360
#[len=5]=15, 1x6+1x45+1x60+1x72+2x90+1x120+2x180+6x360
#[len=6]=21, 1x10+1x15+3x60+2x90+1x120+3x180+9x360+1x720
#[len=7]=24, 1x15+1x60+2x90+2x120+9x180+6x360+3x720
#[len=8]=24, 1x15+1x60+2x90+2x120+9x180+6x360+3x720
#[len=9]=21, 1x10+1x15+3x60+2x90+1x120+3x180+9x360+1x720
#[len=10]=15, 1x6+1x45+1x60+1x72+2x90+1x120+2x180+6x360
#[len=11]=9, 1x30+1x45+2x60+1x90+2x180+2x360
#[len=12]=5, 1x15+1x20+1x60+2x180
#[len=13]=2, 1x45+1x60
#[len=14]=1, 1x15
#[len=15]=1, 1x1
'''

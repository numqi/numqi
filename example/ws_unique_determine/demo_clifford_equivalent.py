import collections

import numqi

num_qubit = 2
ud_list = numqi.unique_determine.load_pauli_ud_example(num_qubit, tag_group_by_size=True)[13]

equivalent_set = numqi.gate.get_pauli_subset_equivalent(ud_list[0], num_qubit)
z0 = [tuple(x) for x in ud_list if len(x)==len(ud_list[0])]
tmp0 = [x for x in z0 if (x not in equivalent_set)]
if len(tmp0)==0:
    print('all equivalent')
else:
    print(f'#not equivalent={len(tmp0)}')


all_equivalent_set = numqi.gate.get_pauli_all_subset_equivalent(num_qubit=2)

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

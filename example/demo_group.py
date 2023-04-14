import numpy as np

import numpyqi


np_rng = np.random.default_rng()
hf_randc = lambda *size: np_rng.normal(size=size) + 1j*np_rng.normal(size=size)

N0 = 4
# z0 = numpyqi.group.get_multiplicative_group_cayley_table(N0)
z0 = numpyqi.group.get_symmetric_group_cayley_table(N0)
# z0 = numpyqi.group.get_symmetric_group_cayley_table(N0, alternating=True)
# z0 = numpyqi.group.get_dihedral_group_cayley_table(N0)
# z0 = numpyqi.group.get_cyclic_group_cayley_table(N0)
# z0 = numpyqi.group.get_klein_four_group_cayley_table()
# z0 = numpyqi.group.get_quaternion_cayley_table()
z1 = numpyqi.group.cayley_table_to_left_regular_form(z0)
irrep_list = numpyqi.group.reduce_group_representation(z1)
character,class_list,character_table = numpyqi.group.get_charater_and_class(irrep_list)
num_element = irrep_list[0].shape[0]
dim_list = [x.shape[1] for x in irrep_list]
print(f'#element={num_element}', f'dim(irrep)={dim_list}', f'#class={[len(x) for x in class_list]}', sep='\n')
print(np.array2string(character_table, precision=3))

# tmp0 = np.round(character_table.real).astype(np.int64)
# character_table_str = [[(str(y0) if abs(y0-y1)<1e-10 else 'xxx') for y0,y1 in zip(x0,x1)] for x0,x1 in zip(tmp0,character_table)]
# print('| $\chi$ | {} |'.format(' | '.join(str(len(x)) for x in class_list)))
# print('| {} |'.format(' | '.join([':-:']*(len(class_list)+1))))
# # assert np.abs(tmp0-character_table).max() < 1e-10
# for ind0 in range(len(irrep_list)):
#     tmp1 = ' | '.join(character_table_str[ind0])
#     tmp2 = '{' + str(ind0) + '}'
#     print(f'| $A_{tmp2}$ | {tmp1} |')

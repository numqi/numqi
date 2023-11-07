import time
import numpy as np

import numqi

case_dict = {
    2: [6,8,10,16,32,64],
    3: [2,3,4,5,6],
    # 4: [3,4], #d=4,k=4 is almost impossible
}

dim = 2
kext_list = case_dict[dim]
alpha_irrep_list = []
alpha_analytic_list = []
alpha_list = []
ret_list = []
dm_werner = numqi.state.Werner(dim, alpha=1)
dm_norm = numqi.gellmann.dm_to_gellmann_norm(dm_werner)
for kext in kext_list:
    _ = numqi.group.symext.get_symmetric_extension_irrep_coeff(dim, kext) #build cache
    t0 = time.time()
    beta = numqi.entangle.symext.get_ABk_symmetric_extension_boundary(dm_werner, (dim,dim), kext, use_ppt=False, use_boson=False)
    alpha_irrep = (beta/dm_norm)*dim/(beta/dm_norm+dim-1)
    alpha_analytical = (kext+dim*dim-dim)/(kext*dim+dim-1)
    tmp0 = time.time() - t0
    print(f'[d={dim},kext={kext}][{tmp0:.3f}s] alpha={alpha_irrep:.6f}, abs(error)={abs(alpha_analytical-alpha_irrep):.5g}')
    ret_list.append((alpha_irrep, alpha_analytical, tmp0))

# gtx3060
# [d=2,kext=6][0.099s] alpha=0.615385, abs(error)=5.0737e-14
# [d=2,kext=8][0.161s] alpha=0.588235, abs(error)=2.4725e-13
# [d=2,kext=10][0.161s] alpha=0.571429, abs(error)=7.5385e-11
# [d=2,kext=16][0.323s] alpha=0.545455, abs(error)=8.6462e-12
# [d=2,kext=32][3.181s] alpha=0.523077, abs(error)=6.4994e-10
# [d=2,kext=64][51.957s] alpha=0.511628, abs(error)=2.9869e-09

# [d=3,kext=2][0.136s] alpha=1.000000, abs(error)=1.6722e-10
# [d=3,kext=3][0.508s] alpha=0.818182, abs(error)=1.8212e-10
# [d=3,kext=4][2.377s] alpha=0.714286, abs(error)=1.7246e-10
# [d=3,kext=5][11.556s] alpha=0.647059, abs(error)=1.4563e-09
# [d=3,kext=6][55.595s] alpha=0.600000, abs(error)=7.9243e-12


dim = 3
kext_list = [3,4,5,6]
for kext in kext_list:
    coeffB_list,multiplicity_list = numqi.group.symext.get_symmetric_extension_irrep_coeff(dim, kext) #build cache
    dim_list = [x.shape[0] for x in coeffB_list]
    assert sum(x*y for x,y in zip(dim_list,multiplicity_list))==dim**kext
    tmp0 = ' + '.join([f'{x}x{y}' for x,y in zip(dim_list,multiplicity_list)])
    # tmp0 = ' + '.join([rf'{x}\times {y}' for x,y in zip(dim_list,multiplicity_list)])
    num_parameter = sum(x*x for x in dim_list)
    print(f'{dim}^{kext} = {tmp0}, #parameter={num_parameter}/{dim**(2*kext)}')
# 3^3 = 10x1 + 8x2 + 1x1, #parameter=165
# 3^4 = 15x1 + 15x3 + 6x2 + 3x3, #parameter=495
# 3^5 = 21x1 + 24x4 + 15x5 + 6x6 + 3x5, #parameter=1287
# 3^6 = 28x1 + 35x5 + 27x9 + 10x5 + 10x10 + 8x16 + 1x5, #parameter=3003
# 3^3 = 10\times 1 + 8\times 2 + 1\times 1, #parameter=165/729
# 3^4 = 15\times 1 + 15\times 3 + 6\times 2 + 3\times 3, #parameter=495/6561
# 3^5 = 21\times 1 + 24\times 4 + 15\times 5 + 6\times 6 + 3\times 5, #parameter=1287/59049
# 3^6 = 28\times 1 + 35\times 5 + 27\times 9 + 10\times 5 + 10\times 10 + 8\times 16 + 1\times 5, #parameter=3003/531441

dim = 4
kext_list = [3,4,5]
for kext in kext_list:
    coeffB_list,multiplicity_list = numqi.group.symext.get_symmetric_extension_irrep_coeff(dim, kext) #build cache
    dim_list = [x.shape[0] for x in coeffB_list]
    assert sum(x*y for x,y in zip(dim_list,multiplicity_list))==dim**kext
    tmp0 = ' + '.join([f'{x}x{y}' for x,y in zip(dim_list,multiplicity_list)])
    tmp0 = ' + '.join([rf'{x}\times {y}' for x,y in zip(dim_list,multiplicity_list)])
    num_parameter = sum(x*x for x in dim_list)
    print(f'{dim}^{kext} = {tmp0}, #parameter={num_parameter}/{dim**(2*kext)}')
# 4^3 = 20x1 + 20x2 + 4x1, #parameter=816
# 4^4 = 35x1 + 45x3 + 20x2 + 15x3 + 1x1, #parameter=3876
# 4^5 = 56x1 + 84x4 + 60x5 + 36x6 + 20x5 + 4x4, #parameter=15504
# 4^3 = 20\times 1 + 20\times 2 + 4\times 1, #parameter=816/4096
# 4^4 = 35\times 1 + 45\times 3 + 20\times 2 + 15\times 3 + 1\times 1, #parameter=3876/65536
# 4^5 = 56\times 1 + 84\times 4 + 60\times 5 + 36\times 6 + 20\times 5 + 4\times 4, #parameter=15504/1048576

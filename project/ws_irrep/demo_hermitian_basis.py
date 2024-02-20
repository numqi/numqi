import numpy as np
import time
import cvxpy

import numqi

def get_symmetric_extension_boundary_hermitian_basis(rho, dim, kext, use_boson=False):
    dimA,dimB = dim
    index_sym,index_skew,factor_skew = numqi.group.symext.get_ABk_symmetry_index(dimA, dimB, kext, use_boson=use_boson)
    rho_vec = (rho - np.eye(dimA*dimB)/(dimA*dimB)) / numqi.gellmann.dm_to_gellmann_norm(rho)
    cvx_sym = cvxpy.Variable(index_sym.max()+1)
    cvx_skew_sym = cvxpy.Variable(index_skew.max())

    rho_ABk = cvx_sym[index_sym] + cvxpy.multiply(cvxpy.hstack([[0], cvx_skew_sym])[index_skew],1j*factor_skew)
    cvx_beta = cvxpy.Variable()
    cvx_sigma = np.eye(dimA*dimB)/(dimA*dimB) + cvx_beta * rho_vec
    constraints = [
        rho_ABk>>0,
        cvxpy.partial_trace(rho_ABk, [dimA*dimB, dimB**(kext-1)], 1)==cvx_sigma,
    ]
    prob = cvxpy.Problem(cvxpy.Maximize(cvx_beta), constraints)
    prob.solve()
    return cvx_beta.value

np_rng = np.random.default_rng()

case_dict = {
    2: [6,8,10],
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
    t0 = time.time()
    beta = get_symmetric_extension_boundary_hermitian_basis(dm_werner, (dim,dim), kext, use_boson=False)
    alpha_irrep = (beta/dm_norm)*dim/(beta/dm_norm+dim-1)
    alpha_analytical = (kext+dim*dim-dim)/(kext*dim+dim-1)
    tmp0 = time.time() - t0
    print(f'[d={dim},kext={kext}][{tmp0:.3f}s] alpha={alpha_irrep:.6f}, abs(error)={abs(alpha_analytical-alpha_irrep):.5g}')
    ret_list.append((alpha_irrep, alpha_analytical, tmp0))
# gtx3060
# [d=2,kext=6][0.771s] alpha=0.615385, abs(error)=1.4866e-08
# [d=2,kext=8][23.835s] alpha=0.588235, abs(error)=3.9728e-08
# [d=2,kext=8,use_boson][17.716s] alpha=0.588235, abs(error)=8.4524e-09
# [d=3,kext=2][0.166s] alpha=1.000000, abs(error)=4.4214e-10
# [d=3,kext=3][0.866s] alpha=0.818182, abs(error)=7.8243e-11
# [d=3,kext=4][6.692s] alpha=0.714286, abs(error)=4.7046e-09
## bad idea, Hermitian basis is not good in timing

'''
| $(d,k)$| QETLAB time (s)| hermitian basis (s)  | irrep time (s) | $\alpha$ |
| :-: | :-: | :-: | :-: | :-: |
| $(2,6)$ | 0.14 | 0.77 | 0.10 | 0.615 |
| $(2,8)$ | 0.19 | 23.84 | 0.16 | 0.588 |
| $(2,10)$ | 12.60 | NA | 0.16 | 0.571 |
| $(2,16)$ | NA | NA | 0.32 | 0.545 |
| $(2,32)$ | NA | NA | 3.18 | 0.523 |
| $(2,64)$ | NA | NA | 51.96 | 0.512 |
| $(3,3)$ | 0.62 | 0.87 | 0.51 | 0.818 |
| $(3,4)$ | 7.96 | 6.69 | 2.38 | 0.714 |
| $(3,5)$ | NA | NA | 11.56 | 0.647 |
| $(3,6)$ | NA | NA | 55.60 | 0.6 |
'''

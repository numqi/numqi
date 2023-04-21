import numpy as np
import functools
import sympy
import sympy.physics.quantum


def get_angular_momentum_op(j_double:int):
    # https://en.wikipedia.org/wiki/Clebsch%E2%80%93Gordan_coefficients
    assert j_double>=0
    if j_double==0:
        jx = jy = jz = np.zeros((1,1), dtype=np.float64)
    else:
        jz = np.diag((np.arange(j_double+1)[::-1]-j_double/2))
        tmp0 = np.arange(1, j_double+1)
        tmp1 = np.sqrt(tmp0 * tmp0[::-1])/2
        jx = np.diag(tmp1, 1) + np.diag(tmp1, -1)
        jy = np.diag(-1j*tmp1, 1) + np.diag(1j*tmp1, -1)
    return jx,jy,jz


def get_clebsch_gordan_coeffient(j1_double:int, j2_double:int):
    # https://en.wikipedia.org/wiki/Table_of_Clebsch%E2%80%93Gordan_coefficients
    j1_double = int(j1_double)
    j2_double = int(j2_double)
    assert j1_double>=0
    assert j2_double>=0
    ret = _get_clebsch_gordan_coeffient_cache(j1_double, j2_double)
    return ret


@functools.lru_cache
def _get_clebsch_gordan_coeffient_cache(j1_double:int, j2_double:int):
    # https://en.wikipedia.org/wiki/Table_of_Clebsch%E2%80%93Gordan_coefficients
    jmax_double = j1_double + j2_double
    jmin_double = abs(j1_double - j2_double)
    j1_sym = sympy.S(j1_double)/2
    j2_sym = sympy.S(j2_double)/2
    ret = []
    for j_double in range(jmin_double, jmax_double+1, 2):
        coeff = np.zeros((j_double+1, j1_double+1, j2_double+1), dtype=np.float64)
        j_sym = sympy.S(j_double)/2
        int_shift = (j1_double+j2_double-j_double)//2
        for n in range(j_double, -1, -1):
            for n1 in range(max(0, n+int_shift-j2_double), min(j1_double, n+int_shift)+1):
                n2 = n + int_shift - n1
                tmp0 = sympy.physics.quantum.cg.CG(j1_sym, -j1_sym+n1, j2_sym, -j2_sym+n2, j_sym, -j_sym+n)
                coeff[j_double-n, j1_double-n1, j2_double-n2] = float(tmp0.doit().evalf())
        ret.append((j_double, coeff))
    return ret

def get_irreducible_tensor_operator(S_double:int):
    # https://en.wikipedia.org/wiki/Tensor_operator
    # https://en.wikipedia.org/wiki/Spherical_basis
    # https://doi.org/10.1006/jmre.2001.2416
    # multiplication operator http://www.physics.umd.edu/grt/taj/623d/tops2.pdf
    assert S_double>=1
    cg_coeff = get_clebsch_gordan_coeffient(S_double, S_double)
    ret = []
    for k_double,coeff in cg_coeff:
        k = k_double//2 #k=0,1,2,...,2S
        tmp0 = np.sqrt(S_double+1)*(1 if (k % 2==0) else -1) * (1 - 2*(np.arange(S_double+1) % 2))
        # q = k,k-1,...,1-k,-k
        # m,mp = S,S-1,...,1-S,S
        ret.append(tmp0[:,np.newaxis]*(coeff[:,:,::-1]))
    return ret


def get_irreducible_hermitian_matrix_basis(S_double:int, tag_norm=False, tag_stack=False):
    # https://doi.org/10.1006/jmre.2001.2416
    T = get_irreducible_tensor_operator(S_double)
    cx = []
    cy = []
    cz = []
    factor = np.sqrt((S_double/2)*(S_double/2+1)/3)
    for k,T_i in enumerate(T):
        if k==0:
            cz.append(T_i[0] * factor)
        else:
            # q=1,2,...,k
            tmp0 = 1-2*(np.arange(1,k+1)%2)
            tmp_positive = T_i[:k][::-1] * tmp0.reshape(-1,1,1)
            tmp_negative = T_i[(k+1):]
            cx.append((tmp_negative+tmp_positive)*(factor/np.sqrt(2)))
            cy.append((tmp_negative-tmp_positive)*(1j*factor/np.sqrt(2)))
            cz.append(T_i[k]*factor)
    cx = np.concatenate(cx, axis=0)
    cy = np.concatenate(cy, axis=0)
    cz = np.stack(cz, axis=0)
    if tag_norm:
        tmp0 = 1 / np.sqrt((S_double/2)*(S_double/2+1)*(S_double+1)/3)
        cx *= tmp0
        cy *= tmp0
        cz *= tmp0
    #make identity the first term
    ret = np.concatenate([cz,cx,cy], axis=0) if tag_stack else (cz,cx,cy)
    return ret

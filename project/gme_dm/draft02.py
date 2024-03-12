import numpy as np
import cvxpy
from tqdm import tqdm

import numqi

np_rng = np.random.default_rng()

def test_flip_op():
    dimA = 3
    dimB = 4
    psi_AB = numqi.random.rand_haar_state(dimA*dimB).reshape(dimA, dimB)

    rdm = psi_AB @ psi_AB.conj().T
    purity = np.vdot(rdm.reshape(-1), rdm.reshape(-1)).real

    flip_op = np.eye(dimA*dimA).reshape(dimA,dimA,dimA,dimA).transpose(0,1,3,2)
    z0 = np.einsum(psi_AB, [0,1], psi_AB.conj(), [2,1], psi_AB, [3,4], psi_AB.conj(), [5,4], flip_op, [2,5,0,3], [], optimize=True).real
    assert abs(purity-z0)<1e-10


def get_linear_entropy_entanglement_ppt(rho:np.ndarray, dim:tuple[int], use_tqdm:bool=False, return_info:bool=False):
    # http://dx.doi.org/10.1103/PhysRevLett.114.160501
    rho,is_single_item,dimA,dimB,use_tqdm = numqi.entangle.symext._check_input_rho_SDP(rho, dim, use_tqdm)
    cvx_rho = cvxpy.Parameter((dimA*dimB,dimA*dimB), complex=True)
    ind_sym = np.arange(dimA*dimB*dimA*dimB, dtype=np.int64).reshape(dimA*dimB,-1).T.reshape(-1)
    cvxW = cvxpy.Variable((dimA*dimB*dimA*dimB,dimA*dimB*dimA*dimB), hermitian=True)
    # numqi.group.symext.get_sud_symmetric_irrep_basis() #TODO
    constraint = [
        cvxW==cvxW[ind_sym],
        cvxW>>0,
        cvxpy.partial_transpose(cvxW, [dimA*dimB,dimA*dimB], 1)>>0,
        cvxpy.partial_trace(cvxW, [dimA*dimB,dimA*dimB], 1)==cvx_rho,
        # cvxpy.partial_trace(cvxW, [dimA*dimB,dimA*dimB], 0)==cvx_rho,
    ]
    tmp0 = cvxpy.partial_trace(cvxW, [dimA,dimB,dimA,dimB], 3)
    tmp1 = cvxpy.partial_trace(tmp0, [dimA,dimB,dimA], 1)
    flip_op = np.eye(dimA*dimA).reshape(dimA,dimA,dimA,dimA).transpose(0,1,3,2)
    tmp2 = np.ascontiguousarray(flip_op.reshape(dimA*dimA,-1).T)
    obj = cvxpy.Maximize(cvxpy.real(cvxpy.sum(cvxpy.multiply(tmp1,tmp2))))
    prob = cvxpy.Problem(obj, constraint)
    ret = []
    for rho_i in (tqdm(rho) if use_tqdm else rho):
        cvx_rho.value = rho_i
        try:
            prob.solve()
            tmp0 = 1 - prob.value
        except cvxpy.error.SolverError: #sometimes error when fail to solve
            tmp0 = np.nan
        if return_info:
            tmp1 = np.ascontiguousarray(cvxW.value) if tmp0 else None
            ret.append((tmp0,tmp1))
        else:
            ret.append(tmp0)
    if not return_info:
        ret = np.array(ret)
    if is_single_item:
        ret = ret[0]
    return ret

def test_get_linear_entropy_entanglement_ppt():
    dimA = 3
    dimB = 3
    rho = numqi.state.get_bes3x3_Horodecki1997(np_rng.uniform(0,1))
    ret,matW = get_linear_entropy_entanglement_ppt(rho, (3,3), return_info=True)

    eps = 1e-7 # TODO, 1e-7 if use_MOSEK else 1e-4
    assert np.abs(matW - matW.T.conj()).max() < eps
    assert abs(np.trace(matW) - 1) < eps
    assert np.linalg.eigvalsh(matW)[0] > -eps
    tmp0 = matW.reshape(dimA*dimB,dimA*dimB,-1).transpose(1,0,2).reshape(dimA*dimB*dimA*dimB,-1)
    assert np.abs(tmp0 - matW).max() < eps
    tmp0 = matW.reshape(dimA*dimB,dimA*dimB,dimA*dimB,dimA*dimB)
    tmp1 = np.einsum(tmp0,[0,1,2,1],[0,2],optimize=True)
    assert np.abs(tmp1-rho).max() < eps
    tmp1 = np.einsum(tmp0,[0,1,0,2],[1,2],optimize=True)
    assert np.abs(tmp1-rho).max() < eps
    assert np.linalg.eigvalsh(tmp0.transpose(0,3,2,1).reshape(dimA*dimB*dimA*dimB,-1))[0]>-eps
    tmp0 = matW.reshape(dimA,dimB,dimA,dimB,dimA,dimB,dimA,dimB)
    tmp1 = np.einsum(tmp0, [0,1,2,3,4,1,6,3], [0,2,4,6], optimize=True).reshape(dimA*dimA,-1)
    flip_op = np.eye(dimA*dimA).reshape(dimA,dimA,dimA,dimA).transpose(0,1,3,2)
    tmp2 = 1-np.trace((flip_op.reshape(dimA*dimA,-1) @ tmp1)).real
    assert abs(tmp2-ret) < eps


rho = numqi.state.get_bes3x3_Horodecki1997(0.23)
ret = get_linear_entropy_entanglement_ppt(rho, (3,3))
# 0.0017232268486448987


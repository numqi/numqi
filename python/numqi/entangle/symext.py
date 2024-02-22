import functools
import numpy as np
import cvxpy
import torch
from tqdm.auto import tqdm

import numqi.dicke
import numqi.group
import numqi.gellmann

from .ppt import cvx_matrix_mlogx
from ._misc import _sdp_ree_solve, _check_input_rho_SDP


@functools.lru_cache
def get_symmetric_extension_index_list(dimA, dimB, kext, kind='2d'):
    assert kind in {'1d','2d'}
    tmp0 = np.arange(dimA*dimB**kext)
    ret = [tmp0.reshape(dimA*dimB**(kext-2), dimB, dimB).transpose(0,2,1).reshape(-1)]
    if kext>2:
        ret.append(np.transpose(tmp0.reshape((dimA,)+(dimB,)*kext), [0]+list(range(2,kext+1))+[1]).reshape(-1))
    if kind=='1d':
        ind_list = ret
        tmp0 = np.arange(dimA*dimA*dimB**(2*kext)).reshape(dimA*dimB**kext, -1)
        ret = []
        for ind0 in ind_list:
            ret.append(np.reshape(tmp0[ind0[:,np.newaxis], ind0].T, -1, order='F'))
    for x in ret:
        x.flags.writeable = False
    return ret


# dimA=2 dimB=3 kext=3 MOSEK: 8GB 10second
# dimA=3 dimB=3 kext=3 MOSEK: 12GB 50s
def is_ABk_symmetric_ext_naive(rho, dim, kext, index_kind='2d'):
    assert len(dim)==2
    assert index_kind in {'1d', '2d'} #2d is a little bit more faster
    dimA = int(dim[0])
    dimB = int(dim[1])
    assert (rho.ndim==2) and (rho.shape[0]==rho.shape[1]) and (rho.shape[0]==dimA*dimB)
    assert kext>=2
    indP_list = get_symmetric_extension_index_list(dimA, dimB, kext, kind=index_kind)
    cvxX = cvxpy.Variable((dimA*dimB**(kext),dimA*dimB**(kext)), hermitian=True)
    constraints = [
        cvxX>>0,
        cvxpy.trace(cvxX)==1,
        cvxpy.partial_trace(cvxX, (dimA*dimB,dimB**(kext-1)), 1)==np.asfortranarray(rho),
    ]
    if index_kind=='1d':
        tmp0 = cvxpy.reshape(cvxX, (dimA*dimA*dimB**(2*kext)), order='F')
        constraints += [cvxpy.reshape(tmp0[x],shape=cvxX.shape,order='F')==cvxX for x in indP_list]
    else:
        constraints += [(cvxX[indP[:,np.newaxis],indP]==cvxX) for indP in indP_list]
    obj = cvxpy.Minimize(1)
    prob = cvxpy.Problem(obj, constraints)
    prob.solve()

    if np.isinf(prob.value):
        ret = False, None
    else:
        ret = True, np.ascontiguousarray(cvxX.value)
    return ret


@functools.lru_cache
def get_cvxpy_transpose0213_indexing(N0, N1, N2=None, N3=None):
    if N2 is None:
        assert N3 is None
        N2 = N0
        N3 = N1
    ret = np.arange(N0*N1*N2*N3).reshape(N2,N3,N0,N1).transpose(3,1,2,0).reshape(-1)
    ret.flags.writeable = False
    return ret


def get_ABk_symmetric_extension_ree(rho, dim, kext, use_ppt=False, use_boson=False, return_info=False, sqrt_order=3, pade_order=3, use_tqdm=False):
    r'''get the relative entropy of entanglement of k-symmetric extension on B-party

    Parameters:
        rho (np.ndarray,list): density matrix, or list of density matrices (3d array)
        dim (tuple(int)): tuple of length 2, dimension of A-party and B-party
        kext (int): number of copies of symmetric extension
        use_ppt (bool): if True, use PPT (positive partial transpose) constraint
        use_boson (bool): if True, use bosonic symmetry
        return_info (bool): if True, return information of the SDP solver
        sqrt_order (int): the order of sqrtm approximation
        pade_order (int): the order of Pade approximation
        use_tqdm (bool): if True, use tqdm to show progress bar

    Returns:
        ret0 (float,np.array):  `ret` is a float indicates relative entropy of entanglement.
            If `rho` is list of density matrices, `ret` is a 1d `np.array` of float
        ret1 (list[dict]): if `return_info` is `True`, then `ret1` is a list of information of the SDP solver.
    '''
    assert kext>=1
    if kext==1:
        assert use_ppt, 'kext=1 with use_ppt=False is meaningless'
    rho,is_single_item,dimA,dimB,use_tqdm = _check_input_rho_SDP(rho, dim, use_tqdm)

    coeffB_list,multiplicity_list = numqi.group.symext.get_symmetric_extension_irrep_coeff(dimB, kext)
    if use_boson:
        assert numqi.dicke.get_dicke_number(kext, dimB)==coeffB_list[0].shape[0]
        coeffB_list = coeffB_list[:1]
        multiplicity_list = multiplicity_list[:1]
    dim_coeffB_list = [x.shape[0] for x in coeffB_list]

    cvxP_list = [cvxpy.Variable((x*dimA,x*dimA), hermitian=True) for x in dim_coeffB_list]
    index0213_list = [get_cvxpy_transpose0213_indexing(dimA,x) for x in dim_coeffB_list]
    index0213_ab = get_cvxpy_transpose0213_indexing(dimA,dimB)
    # TODO replace indexing-matmul with indexing-sum
    cvx_rdm_list = []
    #TODO trace(cvx_rdm) is not correct
    for ind0 in range(len(coeffB_list)):
        tmp0 = cvxP_list[ind0]
        tmp1 = index0213_list[ind0]
        tmp2 = dim_coeffB_list[ind0]**2
        tmp3 = cvxpy.reshape(cvxpy.reshape(tmp0, tmp0.size, order='F')[tmp1], (dimA*dimA,tmp2), order='F')
        cvx_rdm_list.append(tmp3 @ np.asfortranarray(coeffB_list[ind0].reshape(tmp2, dimB*dimB)))
    tmp0 = sum(cvx_rdm_list) #(dimA*dimA,dimB*dimB)
    cvx_rdm = cvxpy.reshape(cvxpy.reshape(tmp0, tmp0.size, order='F')[index0213_ab], (dimA*dimB,dimA*dimB), order='F')
    constraints = [x>>0 for x in cvxP_list]
    if use_ppt:
        constraints = +[cvxpy.partial_transpose(x, [dimA,x.shape[0]//dimA], [1])>>0 for x in cvxP_list]
    constraints += [sum(cvxpy.trace(x)*y for x,y in zip(cvxP_list,multiplicity_list))==1]
    cvxP, tmp0 = cvx_matrix_mlogx(cvx_rdm, sqrt_order=sqrt_order, pade_order=pade_order)
    constraints += tmp0
    # cvxP['X'] is cvxX
    cvx_rho = cvxpy.Parameter((dimA*dimB,dimA*dimB), hermitian=True)
    obj = cvxpy.Minimize(cvxpy.real(cvxpy.trace(cvx_rho @ cvxP['mlogX'])))
    prob = cvxpy.Problem(obj, constraints)
    ret = _sdp_ree_solve(rho, use_tqdm, cvx_rho, cvxP, prob, obj, return_info, is_single_item)
    return ret


def _ABk_symmetric_extension_setup(dimA, dimB, kext, use_boson, use_ppt, cvx_rho=None):
    coeffB_list,multiplicity_list = numqi.group.symext.get_symmetric_extension_irrep_coeff(dimB, kext)
    if use_boson:
        assert numqi.dicke.get_dicke_number(kext, dimB)==coeffB_list[0].shape[0]
        coeffB_list = coeffB_list[:1]
        multiplicity_list = multiplicity_list[:1]
    dim_coeffB_list = [x.shape[0] for x in coeffB_list]

    cvxP_list = [cvxpy.Variable((x*dimA,x*dimA), hermitian=True) for x in dim_coeffB_list]
    index0213_list = [get_cvxpy_transpose0213_indexing(dimA,x) for x in dim_coeffB_list]
    cvx_rdm_list = []
    for ind0 in range(len(coeffB_list)):
        tmp0 = cvxP_list[ind0]
        tmp1 = index0213_list[ind0]
        tmp2 = dim_coeffB_list[ind0]**2
        tmp3 = cvxpy.reshape(cvxpy.reshape(tmp0, tmp0.size, order='F')[tmp1], (dimA*dimA,tmp2), order='F')
        cvx_rdm_list.append(tmp3 @ np.asfortranarray(coeffB_list[ind0].reshape(tmp2, dimB*dimB)))
    cvx_rdm = sum(cvx_rdm_list) #(dimA,dimA,dimB,dimB)
    constraints = [x>>0 for x in cvxP_list]
    if use_ppt:
        constraints += [cvxpy.partial_transpose(x, [dimA,x.shape[0]//dimA], axis=1)>>0 for x in cvxP_list]
    constraints += [sum(cvxpy.trace(x)*y for x,y in zip(cvxP_list,multiplicity_list))==1]
    if cvx_rho is None:
        ret = cvxP_list, constraints, cvx_rdm
    else:
        constraints += [cvx_rdm==cvx_rho]
        ret = cvxP_list, constraints
    return ret


def is_ABk_symmetric_ext(rho, dim, kext, use_ppt=False, use_boson=False, use_tqdm=False, return_info=False):
    '''check if rho has symmetric extension of kext copies on B-party

    Parameters:
        rho (np.ndarray,list): density matrix, or list of density matrices (3d array)
        dim (tuple(int)): tuple of length 2, dimension of A-party and B-party
        kext (int): number of copies of symmetric extension
        use_ppt (bool): if True, use PPT (positive partial transpose) constraint
        use_boson (bool): if True, use bosonic symmetry
        use_tqdm (bool): if True, use tqdm to show progress bar
        return_info (bool): if True, return information of the SDP solver

    Returns:
        ret (bool): If `return_info=False` and rho is single density matrix, `ret` is a bool indicates if rho has symmetric extension.
            If `return_info=False` and rho is list of density matrices, `ret` is a 1d `np.array` of bool
            If `return_info=True` and `rho` is single density matrix, `ret` is a tuple of (bool, info) where
            `info` is a list of information of the SDP solver.
            If `return_info=True` and `rho` is list of density matrices, `ret` is a tuple of (np.array, info) where
            `info` is a dict of information of the SDP solver.
    '''
    rho,is_single_item,dimA,dimB,use_tqdm = _check_input_rho_SDP(rho, dim, use_tqdm)
    rho = rho.reshape(-1,dimA,dimB,dimA,dimB).transpose(0,1,3,2,4).reshape(-1,dimA*dimA,dimB*dimB)
    cvx_rho = cvxpy.Parameter((dimA*dimA,dimB*dimB), complex=True)
    cvxP_list, constraints = _ABk_symmetric_extension_setup(dimA, dimB, kext, use_boson, use_ppt, cvx_rho)
    prob = cvxpy.Problem(cvxpy.Minimize(1), constraints)
    ret = []
    for rho_i in (tqdm(rho) if use_tqdm else rho):
        cvx_rho.value = rho_i
        try:
            prob.solve()
            tmp0 = not np.isinf(prob.value)
        except cvxpy.error.SolverError: #seems error when fail to solve
            tmp0 = False
        if return_info:
            tmp1 = [np.ascontiguousarray(x.value) for x in cvxP_list] if tmp0 else None
            ret.append((tmp0,tmp1))
        else:
            ret.append(tmp0)
    if not return_info:
        ret = np.array(ret)
    if is_single_item:
        ret = ret[0]
    return ret



def get_ABk_extension_numerical_range(op_list, direction, dim, kext, use_ppt=False, use_boson=False, use_tqdm=True, return_info=False):
    r'''get the symmetric extension numerical range of a list of operators

    $$ \max\;\beta $$

    $$ s.t.\;\begin{cases} \rho_{AB^{k}}\succeq0\\ \mathrm{Tr}[\rho_{AB^{k}}]=1\\ P_{B_{i}B_{j}}\rho_{AB^{k}}P_{B_{i}B_{j}}=\rho_{AB^{k}}\\ \mathrm{Tr}\left[\mathrm{Tr}_{B^{k-1}}\left[\rho\right]A_{i}\right]=\beta\hat{n}_{i} \end{cases} $$

    Parameters:
        op_list (list): a list of operators, each operator is a 2d numpy array
        direction (np.ndarrray): the boundary along the direction will be calculated, if 2d, then each row is a direction
        dim (tuple[int]): the dimension of the density matrix, e.g. (2,2) for 2 qubits, must be of length 2
        kext (int): the number of copies of symmetric extension
        use_ppt (bool): if `True`, then use PPT (positive partial transpose) constraint in the pre-image
        use_boson (bool): if `True`, then use bosonic symmetrical extension
        return_info (bool): if `True`, then return the boundary and the boundary's normal vector
        use_tqdm (bool): if `True`, then use tqdm to show the progress

    Returns:
        beta (np.ndarray): the distance from the origin to the boundary along the direction.
            If `direction` is 2d, then `beta` is 1d array.
        boundary (np.ndarray): the boundary along the direction. only returned if `return_info` is `True`
        normal_vector (np.ndarray): the normal vector of the boundary. only returned if `return_info` is `True`
    '''
    op_list = np.stack(op_list, axis=0)
    num_op = op_list.shape[0]
    assert np.abs(op_list-op_list.transpose(0,2,1).conj()).max() < 1e-10, 'op_list must be Hermitian'
    direction = np.asarray(direction)
    assert (direction.ndim==1) or (direction.ndim==2)
    assert direction.shape[-1]==num_op
    is_single = (direction.ndim==1)
    direction = direction.reshape(-1,num_op)
    if direction.shape[0]==1:
        use_tqdm = False
    dimA,dimB = dim
    N0 = dimA*dimB
    cvx_vec = cvxpy.Parameter(num_op)
    cvx_beta = cvxpy.Variable()
    cvxP_list,constraints,cvx_rho = _ABk_symmetric_extension_setup(dimA, dimB, kext, use_boson, use_ppt)
    # cvx_rho is of shape (dimA*dimA,dimB*dimB)
    tmp0 = op_list.reshape(-1,dimA,dimB,dimA,dimB).transpose(0,4,2,3,1).reshape(-1,dimA*dimB*dimA*dimB, order='C')
    cvx_op = cvxpy.real(tmp0 @ cvxpy.reshape(cvx_rho, N0*N0, order='F'))
    constraints.append(cvx_beta*cvx_vec==cvx_op)
    cvx_obj = cvxpy.Maximize(cvx_beta)
    prob = cvxpy.Problem(cvx_obj, constraints)
    obj_list = []
    boundary_list = []
    norm_vec_list = []
    ret = []
    for vec_i in (tqdm(direction) if use_tqdm else direction):
        cvx_vec.value = vec_i
        prob.solve()
        obj_list.append(cvx_obj.value)
        if return_info:
            boundary_list.append(cvx_op.value.copy())
            norm_vec_list.append(constraints[-1].dual_value.copy())
    if is_single:
        if return_info:
            ret = (obj_list[0], boundary_list[0], norm_vec_list[0])
        else:
            ret = obj_list[0]
    else:
        obj_list = np.array(obj_list)
        if return_info:
            ret = obj_list, np.stack(boundary_list, axis=0), np.stack(norm_vec_list, axis=0)
        else:
            ret = obj_list
    return ret


def get_ABk_symmetric_extension_boundary(rho, dim, kext, use_ppt=False, use_boson=False, use_tqdm=False, return_info=False):
    '''get the boundary (in Euclidean space) of k-ext symmetric extension on B-party along rho direction

    Parameters:
        rho (np.ndarray,list): density matrix, or list of density matrices (3d array)
        dim (tuple(int)): tuple of length 2, dimension of A-party and B-party
        kext (int): number of copies of symmetric extension
        use_ppt (bool): if True, use PPT (positive partial transpose) constraint
        use_boson (bool): if True, use bosonic symmetry
        use_tqdm (bool): if True, use tqdm to show progress bar
        return_info (bool): if True, return information of the SDP solver

    Returns:
        beta (float,np.array):  `beta` is a float indicates Euclidean distance. if `rho` is list of density matrices, `beta` is a 1d `np.array` of float
        vecA (np.ndarray): `vecA` is a 1d `np.ndarray` of float indicates the position of the boundary. if `rho` is list of density matrices, `vecA` is a 2d `np.ndarray`
        vecN (np.ndarray): `vecN` is a 1d `np.ndarray` of float indicates the normal vector of the boundary. if `rho` is list of density matrices, `vecA` is a 2d `np.ndarray`
    '''
    # no need to be positive, only direction matters
    rho,is_single_item,dimA,dimB,use_tqdm = _check_input_rho_SDP(rho, dim, use_tqdm, tag_positive=False)
    dm_norm = numqi.gellmann.dm_to_gellmann_norm(rho)
    tmp0 = (rho - np.eye(dimA*dimB)/(dimA*dimB))/dm_norm.reshape(-1,1,1)
    rho_vec_list = tmp0.reshape(-1,dimA,dimB,dimA,dimB).transpose(0,1,3,2,4).reshape(-1,dimA*dimA,dimB*dimB)

    cvx_rho = cvxpy.Parameter((dimA*dimA,dimB*dimB), complex=True)
    cvx_beta = cvxpy.Variable()
    tmp0 = np.eye(dimA*dimB).reshape(dimA,dimB,dimA,dimB).transpose(0,2,1,3).reshape(dimA*dimA,dimB*dimB)/(dimA*dimB)
    cvx_sigma = tmp0 + cvx_beta * cvx_rho
    cvxP_list,constraints = _ABk_symmetric_extension_setup(dimA, dimB, kext, use_boson, use_ppt, cvx_sigma)
    prob = cvxpy.Problem(cvxpy.Maximize(cvx_beta), constraints)
    beta_list = []
    vecA_list = []
    vecN_list = []
    for ind0 in (tqdm(range(len(rho))) if use_tqdm else range(len(rho))):
        cvx_rho.value = rho_vec_list[ind0]
        prob.solve()
        beta = cvx_beta.value
        beta_list.append(beta)
        if return_info:
            tmp0 = numqi.gellmann.dm_to_gellmann_basis(rho[ind0])
            dual = np.ascontiguousarray(constraints[-1].dual_value).reshape(dimA,dimA,dimB,dimB).transpose(0,2,1,3).reshape(dimA*dimB,dimA*dimB)
            tmp1 = numqi.gellmann.dm_to_gellmann_basis(dual + dual.T.conj())
            vecA_list.append((beta/np.linalg.norm(tmp0))*tmp0)
            vecN_list.append(-tmp1/np.linalg.norm(tmp1))
    if is_single_item:
        ret = beta_list[0] if (not return_info) else (beta_list[0], vecA_list[0], vecN_list[0])
    else:
        tmp0 = np.array(beta_list)
        ret = tmp0 if (not return_info) else (tmp0, np.stack(vecA_list,axis=0), np.stack(vecN_list,axis=0))
    return ret


class SymmetricExtABkIrrepModel(torch.nn.Module):
    def __init__(self, dimA:int, dimB:int, kext:int):
        super().__init__()
        assert dimA>=2
        self.dimA = int(dimA)
        self.dimB = int(dimB)
        self.kext = int(kext)
        coeffB_list,multiplicity_list = numqi.group.symext.get_symmetric_extension_irrep_coeff(dimB, kext)
        multiplicity_list = np.array(multiplicity_list, dtype=np.float64)
        self.coeffB_list = [torch.tensor(x.reshape(-1,self.dimB**2), dtype=torch.complex128) for x in coeffB_list] #TODO complex128?
        self.manifold_psd = torch.nn.ModuleList([numqi.manifold.Trace1PSD(self.dimA*x.shape[0], method='cholesky', dtype=torch.complex128) for x in coeffB_list])
        self.manifold_prob = numqi.manifold.DiscreteProbability(len(self.coeffB_list), method='softmax', weight=multiplicity_list, dtype=torch.float64)

        self.dm_target_transpose = None
        self.rhoAB_transpose = None

    def set_dm_target(self, rhoAB, zero_eps=1e-7):
        assert (rhoAB.ndim==2) and (rhoAB.shape[0]==rhoAB.shape[1]) and (rhoAB.shape[0]==self.dimA*self.dimB)
        assert (abs(np.trace(rhoAB)-1)<zero_eps) and (np.abs(rhoAB-rhoAB.T.conj()).max()<zero_eps)
        assert (np.linalg.eigvalsh(rhoAB)[0] + zero_eps) > 0
        tmp0 = rhoAB.reshape(self.dimA, self.dimB, self.dimA, -1).transpose(0,2,1,3).reshape(self.dimA**2,-1)
        self.dm_target_transpose = torch.tensor(tmp0, dtype=torch.complex128)

    def forward(self):
        assert self.dm_target_transpose is not None
        rhoAB_list = []
        for ind0 in range(len(self.coeffB_list)):
            tmp0 = self.manifold_psd[ind0]()
            tmp1 = tmp0.reshape(self.dimA, tmp0.shape[0]//self.dimA, self.dimA, -1)
            rhoAB_list.append(tmp1.transpose(1,2).reshape(self.dimA**2,-1) @ self.coeffB_list[ind0])
        tmp1 = self.manifold_prob()
        rhoAB_transpose = sum([rhoAB_list[x]*tmp1[x] for x in range(len(rhoAB_list))])
        self.rhoAB_transpose = rhoAB_transpose
        tmp0 = (rhoAB_transpose-self.dm_target_transpose).reshape(-1)
        loss = torch.dot(tmp0.conj(), tmp0).real
        return loss

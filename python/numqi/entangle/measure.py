import numpy as np
import torch
import opt_einsum
import cvxpy
from tqdm.auto import tqdm

import numqi.manifold

from .symext import _check_input_rho_SDP
from .eof import get_concurrence_2qubit


def get_relative_entropy_of_entanglement_pure(psi:np.ndarray, dimA:int, dimB:int, return_SEP:bool=False, zero_eps:float=1e-10)->float:
    r'''get the relative entropy of entanglement of pure state

    reference: Closest separable state when measured by a quasi-relative entropy
    [arxiv-link](https://arxiv.org/abs/2009.04982)

    Parameters:
        psi (np.ndarray): the pure state of shape (dimA*dimB,)
        dimA (int): the dimension of subsystem A
        dimB (int): the dimension of subsystem B
        return_SEP (bool): return the separable state or not
        zero_eps (float): the zero threshold

    Returns:
        ret (float): the relative entropy of entanglement
        sigma (np.ndarray): the separable state, if `return_SEP=True`
    '''
    psi = psi.reshape(dimA, dimB)
    if dimA<=dimB:
        rdm = psi @ psi.T.conj()
    else:
        rdm = psi.T.conj() @ psi
    EVL = np.maximum(np.linalg.eigvalsh(rdm), 0)
    assert abs(np.sum(EVL)-1) < (zero_eps*1e3)
    tmp0 = EVL[EVL>zero_eps]
    ret = -np.dot(tmp0, np.log(tmp0))
    if return_SEP:
        U,S,V = np.linalg.svd(psi.reshape(dimA, dimB), full_matrices=False)
        sigma = np.einsum(U, [1,0], U.conj(), [2,0], S**2, [0], V, [0,3], V.conj(), [0,4], [1,3,2,4], optimize=True).reshape(dimA*dimB, dimA*dimB)
        ret = ret,sigma
    return ret


def get_gme_2qubit(rho:np.ndarray):
    r'''Calculate the geometric measure of entanglement (GME) for 2-qubit density matrix.

    Geometric measure of entanglement and applications to bipartite and multipartite quantum states
    [doi-link](https://doi.org/10.1103/PhysRevA.68.042307) (eq-10)

    Parameters:
        rho (np.ndarray): 2-qubit density matrix.

    Returns:
        ret (float): GME.
    '''
    assert rho.shape==(4,4)
    tmp0 = get_concurrence_2qubit(rho)
    ret = (1-np.sqrt(1-tmp0*tmp0)) / 2
    return ret


class DensityMatrixGMEModel(torch.nn.Module):
    r'''Solve geometric measure of entanglement (GME) for density matrix using gradient descent.'''
    def __init__(self, dim_list:tuple[int,...], num_ensemble:int, rank:int|None=None, CPrank:int=1, dtype:str='float64'):
        r'''Initialize the model.

        Parameters:
            dim_list (tuple[int]): dimension of the density matrix.
            num_ensemble (int): number of ensemble to sample.
            rank (int): rank of the density matrix, if None, then rank is set to the maximum.
            CPrank (int): Canonical Polyadic rank rank of the state.
            dtype (str): data type of the state.
        '''
        super().__init__()
        assert dtype in {'float32','float64'}
        self.cdtype = torch.complex64 if dtype=='float32' else torch.complex128
        self.dtype = torch.float32 if dtype=='float32' else torch.float64
        dim_list = tuple(int(x) for x in dim_list)
        assert (len(dim_list)>=2) and all(x>=2 for x in dim_list)
        self.dim_list = dim_list
        N0 = np.prod(np.array(dim_list))
        self.num_ensemble = int(num_ensemble)
        if rank is None:
            rank = N0
        assert rank<=N0
        self.rank = int(rank)
        assert CPrank>=1
        self.CPrank = int(CPrank)

        self.manifold_stiefel = numqi.manifold.Stiefel(num_ensemble, rank, dtype=self.cdtype)
        self.manifold_psi = torch.nn.ModuleList([numqi.manifold.Sphere(x, batch_size=num_ensemble*CPrank, dtype=self.cdtype) for x in dim_list])
        if CPrank>1:
            self.manifold_coeff = numqi.manifold.PositiveReal(num_ensemble*CPrank, dtype=torch.float64)
            N1 = len(dim_list)
            tmp0 = [(num_ensemble,rank),(num_ensemble,rank)] + [(num_ensemble,rank,x) for x in dim_list] + [(num_ensemble,rank,x) for x in dim_list]
            tmp1 = [(N1,N1+1),(N1,N1+2)] + [(N1,N1+1,x) for x in range(N1)] + [(N1,N1+2,x) for x in range(N1)]
            self.contract_psi_psi = opt_einsum.contract_expression(*[y for x in zip(tmp0,tmp1) for y in x], [N1])

        self._sqrt_rho = None
        self.contract_expr = None
        self.contract_coeff = None

    def set_density_matrix(self, rho:np.ndarray):
        r'''Set the density matrix.

        Parameters:
            rho (np.ndarray): density matrix.
        '''
        N0 = np.prod(np.array(self.dim_list))
        assert rho.shape == (N0, N0)
        assert np.abs(rho-rho.T.conj()).max() < 1e-10
        EVL,EVC = np.linalg.eigh(rho)
        EVL = np.maximum(0, EVL[-self.rank:])
        assert abs(EVL.sum()-1) < 1e-10
        EVC = EVC[:,-self.rank:]
        tmp0 = (EVC * np.sqrt(EVL)).reshape(*self.dim_list, self.rank)
        self._sqrt_rho = torch.tensor(tmp0, dtype=self.cdtype)
        N1 = len(self.dim_list)
        if self.CPrank==1:
            tmp0 = [(N1+1,x) for x in range(N1)]
            tmp1 = [(self.num_ensemble,x) for x in self.dim_list]
            tmp2 = [y for x in zip(tmp1,tmp0) for y in x]
            self.contract_expr = opt_einsum.contract_expression(self._sqrt_rho, tuple(range(N1+1)),
                                [self.num_ensemble,self.rank], (N1+1,N1), *tmp2, (N1+1,), constants=[0])
        else:
            tmp0 = [((N1+1,N1+2))] + [(N1+1,N1+2,x) for x in range(N1)]
            tmp1 = [(self.num_ensemble,self.CPrank)] + [(self.num_ensemble,self.CPrank,x) for x in self.dim_list]
            tmp2 = [y for x in zip(tmp1,tmp0) for y in x]
            self.contract_expr = opt_einsum.contract_expression(self._sqrt_rho, tuple(range(N1+1)),
                                [self.num_ensemble,self.rank], (N1+1,N1), *tmp2, (N1+1,), constants=[0])

    def get_state(self, tag_grad:bool=False):
        with torch.set_grad_enabled(tag_grad):
            matX = self.manifold_stiefel()
            psi_list = [x() for x in self.manifold_psi]
            if self.CPrank>1:
                coeff = self.manifold_coeff().reshape(self.num_ensemble, self.CPrank).to(psi_list[0].dtype)
                psi_list = [x.reshape(self.num_ensemble,self.CPrank,-1) for x in psi_list]
                psi_conj_list = [x.conj().resolve_conj() for x in psi_list]
                psi_psi = self.contract_psi_psi(coeff, coeff, *psi_list, *psi_conj_list).real
                coeff = coeff / torch.sqrt(psi_psi).reshape(-1,1)
                ret = matX,psi_list,coeff
            else:
                ret = matX,psi_list
        return ret

    def forward(self):
        if self.CPrank>1:
            matX,psi_list,coeff = self.get_state(tag_grad=True)
            tmp2 = self.contract_expr(matX, coeff, *psi_list, backend='torch')
        else:
            matX,psi_list = self.get_state(tag_grad=True)
            tmp2 = self.contract_expr(matX, *psi_list, backend='torch')
        loss = 1-torch.vdot(tmp2,tmp2).real
        return loss


def get_linear_entropy_entanglement_ppt(rho:np.ndarray, dim:tuple[int], use_tqdm:bool=False, return_info:bool=False):
    r'''Calculate the linear entropy of entanglement for density matrix using PPT approximation.

    Evaluating Convex Roof Entanglement Measures
    [doi-link](http://dx.doi.org/10.1103/PhysRevLett.114.160501)

    Parameters:
        rho (np.ndarray): density matrix. support batch
        dim (tuple[int]): dimension of the density matrix. must be length 2.
        use_tqdm (bool): use tqdm for progress bar.
        return_info (bool): return additional information.

    Returns:
        ret (float,np.ndarray,list): linear entropy of entanglement.
    '''
    rho,is_single_item,dimA,dimB,use_tqdm = _check_input_rho_SDP(rho, dim, use_tqdm)
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
            tmp0 = max(0, 1 - prob.value)
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


class DensityMatrixLinearEntropyModel(torch.nn.Module):
    r'''Solve linear entropy of entanglement for density matrix using gradient descent.'''
    def __init__(self, dim:tuple[int,...], num_ensemble:int, rank:int|None=None, kind:str='convex', method:str='polar'):
        r'''Initialize the model.

        Parameters:
            dim (tuple[int]): dimension of the density matrix. must be length 2.
            num_ensemble (int): number of ensemble to sample.
            rank (int): rank of the density matrix, if None, then rank is set to the maximum.
            kind (str): convex or concave.
            method (str): parameterization method for Stiefel manifold.
        '''
        super().__init__()
        assert kind in {'convex','concave'}
        self.dtype = torch.float64
        self.cdtype = torch.complex128
        dim = tuple(int(x) for x in dim)
        assert (len(dim)==2) and all(x>=2 for x in dim)
        self.dim0 = dim[0]
        self.dim1 = dim[1]
        self.num_ensemble = int(num_ensemble)
        if rank is None:
            rank = dim[0]*dim[1]
        assert rank<=dim[0]*dim[1]
        self.rank = int(rank)
        self.manifold_stiefel = numqi.manifold.Stiefel(num_ensemble, rank, dtype=self.cdtype, method=method)

        self._sqrt_rho = None
        self._eps = torch.tensor(torch.finfo(self.dtype).eps, dtype=self.dtype)
        self._sign = 1 if (kind=='convex') else -1

    def set_density_matrix(self, rho:np.ndarray):
        r'''Set the density matrix.

        Parameters:
            rho (np.ndarray): density matrix.
        '''
        assert rho.shape==(self.dim0*self.dim1, self.dim0*self.dim1)
        assert np.abs(rho-rho.T.conj()).max() < 1e-10
        EVL,EVC = np.linalg.eigh(rho)
        EVL = np.maximum(0, EVL[-self.rank:])
        assert abs(EVL.sum()-1) < 1e-10
        EVC = EVC[:,-self.rank:]
        tmp0 = (EVC * np.sqrt(EVL)).reshape(self.dim0, self.dim1, -1)
        self._sqrt_rho = torch.tensor(tmp0, dtype=self.cdtype)
        tmp0 = self._sqrt_rho.conj().resolve_conj()
        if self.dim0<=self.dim1:
            self.contract_expr = opt_einsum.contract_expression(self._sqrt_rho, [0,3,4], tmp0, [1,3,5],
                                [self.num_ensemble,self.rank], [2,4], [self.num_ensemble,self.rank], [2,5], [2,0,1], constants=[0,1])
        else:
            self.contract_expr = opt_einsum.contract_expression(self._sqrt_rho, [3,0,4], tmp0, [3,1,5],
                                [self.num_ensemble,self.rank], [2,4], [self.num_ensemble,self.rank], [2,5], [2,0,1], constants=[0,1])
        tmp0 = min(self.dim0, self.dim1)
        self.contract_expr1 = opt_einsum.contract_expression([self.num_ensemble,tmp0,tmp0], [0,1,2], [self.num_ensemble,tmp0,tmp0], [0,1,2], [0])

    def forward(self):
        mat_st = self.manifold_stiefel()
        tmp0 = self.contract_expr(mat_st, mat_st.conj(), backend='torch')
        # rdm = tmp0 / torch.einsum(tmp0, [0,1,1], [0]).reshape(-1,1,1)
        tmp1 = torch.maximum(self._eps, torch.einsum(tmp0, [0,1,1], [0]).real)
        loss = self._sign * (1 - (self.contract_expr1(tmp0, tmp0.conj()).real / tmp1).sum())
        return loss

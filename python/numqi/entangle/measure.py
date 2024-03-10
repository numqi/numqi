import numpy as np
import torch
import opt_einsum

import numqi.manifold

from .eof import get_concurrence_2qubit


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
    def __init__(self, dim_list:tuple[int], num_ensemble:int, rank:int, CPrank:int=1, dtype:str='float64'):
        r'''Initialize the model.

        Parameters:
            dim_list (tuple[int]): dimension of the density matrix.
            num_ensemble (int): number of ensemble to sample.
            rank (int): rank of the density matrix.
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

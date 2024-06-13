import numpy as np
import torch

import numqi._torch_op
import numqi.manifold


def get_binegativity(rho:np.ndarray|torch.Tensor, dimA:int, dimB:int):
    '''calculate bi-negativity

    reference: Entanglement Cost under Positive-Partial-Transpose-Preserving Operations
    [doi-link](https://doi.org/10.1103/PhysRevLett.90.027901)

    Parameters:
        rho (numpy.ndarray,torch.Tensor): density matrix, shape=(`dimA`*`dimB`,`dimA`*`dimB`)
        dimA (int): dimension of A
        dimB (int): dimension of B

    Returns:
        ret (numpy.ndarray): bi-negativity, shape=(`dimA`*`dimB`,`dimA`*`dimB`)
    '''
    assert rho.shape==(dimA*dimB, dimA*dimB)
    if isinstance(rho, torch.Tensor):
        rho_pt = rho.reshape(dimA,dimB,dimA,dimB).transpose(1,3).reshape(dimA*dimB,dimA*dimB)
        tmp1 = numqi._torch_op.PSDMatrixSqrtm.apply(rho_pt @ rho_pt)
        ret = tmp1.reshape(dimA,dimB,dimA,dimB).transpose(1,3).reshape(dimA*dimB,dimA*dimB)
    else:
        rho_pt = rho.reshape(dimA,dimB,dimA,dimB).transpose(0,3,2,1).reshape(dimA*dimB,dimA*dimB)
        EVL,EVC = np.linalg.eigh(rho_pt)
        tmp1 = (EVC*np.abs(EVL)) @ EVC.T.conj()
        ret = tmp1.reshape(dimA,dimB,dimA,dimB).transpose(0,3,2,1).reshape(dimA*dimB,dimA*dimB)
    return ret


class SearchMinimumBinegativityModel(torch.nn.Module):
    def __init__(self, dimA:int, dimB:int):
        r'''search the minimum of binegativity

        see also: get_binegativity

        Parameters:
            dimA (int): dimension of A
            dimB (int): dimension of B
        '''
        super().__init__()
        assert (dimA>=2) and (dimB>=2)
        self.manifold_rho = numqi.manifold.Trace1PSD(dimA*dimB, method='cholesky', requires_grad=True, dtype=torch.complex128)
        self.manifold_psi = numqi.manifold.Sphere(dimA*dimB, requires_grad=True, dtype=torch.complex128)
        self.dimA = dimA
        self.dimB = dimB

    def forward(self):
        rho = self.manifold_rho()
        rho_bineg = get_binegativity(rho, self.dimA, self.dimB)
        psi = self.manifold_psi()
        loss = torch.vdot(psi, rho_bineg @ psi).real
        return loss


def get_PPT_entanglement_cost_bound(rho:np.ndarray, dimA:int, dimB:int):
    r'''calculate the lower and upper bound of PPT entanglement cost

    reference: Entanglement Cost under Positive-Partial-Transpose-Preserving Operations
    [doi-link](https://doi.org/10.1103/PhysRevLett.90.027901)

    Parameters:
        rho (numpy.ndarray): density matrix, shape=(`dimA`*`dimB`,`dimA`*`dimB`)
        dimA (int): dimension of A
        dimB (int): dimension of B

    Returns:
        lower_bound (float): lower bound of PPT entanglement cost
        upper_bound (float): upper bound of PPT entanglement cost
    '''
    assert rho.shape==(dimA*dimB, dimA*dimB)
    rho_pt = rho.reshape(dimA,dimB,dimA,dimB).transpose(0,3,2,1).reshape(dimA*dimB,dimA*dimB)
    EVL,EVC = np.linalg.eigh(rho_pt)
    tmp1 = (EVC*np.abs(EVL)) @ EVC.T.conj()
    rho_binegativity = tmp1.reshape(dimA,dimB,dimA,dimB).transpose(0,3,2,1).reshape(dimA*dimB,dimA*dimB)
    rho_pt_trace_norm = np.abs(EVL).sum()
    lower_bound = np.log2(rho_pt_trace_norm)
    upper_bound = np.log2(rho_pt_trace_norm + dimA*dimB*max(0, -np.linalg.eigvalsh(rho_binegativity)[0]))
    return lower_bound, upper_bound

import numpy as np
import torch

from ._internal import DiscreteProbability, Trace1PSD, Sphere, SpecialOrthogonal
from ._stiefel import Stiefel

_CPU = torch.device('cpu')

def quantum_state(dim, batch_size:(int|None)=None, method:str='quotient',
                    requires_grad:bool=True, dtype=torch.complex128, device:torch.device=_CPU):
    r'''manifold of quantum state, wrapper of numqi.manifold.Sphere

    Parameters:
        dim (int): dimension of the quantum state
        batch_size (int|None): batch size of quantum state
        method (str): method to represent quantum state. 'quotient' or 'coordinates'
        requires_grad (bool): whether to require gradient
        dtype (torch.dtype): data type of quantum state
        device (torch.device): device of quantum state

    Returns:
        ret (numqi.manifold.Sphere): manifold of quantum state.
    '''
    assert dtype in {torch.complex64,torch.complex128}
    ret = Sphere(dim, batch_size, method, requires_grad, dtype, device)
    return ret



def density_matrix(dim:int, rank:(int|None)=None, batch_size:(int|None)=None, method:str='cholesky',
            requires_grad:bool=True, dtype:torch.dtype=torch.complex128, device:torch.device=_CPU):
    r'''manifold of density matrix, wrapper of numqi.manifold.Trace1PSD

    Parameters:
        dim (int): dimension of the density matrix
        rank (int|None): rank of the density matrix
        batch_size (int|None): batch size of density matrix
        method (str): method to represent density matrix, 'cholesky' or 'ensemble'
        requires_grad (bool): whether to require gradient
        dtype (torch.dtype): data type of density matrix
        device (torch.device): device of density matrix

    Returns:
        ret (numqi.manifold.Trace1PSD): manifold of density matrix.
    '''
    assert dtype in {torch.complex64,torch.complex128}
    ret = Trace1PSD(dim, rank, batch_size, method, requires_grad, dtype, device)
    return ret



class SeparableDensityMatrix(torch.nn.Module):
    def __init__(self, dimA:int, dimB:int, num_cha:(int|None)=None, batch_size:(int|None)=None,
                 requires_grad:bool=True, dtype:torch.dtype=torch.complex128, device:torch.device=_CPU):
        r'''manifold of separable density matrix

        Parameters:
            dimA (int): dimension of the first subsystem
            dimB (int): dimension of the second subsystem
            num_cha (int|None): number of product states
            batch_size (int|None): batch size of density matrix
            requires_grad (bool): whether to require gradient
            dtype (torch.dtype): data type of density matrix
            device (torch.device): device of density matrix
        '''
        super().__init__()
        if num_cha is None:
            num_cha = 2*dimA*dimB
        tmp0 = torch.float32 if (dtype==[torch.complex64,torch.float32]) else torch.float64
        self.manifold_p = DiscreteProbability(num_cha, batch_size, 'softmax', requires_grad=requires_grad, dtype=tmp0, device=device)
        tmp0 = num_cha if (batch_size is None) else (batch_size*num_cha)
        self.manifold_psiA = Sphere(dimA, tmp0, 'quotient', requires_grad, dtype=dtype, device=device)
        self.manifold_psiB = Sphere(dimB, tmp0, 'quotient', requires_grad, dtype=dtype, device=device)
        self.dimA = dimA
        self.batch_size = batch_size
        self.dimB = dimB
        self.num_cha = num_cha

    def forward(self):
        prob = self.manifold_p()
        psiA = self.manifold_psiA()
        psiB = self.manifold_psiB()
        if self.batch_size is None:
            psiA_conj = psiA.conj()
            psiB_conj = psiB.conj()
            ret = torch.einsum(prob, [0], psiA, [0,1], psiA_conj, [0,3], psiB, [0,2], psiB_conj, [0,4], [1,2,3,4])
        else:
            psiA = psiA.reshape(self.batch_size, self.num_cha, self.dimA)
            psiB = psiB.reshape(self.batch_size, self.num_cha, self.dimB)
            psiA_conj = psiA.conj()
            psiB_conj = psiB.conj()
            ret = torch.einsum(prob, [5,0], psiA, [5,0,1], psiA_conj, [5,0,3], psiB, [5,0,2], psiB_conj, [5,0,4], [5,1,2,3,4])
        return ret


def quantum_gate(dim:int, batch_size:(int|None)=None, method:str='exp', cayley_order:int=2,
                    requires_grad:bool=True, dtype:torch.dtype=torch.complex128, device:torch.device=_CPU):
    r'''manifold of quantum gate, wrapper of numqi.manifold.SpecialOrthogonal

    Parameters:
        dim (int): dimension of the quantum gate
        batch_size (int|None): batch size of quantum gate
        method (str): method to represent quantum gate, 'exp' or 'cayley'
        cayley_order (int): order of Cayley transform
        requires_grad (bool): whether to require gradient
        dtype (torch.dtype): data type of quantum gate
        device (torch.device): device of quantum gate

    Returns:
        ret (numqi.manifold.SpecialOrthogonal): manifold of quantum gate.
    '''
    assert dtype in {torch.complex64,torch.complex128}
    ret = SpecialOrthogonal(dim, batch_size, method, cayley_order, requires_grad, dtype, device)
    return ret


class QuantumChannel(torch.nn.Module):
    def __init__(self, dim_in:int, dim_out:int, choi_rank:(int|None)=None, batch_size:(int|None)=None, method:str='qr',
                return_kind:str='kraus', requires_grad:bool=True, dtype:torch.dtype=torch.complex128, device:torch.device=_CPU):
        r'''manifold of quantum channel, wrapper of numqi.manifold.Stiefel

        Parameters:
            dim_in (int): dimension of the input quantum state
            dim_out (int): dimension of the output quantum state
            choi_rank (int|None): rank of the Choi matrix
            batch_size (int|None): batch size of quantum channel
            method (str): method to represent quantum channel, choleskyL / qr / polar / so-exp / so-cayley
            return_kind (str): return kind of quantum channel, 'kraus' or 'choi'
            requires_grad (bool): whether to require gradient
            dtype (torch.dtype): data type of quantum channel
            device (torch.device): device of quantum channel
        '''
        super().__init__()
        # TODO set polar as default, see chatgpt discussion
        if choi_rank is None:
            choi_rank = dim_in*dim_out
        assert dtype in {torch.complex64,torch.complex128}
        assert return_kind in {'kraus','choi'}
        self.manifold = Stiefel(choi_rank*dim_out, dim_in, batch_size, method, requires_grad, dtype, device)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.choi_rank = choi_rank
        self.batch_size = batch_size
        self.return_kind = return_kind

    def forward(self):
        mat = self.manifold()
        if self.batch_size is None:
            ret = mat.reshape(self.choi_rank, self.dim_out, self.dim_in)
            if self.return_kind=='choi':
                ret = torch.einsum(ret, [0,1,2], ret.conj(), [0,3,4], [1,2,3,4]) #(dim_out, dim_in, dim_out, dim_in)
        else:
            ret = mat.reshape(-1, self.choi_rank, self.dim_out, self.dim_in)
            if self.return_kind=='choi':
                ret = torch.einsum(ret, [5,0,1,2], ret.conj(), [5,0,3,4], [5,1,2,3,4]) #(dim_out, dim_in, dim_out, dim_in)
        return ret

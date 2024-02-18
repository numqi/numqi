import numpy as np
import torch

from ._internal import DiscreteProbability, Trace1PSD, Sphere, SpecialOrthogonal, Stiefel

_CPU = torch.device('cpu')

def quantum_state(dim, batch_size:(int|None)=None, method:str='quotient',
                    requires_grad:bool=True, dtype=torch.complex128, device:torch.device=_CPU):
    assert dtype in {torch.complex64,torch.complex128}
    ret = Sphere(dim, batch_size, method, requires_grad, dtype, device)
    return ret



def density_matrix(dim:int, rank:(int|None)=None, batch_size:(int|None)=None, method:str='cholesky',
            requires_grad:bool=True, dtype:torch.dtype=torch.complex128, device:torch.device=_CPU):
    assert dtype in {torch.complex64,torch.complex128}
    ret = Trace1PSD(dim, rank, batch_size, method, requires_grad, dtype, device)
    return ret



class SeparableDensityMatrix(torch.nn.Module):
    def __init__(self, dimA:int, dimB:int, num_cha:(int|None)=None, batch_size:(int|None)=None,
                 requires_grad:bool=True, dtype:torch.dtype=torch.complex128, device:torch.device=_CPU):
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
    assert dtype in {torch.complex64,torch.complex128}
    ret = SpecialOrthogonal(dim, batch_size, method, cayley_order, requires_grad, dtype, device)
    return ret


class QuantumChannel(torch.nn.Module):
    def __init__(self, dim_in:int, dim_out:int, choi_rank:(int|None)=None, batch_size:(int|None)=None, method:str='qr',
                return_kind:str='kraus', requires_grad:bool=True, dtype:torch.dtype=torch.complex128, device:torch.device=_CPU):
        super().__init__()
        # sqrtm seems to be better than qr TODO
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

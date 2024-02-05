import numpy as np
import torch
import opt_einsum

import numqi.manifold


class ChannelCapacity1InfModel(torch.nn.Module):
    '''
    Capacities of Quantum Channels and How to Find Them

    http://arxiv.org/abs/quant-ph/0304102v1
    '''
    def __init__(self, dim_in:int, num_state:int):
        super().__init__()
        self.manifold_prob = numqi.manifold.DiscreteProbability(num_state, dtype=torch.float64)
        self.manifold_psi = numqi.manifold.Sphere(dim_in, batch_size=num_state, dtype=torch.complex128)
        self.kop = None
        self.kop_conj = None
        self.contract0 = None
        self.contract1 = None

    def set_channel_kraus_op(self, kop):
        num_state = self.manifold_psi.batch_size
        dim_in = self.manifold_psi.dim
        assert (kop.ndim==3) and (kop.shape[2]==dim_in)
        tmp0 = np.einsum(kop, [0,1,2], kop.conj(), [0,1,3], [2,3], optimize=True)
        assert np.abs(tmp0-np.eye(tmp0.shape[0])).max() < 1e-10
        self.kop = torch.tensor(kop, dtype=torch.complex128)
        self.kop_conj = self.kop.conj().resolve_conj()
        self.contract0 = opt_einsum.contract_expression(kop.shape, [0,1,2], kop.shape, [0,3,4], [dim_in,dim_in], [2,4], [1,3])
        self.contract1 = opt_einsum.contract_expression(kop.shape, [0,1,2], kop.shape, [0,3,4], [num_state,dim_in], [5,2], [num_state,dim_in], [5,4], [5,1,3])

    def forward(self):
        prob = self.manifold_prob()
        state = self.manifold_psi()
        rho = (state.T * prob) @ state.conj()
        rho_out = self.contract0(self.kop, self.kop_conj, rho)
        tmp0 = self.contract1(self.kop, self.kop_conj, state, state.conj())
        tmp1 = numqi.utils.get_von_neumann_entropy(rho_out)
        tmp2 = numqi.utils.get_von_neumann_entropy(tmp0)
        ret = -(tmp1 - torch.dot(prob, tmp2))
        # the capacity is to maximize the output entropy, which is equivalent to minimize the negative output entropy
        return ret

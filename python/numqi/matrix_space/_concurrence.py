import numpy as np
import torch
import opt_einsum

import numqi.manifold

from ._public import _get_nontrivial_subset_list

class SubspaceGeneralizedConcurrenceModel(torch.nn.Module):
    # http://doi.org/10.1088/1402-4896/acec15 eq(51) eq(57)
    def __init__(self, dim_subspace:int):
        super().__init__()
        assert dim_subspace>=2
        self.dim_subspace = dim_subspace
        self.manifold = numqi.manifold.Sphere(dim_subspace, dtype=torch.complex128)

        # set in self.set_subspace_basis()
        self.num_partite = None
        self.subset_tensor_list = None
        self.tensor5_contract_expr = None

    def set_subspace_basis(self, np0:np.ndarray, use_tensor5:bool|None=False, zero_eps:float=1e-10):
        N1 = self.dim_subspace
        dim_list = np0.shape[1:]
        N0 = len(dim_list)
        assert np0.shape[0]==N1
        assert (N0>=2) and all(x>=2 for x in dim_list) #TODO what if dim(i)==1
        _,s,v = np.linalg.svd(np0.reshape(N1, -1), full_matrices=False)
        np0 = v[s>zero_eps].reshape(-1, *dim_list) #make basis orthonormal
        self.num_partite = N0
        index_subset = _get_nontrivial_subset_list(N0)
        if use_tensor5: #when N1 is not too large, use_tensor5=True is faster
            self.subset_tensor_list = None
            tmp0 = []
            for ind0 in index_subset:
                ind0 = [x+1 for x in ind0]
                tmp1 = np.prod(np.array([dim_list[x-1] for x in ind0], dtype=np.int64))
                tmp2 = np0.transpose([0] + ind0 + sorted(set(range(1,N0+1))-set(ind0))).reshape(N1, tmp1, -1)
                tmp3 = tmp2.conj()
                tmp0.append(opt_einsum.contract(tmp2, [0,4,7], tmp2, [1,5,6], tmp3, [2,4,6], tmp3, [3,5,7], [0,1,2,3]))
            tmp0 = torch.tensor(np.stack(tmp0), dtype=torch.complex128)
            self.tensor5_contract_expr = opt_einsum.contract_expression(tmp0, [0,1,2,3,4], [N1], [1], [N1], [2], [N1], [3], [N1], [4], [0], constants=[0])
        else:
            self.tensor5_contract_expr = None
            self.subset_tensor_list = []
            for ind0 in index_subset:
                ind0 = [x+1 for x in ind0]
                tmp1 = np.prod(np.array([dim_list[x-1] for x in ind0], dtype=np.int64))
                tmp2 = np0.transpose([0] + ind0 + sorted(set(range(1,N0+1))-set(ind0))).reshape(N1, tmp1, -1)
                self.subset_tensor_list.append(torch.tensor(tmp2, dtype=torch.complex128))

    def forward(self):
        coeff = self.manifold()
        if self.tensor5_contract_expr is not None:
            coeff_conj = coeff.conj()
            purity_list = self.tensor5_contract_expr(coeff,coeff,coeff_conj,coeff_conj).real
        else:
            purity_list = []
            for term_i in self.subset_tensor_list:
                tmp0 = (coeff @ term_i.reshape(coeff.shape[0], -1)).reshape(term_i.shape[1:])
                tmp1 = tmp0 @ tmp0.T.conj() #reduced density matrix
                purity_list.append(torch.vdot(tmp1.reshape(-1), tmp1.reshape(-1)).real)
            purity_list = torch.stack(purity_list)
        loss = 4 - 2**(3-self.num_partite) - 2**(3-self.num_partite) * torch.sum(purity_list)
        #generalized concurrence = sqrt(loss)
        # but sqrt() is non-differentiable at 0 (which means separable state), so we do not take sqrt() when calculating loss
        return loss

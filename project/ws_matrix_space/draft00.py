import time
import itertools
import numpy as np
import torch
import opt_einsum
import scipy.linalg

import numqi

class DetectCPRankModel(torch.nn.Module):
    def __init__(self, basis_orth, rank, dtype='float64'):
        r'''
        Args:
            basis_orth (np.ndarray): shape (N0, N1, N2)
            space_char (str): see numqi.matrix_space.get_matrix_orthogonal_basis
            rank (tuple,int): if int or tuple of length 1, then search for matrix of rank `rank` in the space.
                If tuple (must be of length 3), then search for hermitian matrix in the space of matrices with
                with the inertia `(EVL_free, EVL_positive, EVL_negative)`
            dtype (str): 'float32' or 'float64'
        '''
        super().__init__()
        rank = int(rank)
        assert rank>=1
        self.rank = rank
        self.dim_list = basis_orth.shape[1:]
        np_rng = np.random.default_rng()
        assert basis_orth.ndim>=4
        assert dtype in {'float32','float64'}
        self.dtype = torch.float32 if dtype=='float32' else torch.float64
        self.cdtype = torch.complex64 if dtype=='float32' else torch.complex128
        hf0 = lambda *x: torch.nn.Parameter(torch.tensor(np_rng.uniform(-1,1,size=x), dtype=self.dtype))
        self.theta_EVC = torch.nn.ParameterList([hf0(rank,x,2) for x in self.dim_list])
        self.theta_EVL = hf0(2,rank)
        self.basis_orth = torch.tensor(basis_orth, dtype=self.cdtype)

        N0 = basis_orth.ndim-1
        tmp0 = [(rank,x) for x in self.dim_list]
        tmp1 = [(N0+1,x+1) for x in range(N0)]
        tmp2 = [y for x in zip(tmp0,tmp1) for y in x]
        self.contract_expr = opt_einsum.contract_expression(basis_orth.shape, list(range(N0+1)), *tmp2, [rank], (N0+1,), [0])

    def forward(self):
        tmp0 = self.theta_EVL / torch.linalg.norm(self.theta_EVL)
        EVL = torch.complex(tmp0[0], tmp0[1])
        tmp0 = [x/torch.linalg.norm(x,dim=(1,2),keepdims=True) for x in self.theta_EVC]
        EVC_list = [torch.complex(x[:,:,0],x[:,:,1]) for x in tmp0]
        tmp0 = self.contract_expr(self.basis_orth, *EVC_list, EVL)
        loss = torch.vdot(tmp0, tmp0).real
        return loss


dimA = 3
dimB = 3
dimC = 3
np_list = numqi.matrix_space.get_completed_entangled_subspace(dimA, dimB, dimC, tag_reduce=True)

# mat0 = np_list.reshape(-1,dimA*dimB, dimC)
# # mat0 = np_list.transpose(0,1,3,2).reshape(-1,dimA*dimC,dimB)
# z0 = numqi.matrix_space.has_rank_hierarchical_method(mat0, rank=2, hierarchy_k=4)

# mat0 = np_list.reshape(-1,dimA*dimB, dimC)
# basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(mat0, field='complex')
# model = numqi.matrix_space.DetectRankModel(basis_orth, space_char, rank=1)
# theta_optim = numqi.optimize.minimize(model, 'uniform', num_repeat=3, tol=1e-7)

# mat0 = np_list.reshape(-1,dimA*dimB, dimC)
# basis_orth = numqi.matrix_space.get_vector_orthogonal_basis(np_list.reshape(np_list[0], -1)).reshape(-1, dimA,dimB,dimC)
# basis,basis_orth,space_char = numqi.matrix_space.get_matrix_orthogonal_basis(mat0, field='complex')
# basis_orth = basis_orth.reshape(-1, dimA, dimB, dimC)


basis_orth = numqi.matrix_space.get_vector_orthogonal_basis(np_list.reshape(np_list.shape[0], -1)).reshape(-1, dimA,dimB,dimC)
print(f'[{dimA}x{dimB}x{dimC}] detect rank=2')
t0 = time.time()
model = DetectCPRankModel(basis_orth, rank=2)
theta_optim = numqi.optimize.minimize(model, 'uniform', num_repeat=3, tol=1e-12)
print(f'elapsed time: {time.time()-t0:3f}s', '\n')

print(f'[{dimA}x{dimB}x{dimC}] detect rank=1')
model = DetectCPRankModel(basis_orth, rank=1)
theta_optim = numqi.optimize.minimize(model, 'uniform', num_repeat=3, tol=1e-12)
print(f'elapsed time: {time.time()-t0:3f}s', '\n')


def demo_time_usage():
    case_list = [(2,2,2,2), (2,2,3,2), (2,2,4,2), (2,2,5,2), (2,2,6,2), (2,2,7,2),
                (2,2,8,2), (2,2,9,2), (2,3,3,3), (2,3,4,3), (2,3,5,3), (3,3,3,4)]
    kwargs = dict(theta0='uniform', num_repeat=3, tol=1e-12, print_every_round=0)
    for dimA,dimB,dimC,_ in case_list:
        np_list = numqi.matrix_space.get_completed_entangled_subspace(dimA, dimB, dimC, tag_reduce=True)
        t0 = time.time()
        basis_orth = numqi.matrix_space.get_vector_orthogonal_basis(np_list.reshape(np_list.shape[0], -1)).reshape(-1, dimA,dimB,dimC)
        model = DetectCPRankModel(basis_orth, rank=2)
        theta_optim2 = numqi.optimize.minimize(model, **kwargs)
        model = DetectCPRankModel(basis_orth, rank=1)
        theta_optim1 = numqi.optimize.minimize(model, **kwargs)
        tmp0 = time.time()-t0
        print(f'[{dimA}x{dimB}x{dimC}][{tmp0:.3f}s] loss(r=2)= {theta_optim2.fun:.4e}, loss(r=1)= {theta_optim1.fun:.4e}')
        # [2x2x2][0.095s] loss(r=2)= 1.8304e-14, loss(r=1)= 2.5000e-01
        # [2x2x3][0.087s] loss(r=2)= 5.6645e-14, loss(r=1)= 9.9242e-02
        # [2x2x4][0.091s] loss(r=2)= 1.7311e-13, loss(r=1)= 4.5039e-02
        # [2x2x5][0.140s] loss(r=2)= 2.1488e-13, loss(r=1)= 2.2597e-02
        # [2x2x6][0.165s] loss(r=2)= 1.4117e-13, loss(r=1)= 1.2303e-02
        # [2x2x7][0.244s] loss(r=2)= 9.6170e-14, loss(r=1)= 7.1707e-03
        # [2x2x8][0.223s] loss(r=2)= 3.6233e-13, loss(r=1)= 4.4234e-03
        # [2x2x9][0.408s] loss(r=2)= 7.7931e-13, loss(r=1)= 2.8611e-03
        # [2x3x3][0.114s] loss(r=2)= 4.5884e-14, loss(r=1)= 3.5569e-02
        # [2x3x4][0.137s] loss(r=2)= 3.2846e-13, loss(r=1)= 1.4135e-02
        # [2x3x5][0.202s] loss(r=2)= 1.6536e-13, loss(r=1)= 6.1101e-03
        # [3x3x3][0.178s] loss(r=2)= 5.9082e-13, loss(r=1)= 1.1905e-02

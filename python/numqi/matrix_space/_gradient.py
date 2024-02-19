import numpy as np
import opt_einsum
import torch

import numqi.utils
import numqi.manifold
from ._misc import find_closest_vector_in_space

# cannot be torch.linalg.norm()**2 nan when calculating the gradient when norm is almost zero
# see https://github.com/pytorch/pytorch/issues/99868
# hf_torch_norm_square = lambda x: torch.dot(x.conj(), x).real
hf_torch_norm_square = lambda x: torch.sum((x.conj() * x).real)

class DetectRankModel(torch.nn.Module):
    def __init__(self, basis_orth, space_char, rank, dtype='float64', device='cpu'):
        r'''detect the rank of a matrix subspace

        Parameters:
            basis_orth(np.ndarray): shape (N0,N1,N2)
            space_char(str): see numqi.matrix_space.get_matrix_orthogonal_basis
            rank (tuple,int): if int or tuple of length 1, then search for matrix of rank `rank` in the space.
                If tuple (must be of length 3), then search for hermitian matrix in the space of matrices with
                with the inertia `(EVL_free, EVL_positive, EVL_negative)`
            dtype (str): 'float32' or 'float64'
            device (str): 'cpu' or 'cuda'
        '''
        super().__init__()
        self.is_torch = isinstance(basis_orth, torch.Tensor)
        self.use_sparse = self.is_torch and basis_orth.is_sparse #use sparse only when is a torch.tensor
        assert basis_orth.ndim==3
        assert dtype in {'float32','float64'}
        assert space_char in set('R_T R_A C_T R C C_H R_cT R_c'.split(' '))
        self.dtype = torch.float32 if dtype=='float32' else torch.float64
        self.cdtype = torch.complex64 if dtype=='float32' else torch.complex128
        self.device = torch.device(device)
        self.space_char = space_char
        self.basis_orth_conj = self._setup_basis_orth_conj(basis_orth)
        self._setup_parameter(basis_orth.shape[1], basis_orth.shape[2], space_char, rank, self.dtype, self.cdtype, self.device)

        self.matH = None

    def _setup_basis_orth_conj(self, basis_orth):
        # <A,B>=tr(AB^H)=sum_ij (A_ij, conj(B_ij))
        dtype = self.dtype if (self.space_char in set('R_T R_A R R_cT R_c'.split(' '))) else self.cdtype
        if self.use_sparse:
            assert self.is_torch
            assert self.device==torch.device('cpu'), f'sparse tensor not support device "{self.device}"'
            index = basis_orth.indices()
            shape = basis_orth.shape
            tmp0 = torch.stack([index[0], index[1]*shape[2] + index[2]])
            basis_orth_conj = torch.sparse_coo_tensor(tmp0, basis_orth.values().conj().to(dtype), (shape[0], shape[1]*shape[2]))
        else:
            if self.is_torch:
                basis_orth_conj = basis_orth.conj().reshape(basis_orth.shape[0],-1).to(device=self.device, dtype=dtype)
            else:
                basis_orth_conj = torch.tensor(basis_orth.conj().reshape(basis_orth.shape[0],-1), dtype=dtype, device=self.device)
        return basis_orth_conj

    def _setup_parameter(self, dim0, dim1, space_char, rank, dtype, cdtype, device):
        np_rng = np.random.default_rng()
        rank = numqi.utils.hf_tuple_of_int(rank)
        hf0 = lambda *x: torch.nn.Parameter(torch.tensor(np_rng.uniform(-1,1,size=x), dtype=dtype, device=device))
        if space_char in {'R_T', 'C_H'}:
            assert (len(rank)==1) or (len(rank)==3)
            if len(rank)==1:
                assert 1<=rank[0]<=dim0
                tmp0 = dtype if space_char=='R_T' else cdtype
                self.manifold_U = numqi.manifold.Stiefel(dim0, rank=rank[0], dtype=tmp0, device=device)
                self.EVL_free = hf0(rank[0])
                self.EVL_positive = None
                self.EVL_negative = None
            else:
                assert all(x>=0 for x in rank) and (1<=sum(rank)) and (sum(rank)<=dim0)
                tmp0 = dtype if space_char=='R_T' else cdtype
                tmp1 = sum((0 if (x is None) else x) for x in rank)
                self.manifold_U = numqi.manifold.Stiefel(dim0, rank=tmp1, dtype=tmp0, device=device)
                self.EVL_free = hf0(rank[0]) if (rank[0]>0) else None
                self.EVL_positive = numqi.manifold.PositiveReal(rank[1], dtype=dtype, device=device) if (rank[1]>0) else None
                self.EVL_negative = numqi.manifold.PositiveReal(rank[2], dtype=dtype, device=device) if (rank[2]>0) else None
        elif space_char=='R_A':
            assert (len(rank)==1) or (rank[0]%2==0)
            self.manifold_U = numqi.manifold.Stiefel(dim0, rank=rank[0], dtype=dtype, device=device)
            self.manifold_EVL = numqi.manifold.Sphere(rank[0]//2, dtype=dtype, device=device) if (rank[0]>2) else None
        elif space_char=='C_T':
            assert (len(rank)==1) and 1<=rank[0]<=dim0
            self.manifold_U = numqi.manifold.Stiefel(dim0, rank=rank[0], dtype=cdtype, device=device)
            self.manifold_EVL = numqi.manifold.Sphere(rank[0], dtype=cdtype, device=device) if (rank[0]>1) else None
        elif space_char in {'R','C'}:
            assert (len(rank)==1) and 1<=rank[0]<=min(dim0,dim1)
            tmp0 = dtype if space_char=='R' else cdtype
            self.manifold_U0 = numqi.manifold.Stiefel(dim0, rank=rank[0], dtype=tmp0, device=device)
            self.manifold_U1 = numqi.manifold.Stiefel(dim1, rank=rank[0], dtype=tmp0, device=device)
            self.manifold_EVL = numqi.manifold.Sphere(rank[0], dtype=dtype, device=device) if (rank[0]>1) else None
        elif space_char=='R_cT':
            assert (dim0%2==0) and (len(rank)==1) and (1<=rank[0]<=(dim0//2))
            self.manifold_EVL = numqi.manifold.Sphere(rank[0], dtype=cdtype, device=device) if (rank[0]>1) else None
            self.manifold_U = numqi.manifold.Stiefel(dim0//2, rank=rank[0], dtype=cdtype, device=device)
        elif space_char=='R_c':
            assert (dim0%2==0) and (dim1%2==0) and (len(rank)==1) and (1<=rank[0]<=(min(dim0,dim1)//2))
            self.manifold_EVL = numqi.manifold.Sphere(rank[0], dtype=dtype, device=device) if (rank[0]>1) else None
            self.manifold_U0 = numqi.manifold.Stiefel(dim0//2, rank=rank[0], dtype=dtype, device=device)
            self.manifold_U1 = numqi.manifold.Stiefel(dim1//2, rank=rank[0], dtype=dtype, device=device)

    def forward(self):
        space_char = self.space_char
        if space_char in {'R_T', 'C_H'}:
            tmp0 = [
                self.EVL_free,
                None if (self.EVL_positive is None) else self.EVL_positive(),
                None if (self.EVL_negative is None) else (-self.EVL_negative()),
            ]
            tmp1 = torch.cat([x for x in tmp0 if x is not None])
            EVL = tmp1 / torch.linalg.norm(tmp1)
            unitary = self.manifold_U()
            matH = (unitary.conj()*EVL) @ unitary.T
            loss = hf_torch_norm_square(self.basis_orth_conj @ matH.reshape(-1))
        elif space_char=='R_A':
            EVL = 1 if (self.manifold_EVL is None) else self.manifold_EVL()
            unitary = self.manifold_U()
            tmp0 = unitary[:,::2]
            tmp1 = unitary[:,1::2]
            matH = (tmp0 * EVL) @ tmp1.T - (tmp1 * EVL) @ tmp0.T
            loss = hf_torch_norm_square(self.basis_orth_conj @ matH.reshape(-1))
        elif space_char=='C_T':
            EVL = 1 if (self.manifold_EVL is None) else self.manifold_EVL()
            unitary = self.manifold_U()
            matH = (unitary*EVL) @ unitary.T
            loss = hf_torch_norm_square(self.basis_orth_conj @ matH.reshape(-1))
        elif space_char in {'R','C'}:
            EVL = 1 if (self.manifold_EVL is None) else self.manifold_EVL()
            unitary0 = self.manifold_U0()
            unitary1 = self.manifold_U1()
            matH = (unitary0.conj()*EVL) @ unitary1.T
            loss = hf_torch_norm_square(self.basis_orth_conj @ matH.reshape(-1))
        elif space_char=='R_cT':
            EVL = 1 if (self.manifold_EVL is None) else self.manifold_EVL()
            unitary = self.manifold_U()
            matH = (unitary*EVL) @ unitary.T
            tmp0 = numqi.utils.hf_complex_to_real(matH)
            loss = hf_torch_norm_square(self.basis_orth_conj @ tmp0.reshape(-1))
        elif space_char=='R_c':
            EVL = 1 if (self.manifold_EVL is None) else self.manifold_EVL()
            unitary0 = self.manifold_U0()
            unitary1 = self.manifold_U1()
            matH = (unitary0.conj()*EVL) @ unitary1.T
            tmp0 = numqi.utils.hf_complex_to_real(matH)
            loss = hf_torch_norm_square(self.basis_orth_conj @ tmp0.reshape(-1))
        self.matH = matH.detach()
        return loss

    def get_matrix(self, matrix_subspace):
        with torch.no_grad():
            self()
        matH = self.matH.cpu().numpy().copy()
        field = 'real' if (self.space_char in set('R_T R R_cT R_c C_H'.split(' '))) else 'complex'
        coeff, residual = find_closest_vector_in_space(matrix_subspace, matH, field)
        return matH,coeff,residual


# a canonical polyadic (CP) tensor decomposition
# only field='complex'
# no symmetry is used
class DetectCanonicalPolyadicRankModel(torch.nn.Module):
    def __init__(self, dim_list, rank:int, bipartition=None):
        r'''detect canonical polyadic (CP) tensor decomposition

        Parameters:
            dim_list (tuple): shape of the tensor
            rank (int): rank of the tensor
        '''
        super().__init__()
        dim_list = tuple(int(x) for x in dim_list)
        self.dim_list_ori = dim_list
        if bipartition is not None:
            bipartition = tuple(sorted({int(x) for x in bipartition}))
            assert (len(bipartition)>=1) and (bipartition[0]>=0) and (bipartition[-1]<len(dim_list))
            tmp0 = np.prod([dim_list[x] for x in bipartition]).item()
            dim_list = tmp0, np.prod(dim_list).item()//tmp0
        self.bipartition = bipartition
        self.dim_list = dim_list
        assert len(dim_list)>=2
        assert all(x>1 for x in dim_list)
        assert rank>=1
        self.rank = rank
        self.manifold_psi_list = torch.nn.ModuleList([numqi.manifold.Sphere(x, batch_size=rank, dtype=torch.complex128) for x in dim_list])
        self.manifold_coeff = numqi.manifold.PositiveReal(rank, method='softplus', dtype=torch.float64)

        N0 = len(dim_list)
        tmp0 = [(rank,),(rank,)] + [(rank,x) for x in dim_list] + [(rank,x) for x in dim_list]
        tmp1 = [(N0,),(N0+1,)] + [(N0,x) for x in range(N0)] + [(N0+1,x) for x in range(N0)]
        self.contract_psi_psi = opt_einsum.contract_expression(*[y for x in zip(tmp0,tmp1) for y in x], [])
        self.contract_target_psi = None

        self.target = None
        self.traget_conj = None

    def set_target(self, np0, zero_eps=1e-7):
        N0 = len(self.dim_list)
        if (np0.shape==self.dim_list_ori) or ((np0.shape[1:]==self.dim_list_ori) and np0.shape[0]==1):
            if np0.shape[1:]==self.dim_list_ori:
                np0 = np0[0]
            if self.bipartition is not None:
                tmp0 = self.bipartition + tuple(sorted(set(range(len(self.dim_list_ori))) - set(self.bipartition)))
                np0 = np0.reshape(self.dim_list_ori).transpose(tmp0).reshape(self.dim_list)
            self.target = torch.tensor(np0 / np.linalg.norm(np0.reshape(-1)), dtype=torch.complex128)
            tmp0 = [self.dim_list,(self.rank,)] + [(self.rank,x) for x in self.dim_list]
            tmp1 = [tuple(range(N0)),(N0,)] + [(N0,x) for x in range(N0)]
            self.contract_target_psi = opt_einsum.contract_expression(*[y for x in zip(tmp0,tmp1) for y in x], [])
        elif np0.shape[1:]==self.dim_list_ori:
            _,s,v = np.linalg.svd(np0.reshape(np0.shape[0], -1), full_matrices=False)
            np0 = v[s>zero_eps].reshape(-1, *self.dim_list_ori)
            if self.bipartition is not None:
                tmp0 = self.bipartition + tuple(sorted(set(range(len(self.dim_list_ori))) - set(self.bipartition)))
                np0 = np0.transpose([0]+[x+1 for x in tmp0]).reshape(-1, *self.dim_list)
            self.target = torch.tensor(np0, dtype=torch.complex128)
            tmp0 = [np0.shape,(self.rank,)] + [(self.rank,x) for x in self.dim_list]
            tmp1 = [tuple(range(N0+1)),(N0+1,)] + [(N0+1,x+1) for x in range(N0)]
            self.contract_target_psi = opt_einsum.contract_expression(*[y for x in zip(tmp0,tmp1) for y in x], [0])
        else:
            tmp0 = (-1,) + self.dim_list_ori
            assert False, f'invalid shape, np0.shape should be "({self.dim_list_ori})" or "({tmp0})", but got "{np0.shape}"'
        self.target_conj = self.target.conj().resolve_conj()

    def forward(self):
        psi_list, coeff = self._get_state()
        target_psi = self.contract_target_psi(self.target_conj, coeff, *psi_list)
        if target_psi.ndim==1:
            loss = 1 - torch.vdot(target_psi, target_psi).real
        else:
            loss = 1 - (target_psi.real**2 + target_psi.imag**2)
        return loss

    def _get_state(self):
        psi_list = [x() for x in self.manifold_psi_list]
        coeff = self.manifold_coeff().to(psi_list[0].dtype)
        psi_list_conj = [x.conj().resolve_conj() for x in psi_list]
        psi_psi = self.contract_psi_psi(coeff, coeff, *psi_list, *psi_list_conj).real
        coeff = coeff / torch.sqrt(psi_psi)
        return psi_list, coeff

    def get_state(self):
        with torch.no_grad():
            psi_list, coeff = self._get_state()
            coeff = coeff.numpy().copy()
            psi = [x.numpy().copy() for x in psi_list]
        if self.bipartition is not None:
            assert len(psi)==2
            tmp0 = [self.dim_list_ori[x] for x in self.bipartition]
            psi0 = psi[0].reshape(-1, *tmp0)
            tmp0 = [self.dim_list_ori[x] for x in sorted(set(range(len(self.dim_list_ori))) - set(self.bipartition))]
            psi1 = psi[1].reshape(-1, *tmp0)
            psi = psi0, psi1
        return coeff, psi


# TODO merge with DetectRankModel
class DetectOrthogonalRankOneModel(torch.nn.Module):
    def __init__(self, matB, dtype='float64'):
        super().__init__()
        assert dtype in {'float64','float32'} #TODO complex
        assert not np.iscomplexobj(matB), 'not support complex yet'
        np_rng = np.random.default_rng()
        self.dtype = torch.float32 if dtype=='float32' else torch.float64
        self.theta0 = torch.nn.Parameter(torch.tensor(np_rng.uniform(-1,1,size=matB.shape[1]), dtype=self.dtype))
        self.theta1 = torch.nn.Parameter(torch.tensor(np_rng.uniform(-1,1,size=matB.shape[1]), dtype=self.dtype))
        self.matB = torch.tensor(matB, dtype=self.dtype)
        self.vecX = None
        self.vecY = None
        self.tag_loss_yBx = False

    def forward(self):
        vecX = self.theta0 / torch.linalg.norm(self.theta0)
        vecY = self.theta1 / torch.linalg.norm(self.theta1)
        self.vecX = vecX
        self.vecY = vecY
        loss = hf_torch_norm_square((self.matB @ vecY) @ vecX)
        if self.tag_loss_yBx: #do not help
            loss = loss + hf_torch_norm_square((self.matB @ vecX) @ vecY)
        return loss

    def get_vecX_vecY(self, reshape=None, tag_print=False):
        with torch.no_grad():
            ret = self()
        npX = self.vecX.detach().numpy()
        npY = self.vecY.detach().numpy()
        ret = np.abs(npX @ self.matB.detach().numpy() @ npY).max()
        if tag_print:
            print(ret)
            tmp0 = npX if (reshape is None) else npX.reshape(reshape)
            tmp1 = npY if (reshape is None) else npY.reshape(reshape)
            print(tmp0)
            print(tmp1)
        return npX,npY,ret


# bad, second-svd(a_i A_i)
class DetectRankOneModel(torch.nn.Module):
    def __init__(self, matA, dtype='float64'):
        super().__init__()
        assert dtype in {'float64','float32','complex64','complex128'}
        np_rng = np.random.default_rng()
        self.dtype = torch.float32 if dtype in {'float32','complex64'} else torch.float64
        self.cdtype = torch.complex64 if dtype in {'float32','complex64'} else torch.complex128
        self.use_complex = dtype in {'complex64','complex128'}
        self.matA = torch.tensor(matA, dtype=(self.cdtype if self.use_complex else self.dtype))
        num_para = (2 if self.use_complex else 1) * matA.shape[0]
        self.theta = torch.nn.Parameter(torch.tensor(np_rng.uniform(-1,1,size=num_para), dtype=self.dtype))
        self.matH = None

    def forward(self):
        N0 = self.matA.shape[0]
        coeff = self.theta / torch.linalg.norm(self.theta)
        if self.use_complex:
            coeff = coeff[:N0] + 1j*coeff[N0:]
        matH = (coeff @ self.matA.reshape(self.matA.shape[0],-1)).reshape(self.matA.shape[1], self.matA.shape[2])
        self.matH = matH
        loss = torch.linalg.svd(matH, full_matrices=False)[1][1]
        # loss = torch.linalg.eigvalsh(matH @ matH.T)[-2]
        return loss

    def get_vecX_vecY(self):
        N0 = self.matA.shape[0]
        tmp0 = self.theta.detach().numpy()
        coeff = tmp0 / torch.linalg.norm(tmp0)
        if self.use_complex:
            coeff = coeff[:N0] + 1j*coeff[N0:]
        matH = (coeff @ self.matA.cpu().numpy().reshape(self.matA.shape[0],-1)).reshape(self.matA.shape[1], self.matA.shape[2])
        u,s,v = np.linalg.svd(matH)
        npX = u[:,0]
        npY = v[0]
        return coeff, npX, npY

# not good, space(x B) not full rank
# old name: DetectOrthogonalRank1EigenModel
class DetectOrthogonalRankOneEigenModel(torch.nn.Module):
    def __init__(self, matB):
        super().__init__()
        assert not np.iscomplexobj(matB), 'not support complex yet'
        self.matB = torch.tensor(matB, dtype=torch.float64)
        np_rng = np.random.default_rng()
        tmp0 = np_rng.normal(size=matB.shape[-1])
        self.theta0 = torch.nn.Parameter(torch.tensor(tmp0, dtype=torch.float64))

    def forward(self):
        vecY = self.theta0 / torch.linalg.norm(self.theta0)
        tmp1 = self.matB @ vecY
        loss = torch.linalg.eigvalsh(tmp1.T @ tmp1)[0] #if not full rank, then exist vecX
        return loss

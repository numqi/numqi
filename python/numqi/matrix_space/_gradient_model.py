import numpy as np
import scipy.optimize
import torch

import numqi.utils
import numqi.param
import numqi.optimize
from ._misc import find_closest_vector_in_space

# cannot be torch.linalg.norm()**2 nan when calculating the gradient when norm is almost zero
hf_torch_norm_square = lambda x: torch.dot(x.conj(), x).real

# old_name DetectMatrixSpaceRank
class DetectRankModel(torch.nn.Module):
    def __init__(self, basis_orth, space_char, rank, dtype='float64', device='cpu'):
        super().__init__()
        self.is_torch = isinstance(basis_orth, torch.Tensor)
        self.use_sparse = self.is_torch and basis_orth.is_sparse #use sparse only when is a torch.tensor
        assert basis_orth.ndim==3
        assert dtype in {'float32','float64'}
        assert space_char in set('R_T R_A C_T R C C_H R_cT R_c'.split(' '))
        self.dtype = torch.float32 if dtype=='float32' else torch.float64
        self.cdtype = torch.complex64 if dtype=='float32' else torch.complex128
        self.device = device
        self.space_char = space_char
        self.basis_orth_conj = self._setup_basis_orth_conj(basis_orth)
        self.theta = self._setup_parameter(basis_orth.shape[1], basis_orth.shape[2], space_char, rank, self.dtype, self.device)

        self.matH = None

    def _setup_basis_orth_conj(self, basis_orth):
        # <A,B>=tr(AB^H)=sum_ij (A_ij, conj(B_ij))
        dtype = self.dtype if (self.space_char in set('R_T R_A R R_cT R_c'.split(' '))) else self.cdtype
        if self.use_sparse:
            assert self.is_torch
            assert self.device=='cpu', f'sparse tensor not support device "{self.device}"'
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

    def _setup_parameter(self, dim0, dim1, space_char, rank, dtype, device):
        np_rng = np.random.default_rng()
        rank = numqi.utils.hf_tuple_of_int(rank)
        hf0 = lambda *x: torch.nn.Parameter(torch.tensor(np_rng.uniform(-1,1,size=x), dtype=dtype, device=device))
        if space_char in {'R_T', 'C_H'}:
            assert (len(rank)==1) or (len(rank)==3)
            if len(rank)==1:
                assert 1<=rank[0]<=dim0
                theta = {'unitary0':hf0(dim0, dim0), 'EVL_free':hf0(rank[0]), 'EVL_positive':None, 'EVL_negative':None}
            else:
                assert all(x>=0 for x in rank) and (1<=sum(rank)) and (sum(rank)<=dim0)
                theta = {
                    'unitary0':hf0(dim0, dim0),
                    'EVL_free':hf0(rank[0]) if (rank[0]>0) else None,
                    'EVL_positive':hf0(rank[1]) if (rank[1]>0) else None,
                    'EVL_negative':hf0(rank[2]) if (rank[2]>0) else None,
                }
        elif space_char=='R_A':
            assert (len(rank)==1) or (rank[0]%2==0)
            theta = {'unitary0':hf0(dim0, dim0), 'EVL0':hf0(rank[0]//2)}
        elif space_char=='C_T':
            assert (len(rank)==1) and 1<=rank[0]<=dim0
            theta = {'unitary0':hf0(dim0, dim0), 'EVL0':hf0(rank[0]), 'EVL1':hf0(rank[0])}
        elif space_char in {'R','C'}:
            assert (len(rank)==1) and 1<=rank[0]<=min(dim0,dim1)
            theta = {'unitary0':hf0(dim0, dim0), 'unitary1':hf0(dim1, dim1), 'EVL0':hf0(rank[0])}
        elif space_char=='R_cT':
            assert (dim0%2==0) and (len(rank)==1) and (1<=rank[0]<=(dim0//2))
            theta = {'unitary0':hf0(dim0//2, dim0//2), 'EVL0':hf0(rank[0]), 'EVL1':hf0(rank[0])}
        elif space_char=='R_c':
            assert (dim0%2==0) and (dim1%2==0) and (len(rank)==1) and (1<=rank[0]<=(min(dim0,dim1)//2))
            theta = {'unitary0':hf0(dim0//2, dim0//2), 'unitary1':hf0(dim1//2, dim1//2), 'EVL0':hf0(rank[0])}
        ret = torch.nn.ParameterDict(theta)
        return ret

    def forward(self):
        theta = self.theta
        space_char = self.space_char
        if space_char in {'R_T', 'C_H'}:
            tmp0 = [
                theta['EVL_free'],
                None if (theta['EVL_positive'] is None) else torch.nn.functional.softplus(theta['EVL_positive']),
                None if (theta['EVL_negative'] is None) else (-torch.nn.functional.softplus(theta['EVL_negative'])),
            ]
            tmp1 = torch.cat([x for x in tmp0 if x is not None])
            EVL = tmp1 / torch.linalg.norm(tmp1)
            unitary = numqi.param.real_matrix_to_special_unitary(theta['unitary0'], tag_real=(space_char=='R_T'))[:len(EVL)]
            matH = (unitary.T.conj()*EVL) @ unitary
            loss = hf_torch_norm_square(self.basis_orth_conj @ matH.reshape(-1))
        elif space_char=='R_A':
            tmp0 = theta['EVL0']
            EVL = tmp0 / torch.linalg.norm(tmp0)
            unitary = numqi.param.real_matrix_to_special_unitary(theta['unitary0'], tag_real=True)[:(2*len(EVL))]
            tmp0 = unitary[::2]
            tmp1 = unitary[1::2]
            matH = (tmp0.T * EVL) @ tmp1 - (tmp1.T * EVL) @ tmp0
            loss = hf_torch_norm_square(self.basis_orth_conj @ matH.reshape(-1))
        elif space_char=='C_T':
            tmp0 = theta['EVL0'] + 1j*theta['EVL1']
            EVL = tmp0/torch.linalg.norm(tmp0)
            unitary = numqi.param.real_matrix_to_special_unitary(theta['unitary0'], tag_real=False)[:len(EVL)]
            matH = (unitary.T*EVL) @ unitary
            loss = hf_torch_norm_square(self.basis_orth_conj @ matH.reshape(-1))
        elif space_char in {'R','C'}:
            tag_real = space_char=='R'
            tmp0 = theta['EVL0']
            EVL = tmp0/torch.linalg.norm(tmp0)
            unitary0 = numqi.param.real_matrix_to_special_unitary(theta['unitary0'], tag_real=tag_real)[:len(EVL)]
            unitary1 = numqi.param.real_matrix_to_special_unitary(theta['unitary1'], tag_real=tag_real)[:len(EVL)]
            matH = (unitary0.T.conj()*EVL) @ unitary1
            loss = hf_torch_norm_square(self.basis_orth_conj @ matH.reshape(-1))
        elif space_char=='R_cT':
            tmp0 = theta['EVL0'] + 1j*theta['EVL1']
            EVL = tmp0/torch.linalg.norm(tmp0)
            unitary = numqi.param.real_matrix_to_special_unitary(theta['unitary0'], tag_real=False)[:len(EVL)]
            matH = (unitary.T*EVL) @ unitary
            tmp0 = numqi.utils.hf_complex_to_real(matH)
            loss = hf_torch_norm_square(self.basis_orth_conj @ tmp0.reshape(-1))
        elif space_char=='R_c':
            tmp0 = theta['EVL0']
            EVL = tmp0/torch.linalg.norm(tmp0)
            unitary0 = numqi.param.real_matrix_to_special_unitary(theta['unitary0'], tag_real=tag_real)[:len(EVL)]
            unitary1 = numqi.param.real_matrix_to_special_unitary(theta['unitary1'], tag_real=tag_real)[:len(EVL)]
            matH = (unitary0.T.conj()*EVL) @ unitary1
            tmp0 = numqi.utils.hf_complex_to_real(matH)
            loss = hf_torch_norm_square(self.basis_orth_conj @ tmp0.reshape(-1))
        self.matH = matH
        return loss

    def get_matrix(self, theta, matrix_subspace):
        numqi.optimize.set_model_flat_parameter(self, theta)
        with torch.no_grad():
            self()
        matH = self.matH.detach().cpu().numpy().copy()
        field = 'real' if (self.space_char in set('R_T R R_cT R_c C_H'.split(' '))) else 'complex'
        coeff, residual = find_closest_vector_in_space(matrix_subspace, matH, field)
        return matH,coeff,residual

    # numqi.optimize.minimize(model, 'normal', num_repeat=3, early_stop_threshold=0.01, tol=1e-7, print_freq=20)
    # @deprecated sometimes we need threshold to early stop, like in UDA/UDP
    def minimize(self, num_repeat=3, print_freq=-1, tol=1e-7, threshold=None, seed=None):
        # threshold is used for quick return if fun<threshold
        np_rng = np.random.default_rng(seed)
        num_parameter = len(numqi.optimize.get_model_flat_parameter(self))
        hf_model = numqi.optimize.hf_model_wrapper(self)
        loss_list = []
        for _ in range(num_repeat):
            theta0 = np_rng.normal(size=num_parameter)
            hf_callback = numqi.optimize.hf_callback_wrapper(hf_model, print_freq=print_freq)
            theta_optim = scipy.optimize.minimize(hf_model, theta0, jac=True, method='L-BFGS-B', tol=tol, callback=hf_callback)
            loss_list.append(theta_optim)
            if (threshold is not None) and (theta_optim.fun < threshold):
                break
        ret = min(loss_list, key=lambda x: x.fun)
        numqi.optimize.set_model_flat_parameter(self, ret.x)
        return ret


# old name: DetectOrthogonalRank1Model
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
# old name: DetectMatrixSubspaceRank1Model
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

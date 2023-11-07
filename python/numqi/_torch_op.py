import torch
import numpy as np
import scipy.linalg

class TorchMatrixSqrtm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matA):
        assert not matA.is_cuda, 'TorchMatrixSqrtm() not support GPU'
        assert matA.ndim==2, 'TorchMatrixSqrtm() only suport dim(A)==2, not batch support'
        matA_np = matA.numpy()
        ret = torch.from_numpy(scipy.linalg.sqrtm(matA_np).astype(matA_np.dtype))
        # TODO .astype(matA.dtype) scipy-v1.10 bug https://github.com/scipy/scipy/issues/18250
        ctx.save_for_backward(ret)
        return ret
    @staticmethod
    def backward(ctx, grad_output):
        matB, = ctx.saved_tensors
        matB_np = matB.numpy()
        gradA_np = scipy.linalg.solve_sylvester(matB_np, matB_np, grad_output.numpy())
        gradA = torch.from_numpy(gradA_np)
        return gradA,


def _torch_psd_sqrtm_forward_repeat(matA, repeat=1):
    assert (matA.ndim==2) or (matA.ndim==3)
    repeat = int(repeat)
    assert repeat >= 1
    if (not matA.is_cuda) and (matA.ndim==2):
        ret = matA.numpy()
        if matA.ndim==2:
            for _ in range(repeat):
                # TODO .astype(matA.dtype) scipy-v1.10 bug https://github.com/scipy/scipy/issues/18250
                ret = scipy.linalg.sqrtm(ret).astype(ret.dtype)
        else: #TOO slow, not good (dead code)
            for _ in range(repeat):
                ret = [scipy.linalg.sqrtm(x).astype(x.dtype) for x in ret]
            ret = np.stack(ret)
        ret = torch.from_numpy(ret)
        ctx_tensor = ret,
    else:
        N0 = matA.shape[-1]
        shape = matA.shape[:-2]
        matA = matA.view(-1, N0, N0)
        EVL, EVC = torch.linalg.eigh(matA)
        sqrt_EVL = torch.maximum(torch.zeros(1, dtype=EVL.dtype, device=EVL.device), EVL)
        for _ in range(repeat):
            sqrt_EVL = torch.sqrt(sqrt_EVL)
        ret = ((EVC*sqrt_EVL.view(-1,1,N0)) @ EVC.transpose(1,2).conj()).view(*shape, N0, N0)
        ctx_tensor = sqrt_EVL, EVC
    return ret,ctx_tensor


def _torch_psd_sqrtm_backward_repeat(grad_output, ctx_tensor, repeat=1):
    assert (grad_output.ndim==2) or (grad_output.ndim==3)
    repeat = int(repeat)
    assert repeat>=1
    if (not grad_output.is_cuda) and (grad_output.ndim==2):
        tmp0 = ctx_tensor[0].numpy()
        ret = grad_output.numpy()
        for ind0 in range(repeat):
            if grad_output.ndim==2:
                ret = scipy.linalg.solve_sylvester(tmp0, tmp0, ret)
            else: #too slow, not good (dead code)
                ret = np.stack([scipy.linalg.solve_sylvester(x, x, y) for x,y in zip(tmp0, ret)])
            if ind0!=repeat-1:
                tmp0 = tmp0 @ tmp0
        ret = torch.from_numpy(ret)
    else:
        N0 = grad_output.shape[-1]
        shape = grad_output.shape[:-2]
        grad_output = grad_output.view(-1, N0, N0)
        sqrt_EVL,EVC = ctx_tensor
        EVCh = EVC.transpose(1,2).conj()
        ret = grad_output
        if torch.any(sqrt_EVL==0).item():
            ind_zero = torch.nonzero(sqrt_EVL==0)
        else:
            ind_zero = None
        for ind0 in range(repeat):
            tmp0 = sqrt_EVL.view(-1, 1, N0) + sqrt_EVL.view(-1, N0, 1)
            tmp1 = (EVCh @ ret @ EVC) / tmp0
            if ind_zero is not None:
                tmp1[ind_zero[:,0],ind_zero[:,1],ind_zero[:,1]] = 0
            ret = (EVC @ tmp1 @ EVCh).view(*shape, N0, N0)
            if ind0!=repeat-1:
                sqrt_EVL = sqrt_EVL**2
    return ret


class TorchPSDMatrixSqrtm(torch.autograd.Function):
    # it's user's duty to check the Hermitian, PSD
    # https://github.com/pytorch/pytorch/issues/25481#issuecomment-544465798
    @staticmethod
    def forward(ctx, matA):
        ret,tmp0 = _torch_psd_sqrtm_forward_repeat(matA, repeat=1)
        # ret,tmp0 = _torch_psd_sqrtm_forward(matA)
        ctx.save_for_backward(*tmp0)
        return ret
    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.ndim>=2
        ret = _torch_psd_sqrtm_backward_repeat(grad_output, ctx.saved_tensors, repeat=1)
        return ret,


class PartPSDMatrixLogarithm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matA, s):
        ret,tmp0 = _torch_psd_sqrtm_forward_repeat(matA, repeat=s)
        # matA_np = matA.numpy()
        # for _ in range(s):
        #     matA_np = scipy.linalg.sqrtm(matA_np)
        # matB = torch.from_numpy(matA_np)
        ctx.save_for_backward(*tmp0, torch.tensor(s))
        return ret
    @staticmethod
    def backward(ctx, grad_output):
        tmp0 = ctx.saved_tensors
        ret = _torch_psd_sqrtm_backward_repeat(grad_output, tmp0[:-1], repeat=tmp0[-1].item())
        return ret,None


class TorchMatrixLogm(torch.nn.Module):
    def __init__(self, num_sqrtm=8, pade_order=8, device='cpu'):
        super().__init__()
        node,weight = np.polynomial.legendre.leggauss(pade_order)
        self.alpha = torch.tensor(weight * 2**(num_sqrtm-1), device=device).view(-1, 1, 1, 1)
        self.beta = torch.tensor((node + 1) / 2, device=device).view(-1, 1, 1, 1)
        self.num_sqrtm = num_sqrtm

    def forward(self, matA):
        assert matA.ndim >= 2
        shape = matA.shape
        N0 = shape[-1]
        matA = matA.reshape(-1, N0, N0)
        torch1 = PartPSDMatrixLogarithm.apply(matA, self.num_sqrtm)
        eye0 = torch.eye(N0, device=matA.device)
        # ret = sum(torch.linalg.solve((1-b)*eye0+b*torch1, a*torch1-a*eye0) for a,b in zip(self.alpha,self.beta))
        tmp0 = (1-self.beta)*eye0+self.beta*torch1
        tmp1 = self.alpha*torch1 - self.alpha*eye0
        ret = torch.linalg.solve(tmp0, tmp1).sum(dim=0).reshape(*shape)
        return ret

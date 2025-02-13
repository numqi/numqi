import numpy as np
import torch

import numqi

np_rng = np.random.default_rng()
hf_randc = lambda *sz: np_rng.normal(size=sz) + 1j*np_rng.normal(size=sz)


def test_knill_laflamme_hermite_mul():
    N0 = 2
    N1 = 3
    dim = 5
    tmp0 = hf_randc(N0, dim, dim)
    matA = torch.tensor(tmp0 + tmp0.transpose(0,2,1).conj(), dtype=torch.complex128)
    psi = torch.tensor(hf_randc(dim, N1), dtype=torch.complex128, requires_grad=True)
    matB = torch.tensor(hf_randc(N0, N1, N1), dtype=torch.complex128)
    hf0 = lambda x,y: torch.sum(matB * numqi.qec._grad.knill_laflamme_hermite_mul(x, y)).real
    assert torch.autograd.gradcheck(hf0, (matA, psi))

import numpy as np
import torch
import itertools

import numqi

np_rng = np.random.default_rng()

def test_OpenInterval():
    a = -1
    b = 1
    manifold = numqi.manifold.OpenInterval(a, b)
    x0 = manifold().item()
    assert a < x0 < b
    tmp0 = manifold.theta.detach().numpy()
    x1 = numqi.manifold.to_open_interval(tmp0, a, b)
    assert np.abs(x0 - x1) < 1e-10

    manifold = numqi.manifold.OpenInterval(a, b, batch_size=233)
    x0 = manifold().detach().numpy()
    assert np.all(x0 > a) and np.all(x0 < b)
    tmp0 = manifold.theta.detach().numpy()
    x1 = numqi.manifold.to_open_interval(tmp0, a, b)
    assert np.abs(x0 - x1).max() < 1e-10


def test_Trace1PSD():
    batch_size = 3
    dim = 7
    for rank in [1,3,7]:
        for dtype in [torch.float64, torch.complex128]:
            manifold = numqi.manifold.Trace1PSD(dim, rank, batch_size, dtype=dtype)
            x0 = manifold().detach().numpy()
            assert np.abs(x0 - x0.transpose(0,2,1).conj()).max() < 1e-10
            assert np.abs(np.trace(x0, axis1=1, axis2=2)-1).max() < 1e-10
            EVL = np.linalg.eigvalsh(x0)
            assert np.all((EVL>1e-10).sum(axis=1) == rank)

            tmp0 = manifold.theta.detach().numpy()
            x1 = numqi.manifold.to_trace1_psd_cholesky(tmp0, dim, rank)
            assert np.abs(x0 - x1).max() < 1e-10

class DummyModel00(torch.nn.Module):
    def __init__(self, dm0):
        super().__init__()
        self.dm0 = torch.tensor(dm0, dtype=torch.complex128)
        self.manifold = numqi.manifold.Trace1PSD(dm0.shape[0], dtype=torch.complex128)

    def forward(self):
        dm1 = self.manifold()
        tmp0 = (dm1-self.dm0).reshape(-1)
        loss = torch.vdot(tmp0, tmp0).real
        return loss

def test_Trace1PSD_grad():
    dim = 5
    dm0 = numqi.random.rand_density_matrix(dim)
    model = DummyModel00(dm0)
    numqi.optimize.check_model_gradient(model, tol=1e-6)


def test_SymmetricMatrix():
    batch_size = 3
    dim = 7
    for is_trace0,is_norm1,dtype in itertools.product([False,True], [False,True], [torch.float64, torch.complex128]):
        manifold = numqi.manifold.SymmetricMatrix(dim, batch_size, is_trace0=is_trace0, is_norm1=is_norm1, dtype=dtype)
        x0 = manifold().detach().numpy()
        if dtype==torch.complex128:
            assert np.abs(x0 - x0.transpose(0,2,1).conj()).max() < 1e-10
        else:
            assert np.abs(x0 - x0.transpose(0,2,1)).max() < 1e-10
        if is_trace0:
            assert np.abs(np.trace(x0, axis1=1, axis2=2)).max() < 1e-10
        if is_norm1:
            assert np.abs(np.linalg.norm(x0, axis=(1,2))-1).max() < 1e-10

        tmp0 = manifold.theta.detach().numpy()
        x1 = numqi.manifold.to_symmetric_matrix(tmp0, dim, is_trace0=is_trace0, is_norm1=is_norm1)
        assert np.abs(x0-x1).max() < 1e-10


def test_Sphere():
    batch_size = 3
    dim = 7
    for dtype in [torch.float64, torch.complex128]:
        manifold = numqi.manifold.Sphere(dim, batch_size, dtype=dtype, method='quotient')
        x0 = manifold().detach().numpy()
        assert np.abs(np.linalg.norm(x0, axis=1)-1).max() < 1e-10

        tmp0 = manifold.theta.detach().numpy()
        is_real = dtype in [torch.float64, torch.float32]
        x1 = numqi.manifold.to_sphere_quotient(tmp0, is_real)
        assert np.abs(x0-x1).max() < 1e-10

        manifold = numqi.manifold.Sphere(dim, batch_size, dtype=dtype, method='coordinate')
        x0 = manifold().detach().numpy()
        assert np.abs(np.linalg.norm(x0, axis=1)-1).max() < 1e-10

        tmp0 = manifold.theta.detach().numpy()
        is_real = dtype in [torch.float64, torch.float32]
        x1 = numqi.manifold.to_sphere_coordinate(tmp0, is_real)
        assert np.abs(x0-x1).max() < 1e-10


def test_DiscreteProbabilitty():
    batch_size = 3
    dim = 7
    dtype = torch.float64
    for method in ['softmax', 'sphere']:
        manifold = numqi.manifold.DiscreteProbability(dim, batch_size, method=method, dtype=dtype)
        x0 = manifold().detach().numpy()
        assert np.all(x0>=0) and (np.abs(np.sum(x0,axis=1).max()-1) < 1e-10)
        tmp0 = manifold.theta.detach().numpy()
        if method=='softmax':
            x1 = numqi.manifold.to_discrete_probability_softmax(tmp0)
        else:
            x1 = numqi.manifold.to_discrete_probability_sphere(tmp0)
        assert np.abs(x0-x1).max() < 1e-10


def test_Stiefel():
    batch_size = 3
    for dim,rank in [(7,1), (7,3), (7,7)]:
        for dtype in [torch.float64, torch.complex128]:
            for method in ['choleskyL','qr','so-cayley','so-exp','sqrtm']:
                manifold = numqi.manifold.Stiefel(dim, rank, batch_size, method=method, dtype=dtype)
                x0 = manifold().detach().numpy()
                assert np.abs(x0.conj().transpose(0,2,1) @ x0 - np.eye(rank)).max() < 1e-10
                tmp0 = manifold.theta.detach().numpy()
                if method=='cholesky':
                    x1 = numqi.manifold.to_stiefel_choleskyL(tmp0, dim, rank)
                    assert np.abs(x0-x1).max() < 1e-10
                elif method=='qr':
                    x1 = numqi.manifold.to_stiefel_qr(tmp0, dim, rank)
                    assert np.abs(x0-x1).max() < 1e-10
                elif method=='sqrtm':
                    x1 = numqi.manifold.to_stiefel_sqrtm(tmp0, dim, rank)
                    assert np.abs(x0-x1).max() < 1e-10

def test_SpecialOrthogonal():
    batch_size = 3
    dim = 7
    for dtype in [torch.float64,torch.complex128]:
        for method in ['exp','cayley']:
            dtype = torch.complex128
            manifold = numqi.manifold.SpecialOrthogonal(dim, batch_size, method=method, dtype=dtype)
            tmp0 = manifold()
            assert tmp0.dtype==dtype
            x0 = tmp0.detach().numpy()
            assert np.abs(x0 @ x0.transpose(0,2,1).conj() - np.eye(dim)).max() < 1e-10

            tmp0 = manifold.theta.detach().numpy()
            if method=='exp':
                x1 = numqi.manifold.to_special_orthogonal_exp(tmp0, dim)
            else:
                x1 = numqi.manifold.to_special_orthogonal_cayley(tmp0, dim)
            assert np.abs(x0-x1).max() < 1e-10


def naive_lu_decomposition(mat):
    # https://stackoverflow.com/a/55131490/7290857
    # scipy.linalg.lu always do the pivoting which is not what we want
    assert (mat.ndim==2)
    mat = mat.copy()
    m,n = mat.shape
    for k in range(min(m,n)-1):
        for i in range(k+1,m):
            mat[i,k] /= mat[k,k]
            tmp0 = np.arange(k+1,n)
            mat[i,tmp0] -= mat[i,k]*mat[k,tmp0]
    if m>n:
        mat[n:,n-1] /= mat[n-1,n-1]
    if m>=n:
        matL = np.tril(mat,-1) + np.eye(m,n)
        matU = np.triu(mat)[:n]
    else:
        matL = np.tril(mat,-1)[:,:m] + np.eye(m,m)
        matU = np.triu(mat)
    return matL,matU

def test_naive_lu_decomposition():
    np_rng = np.random.default_rng()
    for _ in range(100):
        dim = np_rng.integers(2, 20, size=2)
        mat = np_rng.normal(size=dim) + 1j*np_rng.normal(size=dim)
        matL,matU = naive_lu_decomposition(mat)
        assert np.abs(np.triu(matL,1)).max() < 1e-10
        assert np.abs(np.tril(matU,-1)).max() < 1e-10
        assert np.abs(np.diag(matL)-1).max() < 1e-10
        assert np.abs(mat - matL@matU).max() < 1e-10

import numpy as np

def get_qutrit_projector_basis(num_qutrit=1):
    # TODO with-identity
    tmp0 = [
        [(0,1)], [(1,1)], [(2,1)],
        [(0,1),(1,1)], [(0,1),(1,-1)], [(0,1),(1,1j)], [(0,1),(1,-1j)],
        [(0,1),(2,1)], [(0,1),(2,-1)], [(0,1),(2,1j)], [(0,1),(2,-1j)],
        [(1,1),(2,1)], [(1,1),(2,-1)], [(1,1),(2,1j)], [(1,1),(2,-1j)],
    ]
    matrix_subspace = []
    for x in tmp0:
        tmp1 = np.zeros(3, dtype=np.complex128)
        tmp1[[y[0] for y in x]] = [y[1] for y in x]
        matrix_subspace.append(tmp1[:,np.newaxis] * tmp1.conj())
    matrix_subspace = np.stack(matrix_subspace)

    if num_qutrit>=1:
        tmp0 = matrix_subspace
        for _ in range(num_qutrit-1):
            tmp1 = np.einsum(tmp0, [0,1,2], matrix_subspace, [3,4,5], [0,3,1,4,2,5], optimize=True)
            tmp2 = [x*y for x,y in zip(tmp0.shape, matrix_subspace.shape)]
            tmp0 = tmp1.reshape(tmp2)
        matrix_subspace = tmp0
    tmp0 = np.eye(matrix_subspace.shape[1])[np.newaxis]
    matrix_subspace = np.concatenate([tmp0,matrix_subspace], axis=0)
    return matrix_subspace


hf_chebval_n = lambda x, n: np.polynomial.chebyshev.chebval(x, np.array([0]*n+[1]))*(1 if n==0 else np.sqrt(2))

def get_chebshev_orthonormal(dim_qudit, alpha, with_computational_basis=False, return_basis=False):
    # with_computational_basis=False: 4PB
    # with_computational_basis=True: 5PB
    rootd = np.cos(np.pi*(np.arange(dim_qudit)+0.5)/dim_qudit)
    basis0 = np.stack([hf_chebval_n(rootd, x) for x in range(dim_qudit)], axis=1)/np.sqrt(dim_qudit)

    rootd1 = np.cos(np.pi*(np.arange(dim_qudit-1)+0.5)/(dim_qudit-1))
    tmp1 = np.stack([hf_chebval_n(rootd1, x) for x in range(dim_qudit)], axis=1)/np.sqrt(dim_qudit-1)
    tmp2 = np.array([0]*(dim_qudit-1)+[1])
    basis1 = np.concatenate([tmp1,tmp2[np.newaxis]], axis=0)

    basis2 = np.stack([hf_chebval_n(rootd, x)*np.exp(1j*alpha*x) for x in range(dim_qudit)], axis=1)/np.sqrt(dim_qudit)

    tmp1 = np.stack([hf_chebval_n(rootd1, x)*np.exp(1j*alpha*x) for x in range(dim_qudit)], axis=1)/np.sqrt(dim_qudit-1)
    tmp2 = np.array([0]*(dim_qudit-1)+[1])
    basis3 = np.concatenate([tmp1,tmp2[np.newaxis]], axis=0)

    basis_list = [basis0,basis1,basis2,basis3]
    if with_computational_basis:
        basis_list.append(np.eye(dim_qudit))
        tmp0 = np.eye(dim_qudit)

    tmp0 = np.concatenate(basis_list, axis=0)
    ret = tmp0[:,:,np.newaxis]*(tmp0[:,np.newaxis].conj())
    if return_basis:
        ret = ret,basis_list
    return ret

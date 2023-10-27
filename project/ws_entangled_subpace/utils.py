import numpy as np
import numqi 
import cvxpy

def get_bipartite_ces(d, theta, random_basis=False, xi=0):
    vec_list = []
    if random_basis:
        random_unitary = numqi.random.rand_unitary_matrix(d)
        basis = [random_unitary[:,i].T for i in range(d)]  
    else:
        basis = [np.eye(d)[i] for i in range(d)] 
    for i in range(d-1):
        psi_i = np.cos(theta/2)*np.kron(np.eye(2)[0], basis[i]) + np.exp(1j*xi)*np.sin(theta/2)*np.kron(np.eye(2)[1], basis[i+1])
        vec_list.append(psi_i)
    # return a numpy array
    return np.array(vec_list)

def analytical_GM(d, theta):
    tmp0 = np.sin(theta)**2*np.sin(np.pi/d)**2
    tmp1 = np.sqrt(1-tmp0)
    tmp2 = (1/2)*(1-tmp1)
    return tmp2

def PPT_GM(basis_orth):
    # basis_orth: a numpy array of shape (num_basis, dimA, dimB)
    num_basis = basis_orth.shape[0]
    dimA = basis_orth.shape[1]
    dimB = basis_orth.shape[2]
    dim = dimA*dimB
    basis = basis_orth.reshape(num_basis, dimA*dimB)
    proj_list = [vec[:, np.newaxis] @ vec[np.newaxis, :].conj() for vec in basis]
    # sum up all the projectors
    P = np.sum(proj_list, axis=0)
    rho = cvxpy.Variable((dim, dim), hermitian=True)
    constraints = [
        rho>>0,
        cvxpy.trace(rho)==1,
        cvxpy.partial_transpose(rho, [dimA,dimB], 1)>>0,
    ]
    objective = cvxpy.Minimize(cvxpy.real(cvxpy.trace(P @ rho)))
    prob = cvxpy.Problem(objective, constraints)
    prob.solve(solver=cvxpy.MOSEK)
    return prob.value

def random_hermitian(d, trace_norm):
    Z = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    H = Z + Z.conj().T
    eigvals = np.linalg.eigvalsh(H)
    tr_norm = np.sum(np.abs(eigvals))
    H = H /tr_norm  * trace_norm
    return H
import functools
import numpy as np
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from tqdm import tqdm
import cvxpy
import torch

import numqi.optimize
import numqi.gate
import numqi.gellmann
import numqi.dicke

hf_kron = lambda *x: functools.reduce(np.kron, x)

def get_maximum_entropy_model_boundary(model, radius, index=(0,1),
            term_value_target=None, num_point=100, num_repeat=1, tol=1e-10):
    theta_list = np.linspace(0, 2*np.pi, num_point)
    num_term = model.term.shape[0]
    if term_value_target is None:
        term_value_target = np.zeros((num_point, num_term), dtype=np.float64)
    else:
        assert term_value_target.ndim==1 and (term_value_target.shape[0]==num_term)
        term_value_target = np.ones((num_point,1)) * term_value_target
    term_value_target[:,index[0]] = radius*np.cos(theta_list)
    term_value_target[:,index[1]] = radius*np.sin(theta_list)

    term_value_list = []
    EVL_list = []
    for ind0 in tqdm(range(num_point)):
        model.set_target(term_value_target[ind0])
        numqi.optimize.minimize(model, theta0='uniform', num_repeat=num_repeat, tol=tol, print_freq=0, print_every_round=0)
        term_value_list.append(model.term_value.detach().numpy().copy())
        EVL_list.append(np.linalg.eigvalsh(model.dm_torch.detach().numpy()))
    term_value_list = np.stack(term_value_list, axis=0)
    EVL_list = np.stack(EVL_list, axis=0)
    return term_value_target, term_value_list, EVL_list


def draw_maximum_entropy_model_boundary(term_value_target, term_value_list, EVL_list, index=(0,1),
            witnessA=None, witnessC=None, rank_radius=0.2, zero_eps=1e-4, show_target=True):
    if rank_radius>0:
        tmp0 = np.angle(term_value_list[:,index[0]] + 1j*term_value_list[:,index[1]])
        tmp1 = (EVL_list>zero_eps).sum(axis=1)
        rank_list = rank_radius*np.stack([tmp1*np.cos(tmp0), tmp1*np.sin(tmp0)], axis=1)

    fig,ax = plt.subplots()
    if show_target:
        tmp0 = np.stack([term_value_target[:,index[0]], term_value_list[:,index[0]], term_value_list[:,index[0]]*np.nan], axis=1).reshape(-1)
        tmp1 = np.stack([term_value_target[:,index[1]], term_value_list[:,index[1]], term_value_list[:,index[1]]*np.nan], axis=1).reshape(-1)
        ax.plot(term_value_target[:,index[0]], term_value_target[:,index[1]], '-', label='target')
    ax.plot(term_value_list[:,index[0]], term_value_list[:,index[1]], label='boundary')
    ax.plot(tmp0, tmp1, '--')
    tmp0 = np.linspace(0, 2*np.pi)
    if rank_radius>0:
        ax.plot(rank_radius*np.cos(tmp0), rank_radius*np.sin(tmp0), ':', label='rank=1')
        ax.plot(rank_list[:,0], rank_list[:,1], label=r'rank')
    ax.set_aspect('equal')
    if witnessA is not None:
        assert witnessC is not None
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        tmp0 = term_value_list - witnessA
        tmp0[:,index[0]] = 0
        tmp0[:,index[1]] = 0
        bias = -np.dot(tmp0, witnessC).max()
        if abs(witnessC[index[0]])>abs(witnessC[index[1]]):
            tmp0 = [bias-(y-witnessA[index[1]])*witnessC[index[1]] for y in ylim]
            xdata = [(x/witnessC[index[0]]+witnessA[index[0]]) for x in tmp0]
            ydata = ylim
        else:
            xdata = xlim
            tmp0 = [bias-(x-witnessA[index[0]])*witnessC[index[0]] for x in xlim]
            ydata = [(x/witnessC[index[1]]+witnessA[index[1]]) for x in tmp0]
        ax.plot(xdata, ydata, linestyle='dashed', label='witness')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    ax.legend()
    fig.tight_layout()
    return fig,ax


def get_1dchain_2local_pauli_basis(num_qubit, with_I=False):
    assert num_qubit>=2
    pauli_list = [numqi.gate.I, numqi.gate.X, numqi.gate.Y, numqi.gate.Z]
    pauli2_list = [np.kron(x,y) for x in pauli_list for y in pauli_list][1:]
    ret = [np.eye(num_qubit)] if with_I else []
    for ind0 in range(num_qubit-1):
        tmp0 = [None,None,None]
        if ind0>0:
            tmp0[0] = np.eye(2**ind0)
        if ind0<num_qubit-2:
            tmp0[2] = np.eye(2**(num_qubit-2-ind0))
        for ind1 in range(len(pauli2_list)):
            tmp0[1] = pauli2_list[ind1]
            ret.append(hf_kron(*[x for x in tmp0 if x is not None]))
    ret = np.stack(ret, axis=0)
    return ret


def sdp_2local_rdm_solve(term_value_target):
    assert (term_value_target.shape[0]%15)==0
    op_2qubit_flat = get_1dchain_2local_pauli_basis(2, with_I=False).transpose(0,2,1).reshape(-1,16)
    num_qubit = term_value_target.shape[0]//15 + 1
    cvxX = cvxpy.Variable((2**num_qubit,2**num_qubit), hermitian=True)
    tmp0 = term_value_target.reshape(-1, 15)
    constraints = [cvxX>>0, cvxpy.real(cvxpy.trace(cvxX))==1]
    rho_rdm_list = []
    for ind0 in range(num_qubit-1):
        if ind0>0:
            rho_rdm = cvxpy.partial_trace(cvxX, [2**ind0, 2**(num_qubit-ind0)], axis=0)
        else:
            rho_rdm = cvxX
        if ind0<(num_qubit-2):
            rho_rdm = cvxpy.partial_trace(rho_rdm, [4, rho_rdm.shape[0]//4], axis=1)
        rho_rdm_list.append(rho_rdm)
        constraints.append(cvxpy.real(op_2qubit_flat @ cvxpy.reshape(rho_rdm, 16, order='C'))==tmp0[ind0])
    prob = cvxpy.Problem(cvxpy.Minimize(1), constraints)
    prob.solve()
    return cvxX.value


def sdp_op_list_solve(op_list, term_value_target):
    assert (op_list.ndim==3) and (op_list.shape[1]==op_list.shape[2])
    num_qubit = int(np.log2(op_list.shape[1]))
    cvxX = cvxpy.Variable((2**num_qubit,2**num_qubit), hermitian=True)
    tmp0 = op_list.reshape(op_list.shape[0],-1) @ cvxpy.reshape(cvxX, 4**num_qubit)
    constraints = [
        cvxX>>0,
        cvxpy.real(cvxpy.trace(cvxX))==1,
        tmp0==term_value_target,
    ]
    prob = cvxpy.Problem(cvxpy.Minimize(1), constraints)
    prob.solve() #fail with mosek
    return cvxX.value


def get_ABk_gellmann_preimage_op(dimA:int, dimB:int, kext:int, kind:str='boson'):
    kind = kind.lower()
    assert kind in {'boson','symmetric'}
    assert (dimA>=2) and (dimB>=2) and (kext>=2)
    N0 = dimA*dimB*dimA*dimB-1
    matG = numqi.gellmann.all_gellmann_matrix(dimA*dimB, with_I=False).reshape(N0,dimA,dimB,dimA,dimB)
    if kind=='boson':
        Brsab = numqi.dicke.get_partial_trace_ABk_to_AB_index(kext, dimB, return_tensor=True)
        tmp0 = dimA*Brsab.shape[2]
        ret = np.einsum(matG, [0,1,2,3,4], Brsab, [2,4,5,6], [0,1,5,3,6], optimize=True).reshape(N0, tmp0, tmp0)
    else:
        ret = 0
        for ind0 in range(1,kext+1):
            tmp0 = matG
            if ind0>1:
                tmp1 = np.eye(dimB**(ind0-1)).reshape(dimB**(ind0-1),1,dimB**(ind0-1),1)
                tmp0 = (tmp0.reshape(N0,dimA,1,dimB*dimA,1,dimB)*tmp1).reshape(N0,dimA*(dimB**ind0),dimA*(dimB**ind0))
            if ind0<kext:
                tmp1 = np.eye(dimB**(kext-ind0)).reshape(dimB**(kext-ind0),1,dimB**(kext-ind0))
                tmp0 = (tmp0.reshape(N0*dimA*(dimB**ind0),1,dimA*(dimB**ind0),1)*tmp1).reshape(N0,dimA*(dimB**kext),dimA*(dimB**kext))
            ret = ret + tmp0
        ret /= kext
    return ret


def eigvalsh_largest_power_iteration(matA, maxiter=20, vec0=None, tol=1e-7, tag_force_full=False):
    N0 = matA.shape[0]
    tmp0 = matA.detach().cpu().numpy()
    assert N0>=2
    if tag_force_full or (N0<5):
        EVL = np.linalg.eigvalsh(tmp0)
        EVL_min = EVL[0]
        EVL_max = EVL[-1]
    else:
        EVL_max = scipy.sparse.linalg.eigsh(tmp0, k=1, which='LA', return_eigenvectors=False)[0]
        EVL_min = scipy.sparse.linalg.eigsh(tmp0, k=1, which='SA', return_eigenvectors=False)[0]
    # (min,max)->(-0.5,1.0): y=ax+b
    y0 = -0.5
    a0 = (1-y0)/(EVL_max-EVL_min)
    b0 = y0-a0*EVL_min
    matB = matA*a0+b0*torch.eye(N0, dtype=matA.dtype, device=matA.device)
    if vec0 is None:
        vec0 = torch.randn(N0, dtype=matB.dtype, device=matB.device)
    for ind_step in range(1,maxiter+1):
        matB = matB @ matB
        vec1 = matB @ vec0
        vec1 = vec1 / torch.linalg.norm(vec1)
        if torch.linalg.norm(vec1-vec0) < tol:
            break
        vec0 = vec1
    return vec1, ind_step


# not good
class NANGradientToNumber(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, beta=0.01):
        ctx.save_for_backward(input)
        ctx._data = dict(beta=beta)
        return input

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        mask = torch.isnan(grad_output)
        if mask.any():
            tmp0 = torch.rand(grad_output.shape, dtype=grad_output.dtype, device=grad_output.device)
            tmp1 = tmp0*(2*ctx._data['beta'])-ctx._data['beta']
            ret = torch.where(torch.logical_not(mask), grad_output, tmp1)
        else:
            ret = grad_output
        return ret,None


def draw_line_list(ax, xydata, norm_theta_list, kind='norm', color='#ABABAB', radius=2.5, label=None):
    assert kind in {'norm', 'tangent'}
    N0 = len(norm_theta_list)
    assert xydata.shape==(N0,2)
    xdata = np.zeros((N0,3), dtype=np.float64)
    ydata = np.zeros((N0,3), dtype=np.float64)
    xdata[:,2] = np.nan
    ydata[:,2] = np.nan
    if kind=='tangent':
        xdata[:,0] = xydata[:,0] + radius*np.cos(norm_theta_list-np.pi/2)
        xdata[:,1] = xydata[:,0] + radius*np.cos(norm_theta_list+np.pi/2)
        ydata[:,0] = xydata[:,1] + radius*np.sin(norm_theta_list-np.pi/2)
        ydata[:,1] = xydata[:,1] + radius*np.sin(norm_theta_list+np.pi/2)
    else:
        xdata[:,0] = xydata[:,0]
        xdata[:,1] = xydata[:,0] + radius*np.cos(norm_theta_list)
        ydata[:,0] = xydata[:,1]
        ydata[:,1] = xydata[:,1] + radius*np.sin(norm_theta_list)
    ax.plot(xdata.reshape(-1), ydata.reshape(-1), color=color, label=label)


def get_supporting_plane_2d_projection(vecA, vecN, basis0, basis1, theta_list):
    assert (vecA.ndim==2) and (vecA.shape==vecN.shape)
    dim = vecA.shape[1]
    assert (basis0.shape==basis1.shape) and basis0.shape==(dim,)
    assert theta_list.shape==(vecA.shape[0],)
    basis0 = basis0/np.linalg.norm(basis0)
    basis1 = basis1 - np.dot(basis1, basis0)*basis0
    basis1 = basis1/np.linalg.norm(basis1)
    tmp0 = vecN @ basis0
    tmp1 = vecN @ basis1
    vec_proj_N = np.angle(tmp0 + 1j*tmp1)
    tmp2 = np.einsum(vecA, [0,1], vecN, [0,1], [0], optimize=True)
    tmp3 = tmp2 / (np.cos(theta_list)*tmp0 + np.sin(theta_list)*tmp1)
    vec_proj_A = np.stack([tmp3*np.cos(theta_list), tmp3*np.sin(theta_list)], axis=1)
    return vec_proj_A, vec_proj_N


def op_list_numerical_range_SDP(op_list, theta_list):
    op_list = np.stack(op_list, axis=0)
    num_op,dim,_ = op_list.shape
    assert num_op==2
    cvxX = cvxpy.Variable((dim,dim), hermitian=True)
    cvxB = cvxpy.Parameter(num_op)
    cvx_beta = cvxpy.Variable()
    cvxO = cvxpy.real(op_list.transpose(0,2,1).reshape(-1, dim*dim) @ cvxpy.reshape(cvxX, dim*dim, order='F'))
    constraints = [
        cvxX>>0,
        cvxpy.real(cvxpy.trace(cvxX))==1,
        cvx_beta*cvxB==cvxO,
    ]
    obj = cvxpy.Maximize(cvx_beta)
    prob = cvxpy.Problem(obj, constraints)
    theta_list = np.asarray(theta_list).reshape(-1)
    ret = []
    for theta_i in theta_list:
        cvxB.value = np.array([np.cos(theta_i), np.sin(theta_i)])
        prob.solve()
        ret.append((cvx_beta.value.item(), cvxO.value.copy(), constraints[-1].dual_value.copy()))
    tmp0 = np.array([x[0] for x in ret])
    tmp1 = np.stack([x[1] for x in ret], axis=0)
    tmp2 = np.stack([x[2] for x in ret], axis=0)
    return tmp0,tmp1,tmp2

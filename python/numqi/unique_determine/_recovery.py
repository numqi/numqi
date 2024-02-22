import numpy as np
import torch
import cvxpy

import numqi.random
import numqi.optimize

class FindStateWithOpModel(torch.nn.Module):
    def __init__(self, op_list, use_dm:bool, dtype:str='float64'):
        super().__init__()
        assert dtype in {'float32','float64'}
        assert (op_list.ndim==3) and (op_list.shape[1]==op_list.shape[2])
        self.cdtype = torch.complex64 if (dtype=='float32') else torch.complex128
        if use_dm:
            self.manifold = numqi.manifold.Trace1PSD(op_list.shape[1], dtype=self.cdtype)
        else:
            self.manifold = numqi.manifold.Sphere(op_list.shape[1], dtype=self.cdtype)
        self.op_list = torch.tensor(op_list, dtype=self.cdtype)
        self.expectation = None
        self.state = None

    def set_expectation(self, x):
        assert len(x)==self.op_list.shape[0]
        tmp0 = torch.float32 if (self.cdtype==torch.complex64) else torch.float64
        self.expectation = torch.tensor(x, dtype=tmp0)

    def get_state(self):
        with torch.no_grad():
            self()
        ret = self.state.detach().numpy().copy()
        return ret

    def forward(self):
        assert self.expectation is not None
        state = self.manifold()
        if state.ndim==1: #pure state vector
            tmp0 = ((self.op_list @ state) @ state.conj()).real
        else: #density matrix
            tmp0 = (self.op_list.reshape(self.op_list.shape[0],-1) @ state.reshape(-1).conj()).real
        self.state = state.detach()
        loss = torch.mean((tmp0 - self.expectation)**2)
        return loss


def check_UD_is_UD(matB, kind='uda', num_round=100, num_repeat_sgd=10, zero_eps=1e-7, use_sdp=False, seed=None):
    kind = kind.lower()
    assert kind in {'uda','udp'}
    assert (matB.ndim==3) and (matB.shape[1]==matB.shape[2])
    np_rng = numqi.random.get_numpy_rng(seed)
    dim = matB.shape[1]
    if use_sdp:
        assert kind!='udp', 'udp is not a SDP'
        assert kind!='uda', 'not implemented yet'
    else:
        model = FindStateWithOpModel(matB, use_dm=(kind=='uda'))
        for _ in range(num_round):
            state = numqi.random.rand_haar_state(dim, tag_complex=True, seed=np_rng)
            matB_exp = ((matB @ state) @ state.conj()).real
            model.set_expectation(matB_exp)
            theta_optim0 = numqi.optimize.minimize(model, theta0='uniform', num_repeat=num_repeat_sgd,
                            tol=zero_eps/10, early_stop_threshold=zero_eps/10, print_every_round=0, print_freq=0)
            state1 = model.get_state()
            assert theta_optim0.fun < zero_eps
            if kind=='uda':
                assert abs(((state1 @ state) @ state.conj()).real-1) < zero_eps
            else:
                assert abs(abs(np.dot(state1.conj(), state)) - 1) < zero_eps


def density_matrix_recovery_SDP(op_list, measure, converge_eps=None):
    dim = op_list.shape[1]
    rho = cvxpy.Variable((dim,dim), hermitian=True)
    tmp0 = np.asarray(op_list).reshape(-1, dim*dim).T
    tmp1 = cvxpy.real(cvxpy.reshape(rho, (dim*dim,), order='F') @ tmp0)
    # objective = cvxpy.Minimize(cvxpy.sum_squares(tmp1 - measure))
    objective = cvxpy.Minimize(cvxpy.norm(tmp1-measure, 2))
    constraints = [rho>>0, cvxpy.trace(rho)==1]
    prob = cvxpy.Problem(objective, constraints)
    if converge_eps is not None:
        # TODO mosek is faster
        prob.solve(solver=cvxpy.SCS, eps=converge_eps)
    else:
        prob.solve()
    return np.ascontiguousarray(rho.value), prob.value

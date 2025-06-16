import numpy as np
import torch
import cvxpy

import numqi.random
import numqi.optimize

class FindStateWithOpModel(torch.nn.Module):
    '''recover quantum states from expectation values of a list of operators'''
    def __init__(self, op_list:np.ndarray, use_dm:bool, dtype:str='float64'):
        r'''initialize the model

        Parameters:
            op_list (np.ndarray): measurement operator, `ndim=3`, `shape=(num_op, dim, dim)`
            use_dm (bool): if `True`, the model will optimize over density matrices, otherwise pure states
            dtype (str): data type of the model, either `'float32'` or `'float64'`
        '''
        super().__init__()
        assert dtype in {'float32','float64'}
        assert (op_list.ndim==3) and (op_list.shape[1]==op_list.shape[2])
        assert np.abs(op_list-op_list.transpose(0,2,1).conj()).max() < 1e-10
        self.cdtype = torch.complex64 if (dtype=='float32') else torch.complex128
        if use_dm:
            self.manifold = numqi.manifold.Trace1PSD(op_list.shape[1], dtype=self.cdtype)
        else:
            self.manifold = numqi.manifold.Sphere(op_list.shape[1], dtype=self.cdtype)
        self.op_list = torch.tensor(op_list, dtype=self.cdtype)
        self.expectation = None
        self.state:None|torch.Tensor = None

    def set_expectation(self, x:np.ndarray):
        r'''set the expectation values of the operators

        Parameters:
            x (np.ndarray): expectation values, `shape=(num_op,)`
        '''
        assert len(x)==self.op_list.shape[0]
        tmp0 = torch.float32 if (self.cdtype==torch.complex64) else torch.float64
        self.expectation = torch.tensor(x, dtype=tmp0)

    def get_state(self):
        r'''get the recovered state

        Returns:
            ret (np.ndarray): recovered state, `shape=(dim,)` (pure state) or `shape=(dim,dim)` (density matrix)
        '''
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


def check_UD_is_UD(op_list:np.ndarray, kind:str='uda', num_round:int=100, num_repeat_sgd:int=10, zero_eps:float=1e-7, seed:int|None=None):
    r'''check if the given operators are unique-determine (UD) operators
    Raise AssertionError if the operators are not UD

    Parameters:
        op_list (np.ndarray): measurement operator, `ndim=3`, `shape=(num_op, dim, dim)`
        kind (str): `'uda'` or `'udp'`
        num_round (int): number of rounds for the test
        num_repeat_sgd (int): number of SGD repeats for each round
        zero_eps (float): tolerance for the test
        seed (int,None): random seed
    '''
    kind = kind.lower()
    assert kind in {'uda','udp'}
    assert (op_list.ndim==3) and (op_list.shape[1]==op_list.shape[2])
    np_rng = numqi.random.get_numpy_rng(seed)
    dim = op_list.shape[1]
    # TODO SDP for kind='uda'
    model = FindStateWithOpModel(op_list, use_dm=(kind=='uda'))
    for _ in range(num_round):
        state = numqi.random.rand_haar_state(dim, tag_complex=True, seed=np_rng)
        matB_exp = ((op_list @ state) @ state.conj()).real
        model.set_expectation(matB_exp)
        theta_optim0 = numqi.optimize.minimize(model, theta0='uniform', num_repeat=num_repeat_sgd,
                        tol=zero_eps/10, early_stop_threshold=zero_eps/10, print_every_round=0, print_freq=0)
        state1 = model.get_state()
        assert theta_optim0.fun < zero_eps
        if kind=='uda':
            assert abs(((state1 @ state) @ state.conj()).real-1) < zero_eps
        else:
            assert abs(abs(np.dot(state1.conj(), state)) - 1) < zero_eps


def density_matrix_recovery_SDP(op_list:np.ndarray, measure:np.ndarray, converge_eps:float|None=None):
    r'''recover density matrix from expectation values of a list of operators using SDP

    Parameters:
        op_list (np.ndarray): measurement operator, `ndim=3`, `shape=(num_op, dim, dim)`
        measure (np.ndarray): expectation values, `shape=(num_op,)`
        converge_eps (float,None): convergence tolerance for the SDP solver, if not None, use `solver=SCS`

    Returns:
        rho (np.ndarray): recovered density matrix, `shape=(dim,dim)`
        value (float): objective value of the SDP
    '''
    dim = op_list.shape[1]
    rho = cvxpy.Variable((dim,dim), hermitian=True)
    tmp0 = np.asarray(op_list).reshape(-1, dim*dim).T
    tmp1 = cvxpy.real(cvxpy.reshape(rho, (dim*dim,), order='F') @ tmp0)
    # obj = cvxpy.Minimize(cvxpy.sum_squares(tmp1 - measure))
    obj = cvxpy.Minimize(cvxpy.norm(tmp1-measure, 2))
    constraints = [rho>>0, cvxpy.trace(rho)==1]
    prob = cvxpy.Problem(obj, constraints)
    if converge_eps is not None:
        # TODO mosek is faster
        prob.solve(solver=cvxpy.SCS, eps=converge_eps)
    else:
        prob.solve()
    return np.ascontiguousarray(rho.value), prob.value

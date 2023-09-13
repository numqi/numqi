import numpy as np
import scipy.linalg
import scipy.special
import torch

import numqi

np_rng = np.random.default_rng()


def real_parameter_to_bound(theta, lower, upper):
    assert lower < upper
    if isinstance(theta, torch.Tensor):
        tmp0 = torch.sigmoid(theta) #1/(1+exp(-theta))
    else:
        tmp0 = scipy.special.expit(theta)
    ret = tmp0 * (upper - lower) + lower
    return ret

class DummyCharacterizeHamiltonianModel(torch.nn.Module):
    def __init__(self, term_list, initial_state, measure_op, t_measure_list, bound, measurement_result):
        super().__init__()
        assert (term_list.ndim==3) and np.abs(term_list - term_list.transpose(0,2,1).conj()).max() < 1e-10
        self.term_list = torch.tensor(term_list, dtype=torch.complex128)
        assert (len(bound)==2) and (bound[0] < bound[1])
        self.bound = float(bound[0]), float(bound[1])
        self.initial_state = torch.tensor(initial_state, dtype=torch.complex128)
        self.measure_op = torch.tensor(measure_op, dtype=torch.complex128)
        self.t_measure_list = [float(x) for x in t_measure_list]
        self.measurement_result = torch.tensor(measurement_result, dtype=torch.float64)

        np_rng = np.random.default_rng()
        self.theta = torch.nn.Parameter(torch.tensor(np_rng.uniform(-1, 1, size=len(term_list)), dtype=torch.float64))
        self._remove_bound = False

    def get_coefficient(self):
        ret  = real_parameter_to_bound(self.theta, *self.bound)
        return ret

    def forward(self):
        coeff = self.theta if self._remove_bound else self.get_coefficient()
        N0 = self.term_list.shape[1]
        tmp0 = (coeff.to(self.term_list.dtype) @ self.term_list.reshape(-1, N0*N0)).reshape(N0,N0)
        hamiltonian = tmp0 - (torch.trace(tmp0)/N0) * torch.eye(N0, dtype=torch.complex128) #trace is meaningless
        expectation = []
        for x in self.t_measure_list:
            tmp0 = (torch.linalg.matrix_exp(-1j*x*hamiltonian)@self.initial_state.T)
            expectation.append(torch.einsum(self.measure_op, [2,0,1], tmp0, [1,2], tmp0.conj(), [0,2], [2]).real)
        expectation = torch.stack(expectation, dim=1)
        loss = torch.mean((expectation - self.measurement_result)**2)
        return loss

    def get_hessian(self):
        theta_backup = self.theta.detach().clone()
        self.theta.data[:] = self.get_coefficient().detach()
        self._remove_bound = True
        ret = numqi.optimize.get_model_hessian(self)
        self._remove_bound = False
        self.theta.data[:] = theta_backup
        return ret

def generate_random_data(num_measure=20, seed=None):
    # https://docs.q-ctrl.com/boulder-opal/user-guides/how-to-perform-parameter-estimation-with-a-small-amount-of-data
    time_duration = 0.5 #in microsecond
    t_measure_list = np.linspace(1/num_measure, 1, num_measure)*time_duration
    frequency_bound = num_measure / time_duration #MHz
    parameter = (2 * np.pi) * np.array([0.5, 1.5, 1.8])  # MHz

    np_rng = np.random.default_rng(seed)
    tmp0 = np.array([1,1]) / np.sqrt(2)
    tmp1 = np.array([1,1j]) / np.sqrt(2)
    tmp2 = np.array([1,0])
    initial_state = np.array([tmp2, tmp0, tmp1])
    measure_op = np.stack([x.reshape(-1,1)*x.conj() for x in [tmp1,tmp2,tmp0]], axis=0)

    hamiltonian = (parameter[0] * numqi.gate.X + parameter[1] * numqi.gate.Y + parameter[2] * numqi.gate.Z)/ 2
    tmp0 = np.einsum(scipy.linalg.expm((-1j*t_measure_list).reshape(-1,1,1)*hamiltonian), [0,1,2], initial_state, [3,2], [3,0,1], optimize=True)
    expectation = np.einsum(measure_op, [0,1,2], tmp0, [0,3,2], tmp0.conj(), [0,3,1], [0,3], optimize=True).real
    noise = np_rng.normal(0, 0.01, size=expectation.shape)
    measurement_result = expectation + noise
    return measurement_result, t_measure_list, frequency_bound, initial_state, measure_op, parameter


def get_confidence_ellipse_matrix(hessian, loss, num_data, confidence_fraction=0.95):
    # https://docs.q-ctrl.com/references/qctrl-visualizer/qctrlvisualizer/confidence_ellipse_matrix.html
    # https://docs.q-ctrl.com/boulder-opal/topics/characterizing-your-hardware-using-system-identification-in-boulder-opal
    num_parameter = hessian.shape[0]
    covariance_matrix = np.linalg.inv(((num_data - num_parameter) / (2*loss))*hessian)
    tmp0 = scipy.special.betaincinv((num_data-num_parameter)/2, num_parameter/2, 1-confidence_fraction)
    tmp1 = (num_data - num_parameter) / num_parameter * (1 / tmp0 - 1)
    ret = np.sqrt(num_parameter * tmp1) * scipy.linalg.sqrtm(covariance_matrix)
    return ret


measurement_result, t_measure_list, frequency_bound, initial_state, measure_op, parameter_real = generate_random_data()
term_list = np.stack([numqi.gate.X, numqi.gate.Y, numqi.gate.Z], axis=0)/2

model = DummyCharacterizeHamiltonianModel(term_list, initial_state,
            measure_op, t_measure_list, (-frequency_bound, frequency_bound), measurement_result)
theta_optim = numqi.optimize.minimize(model, ('uniform',-1,1), num_repeat=3, tol=1e-7)

parameter_estimate = model.get_coefficient()
hessian = model.get_hessian()


confidence_region = get_confidence_ellipse_matrix(hessian, model().item(), measurement_result.size)

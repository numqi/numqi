import numpy as np
import torch

class GrapeModel(torch.nn.Module):
    def __init__(self, ham_drift, ham_control, tspan, smooth_weight=0.01):
        super().__init__()
        deltaT = tspan[1:] - tspan[:-1]
        assert (deltaT.ndim==1) and (deltaT.shape[0]>0) and (deltaT.min()>0)
        self.deltaT = torch.tensor(deltaT, dtype=torch.float64)
        self.tspan = tspan
        assert smooth_weight >= 0
        self.smooth_weight = smooth_weight
        assert (ham_drift.ndim==2) and (ham_drift.shape[0]==ham_drift.shape[1])
        N0 = ham_drift.shape[0]
        assert (ham_control.ndim==3) and (ham_control.shape[1:]==(N0,N0))
        self.ham_drift = torch.tensor(ham_drift, dtype=torch.complex128)
        self.ham_control = torch.tensor(ham_control, dtype=torch.complex128)
        tmp0 = np.random.default_rng().uniform(-0.001, 0.001, size=(len(tspan),ham_control.shape[0]))
        self.theta = torch.nn.Parameter(torch.tensor(tmp0, dtype=torch.float64))

        self.q_init = None
        self.q_target = None
        self.q_predict = None
        self.fidelity = None

    def set_state_vector(self, q_init, q_target):
        assert (q_init.ndim==1) and (q_init.shape[0]==self.ham_drift.shape[0])
        assert q_init.shape==q_target.shape
        self.q_init = torch.tensor(q_init/np.linalg.norm(q_init), dtype=torch.complex128)
        self.q_target = torch.tensor(q_target/np.linalg.norm(q_target), dtype=torch.complex128)

    def forward(self):
        q0 = self.q_init
        N0 = q0.shape[0]
        for ind0 in range(self.deltaT.shape[0]):
            tmp0 = (self.theta[ind0] + self.theta[ind0+1])/2
            tmp1 = self.ham_drift + (tmp0.to(self.ham_control.dtype) @ self.ham_control.reshape(-1,N0*N0)).reshape(N0,N0)
            q0 = torch.linalg.matrix_exp((-1j*self.deltaT[ind0])*tmp1) @ q0
        self.q_predict  = q0.detach()
        tmp0 = torch.vdot(q0, self.q_target)
        fidelity = (tmp0*tmp0.conj()).real
        self.fidelity = fidelity.detach()
        loss_smooth = self.smooth_weight * torch.sum((self.theta[1:] - self.theta[:-1])**2)
        loss = loss_smooth - fidelity
        return loss

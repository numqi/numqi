import functools
import types
import numpy as np

from ..utils import is_torch

try:
    import torch
except ImportError:
    pass


def _make_pauli():
    s0=np.array([[1.0, 0.0], [0.0, 1.0]])
    sx=np.array([[0.0, 1.0], [1.0, 0.0]])
    sy=np.array([[0.0, -1j], [1j, 0.0]])
    sz=np.array([[1.0, 0.0], [0.0, -1.0]])
    ret = types.SimpleNamespace(
        s0 = s0,
        sx = sx,
        sy = sy,
        sz = sz,
        s0s0 = np.kron(s0, s0),
        s0sx = np.kron(s0, sx),
        s0sy = np.kron(s0, sy),
        s0sz = np.kron(s0, sz),
        sxs0 = np.kron(sx, s0),
        sxsx = np.kron(sx, sx),
        sxsy = np.kron(sx, sy),
        sxsz = np.kron(sx, sz),
        sys0 = np.kron(sy, s0),
        sysx = np.kron(sy, sx),
        sysy = np.kron(sy, sy),
        sysz = np.kron(sy, sz),
        szs0 = np.kron(sz, s0),
        szsx = np.kron(sz, sx),
        szsy = np.kron(sz, sy),
        szsz = np.kron(sz, sz),
    )
    return ret

pauli = _make_pauli()
I = pauli.s0
X = pauli.sx
Y = pauli.sy
Z = pauli.sz
H = np.array([[1,1],[1,-1]])/np.sqrt(2) #Hadamard
S = np.array([[1,0],[0,1j]])
T = np.array([[1,0],[0,np.exp(1j*np.pi/4)]])
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
CZ = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])
Swap = np.array([[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]])

def pauli_exponential(a, theta, phi):
    r'''
    $$ exp\{ ia \hat{n} \cdot \vec{\sigma}\} $$

    see wiki https://en.wikipedia.org/wiki/Pauli_matrices in section "Exponential of a Pauli vector" equation (2)
    '''
    if is_torch(a):
        ca = torch.cos(a)
        sa = torch.sin(a)
        ct = torch.cos(theta)
        st = torch.sin(theta)
        cp = torch.cos(phi)
        sp = torch.sin(phi)
        tmp0 = ca + 1j*sa*ct
        tmp3 = ca - 1j*sa*ct
        tmp1 = sa*st*(sp + 1j*cp)
        tmp2 = sa*st*(-sp + 1j*cp)
        tmp4 = [x.view(-1,1) for x in (tmp0,tmp1,tmp2,tmp3)] #torch.stack error for float/complex mix input
        ret = torch.concat(tmp4, axis=-1).reshape(*a.shape, 2, 2)
    else:
        ca = np.cos(a)
        sa = np.sin(a)
        ct = np.cos(theta)
        st = np.sin(theta)
        cp = np.cos(phi)
        sp = np.sin(phi)
        tmp0 = ca + 1j*sa*ct
        tmp3 = ca - 1j*sa*ct
        tmp1 = sa*st*(sp + 1j*cp)
        tmp2 = sa*st*(-sp + 1j*cp)
        ret = np.stack([tmp0,tmp1,tmp2,tmp3], axis=-1).reshape(*ca.shape, 2, 2)
    return ret


def u3(theta, phi, lambda_):
    r'''
    $$ R_z( \phi ) R_y( \theta ) R_z( \lambda ) e^{0.5j (\phi+\lambda)} $$

    https://qiskit.org/documentation/stubs/qiskit.circuit.library.UGate.html?highlight=ugate#qiskit.circuit.library.UGate
    '''
    if is_torch(theta):
        ct = torch.cos(theta/2)
        st = torch.sin(theta/2)
        tmp0 = torch.exp(1j*lambda_)
        tmp1 = torch.exp(1j*phi)
        tmp2 = [ct,-st*tmp0,st*tmp1,ct*tmp0*tmp1]
        ret = torch.concat([x.view(-1,1) for x in tmp2], axis=-1).view(*ct.shape, 2, 2)
    else:
        ct = np.cos(theta/2)
        st = np.sin(theta/2)
        tmp0 = np.exp(1j*lambda_)
        tmp1 = np.exp(1j*phi)
        ret = np.stack([ct,-st*tmp0,st*tmp1,ct*tmp0*tmp1], axis=-1).reshape(*ct.shape, 2, 2)
    return ret


def get_quditX(d):
    # Weyl–Heisenberg matrices https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices
    ret = np.diag(np.ones(d-1), 1)
    ret[-1,0] = 1
    return ret


def get_quditH(d):
    # Weyl–Heisenberg matrices https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices
    tmp0 = np.exp(2j*np.pi*np.arange(d)/d)
    ret = np.vander(tmp0, d, increasing=True) / np.sqrt(d)
    return ret


def get_quditZ(d):
    # Weyl–Heisenberg matrices https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices
    ret = np.diag(np.exp(2j*np.pi*np.arange(d)/d))
    return ret


@functools.lru_cache
def _get_quditX_eigen(d:int, is_torch:bool):
    tmp0 = np.arange(d) - (np.arange(d)>(d/2))*d
    EVL_log = tmp0 * (2*np.pi / d)
    EVC = get_quditH(d)
    if is_torch:
        EVL_log_torch = torch.tensor(EVL_log, dtype=torch.complex128)
        EVC_torch = torch.tensor(EVC, dtype=torch.complex128)
        ret = EVL_log_torch, EVC_torch
    else:
        ret = EVL_log, EVC
    return ret


def _rx_qudit(theta, d):
    '''qudit Rx rotation
    1. periodic 4*np.pi
    2. special unitary
    3. when d=2, restore to qubit Rx
    '''
    assert d>1
    if is_torch(theta):
        EVL_log,EVC = _get_quditX_eigen(int(d), True)
        shape = theta.shape
        tmp0 = theta.view(-1,1,1)
        tmp1 = tmp0*EVL_log*(d/(2*np.pi))
        ret = (EVC*torch.exp(1j*tmp1)) @ EVC.T.conj()
        if d%2==0:
            ret = ret * torch.exp(-1j*tmp0/2)
        ret = ret.view(*shape, d, d)
    else:
        theta = np.asarray(theta)
        shape = theta.shape
        EVL_log,EVC = _get_quditX_eigen(int(d), False)
        tmp0 = theta.reshape(-1,1,1)
        tmp1 = tmp0*EVL_log*(d/(2*np.pi))  #periodic 4*np.pi
        ret = (EVC*np.exp(1j*tmp1)) @ EVC.T.conj()
        if d%2==0:
            ret = ret * np.exp(-1j*tmp0/2)
        ret = ret.reshape(*shape, d, d)
    return ret


def rx(theta, d=2):
    r'''
    $$ exp \{ -i\theta \sigma _x/2 \} $$
    '''
    if d==2:
        if is_torch(theta):
            # TODO maybe fixed at some release for pytorch
            # error when back-propagation without multiplying 1. I have no idea why
            # if theta.dtype==torch.float32:
            #     theta = theta*torch.tensor(1, dtype=torch.complex64)
            # elif theta.dtype==torch.float64:
            #     theta = theta*torch.tensor(1, dtype=torch.complex128)
            ca = torch.cos(theta/2)
            isa = 1j*torch.sin(theta/2)
            tmp0 = [ca,-isa,-isa,ca]
            ret = torch.concat([x.view(-1,1) for x in tmp0], axis=-1).view(*theta.shape,2,2)
        else:
            ca = np.cos(theta/2)
            isa = 1j*np.sin(theta/2)
            ret = np.stack([ca,-isa,-isa,ca], axis=-1).reshape(*ca.shape,2,2)
    else:
        ret = _rx_qudit(theta, d)
    return ret


def ry(theta):
    r'''
    $$ exp \{ -i\theta \sigma _y/2 \} $$
    '''
    if is_torch(theta):
        if theta.dtype==torch.float32:
            theta = theta*torch.tensor(1, dtype=torch.complex64) #if not to(complex), error in the torch.stack later
        elif theta.dtype==torch.float64:
            theta = theta*torch.tensor(1, dtype=torch.complex128)
        ca = torch.cos(theta/2)
        sa = torch.sin(theta/2)
        tmp0 = [ca,-sa,sa,ca]
        ret = torch.concat([x.view(-1,1) for x in tmp0], dim=-1).view(*theta.shape,2,2)
    else:
        ca = np.cos(theta/2)
        sa = np.sin(theta/2)
        ret = np.stack([ca,-sa,sa,ca], axis=-1).reshape(*ca.shape,2,2)
    return ret


def _rz_qudit(theta, d, diag_only):
    assert d>1
    if is_torch(theta):
        shape = theta.shape
        tmp0 = torch.exp(1j*theta.view(-1))
        ret = torch.linalg.vander(tmp0, N=d)
        # torch.vander not support autograd, use torch.linalg.vander instead
        # https://github.com/pytorch/pytorch/issues/60197
        ret = ret * torch.exp(-1j*(torch.arange(d)>(d/2))*d*theta.view(-1,1))
        if d%2==0:
            ret = ret * torch.exp(-1j*theta.reshape(-1,1)/2)
        if diag_only:
            ret = ret.view(*theta.shape, d)
        else:
            ret = torch.diag_embed(ret).view(*shape, d, d)
    else:
        theta = np.asarray(theta)
        shape = theta.shape
        tmp0 = np.exp(1j*theta.reshape(-1))
        ret = np.vander(tmp0, N=d, increasing=True)
        ret *= np.exp(-1j*(np.arange(d)>(d/2))*d*theta.reshape(-1,1))
        if d%2==0:
            ret *= np.exp(-1j*theta.reshape(-1,1)/2)
        if diag_only:
            ret = ret.reshape(*theta.shape, d)
        else:
            tmp0 = np.zeros((ret.shape[0],d,d), dtype=ret.dtype)
            ind0 = np.arange(d)
            tmp0[:,ind0,ind0] = ret
            ret = tmp0.reshape(*shape, d, d)
    return ret


def rz(theta, d=2, diag_only=False):
    r'''
    $$ exp \{ -i\theta \sigma _z/2 \} $$
    '''
    if d==2:
        if is_torch(theta):
            ca = torch.cos(theta/2)
            isa = 1j*torch.sin(theta/2)
            if diag_only:
                tmp0 = [ca-isa,ca+isa]
                ret = torch.concat([x.view(-1,1) for x in tmp0], dim=-1).view(*theta.shape,2)
            else:
                zero = torch.zeros_like(ca)
                tmp0 = [ca-isa,zero,zero,ca+isa]
                ret = torch.concat([x.view(-1,1) for x in tmp0], dim=-1).view(*theta.shape,2,2)
        else:
            ca = np.cos(theta/2)
            isa = 1j*np.sin(theta/2)
            if diag_only:
                ret = np.stack([ca-isa,ca+isa], axis=-1).reshape(*ca.shape,2)
            else:
                zero = np.zeros_like(ca)
                ret = np.stack([ca-isa,zero,zero,ca+isa], axis=-1).reshape(*ca.shape,2,2)
    else:
        ret = _rz_qudit(theta, d, diag_only)
    return ret


def rzz(theta):
    r'''
    $$ exp \{ -i\theta \sigma _z \otimes \sigma_z /2 \} $$
    '''
    if is_torch(theta):
        ca = torch.cos(theta/2)
        isa = 1j*torch.sin(theta/2)
        zero = torch.zeros_like(ca)
        tmp0 = [ca-isa,zero,zero,zero, zero,ca+isa,zero,zero, zero,zero,ca+isa,zero, zero,zero,zero,ca-isa]
        ret = torch.concat([x.view(-1,1) for x in tmp0], dim=-1).view(*theta.shape,4,4)
    else:
        ca = np.cos(theta/2)
        isa = 1j*np.sin(theta/2)
        zero = np.zeros_like(ca)
        tmp0 = [ca-isa,zero,zero,zero, zero,ca+isa,zero,zero, zero,zero,ca+isa,zero, zero,zero,zero,ca-isa]
        ret = np.stack(tmp0, axis=-1).reshape(*ca.shape,4,4)
    return ret

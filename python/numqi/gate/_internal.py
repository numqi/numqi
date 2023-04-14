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


def rx(theta):
    r'''
    $$ exp \{ -i\theta \sigma _x/2 \} $$
    '''
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


def rz(theta):
    r'''
    $$ exp \{ -i\theta \sigma _z/2 \} $$
    '''
    if is_torch(theta):
        ca = torch.cos(theta/2)
        isa = 1j*torch.sin(theta/2)
        zero = torch.zeros_like(ca)
        tmp0 = [ca-isa,zero,zero,ca+isa]
        ret = torch.concat([x.view(-1,1) for x in tmp0], dim=-1).view(*theta.shape,2,2)
    else:
        ca = np.cos(theta/2)
        isa = 1j*np.sin(theta/2)
        zero = np.zeros_like(ca)
        ret = np.stack([ca-isa,zero,zero,ca+isa], axis=-1).reshape(*ca.shape,2,2)
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

import types
import collections
import numpy as np

from ..utils import is_torch, hf_tuple_of_int

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
        ret = torch.stack([tmp0,tmp1,tmp2,tmp3], axis=-1).reshape(*a.shape, 2, 2)
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


def u(theta, phi, lambda_):
    r'''
    $$ R_z( \phi ) R_y( \theta ) R_z( \lambda ) e^{0.5j (\phi+\lambda)} $$

    https://qiskit.org/documentation/stubs/qiskit.circuit.library.UGate.html?highlight=ugate#qiskit.circuit.library.UGate
    '''
    if is_torch(theta):
        ct = torch.cos(theta/2)
        st = torch.sin(theta/2)
        tmp0 = torch.exp(1j*lambda_)
        tmp1 = torch.exp(1j*phi)
        ret = torch.stack([ct,-st*tmp0,st*tmp1,ct*tmp0*tmp1], axis=-1).view(*ct.shape, 2, 2)
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
        if theta.dtype==torch.float32:
            theta = theta*torch.tensor(1, dtype=torch.complex64)
        elif theta.dtype==torch.float64:
            theta = theta*torch.tensor(1, dtype=torch.complex128)
        ca = torch.cos(theta/2)
        isa = 1j*torch.sin(theta/2)
        ret = torch.stack([ca,-isa,-isa,ca], axis=-1).view(*theta.shape,2,2)
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
            theta = theta*torch.tensor(1, dtype=torch.complex64)
        elif theta.dtype==torch.float64:
            theta = theta*torch.tensor(1, dtype=torch.complex128)
        ca = torch.cos(theta/2)
        sa = torch.sin(theta/2)
        ret = torch.stack([ca,-sa,sa,ca], dim=-1).view(*theta.shape,2,2)
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
        ret = torch.stack([ca-isa,zero,zero,ca+isa], dim=-1).view(*theta.shape,2,2)
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
        ret = torch.stack(tmp0, dim=-1).view(*theta.shape,4,4)
    else:
        ca = np.cos(theta/2)
        isa = 1j*np.sin(theta/2)
        zero = np.zeros_like(ca)
        tmp0 = [ca-isa,zero,zero,zero, zero,ca+isa,zero,zero, zero,zero,ca+isa,zero, zero,zero,zero,ca-isa]
        ret = np.stack(tmp0, axis=-1).reshape(*ca.shape,4,4)
    return ret


class Gate:
    def __init__(self, kind, array, requires_grad=False, name=None):
        assert kind in {'unitary', 'kraus', 'control'}
        self.kind = kind
        self.name = name
        self.array = array #numpy
        self.requires_grad = requires_grad

    def copy(self):
        ret = Gate(self.kind, self.array.copy(), requires_grad=self.requires_grad, name=self.name)
        return ret

    def __repr__(self):
        tmp0 = repr(self.array)
        ret = f'Gate({self.kind}, {self.name}, requires_grad={self.requires_grad}, {tmp0})'
        return ret
    __str__ = __repr__



class ParameterGate(Gate):
    def __init__(self, kind, hf0, args, name=None, requires_grad=True):
        args = [(np.asarray(x)) for x in args]
        array = hf0(*args)
        super().__init__(kind, array, requires_grad=requires_grad, name=name)
        self.args = args
        self.hf0 = hf0
        self.grad = np.zeros(array.shape, dtype=np.complex128) #WARNING do NOT do in-place operation

    def set_args(self, args, array=None):
        self.args = [(np.asarray(x)) for x in args]
        self.array = self.hf0(*self.args)
        if array is None:
            self.array = self.hf0(*args)
        else:
            self.array = np.asarray(array)

    def requires_grad_(self, tag=True):
        self.requires_grad = tag

    def zero_grad_(self):
        self.grad *= 0

    def copy(self):
        ret = ParameterGate(self.kind, self.hf0, self.args, name=self.name, requires_grad=self.requires_grad)
        return ret

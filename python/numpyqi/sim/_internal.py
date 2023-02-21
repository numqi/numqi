import numpy as np

from numpyqi.utils import hf_tuple_of_any

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
        args = hf_tuple_of_any(args, float)
        array = hf0(*args)
        super().__init__(kind, array, requires_grad=requires_grad, name=name)
        self.args = args
        self.hf0 = hf0
        self.grad = np.zeros(array.shape, dtype=np.complex128) #WARNING do NOT do in-place operation

    def set_args(self, args, array=None):
        self.args = hf_tuple_of_any(args, float)
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

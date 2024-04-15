import numpy as np

from numqi.utils import hf_tuple_of_any

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


class _ParameterHolder:
    def __init__(self, root:dict, key:(str|None)=None, index=None):
        self.root = root #circ._P
        self.key:(str|None) = key #None for root only
        self.index = index #None for root and leaf node

    def __getitem__(self, key):
        if isinstance(key, str):
            assert (self.key is None) and (self.index is None)
            ret = _ParameterHolder(self.root, key)
        else:
            assert self.index is None
            tmp0 = '' if self.key is None else self.key
            ret = _ParameterHolder(self.root, tmp0, index=key)
        return ret

    def resolve(self):
        if self.index is None:
            ret = self.root[self.key]
        else:
            ret = self.root[self.key][self.index]
        return ret

    def __str__(self):
        tmp0 = '' if self.key is None else self.key
        tmp1 = '' if self.index is None else self.index
        ret = f'_ParameterHolder({tmp0}, {tmp1})'
        return ret

    __repr__ = __str__

class ParameterGate(Gate):
    def __init__(self, kind, hf0, args, name=None, requires_grad=True):
        if isinstance(args, _ParameterHolder):
            array = None
        else:
            args = hf_tuple_of_any(args, float)
            array = hf0(*args)
        super().__init__(kind, array, requires_grad=requires_grad, name=name)
        self.args = args
        self.hf0 = hf0

    def set_args(self, args, array=None):
        self.args = hf_tuple_of_any(args, float)
        if array is None:
            self.array = self.hf0(*args)
        else:
            self.array = np.asarray(array)

    def requires_grad_(self, tag=True):
        self.requires_grad = tag

    def copy(self):
        ret = ParameterGate(self.kind, self.hf0, self.args, name=self.name, requires_grad=self.requires_grad)
        return ret

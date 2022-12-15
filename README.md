# numpyqi: a quantum information toolbox implemented in numpy

A Basic Quantum Information toolbox implemented in numpy

1. A numpy based quantum simulator, for quick development only, not performance-optimized
   * `.state`, `.dm`, `.gate`, `.utils` module
2. quantum information related function
   * `.random`, `.utils`, `.param`, `.gellmann`, `.channel`
3. Pytorch is an optional requirement, most functions should work for non `torch.Tensor` input.
   * pytorch and cvxpy  should be put in `pyqet` package

```python
import numpyqi
```

gradient back-propagation can be supported in this module

1. `numpyqi.circuit._torch_only`
2. gradient back-propagation for density matrix

QECC

1. link
   * [github/qecsim](https://github.com/qecsim/qecsim)
   * [github/qtcodes](https://github.com/yaleqc/qtcodes)
   * [github/pymatching](https://github.com/oscarhiggott/PyMatching)
   * [github/qsurface](https://github.com/watermarkhu/qsurface)
   * [encoding-circuits](https://markus-grassl.de/QECC/circuits/index.html)

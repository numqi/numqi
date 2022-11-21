# numpyqi: a quantum information toolbox implemented in numpy

A Basic Quantum Information toolbox implemented in numpy

1. A numpy based quantum simulator, for quick development only, not performance-optimized
   * `.state`, `.dm`, `.gate`, `.utils` module
2. quantum information related function
   * `.random`, `.utils`
3. basic: with minimum dependency, no pytorch, no cvxpy (those should be put in `pyqet`)

Pytorch is an optional requirement, all function should work for non torch tensor input.

```txt
import numpyqi
```

gradient back-propagation can be supported in this module

1. `numpyqi.circuit._torch_only`
2. gradient back-propagation for density matrix
3. VarQEC

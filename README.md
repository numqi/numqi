# numpyqi: a quantum information toolbox implemented in numpy

WARNING: no backward compatibility guarantee until version `1.0.0`

1. A numpy based quantum simulator, for quick development only, not performance-optimized
   * `.state`, `.dm`, `.gate` module
   * gradient back-propagation is supported
   * TODO gradient back-propagation for density matrix
2. parameterization of various stuff
   * `.param`: special unitary matrix, special orthogonal matrix, density matrix, quantum channel, etc.
   * `.random`: randomly generate these kind of things
3. special states, matrices
   * `.gellmann`: convert matrix into gellmann basis and reversally
   * `.dicke`: convert bosonic symmetrically state into Dicke basis and reversally
4. quantum operation
   * `.gate`: various unitary matrix, like Pauli gate, Clifford gate, etc.
   * `.channel`: different representation of quantum channel, like Choi, Kraus, Super-op, partial trace of unitary
5. Pytorch is an optional requirement, most functions should work for non `torch.Tensor` input.
   * pytorch and cvxpy should be put in `pyqet` package
6. optional dependency
   * `pytorch`: without it, gradient related functions will not work
   * `matplotlib`: with it, `MaxEntModel`

design philosophy

1. open and available
   * open source
   * most users can install `numpyqi`, no matter windows/linux/macOS, no matter CPU/GPU
   * minimum dependency: `scipy,matplotlib,tqdm,sympy,opt_einsum,torch-wrapper`
   * Pytorch is an optional requirement, most functions should work for non `torch.Tensor` input
2. the fundamental functions should be here. Quantum applications should go to `pyqet` package
   * fundamental: (personally hope) more packages can be developed beyond `pyqet`

the relation betwen `numpyqi` and `pyqet`

1. just like `numpy` and `scipy`
   * all `numpyqi` function will port into `pyqet`
2. researchers who cares more about quantum application, like how to detect entanglement, VarQEC, MaxEnt, PureB etc.
   * use `pyqet`, no need to warry about `numpyqi`
3. developers who are ambitious beyond quantum applications
   * please try to use `numpyqi`
   * based on the utilities provided by `numpyqi`, you can develop more powerful tools, like `cupysim` is in this direction

```python
import numpyqi
numpyqi.param
numpyqi.random
numpyqi.gellmann
numpyqi.dicke
numpyqi.gate
numpyqi.channel
numpyqi.sim #.circuit .state .dm
numpyqi.utils
numpyqi.matrix_subspace #not finished
```

## documentation

package requried for building the documentation

```bash
pip install mkdocs mkdocs-material pymdown-extensions
```

Build and Serve the documentation locally, then brower the website `127.0.0.1:23333`

```bash
mkdocs serve --dev-addr=127.0.0.1:23333
# --dev-addr=0.0.0.0:23333
```

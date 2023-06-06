# numqi documentation

**WARNING**: the package is still under development, the documentation page is for preview only.

**WARNING**: the documentation is generated with the help of GitHub Copilot, so it may contain some errors.

`numqi` (pronouce: num(-py)-q-i) is designed as the "basic" quantum computing toolbox. With minimum dependency required so that `numqi` can be installed on most of platform. `numqi` provides these modules

1. quantum circuit simulator, variational algorithm with gradient back-propagation
     * `numqi.sim.Circuit`: construct a circuit
     * `numqi.gate`: various gate, like Puali-XYZ
     * `numqi.sim.state`: function for pure state simulation, already wrapped in `numqi.sim.circuit` module
     * `numqi.sim.dm`: function for density matrix simulation, already wrapped in `numqi.sim.circuit` module
2. `numqi.random`: generate various random "stuff", like random pure state, untiary gate, quantum channel, density matrix, seprable density matrix, etc.
3. `numqi.param`: parameterize various "stuff", like parameterized unitary matrices, hermitian matrices, quantum channel etc.
4. `numqi.gellman`

## installation

(TODO, when the repo `numqi` is public available) The following command should be okay for `win/mac/linux`. If you might use `torch` to do gradient backpropagation tasks, then you need add the optional arguments `[torch]`. Without `torch`, most of functions should still be runable.

```bash
pip install numqi
# pip install numqi[torch]
```

Since `numqi` is still not public available right now, please download the source code and install it manually.

```bash
git clone git@github.com:husisy/numqi.git
cd numqi
pip install .
# pip install ".[torch]"
```

test whether succuessfully installed (run it in `python/ipython` REPL)

```Python
import numqi
```

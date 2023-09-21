# numqi

**WARNING**: the package is still under development, the documentation page is for preview only.

**WARNING**: the documentation is generated with the help of GitHub Copilot, so it may contain errors.

**WARNING**: no backward compatiblity until version `1.0.0` (TODO roadmap). Right now, feel free to ask questions on GitHub issue page (TODO to be public).

`numqi` (pronouce: num(-py)-q-i) a numpy-based quantum information toolbox. Currently, `numqi` provides these modules

1. specific quantum information tasks
    * `numqi.sim`: a numpy based quantum circuit simulator, variational algorithm with gradient back-propagation supported [wiki/quantum-simulator](https://en.wikipedia.org/wiki/Quantum_simulator)
        * `numqi.sim.Circuit`: construct a circuit
        * `numqi.gate`: various gate, like Puali-XYZ
        * `numqi.sim.state`: function for pure state simulation, wrapped in `numqi.sim.circuit` module
        * `numqi.sim.dm`: function for density matrix simulation, wrapped in `numqi.sim.circuit` module
    * `numqi.gate`: all kinds of quantum gate [wiki/quantum-logic-gate](https://en.wikipedia.org/wiki/Quantum_logic_gate)
    * `numqi.channel`: utilities related to quantum channel, like conversion between different representations of a quantum channel (Kraus operator, super-operator, Choi state, etc.) [wiki/quantum-channel](https://en.wikipedia.org/wiki/Quantum_channel)
    * `numqi.entangle`: detect entanglement, including convex hull approximation (CHA), symmetric extension, pure Bosonic extension, etc. [wiki/quantum-entanglement](https://en.wikipedia.org/wiki/Quantum_entanglement)
    * `numqi.qec`: finding quantum error correcting code, including VarQECC model [wiki](https://en.wikipedia.org/wiki/Quantum_error_correction)
    * `numqi.maximum_entropy`: the relation between Hamiltonian, groud state, and marginal problem
    * `numqi.unique_determine`: UDA/UDP measurement scheme. Given part of measurement results, determine what state it should be with some prior knowledge (e.g. pure state assumption)
    * `numqi.optimal_control`: optimal control for quantum system, e.g. finding the optimal control pulse to implement a quantum gate
2. fundamental tools
   * `numqi.random`: generate various random "stuff", like random pure state, untiary gate, quantum channel, density matrix, seprable density matrix, etc.
   * `numqi.param`: parameterize various "stuff", like parameterized unitary matrices, hermitian matrices, quantum channel etc.
   * `numqi.gellmann`: Gell-Mann matrix. E.g., converting density matrix into Gell-Mann matrix representation [wiki/gellmann](https://en.wikipedia.org/wiki/Gell-Mann_matrices) [wiki/generalized-gellmann](https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices)
   * `numqi.dicke`: Dicke state (Bosonic pure symmetric state)
   * `numqi.group`: utilities related to group [wiki/group](https://en.wikipedia.org/wiki/Group_(mathematics)) (like symmetric group, cyclic group)
   * `numqi.optimize`: optimization related tools, like gradient descent
   * `numqi.matrix_space`: the linear space of matrices, and determine its entanglement property

Generally, these specific quantum information tasks are implemented based on these fundamental tools. For example, the `numqi.entangle` module use quite a lots of the module `numqi.group` and `numqi.optimize`. For those who are more interested in quantum information problems, you may directly dive into these specific modules. For those who are more interested in the underlying algorithms or math concepts, you may start from the fundamental modules.

*PS*: Stay relaxing if none of these terminologies make sense, I will (try to) explain these words in the following pages.

## installation

(TODO, when the repo `numqi` is public available) The following command should be okay for `win/mac/linux`.

```bash
pip install numqi
# TODO upload to pypi.org
```

Since `numqi` is still not public available right now, please download the source code and install it manually.

```bash
git clone git@github.com:husisy/numqi.git
cd numqi
pip install .
```

test whether succuessfully installed (run it in `python/ipython` REPL)

```Python
import numqi
```

A simple example to detect whether Bell state [wiki](https://en.wikipedia.org/wiki/Bell_state) is entangle or not using positive partial transpose (PPT) criteria.

```Python
import numqi
bell_state = numqi.entangle.get_werner_state(d=2, alpha=1)
print(bell_state) #a numpy array
# [[ 0.   0.   0.   0. ]
#  [ 0.   0.5 -0.5  0. ]
#  [ 0.  -0.5  0.5  0. ]
#  [ 0.   0.   0.   0. ]]
print(numqi.entangle.is_ppt(bell_state)) #True if seperable state, False is entangle state (small probability also return True)
# False
```

`numqi` also include a `numpy` based quantum simulator. Let's try a "non-trival" quantum circuit (we will re-visit this circuit in quantum error correction section)

```Python
import numpy as np
import numqi
circ = numqi.sim.Circuit()
for x in range(4):
    circ.H(x)
circ.cz(3, 4)
circ.cy(2, 3)
circ.cz(2, 4)
circ.cx(1, 2)
circ.cz(1, 3)
circ.cx(1, 4)
circ.cy(0, 2)
circ.cx(0, 3)
circ.cx(0, 4)

# numqi store quantum state using numpy array
initial_state = np.zeros(2**5, dtype=np.complex128)
initial_state[0] = 1
final_state = circ.apply_state(initial_state)
```

## similar packages

1. QETLAB [documentation](https://qetlab.com/) [github](https://github.com/nathanieljohnston/QETLAB)
2. google/quantumlib [google/quantumAI](https://quantumai.google/software) [github](https://github.com/quantumlib) [openfermion](https://github.com/quantumlib/OpenFermion) [stim](https://github.com/quantumlib/Stim)
3. pennylane [documentation](https://docs.pennylane.ai/en/stable/)
4. [github/qustop](https://github.com/vprusso/qustop) [github/toqito](https://github.com/vprusso/toqito)
5. [github/qutip](https://github.com/qutip)

> `qetlab`

`qetlab` is a matlab toolbox designed for quantum entanglement detection and do a greak work in this field. Honestly, `numqi` is mainly inspired by `qetlab` and the name segment `py-q-e-t` is taken from `py-thon` and `qet-lab`. In my personal view, the python package`numpy` and various python deep learning packages `pytorch/tensorflow/...` are much more powerful than the matlab builtin array manipulations. Then, why not combining them together.

## publications

This package is to support following papers

1. detecting entanglement by pure bosonic extension [arxiv-link](https://arxiv.org/abs/2209.10934) `numqi.entangle`
2. Quantum variational learning for quantum error-correcting codes [arxiv-link](https://arxiv.org/abs/2204.03560) ``
3. Tapping into Permutation Symmetry for Improved Detection of k-Symmetric Extensions [arxiv-link](https://arxiv.org/abs/2309.04144) `numqi.entangle` `numqi.group.symext`
4. Variational learning algorithms for quantum query complexity [arxiv-link](https://arxiv.org/abs/2205.07449) `numqi.query`
5. A Variational Approach to Unique Determinedness in Pure-state Tomography [arxiv-link](https://arxiv.org/abs/2305.10811) `numqi.unique_determine`
6. Maximum entropy methods for quantum state compatibility problems [arxiv-link](https://arxiv.org/abs/2207.11645) `numqi.maximum_entropy`
7. Simulating Noisy Quantum Circuits with Matrix Product Density Operators [arxiv-link](https://arxiv.org/abs/2004.02388) `numqi.sim`

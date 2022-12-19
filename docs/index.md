# numpyqi documentation

`numpyqi` (pronouce: num-py-q-i) is designed as the "basic" quantum computing toolbox. With minimum dependency required so that `numpyqi` can be installed on most of platform. `numpyqi` provides these modules

1. quantum circuit simulator, variational algorithm with gradient back-propagation
   * `numpyqi.circuit`: construct a circuit
   * `numpyqi.gate`: various gate, like Puali-XYZ
   * `numpyqi.state`: function for pure state simulation, already wrapped in `numpyqi.circuit` module
   * `numpyqi.dm`: function for density matrix simulation, already wrapped in `numpyqi.circuit` module
2. `numpyqi.random`: generate various random "stuff", like random pure state, untiary gate, quantum channel, density matrix, seprable density matrix, etc.
3. `numpyqi.param`: parameterize various "stuff", like parameterized unitary matrices, hermitian matrices, quantum channel etc.
4. `numpyqi.gellman`
5. `numpyqi.qec`

## installation

the following command should be okay for `win/mac/linux`

`pip install numpyqi`

`pytorch` is an optional dependency. Without `pytorch`, you should still be able to use most of functions.

test whether succuessfully installed (run it in `python/ipython` REPL)

`import numpyqi`

## guide for developer

TODO

1. [ ] VarQEC

# gate

## Pauli

`numpyqi.gate.X`

`numpyqi.gate.Y`

`numpyqi.gate.Z`

`numpyqi.gate.I`

`numpyqi.gate.pauli.s0`: `.sx` `.sy` `.sz` `.s0` `.s0s0` `.s0sx` ...

## Clifford gate

`numpyqi.gate.H`

`numpyqi.gate.S`

`numpyqi.gate.CNOT`

`numpyqi.gate.CZ`

`numpyqi.gate.Swap`

## non-Clifford gate

`numpyqi.gate.T`

## parameterized gate

`numpyqi.gate.pauli_exponential(a, theta, phi)`

$$
e^{ia\hat{n}\cdot \vec{\sigma}}
$$

`numpyqi.gate.rx()`

`numpyqi.gate.ry()`

`numpyqi.gate.rz()`

`numpyqi.gate.u3(theta, phi, lambda)`

`numpyqi.gate.rzz()`

## misc

`numpyqi.gate.Gate`

`numpyqi.gate.ParameterGate`

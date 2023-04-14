# gate

## Pauli

`numqi.gate.X`

`numqi.gate.Y`

`numqi.gate.Z`

`numqi.gate.I`

`numqi.gate.pauli.s0`: `.sx` `.sy` `.sz` `.s0` `.s0s0` `.s0sx` ...

## Clifford gate

`numqi.gate.H`

`numqi.gate.S`

`numqi.gate.CNOT`

`numqi.gate.CZ`

`numqi.gate.Swap`

## non-Clifford gate

`numqi.gate.T`

## parameterized gate

`numqi.gate.pauli_exponential(a, theta, phi)`

$$
e^{ia\hat{n}\cdot \vec{\sigma}}
$$

`numqi.gate.rx()`

`numqi.gate.ry()`

`numqi.gate.rz()`

`numqi.gate.u3(theta, phi, lambda)`

`numqi.gate.rzz()`

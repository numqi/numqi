# Entanglement basic

This is a place to put basic knowledge, like what is quantum pure state.

You may skip this part continue with the usage part. When you encounter unknown notation later, come back and go through that section.

We assume some basic knowledge about how to use `numpy` [numpy-documentation](https://numpy.org/doc/)

*TO developer*: please use the knowledge as basic as possible, if necessary, you can also put a coursera/youtube link.

**WARNING**: this page is **NOT** going to be formal, rigorous

For conciseness, we put necessary package import here.

```Python
import numpy as np
import numqi
```

## Quantum state

Quantum state can be divided into

1. pure state: one dimensional array (informal)
2. density matrix: two dimensional matrix (informal)

## Pure state

**definition**: a complex vector (array) with length 1, usually denoted using Dirac (bra-ket) notation [wiki/bra-ket-notation](https://en.wikipedia.org/wiki/Bra%E2%80%93ket_notation) $|\phi\rangle,|\psi\rangle$. If the vector has $d$ components, we call it qudit with $d$. Specially, when $d=2$, we call it qubit. let's see some examples.

qubit: be care the state $|0\rangle,|1\rangle$ are used conventional in this field and they are also called basis (computational basis)

$$ |0\rangle=[1,0]$$

$$ |1\rangle=[0,1] $$

$$ |\psi\rangle=[0.6, 0.48+0.64j] $$

qutrit (qudit with $d=3$):

$$ |\psi\rangle = [0.6,0.48,0.64] $$

2 qubits (or qudit with $d=4$)

$$ |\psi\rangle = [0.36,0.48j,0.48,0.64] $$

more generally for any qudit of size $d$

$$ |\psi\rangle=\sum_{x=0}^{d-1}\psi_x|x\rangle $$

The **length** (norm) is measured in complex 2-norm and its norm should always be one (unit)

$$\lVert |\psi \rVert_2=\sum_{x=0}^{d-1}\psi_x\psi_x^*=1$$

We can create pure states using `numpy` or `numqi` (they are simply `numpy` array)

```Python
# qubit
psi0 = np.array([1,0])
psi1 = np.array([0,1])
psi2 = np.array([0.6,0.8j])

# qudit (d=4)
psi0 = np.array([0.36, 0.48j, 0.48, 0.64])
psi1 = np.array([1,2,3,4])/np.sqrt(30)
psi2 = np.array([0,1,-1,0])/np.sqrt(2) #Bell state (pure state)

# qudit (d=23): random qudit state
psi0 = numqi.random.rand_state(23)
```

and evaluate its norm which should always be almost `1`

```Python
psi0 = numqi.random.rand_state(233)
print(np.linalg.norm(psi0)) #1.0
```

Inner product between two pure states *TODO*

Some special pure states

1. GHZ state
2. Bell state

## Density matrix

**definition**: a Hermitian, semi-definite matrix with trace 1, usually denoted with symbol $\rho$. If this matrix has $d$ column and $d$ row (columns must be equal to rows), we call it density matrix of qudit with $d$ dimension and write it as

$$\rho=\sum_{i,j=0}^{d-1} \rho_{ij}|i\rangle\langle j|$$

where $\rho_{ij}$ is the numerical value at i-th row j-th column, $|i\rangle$ is the i-th basis (ket, pure state with only $i$-th component 1), $\langle j|$ is the j-th basis (a row vector, complex transpose of the corresponding $|j\rangle$). Let's explain each word in the definition

1. complex matrix: $\rho\in\mathbb{C}^{d\times d}$
2. Hermitian: $\rho_{ij}=\rho_{ji}^*$
3. semi-definite: for any nonzero complex vector $v$, $\sum_{ij}v^*_i\rho_{ij}v_j>0$
    * equivalent: the minimum eigenvalues of $\rho$ is non-negative
    * usually denoted as $\rho\succeq 0$, the symbol $0$ denotes zero matrix of size $d\times d$
4. trace 1: $\sum_i\rho_{ii}=1$
    * equivalent: summation of all eigenvalues gives one, $\sum_i \lambda_i(\rho)=1$

We can generate random density matrix in `numqi`

```Python
# d=4
rho0 = np.array([[0.1,0,0,0], [0,0.2,0,0], [0,0,0.3,0], [0,0,0,0.4]])
rho1 = numqi.entangle.get_werner_state(d=2, alpha=1) #Bell state (density matrix)
rho2 = numqi.random.rand_density_matrix(4)
```

and evaluate its trace which should almost 1, and its eigenvalue which should be all non-negative

```Python
rho = numqi.random.rand_density_matrix(4)
print(rho)
# [[ 0.311+3.986e-19j  0.072+7.568e-03j  0.081+6.956e-03j -0.005+6.121e-02j]
#  [ 0.072-7.568e-03j  0.095+3.608e-19j  0.122+3.662e-02j  0.074-5.899e-02j]
#  [ 0.081-6.956e-03j  0.122-3.662e-02j  0.274-4.513e-19j -0.063-1.278e-01j]
#  [-0.005-6.121e-02j  0.074+5.899e-02j -0.063+1.278e-01j  0.32 -3.081e-19j]]
print(f'{np.trace(rho)=}')
# np.trace(rho)=(0.9999999999999999+4.81482486096809e-34j)
print(f'{np.linalg.eigvalsh(rho)=}')
# np.linalg.eigvalsh(rho)=array([0.001, 0.123, 0.381, 0.495])
print(f'{np.linalg.eigvalsh(rho).sum()=}')
# np.linalg.eigvalsh(rho).sum()=1.0
```

Connection between pure state and density matrix: when the density matrix $\rho$ has only one non-zero eigenvalue (must be 1), let's denote the associated eigenvector $|\psi\rangle$, then we say the density matrix $\rho$ and the pure state $|\psi\rangle$ are equivalent and have the following equation

$$\rho=|\psi\rangle\langle\psi|$$

The matrix with only one nonzero eigenvalue is said of rank one.

Maximum mixed state $\rho_0=I_d/d$.

Given a density matrix $\rho$, we can convert it into Gell-Mann basis. Its vector form $\vec{\rho}$, unit vector $\hat{\rho}$ and its vector norm $\lVert \vec{\rho}\rVert_2$.

$$ \rho=\rho_{0}+\sum_{i=0}^{d^{2}-2}\vec{\rho}_{i}M_{i}=\rho_{0}+\vec{\rho}\cdot\vec{M}=\rho_{0}+\left\Vert \vec{\rho}\right\Vert _{2}\hat{\rho}\cdot\vec{M} $$

Two interpolation schemes might be useful

$$ \rho\left(\alpha\right)=\left(1-\alpha\right)\rho_{0}+\alpha\rho,\rho\left(\alpha=1\right)=\rho $$

$$ \rho\left(\beta,\hat{\rho}\right)=\rho_{0}+\beta\hat{\rho}\cdot\vec{M},\rho\left(\beta=\left\Vert \vec{\rho}\right\Vert _{2},\hat{\rho}\right)=\rho $$

Their connection is $\beta=\alpha \lVert \vec{\rho}\rVert_2$. In `numqi`, the second form is used in most cases to avoid confusion but you can always convert between them.

### Werner state

1. reference
   * [wiki/werner-state](https://en.wikipedia.org/wiki/Werner_state)
   * [quantiki/werner-state](https://www.quantiki.org/wiki/werner-state)
   * [PRA88.032323](http://dx.doi.org/10.1103/PhysRevA.88.032323) Compatible quantum correlations: Extension problems for Werner and isotropic states

$$ \rho_{W,d}\left(a\right)=\frac{d-a}{d\left(d^{2}-1\right)}I+\frac{ad-1}{d\left(d^{2}-1\right)}\sum_{ij}\left|ij\right\rangle \left\langle ji\right|,\quad a\in\left[-1,1\right] $$

$$ \rho_{W,d}^{\prime}\left(\alpha\right)=\frac{1}{d^{2}-d\alpha}I-\frac{\alpha}{d^{2}-d\alpha}\sum_{ij}\left|ij\right\rangle \left\langle ji\right|,\quad\alpha\in\left[-1,1\right] $$

$$ \rho_{W,d}\left(a\right)=\rho_{W,d}^{\prime}\left(\frac{1-ad}{d-a}\right),\quad\rho_{W,d}\left(\frac{1-\alpha d}{d-\alpha}\right)=\rho_{W,d}^{\prime}\left(\alpha\right) $$

| $a$ | $\alpha$ | state |
| :-: | :-: | :-: |
| 1 | -1 | $xI+y\sum_{ij}\lvert ij\rangle\langle ji\rvert$ |
| $1/d$ | 0 | $I/d^2$ |
| 0 | $1/d$ | xx |
| -1 | 1 | xx |

SEP boundary

$$ a\in\left[0,1\right],\alpha\in\left[-1,\frac{1}{d}\right] $$

(1,k) extension boundary

$$ \left(1,k\right)\;\mathrm{ext}:\quad a\in\left[\frac{1-d}{k},1\right],\alpha\in\left[-1,\frac{k+d^{2}-d}{kd+d-1}\right] $$

### Isotropic state

1. reference
   * [quantiki/isotropic-state](https://www.quantiki.org/wiki/isotropic-state)
   * [PRA88.032323](http://dx.doi.org/10.1103/PhysRevA.88.032323) Compatible quantum correlations: Extension problems for Werner and isotropic states

$$ \rho_{I,d}\left(a\right)=\frac{d-a}{d\left(d^{2}-1\right)}I+\frac{ad-1}{d\left(d^{2}-1\right)}\sum_{i}\left|ii\right\rangle \left\langle ii\right|,\quad a\in\left[0,d\right] $$

$$ \rho_{I,d}^{\prime}\left(\alpha\right)=\frac{1-\alpha}{d^{2}}I+\frac{\alpha}{d}\sum_{i}\left|ii\right\rangle \left\langle ii\right|,\quad\alpha\in\left[-\frac{1}{d^{2}-1},1\right] $$

$$ \rho_{I,d}\left(\frac{1+\alpha d^{2}-\alpha}{d}\right)=\rho_{I,d}^{\prime}\left(\alpha\right),\quad\rho_{I,d}\left(a\right)=\rho_{I,d}^{\prime}\left(\frac{ad-1}{d^{2}-1}\right) $$

| $a$ | $\alpha$ | state |
| :-: | :-: | :-: |
| 0 | $-\frac{1}{d^2-1}$ | xx |
| $1/d$ | 0 | $I/d$ |
| $d$ | 1 | $\sum_{i}\lvert ii\rangle\langle ii\rvert$ |

SEP boundary

$$ a\in\left[0,1\right],\alpha\in\left[-\frac{1}{d^{2}-1},\frac{1}{d+1}\right] $$

(1,k) extension boundary

$$ a\in\left[0,1+\frac{d-1}{k}\right],\alpha\in\left[-\frac{1}{d^{2}-1},\frac{kd+d^{2}-d-k}{k\left(d^{2}-1\right)}\right] $$

### maximum entangled state

TODO

## unitary matrix

TODO

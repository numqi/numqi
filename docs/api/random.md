# `numqi.random`

Import necessary modules

```Python
import numpy as np
import numqi
```

This module use random seed widely, most likely you will get different results. Most functions accept `seed` as an argument, you can set it to a fixed value to get reproducible results.

## quantum state

::: numqi.random.rand_haar_state
    options:
      heading_level: 3

TODO explain Haar measure

```Python
>>> numqi.random.rand_haar_state(3)
array([-0.36622244-0.61061589j,  0.18964285+0.08874409j,
       -0.18312026-0.64471421j])
>>> numqi.random.rand_haar_state(3, seed=233)
array([0.4757064 +0.46866932j, 0.43667786-0.12461279j,
       0.48188252+0.34003799j])
```

::: numqi.random.rand_bipartite_state
    options:
      heading_level: 3

```Python
>>> numqi.random.rand_bipartite_state(2, 3)
array([ 0.24727854-0.1190505j ,  0.48835483-0.06843491j,
        0.50264365-0.01684556j, -0.3551116 -0.41708912j,
       -0.12957505-0.20932634j,  0.21153852+0.15214723j])
```

## density matrix

::: numqi.random.rand_density_matrix
    options:
      heading_level: 3

::: numqi.random.rand_separable_dm
    options:
      heading_level: 3

## hermitian matrix

`numqi.random.rand_hermite_matrix(d)`

$$
\lbrace x\in \mathbb{C} ^{d\times d}\,\,: x=x^{\dagger} \rbrace
$$

## unitary matrix

`numqi.random.rand_haar_unitary(d)`, `numqi.random.rand_unitary_matrix(d)`

$$
\lbrace x\in \mathbb{C} ^{d\times d}\,\,: xx^{\dagger}=I_d \rbrace
$$

## quantum channel

`numqi.random.rand_kraus_op(k, d0, d1)`

$$
\lbrace K\in \mathbb{C} ^{k\times d_o\times d_i}\,\,: \sum_s{K_{s}^{\dagger}K_s}=I_{d_i} \rbrace
$$

`numqi.random.rand_choi_op(d0, d1)`

$$
\lbrace C\in \mathbb{C} ^{d_id_o\times d_id_o}\,\,:C\succeq 0,\mathrm{Tr}_{d_o}\left[ C \right] =I_{d_i} \rbrace
$$

## misc

`numqi.random.get_numpy_rng`

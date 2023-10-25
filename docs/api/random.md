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

::: numqi.random.rand_bipartite_state
    options:
      heading_level: 3

## density matrix

::: numqi.random.rand_density_matrix
    options:
      heading_level: 3

::: numqi.random.rand_separable_dm
    options:
      heading_level: 3

## hermitian matrix

::: numqi.random.rand_hermitian_matrix
    options:
      heading_level: 3

## unitary matrix

::: numqi.random.rand_haar_unitary
    options:
      heading_level: 3

::: numqi.random.rand_unitary_matrix
    options:
      heading_level: 3

## quantum channel

::: numqi.random.rand_kraus_op
    options:
      heading_level: 3

::: numqi.random.rand_choi_op
    options:
      heading_level: 3

## miscellaneous

::: numqi.random.get_random_rng
    options:
      heading_level: 3

::: numqi.random.get_numpy_rng
    options:
      heading_level: 3

# `numpyqi.random`

## quantum state

`numpyqi.random.rand_state(d)`, `numpyqi.random.rand_haar_state(d)`

$$
\left\{ |\psi \rangle \in \mathbb{C} ^d\,\,: \left\| |\psi \rangle \right\| _2=1 \right\}
$$

TODO explain Haar measure

`numpyqi.random.rand_bipartitle_state(d0, d1)`

$$
\left\{ |\psi \rangle \in \mathbb{C} ^{d_1d_2}\,\,: \left\| |\psi \rangle \right\| _2=1 \right\}
$$

## density matrix

`numpyqi.random.rand_density_matrix`

$$
\left\{ \rho \in \mathbb{C} ^{d\times d}\,\,: \rho \succeq 0,\mathrm{Tr}\left[ \rho \right] =1 \right\}
$$

`numpyqi.random.rand_separable_dm`

$$
\left\{ \rho \in \mathbb{C} ^{d_1d_2\times d_1d_2}\,\,: \rho =\sum_k{p_i\rho _{i}^{\left( A \right)}\otimes \rho _{i}^{\left( B \right)}} \right\}
$$

## hermitian matrix

`numpyqi.random.rand_hermite_matrix(d)`

$$
\left\{ x\in \mathbb{C} ^{d\times d}\,\,: x=x^{\dagger} \right\}
$$

## unitary matrix

`numpyqi.random.rand_haar_unitary(d)`, `numpyqi.random.rand_unitary_matrix(d)`

$$
\left\{ x\in \mathbb{C} ^{d\times d}\,\,: xx^{\dagger}=I_d \right\}
$$

## quantum channel

`numpyqi.random.rand_kraus_op(k, d0, d1)`

$$
\left\{ K\in \mathbb{C} ^{k\times d_o\times d_i}\,\,: \sum_s{K_{s}^{\dagger}K_s}=I_{d_i} \right\}
$$

`numpyqi.random.rand_choi_op(d0, d1)`

$$
\left\{ C\in \mathbb{C} ^{d_id_o\times d_id_o}\,\,:C\succeq 0,\mathrm{Tr}_{d_o}\left[ C \right] =I_{d_i} \right\}
$$

## misc

`numpyqi.random.get_numpy_rng`

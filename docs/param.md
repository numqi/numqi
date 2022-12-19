# Parameterization

## unitary matrix

`numpyqi.param.real_matrix_to_unitary(x)`

$$
x\in \mathbb{R} ^{d\times d}
$$

$$
y=x_l+x_{l}^{T}+i\left( x_u-x_{u}^{T} \right)
$$

$$
z=\mathrm{e}^{iy}
$$

where $x_l$ denotes the lower triangle matrix with diagonal elements, and $x_u$ denotes the upper triagnle matrix without diagonal elements

## orthogonal matrix

`numpyqi.param.real_matrix_to_orthogonal(x)`

$$
x\in \mathbb{R} ^{d\times d}
$$

$$
z=\mathrm{e}^{\left( x_u-x_{u}^{T} \right)}
$$

## Positive Semi-definite matrix

`numpyqi.param.real_matrix_to_PSD(x)`

$$
x\in \mathbb{R} ^{d\times d}
$$

$$
y=x_l+x_{l}^{T}+i\left( x_u-x_{u}^{T} \right)
$$

$$
z=\mathrm{e}^y
$$

## quantum channel

`numpyqi.param.real_matrix_to_choi_op`

TODO

`numpyqi.param.real_to_kraus_op`

TODO

`numpyqi.param.PSD_to_choi_op`

TODO

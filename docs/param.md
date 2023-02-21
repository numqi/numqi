# Parameterization

## special unitary matrix

`numpyqi.param.real_matrix_to_special_unitary(x)`

$$
x\in \mathbb{R} ^{d\times d}
$$

$$
y=x_l+x_{l}^{T}+i\left( x_u-x_{u}^{T} \right) - \frac{2\mathrm{tr}(x)}{d}I
$$

$$
z=\mathrm{e}^{iy}
$$

where $x_l$ denotes the lower triangle matrix with diagonal elements, and $x_u$ denotes the upper triagnle matrix without diagonal elements. The traceless matrix $y$ make sure that $det(z)=1$

Usually, the unitary matrix is unnecessary. But still, one can still parameterize a general unitary matrix as $e^{ir}e^{iy}$ where $r$ is a real number and $e^{iy}$ is a special unitary matrix as above

## special orthogonal matrix

`numpyqi.param.real_matrix_to_special_unitary(x, tag_real=True)`

$$
x\in \mathbb{R} ^{d\times d}
$$

$$
z=\mathrm{e}^{\left( x_u-x_{u}^{T} \right)}
$$

## Positive Semi-definite matrix

`numpyqi.param.real_matrix_to_trace1_PSD(x)`

$$
x\in \mathbb{R} ^{d\times d}
$$

$$
y=x_l+x_{l}^{T}+i\left( x_u-x_{u}^{T} \right)
$$

$$
z=\mathrm{e}^y
$$

for numerical stability, this function only produce trace one PSD matrices. To generate matrix with different trace, you may multiply the trace-1 matrix with another trace parameter.

## quantum channel

`numpyqi.param.real_matrix_to_choi_op`

TODO

`numpyqi.param.real_to_kraus_op`

TODO

`numpyqi.param.PSD_to_choi_op`

TODO

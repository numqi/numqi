# Gell-Mann matrix

[wiki/generalizations-of-pauli-matrices](https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices)

In traceless matrix space $\lbrace x\in\mathbb{C}^{d\times d}:\mathrm{Tr}[x]=0\rbrace$, the Gell-Mann matrices make a complete and orthogonal basis. When $d=2$, Gell-Mann matrices are simply three Pauli matrices. For general dimension $d$, the number of Gell-Mann matrices is $d^2-1$, $M_0,M_1,M_2,\cdots,M_{d^2-2}$. For $d=3$, the specifc form of Gell-Mann matrices can be found in [wiki/Gell-Mann-matrices](https://en.wikipedia.org/wiki/Gell-Mann_matrices).

`numqi` add the identity matrix as an extra Gell-Mann matrix $M_{d^2-1}=\sqrt{2/d}I$ to make a complete basis in matrix space $\mathbb{C}^{d\times d}$. The factor is choosen such that

$$\mathrm{Tr}[M_iM_j]=2\delta_{ij}$$

**Its connection with Frobenius norm**: the Frobenius norm of a matrix is defined as

$$x\in\mathbb{C}^{m\times n},\left\Vert x\right\Vert _{\mathrm{fro}}^{2}=\sum_{ij}\left|x_{ij}\right|^{2}$$

For any square matrix $x\in\mathbb{C}^{d\times d}$, it can be expanded in the Gell-Mann basis

$$ x=\sum_{i=0}^{d^{2}-1}\vec{x}_{i}M_{i} $$

where $\vec{x}$ is a vector with $d^2$ components. The Frobenius norm of $x$ is related to the norm of $\vec{x}$ as

$$ 2\left\Vert \vec{x}\right\Vert _{2}^{2}=\left\Vert x\right\Vert _{\mathrm{fro}}^{2} $$

**Its connection with the density matrix**: since density matrix $\rho\in\mathbb{C}^{d\times d}$ is trace-one Hermitian matrix, it can be expanded in the Gell-Mann basis

$$ \rho=\frac{1}{d}I+\sum_{i=0}^{d^2-2}\vec{\rho}_iM_i $$

where $\vec{\rho}$ is a real vector with $d^2-1$ components.

**Its connection with purity**: the purity $\gamma_{\rho}$ of a density matrix $\rho$ is

$$\gamma_{\rho}=\mathrm{Tr}\left[ \rho ^2 \right] =\frac{1}{d}+2\lVert\vec{\rho}\rVert^{2}$$

It's easy to see that the purity is always larger than $1/d$ and the maximum mixed state $\rho=I/d$ gives the minimum purity $1/d$.

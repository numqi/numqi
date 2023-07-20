# Pauli group and Clifford group

This note explain how Clifford circuit can be simulated efficiently on classical computers.

1. link
    * [wiki/symplectic-group](https://en.wikipedia.org/wiki/Symplectic_group)
    * How to efficiently select an arbitrary clifford group element [doi-link](https://doi.org/10.1063%2F1.4903507)
    * 2004 Improved simulation of stabilizer circuits [doi-link](https://doi.org/10.1103/PhysRevA.70.052328)
    * Stabilizer Codes and Quantum Error Correction [arxiv-link](https://arxiv.org/abs/quant-ph/9705052)
    * Quantum Error Correction via Codes over GF(4) [arxiv-link](https://arxiv.org/abs/quant-ph/9608006)
    * Fast simulation of stabilizer circuits using a graph state representation [arxiv-link](https://arxiv.org/abs/quant-ph/0504117)
    * python-quaec [documentation](http://www.cgranade.com/python-quaec/index.html) [github-link](https://github.com/cgranade/python-quaec)
    * [github/hongyehu/pyclifford](https://github.com/hongyehu/PyClifford)
    * [github/abp](https://github.com/peteshadbolt/abp) Fast simulation of Clifford circuits
    * quantumclifford.jl [documentation](https://quantumsavory.github.io/QuantumClifford.jl/stable/) [github-link](https://github.com/QuantumSavory/QuantumClifford.jl)
    * the canonical name of Clifford group: Involutions on the the Barnes-Wall lattices and their fixed point sublattices [arxiv-link](https://arxiv.org/abs/math/0511084)
    * the canonical name of Pauli group: extra special group with $p=2$ [wiki-link](https://en.wikipedia.org/wiki/Extra_special_group) the Heisenberg group over a finite field [stackexchange-link](https://quantumcomputing.stackexchange.com/q/26351)
2. notation
    * $\mathbb{F}_n$: finite field
    * $\mathbb{R}$: real field
    * $\mathbb{C}$: complex field
    * $U(n)=\lbrace x\in\mathbb{C}^{n\times n}:x^\dagger x=I_n \rbrace$: unitary group
    * $\Lambda_n=\sigma_x\otimes I_n$

## Symplectic vector space

$$x,y\in \mathbb{F}_2^{2n}\Rightarrow \langle x,y\rangle=\sum_{i=1}^n x_iy_{i+n}+x_{i+n}y_i\pmod{2}$$

1. $x\in\mathbb{F}_2^{2n},\langle x,x\rangle=0$

## Pauli group

$$P_{n}=\lbrace e^{ij_{0}\pi/2}\sigma_{j_{1}}\otimes\cdots\sigma_{j_{n}}:j_{k}\in\mathbb{F}_{4}\rbrace =\lbrace \pm1,\pm i\rbrace \otimes\lbrace I,X,Y,Z\rbrace ^{n}$$

1. there exists a one-to-one mapping $P_{n}\cong \mathbb{F}_{2}^{2n+2}$ (group isomorphism)

     $$ x \in P_{n},x=\left(-1\right)^{x_0}\left(i\right)^{x_{0}^{\prime}}\prod_{i=1}^{n}X_{i}^{x_{i}}Z_{i}^{x_{i+n}}\sim x_{\mathbb{F}}=\left(x_{0},x_{0}^{\prime},x_{1},x_{2},x_{3},\cdots,x_{2n}\right) $$

2. group center $Z(P_n)=\left\{I,-I,iI,-iI\right\}$. $P/Z(P_n)$ is an Abelian group, no phase factor
3. identity $I_n\sim (0,0,\cdots,0)\in\mathbb{F}^{2n+2}_2$
4. commutation relation $x,y\in P_{n}$

     $$ xy=\left(-1\right)^{f\left(x_{\mathbb{F}},y_{\mathbb{F}}\right)}yx $$

     $$ f\left(x_{\mathbb{F}},y_{\mathbb{F}}\right)=\sum_{i=1}^{n}x_{i}y_{i+n}+x_{i+n}y_{i}\simeq\langle x_{\mathbb{F}},y_{\mathbb{F}}\rangle $$

5. inverse $x\in P_{n},y=x^{-1}$

     $$ y_{0}=x_{0}+x_{0}^{\prime}+\sum_{i=1}^{n}x_{i}x_{i+n},y_{0}^{\prime}=x_{0}^{\prime},y_{i}=x_{i} $$

6. multiplication $x,y\in P_n,z=xy$

     $$ z_{0}=x_{0}+y_{0}+\left\lfloor \frac{x_{0}^{\prime}+y_{0}^{\prime}}{2}\right\rfloor +\sum_{i=1}^{n}x_{i}y_{i+n},z_{0}^{\prime}=x_{0}^{\prime}+y_{0}^{\prime},z_{i}=x_{i}+y_{i} $$

7. one qubit example: $I=0000,X=0010,Y=0111,Z=0001$

## Symplectic pair vector set

$$ S_{n}=\lbrace \left(x,y\right)\in\mathbb{F}_{2}^{2n}\times\mathbb{F}_{2}^{2n}:\langle x,y\rangle=x^{T}\Lambda_{n}y=1\rbrace $$

1. order of the set

    $$ \left|S_{n}\right|=10\left|S_{n-1}\right|+6\left(16^{n-1}-\left|S_{n-1}\right|\right)=2^{2n-1}\times\left(4^{n}-1\right) $$

## Symplectic group

$$ Sp\left(2n,\mathbb{F}_{2}\right)=\lbrace x\in\mathbb{F}_{2}^{2n\times2n}:x\Lambda_{n}x^{T}=\Lambda_{n}\rbrace $$

1. $x\in Sp\left(2n,\mathbb{F}_{2}\right)\Rightarrow x^{T}\in Sp\left(2n,\mathbb{F}_{2}\right)$
2. $x,y\in Sp\left(2n,\mathbb{F}_{2}\right)\Rightarrow xy\in Sp\left(2n,\mathbb{F}_{2}\right)$
3. $x\in Sp\left(2n,\mathbb{F}_{2}\right)\Rightarrow x^{-1}=\Lambda_{n}x^{T}\Lambda_{n}$
4. order of the group

    $$ \left|Sp\left(2n,\mathbb{F}_{2}\right)\right|=\prod_{i=1}^{n}\left|S_{i}\right|=\prod_{i=1}^{n}(4^{i}-1)2^{2n-1} $$

5. there exists a one-to-one mapping: $x\in Sp\left(2n,\mathbb{F}_{2}\right),x\mapsto\left(a_{1},b_{1},a_{2},b_{2},\cdots,a_{n},b_{n}\right)$
    * $0\leq a_{i}<4^{i}-1$
    * $0\leq b_{i}<2^{2i-1}$
    * the mapping is constructed using transvection, details see paper [doi-link](https://doi.org/10.1063%2F1.4903507)

transvection

$$h\in\mathbb{F}_{2}^{2n},T_{h}\left(x\right)=T_{h}x=x+\langle x,h\rangle h:\mathbb{F}_{2}^{2n}
\mapsto\mathbb{F}_{2}^{2n}$$

1. $\forall h\in\mathbb{F}_{2}^{2n},T_{h}T_{h}x=x=T_{0}x$
2. $\forall x,y\in\mathbb{F}_{2}^{2n}\setminus\lbrace 0\rbrace \Rightarrow\exists a,b\in\mathbb{F}_{2}^{2n},y=T_{b}T_{a}x$
    * Lemma 2 in paper [doi-link](https://doi.org/10.1063%2F1.4903507)

## Clifford group

$$C_{n}=\lbrace x\in U\left(2^{n}\right):xP_{n}x^{\dagger}=P_{n}\rbrace /U\left(1\right)$$

1. group isomorphism $C_{n}\cong\mathbb{F}_{2}^{2n}\times Sp\left(2n,\mathbb{F}_{2}\right)$

    $$ U\in C_{n},r\in\mathbb{F}_{2}^{2n},S\in Sp\left(2n,\mathbb{F}_{2}\right) $$

    $$UX_{j}U^{\dagger}=\left(-1\right)^{r_{j}}\left(i\right)^{\sum_{k}S_{kj}S_{k+n,j}}\prod_{k=1}^{n}X_{k}^{S_{kj}}Z_{k}^{S_{k+n,j}}$$

    $$UZ_{j}U^{\dagger}=\left(-1\right)^{r_{j+n}}\left(i\right)^{\sum_{k}S_{k,j+n}S_{k+n,j+n}}\prod_{k=1}^{n}X_{k}^{S_{k,j+n}}Z_{k}^{S_{k+n,j+n}}$$

    $$\Delta=x_{0}^{\prime}+\sum_{i=1}^{n}\sum_{j=1}^{2n}x_{j}S_{ij}S_{i+n,j}$$

    $$y_{0}=x_{0}+\sum_{i=1}^{2n}x_{i}r_{i}+\sum_{i=1}^{n}\sum_{j=1}^{2n}\sum_{k=j+1}^{2n}x_{j}x_{k}S_{i+n,j}S_{ik}+\left\lfloor \frac{\Delta\%4}{2}\right\rfloor$$

    $$y_{0}^{\prime}=\Delta\%2,y_{i}=\sum_{j=1}^{2n}S_{ij}x_{j}$$

2. group center $Z(C_n)\cong P_n/Z(P_n)$
3. identity $I_{2^n} \mapsto 0^{n}\times I_n$
4. inverse *TODO*
5. multiplication: $x,y\in C{}_{n},z=y\circ x,x\mapsto r^{(x)}\times S^{(x)},y\mapsto r^{(y)}\times S^{(y)},z\mapsto r^{(z)}\times S^{(z)}$

    $$ a=xX_{\alpha}x^{\dagger},xZ_{\alpha}x^{\dagger} $$

    $$ b=yay^{\dagger} $$

    $$ a_{0}=r_{\alpha}^{(x)},a_{0}^{\prime}=\sum_{k=1}^{n}S_{k\alpha}^{(x)}S_{k+n,\alpha}^{(x)},a_{i}=S_{i\alpha}^{(x)} $$

    $$ \Delta=a_{0}^{\prime}+\sum_{i=1}^{n}\sum_{j=1}^{2n}a_{j}S_{ij}^{(y)}S_{i+n,j}^{(y)}-\sum_{k}S_{k\alpha}^{(z)}S_{k+n,\alpha}^{(z)} $$

    $$ r_{\alpha}^{(z)}=b_{0}=a_{0}+\sum_{i=1}^{2n}a_{i}r_{i}^{(y)}+\sum_{i=1}^{n}\sum_{j=1}^{2n}\sum_{k=j+1}^{2n}a_{j}a_{k}S_{i+n,j}^{(y)}S_{ik}^{(y)}+\left\lfloor \frac{\Delta\%4}{2}\right\rfloor $$

    $$ S_{i\alpha}^{(z)}=b_{i}=\sum_{j=1}^{2n}S_{ij}^{(y)}a_{j}=\sum_{j=1}^{2n}S_{ij}^{(y)}S_{j\alpha}^{(x)} $$

6. example

    $$ HXH=Z,HYH=-Y,HZH=X,H\simeq 00\times\begin{bmatrix}0 & 1\\1 & 0\end{bmatrix} $$

    $$ XXX=X,XYX=-Y,XZX=-Z,X\simeq 01\times\begin{bmatrix}1 & 0\\0 & 1\end{bmatrix} $$

    $$ YXY=-X,YYY=Y,YZY=-Z,Y\simeq 11\times\begin{bmatrix}1 & 0\\0 & 1\end{bmatrix} $$

    $$ ZXZ=-X,ZYZ=-Y,ZZZ=Z,Z\simeq 01\times\begin{bmatrix}1 & 0\\0 & 1\end{bmatrix} $$

    $$ SXS^\dagger=Y,SYS^\dagger=-X,SZS^\dagger=Z,S\simeq 00\times\begin{bmatrix}1 & 0\\1 & 1\end{bmatrix} $$

    $$ \mathrm{CNOT}\simeq 0000\times\begin{bmatrix}1&0&0&0\\ 1&1&0&0\\0&0&1&1\\0&0&0&1\end{bmatrix} $$

TODO

1. clifford circuit

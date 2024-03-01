# Symplectic group over F2

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
    * 6-qubit optimal Clifford circuits [doi-link](https://doi.org/10.1038/s41534-022-00583-7)
2. notation
    * $\mathbb{F}_n$: finite field
    * $\mathbb{R}$: real field
    * $\mathbb{C}$: complex field
    * $U(n)=\lbrace x\in\mathbb{C}^{n\times n}:x^\dagger x=I_n \rbrace$: unitary group
    * $\Lambda_n=\sigma_x\otimes I_n$

## Symplectic vector space

$$x,y\in \mathbb{F}_2^{2n}\Rightarrow \langle x,y\rangle=\sum_{i=1}^n x_iy_{i+n}+x_{i+n}y_i\pmod{2}$$

1. $x\in\mathbb{F}_2^{2n},\langle x,x\rangle=0$

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

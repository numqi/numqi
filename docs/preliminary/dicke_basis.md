# Dicke basis

See paper for detail explaination [arxiv-link](http://arxiv.org/abs/1904.07358)

For $n$ qudit-$d$ Hilbert space $\mathcal{H}_d^{\otimes n}$, you may think of $n$ Boson particles $A_1,A_2,\cdots,A_n$, the pure state vector must satisfy the permutation symmetry

$$ P_{A_rA_s} \psi=\psi,\forall r,s=1,2,\cdots,n$$

with the permutation operator $P_{A_rA_s}$ interchanging $B_r$-party and $B_s$-party. If take $\psi$ as a high-dimensioanl array (tensor), it's equivalent to say

$$ \left[P_{B_{r}B_{s}}\psi\right]_{i_{1}\cdots i_{r-1}i_{r}i_{r+1}\cdots i_{s-1}i_{s}i_{s+1}\cdots i_{n}}=\psi_{i_{1}\cdots i_{r-1}i_{s}i_{r+1}\cdots i_{s-1}i_{r}i_{s+1}\cdots i_{n}} $$

All such states are located in a so-called Bosonic-symmetric subspace with Dick states as basis

$$ \left|D_{n,\vec{k}}\right\rangle =\left(\frac{n!}{k_{0}!k_{1}!\cdots k_{d-1}!}\right)^{-1/2}\sum_{wt_{i}\left(x\right)=k_{i}}\left|x\right\rangle $$

$$ \vec{k}=\left[k_{0},k_{1},\cdots,k_{d-1}\right],k_{i}\geq0,\sum_{i}k_{i}=n $$

$$ \left\langle D_{n,\vec{k}'}\right.\left|D_{n,\vec{k}}\right\rangle =\delta_{\vec{k}\vec{k}'} $$

For two qubits, the basis can be explicitly written down.

$$ \left|D_{2,00}\right\rangle =\left|00\right\rangle ,\sqrt{2}\left|D_{2,01}\right\rangle =\left|01\right\rangle +\left|10\right\rangle ,\left|D_{2,11}\right\rangle =\left|11\right\rangle $$

One can prove these basis are orthogonal and normalized and the spanned subspace is of dimension $\binom{n+d-1}{d-1}$. The dimensions for some specific values are listed below

| $n$ | 8 | 16 | 32 | 64 | 128 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| $d=3$ | 45 | 153 | 561 | 2145 | 8384 |
| $d=4$ | 165 | 969 | 6545 | 47905 | 366145 |
| $d=5$ | 495 | 4845 | 58904 | 814385 | 12082785 |

## Partial trace of Dicke states

When doing Bosonic extension related tasks, it's common to take partial trace of Dicke states. For example, given a pure state $|\psi_{AB^n}\rangle$ in (k+1) partite Hilbert space $A,B_0,B_1,B_2,\cdots,B_{k-1}$ and it's symmetric in $B_i$s' subspace

$$ P_{B_rB_s}|\psi_{AB^n}\rangle=|\psi_{AB^n}\rangle $$

So we can write it in Dicke basis

$$ |\psi_{AB^{k}}\rangle=\sum_{i\vec{k}}p_{i\vec{k}}|i\rangle\otimes|D_{n,\vec{k}}\rangle $$

with $|i\rangle$ belongs to $A$'s Hilbert space. Then its reduced density matrix of $AB$ is

$$ \mathrm{Tr}_{B^{n-1}}\left[|\psi_{AB^{n}}\rangle\left\langle \psi_{AB^{n}}\right|\right]=\sum_{ij\vec{k}\vec{k}'}p_{i\vec{k}}p_{j\vec{k}'}^{*}\mathcal{B}_{rs\vec{k}\vec{k}'}|ir\rangle\left\langle js\right| $$

with $|r \rangle,|s\rangle$ belongs to $B_i$'s Hilbert space. The $\mathcal{B}_{rs\vec{k}\vec{k}'}$ can be calculated by

$$ \mathcal{B}_{rs\vec{k}\vec{k}'}=\mathrm{Tr}_{B^{n-1}}\left[\left\langle r\right.|D_{n,\vec{k}}\rangle\left\langle D_{n,\vec{k}'}\right|\left.s\right\rangle \right]=\frac{1}{n}\sqrt{k_{r}k_{s}^{\prime}}\prod_{i}\delta_{k_{i},k_{i}^{\prime}} $$

For n qubits, the tensor $\mathcal{B}$ can be explicitly written down.

$$ |D_{n,0}\rangle=|D_{n,0n}\rangle=|1^{\otimes n}\rangle,|D_{n,n}\rangle=|D_{n,n0}\rangle=|0^{\otimes n}\rangle $$

$$ n\mathcal{B}_{00,\alpha\beta}=\alpha\delta_{\alpha\beta},n\mathcal{B}_{11,\alpha\beta}=\left(n-\alpha\right)\delta_{\alpha\beta},n\mathcal{B}_{01,\alpha\beta}=n\mathcal{B}_{10,\beta\alpha}=\sqrt{\left(\alpha+1\right)\left(n-\alpha\right)}\delta_{\alpha+1,\beta} $$

## Quantum Gates in Qubits Symmetric Subspace

The following results are obtained numerically and it **seems** to be correct.

### x-rotation gate

$$ R_{x}\left(\theta\right)=e^{-i\sigma_{x}/2} $$

$$ \langle D_{n,\alpha}|\bigotimes_{i=0}^{n-1}R_{x}^{\left(i\right)}\left(\theta\right)|D_{n,\beta}\rangle=\left[\mathrm{e}^{-iA^{\left(n\right)}\theta/2}\right]_{\alpha\beta} $$

$$ A^{\left(1\right)}=\sigma_{x} $$

$$ A^{\left(2\right)}=\sqrt{2}\left[\begin{matrix} & 1\\
1 &  & 1\\
 & 1
\end{matrix}\right] $$

$$ A^{\left(3\right)}=\left[\begin{matrix} & \sqrt{3}\\
\sqrt{3} &  & \sqrt{4}\\
 & \sqrt{4} &  & \sqrt{3}\\
 &  & \sqrt{3}
\end{matrix}\right] $$

$$ \left[A^{\left(n\right)}\right]_{ij}=\delta_{i,j+1}\sqrt{j\left(n-j\right)}+\delta_{i+1,j}\sqrt{i\left(n-i\right)} $$

### z-rotation gate

$$ R_{z}\left(\theta\right)=e^{-i\sigma_{z}/2} $$

$$ \langle D_{n,\alpha}|\bigotimes_{i=0}^{n-1}R_{z}^{\left(i\right)}\left(\theta\right)|D_{n,\beta}\rangle=\left[\mathrm{e}^{-iB^{\left(n\right)}\theta/2}\right]_{\alpha\beta} $$

$$ B^{\left(1\right)}=\sigma_{z}=\mathrm{diag}\left\{ 1,-1\right\}  $$

$$ B^{\left(2\right)}=\mathrm{diag}\left\{ 2,0,-2\right\} $$

$$ B^{\left(3\right)}=\mathrm{diag}\left\{ 3,1,-1,3\right\} $$

$$ \left[B^{\left(n\right)}\right]_{ij}=\delta_{ij}\left(n+2-2i\right) $$

### ZZ double qubit rotation gate

$$ R_{zz}^{\left(ij\right)}\left(\theta\right)=e^{-i\sigma_{z}^{\left(i\right)}\sigma_{z}^{\left(j\right)}/2} $$

$$ \bigotimes_{i=0}^{n-1}R_{zz}^{\left(AB_{i}\right)}\left(\theta\right)\in\mathbb{C}^{2^{n+1}\times2^{n+1}} $$

$$ \langle x,D_{n,\alpha}|\bigotimes_{i=0}^{n-1}R_{zz}^{\left(AB_{i}\right)}\left(\theta\right)|y,D_{n,\beta}\rangle=\left[\mathrm{e}^{-iC^{\left(n\right)}\theta/2}\right]_{xk,\alpha\beta} $$

$$ C^{\left(n\right)}=\sigma_{z}\otimes B^{\left(n\right)} $$

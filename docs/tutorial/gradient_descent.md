# Gradient Descent

Gradient Descent is not "silver bullet". The following problems are identitied as difficult problems for gradient optimization by numerical experiments.

> Example 1: given $m$ real vectors $x_1,x_2,\cdots,x_m\in \mathbb{R}^n$, find the maximum possible value $\alpha$ which could also be nonexistent

$$\max_{\lambda_i} \alpha$$

$$s.t.\begin{cases}
\alpha u=\sum_{i=1}^{m}\lambda_{i}x_{i}\\
\sum_{i=1}^{m}\lambda_{i}=1\\
\lambda_{i\geq0}
\end{cases}$$

comment: easy for convex optimization (linear programming), almost impossible for gradient descent (the first constraint equation cannot be parameterized)

> Example 2: Superactivation, given a series of real matrix $B^{\alpha}\in \mathbb{R}^{16\times 16},\alpha=1,2,\cdots,64$ (see [github-link](https://github.com/husisy/biquadratic-optimization) for detailed)

$$\min_{a,b} \sum_{\alpha}\left|\left\langle a\right|B^{\alpha}\left|b\right\rangle \right|^{2}$$

comment: the solution is almost unique, non convex, the success probability for random initialization is almost 1 over thousand

> Example 3: Given a three-partites $\mathcal{H}^A\otimes\mathcal{H}^{B_1}\otimes\mathcal{H}^{B_2}$ density matrix $\rho_{AB_1B_2}$ with $B_1/B_2$ permutation symmetry

$$\rho_{AB_{1}B_{2}}\in\mathcal{H}_{d_{A}}\otimes\mathcal{H}_{3}\otimes\mathcal{H}_{3},\quad \rho_{AB_{1}B_{2}}\succeq 0,\quad\rho_{AB_{1}B_{2}}=\rho_{AB_{2}B_{1}}=P_{B_{1}B_{2}}\rho_{AB_{1}B_{2}}P_{B_{1}B_{2}}$$

where $P_{B_1B_2}$ is the permutation operator, find the parameters $p_{x\alpha,y\beta},q_{x\alpha,y\beta}$ satisifying

$$\rho_{AB_{1}B_{2}}=\sum_{x,y,\alpha,\beta}p_{x\alpha,y\beta}\left|x,\psi_{B}^{\alpha}\right\rangle \left\langle y,\psi_{B}^{\beta}\right|+\sum_{x,y,\alpha,\beta}q_{x\alpha,y\beta}\left|x,\psi_{F}^{\alpha}\right\rangle \left\langle y,\psi_{F}^{\beta}\right|$$

$$p_{x\alpha,y\beta}\succeq0,\quad q_{x\alpha,y\beta}\succeq0,\quad\sum_{x\alpha}p_{x\alpha,x\alpha}+q_{x\alpha,x\alpha}=1$$

with the complete basis set (Bosonic and Fermionic parts)

$$\left|\psi_{B}^{\alpha}\right\rangle \in\lbrace \left|00\right\rangle ,\left|11\right\rangle ,\left|22\right\rangle ,\frac{1}{\sqrt{2}}\left(\left|01\right\rangle +\left|10\right\rangle \right),\frac{1}{\sqrt{2}}\left(\left|02\right\rangle +\left|20\right\rangle \right),\frac{1}{\sqrt{2}}\left(\left|12\right\rangle +\left|21\right\rangle \right)\rbrace $$

$$ \left|\psi_{F}^{\alpha}\right\rangle \in\lbrace \frac{1}{\sqrt{2}}\left(\left|01\right\rangle -\left|10\right\rangle \right),\frac{1}{\sqrt{2}}\left(\left|02\right\rangle -\left|20\right\rangle \right),\frac{1}{\sqrt{2}}\left(\left|12\right\rangle -\left|21\right\rangle \right)\rbrace  $$

comment: easy for convex optimization (semidefinite programming), hard for `L-BFGS-B` algorithm, feasible for `BFGS` algorithm, feasible for `adam` optimizer

The above three examples are obtained by many tries of numerical experiments which may not "truely" hard examples for gradient descents. Idea of trials are welcome and the code snippets are provided colab-link (TODO) and python-scirpts (TODO)

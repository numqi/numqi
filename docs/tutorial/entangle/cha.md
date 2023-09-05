# Convex Hull Approximation

The bipartite  density matrix $\rho$ is separated if the following decomposition exists

$$ \rho =\sum_i{p_i\rho_A\otimes \rho _B} $$

$$ s.t.\begin{cases}
\sum_{i}p_{i}=1\\
p_{i}\geq0\\
\rho_{A},\rho_{B}\succeq0\\
\mathrm{Tr}\left[\rho_{A}\right]=\mathrm{Tr}\left[\rho_{B}\right]=1
\end{cases} $$

otherwise entangled. All separated density matrices make the convex set SEP. The quantum entropy of entanglement (REE) of a given density matrix $\rho$ is defined as the minimum relative entropy $S$ with respect to the SEP.

$$ S\left( \rho ||\sigma \right) =\mathrm{Tr}\left[ \rho \ln \rho -\rho \ln \sigma \right] $$

$$ \mathrm{REE}\left( \rho \right) =\min_{\sigma \in \mathrm{SEP}} S\left( \rho ||\sigma \right) $$

The REE is hard to calculate. Below, we use a variational method `AutodiffREE` to approximate REE. According to paper "separability criterion and inseparable mixed states with positive paretial transposition" [arxiv-link](https://arxiv.org/abs/quant-ph/9703004v2), all elements in SEP can be represented as

$$ \sigma=\sum_{i=1}^{d_Ad_B}p_{i}|\psi_{A,i}\rangle\langle\psi_{A,i}|\otimes|\psi_{B,i}\rangle\langle\psi_{B,i}|. $$

For easier optimization, a larger number of pure states can be choosen $i=1,2,\cdots, N$ for $N>d_Ad_B$. The optimization target for AutodiffREE is

$$ \min_{p_i,\psi _A,\psi _B} S\left( \rho ||\sigma \right) $$

$$ s.t.\; \sum_i{p_i}=\langle \psi_{A,i}|\psi_{A,i}\rangle =\langle \psi_{B,i}|\psi_{B,i}\rangle =1 $$

with trainable parameters $p_i,| \psi_{A,i} \rangle,| \psi_{B,i} \rangle$.

```Python
import numpy as np
import matplotlib.pyplot as plt
import numqi
```

## CHA boundary

`numqi.entangle.CHABoundaryAutodiff` is not good, use `numqi.entangle.AutodiffCHAREE` instead.

```Python
dimA = 3
dimB = 3
dm_target = numqi.random.rand_density_matrix(dimA*dimB, kind='haar')
beta_l,beta_u = numqi.entangle.get_ppt_boundary(dm_target, (dimA,dimB))
print(beta_l,beta_u)

## TODO replace with NEW CHA
## sometimes num_state=200 is not enough, maxiter also need finetune
# model = numqi.entangle.CHABoundaryAutodiff(dimA, dimB=dimB)
# model.set_dm_target(dm_target)
# loss = numqi.optimize.minimize_adam(model, num_step=150, theta0='uniform', tqdm_update_freq=1)
# beta_cha = -loss
```

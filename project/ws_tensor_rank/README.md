# tensor rank

1. link
   * [github/tensorspace](https://github.com/thetensor-space/TensorSpace) magma based
2. completely entangled subspace (CES)
   * maximal dimension in space $\bigotimes_{i=1}^{k}\mathcal{H}_k$: $d_1d_2\cdots d_k-(d_1+\cdots+d_k)+k-1$ [doi-link](https://doi.org/10.1007/BF02829441)
3. perfectly entangled subspace

`demo_schmidt_rank.py`, `GenTiles1`

| dim | `loss(1)` | `loss(2)` | time (second) |
| :-: | :-: | :-: | :-: |
| `4` | $0.030$ | $9.2\times 10^{-15}$ | `0.82` |
| `8` | $0.016$ | $2.2\times 10^{-12}$ | `2.5` |
| `16` | $0.0079$ | $2.3\times 10^{-12}$ | `3.4` |
| `32` | $0.0038$ | $4.3\times 10^{-12}$ | `20.0` |
| `64` | $0.0018$ | $5.8\times 10^{-12}$ | `352` |

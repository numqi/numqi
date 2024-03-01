# Entanglement

1. abbreviations
   * PB: product basis
   * UCPB: uncompletable product basis
   * SUCPB: strongly uncompletable product basis
   * UPB: unextendable product basis
   * BES: bound entangled state
2. entanglement measure
   * convex
   * non-increasing under LOCC on average
   * non-increasing under local measurement
   * asymptotic continuity

If a bipartite density matrix $\rho$ can be written as

$$ \rho_{AB}=\sum_i{p_i\rho_A^{(i)}\otimes \rho_B^{(i)}} $$

we call it a separable density matrix, otherwise a entangled density matrix. Determining whether a density matrix is separable or entangled is important in quantum information field. This package is served to do jobs related to quantum entanglement.

Density matrix can be divided into separable states and entangled states. The separable set is defined as all matrices satisfying

PPT

UPB/BES

## Numerical range

PPT Numerical range: given hermitian operators

$$ O=\left\{ O_{1},O_{2},\cdots,O_{n}\right\} \subset\mathbb{C}^{d_{A}d_{B}\times d_{A}d_{B}} $$

$$ R_{\mathrm{PPT}}\left[O\right]=\left\{ \left(\mathrm{Tr}\left[xO_{1}\right],\mathrm{Tr}\left[xO_{2}\right],\cdots,\mathrm{Tr}\left[xO_{n}\right]\right):x\in S_{\mathrm{PPT},d_{A},d_{B}}\right\} $$

## k-bosonic extendible

Carathéodory's theorem [wiki-link](https://en.wikipedia.org/wiki/Carath%C3%A9odory%27s_theorem_(convex_hull))

## unextandable product basis

1. reference
   * `[DiVincenzo2003]` Unextendible Product Bases, Uncompletable Product Bases and Bound Entanglement [arxiv-link](https://arxiv.org/abs/quant-ph/9908070v3)
   * Unextendible and uncompletable product bases in every bipartition [doi-link](https://doi.org/10.1088/1367-2630/ac9e14)
   * `[Horodecki1999]` Rank two bipartite bound entangled states do not exist [doi-link](https://arxiv.org/abs/quant-ph/9910122)
2. product basis (OPB)
   * definition: a multipartite quantum system $\mathcal{H}=\bigotimes_{i=1}^m \mathcal{H}_i$, a set $S$ of pure orthogonal product states spanning a subspace $\mathcal{H}_S$ of $\mathcal{H}$
3. uncompletable product basis (UCPB)
   * definition: PB, its orthogonal space $\mathcal{H}_S^\bot$ contains fewer mutually orthogonal product states than the dimension of $\mathcal{H}_S^\bot$
   * if $S$ is exactly distinguishable by local von Neumann measurements and classical communication, then $S$ is completable
   * any PB with less than three elements $\leq 3$ is NOT UCPB (proposition 2 in `[DiVincenzo2003]`)
   * any bipartite PB with less than four elements $\leq 4$ is NOT UCPB (proposition 2 in `[DiVincenzo2003]`)
   * for $d_A=3,d_B=3$ bipartite system, any PB with more than six elements $\geq 6$ is NOT UCPB (proposition 2 in `[DiVincenzo2003]`)
   * Rank two bound entangled states do not exist `[Horodecki1999]`
   * for $d_A=2,d_B\geq 2$ bipartite system, any PB is NOT UCPB (theorem 4 in `[DiVincenzo2003]`)
4. locally extended Hilbert space: $\mathcal{H}_{ext}=\bigotimes_{i=1}^m(\mathcal{H}_i\oplus \mathcal{H}_i^{\prime})$
5. strongly uncompletable product basis (SUCPB)
   * definition: PB, PB, for all $\mathcal{H}_{ext}$, its orthogonal space $\mathcal{H}_S^\bot$ contains fewer mutually orthogonal product states than the dimension of $\mathcal{H}_S^\bot$
   * if $S$ is exactly distinguishable by local POVMs and classical communication, then $S$ is completable in some $\mathcal{H}_{ext}$
   * if $S$ is not SUCPB, then its orthogonal projector is separable
6. unextendable product basis (UPB)
   * definition: PB, its orthogonal space $\mathcal{H}_S^\bot$ contains no product states
   * iff condition: Lemma 1 in `[DiVincenzo2003]`
   * check whether a bi-partite PB is UPB: `numqi.matrix_space.DetectRankOneModel`
   * projector to orthogonal space is proportional to a bounded entangled state
7. set inclusion
   * $\mathrm{UPB}\subset \mathrm{SUCPB}\subset \mathrm{UCPB}$
   * example
     * `pyramid(3x3,5)`: UPB
     * `Pyr34(3x4,5)`: UCPB, not SUCPB (SUCPB in `3x5`)
     * `Pyr34+(3x4,6)`: SUCPB, not UPB
     * `Shifts(2x2x2,4)`
     * `GenShifts(2^(2k-1),2k)`: `(2x2x2,4)`, `(2x2x2x2x2,6)`, ...
     * `GenTiles1`: $n\otimes n,n/2\in \mathbb{Z}$, $n^2-2n+1$ members, $(4\otimes 4,9)$, $(6\otimes 6,25)$
     * `GenTiles2`: $m\times n,n>3,m\geq 3n\geq m$, $mn-2m+1$ members

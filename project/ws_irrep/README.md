# symext irrep

Werner state boundary $\frac{k+d^2-d}{kd^2+d-1}$, the absolute error is within $10^{-8}$

| $(d,k)$| QETLAB time (s)| hermitian basis (s)  | irrep time (s) | $\alpha$ |
| :-: | :-: | :-: | :-: | :-: |
| $(2,6)$ | 0.14 | 0.77 | 0.10 | 0.615 |
| $(2,8)$ | 0.19 | 23.84 | 0.16 | 0.588 |
| $(2,10)$ | 12.60 | NA | 0.16 | 0.571 |
| $(2,16)$ | NA | NA | 0.32 | 0.545 |
| $(2,32)$ | NA | NA | 3.18 | 0.523 |
| $(2,32)$ | NA | NA | 51.96 | 0.512 |
| $(3,3)$ | 0.62 | 0.87 | 0.51 | 0.818 |
| $(3,4)$ | 7.96 | 6.69 | 2.38 | 0.714 |
| $(3,5)$ | NA | NA | 11.56 | 0.647 |
| $(3,6)$ | NA | NA | 55.60 | 0.6 |

1. computer spec: AMD R7-5800H, 16 cpu (hyperthreaded), 16GB memory
2. naive SDP: the naive kext SDP as in qetlab
    * python/cvxpy with MOSEK solver, should be comparable to qetlab
    * momery is the bottleneck, time may be slightly overestimated
3. SDP with irreducible representation: @LYN-paper
    * sparse trick cannot give correct werner/isotropic kext bounary
    * python/cvxpy with MOSEK solver

count number of parameters

| $(d,k)$ | Young Tableau $d^k=\sum_i m_i\times s_i$ | parameters $(\sum_i s_i^2,d^{2k})$ |
| :-: | :-: | :-: |
| $(3,3)$ | $3^3 = 10\times 1 + 8\times 2 + 1\times 1$ | $(165,729)$ |
| $(3,4)$ | $3^4 = 15\times 1 + 15\times 3 + 6\times 2 + 3\times 3$ | $(495,6561)$ |
| $(3,5)$ | $3^5 = 21\times 1 + 24\times 4 + 15\times 5 + 6\times 6 + 3\times 5$ | $(1287,59049)$ |
| $(3,6)$ | $3^6 = 28\times 1 + 35\times 5 + 27\times 9 + 10\times 5 + 10\times 10 + 8\times 16 + 1\times 5$ | $(3003,531441)$ |
| $(4,3)$ | $4^3 = 20\times 1 + 20\times 2 + 4\times 1$ | $(816,4096)$ |
| $(4,4)$ | $4^4 = 35\times 1 + 45\times 3 + 20\times 2 + 15\times 3 + 1\times 1$ | $(3876,65536)$ |
| $(4,5)$ | $4^5 = 56\times 1 + 84\times 4 + 60\times 5 + 36\times 6 + 20\times 5 + 4\times 4$ | $(15504,1048576)$ |

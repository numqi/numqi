# misc

## Q-and-A

> how this package name `numqi` comes?

Initially, this package is named `numpyqi`, later it's shortened to be `numqi` (pronounce: num-py-q-i). The reason is that `numpy` is the most popular python package for numerical computation, and `numqi` is based on `numpy`. The name `numqi` is also a play on the word `numpy` and quantum information.

1. short, no more than 7 letters
2. keyword: quantum information, numpy, python, optimization
3. NEP41: should not include the whole word "numpy"
4. example (good and bad)
    * `numpyqi`: bad
    * `numqi`: pronounced as "num py q i", emphasize that it's based on numpy and focuses on quantum information field
    * `numqy`: bad, confused with `numpy`

> why `233` appears so frequently?

`233` is a prime number! Internet slang that essentially means “LOL.”

## Publications

This package is to support following papers

1. detecting entanglement by pure bosonic extension [arxiv-link](https://arxiv.org/abs/2209.10934) `numqi.entangle`
2. Quantum variational learning for quantum error-correcting codes [doi-link](https://doi.org/10.22331/q-2022-10-06-828) `numqi.qec`
3. Tapping into Permutation Symmetry for Improved Detection of k-Symmetric Extensions [doi-link](https://doi.org/10.3390/e25101425) `numqi.entangle` `numqi.group.symext`
4. Variational learning algorithms for quantum query complexity [arxiv-link](https://arxiv.org/abs/2205.07449) `numqi.query`
5. Variational approach to unique determinedness in pure-state tomography [arxiv-link](https://arxiv.org/abs/2305.10811) [doi-link](https://doi.org/10.1103/PhysRevA.109.022425) `numqi.unique_determine`
6. Maximum entropy methods for quantum state compatibility problems [arxiv-link](https://arxiv.org/abs/2207.11645) `numqi.maximum_entropy`
7. Simulating Noisy Quantum Circuits with Matrix Product Density Operators [arxiv-link](https://arxiv.org/abs/2004.02388) `numqi.sim`
8. Quantifying Subspace Entanglement with Geometric Measures [arxiv-link](https://arxiv.org/abs/2311.10353) `numqi.matrix_space`

## Acknowledgement

Thanks to (alphabetical order)

1. Bei ZENG
2. Chenfeng CAO [github](https://github.com/caochenfeng)
3. Shiyao HOU [github](https://github.com/houbigdream)
4. Xuanran ZHU [github](https://github.com/Sunny-Zhu-613)
5. Yichi ZHANG [github](https://github.com/Yichi-Lionel-Cheung)
6. Youning LI
7. Zheng AN [github](https://github.com/Plmono)
8. Zipeng WU [github](https://github.com/wuzp15)

Thanks to the following open source projects (alphabetical order)

1. cvxpy [github](https://github.com/cvxpy/cvxpy)
2. cvxquad [github](https://github.com/hfawzi/cvxquad)
3. pytorch [github](https://github.com/pytorch/pytorch)
4. QETLAB [github](https://github.com/nathanieljohnston/QETLAB)

## abbreviation

1. CHA: convex hull approximation
2. PPT: positive partial transpose
3. QEC: quantum error correction
4. QECC: quantum error correction code

model name

1. `pureb`: pure bosonic extension
2. `symext`: symmetric extension
3. `varqecc`: variational quantum

## license

GNU Affero General Public License v3.0, see [github-repo/LICENSE](https://github.com/husisy/numqi/blob/main/LICENSE)

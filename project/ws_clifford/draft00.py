import numpy as np

import numqi

# TODO given (cli_r,cli_R), find the corresponding unitary matrix, furthermore the clifford quantum circuit


def clifford_inverse(rx, Sx, ry, Sy):
    # z=y \circ x
    assert rx.shape[0]%2==0
    N0 = rx.shape[0]//2
    assert (rx.shape==(2*N0,)) and (Sx.shape==(2*N0,2*N0))
    assert (ry.shape==(2*N0,)) and (Sy.shape==(2*N0,2*N0))
    assert all(x.dtype.type==np.uint8 for x in [rx,ry,Sx,Sy])
    Sz = (Sy @ Sx) % 2
    tmp0 = np.einsum(Sx[:N0], [0,1], Sx[N0:], [0,1], [1], optimize=True)
    tmp1 = np.einsum(Sx, [0,1], Sy[:N0], [2,0], Sy[N0:], [2,0], [1], optimize=True)
    tmp2 = np.einsum(Sz[:N0], [0,1], Sz[N0:], [0,1], [1], optimize=True)
    # assert np.all((tmp0+tmp1+tmp2)%2==0)
    delta = ((tmp0 + tmp1 - tmp2)%4).astype(np.uint8)
    # alpha=0 j=1 k=2 i=3
    tmp0 = np.triu(np.ones(2*N0, dtype=np.uint8), k=1)
    tmp1 = np.einsum(Sx, [1,0], Sx, [2,0], Sy[N0:], [3,1], Sy[:N0], [3,2], tmp0, [1,2], [0], optimize=True)
    rz = (rx + ry@Sx + tmp1 + delta//2) % 2
    return rz, Sz


def clifford_inverse(rx, Sx):
    assert rx.shape[0]%2==0
    N0 = rx.shape[0]//2
    assert (rx.shape==(2*N0,)) and (Sx.shape==(2*N0,2*N0))
    Sy = numqi.group.spf2.inverse(Sx)
    tmp0 = np.einsum(Sx[:N0], [0,1], Sx[N0:], [0,1], [1], optimize=True)
    tmp1 = np.einsum(Sx, [0,1], Sy[:N0], [2,0], Sy[N0:], [2,0], [1], optimize=True)
    delta = (tmp0 + tmp1)%4
    tmp0 = np.triu(np.ones(2*N0, dtype=np.uint8), k=1)
    tmp1 = np.einsum(Sx, [1,0], Sx, [2,0], Sy[N0:], [3,1], Sy[:N0], [3,2], tmp0, [1,2], [0], optimize=True)
    ry = (Sy @ (rx + tmp1 + delta//2)) % 2
    return ry,Sy


# N0 = 2
# for _ in range(10):
#     rx = numqi.random.rand_F2(2*N0)
#     Sx = numqi.random.rand_SpF2(N0)
#     ry, Sy = clifford_inverse(rx, Sx)
#     # for _ in range(10):
#     pauli = numqi.random.rand_F2(2*N0+2)

#     tmp0 = numqi.sim.clifford.apply_clifford_on_pauli(pauli, rx, Sx)
#     ret0 = numqi.sim.clifford.apply_clifford_on_pauli(tmp0, ry, Sy)
#     print(np.array_equal(pauli, ret0))

# Simple DMRG tutorial.  This code contains a basic implementation of the infinite system algorithm
#
#
# Copyright 2013 James R. Garrison and Ryan V. Mishmash.
# Open source under the MIT license.  Source code at
# <https://github.com/simple-dmrg/simple-dmrg/>
import types
import collections
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

def _make_pauli():
    s0=np.array([[1.0, 0.0], [0.0, 1.0]])
    sx=np.array([[0.0, 1.0], [1.0, 0.0]])
    sy=np.array([[0.0, -1j], [1j, 0.0]])
    sz=np.array([[1.0, 0.0], [0.0, -1.0]])
    ret = types.SimpleNamespace(
        s0 = s0,
        sx = sx,
        sy = sy,
        sz = sz,
        sp = np.array([[0,1],[0,0]]),
        sm = np.array([[0,0],[1,0]]),
        s0s0 = np.kron(s0, s0),
        s0sx = np.kron(s0, sx),
        s0sy = np.kron(s0, sy),
        s0sz = np.kron(s0, sz),
        sxs0 = np.kron(sx, s0),
        sxsx = np.kron(sx, sx),
        sxsy = np.kron(sx, sy),
        sxsz = np.kron(sx, sz),
        sys0 = np.kron(sy, s0),
        sysx = np.kron(sy, sx),
        sysy = np.kron(sy, sy),
        sysz = np.kron(sy, sz),
        szs0 = np.kron(sz, s0),
        szsx = np.kron(sz, sx),
        szsy = np.kron(sz, sy),
        szsz = np.kron(sz, sz),
    )
    return ret

Block = collections.namedtuple("Block", ["length", "basis_size", "operator_dict"])

def is_valid_block(block):
    for op in block.operator_dict.values():
        if op.shape[0] != block.basis_size or op.shape[1] != block.basis_size:
            return False
    return True


def H2(Sz1, Sp1, Sz2, Sp2):  # two-site part of H
    """Given the operators S^z and S^+ on two sites in different Hilbert spaces
    (e.g. two blocks), returns a Kronecker product representing the
    corresponding two-site term in the Hamiltonian that joins the two sites.
    """
    J = Jz = 1.
    ret = (J / 2) * (scipy.sparse.kron(Sp1, Sp2.T.conj()) + scipy.sparse.kron(Sp1.T.conj(), Sp2)) + Jz * scipy.sparse.kron(Sz1, Sz2)
    return ret

def enlarge_block(block):
    """enlarges the provided Block by a single site, returning an EnlargedBlock"""
    eye0 = scipy.sparse.identity(block.basis_size)
    tmp0 = H2(block.operator_dict["conn_Sz"], block.operator_dict["conn_Sp"], pauli.sz/2, pauli.sp)
    tmp1 = {
        "H": scipy.sparse.kron(block.operator_dict["H"], scipy.sparse.identity(model_d)) + scipy.sparse.kron(eye0, H1) + tmp0,
        "conn_Sz": scipy.sparse.kron(eye0, pauli.sz/2),
        "conn_Sp": scipy.sparse.kron(eye0, pauli.sp),
    }
    ret = Block(length=(block.length + 1), basis_size=(block.basis_size * model_d), operator_dict=tmp1)
    return ret

def single_dmrg_step(sys, dim_cut):
    """Performs a single DMRG step using `sys` as the system and `env` as the
    environment, keeping a maximum of `dim_cut` states in the new basis"""
    sys2 = enlarge_block(sys)
    env2 = sys2

    # Construct the full superblock Hamiltonian.
    tmp0 = scipy.sparse.kron(sys2.operator_dict["H"], scipy.sparse.identity(env2.basis_size))
    tmp1 = scipy.sparse.kron(scipy.sparse.identity(sys2.basis_size), env2.operator_dict["H"])
    tmp2 = H2(sys2.operator_dict["conn_Sz"], sys2.operator_dict["conn_Sp"], env2.operator_dict["conn_Sz"], env2.operator_dict["conn_Sp"])
    energy, psi0 = scipy.sparse.linalg.eigsh(tmp0+tmp1+tmp2, k=1, which="SA")
    energy = energy.item()

    # Construct the reduced density matrix of the system by tracing out the environment
    tmp0 = psi0.reshape(sys2.basis_size, -1)
    rho = tmp0 @ tmp0.T.conj()
    EVL, EVC = np.linalg.eigh(rho)
    # truncation_error = 1-EVL[(-dim_cut):].sum()
    EVCi = EVC[:,(-dim_cut):] #keep the largest-several eigenvectors

    tmp0 = {k:(EVCi.T.conj() @ v @ EVCi) for k,v in sys2.operator_dict.items()}
    ret = Block(length=sys2.length, basis_size=EVCi.shape[1], operator_dict=tmp0)
    return ret, energy

pauli = _make_pauli()

# Model-specific code for the Heisenberg XXZ chain
H1 = np.array([[0, 0], [0, 0]], dtype=np.float64)  # single-site portion of H is zero

# infinite_system_algorithm
dim_cut = 20
model_d = 2  # single-site basis size

# connection (conn) operator: the operator on the edge of
# the block, on the interior of the chain.  We need to be able to represent S^z
# and S^+ on that site in the current basis in order to grow the chain.
block = Block(length=1, basis_size=model_d, operator_dict={
    "H": H1,
    "conn_Sz": pauli.sz/2,
    "conn_Sp": pauli.sp,
})
# Repeatedly enlarge the system by performing a single DMRG step, using a reflection of the current block as the environment.
ret_list = []
for _ in  range(100):
    block, energy = single_dmrg_step(block, dim_cut)
    ret_list.append(energy / (2*block.length))
ret_list = np.array(ret_list)

ret_analytical = 1/4 - np.log(2)

import matplotlib.pyplot as plt
fig,ax = plt.subplots()
ax.plot(ret_list - ret_analytical)
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)

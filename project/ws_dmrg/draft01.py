# Simple DMRG tutorial.  This code integrates the following concepts:
#  - Infinite system algorithm
#  - Finite system algorithm
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

def H2(Sz1, Sp1, Sz2, Sp2):  # two-site part of H
    """Given the operators S^z and S^+ on two sites in different Hilbert spaces
    (e.g. two blocks), returns a Kronecker product representing the
    corresponding two-site term in the Hamiltonian that joins the two sites.
    """
    J = Jz = 1.
    ret = (J / 2) * (scipy.sparse.kron(Sp1, Sp2.T.conj()) + scipy.sparse.kron(Sp1.T.conj(), Sp2)) + Jz * scipy.sparse.kron(Sz1, Sz2)
    return ret


def enlarge_block(block):
    """enlarges the provided Block by a single site"""
    # Create the new operators for the enlarged block.  Our basis becomes a
    # Kronecker product of the Block basis and the single-site basis
    eye0 = scipy.sparse.identity(block.basis_size)
    tmp0 = scipy.sparse.kron(block.operator_dict["H"], scipy.sparse.identity(model_d))
    tmp1 = scipy.sparse.kron(eye0, H1)
    tmp2 = H2(block.operator_dict["conn_Sz"], block.operator_dict["conn_Sp"], pauli.sz/2, pauli.sp)
    tmp3 = {
        "H": tmp0 + tmp1 + tmp2,
        "conn_Sz": scipy.sparse.kron(eye0, pauli.sz/2),
        "conn_Sp": scipy.sparse.kron(eye0, pauli.sp),
    }
    ret = Block(length=(block.length+1), basis_size=(block.basis_size * model_d), operator_dict=tmp3)
    return ret

def single_dmrg_step(sys, env, m):
    """Performs a single DMRG step using `sys` as the system and `env` as the
    environment, keeping a maximum of `m` states in the new basis.
    """
    # Enlarge each block by a single site.
    sys_enl = enlarge_block(sys)
    if sys is env:  # no need to recalculate a second time
        env_enl = sys_enl
    else:
        env_enl = enlarge_block(env)

    # Construct the full superblock Hamiltonian.
    m_sys_enl = sys_enl.basis_size
    m_env_enl = env_enl.basis_size
    sys_enl_op = sys_enl.operator_dict
    env_enl_op = env_enl.operator_dict
    tmp0 = scipy.sparse.kron(sys_enl_op["H"], scipy.sparse.identity(m_env_enl))
    tmp1 = scipy.sparse.kron(scipy.sparse.identity(m_sys_enl), env_enl_op["H"])
    tmp2 = H2(sys_enl_op["conn_Sz"], sys_enl_op["conn_Sp"], env_enl_op["conn_Sz"], env_enl_op["conn_Sp"])
    (energy,), psi0 = scipy.sparse.linalg.eigsh(tmp0+tmp1+tmp2, k=1, which="SA")

    # Construct the reduced density matrix of the system by tracing out the environment
    tmp0 = psi0.reshape([sys_enl.basis_size, -1])
    rho = tmp0 @ tmp0.T.conj()
    EVL, EVC = np.linalg.eigh(rho)
    # truncation_error = 1 - EVL[(-m):].sum()
    EVCi = EVC[:,(-m):]

    # Rotate and truncate each operator.
    tmp0 = {k:(EVCi.T.conj() @ v @ EVCi) for k,v in sys_enl.operator_dict.items()}
    ret = Block(length=sys_enl.length, basis_size=EVCi.shape[1], operator_dict=tmp0)
    return ret, energy

def graphic(sys_block, env_block, sys_label="l"):
    """Returns a graphical representation of the DMRG step we are about to
    perform, using '=' to represent the system sites, '-' to represent the
    environment sites, and '**' to represent the two intermediate sites.
    """
    assert sys_label in ("l", "r")
    graphic = ("=" * sys_block.length) + "**" + ("-" * env_block.length)
    if sys_label == "r":
        # The system should be on the right and the environment should be on
        # the left, so reverse the graphic.
        graphic = graphic[::-1]
    return graphic

pauli = _make_pauli()

# Model-specific code for the Heisenberg XXZ chain
model_d = 2  # single-site basis size

H1 = np.array([[0, 0], [0, 0]], dtype=np.float64)  # single-site portion of H is zero

L = 20
m_warmup = 10
m_sweep_list = [10, 20, 30, 40, 40]

assert L % 2 == 0

block_disk = {}

# Use the infinite system algorithm to build up to desired size.  Each time
# we construct a block, we save it for future reference as both a left
# ("l") and right ("r") block, as the infinite system algorithm assumes the
# environment is a mirror image of the system.

# connection (conn) operator: the operator on the edge of
# the block, on the interior of the chain.  We need to be able to represent S^z
# and S^+ on that site in the current basis in order to grow the chain.
block = Block(length=1, basis_size=model_d, operator_dict={
    "H": H1,
    "conn_Sz": pauli.sz/2,
    "conn_Sp": pauli.sp,
})
block_disk["l", block.length] = block
block_disk["r", block.length] = block
while 2 * block.length < L:
    # Perform a single DMRG step and save the new Block to "disk"
    # print(graphic(block, block))
    block, energy = single_dmrg_step(block, block, m=m_warmup)
    print("E/L =", energy / (block.length * 2))
    block_disk["l", block.length] = block
    block_disk["r", block.length] = block

# Now that the system is built up to its full size, we perform sweeps using
# the finite system algorithm.  At first the left block will act as the
# system, growing at the expense of the right block (the environment), but
# once we come to the end of the chain these roles will be reversed.
sys_label, env_label = "l", "r"
sys_block = block
for m in m_sweep_list:
    while True:
        # Load the appropriate environment block from "disk"
        env_block = block_disk[env_label, L - sys_block.length - 2]
        if env_block.length == 1:
            # We've come to the end of the chain, so we reverse course.
            sys_block, env_block = env_block, sys_block
            sys_label, env_label = env_label, sys_label

        # Perform a single DMRG step.
        # print(graphic(sys_block, env_block, sys_label))
        sys_block, energy = single_dmrg_step(sys_block, env_block, m=m)

        print("E/L =", energy / L)

        # Save the block from this step to disk.
        block_disk[sys_label, sys_block.length] = sys_block

        # Check whether we just completed a full sweep.
        if sys_label == "l" and 2 * sys_block.length == L:
            break  # escape from the "while True" loop

ret0 = energy / L
assert abs(ret0+0.4341236667) < 1e-6

'''
E/L = -0.43412366670031977
E/L = -0.43412366670032
E/L = -0.434123666700319
E/L = -0.4341236667003202
E/L = -0.4341236667003784
E/L = -0.4341236667015228
E/L = -0.43412366670287145
E/L = -0.4341236667040528
E/L = -0.4341236667042415
'''

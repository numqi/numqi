import numpy as np

import numqi.gate

# see numqi.group.spf2

def apply_clifford_on_pauli(pauli_bit, cli_r, cli_mat):
    for x in [pauli_bit, cli_r, cli_mat]:
        assert x.dtype.type==np.uint8
    assert pauli_bit.max()<=1
    N0 = cli_r.shape[0]//2
    XZin = pauli_bit[2:]
    XZout = (cli_mat @ XZin)%2
    delta = pauli_bit[1] + np.dot((cli_mat[:N0]*XZin).reshape(-1), cli_mat[N0:].reshape(-1))
    bit1 = delta%2
    tmp0 = cli_mat * XZin
    tmp_jk = tmp0[N0:].T @ tmp0[:N0]
    tmp1 = ((delta%4)//2).astype(np.uint8)
    bit0 = (pauli_bit[0] + np.dot(XZin, cli_r) + np.triu(tmp_jk, 1).sum() + tmp1) % 2
    ret = np.concatenate([np.array([bit0,bit1], dtype=np.uint8), XZout], axis=0)
    return ret


def pauli_F2_to_array(bit_np):
    assert (bit_np.dtype==np.uint8) and (bit_np.ndim==1) and (bit_np.shape[0]%2==0) and (bit_np.shape[0]>=4)
    num_qubit = (bit_np.shape[0]-2)//2
    phase = (1j)**(bit_np[0]*2 + bit_np[1])
    int_to_map = {0:numqi.gate.I, 1:numqi.gate.Z, 2:numqi.gate.X, 3:-1j*numqi.gate.Y}
    tmp0 = (bit_np[2:(2+num_qubit)]*2 + bit_np[(2+num_qubit):]).tolist()
    np0 = int_to_map[tmp0[0]]
    for x in tmp0[1:]:
        np0 = np.kron(np0, int_to_map[x])
    ret = phase*np0
    return ret


def pauli_array_to_F2(np0):
    assert (np0.ndim==2) and (np0.shape[0]==np0.shape[1]) and (np0.shape[0]>=2)
    N0 = int(np.log2(np0.shape[0]))
    assert 2**N0 == np0.shape[0]
    IXYZ = [numqi.gate.I,numqi.gate.X,-1j*numqi.gate.Y,numqi.gate.Z]
    bitXZ = np.zeros(2*N0, dtype=np.uint8)
    for ind0 in range(N0):
        if np0.shape[0]==2:
            np1 = np0.reshape(-1)
        else:
            tmp0 = np0.reshape(2,2**(N0-ind0-1),2,2**(N0-ind0-1))
            tmp1 = np.einsum(tmp0, [0,1,2,3], tmp0.conj(), [4,1,5,3], [0,2,4,5], optimize=True)
            EVL,EVC = np.linalg.eigh(tmp1.reshape(4,4))
            assert np.abs(EVL - np.array([0,0,0,4])).max() < 1e-7
            np1 = EVC[:,3] * np.sqrt(2)
            # tmp2.reshape(2,2) #should be one of the I,X,Y,Z (ignore phase factor)
        for ind1,pauli in enumerate(IXYZ):
            if abs(abs(np.vdot(pauli.reshape(-1), np1))-2) < 1e-7:
                bitXZ[ind0] = 1 if (ind1 in (1,2)) else 0
                bitXZ[ind0+N0] = 1 if (ind1 in (2,3)) else 0
                tmp0 = np0.reshape(2,2**(N0-ind0-1),2,2**(N0-ind0-1))
                np0 = np.einsum(tmp0, [0,1,2,3], pauli, [0,2], [1,3], optimize=True)/2
                break
        else: #no break path
            assert False, 'not a Pauli operator'
    assert (np0.shape==(1,1)) and (abs(abs(np0.item())-1)<1e-7)
    tmp0 = round(np.angle(np0.item())*2/np.pi) % 4
    ret = np.array([tmp0>>1, tmp0&1] + list(bitXZ), dtype=np.uint8)
    return ret


def clifford_array_to_F2(np0):
    assert (np0.ndim==2) and (np0.shape[0]==np0.shape[1]) and np0.shape[0]>=2
    N0 = int(np.log2(np0.shape[0]))
    assert 2**N0 == np0.shape[0]
    assert np.abs(np0 @ np0.T.conj() - np.eye(np0.shape[0])).max() < 1e-12

    cli_r = np.zeros(2*N0, dtype=np.uint8)
    cli_mat = np.zeros((2*N0,2*N0), dtype=np.uint8)
    for ind0 in range(N0):
        tmp0 = np0.reshape(2**N0, 2**ind0, 2, 2**(N0-ind0-1))

        tmp1 = np.einsum(tmp0, [0,1,2,3], numqi.gate.X, [2,4], tmp0.conj(), [5,1,4,3], [0,5], optimize=True)
        Xbit = pauli_array_to_F2(tmp1)
        cli_mat[:,ind0] = Xbit[2:]
        cli_r[ind0] = (Xbit[0] + (np.dot(Xbit[2:(N0+2)], Xbit[(N0+2):]) % 4)//2) % 2

        tmp1 = np.einsum(tmp0, [0,1,2,3], numqi.gate.Z, [2,4], tmp0.conj(), [5,1,4,3], [0,5], optimize=True)
        Zbit = pauli_array_to_F2(tmp1)
        cli_mat[:,ind0+N0] = Zbit[2:]
        cli_r[ind0+N0] = (Zbit[0] + (np.dot(Zbit[2:(N0+2)], Zbit[(N0+2):]) % 4)//2) % 2
    return cli_r,cli_mat

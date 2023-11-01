import numpy as np

import numqi

def restore_ABC_from_2rdm(rhoAB, rhoBC, INDEX=0, zero_eps=1e-10):
    # iUDP, not unique when degenrate
    # https://doi.org/10.1103/PhysRevA.70.010302
    # Three-party pure quantum states are determined by two two-party reduced states
    assert (rhoAB.ndim==4) and (rhoBC.ndim==4) and (rhoAB.shape[1]==rhoBC.shape[0])
    dimA,dimB,_,_ = rhoAB.shape
    dimC = rhoBC.shape[1]
    assert (0<=INDEX) and (INDEX<dimC)
    assert np.abs(rhoAB-rhoAB.transpose(2,3,0,1).conj()).max() < zero_eps
    assert np.abs(rhoBC-rhoBC.transpose(2,3,0,1).conj()).max() < zero_eps
    rhoA = np.trace(rhoAB, axis1=1, axis2=3)
    rhoB = np.trace(rhoAB, axis1=0, axis2=2)
    rhoC = np.trace(rhoBC, axis1=0, axis2=2)
    assert np.abs(rhoB-np.trace(rhoBC, axis1=1, axis2=3)).max() < zero_eps

    EVL_A,EVC_A = np.linalg.eigh(rhoA)
    EVL_B,EVC_B = np.linalg.eigh(rhoB)
    EVL_C,EVC_C = np.linalg.eigh(rhoC)

    EVL_BC,EVC_BC = np.linalg.eigh(rhoBC.reshape(dimB*dimC,-1))
    EVC_BC = EVC_BC[:,(-dimA):]
    coeff0 = np.einsum(EVC_BC.reshape(dimB,dimC,-1), [0,1,2], EVC_B.conj(), [0,3], EVC_C[:,INDEX].conj(), [1], [2,3], optimize=True)

    EVL_AB,EVC_AB = np.linalg.eigh(rhoAB.reshape(dimA*dimB,-1))
    tmp0 = EVC_AB[:,(-dimC):][:,INDEX].reshape(dimA,dimB)
    coeff1 = np.einsum(tmp0, [0,1], EVC_A.conj(), [0,3], EVC_B.conj(), [1,4], [3,4], optimize=True)
    angle = np.angle(np.einsum(coeff0.conj(), [0,1], coeff1, [0,1], [0], optimize=True))

    tmp0 = np.sqrt(np.maximum(EVL_A,0)) * np.exp(1j*angle)
    ret = ((EVC_A * tmp0) @ EVC_BC.T).reshape(dimA,dimB,dimC)
    return ret


def test_restore_ABC_from_2rdm():
    dimA = 3
    dimB = 4
    dimC = 5

    psiABC = numqi.random.rand_haar_state(dimA*dimB*dimC)

    tmp0 =  psiABC.reshape(dimA,dimB,dimC)
    rhoAB = np.einsum(tmp0, [0,1,2], tmp0.conj(), [3,4,2], [0,1,3,4], optimize=True)
    rhoBC = np.einsum(tmp0, [0,1,2], tmp0.conj(), [0,3,4], [1,2,3,4], optimize=True)

    for index in range(dimC):
        z0 = restore_ABC_from_2rdm(rhoAB, rhoBC, index)
        fidelity = abs(np.vdot(psiABC, z0.reshape(-1)).item())
        assert abs(fidelity-1) < 1e-10

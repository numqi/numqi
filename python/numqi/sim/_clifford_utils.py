import numpy as np

import numqi.gate

def pauli_F2_to_pauli_op(pauli_F2):
    assert (pauli_F2.dtype.type==np.uint8)
    assert (pauli_F2.ndim==1) and (pauli_F2.shape[0]%2==0) and (pauli_F2.shape[0]>=2)
    num_qubit = (pauli_F2.shape[0]-2)//2
    tmp0 = {(0,0):numqi.gate.I, (1,0):numqi.gate.X, (0,1):numqi.gate.Z, (1,1):numqi.gate.Y}
    bitX = pauli_F2[2:(2+num_qubit)]
    bitZ = pauli_F2[(2+num_qubit):]
    op = [(tmp0[(int(x),int(y))], i) for i,(x,y) in enumerate(zip(bitX,bitZ))]
    tmp1 = (2*pauli_F2[0] + pauli_F2[1] + 3*np.dot(bitX,bitZ))%4 #XZ=-iY
    coeff = 1j**tmp1
    ret = [(coeff,op)]
    return ret

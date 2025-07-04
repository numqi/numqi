import numpy as np
import galois
import numqi

def test_get_logical_from_stab():
    stab_str_list = [
        'XIIIZII IXIIZII IIXIIZI IIIXIIZ IIZZIYY ZZZIXXZ', #bare 7-qubit code
        'ZZIZIIZ ZIZZIZI IYYYYII IIIZZZZ YYIIYYI YIYIYIY', #steane
        'XXXXXXXX ZZZZZZZZ IXIXYZYZ IXZYIXZY IYXZXZIY', #((8,8,3))
    ]
    for stab_str in stab_str_list:
        stab = galois.GF2(numqi.qec.str_to_gf4(stab_str.split(' ')))
        logicalX, logicalZ = numqi.qec.get_logical_from_stabilizer(stab)
        assert np.all(numqi.qec.matmul_gf4(logicalX, logicalZ.T)==np.eye(len(logicalX), dtype=np.uint8))
        assert np.all(numqi.qec.matmul_gf4(logicalX, stab.T)==0)
        assert np.all(numqi.qec.matmul_gf4(logicalZ, stab.T)==0)

import numpy as np
import numqi


def test_load_upb_basic():
    test_list = [('tiles',None)]
    test_list += [('quadres',x) for x in (3,7,9,15,19)]
    test_list += [('GenTiles1',x) for x in (4,6,8,10)]

    for kind,args in test_list:
        (upbA,upbB),rho = numqi.entangle.load_upb(kind, args=args, return_bes=True)
        dimA = upbA.shape[1]
        dimB = upbB.shape[1]
        assert numqi.entangle.is_ppt(rho, [dimA,dimB])

        matrix_subspace = np.einsum(upbA, [0,1], upbB, [0,2], [0,1,2], optimize=True)
        EVC = matrix_subspace.reshape(-1, dimA*dimB)
        assert np.abs(EVC @ EVC.T.conj() - np.eye(EVC.shape[0])).max() < 1e-10
        assert np.abs(rho @ EVC.T.conj() @ EVC).max() < 1e-10


def hf_check_upb_hierarchical(kind, args, klist):
    upb,rho = numqi.entangle.load_upb(kind, args=args, return_bes=True, ignore_warning=True)
    assert len(upb)==2
    dimA,dimB = [x.shape[1] for x in upb]
    EVL,EVC = np.linalg.eigh(rho)
    EVC = EVC[:,EVL>1e-5] #remove zero eigenvalue
    matrix_subspace = EVC.reshape(dimA,dimB,EVC.shape[1]).transpose(2,0,1)
    ret = [numqi.matrix_space.has_rank_hierarchical_method(matrix_subspace, rank=2, hierarchy_k=k) for k in klist]
    return ret

def test_matrix_space_hierarchical():
    para_list = [
        ('pyramid', None),
        ('tiles', None),
        ('feng4x4', None),
        ('min4x4', None),
        ('quadres', 3),
        ('gentiles1', 4),
        ('gentiles1', 6),
        ('gentiles1', 8), #2 seconds
        # ('gentiles1', 10), #21 seconds
    ]
    for kind,args in para_list:
        tmp0 = hf_check_upb_hierarchical(kind, args, [1,2])
        assert tmp0[0] == False
        assert tmp0[1] == True

    # # quadres(7) hierarical(k=2) 57 seconds
    # tmp0 = hf_check_upb_hierarchical('quadres', 7, [1,2]) #False False
    # assert tmp0[0] == False
    # assert tmp0[1] == False
    # # quadres(7) hierarical(k=3) need at least 100GB memory

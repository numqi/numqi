import numpy as np
import time

import numqi

def benchmark_is_ABC_completely_entangled_subspace():
    case_list = [(2,2,2,2), (2,2,3,2), (2,2,4,2), (2,2,5,2), (2,2,6,2), (2,2,7,2),
                (2,2,8,2), (2,2,9,2),]# (2,3,3,3), (2,3,4,3), (2,3,5,3)]
    info_list = []
    for dimA,dimB,dimC,kmax in case_list:
        np_list = numqi.matrix_space.get_completed_entangled_subspace((dimA, dimB, dimC), kind='quant-ph/0409032')[0]
        for k in [kmax-1,kmax]:
            t0 = time.time()
            ret = numqi.matrix_space.is_ABC_completely_entangled_subspace(np_list, hierarchy_k=k)
            info_list.append((dimA,dimB,dimC,k, ret, time.time()-t0))
        print(f'[{dimA}x{dimB}x{dimC}] {info_list[-2][-2]}@(k={kmax-1}) {info_list[-1][-2]}@(k={kmax}) time(k={kmax})={info_list[-1][-1]:.2f}s')
    # mac-studio 20230826
    # [2x2x2] False@(k=1) True@(k=2) time(k=2)=0.01s
    # [2x2x3] False@(k=1) True@(k=2) time(k=2)=0.22s
    # [2x2x4] False@(k=1) True@(k=2) time(k=2)=0.52s
    # [2x2x5] False@(k=1) True@(k=2) time(k=2)=0.55s
    # [2x2x6] False@(k=1) True@(k=2) time(k=2)=0.69s
    # [2x2x7] False@(k=1) True@(k=2) time(k=2)=0.96s
    # [2x2x8] False@(k=1) True@(k=2) time(k=2)=1.40s
    # [2x2x9] False@(k=1) True@(k=2) time(k=2)=2.18s
    # [2x3x3] False@(k=2) True@(k=3) time(k=3)=1.30s
    # [2x3x4] False@(k=2) True@(k=3) time(k=3)=8.59s
    # [2x3x5] False@(k=2) True@(k=3) time(k=3)=328.46s

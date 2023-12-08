import time
import numpy as np
import scipy.linalg
# https://stackoverflow.com/a/71234790/7290857

np_rng = np.random.default_rng()
hf_randc = lambda *x: np_rng.normal(size=x) + 1j*np_rng.normal(size=x)

para_list = [(x,y) for x in [1,8,32] for y in [32, 64, 128]]
time_dict = dict()
for batch_size,dim in para_list:
    print(batch_size,dim)
    tmp0 = np_rng.uniform(0, 1, size=(batch_size,dim))
    tmp1 = hf_randc(batch_size,dim,dim)
    EVC = np.linalg.eigh(tmp1 + tmp1.transpose(0,2,1).conj())[1]
    np0 = (EVC * tmp0.reshape(-1,1,dim)) @ EVC.transpose(0,2,1).conj()

    # cholesky
    tmp0 = time.time()
    _ = np.linalg.cholesky(np0)
    t0 = time.time() - tmp0

    # eigen
    tmp0 = time.time()
    EVL,EVC = np.linalg.eigh(np0)
    ret_ = (EVC*np.sqrt(EVL).reshape(-1,1,EVL.shape[1])) @ EVC.transpose(0,2,1).conj()
    t1 = time.time() - tmp0

    # sqrtm
    tmp0 = time.time()
    ret0 = np.stack([scipy.linalg.sqrtm(np0[x]) for x in range(len(np0))])
    t2 = time.time() - tmp0
    time_dict[(batch_size,dim)] = (t0,t1,t2)

for k0,k1 in para_list:
    v0,v1,v2 = time_dict[(k0,k1)]
    print(f'| {k0} | {k1} | {v0:.3g} | {v1:.3g} | {v2:.3g} |')

# mac-studio 20231208
# | 1 | 32 | 9.39e-05 | 0.000687 | 0.00137 |
# | 1 | 64 | 0.000123 | 0.00157 | 0.00366 |
# | 1 | 128 | 0.000262 | 0.00487 | 0.485 |
# | 8 | 32 | 7.82e-05 | 0.00288 | 0.00301 |
# | 8 | 64 | 0.000421 | 0.00884 | 0.0165 |
# | 8 | 128 | 0.00174 | 0.0372 | 4.17 |
# | 32 | 32 | 0.000314 | 0.0109 | 0.0117 |
# | 32 | 64 | 0.00266 | 0.0785 | 0.0726 |
# | 32 | 128 | 0.00787 | 0.179 | 19.8 |

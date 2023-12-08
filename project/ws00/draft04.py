import numpy as np
import torch
import scipy.linalg

import numqi

# https://doi.org/10.1017/9781316882177
# Algorithmic Aspects of Machine Learning page28/theorem-3.1.3

np_rng = np.random.default_rng()

dimA,dimB,dimC = 3,5,7
rank = 4
zero_eps = 1e-10

z0 = [[np_rng.normal(size=x)+0j*np_rng.normal(size=x) for x in [dimA,dimB,dimC]] for _ in range(rank)]

z1 = sum(x.reshape(-1,1,1)*y.reshape(-1,1)*z for x,y,z in z0)

tmp0 = np_rng.normal(size=(2,dimC))
veca,vecb = tmp0 / np.linalg.norm(tmp0, axis=1, keepdims=True)

tmp0 = z1 @ veca
tmp1 = z1 @ vecb
EVL0,EVC0 = np.linalg.eig(tmp0 @ tmp1.T.conj())
index = np.argsort(-np.abs(EVL0))
EVL0,EVC0 = EVL0[index],EVC0[:,index]
EVL1,EVC1 = np.linalg.eig(tmp1 @ tmp0.T.conj())
index = np.argsort(-np.abs(EVL1))
EVL1,EVC1 = EVL1[index],EVC1[:,index]

# EVL0,EVC0 = np.linalg.eig(tmp0 @ scipy.linalg.pinv(tmp1))
# EVL1,EVC1 = np.linalg.eig(tmp1 @ scipy.linalg.pinv(tmp0))
# print(np.abs(EVL0.reshape(-1,1)*EVL1))

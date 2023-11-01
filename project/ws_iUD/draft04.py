import numpy as np

import numqi

np_rng = np.random.default_rng()
hf_randc = lambda *x: np_rng.normal(size=x) + 1j*np_rng.normal(size=x)

dicke_basis_4q = numqi.dicke.get_dicke_basis(4, 2)
bell_basis = numqi.dicke.get_dicke_basis(2, 2)

tmp0 = np_rng.normal(size=(3))
state = (tmp0 / np.linalg.norm(tmp0)) @ dicke_basis_4q[[0,2,4]]


tmp0 = state.reshape(4,4)
EVL,EVC = np.linalg.eigh(tmp0)

tmp0 = bell_basis @ state.reshape(4,4) @ bell_basis.T


tmp0 = hf_randc(3)
state1 = (tmp0 / np.linalg.norm(tmp0)) @ dicke_basis_4q[[0,2,4]]
z0 = (bell_basis @ state1.reshape(4,4) @ bell_basis.T)[::2,::2]

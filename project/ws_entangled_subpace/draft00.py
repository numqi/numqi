import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import numqi

dimB = 4
num_party = 3

dim_list = [2]+[dimB]*(num_party-1)
bipartition_list = [tuple(range(x)) for x in range(1,num_party)]

theta_list = np.linspace(0, np.pi, 20)

ret_analytical = numqi.matrix_space.get_GM_Maciej2019(dimB, theta_list)

model_list = [numqi.matrix_space.DetectCanonicalPolyadicRankModel(dim_list, rank=1, bipartition=x) for x in bipartition_list]
model = numqi.matrix_space.DetectCanonicalPolyadicRankModel(dim_list, rank=1)
kwargs = dict(theta0='uniform', num_repeat=3, tol=1e-12, print_every_round=0)
ret_gd = []
ret_ppt = []
for theta_i in tqdm(theta_list):
    matrix_subspace = numqi.matrix_space.get_GES_Maciej2019(dimB, num_party=num_party, theta=theta_i)
    for model in model_list:
        model.set_target(matrix_subspace)
        ret_gd.append(numqi.optimize.minimize(model, **kwargs).fun)
    ret_ppt.append(numqi.matrix_space.get_generalized_geometric_measure_ppt(matrix_subspace, dim_list, bipartition_list))
ret_gd_bipartition = np.array(ret_gd).reshape(len(theta_list), -1)
ret_gd = ret_gd_bipartition.min(axis=1)
ret_ppt = np.array(ret_ppt)

fig,ax = plt.subplots()
ax.plot(theta_list, ret_analytical, 'x', label='analytical')
ax.plot(theta_list, ret_ppt, 'o', label='ppt', markersize=4)
ax.plot(theta_list, ret_gd, label='numerical')
ax.legend()
ax.set_title(f'dimB={dimB}, num_party={num_party}')

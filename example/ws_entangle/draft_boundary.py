import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
tableau = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']

import numqi

np_rng = np.random.default_rng()

def demo_ppt_cha_gap():
    # num_sample=100, dim=3, 9min
    # num_sample=100, dim=4, 1hour
    num_sample = 100
    dim_list = [3,4]
    seed_list = np_rng.integers(0, 2**31-1, size=len(dim_list)).tolist() #for reproducibility

    data = []
    for dim,seed_i in zip(dim_list,seed_list):
        np_rng_i = np.random.default_rng(seed_i)
        model = numqi.entangle.CHABoundaryBagging((dim,dim))
        # model = numqi.entangle.AutodiffCHAREE((dim, dim), distance_kind='ree')
        for _ in tqdm(range(num_sample)):
            dm0 = numqi.random.rand_density_matrix(dim*dim, seed=np_rng_i)
            beta_ppt = numqi.entangle.get_ppt_boundary(dm0, (dim,dim))[1]
            beta_cha = model.solve(dm0, use_tqdm=False)
            # beta_cha = model.get_boundary(dm0, xtol=1e-4, converge_tol=1e-10, threshold=1e-7, num_repeat=3, use_tqdm=False)
            data.append((beta_ppt,beta_cha))
    data = np.array(data).reshape(len(dim_list), num_sample, 2)

    fig,tmp0 = plt.subplots(2, 2)
    ax0,ax1,ax2,ax3 = tmp0[0][0], tmp0[0][1], tmp0[1][0], tmp0[1][1]
    ax0.hist(data[0,:,0], bins=20)
    ax0.set_title('PPT, dim=3')
    ax1.hist(data[1,:,0], bins=20)
    ax1.set_title('PPT, dim=4')
    ax2.hist(data[0,:,0]-data[0,:,1], bins=20)
    ax2.set_title('PPT-CHA, dim=3')
    ax3.hist(data[1,:,0]-data[1,:,1], bins=20)
    ax3.set_title('PPT-CHA, dim=4')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)

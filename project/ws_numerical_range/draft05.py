import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import numqi

np_rng = np.random.default_rng()

dimA = 2
dimB = 2
kext = 3
kext_list = [3, 4, 8, 16]
num_sample = 5000
num_bins = 30

count_np = []
for kext in kext_list:
    matGext = numqi.maximum_entropy.get_ABk_gellmann_preimage_op(dimA, dimB, kext, kind='boson')
    for _ in tqdm(range(num_sample)):
        tmp0 = np_rng.uniform(-1, 1, size=matGext.shape[0])
        tmp0 /= np.linalg.norm(tmp0)
        tmp1 = (tmp0 @ matGext.reshape(tmp0.shape[0],-1)).reshape(matGext.shape[1:])
        EVL = np.linalg.eigvalsh(tmp1)
        count_np.append(EVL[1]-EVL[0])
count_np = np.array(count_np).reshape(len(kext_list), -1)
count_np_log = np.log10(count_np)
bin_list = np.linspace(count_np_log.min()*0.9, count_np_log.max()*1.1, num_bins)
distribution_list = [np.histogram(x, bin_list, density=True)[0] for x in count_np_log]

fig,ax = plt.subplots()
bin_center = (bin_list[1:]+bin_list[:-1])/2
for ind0 in range(len(kext_list)):
    ax.fill_between(bin_center, distribution_list[ind0], label=f'k={kext_list[ind0]}', alpha=0.3)
ax.legend()
ax.set_title(f'dA={dimA}, dB={dimB}')
fig.tight_layout()
fig.savefig('tbd02.png', dpi=200)

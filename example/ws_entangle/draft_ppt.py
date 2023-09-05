import numpy as np
import matplotlib.pyplot as plt

import numqi


def demo_generalized_ppt():
    dm_tiles = numqi.entangle.load_upb('tiles', return_bes=True)[1]
    dm_norm = numqi.gellmann.dm_to_gellmann_norm(dm_tiles)
    beta_dm = numqi.entangle.get_density_matrix_boundary(dm_tiles)[1]
    beta_list = np.linspace(0, beta_dm, 100)

    dm_target_list = [numqi.entangle.hf_interpolate_dm(dm_tiles,beta=x,dm_norm=dm_norm) for x in beta_list]

    z0 = []
    for dm_target_i in dm_target_list:
        info = numqi.entangle.is_generalized_ppt(dm_target_i, dim=(3,3), return_info=True)[1]
        z0.append(max(x[2] for x in info))
    beta_gppt = numqi.entangle.get_generalized_ppt_boundary(dm_tiles, dim=(3,3))
    fig,ax = plt.subplots()
    ax.plot(beta_list, z0)
    ax.axvline(beta_gppt, color='k', linestyle=':')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel('nuclear norm')
    ax.set_title('tiles UPB/BES')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)

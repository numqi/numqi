import numpy as np
import matplotlib.pyplot as plt


def demo_dirichlet_distribution():
    # https://en.wikipedia.org/wiki/Dirichlet_distribution
    np_rng = np.random.default_rng()
    alpha = np.array([0.3, 0.3, 0.3])
    N0 = 5000
    np0 = np_rng.dirichlet(alpha, size=N0)
    assert np.abs(np0.sum(axis=1)-1).max()<1e-7

    tmp0 = [0, np.pi*2/3, np.pi*4/3]
    tmp1 = np0 @ np.array([[np.cos(x),np.sin(x)] for x in tmp0])
    fig,ax = plt.subplots()
    ax.scatter(tmp1[:,0], tmp1[:,1], s=3, alpha=0.3)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    fig.savefig('tbd00.png', dpi=200)

import numpy as np
import scipy.linalg
import seaborn as sns
import matplotlib.pyplot as plt

import pyqet

np_rng = np.random.default_rng()

# Pyramid(3x3) UPB
pbA,pbB = pyqet.entangle.load_upb('pyramid')
tmp0 = np.einsum(pbA, [0,1], pbB, [0,2], [0,1,2], optimize=True)
basis,basis_orth,space_char = pyqet.matrix_space.get_matrix_orthogonal_basis(tmp0, field='complex')

model = pyqet.matrix_space.DetectRankOneModel(basis_orth)
theta_optim = pyqet.optimize.minimize(model, 'normal', num_repeat=10, tol=1e-13)
theta_optim.fun #0.177


# Pyr34(3x4) UCPB, not SUCPB
pbA,_ = pyqet.entangle.load_upb('pyramid')
tmp0 = np.sqrt(np.cos(np.pi/5))
tmp1 = np.sqrt(np.cos(np.pi*2/5))
tmp2 = [(tmp0*np.cos(2*x*np.pi/5), tmp0*np.sin(2*x*np.pi/5), tmp1*np.cos(4*x*np.pi/5), tmp1*np.sin(4*x*np.pi/5)) for x in range(5)]
pbB = np.array(tmp2)*np.sqrt(2/np.sqrt(5))
assert np.abs(np.linalg.norm(pbB,axis=1)-1).max() < 1e-10
assert np.abs(np.sum(pbB * np.roll(pbB, -1, axis=0), axis=1)).max() < 1e-10
tmp0 = (pbA @ pbA.T) * (pbB @ pbB.T)
assert np.abs(tmp0 - np.eye(tmp0.shape[0])).max() < 1e-10 #orthogonal product basis

# eq-2.7 https://arxiv.org/abs/quant-ph/9908070v3
tmp0 = [(-np.sin(2*x*np.pi/5), np.cos(2*x*np.pi/5), -np.sin(4*x*np.pi/5), np.cos(4*x*np.pi/5), 0) for x in range(5)]
vec_u_list = np.sqrt(1/2) * np.array(tmp0)
assert np.abs(np.linalg.norm(vec_u_list, axis=1)-1).max() < 1e-10
vec_x_list = vec_u_list + np.array([0,0,0,0,1/2])
pbB_ext = np.pad(pbB, [(0,0),(0,1)])
hf0 = lambda x,y: np.linalg.eigh(np.eye(3) - x.reshape(-1,1)*x - y.reshape(-1,1)*y)[1][:,2]
tmp0 = [(1,4), (0,2), (1,3), (2,4), (0,3)]
z0 = [(hf0(pbA[x],pbA[y]), vec_x_list[i]) for i,(x,y) in enumerate(tmp0)]
hf1 = lambda v0,v1,u0: (np.dot(v1,u0)*v0 - np.dot(v0,u0)*v1)
tmp0 = [(1,4), (0,2), (1,3), (2,4), (0,3)]
z1 = [(pbA[i], hf1(vec_x_list[x], vec_x_list[y], pbB_ext[i])) for i,(x,y) in enumerate(tmp0)]
tmp0 = np.stack([x[0] for x in z0] + [x[0] for x in z1], axis=0)
pb_orth_A = tmp0 / np.linalg.norm(tmp0, axis=1, keepdims=True)
tmp1 = np.stack([x[1] for x in z0] + [x[1] for x in z1], axis=0)
pb_orth_B = tmp1 / np.linalg.norm(tmp1, axis=1, keepdims=True)
tmp0 = np.concatenate([pbA, pb_orth_A], axis=0)
tmp1 = np.concatenate([pbB_ext, pb_orth_B], axis=0)
tmp2 = (tmp0 @ tmp0.T) * (tmp1 @ tmp1.T)
assert np.abs(tmp2 - np.eye(tmp2.shape[0])).max() < 1e-10


# graph to product basis
graph_pyramid = np.zeros((5,5), dtype=np.int32)
tmp0 = [
    [(0,2), (0,3), (1,3), (1,4), (2,4)], #partite-A
    [(0,1), (0,4), (1,2), (2,3), (3,4)],
]
tmp0 = [np.array(x) for x in tmp0]
for ind0,x in enumerate(tmp0, start=1):
    graph_pyramid[x[:,0], x[:,1]] = ind0
    graph_pyramid[x[:,1], x[:,0]] = ind0
assert np.all(graph_pyramid==graph_pyramid.T)
assert np.all(np.diag(graph_pyramid)==0)
assert np.all(np.logical_or(np.eye(graph_pyramid.shape[0],dtype=np.bool_), (graph_pyramid>0)))

graph = graph_pyramid==1



def pb_to_graph(pb_list, zero_eps=1e-7):
    tmp0 = [(np.abs(x.conj() @ x.T)<zero_eps) for x in pb_list]
    ret = np.stack(tmp0).astype(np.bool_)
    return ret

def get_genshifts_graph(k:int):
    assert k>1
    tmp0 = np.array([0] + list(range(1,k)) + list(range(1-k,0)), dtype=np.int32)
    tmp0a = np.stack([np.roll(tmp0,i) for i in range(2*k-1)])
    tmp1 = np.nonzero((tmp0a[0] + tmp0a[1:])==0)[1]
    tmp2 = np.insert(tmp1+1, 0, 0)
    tmp3 = scipy.linalg.hankel(tmp2, np.insert(tmp1, 2*k-2, 0))
    np.fill_diagonal(tmp3, 0)

    z0 = np.zeros((2*k,2*k), dtype=np.uint32)
    z0[0] = np.arange(2*k, dtype=np.uint32)
    z0[:,0] = z0[0]
    z0[1:,1:] = tmp3
    ret = np.stack([z0==x for x in range(1,2*k)]).astype(np.bool_)
    return ret


def test_upb_genshifts():
    for k in [2,3,4,5]:
        graph = get_genshifts_graph(k)
        pb_list = pyqet.entangle.load_upb('genshifts', args=2*k-1)
        for x,pb_i in zip(graph,pb_list):
            tmp0 = np.abs(pb_i.conj() @ pb_i.T)
            assert (x*tmp0).max() < 1e-10


def plot_pb_graph(graph, ax=None):
    assert (graph.ndim==3) and (graph.shape[1]==graph.shape[2])
    num_party,num_node,_ = graph.shape
    if num_party<=4:
        color_list = sns.color_palette("hls", num_party)
        linestyle_list = ['-'] * num_party
    else:
        tmp0 = sns.color_palette("hls", (num_party+1)//2)
        color_list = [tmp0[x//2] for x in range(num_party)]
        tmp0 = ['-', '--']
        linestyle_list = [tmp0[x%2] for x in range(num_party)]
    theta_list = np.pi/2 - np.linspace(0, 2*np.pi, num_node+1)[:-1]
    xy_list = np.stack([np.cos(theta_list), np.sin(theta_list)], axis=1)
    if ax is None:
        fig,ax = plt.subplots()
    else:
        fig = ax.get_figure()
    for ind0 in range(num_party):
        ind1,ind2 = np.nonzero(graph[ind0])
        tmp0 = sorted([(x,y) for x,y in zip(ind1.tolist(),ind2.tolist()) if x<y])
        for ind3,(x,y) in enumerate(tmp0):
            xdata,ydata = xy_list[[x,y]].T
            label = str(ind0) if ind3==0 else None
            num_line = graph[:,x,y].sum()
            kwargs = dict(linestyle=linestyle_list[ind0], color=color_list[ind0], label=label)
            if num_line==1:
                ax.plot(xdata, ydata, **kwargs)
            else:
                num_point = 20
                index_line = graph[:ind0,x,y].sum()
                radius = np.sin(np.linspace(0, np.pi, num_point)) * 0.03 * (1-2*(index_line%2))
                shift_theta = np.random.default_rng().uniform(-np.pi/2, np.pi/2)
                xdata = np.linspace(xdata[0], xdata[1], num_point) + radius*np.cos(shift_theta)
                ydata = np.linspace(ydata[0], ydata[1], num_point) + radius*np.sin(shift_theta)
                ax.plot(xdata, ydata, **kwargs)
    ax.plot(xy_list[:,0], xy_list[:,1], '.', color='k', markersize=10)
    tmp0 = ax.get_xlim()
    ax.set_xlim(tmp0[0], tmp0[1]+0.3) #for legend
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    return fig,ax

def demo_genshifts():
    fig,(ax0,ax1,ax2) = plt.subplots(1, 3, figsize=(12,4))
    graph = get_genshifts_graph(2)
    plot_pb_graph(graph, ax=ax0)
    ax0.set_title(r'$\mathcal{H}_2^{\otimes 3}$')
    graph = get_genshifts_graph(3)
    plot_pb_graph(graph, ax=ax1)
    ax1.set_title(r'$\mathcal{H}_2^{\otimes 5}$')
    graph = get_genshifts_graph(4)
    plot_pb_graph(graph, ax=ax2)
    ax2.set_title(r'$\mathcal{H}_2^{\otimes 7}$')
    fig.tight_layout()
    # fig.savefig('upb-genshifts-graph.png', dpi=200)


def demo_gentiles1():
    fig,(ax0,ax1) = plt.subplots(1, 2, figsize=(8,4))
    pb_list = pyqet.entangle.load_upb('gentiles1', args=4)
    graph = pb_to_graph(pb_list)
    plot_pb_graph(graph, ax=ax0)
    ax0.set_title(r'$4\otimes 4$')

    pb_list = pyqet.entangle.load_upb('gentiles1', args=6)
    graph = pb_to_graph(pb_list)
    plot_pb_graph(graph, ax=ax1)
    ax1.set_title(r'$6\otimes 6$')
    fig.tight_layout()
    # fig.savefig('upb-gentiles1-graph.png', dpi=200)


def demo_gentiles1():
    fig,(ax0,ax1,ax2) = plt.subplots(1, 3, figsize=(12,4))
    pb_list = pyqet.entangle.load_upb('gentiles2', args=(3,4))
    graph = pb_to_graph(pb_list)
    plot_pb_graph(graph, ax=ax0)
    ax0.set_title(r'$3\otimes 4$')

    pb_list = pyqet.entangle.load_upb('gentiles2', args=(3,5))
    graph = pb_to_graph(pb_list)
    plot_pb_graph(graph, ax=ax1)
    ax1.set_title(r'$3\otimes 5$')

    pb_list = pyqet.entangle.load_upb('gentiles2', args=(4,4))
    graph = pb_to_graph(pb_list)
    plot_pb_graph(graph, ax=ax2)
    ax2.set_title(r'$4\otimes 4$')
    fig.tight_layout()
    # fig.savefig('upb-gentiles2-graph.png', dpi=200)




graph = get_genshifts_graph(2)
graph = get_genshifts_graph(3)
# plot_pb_graph(graph)

pb_list = pyqet.entangle.load_upb('gentiles1', args=4)
graph = pb_to_graph(pb_list)
# plot_pb_graph(graph)

# genshift graph
N0 = 7
pb_list = pyqet.entangle.load_upb('genshifts', args=N0)
graph = pb_to_graph(pb_list)
print(graph)

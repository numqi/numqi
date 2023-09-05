import numpy as np
import cvxpy
import matplotlib.pyplot as plt

import pyqet

np_rng = np.random.default_rng()

op0 = np.diag([1,1,-1])
op1 = np.array([[1,0,1], [0,1,1], [1,1,-1]])
# op1 = np.array([[1,0,1], [0,0,1], [1,1,-1]])
# op1 = np.diag([1,0,-1])

theta_list = np.linspace(0, 2*np.pi, 200)
beta_list,op_nr_list,dual_value_list = pyqet.maximum_entropy.op_list_numerical_range_SDP([op0,op1], theta_list)

fig,ax = plt.subplots()
tmp0 = slice(None, None, 3)
tmp1 = np.angle(dual_value_list[:,0] + 1j*dual_value_list[:,1])
# pyqet.maximum_entropy.draw_line_list(ax, op_nr_list[tmp0], tmp1[tmp0], kind='norm', color='#CCCCCC', radius=0.06)
pyqet.maximum_entropy.draw_line_list(ax, op_nr_list[tmp0], tmp1[tmp0], kind='tangent', color='#CCCCCC', radius=1)
ax.plot(op_nr_list[:,0], op_nr_list[:,1])
ax.set_aspect('equal')
fig.tight_layout()
fig.savefig('tbd01.png', dpi=200)


theta_list = np.linspace(0, 2*np.pi, 200)
op0 = np.diag([1,1,-1])
op1_list = [
    np.array([[1,0,1], [0,1,1], [1,1,-1]]),
    np.array([[1,0,1], [0,0,1], [1,1,-1]]),
    np.diag([1,0,-1]),
]
op1 = np.array([[1,0,1], [0,1,1], [1,1,-1]])
# op1 = np.array([[1,0,1], [0,0,1], [1,1,-1]])
# op1 = np.diag([1,0,-1])

data_list = [pyqet.maximum_entropy.op_list_numerical_range_SDP([op0,x], theta_list) for x in op1_list]

fig,ax_list = plt.subplots(1, 3, figsize=(9,4))
for ind0 in range(3):
    ax = ax_list[ind0]
    beta_list,op_nr_list,dual_value_list = data_list[ind0]
    tmp0 = slice(None, None, 3)
    tmp1 = np.angle(dual_value_list[:,0] + 1j*dual_value_list[:,1])
    # pyqet.maximum_entropy.draw_line_list(ax, op_nr_list[tmp0], tmp1[tmp0], kind='norm', color='#CCCCCC', radius=0.06)
    pyqet.maximum_entropy.draw_line_list(ax, op_nr_list[tmp0], tmp1[tmp0], kind='tangent', color='#CCCCCC', radius=1)
    ax.plot(op_nr_list[:,0], op_nr_list[:,1])
    ax.set_aspect('equal')
fig.tight_layout()
fig.savefig('tbd01.png', dpi=200)
# fig.savefig('data/example-abc.png', dpi=200)

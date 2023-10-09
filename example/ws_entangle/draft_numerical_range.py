import numpy as np
import matplotlib.pyplot as plt

import numqi

np_rng = np.random.default_rng()


def demo_PPT_numerical_range():
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    op0 = np.kron(np.array([[1/np.sqrt(2),0],[0,0]]), sz)
    op1 = (np.kron(sy, sx) - np.kron(sx, sy)) / 2
    theta_list = np.linspace(0, 2*np.pi, 400)
    direction = np.stack([np.cos(theta_list), np.sin(theta_list)], axis=1)
    beta_ppt = numqi.entangle.get_ppt_numerical_range([op0,op1], direction, dim=(2,2))
    op_nr_ppt = beta_ppt.reshape(-1,1)*direction

    fig,ax = plt.subplots()
    ax.plot(op_nr_ppt[:,0], op_nr_ppt[:,1])
    ax.set_title('PPT numerical range')
    ax.grid()
    ax.set_xlabel('$O_0$')
    ax.set_ylabel('$O_1$')
    fig.tight_layout()
    fig.savefig('tbd00.png')


def demo_2qubit_numerical_range():
    # https://doi.org/10.1103/PhysRevA.98.012315 Fig 4
    # about 1 minute
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    op0 = np.kron(np.array([[1/np.sqrt(2),0],[0,0]]), sz)
    op1 = (np.kron(sy, sx) - np.kron(sx, sy)) / 2
    dimA = 2
    dimB = 2
    num_theta = 100
    kext_list = [8,16,32,64,128]

    theta_list = np.linspace(0, 2*np.pi, num_theta)
    direction = np.stack([np.cos(theta_list), np.sin(theta_list)], axis=1)
    beta_ppt = numqi.entangle.get_ppt_numerical_range([op0,op1], direction, (dimA,dimB))
    op_nr_ppt = beta_ppt.reshape(-1,1)*direction
    model_cha = numqi.entangle.AutodiffCHAREE(dimA, dimB)
    ret_cha = model_cha.get_numerical_range(op0, op1, num_theta=num_theta, num_repeat=3)
    ret_pureb = []
    for kext_i in kext_list:
        model = numqi.entangle.PureBosonicExt(2, 2, kext=kext_i)
        ret_pureb.append(model.get_numerical_range(op0, op1, num_theta=num_theta, num_repeat=3))
    ret_pureb = np.stack(ret_pureb)

    fig,ax = plt.subplots()
    ax.plot(op_nr_ppt[:,0], op_nr_ppt[:,1], 'x', label=f'PPT')
    ax.plot(ret_cha[:,0], ret_cha[:,1], linewidth=0.5, label=f'CHA')
    for ind0 in range(len(kext_list)):
        ax.plot(ret_pureb[ind0,:,0], ret_pureb[ind0,:,1], linewidth=0.5, label=f'PureB({kext_list[ind0]})')
    ax.legend()
    ax.set_xlabel('$O_1$')
    ax.set_ylabel('$O_2$')
    ax.set_xlabel(r'$((\sigma_z+\sigma_0)\sigma_z)/(2\sqrt{2})$')
    ax.set_ylabel(r'$(\sigma_y\sigma_x-\sigma_x\sigma_y)/2$')
    ax.set_title(f'numerical range {dimA}x{dimB}')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)


def demo_random_numerical_range():
    # about 1 minute
    hf_trace0 = lambda x: x - np.eye(x.shape[0])*np.trace(x)/x.shape[0]
    dimA = 3
    dimB = 3
    kext = 32
    op0 = hf_trace0(numqi.random.rand_hermite_matrix(dimA*dimB))
    op1 = hf_trace0(numqi.random.rand_hermite_matrix(dimA*dimB))
    num_theta = 100

    theta_list = np.linspace(0, 2*np.pi, num_theta)
    direction = np.stack([np.cos(theta_list), np.sin(theta_list)], axis=1)
    beta_ppt = numqi.entangle.get_ppt_numerical_range([op0, op1], direction, (dimA,dimB))
    op_nr_ppt = beta_ppt.reshape(-1,1)*direction
    model_cha = numqi.entangle.AutodiffCHAREE(dimA, dimB)
    ret_cha = model_cha.get_numerical_range(op0, op1, num_theta=num_theta, num_repeat=3)
    model_pureb = numqi.entangle.PureBosonicExt(dimA, dimB, kext=kext)
    ret_pureb = model_pureb.get_numerical_range(op0, op1, num_theta=num_theta, num_repeat=3)

    fig,ax = plt.subplots()
    ax.plot(op_nr_ppt[:,0], op_nr_ppt[:,1], 'x', label=f'PPT')
    ax.plot(ret_cha[:,0], ret_cha[:,1], linewidth=0.5, label=f'CHA')
    ax.plot(ret_pureb[:,0], ret_pureb[:,1], linewidth=0.5, label=f'PureB({kext})')
    ax.legend()
    ax.grid()
    ax.set_xlabel('random $O_1$')
    ax.set_ylabel('random $O_2$')
    ax.set_title(f'numerical range {dimA}x{dimB}')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)

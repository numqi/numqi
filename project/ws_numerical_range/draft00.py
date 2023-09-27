import numpy as np
import matplotlib.pyplot as plt

import numqi

np_rng = np.random.default_rng()
tableau = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
hf_trace0 = lambda x: x-np.trace(x)/x.shape[0]*np.eye(x.shape[0])

def demo_maxent_2op():
    op0 = np.diag([1,1,-1])
    op1 = np.array([[1,0,1], [0,1,1], [1,1,-1]])
    # op1 = np.array([[1,0,1], [0,0,1], [1,1,-1]])
    # op1 = np.diag([1,0,-1])

    # N0 = 3
    # tmp0 = np_rng.normal(size=(2,N0,N0)) + 1j*np_rng.normal(size=(2,N0,N0))
    # op0,op1 = [hf_trace0(x) for x in (tmp0 + tmp0.transpose(0,2,1).conj())]

    # tmp0 = numqi.matrix_space.get_matrix_numerical_range(op0+1j*op1, num_point=num_point)
    # op_nr = np.stack([tmp0.real, tmp0.imag], axis=1)

    num_point = 200
    theta_list = np.linspace(-np.pi, np.pi, num_point)
    phi_target = np.pi/6
    op_nr_list = []
    t_list = []
    vecB = np.array([np.cos(phi_target), np.sin(phi_target)])
    for theta_i in theta_list:
        vecN = np.array([np.cos(theta_i), np.sin(theta_i)])
        matH = op0*vecN[0] + op1*vecN[1]
        EVC = np.linalg.eigh(matH)[1][:,-1]
        pointA = np.array([np.vdot(EVC, x@EVC).real for x in [op0,op1]])
        op_nr_list.append(pointA)
        t_list.append(np.dot(pointA, vecN)/np.dot(vecB, vecN))
    t_list = np.array(t_list)
    op_nr_list = np.array(op_nr_list)

    fig,(ax0,ax1) = plt.subplots(1, 2, figsize=(8,4))
    ax0.plot(op_nr_list[:,0], op_nr_list[:,1], color=tableau[4])
    ind0 = slice(None, None, 5)
    numqi.matrix_space.draw_line_list(ax0, op_nr_list[ind0], theta_list[ind0], kind='norm', radius=0.5, color=tableau[2])
    tmp0 = ax0.get_xlim()[1]*0.8
    ax0.plot([0,tmp0], [0,tmp0*np.tan(phi_target)], color=tableau[0])
    ax0.set_aspect('equal')
    tmp0 = t_list.copy()
    tmp0[np.nonzero(tmp0[:-1]*tmp0[1:]<0)] = np.nan
    ax1.plot(theta_list, tmp0)
    ax1.set_xlabel(r'$\theta$')
    ax1.set_ylabel('$t$')
    ax1.set_ylim(-10, 10)
    fig.tight_layout()
    # ax0.legend()
    fig.savefig('tbd00.png', dpi=200)
    # fig.savefig('data/example-1c.png', dpi=200)


def demo_maxent_3op_loss():
    op0 = np.diag([1,1,-1])
    # op1 = np.array([[1,0,1], [0,1,1], [1,1,-1]])
    op1 = np.array([[1,0,1], [0,0,1], [1,1,-1]])
    # op1 = np.diag([1,0,-1])
    tmp0 = np_rng.normal(size=(3,3)) + 1j*np_rng.normal(size=(3,3))
    op2 = hf_trace0(tmp0+tmp0.T.conj())

    # N0 = 4
    # tmp0 = np_rng.normal(size=(3,N0,N0)) + 1j*np_rng.normal(size=(3,N0,N0))
    # op0,op1,op2 = [hf_trace0(x) for x in (tmp0 + tmp0.transpose(0,2,1).conj())]

    num_point = 100
    theta_list = np.linspace(0, np.pi, num_point)
    phi_list = np.linspace(-np.pi, np.pi, num_point)
    tmp0 = np_rng.uniform(-1,1,size=3)
    vecB = tmp0/np.linalg.norm(tmp0)

    t_list = []
    for theta_i in theta_list:
        for phi_i in phi_list:
            vecN = np.array([np.sin(theta_i)*np.cos(phi_i), np.sin(theta_i)*np.sin(phi_i), np.cos(theta_i)])
            matH = op0*vecN[0] + op1*vecN[1] + op2*vecN[2]
            EVC = np.linalg.eigh(matH)[1][:,-1]
            vecA = np.array([np.vdot(EVC, x@EVC).real for x in [op0,op1,op2]])
            t_list.append(np.dot(vecA, vecN)/np.dot(vecB, vecN))
    t_list = np.array(t_list).reshape(num_point, num_point)

    fig,ax = plt.subplots()
    tmp2 = np.clip(t_list, -100, 20)
    tmp2[tmp2<0] = np.nan
    himage = ax.imshow(tmp2, cmap=plt.get_cmap('winter'), extent=[-np.pi, np.pi, np.pi, 0], origin='lower')
    fig.colorbar(himage, shrink=0.5, aspect=5)
    fig.tight_layout()


def demo_energy_band_2op():
    op0 = np.diag([1,1,-1])
    op1 = np.array([[1,0,1], [0,1,1], [1,1,-1]])
    # op1 = np.array([[1,0,1], [0,0,1], [1,1,-1]])
    # op1 = np.diag([1,0,-1])

    theta_list = np.linspace(-np.pi, np.pi, 200)
    tmp0 = np.cos(theta_list)
    tmp1 = np.sin(theta_list)
    EVL = np.array([np.linalg.eigvalsh(x*op0+y*op1) for x,y in zip(tmp0, tmp1)])
    fig,ax = plt.subplots()
    ax.plot(theta_list, EVL)
    ax.set_xlabel('cos(theta) F1 + sin(theta) F2')
    ax.set_ylabel('eigenvalues')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def get_unitary_infidelity(np0, np1):
    assert (np0.ndim==2) and (np0.shape[0]==np0.shape[1]) and (np0.shape==np1.shape)
    tmp0 = abs(np.trace(np0 @ np1.T.conj()))
    # ret = 1 - tmp0 / np0.shape[0]
    ret = 1 - tmp0*tmp0 / np0.size
    return ret

def get_unitary_fidelity(np0, np1):
    tmp0 = abs(np.trace(np0 @ np1.T.conj()))
    ret = tmp0*tmp0 / np0.size
    return ret

PauliX = np.array([[0, 1], [1, 0]])
PauliY = np.array([[0, -1j], [1j, 0]])
PauliZ = np.array([[1, 0], [0, -1]])
Pauli0 = np.eye(2)


def demo_1qubit():
    alpha = np.pi/6
    theta_list = np.linspace(0, np.pi, 50)[1:]
    fidelity_trotter_list = []
    fidelity_improved_list = []
    for theta in theta_list:
        tmp0 = scipy.linalg.expm(-1j*alpha*PauliZ/2)
        ret_ = tmp0 @ scipy.linalg.expm(-1j*theta*PauliX/2) @ tmp0
        ret0 = scipy.linalg.expm(-1j*theta*PauliX/2 - 1j*2*alpha*PauliZ/2)
        ret1 = scipy.linalg.expm(-1j*theta*PauliX/2 - 1j*alpha*(theta/np.tan(theta/2))*PauliZ/2)
        fidelity_trotter_list.append(get_unitary_fidelity(ret_, ret0))
        fidelity_improved_list.append(get_unitary_fidelity(ret_, ret1))
    fidelity_trotter_list = np.array(fidelity_trotter_list)
    fidelity_improved_list = np.array(fidelity_improved_list)

    fig,ax = plt.subplots()
    ax.plot(theta_list, fidelity_trotter_list, label=r'$\lambda=2$')
    ax.plot(theta_list, fidelity_improved_list, label=r'$\lambda=\theta/\tan(\theta/2)$')
    ax.legend()
    ax.set_xlabel('theta')
    ax.set_title(f'alpha={alpha:.3f}')
    ax.set_ylabel('fidelity')
    ax.grid()
    fig.tight_layout()
    fig.savefig('tbd01.png', dpi=200)


def demo_2qubits():
    matA = np.kron(PauliZ, PauliZ)
    matB = (np.kron(PauliZ, Pauli0) + np.kron(Pauli0, PauliZ))
    matC = np.kron(PauliX, Pauli0) + np.kron(Pauli0, PauliX)
    alpha = 0.1
    t_list = np.linspace(0, 2, 50)[1:-1]

    fidelity_improved = []
    fidelity_trotter = []
    for ti in t_list:
        tmp0 = scipy.linalg.expm(1j*alpha*ti/2 * (matB+matC))
        ret_ = tmp0 @ scipy.linalg.expm(1j*ti * matA) @ tmp0
        ret0 = scipy.linalg.expm(1j*ti * (matA + alpha*(matB/alpha+matC*ti/np.tan(ti))))
        fidelity_improved.append(get_unitary_fidelity(ret_, ret0))
        ret1 = scipy.linalg.expm(1j*ti * (matA + alpha*(matB+matC)))
        fidelity_trotter.append(get_unitary_fidelity(ret_, ret1))
    fidelity_improved = np.array(fidelity_improved)
    fidelity_trotter = np.array(fidelity_trotter)

    fig,ax = plt.subplots()
    ax.plot(t_list, fidelity_improved, label=r'$p=B/\alpha+tC/tan(t)$')
    ax.plot(t_list, fidelity_trotter, label=r'$p=B+C$')
    ax.set_xlabel('t')
    ax.legend()
    ax.set_title(f'alpha={alpha:.3f}')
    ax.set_ylabel('fidelity')
    ax.grid()
    fig.tight_layout()
    fig.savefig('tbd02.png', dpi=200)


def demo_20231124():
    matA = np.kron(PauliZ, PauliZ)
    matB = (np.kron(PauliZ, Pauli0) + np.kron(Pauli0, PauliZ))
    matC = np.kron(PauliX, Pauli0) + np.kron(Pauli0, PauliX)
    alpha = 1
    alpha_list = []
    t_list = np.linspace(0, 0.95, 50)[1:-1]

    fidelity01_list = []
    fidelity23_list = []
    for ti in t_list:
        tmp0 = scipy.linalg.expm(0.5j*(ti*matB + alpha*matC*np.tan(ti)))
        ret0 = tmp0 @ scipy.linalg.expm(1j*ti * matA) @ tmp0
        ret1 = scipy.linalg.expm(1j*ti * (matA + matB + alpha*matC))
        # ret2 = scipy.linalg.expm(1j*ti * (matA + matB + alpha*matC))
        tmp0 = scipy.linalg.expm(0.5j*(ti*matB + alpha*ti*matC))
        ret3 = tmp0 @ scipy.linalg.expm(1j*ti * matA) @ tmp0
        fidelity01_list.append(get_unitary_fidelity(ret0, ret1))
        fidelity23_list.append(get_unitary_fidelity(ret1, ret3))
    fidelity01_list = np.array(fidelity01_list)
    fidelity23_list = np.array(fidelity23_list)

    fig,ax = plt.subplots()
    ax.plot(t_list, fidelity01_list, label=r'$F(U_0,U_1)$')
    ax.plot(t_list, fidelity23_list, label=r'$F(U_1,U_3)$')
    ax.set_xlabel('t')
    ax.legend()
    ax.set_title(f'alpha={alpha:.3f}')
    ax.set_ylabel('fidelity')
    ax.grid()
    fig.tight_layout()
    fig.savefig('tbd02.png', dpi=200)


## TODO debug
# z0 = []
# for ti in t_list:
#     tmp0 = scipy.linalg.expm(1j*alpha*ti/2 * (matB+matC))
#     ret_ = tmp0 @ scipy.linalg.expm(1j*ti * matA) @ tmp0
#     tmp1 = scipy.linalg.logm(tmp0 @ scipy.linalg.expm(1j*ti * matA) @ tmp0)/1j
#     z0.append(tmp1 - np.trace(tmp1)*np.eye(4)/4)
# z0 = np.stack(z0)

# np.trace(z0 @ matA, axis1=1, axis2=2).real/t_list

# hf0 = lambda x: np.trace(x @ matA, axis1=1, axis2=2).real
# hf0(z0)/t_list
# hf0(z0 - matA*t_list[:,None,None])/t_list

# z1 = (z0/t_list[:,None,None] - matA)/alpha
# hf0 = lambda x: np.trace(x @ matB, axis1=1, axis2=2).real
# hf0(z1)
# hf0(z1 - matB)

# z2 = z1 - matB
# hf0 = lambda x: np.trace(x @ matC, axis1=1, axis2=2).real
# hf0(z2) / (t_list/np.tan(t_list))

# hf1 = lambda s: np.linalg.norm(hf0(z1-s*matB)/t_list)


if __name__=='__main__':
    demo_1qubit()

    demo_2qubits()

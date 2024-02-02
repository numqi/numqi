import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import numqi

cp_tableau = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
np_rng = np.random.default_rng()


def demo_search_UD_in_pauli_group():
    num_qubit = 3
    num_repeat = {2:10, 3:10, 4:80, 5:80}[num_qubit]
    num_init_sample = {2:0, 3:10, 4:80, 5:400}[num_qubit]
    matrix_subspace = numqi.gate.get_pauli_group(num_qubit, use_sparse=True)
    kwargs = {'num_repeat':num_repeat,  'num_init_sample':num_init_sample, 'indexF':[0],
                'num_worker':1, 'tag_print':True}
    z0 = numqi.unique_determine.find_optimal_UD('udp', num_round=2, mat_list=matrix_subspace, **kwargs)
    # all_index = numqi.unique_determine.load_pauli_ud_example(num_qubit)
    matB_list = [numqi.unique_determine.get_matrix_list_indexing(matrix_subspace, x) for x in z0]
    z1 = numqi.unique_determine.check_UD('uda', matB_list, num_repeat=num_repeat*5, num_worker=1)


def demo_search_UD_in_gellmann_group():
    num_qudit = 2
    dim_qudit = 3
    num_repeat = {(2,3):80, (1,3):10}[(num_qudit,dim_qudit)]
    num_init_sample = {(2,3):10, (1,3):0}[(num_qudit,dim_qudit)]

    gellmann_basis = numqi.gellmann.all_gellmann_matrix(num_qudit, tensor_n=dim_qudit, with_I=True)
    # last item is identity, so put it in indexF (fixed index)
    z0 = numqi.unique_determine.find_optimal_UD('udp', num_round=1, mat_list=gellmann_basis,
                indexF=[len(gellmann_basis)-1], num_repeat=num_repeat, num_init_sample=num_init_sample,
                num_worker=1, tag_print=True)
    # all_index = numqi.unique_determine.save_index_to_file('gellmann-UD.json', key=f'{num_qudit},{dim_qudit},udp', index=z0)
    matB_list = [gellmann_basis[x] for x in z0]
    numqi.unique_determine.check_UD('udp', matB_list, num_repeat, early_stop_threshold=1e-4, converge_tol=1e-7, dtype='float64', num_worker=19)


def demo_4PB_5PB():
    dim_list = list(range(3, 9))
    kwargs = dict(num_repeat=5, early_stop_threshold=1e-10, converge_tol=1e-12, dtype='float64')
    # ~3 min (num_repeat=5)

    udp_3pb_loss_list = []
    uda_4pb_loss_list = []
    uda_5pb_loss_list = []
    udp_4pb_loss_list = []
    udp_5pb_loss_list = []
    for dim in dim_list:
        print(f'd={dim}')
        alpha = np.pi/dim

        matB = numqi.unique_determine.get_chebshev_orthonormal(dim, alpha, with_computational_basis=False)[:(-dim)]
        # matB = numqi.matrix_space.get_matrix_orthogonal_basis(matB, field='real')[0]
        udp_3pb_loss_list.append(numqi.unique_determine.check_UD('udp', matB, **kwargs)[1])

        matB = numqi.unique_determine.get_chebshev_orthonormal(dim, alpha, with_computational_basis=False)
        udp_4pb_loss_list.append(numqi.unique_determine.check_UD('udp', matB, **kwargs)[1])
        uda_4pb_loss_list.append(numqi.unique_determine.check_UD('uda', matB, **kwargs)[1])

        matB = numqi.unique_determine.get_chebshev_orthonormal(dim, alpha, with_computational_basis=True)
        udp_5pb_loss_list.append(numqi.unique_determine.check_UD('udp', matB, **kwargs)[1])
        uda_5pb_loss_list.append(numqi.unique_determine.check_UD('uda', matB, **kwargs)[1])

    fig,(ax0,ax1) = plt.subplots(1,2,figsize=(8,4))
    ax0.plot(dim_list, udp_4pb_loss_list, label='4PBs', marker='.')
    ax0.plot(dim_list, udp_3pb_loss_list, label='3PBs', marker='x')
    ax1.plot(dim_list, uda_5pb_loss_list, label='5PBs', marker='.')
    ax1.plot(dim_list, uda_4pb_loss_list, label='4PBs', marker='x')
    for ax in [ax0,ax1]:
        ax.set_xlabel('qudit $d$')
        ax.legend()
        ax.set_yscale('log')
        ax.grid()
        ax.set_ylim(3e-14, 0.4)
    ax0.set_ylabel(r'UDP loss')
    ax1.set_ylabel(r'UDA loss')
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)


def demo_search_UD_in_qutrit_projector():
    ## 1 qutrit, projector
    matrix_subspace = numqi.unique_determine.get_qutrit_projector_basis(num_qutrit=1)
    z0 = numqi.unique_determine.find_optimal_UD('udp', num_round=3, mat_list=matrix_subspace, indexF=[0],
            num_repeat=10, num_init_sample=1, tag_print=True)
    numqi.unique_determine.check_UD('udp', matrix_subspace[z0[0]], num_repeat=300, early_stop_threshold=1e-4)

    ## 2 qutrit, projector
    matrix_subspace = numqi.unique_determine.get_qutrit_projector_basis(num_qutrit=2)
    z0 = numqi.unique_determine.find_optimal_UD('udp', num_round=3, mat_list=matrix_subspace, indexF=[0],
            early_stop_threshold=0.001, num_repeat=80, num_init_sample=130, converge_tol=1e-6, tag_print=True)


def get_qubit_projector_basis(num_qubit):
    assert num_qubit>=1
    hf0 = lambda x,y,z=1: np.array([[x*np.conj(x),x*np.conj(y)], [y*np.conj(x), y*np.conj(y)]])/(z*z)
    s2 = np.sqrt(2)
    tmp0 = [(1,0), (0,1), (1,1,s2), (1,-1,s2), (1,1j,s2), (1,-1j,s2)]
    np0 = np.stack([hf0(*x) for x in tmp0])
    ret = np0
    for _ in range(num_qubit-1):
        tmp0 = np.einsum(ret, [0,1,2], np0, [3,4,5], [0,3,1,4,2,5], optimize=True)
        tmp1 = [x*y for x,y in zip(ret.shape, np0.shape)]
        ret = tmp0.reshape(tmp1)
    ret = np.concatenate([np.eye(ret.shape[1])[np.newaxis], ret], axis=0)
    return ret

def demo_qubit_ud_orthonormal():
    # 2 qubits, non-orthogonal matrix, UDA!=UDP
    num_qubit = 2
    matrix_subspace = get_qubit_projector_basis(num_qubit)
    z0 = numqi.unique_determine.find_UD('udp', num_round=10, matrix_basis=matrix_subspace, indexF=[0],
                num_repeat=100, num_random_select=15, tag_reduce=True, early_stop_threshold=0.0001, num_worker=10)
    matB_list = [matrix_subspace[x] for x in z0]
    z1 = numqi.unique_determine.check_UD('udp', matB_list, num_repeat=1000, early_stop_threshold=0.0001, converge_tol=1e-7, dtype='float64')
    z2 = numqi.unique_determine.check_UD('uda', matB_list, num_repeat=1000, early_stop_threshold=0.0001, converge_tol=1e-7, dtype='float64')


def demo_EP_POVM():
    dim0 = 3
    matrix_subspace = numqi.unique_determine.get_element_probing_POVM('eq8', dim0)
    tag_ud,loss,model = numqi.unique_determine.check_UD('udp', matrix_subspace, num_repeat=3,
                early_stop_threshold=1e-11, converge_tol=1e-11, dtype='float64', return_model=True)
    matH = model.matH.numpy().copy()
    # |11> - |22>

    dim0 = 6
    matrix_subspace = numqi.unique_determine.get_element_probing_POVM('eq9', dim0)
    tag_ud,loss,model = numqi.unique_determine.check_UD('udp', matrix_subspace, num_repeat=3,
                early_stop_threshold=1e-13, converge_tol=1e-13, dtype='float64', return_model=True)
    matH = model.matH.numpy().copy()
    assert np.abs(np.trace(matrix_subspace @ matH, axis1=1, axis2=2)).max() < 1e-6



def demo_UDA_5PB_stability():
    dim_qudit = 6
    num_random = 50
    cvxpy_eps = 1e-6
    noise_rate_list = np.logspace(-6, -3, 10)

    matrix_subspace = numqi.unique_determine.get_chebshev_orthonormal(dim_qudit, alpha=np.pi/dim_qudit, with_computational_basis=True)
    # about 5 mins
    tag_ud,loss,model = numqi.unique_determine.check_UD('uda', matrix_subspace, num_repeat=100,
            converge_tol=1e-10, early_stop_threshold=1e-10, dtype='float64', tag_print=2, return_model=True)
    matH = model.matH.numpy().copy()
    EVL,EVC = np.linalg.eigh(matH)
    state_list = [EVC[:,0]] + [numqi.random.rand_haar_state(dim_qudit) for _ in range(2)]

    data = []
    for state_i in state_list:
        measure_no_noise = ((matrix_subspace @ state_i) @ state_i.conj()).real
        for noise_rate in noise_rate_list:
            for _ in tqdm(range(num_random)):
                tmp0 = np_rng.normal(size=len(matrix_subspace))
                noise = tmp0 * (noise_rate/np.linalg.norm(tmp0))
                tmp1,eps = numqi.unique_determine.density_matrix_recovery_SDP(matrix_subspace, measure_no_noise + noise, converge_eps=cvxpy_eps)
                tmp2 = np.linalg.norm(tmp1 - state_i[:,np.newaxis]*state_i.conj(), ord='fro') #frob norm
                data.append((np.linalg.norm(noise), eps, tmp2))
    data = np.array(data).reshape(-1, len(noise_rate_list), num_random, 3).transpose(0,3,1,2)

    fig,(ax0,ax1) = plt.subplots(1, 2, figsize=(8,4))
    for ind0 in range(data.shape[0]):
        tmp0= noise_rate_list
        tmp1 = data[ind0,1]
        ax0.fill_between(tmp0, tmp1.min(axis=1), tmp1.max(axis=1), alpha=0.2, color=cp_tableau[ind0])
        tmp2 = r'$\psi_-$' if ind0==0 else r'random $\sigma_'+f'{ind0}$'
        ax0.plot(tmp0, tmp1.mean(axis=1), color=cp_tableau[ind0], label=tmp2)

        tmp1 = data[ind0,2] / (data[ind0,0] + data[ind0,1])
        ax1.fill_between(tmp0, tmp1.min(axis=1), tmp1.max(axis=1), alpha=0.2, color=cp_tableau[ind0])
        ax1.plot(tmp0, tmp1.mean(axis=1), color=cp_tableau[ind0])
    ax0.set_ylabel(r'$\epsilon$')
    ax1.set_ylabel(r'$\frac{||Y-\sigma||_F}{\epsilon+||f||_2}$')
    ax1.axhline(1/np.sqrt(loss), color='r', linestyle='--', label=r'$1/\sqrt{\mathcal{L}}$')
    fig.suptitle(f'5PB(d={dim_qudit})')
    ax0.legend()
    for ax in [ax0,ax1]:
        ax.set_xlabel(r'$||f||_2$')
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)


def demo_pauli_loss_function():
    num_worker = 4
    num_qubit_to_num_repeat = {2:50, 3:50, 4:400, 5:3200}
    pauli_len_dict = {2:11, 3:31, }# 4:106, 5:398
    z0 = numqi.unique_determine.load_pauli_ud_example(tag_group_by_size=True)
    z0 = {k:{k1:v1[:6] for k1,v1 in v.items()} for k,v in z0.items()} #each group only keep 6 items
    z1 = dict()
    for num_qubit,max_len in pauli_len_dict.items():
        if num_qubit not in z0:
            continue
        num_repeat = num_qubit_to_num_repeat[num_qubit]
        matrix_subspace = numqi.gate.get_pauli_group(num_qubit, use_sparse=True)

        tmp0 = [y for x,y in z0[num_qubit].items() if x<=max_len]
        index_list = [y for x in tmp0 for y in x]
        matB_list = [numqi.unique_determine.get_matrix_list_indexing(matrix_subspace, x) for x in index_list]
        uda_loss = [x[1] for x in numqi.unique_determine.check_UD('uda', matB_list, num_repeat, num_worker=num_worker)]
        udp_loss = [x[1] for x in numqi.unique_determine.check_UD('udp', matB_list, num_repeat, num_worker=num_worker)]
        tmp0 = itertools.groupby(zip(index_list,uda_loss,udp_loss), key=lambda x:len(x[0]))
        tmp1 = {x0:list(zip(*x1)) for x0,x1 in tmp0}
        for x0,x1 in tmp1.items():
            print(f'[{num_qubit},{x0},UDA]', np.sort(np.array(x1[1])))
            print(f'[{num_qubit},{x0},UDP]', np.sort(np.array(x1[2])))
        z1[num_qubit] = tmp1

# [2,11,UDA] [1.33333623 1.33333671 1.33333683 1.33334064 1.33334363 1.3333447 ]
# [2,11,UDP] [1.99999869 1.99999881 1.99999905 1.99999917 1.99999917 1.99999917]
# [3,31,UDA] [0.98112118 0.98115534 0.98131108 0.98138881 0.98143536 0.98149371]
# [3,31,UDP] [1.99999905 1.99999952 1.99999952 1.99999952 1.99999976 1.99999976]
# [4,106,UDA] [0.72969043 0.78669071 0.80056381]
# [4,106,UDP] [1.78830206 1.81233203 1.82333124]
# [5,393,UDA] [0.66714954]
# [5,393,UDP] [1.95155692]
# [5,395,UDA] [0.66707623]
# [5,395,UDP] [1.90302622]
# [5,397,UDA] [0.66705465 0.66711843]
# [5,397,UDP] [1.71428847 1.89451289]
# [5,398,UDA] [0.6671102  0.66723371]
# [5,398,UDP] [1.94959903 1.97744238]

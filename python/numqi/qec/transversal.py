import itertools
import numpy as np
import torch
import opt_einsum
import scipy.optimize
import cvxpy

import numqi.utils
import numqi.manifold
import numqi.gate
import numqi.optimize
import numqi.group

a = (1+np.sqrt(5))/2
b = (np.sqrt(5)-1)/2
su2_finite_subgroup_gate_dict = {
    'I': np.eye(2),
    'X': -1j*np.array([[0,1], [1,0]]),
    'Y': -1j*np.array([[0,-1j], [1j,0]]),
    'Z': -1j*np.array([[1,0], [0,-1]]),
    'H': -1j*np.array([[1,1], [1,-1]])/np.sqrt(2),
    'S': np.array([[np.exp(-1j*np.pi/4),0], [0,np.exp(1j*np.pi/4)]]),
    'T': np.array([[np.exp(-1j*np.pi/8),0], [0,np.exp(1j*np.pi/8)]]),
    'F': np.exp(-1j*np.pi/4)/np.sqrt(2) * np.array([[1,-1j], [1,1j]]), #-1j H @ Z(-pi/2)
    'Phi': np.array([[a+1j*b,1], [-1,a-1j*b]])/2,
    'Phi*': np.array([[-b+1j*a,1], [-1,-b-1j*a]])/2,
    # 'tau60': np.array([[1j*(2+a),1+1j], [-1+1j,-1j*(2+a)]])/np.sqrt(5*a+7),
}


def get_su2_finite_subgroup_generator(key:str):
    if key=='2T':
        ret = np.stack([su2_finite_subgroup_gate_dict[x] for x in 'XF'])
        # ret = np.stack([su2_finite_subgroup_gate_dict[x] for x in 'XZF'])
    elif key.startswith('2O'): #Clifford
        if key=='2O':
            ret = np.stack([su2_finite_subgroup_gate_dict[x] for x in 'SH']) #2 generators are enough
            # ret = np.stack([su2_finite_subgroup_gate_dict[x] for x in 'SHF'])
        else:
            assert key=='2Ox' #X, Ry(pi/4) S Ry(-pi/4)
            ret = np.stack([su2_finite_subgroup_gate_dict['X'], 0.5*np.array([[np.sqrt(2)-1j,-1j], [-1j,np.sqrt(2)+1j]])])
    elif key=='2I': #Clifford
        tmp0 = su2_finite_subgroup_gate_dict['Z'] @ su2_finite_subgroup_gate_dict['Phi']
        ret = np.stack([su2_finite_subgroup_gate_dict['X'], tmp0], axis=0) #2 generators are enough
        # ret = np.stack([su2_finite_subgroup_gate_dict[x] for x in ['Y','Phi']]) #2 generators are enough
        # ret = np.stack([su2_finite_subgroup_gate_dict[x] for x in ['X','Z','F','Phi']])
    elif key.startswith('C'):
        n2 = int(key[1:])
        assert n2%2==0
        ret = np.array([[np.exp(-2j*np.pi/n2),0], [0,np.exp(2j*np.pi/n2)]]).reshape(1,2,2)
    elif key.startswith('BD'):
        n2 = int(key[2:])
        assert n2%2==0
        ret = np.stack([
            su2_finite_subgroup_gate_dict['X'],
            np.array([[np.exp(-2j*np.pi/n2),0], [0,np.exp(2j*np.pi/n2)]])
        ])
    else:
        raise ValueError(f'invalid transversal group "key={key}"')
    return ret


class SearchTransversalGateModel(torch.nn.Module):
    def __init__(self, code:np.ndarray, tag_rz:bool=False, tag_phase:bool=False):
        super().__init__()
        assert (code.ndim==2) and (code.shape[0]>=2)
        dimK = code.shape[0]
        num_qubit = numqi.utils.hf_num_state_to_num_qubit(code.shape[1], kind='exact')
        self.code = torch.tensor(code.T.copy().reshape([2]*num_qubit+[dimK]), dtype=torch.complex128)
        if tag_rz:
            self.theta_rz = torch.nn.Parameter(torch.randn(num_qubit, dtype=torch.float64))
            self.manifold = None
        else:
            self.theta_rz = None
            self.manifold = numqi.manifold.SpecialOrthogonal(2, batch_size=num_qubit, dtype=torch.complex128)
        self.eyeK = torch.eye(dimK, dtype=torch.complex128)
        if tag_phase:
            self.theta_phase = torch.nn.Parameter(torch.randn(1, dtype=torch.float64))

        N = num_qubit
        tmp0 = [y for x in range(N) for y in [(2,2), (N+x,x)]]
        self.contract_expr = opt_einsum.contract_expression(self.code, list(range(N))+[2*N],
                            self.code.conj().resolve_conj(), list(range(N,2*N))+[2*N+1], *tmp0, [2*N+1,2*N], constants=[0,1])
        self.target_gate = None

    def set_target_gate(self, x:None|np.ndarray):
        if x is None:
            self.target_gate = None
        else:
            assert np.abs(x@x.T.conj() - np.eye(self.eyeK.shape[0])).max() < 1e-10
            self.target_gate = torch.tensor(x, dtype=torch.complex128)

    def forward(self, return_info=False):
        if self.theta_rz is not None:
            su2 = numqi.gate.rz(self.theta_rz)
        else:
            su2 = self.manifold()
        logicalU = self.contract_expr(*su2)
        if self.target_gate is None:
            tmp0 = (logicalU @ logicalU.T.conj() - self.eyeK).reshape(-1)
        else:
            tmp0 = self.target_gate * (torch.exp(1j*self.theta_phase) if hasattr(self,'theta_phase') else 1)
            tmp0 = (logicalU - tmp0).reshape(-1)
        ret = torch.vdot(tmp0, tmp0).real
        if return_info:
            ret = ret, su2.detach().numpy(), logicalU.detach().numpy()
        return ret


def get_transversal_group(code:np.ndarray, num_round=100, optim_tol=1e-15, tag_print=False, optim_kwargs=None,
                max_num_logical:int|None=150, zero_eps_same=1e-5, zero_eps_minimize=1e-10):
    if optim_kwargs is None:
        optim_kwargs = dict(theta0='uniform', num_repeat=20, tol=1e-10, early_stop_threshold=1e-7, print_every_round=0)
    hf_is_same = lambda x,y: (np.abs(x - y).max() < zero_eps_same)
    model = SearchTransversalGateModel(code)
    logical_list = [] #list of tuple (logicalU,physicalU)
    for ind0 in range(num_round):
        theta_optim = numqi.optimize.minimize(model, **optim_kwargs)
        theta_optim = numqi.optimize.minimize(model, theta0=theta_optim.x, num_repeat=1, tol=optim_tol, print_every_round=0)
        if theta_optim.fun > zero_eps_minimize:
            continue
        with torch.no_grad():
            loss,physicalU,logicalU = model(return_info=True) # loss, physicalU, logicalU
            logicalU = logicalU * np.exp(-1j*np.angle(np.linalg.det(logicalU))/2) #not necessary SU(2), e.g., 623-SO5 code
        if any(hf_is_same(logicalU, x[0]) for x in logical_list):
            continue
        logical_list.append((logicalU,physicalU))
        if tag_print:
            print(f'[{len(logical_list)}/{ind0+1}/{theta_optim.fun:.5g}]\n{np.around(logicalU,3)}')
    # complete group, almost the same as numqi.group.get_complete_group
    check_list = [(x,y) for x in range(len(logical_list)) for y in range(len(logical_list))]
    # TODO replace with np.append
    while len(check_list):
        ind0,ind1 = check_list.pop(0)
        np1 = logical_list[ind0][0] @ logical_list[ind1][0]
        if not any(hf_is_same(np1, x[0]) for x in logical_list):
            np2 = logical_list[ind0][1] @ logical_list[ind1][1]
            logical_list.append((np1,np2))
            check_list += [(x,len(logical_list)-1) for x in range(len(logical_list))]
            check_list += [(len(logical_list)-1,x) for x in range(len(logical_list))]
        if (max_num_logical is not None) and len(logical_list)>=max_num_logical: #numerical is not stable
            break
    return logical_list


def get_transversal_group_info(np_list, tag_print=True, zero_eps=1e-7):
    np_list = np.stack(np_list, axis=0)
    assert (np_list.ndim==3) and (np_list.shape[1]==np_list.shape[2])
    N0 = np_list.shape[1]
    cayley_table = np.zeros((len(np_list), len(np_list)), dtype=np.int64)
    for ind0 in range(len(np_list)):
        tmp0 = (np_list[ind0] @ np_list).reshape(-1,1,N0*N0)
        tmp1 = np.abs(tmp0 - np_list.reshape(-1,N0*N0)).max(axis=2) < zero_eps
        assert np.all(tmp1.sum(axis=0)==1) and np.all(tmp1.sum(axis=1)==1)
        cayley_table[ind0] = tmp1.nonzero()[1]
    left_regular_form = numqi.group.cayley_table_to_left_regular_form(cayley_table) #(np,int,(N0,N0,N0))
    irrep_list = numqi.group.reduce_group_representation(left_regular_form, tagI=True) #(list,(np,complex,(N0,N1,N1)))
    _,class_list,character_table = numqi.group.get_character_and_class(irrep_list, tagIFirst=True)
    ret = {
        'cayley_table': cayley_table,
        'irrep_list': irrep_list,
        'class_list': class_list,
        'character_table': character_table,
        'dim_irrep': np.array([x.shape[1] for x in irrep_list], dtype=np.int64),
        'num_class': np.array([len(x) for x in class_list], dtype=np.int64),
    }
    if tag_print:
        print('order(group):', np_list.shape[0])
        print('dim(irrep):', ret['dim_irrep'])
        print('class:', ret["num_class"])
    return ret


def pick_indenpendent_vector(np0:np.ndarray, tag_pure_imag:bool=False, zero_eps:float=1e-10):
    assert np0.ndim==2
    if tag_pure_imag:
        ind0 = np.abs(np0.real).max(axis=1) > zero_eps
        ind1 = np.abs(np0.imag).max(axis=1) > zero_eps
        assert np.all((ind0.astype(np.int64) + ind1.astype(np.int64))<2)
        np0[ind1] = np0[ind1].imag
        np0 = np0.real
    np0 = np0[np.abs(np0).max(axis=1) > zero_eps]
    rank = (np.linalg.svd(np0, compute_uv=False) > zero_eps).sum()
    if rank==1:
        ret = np0[0]
    else:
        ind0 = [0]
        while len(ind0)<rank:
            for x in range(ind0[-1]+1, np0.shape[0]):
                if np.linalg.svd(np0[ind0 + [x]], compute_uv=False)[-1] > zero_eps:
                    ind0.append(x)
                    break
        ret = np0[ind0]
    return ret


def get_chebshev_center_Axb(A, b, C=None, d=None):
    # https://en.wikipedia.org/wiki/Chebyshev_center
    m,n = A.shape
    if C is None:
        assert d is None
        C = -np.eye(n)
        d = np.zeros(n)
    cvxX = cvxpy.Variable(n)
    cvxR = cvxpy.Variable(1)
    constraint = [
        A @ cvxX==b,
        C @ cvxX + np.linalg.norm(C, axis=1)*cvxR <= d,
    ]
    obj = cvxpy.Maximize(cvxR)
    prob = cvxpy.Problem(obj, constraint)
    prob.solve()
    ret = cvxX.value, cvxR.value
    return ret


def get_BD2m_submultiset(m:int, veca:np.ndarray):
    veca = np.asarray(veca, dtype=np.int64)
    assert len(veca)<=15, 'veca cannot be longer than 15'
    assert np.all(veca[:-1]<=veca[1:])
    ind0 = np.array(list(itertools.product([0,1], repeat=len(veca))), dtype=np.int64)
    ind1 = np.nonzero((ind0 @ veca)%m==0)[0]
    return ind1


def get_C2m_submultiset(m:int, veca:np.ndarray):
    veca = np.asarray(veca, dtype=np.int64)
    assert len(veca)<=15, 'veca cannot be longer than 15'
    assert np.all(veca[:-1]<=veca[1:])
    ind0 = np.array(list(itertools.product([0,1], repeat=len(veca))), dtype=np.int64)
    ind1 = np.nonzero((ind0 @ veca)%m==0)[0]
    ind2 = np.nonzero(((ind0 @ veca)+1)%m==0)[0]
    return ind1,ind2


def _BD_group_LP(bsi:np.ndarray):
    s,n = bsi.shape
    tmp0 = np.concatenate([bsi.T, np.ones((1, s))], axis=0)
    tmp1 = np.ones(n+1)
    tmp1[:-1] *= 0.5
    tmp2 = [(0, 1)]*s
    res = scipy.optimize.linprog(np.zeros(s), A_eq=tmp0, b_eq=tmp1, bounds=tmp2, options={'disp': False})
    return (res.success and res.status == 0)

def search_veca_BD_group(n:int, m:int, tag_print=True, min_value:int=0, k:int|None=2, min_term=None):
    if min_term is None:
        min_term = n
    assert min_term>=2
    ind0 = np.array(list(itertools.product([0,1], repeat=n)), dtype=np.int32)
    if k is None:
        tmp0 = (x for x in itertools.combinations_with_replacement(range(min_value, m), n) if (sum(x)+1)%m==0)
    else:
        assert k>=2
        tmp0 = (x for x in itertools.combinations_with_replacement(range(min_value, m), n) if sum(x)==(k*m-1))
    ret = []
    for a in tmp0:
        ind1 = (ind0 @ np.array(a))%m==0
        if (ind1.sum()>=min_term) and _BD_group_LP(ind0[ind1]): #TODO, this results requires bsi has rank n
            ret.append(a)
            if tag_print:
                print(a)
    return ret

def _C_group_LP(bsi0, bsij0, bsi1, bsij1):
    s,n = bsi0.shape
    s1,_ = bsi1.shape
    tmp0 = np.concatenate([bsi0.T, -bsi1.T], axis=1)
    tmp0b = np.zeros(n)
    tmp1 = np.concatenate([bsij0.T, -bsij1.T], axis=1)
    tmp1b = np.zeros(bsij0.shape[1])
    tmp2 = np.stack([np.concatenate([np.ones(s),np.zeros(s1)], axis=0), np.concatenate([np.zeros(s),np.ones(s1)], axis=0)], axis=0)
    tmp2b = np.ones(2)
    tmp3 = np.concatenate([tmp0, tmp1, tmp2], axis=0)
    tmp3b = np.concatenate([tmp0b, tmp1b, tmp2b], axis=0)
    tmp4 = [(0,1)]*(s+s1)
    res = scipy.optimize.linprog(np.zeros(s+s1), A_eq=tmp3, b_eq=tmp3b, bounds=tmp4, method='highs', options={'disp': False, 'presolve':True})
    return (res.success and res.status == 0)


def search_veca_C_group(n:int, m:int, tag_print=True, min_value:int=0, min_term=None):
    if min_term is None:
        min_term = n
    assert min_term>=2
    ind0 = np.array(list(itertools.product([0,1], repeat=n)), dtype=np.int32)
    i0,i1 = np.triu_indices(n,1)
    ind0ij = ind0[:,i0]*ind0[:,i1]
    ret = []
    for a in itertools.combinations_with_replacement(range(min_value, m), n):
        tmp0 = (ind0 @ np.array(a, dtype=np.int32))%m
        ind1 = tmp0==0
        ind2 = tmp0==(m-1)
        if (ind1.sum() >= min_term) and (ind2.sum() >= min_term) and _C_group_LP(ind0[ind1], ind0ij[ind1], ind0[ind2], ind0ij[ind2]):
            ret.append(tuple(a))
            if tag_print:
                print(a)
    return ret

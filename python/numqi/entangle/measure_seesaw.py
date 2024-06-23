import numpy as np
import scipy.linalg
import opt_einsum

import numqi.matrix_space
import numqi.manifold

def _get_GME_pure_seesaw_contract_expr(np0, dim_list):
    N0 = len(dim_list)
    ret = []
    for ind0 in range(N0):
        tmp0 = [((dim_list[x],),(x,)) for x in range(N0) if x!=ind0]
        tmp1 = [y for x in tmp0 for y in x]
        ret.append(opt_einsum.contract_expression(np0.conj(), list(range(N0)), *tmp1, [ind0], constants=[0]))
    return ret

def _get_GME_pure_seesaw_hf0(np0, converge_eps, maxiter, psi_list, contract_expr_list=None):
    # this function is for testing purpose
    dim_list = np0.shape
    N0 = len(dim_list)
    if contract_expr_list is None:
        contract_expr_list = _get_GME_pure_seesaw_contract_expr(np0, dim_list)
    for _ in range(maxiter):
        fidelity_list = np.zeros(N0, dtype=np.float64)
        for ind0 in range(N0):
            tmp0 = [psi_list[x] for x in range(N0) if x!=ind0]
            tmp1 = contract_expr_list[ind0](*tmp0).conj()
            ret_fidelity = np.vdot(tmp1, tmp1).real
            tmp2 = tmp1 / np.sqrt(ret_fidelity)
            fidelity_list[ind0] = abs(np.vdot(tmp2, psi_list[ind0]))**2
            psi_list[ind0] = tmp2
        if (1-fidelity_list.min())<converge_eps:
            break
    ret = max(0,1-ret_fidelity), psi_list
    return ret


def get_GME_pure_seesaw(np0:np.ndarray, converge_eps:float=1e-7, num_repeat:int=1, maxiter=1000, psi_list=None, seed=None):
    r'''Get the Geometric Measure of Entanglement (GME) of a pure state using the seesaw algorithm.

    reference: Simple algorithm for computing the geometric measure of entanglement
    [doi-link](https://doi.org/10.1103/PhysRevA.84.022323)

    Parameters:
        np0 (np.ndarray): pure state, `shape=(d1,d2,...,dN)`
        converge_eps (float): convergence threshold
        num_repeat (int): number of repeat to get the best result
        maxiter (int): maximum number of iterations
        psi_list (list): initial states, `len(psi_list)==N`, `psi_list[ind0].shape==(dim_list[ind0],)`.
                If None, random initial states will be used. When `num_repeat>1`, `psi_list` must be None.
        seed (int,None): random seed

    Returns:
        GME (float): non-negative number, the Geometric Measure of Entanglement
        psi_list (list): final states, `len(psi_list)==N`, `psi_list[ind0].shape==(dim_list[ind0],)`
    '''
    N0 = np0.ndim
    assert (N0>=2) and all(x>=2 for x in np0.shape)
    np0 = np0 / np.linalg.norm(np0.reshape(-1), ord=2)
    if N0==2:
        U,S,V = np.linalg.svd(np0, full_matrices=False)
        ret = 1-S[0]**2, [U[:,0], V[0]]
    else:
        dim_list = np0.shape
        contract_expr_list = _get_GME_pure_seesaw_contract_expr(np0, dim_list)
        if psi_list is not None:
            assert num_repeat==1, "num_repeat>1 is not supported with psi_list"
            assert len(psi_list)==N0
            assert all(x.shape==(y,) for x,y in zip(psi_list, dim_list)), "Inconsistent shape"
        ret_repeat_list = []
        np_rng = np.random.default_rng(seed)
        for _ in range(num_repeat):
            if (num_repeat>1) or (psi_list is None):
                psi_list = []
                for ind0 in range(N0):
                    tmp0 = np_rng.normal(size=dim_list[ind0]) + 1j * np_rng.normal(size=dim_list[ind0])
                    psi_list.append(tmp0/np.linalg.norm(tmp0, ord=2))
            ret_repeat_list.append(_get_GME_pure_seesaw_hf0(np0, converge_eps, maxiter, psi_list, contract_expr_list))
        ret = min(ret_repeat_list, key=lambda x: x[0])
    return ret


def get_GME_subspace_seesaw(np0:np.ndarray, converge_eps:float=1e-7, num_repeat:int=1, maxiter=1000, psi_list=None, seed=None):
    r'''Get the Geometric Measure of Entanglement (GME) of a subspace using the seesaw algorithm.

    reference: Simple algorithm for computing the geometric measure of entanglement
    [doi-link](https://doi.org/10.1103/PhysRevA.84.022323)

    Parameters:
        np0 (np.ndarray): pure state, `shape=(N,d1,d2,...,dN)`
        converge_eps (float): convergence threshold
        num_repeat (int): number of repeat to get the best result
        maxiter (int): maximum number of iterations
        psi_list (list): initial states, `len(psi_list)==N`, `psi_list[ind0].shape==(dim_list[ind0],)`.
                If None, random initial states will be used. When `num_repeat>1`, `psi_list` must be None.
        seed (int,None): random seed

    Returns:
        GME (float): non-negative number, the Geometric Measure of Entanglement
        psi_list (list): final states, `len(psi_list)==N`, `psi_list[ind0].shape==(dim_list[ind0],)`
    '''
    N0 = np0.ndim-1
    assert (N0>=2) and all(x>=2 for x in np0.shape[1:])
    if np0.shape[0]==1:
        ret = get_GME_pure_seesaw(np0[0], converge_eps, num_repeat, maxiter, psi_list, seed)
    else:
        dim_list = np0.shape[1:]
        np0 = numqi.matrix_space.reduce_vector_space(np0.reshape(np0.shape[0],-1), zero_eps=1e-10).reshape(-1, *dim_list)
        contract_expr_list = []
        for ind0 in range(N0):
            tmp0 = [((dim_list[x],),(x+1,)) for x in range(N0) if x!=ind0]
            tmp1 = [y for x in tmp0 for y in x]
            contract_expr_list.append(opt_einsum.contract_expression(np0.conj(), list(range(N0+1)), *tmp1, [0,ind0+1], constants=[0]))
        if psi_list is not None:
            assert num_repeat==1, "num_repeat>1 is not supported with psi_list"
            assert len(psi_list)==N0
            assert all(x.shape==(y,) for x,y in zip(psi_list, dim_list)), "Inconsistent shape"
        ret_repeat_list = []
        for _ in range(num_repeat):
            if (num_repeat>1) or (psi_list is None):
                np_rng = np.random.default_rng(seed)
                psi_list = []
                for ind0 in range(N0):
                    tmp0 = np_rng.normal(size=dim_list[ind0]) + 1j * np_rng.normal(size=dim_list[ind0])
                    psi_list.append(tmp0/np.linalg.norm(tmp0, ord=2))
            for _ in range(maxiter):
                fidelity_list = np.zeros(N0, dtype=np.float64)
                for ind0 in range(N0):
                    tmp0 = [psi_list[x] for x in range(N0) if x!=ind0]
                    tmp1 = contract_expr_list[ind0](*tmp0)
                    EVL,EVC = np.linalg.eigh(tmp1.T.conj() @ tmp1)
                    EVCi = EVC[:,-1]
                    fidelity_list[ind0] = abs(np.vdot(EVCi, psi_list[ind0]))**2
                    gme = max(0, 1-EVL[-1])
                    psi_list[ind0] = EVCi
                if (1-fidelity_list.min())<converge_eps:
                    break
            ret_repeat_list.append((gme, psi_list))
        ret = min(ret_repeat_list, key=lambda x: x[0])
    return ret


def get_GME_seesaw(rho:np.ndarray, dim_list:tuple[int], maxiter:int=1000, init_coeffq:np.ndarray|None=None,
                init_phi_list:list|None=None, num_state:int|None=None, converge_eps:float=1e-10,
                num_repeat:int=1, maxiter_inner:int=100, converge_eps_inner:float=1e-7, zero_eps:float=1e-10,
                return_info:bool=False, seed:int|None=None):
    r'''Get the Geometric Measure of Entanglement (GME) of a density matrix using the seesaw algorithm.

    reference: Simple algorithm for computing the geometric measure of entanglement
    [doi-link](https://doi.org/10.1103/PhysRevA.84.022323)

    Parameters:
        rho (np.ndarray): density matrix, `shape=(d1*d2*...*dN,d1*d2*...*dN)`
        dim_list (tuple[int]): dimension list, `len(dim_list)==N`, `dim_list[ind0]` is the dimension of the ind0-th party
        maxiter (int): maximum number of iterations
        init_coeffq (np.ndarray,None): initial probability distribution, `shape=(num_state,)`
        init_phi_list (list,None): initial states, `len(init_phi_list)==N`, `init_phi_list[ind0].shape==(num_state,dim_list[ind0])`
        num_state (int,None): number of states, default is `2*dim_total`
        converge_eps (float): convergence threshold
        num_repeat (int): number of repeat to get the best result
        maxiter_inner (int): maximum number of iterations for inner loop
        converge_eps_inner (float): convergence threshold for inner loop
        zero_eps (float): zero threshold
        return_info (bool): return additional information
        seed (int,None): random seed

    Returns:
        GME (float): non-negative number, the Geometric Measure of Entanglement
        info (dict): additional information, if `return_info==True`
    '''
    assert num_repeat>=1
    if num_repeat>1:
        assert (init_coeffq is None) and (init_phi_list is None)
    if (init_coeffq is None) or (init_phi_list is None):
        np_rng = np.random.default_rng(seed)
    else:
        np_rng = None
    dim_list = tuple(dim_list)
    assert (len(dim_list)>=2) and all(x>1 for x in dim_list)
    dim_total = int(np.prod(dim_list))
    assert (rho.ndim==2) and (rho.shape==(dim_total,dim_total))
    assert np.abs(rho-rho.T.conj()).max() < zero_eps
    EVL,EVC = np.linalg.eigh(rho)
    ind0 = EVL>zero_eps
    rank = ind0.sum()
    EVL = EVL[ind0]
    EVC = EVC[:,ind0]
    if init_coeffq is not None:
        if num_state is not None:
            assert num_state == init_coeffq.shape[0]
        else:
            num_state = init_coeffq.shape[0]
    if init_phi_list is not None:
        if num_state is not None:
            assert all(x.shape[0]==num_state for x in init_phi_list)
        else:
            num_state = init_phi_list[0].shape[0]
        assert len(init_phi_list)==len(dim_list)
        assert all(x.shape[1]==y for x,y in zip(init_phi_list,dim_list))
    if num_state is None:
        num_state = 3*dim_total
    assert num_state >= rank

    EVL_sqrt = np.sqrt(np.maximum(EVL,0))
    EVC_conj = EVC.conj().T.reshape((rank,)+dim_list)

    N0 = len(dim_list) #number of parties
    contract_expr = dict()
    tmp0 = [y for x0,x1 in enumerate(dim_list) for y in [[num_state,x1],[N0+1,x0]]]
    contract_expr['step0'] = opt_einsum.contract_expression(EVL_sqrt, [N0],
                EVC_conj, [N0]+list(range(N0)), [num_state], [N0+1], *tmp0, [N0,N0+1], constants=[0,1])
    contract_expr['step1'] = opt_einsum.contract_expression(EVL_sqrt, [N0], EVC_conj, [N0]+list(range(N0)),
                [num_state], [N0+1], [num_state,rank], [N0+1,N0], [N0+1]+list(range(N0)), constants=[0,1])
    if N0>2:
        tmp0 = (num_state,)+dim_list, [N0]+list(range(N0))
        for ind0 in range(N0):
            tmp1 = [((num_state,dim_list[x],),(N0,x)) for x in range(N0) if x!=ind0]
            tmp2 = [y for x in tmp1 for y in x]
            contract_expr[('step1',ind0)] = opt_einsum.contract_expression(*tmp0, *tmp2, [N0,ind0])
    tmp0 = [y for x0,x1 in enumerate(dim_list) for y in [[num_state,x1],[N0+1,x0]]]
    contract_expr['step2'] = opt_einsum.contract_expression(EVL_sqrt, [N0],
                    EVC_conj, [N0]+list(range(N0)), [num_state,rank], [N0+1,N0], *tmp0, [N0+1], constants=[0,1])

    history_info = []
    for _ in range(num_repeat):
        history_info_i = []
        if init_coeffq is None:
            coeffq = numqi.manifold.to_discrete_probability_softmax(np_rng.normal(size=num_state))
        else:
            coeffq = init_coeffq
            assert (np.all(coeffq)>=0) and (abs(coeffq.sum()-1)<zero_eps)
        if init_phi_list is None:
            phi_list = [numqi.manifold.to_sphere_quotient(np_rng.normal(size=(num_state,2*x)), is_real=False) for x in dim_list]
        else:
            phi_list = init_phi_list
            assert all(np.abs(np.linalg.norm(x,axis=1,ord=2)-1).max() < zero_eps for x in phi_list)
        coeffq_sqrt = np.sqrt(coeffq)

        for _ in range(maxiter):
            matM0 = contract_expr['step0'](coeffq_sqrt, *phi_list)
            matX = scipy.linalg.polar(matM0, side='left')[0].T.conj()
            # z0 = np.trace(matM0@matX)
            # assert np.abs(z0.imag) < 1e-12
            # print(z0.real)

            matM1 = contract_expr['step1'](coeffq_sqrt, matX)
            if N0==2:
                tmp0 = [scipy.linalg.svd(x, full_matrices=False) for x in matM1]
                phi_list = np.stack([x[0][:,0].conj() for x in tmp0]), np.stack([x[2][0].conj() for x in tmp0])
            else:
                for _ in range(maxiter_inner):
                    fidelity_list = np.zeros(N0, dtype=np.float64)
                    for ind0 in range(N0):
                        tmp0 = [phi_list[x] for x in range(N0) if x!=ind0]
                        tmp1 = contract_expr[('step1',ind0)](matM1, *tmp0).conj()
                        tmp1 = tmp1 / np.linalg.norm(tmp1, axis=1, ord=2, keepdims=True)
                        fidelity_list[ind0] = abs(np.vdot(tmp1.reshape(-1), phi_list[ind0].reshape(-1))/num_state)**2
                        phi_list[ind0] = tmp1
                    if (1-fidelity_list.min())<converge_eps_inner:
                        break
            # tmp0 = [y for x0,x1 in enumerate(phi_list) for y in [x1,[N0+1,x0]]]
            # z0 = opt_einsum.contract(matM1, [N0+1]+list(range(N0)), *tmp0, [N0+1])
            # assert np.abs(z0.imag).max() < 1e-12
            # print(z0.sum().real, z0.real)

            matM2 = contract_expr['step2'](matX, *phi_list).real
            coeffq_sqrt = matM2 / np.linalg.norm(matM2, ord=2)
            history_info_i.append(np.dot(coeffq_sqrt, matM2)**2)
            if (len(history_info_i)>1) and (abs(history_info_i[-1]-history_info_i[-2])<converge_eps):
                break
        history_info.append(1-np.array(history_info_i))
    history_info = max(history_info, key=lambda x: x[-1])
    ret = history_info[-1]
    if return_info:
        info = {
            'history_value': history_info,
            'coeffq': coeffq_sqrt**2,
            'phi_list': phi_list,
            'matX': matX,
        }
        ret = history_info[-1],info
    else:
        ret = history_info[-1]
    return ret

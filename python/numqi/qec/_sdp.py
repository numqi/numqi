import math
import functools
import itertools
import numpy as np
import scipy.special
import sympy
import cvxpy

# https://arxiv.org/abs/2408.10323v1 SDP bounds for quantum codes

def get_ijtp_range(n:int):
    # maps 4-tuple (i,j,t,p) to 1d indexing. eq(75)
    assert n>=1
    ret = []
    for i,j in itertools.product(range(n+1), repeat=2):
        for t in range(max(0,i+j-n), min(i,j)+1):
            for p in range(t+1):
                ret.append((i,j,t,p))
    return ret

def get_ak_range(n:int):
    # maps a tuple (a,k) to a 1d indexing. eq(84)
    ret = [(a,k) for a in range(n+1) for k in range(a,(n+a)//2+1)]
    return ret


def get_gamma_dict(ijtp_list):
    # eq(76)
    n = max(x[0] for x in ijtp_list)
    nf = math.factorial(n)
    hf0 = lambda a,b,c,d: (((nf // math.factorial(a)) // math.factorial(b)) // math.factorial(c)) // math.factorial(d) // math.factorial(n-a-b-c-d)
    ret = {(i,j,t,p):((3**(i+j-t)) * (2**(t-p)) * hf0(p,t-p,i-t,j-t)) for i,j,t,p in ijtp_list}
    return ret

def get_beta_dict(ijtp_list, ak_list):
    # eq(83)
    n = max(x[0] for x in ijtp_list)
    key_list = {(i-a,j-a,k-a,n-a,t-a) for i,j,t,_ in ijtp_list for a,k in ak_list}
    hf0 = lambda i,j,k,m,t,u: (-1)**(t-u)*math.comb(u,t)*math.comb(m-2*k,m-k-u)*math.comb(m-k-u,i-u)*math.comb(m-k-u,j-u)
    ret = {(i,j,k,m,t):sum(hf0(i,j,k,m,t,u) for u in range(min(m-k,i,j)+1)) for i,j,k,m,t in key_list if t>=0}
    ret = {k:v for k,v in ret.items() if v!=0} # most entry are zero
    return key_list,ret

def get_alpha_dict(ijtp_list, ak_list):
    # eq(83)
    n = max(x[0] for x in ijtp_list)
    _,beta_dict = get_beta_dict(ijtp_list, ak_list)
    key_list = [(i,j,t,p,a,k) for i,j,t,p in ijtp_list for a,k in ak_list]
    shift = {(a,t,p):min(0,(t-a-p+max(0,p+a-t))) for _,_,t,p,a,_ in key_list if t>=a}
    tmp0 = {(a,t,p):(sum((-1)**(a-g)*math.comb(a,g)*math.comb(t-a,p-g)*(2**(t-a-p+g-v)) for g in range(max(0,p+a-t),min(a,p)+1))) for (a,t,p),v in shift.items()}
    # "t-a-p+g" seems to be always positive
    ret = {(i,j,t,p,a,k):(beta_dict.get((i-a,j-a,k-a,n-a,t-a), 0) * (3**((i+j)/2-t)) * (2**(shift[(a,t,p)])) * tmp0[(a,t,p)]) for i,j,t,p,a,k in key_list if (t>=a)}
    # still a lots of zero
    return ret


def _MacWilliams_transform(expr0, expr1, sx, sy, num_qubit):
    ret = []
    for j in range(num_qubit+1):
        expr2 = sympy.poly((expr0**(num_qubit-j)) * (expr1**j), [sx,sy], domain=sympy.RR)
        tmp0 = {y:float(x) for x,y in zip(expr2.coeffs(), expr2.monoms())}
        ret.append([tmp0.get((num_qubit-y,y), 0) for y in range(num_qubit+1)])
    ret = np.ascontiguousarray(np.array(ret).T)
    return ret


def get_code_feasible_constraint(num_qubit:int, dimK:int, distance:int):
    r'''get the SDP constraint for the feasibility of a quantum code

    [arxiv-link](https://arxiv.org/abs/2408.10323v1) SDP bounds for quantum codes

    see eq(142)

    Parameters:
        num_qubit (int): the number of qubits
        dimK (int): the dimension of the code space
        distance (int): the distance of the code

    Returns:
        cvxX (dict): the variable for the code
        cvxA (cvxpy.Expression): the approximation of the quantum weight enumerator (Shor-Laflamme)
        cvxB (cvxpy.Expression): the approximation of the dual quantum weight enumerator (Shor-Laflamme)
        cvxS (cvxpy.Expression): the approximation of the quantum weight enumerator (Rains' shadow)
        constraint (list): the list of constraints
    '''
    ijtp_list = get_ijtp_range(num_qubit)
    ak_list = get_ak_range(num_qubit)
    gamma_dict = get_gamma_dict(ijtp_list)
    alpha_dict = get_alpha_dict(ijtp_list, ak_list)

    ind_zero = {(i,j,t,p) for i,j,t,p in ijtp_list if (t-p)%2==1}
    if dimK==1: #eq(145), all dimK=1 code are defined to be pure (eq8)
        tmp0 = set(range(1,distance))
        ind_zero |= {(i,j,t,p) for i,j,t,p in ijtp_list if any(x in tmp0 for x in (i,j,i+j-t-p))}
    ind_equal = dict()
    for ind0 in range(len(ijtp_list)):
        i0,j0,t0,p0 = ijtp_list[ind0]
        if (t0-p0)%2!=0:
            continue
        tmp0 = tuple(sorted([i0,j0,i0+j0-t0-p0]))
        for ind1 in range(ind0+1,len(ijtp_list)):
            i1,j1,t1,p1 = ijtp_list[ind1]
            if (t0-p0)!=(t1-p1):
                continue
            if tmp0==tuple(sorted([i1,j1,i1+j1-t1-p1])):
                tmp1 = (i0,j0,t0,p0)
                while tmp1 in ind_equal:
                    if tmp1 in ind_zero:
                        break
                    tmp1 = ind_equal[tmp1]
                if tmp1 in ind_zero:
                    ind_zero.add((i1,j1,t1,p1))
                else:
                    ind_equal[(i1,j1,t1,p1)] = tmp1
    tmp0 = ind_zero | set(ind_equal.keys()) | {(0,0,0,0)}
    assert (0,0,0,0) not in set(ind_equal.values())
    tmp1 = set(ind_equal.keys()) | set(ind_equal.values())
    assert all((x not in tmp1) for x in ind_zero)
    cvxX = {ijtp:cvxpy.Variable() for ijtp in ijtp_list if (ijtp not in tmp0)}
    cvxX[(0,0,0,0)] = 1
    for ijtp in ind_zero:
        cvxX[ijtp] = 0
    for k,v in ind_equal.items():
        cvxX[k] = cvxX[v]
    ## cvxA and cvxB are in the convention with SDP-paper
    cvxA = cvxpy.hstack([gamma_dict[(i,0,0,0)]*cvxX[(i,0,0,0)] for i in range(num_qubit+1)])
    sx = sympy.symbols('x')
    sy = sympy.symbols('y')
    cvxB = _MacWilliams_transform((sx+3*sy)/2, (sx-sy)/2, sx, sy, num_qubit) @ cvxA
    # cvxpy.sum(cvxA)==(2**num_qubit/dimK) #this is equal to K*B0=A0
    cvxS = _MacWilliams_transform((sx+3*sy)/2, (sy-sx)/2, sx, sy, num_qubit) @ cvxA
    constraint = [
        dimK*cvxB[:distance]==cvxA[:distance],
        dimK*cvxB[distance:]>=cvxA[distance:],
    ]
    if dimK==1: #eq143
        constraint += [cvxS[i]==0 for i in range(1-(num_qubit%2), num_qubit+1, 2)]
        constraint += [cvxS[i]>=0 for i in range((num_qubit%2), num_qubit+1, 2)]
    else:
        constraint.append(cvxS>=0)
    for k in range(num_qubit+1):
        tmp0 = sum(gamma_dict[(i,j,t,p)]*cvxX[(i,j,t,p)] for i,j,t,p in ijtp_list if (i+j-t-p)==k)
        constraint.append(tmp0 == (2**num_qubit/dimK) * gamma_dict[(k,0,0,0)] * cvxX[(k,0,0,0)])
    tmp0 = sorted([(a,k,i,j,t,p) for i,j,t,p in ijtp_list for a,k in ak_list])
    tmp0 = itertools.groupby(tmp0, key=lambda x: x[:4])
    akij_list = {k:[x[4:] for x in v] for k,v in tmp0}
    for a,k in ak_list:
        N0 = num_qubit+a-k
        indIJ = list(itertools.product(range(k,N0+1), repeat=2))
        tmp0 = [sum((alpha_dict[(i,j,t,p,a,k)]*cvxX[(i,j,t,p)]
                    if ((i,j,t,p,a,k) in alpha_dict) else 0) for t,p in akij_list[(a,k,i,j)]) for i,j in indIJ]
        tmp1 = [sum((alpha_dict[(i,j,t,p,a,k)]*(cvxX[(i+j-t-p,0,0,0)]-cvxX[(i,j,t,p)])
                    if ((i,j,t,p,a,k) in alpha_dict) else 0) for t,p in akij_list[(a,k,i,j)]) for i,j in indIJ]
        if (num_qubit+a-2*k+1)==1:
            constraint.append(tmp0[0]>=0)
            constraint.append(tmp1[0]>=0)
        else:
            constraint.append(cvxpy.reshape(cvxpy.hstack(tmp0), (N0-k+1,N0-k+1), order='C')>>0)
            constraint.append(cvxpy.reshape(cvxpy.hstack(tmp1), (N0-k+1,N0-k+1), order='C')>>0)
    return cvxX, cvxA, cvxB, cvxS, constraint


def is_code_feasible(num_qubit:int, dimK:int, distance:int, drop_constraint=None, solver=None):
    r'''check if a quantum code is feasible

    [arxiv-link](https://arxiv.org/abs/2408.10323v1) SDP bounds for quantum codes

    see eq(142)

    Parameters:
        num_qubit (int): the number of qubits
        dimK (int): the dimension of the code space
        distance (int): the distance of the code
        drop_constraint (list): the list of constraint to be dropped. if all constraints are kept, SDP problem probably raises NumericalError
        solver (str): the solver to be used, we find MOSEK is generally more stable

    Returns:
        ret (bool): if False, the code is definitely infeasible. if True, the code could be feasible
    '''
    cvxX, cvxA, cvxB, cvxS, constraint = get_code_feasible_constraint(num_qubit, dimK, distance)
    if drop_constraint is not None:
        tmp0 = {int(x) for x in drop_constraint}
        constraint = [constraint[x] for x in range(len(constraint)) if x not in tmp0]
    prob = cvxpy.Problem(cvxpy.Minimize(0), constraint)
    prob.solve(solver=solver) #solver='CLARABEL'
    ret = prob.status!='infeasible'
    # prob.solve(solver='CLARABEL')
    # prob.solve(solver='MOSEK', mosek_params={"MSK_DPAR_INTPNT_TOL_PFEAS": 1e-8})
    # prob.solve(solver='CVXOPT', verbose=True, max_iters=1000) #CVXOPT CLARABEL
    #cvxopt: singular KKT matrix
    return ret


def get_Krawtchouk_polynomial(q:int, k:int):
    r'''get the Krawtchouk polynomial

    [wiki-link](https://en.wikipedia.org/wiki/Kravchuk_polynomials)

    Parameters:
        q (int): the prime power
        k (int): the degree of the polynomial

    Returns:
        ret (np.ndarray): the Krawtchouk polynomial of shape `(k+1,k+1)`
    '''
    assert (q>=2) and (len(sympy.primefactors(q))==1), 'q must be prime power'
    assert k>=0, 'k must be positive integer'
    # row index (1,x,x^2,...,x^k)
    # column index (1,n,n^2,...,n^k)
    ret = np.zeros((k+1,k+1), dtype=np.float64)
    sx = sympy.symbols('x')
    sn = sympy.symbols('n')
    s1 = sympy.N(1)
    hf0 = lambda x,y: x*y
    hf_prod = lambda x: (functools.reduce(hf0, x[1:], x[0]) if len(x) else s1)
    for j in range(k+1):
        coeff = ((-q)**j) * ((q-1)**(k-j))
        tmp0 = sympy.poly(hf_prod([(sn-j-a) for a in range(k-j)])/scipy.special.factorial(k-j), sn, domain=sympy.RR)
        tmp1 = {y[0]:float(x) for x,y in zip(tmp0.coeffs(), tmp0.monoms())}
        coeffN = np.array([tmp1.get(a,0) for a in range(k+1)])
        tmp0 = sympy.poly(hf_prod([(sx-a) for a in range(j)])/scipy.special.factorial(j), sx, domain=sympy.RR)
        tmp1 = {y[0]:float(x) for x,y in zip(tmp0.coeffs(), tmp0.monoms())}
        coeffX = np.array([tmp1.get(a,0) for a in range(k+1)])
        ret += coeff * coeffX.reshape(-1,1) * coeffN
    return ret


def _get_Shor_weight_enumerator_dual_map(n:int, dimK:int):
    sx = sympy.symbols('x')
    sy = sympy.symbols('y')
    ret = dimK*_MacWilliams_transform((sx+3*sy)/2, (sx-sy)/2, sx, sy, n)
    return ret


def is_code_feasible_linear_programming(num_qubit:int, dimK:int, distance:int):
    r'''check if a quantum code is feasible using linear programming

    [arxiv-link](https://arxiv.org/abs/2408.10323v1) SDP bounds for quantum codes

    see eq(19) and eq(20)

    Parameters:
        num_qubit (int): the number of qubits
        dimK (int): the dimension of the code space
        distance (int): the distance of the code

    Returns:
        ret (bool): if False, the code is definitely infeasible. if True, the code could be feasible
        info (dict): the information of the code
    '''
    # https://arxiv.org/abs/2408.10323
    assert distance <= num_qubit
    cvxA = cvxpy.Variable(num_qubit+1)
    sx = sympy.symbols('x')
    sy = sympy.symbols('y')
    cvxB = dimK*_MacWilliams_transform((sx+3*sy)/2, (sx-sy)/2, sx, sy, num_qubit) @ cvxA
    cvxS = (dimK*dimK)*_MacWilliams_transform((sx+3*sy)/2, (sy-sx)/2, sx, sy, num_qubit) @ cvxA
    if dimK==1:
        constraint = [
            cvxA[0]==1,
            cvxA>=0,
            cvxA[1:distance]==0,
            cvxB>=0,
            cvxS>=0,
            cvxpy.sum(cvxA)==2**num_qubit,
        ]
    else: #eq(19)
        constraint = [
            cvxA[0]==1,
            cvxA>=0,
            cvxS>=0,
            cvxB[:distance]==cvxA[:distance],
            (cvxB-cvxA)[distance:]>=0,
        ]
    prob = cvxpy.Problem(cvxpy.Minimize(0), constraint)
    prob.solve()
    if prob.status=='infeasible':
        ret = False,dict()
    else:
        ret = True, dict(A=cvxA.value, B=cvxB.value, S=cvxS.value)
    return ret

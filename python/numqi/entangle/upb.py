import math
import numpy as np
import scipy.linalg

hf_is_prime = lambda n: (n>=2) and ((n==2) or ((n%2==1) and all(n%x>0 for x in range(3, int(math.sqrt(n))+1, 2))))
# TODO sympy.isprime()
# https://stackoverflow.com/q/18114138
# True: 2,3,5,7
# False: -1,0,1,4,6


def upb_product(upb):
    ret = upb[0]
    for x in upb[1:]:
        ret = (ret[:,:,np.newaxis]*x[:,np.newaxis]).reshape(ret.shape[0], -1)
    return ret


def load_upb(kind, args=None, return_product=False, return_bes=False, ignore_warning=False):
    r'''load unextendible product basis (UPB)

    see http://www.qetlab.com/UPB

    TODO John2^8 John2^4k CJ4k1 CJBip CJBip46 Feng2x2x5 Feng4m2 Feng2x2x2x4 Feng2x2x2x2x5 AlonLovasz

    Parameters:
        kind (str): UPB name, case insensitive, including:
            tiles pyramid feng4x4 min4x4 quadres genshifts feng2x2x2x2 sixparam gentiles1 gentiles2 john2^8
        args (int, tuple): UPB arguments
        return_product (bool): return product of UPB
        return_bes (bool): return bounding entangled states

    Returns:
        upb (list of np.ndarray, np.ndarray): If `return_product=False`, the length of list is the number of partites. Each np.ndarray is of shape `(N,dimI)`
            where `N` is the number of product states, `dimI` is the dimension of the i-th partite. If `return_product=True`,
            the return value is a `np.ndarray` of shape (N,dim0*dim1*dim2*...)
        bes (np.ndarray): only when `return_bes=True`, the return value is a `np.ndarray` of shape `(dim0*dim1*dim2*...,dim0*dim1*dim2*...)`
    '''
    assert kind is not None
    kind = str(kind).lower()
    s12 = 1/np.sqrt(2)
    s13 = 1/np.sqrt(3)
    if kind=='tiles':
        # C.H. Bennett, D.P. DiVincenzo, T. Mor, P.W. Shor, J.A. Smolin, and B.M. Terhal.
        # Unextendible product bases and bound entanglement. Phys. Rev. Lett. 82, 5385-5388, 1999.
        tmp0 = [[1,0,0],[s12,-s12,0],[0,0,1],[0,s12,-s12],[s13,s13,s13]]
        tmp1 = [[s12,-s12,0],[0,0,1],[0,s12,-s12],[1,0,0],[s13,s13,s13]]
        upb = [np.array(tmp0), np.array(tmp1)]
    elif kind=='pyramid':
        h = np.sqrt(1+np.sqrt(5))/2
        tmp0 = np.array([(np.cos(2*np.pi*x/5), np.sin(2*np.pi*x/5), h) for x in range(5)])
        tmp1 = (2/np.sqrt(5+np.sqrt(5)))*tmp0
        upb = [tmp1,tmp1[[0,2,4,1,3]]]
    elif kind=='feng4x4':
        # K. Feng. Unextendible product bases and 1-factorization of complete graphs. Discrete Applied Mathematics, 154:942-949, 2006.
        tmp0 = np.array([[0,s13,s13,s13],[s13,0,-s13,s13],[s13,s13,0,-s13],[s13,-s13,s13,0]])
        tmp1 = np.concatenate([np.eye(4),tmp0], axis=0)
        tmp2 = np.zeros((8,4), dtype=np.float64)
        tmp2[7] = np.array([0,s13,s13,s13])
        tmp2[[0,6,5,3]] = np.eye(4)
        tmp2[4] = np.array([s13,-s13,s13,0])
        tmp2[1] = np.array([s13,0,-s13,s13])
        tmp2[2] = np.array([s13,s13,0,-s13])
        upb = [tmp1, tmp2]
    elif kind=='min4x4':
        s16 = 1/np.sqrt(6)
        s112 = 1/np.sqrt(12)
        tmp0 = np.array([[s112,-3*s112,s112,s112],[1,0,0,0],[0,s16,2*s16,s16],
                [s12,0,0,-s12],[0,1,0,0],[3*s112,s112,-s112,s112],[0,s12,s12,0],[0,0,1,0]])
        tmp1 = np.zeros((8,4), dtype=np.float64)
        tmp1[0] = np.array([0,1,-3-np.sqrt(2),-1-np.sqrt(2)])/np.sqrt(15+8*np.sqrt(2))
        tmp1[1] = np.array([1,0,0,0])
        tmp1[2] = np.array([1,0,np.sqrt(2)-1,1])/np.sqrt(5-2*np.sqrt(2))
        tmp1[3] = np.array([0,1,0,0])
        tmp1[4] = np.array([-1,1+np.sqrt(2),0,1])/np.sqrt(5+2*np.sqrt(2))
        tmp1[5] = np.array([0,0,1,0])
        tmp1[6] = np.array([1,1,1,-np.sqrt(2)])/np.sqrt(5)
        tmp1[7] = np.array([-1,1+np.sqrt(2),0,1])/np.sqrt(5+2*np.sqrt(2))
        upb = [tmp0,tmp1]
    elif kind=='quadres':
        # D.P. DiVincenzo, T. Mor, P.W. Shor, J.A. Smolin, and B.M. Terhal.
        # Unextendible product bases, uncompletable product bases and bound entanglement. Commun. Math. Phys. 238, 379-410, 2003.
        # import sympy
        # sympy.sieve.extend(50)
        # allowed_dim = [(x+1)//2 for x in sympy.sieve._list[1:] if x%4==1]
        # [3, 7, 9, 15, 19, 21]
        dim = int(args)
        assert ((dim%2)==1) and hf_is_prime(2*dim-1)
        p = 2*dim-1
        q = np.array(sorted(set(np.mod(np.arange(1, p//2+1, dtype=np.int64)**2, p).tolist())), dtype=np.int64)
        s = np.array(sorted(set(range(1,p)) - set(q.tolist())))
        sm = np.exp(2j*np.pi*q/p).sum().real
        N = max(-sm, 1+sm)
        F = fourier_matrix(p)
        F[0] *= np.sqrt(N)
        tmp0 = F[[0,*q]].T
        tmp1 = F[[0,*np.mod(s[0]*q,p)]].T
        upb = [x/np.linalg.norm(x,axis=1,keepdims=True) for x in [tmp0,tmp1]]
    elif kind=='genshifts':
        # D.P. DiVincenzo, T. Mor, P.W. Shor, J.A. Smolin, and B.M. Terhal.
        # Unextendible product bases, uncompletable product bases and bound entanglement. Commun. Math. Phys. 238, 379-410, 2003.
        # TODO replace args with k
        dim = int(args)
        assert dim%2==1
        k = (dim+1)//2
        tmp0 = [0,*list(reversed(range(1,k+1))),*list(range(k+1,2*k))]
        tmp1 = np.arange(2*k)[tmp0]*(np.pi/(2*k))
        # [1,k+1:-1:2,k+2:end]
        tmp2 = np.stack([np.cos(tmp1),np.sin(tmp1)], axis=1)
        tmp3 = (np.concatenate([np.array([0],dtype=np.int64),np.roll(np.arange(1,2*k), x)], axis=0) for x in range(dim))
        upb = [tmp2[x] for x in tmp3]
    elif kind=='feng2x2x2x2':
        # K. Feng. Unextendible product bases and 1-factorization of complete graphs. Discrete Applied Mathematics, 154:942-949, 2006.
        b1 = np.eye(2)
        b2 = np.array([[1,1],[1,-1]])/np.sqrt(2)
        b3 = np.array([[0.5, -np.sqrt(3)/2], [np.sqrt(3)/2, 0.5]])
        tmp0 = np.stack([b1[0],b1[1],b1[0],b2[0],b2[1],b2[0]], axis=0)
        tmp1 = np.stack([b1[0],b2[0],b1[1],b1[1],b2[1],b1[0]], axis=0)
        tmp2 = np.stack([b1[0],b2[0],b3[0],b2[1],b3[1],b1[1]], axis=0)
        tmp3 = np.stack([b1[0],b2[0],b3[0],b3[1],b1[1],b2[1]], axis=0)
        upb = [tmp0,tmp1,tmp2,tmp3]
    elif kind=='sixparam':
        # Section IV.A of DMSST03
        if args is None:
            np_rng = np.random.default_rng()
            para = np_rng.uniform(0, 2*np.pi, size=6)
        else:
            para = np.asarray(args) #gammaA thetaA phiA gammaB thetaB phiB
            assert para.shape==(6,)
        if not ignore_warning:
            tmp0 = para[[0,1,3,4]]
            if np.any(np.abs(np.cos(tmp0))<1e-10) or np.any(np.abs(np.sin(tmp0))<1e-10):
                # #gamma/theta cannot be multiple of np.pi/2
                print('[WARNING] pyqet.entangle.load_upb(sixparam) is NOT a upb for current parameter')
        gammaA,thetaA,phiA,gammaB,thetaB,phiB = para
        NA = np.maximum(np.sqrt(np.cos(gammaA)**2 + np.sin(gammaA)**2 * np.cos(thetaA)**2), 1e-12)
        NB = np.maximum(np.sqrt(np.cos(gammaB)**2 + np.sin(gammaB)**2 * np.cos(thetaB)**2), 1e-12)
        tmp0 = np.stack([
            [1,0,0],[0,1,0],[np.cos(thetaA),0,np.sin(thetaA)],
            [np.sin(gammaA)*np.sin(thetaA),np.cos(gammaA)*np.exp(1j*phiA),-np.sin(gammaA)*np.cos(thetaA)],
            [0,np.sin(gammaA)*np.cos(thetaA)*np.exp(1j*phiA)/NA,np.cos(gammaA)/NA]
        ], axis=0)
        tmp1 = np.stack([
            [0,1,0],
            [np.sin(gammaB)*np.sin(thetaB),np.cos(gammaB)*np.exp(1j*phiB),-np.sin(gammaB)*np.cos(thetaB)],
            [1,0,0], [np.cos(thetaB),0,np.sin(thetaB)],
            [0,np.sin(gammaB)*np.cos(thetaB)*np.exp(1j*phiB)/NB,np.cos(gammaB)/NB],
        ], axis=0)
        upb = [tmp0,tmp1]
    elif kind=='gentiles1':
        dim = int(args)
        assert (dim%2==0) and (dim>=4)
        upb = [[], []]
        matI = np.eye(dim)
        for m in range(1,dim//2):
            tmp0 = np.concatenate([np.exp((4j*np.pi*m/dim)*np.arange(dim//2)), np.zeros(dim//2)])
            matW = scipy.linalg.circulant(tmp0/np.sqrt(dim/2))
            upb[0].append(np.stack([matI,matW], axis=2).reshape(dim,dim*2))
            upb[1].append(np.stack([np.roll(matW,-1,axis=1),matI], axis=2).reshape(dim,dim*2))
        upb[0].append(np.ones((dim,1))/np.sqrt(dim)) #(dim,dim*dim-2*dim+1)
        upb[1].append(np.ones((dim,1))/np.sqrt(dim))
        upb = [np.concatenate(x,axis=1).T for x in upb]
    elif kind=='gentiles2':
        assert len(args)==2
        dimA = int(args[0]) #m
        dimB = int(args[1]) #n
        assert (dimA>=3) and (dimB>=4) and (dimB>=dimA)
        matIA = np.eye(dimA)
        matIB = np.eye(dimB)
        upb0 = [
            (matIA - np.roll(matIA, -1, axis=1))/np.sqrt(2),
            np.repeat(matIA, dimB-3, axis=1),
            np.ones((dimA,1))/np.sqrt(dimA),
        ] #(dimA,dimA*dimB-2*dimA+1)
        upb1 = [matIB[:,:dimA]]
        for j in range(dimA):
            tmp0 = np.zeros((dimB,dimB-3), dtype=np.complex128)
            tmp1 = np.exp((2j*np.pi/(dimB-2))*np.arange(dimA-2)[:,np.newaxis]*np.arange(1,dimB-2))
            tmp0[(np.arange(dimA-2)+j+1)%dimA] = tmp1
            tmp1 = np.exp((2j*np.pi/(dimB-2)) * np.arange(dimA-2,dimB-2)[:,np.newaxis]*np.arange(1,dimB-2))
            tmp0[np.arange(dimA,dimB)[:,np.newaxis], np.arange(dimB-3)] = tmp1
            upb1.append(tmp0/np.sqrt(dimB-2))
        upb1.append(np.ones((dimB,1))/np.sqrt(dimB))
        upb = [np.concatenate(x,axis=1).T for x in [upb0,upb1]]
    elif kind=='john2^8':
        # N. Johnston. The minimum size of qubit unextendible product bases.
        # In Proceedings of the 8th Conference on the Theory of Quantum Computation, Communication and Cryptography (TQC 2013).
        # E-print: arXiv:1302.1604 [quant-ph], 2013.
        assert False, 'not implemented'
    else:
        assert False
    if return_product:
        upb = upb_product(upb)
    ret = (upb,upb_to_bes(upb)) if return_bes else upb
    return ret


def upb_to_bes(upb):
    # unextendible product basis (UPB), bound entangled state (BES)
    if not isinstance(upb, np.ndarray):
        upb = upb_product(upb)
    ret = np.eye(upb.shape[1]) - upb.T @ upb.conj()
    ret /= np.trace(ret)
    return ret


def fourier_matrix(dim):
    w = np.exp(2j*np.pi/dim)
    tmp0 = np.arange(dim)
    ret = (w**(tmp0[:,np.newaxis]*tmp0))/np.sqrt(dim)
    return ret

import itertools
import functools
import numpy as np
import scipy.linalg

from ..gellmann import matrix_to_gellmann_basis, gellmann_basis_to_matrix
from ..random import get_numpy_rng

def build_matrix_with_index_value(dim0, dim1, index_value):
    value = np.array([x[2] for x in index_value])
    ret = np.zeros((dim0,dim1), dtype=value.dtype)
    ind0 = [x[0] for x in index_value]
    ind1 = [x[1] for x in index_value]
    ret[ind0, ind1] = value
    return ret


def is_vector_linear_independent(np0, field, zero_eps=1e-7):
    assert field in {'real','complex'}
    assert np0.ndim>=2
    np0 = np0.reshape(np0.shape[0], -1)
    if (field=='real') and np.iscomplexobj(np0):
        np0 = np.concatenate([np0.real, np0.imag], axis=1)
    if np0.shape[0]>np0.shape[1]:
        ret = False
    else:
        U = scipy.linalg.lu(np0.conj() @ np0.T)[2]
        ret = np.abs(np.diag(U)).min() > zero_eps
    return ret


def is_vector_space_equivalent(space0, space1, field, zero_eps=1e-10):
    assert field in {'real','complex'}
    assert (space0.ndim>=2) and (space1.ndim>=2)
    if space0.ndim>2:
        space0 = space0.reshape(space0.shape[0], -1)
    if space1.ndim>2:
        space1 = space1.reshape(space1.shape[0], -1)
    assert space0.shape[1]==space1.shape[1]
    tag0 = np.iscomplexobj(space0)
    tag1 = np.iscomplexobj(space1)
    # assert tag0==tag1, 'space0 and space1 must be all real, or all complex'
    if (tag0 or tag1) and (field=='real'):
        space0 = np.concatenate([space0.real, space0.imag], axis=1)
        space1 = np.concatenate([space1.real, space1.imag], axis=1)
    x = np.linalg.lstsq(space0.T, space1.T, rcond=None)[0]
    ret = np.abs(x.T @ space0 - space1).max() < zero_eps
    if ret:
        x = np.linalg.lstsq(space1.T, space0.T, rcond=None)[0]
        ret = np.abs(x.T @ space1 - space0).max() < zero_eps
    return ret


def reduce_vector_space(np0, zero_eps=1e-10):
    # span_R(R^n) span_C(C^n)
    assert np0.ndim==2
    _,S,V = np.linalg.svd(np0, full_matrices=False)
    ret = V[:(S>zero_eps).sum()]
    return ret


def get_vector_orthogonal_basis(np0, tag_reduce=True, zero_eps=1e-10):
    # span_R(R^n) span_C(C^n)
    assert np0.ndim==2
    if tag_reduce:
        np0 = reduce_vector_space(np0, zero_eps)
    else:
        assert np.abs(np0.conj() @ np0.T - np.eye(np0.shape[0])).max() < np.sqrt(zero_eps)
    N0,N1 = np0.shape
    if N0==N1:
        ret = np.zeros((0,N1), dtype=np0.dtype)
    else:
        _,EVC = np.linalg.eigh(np.eye(N1) - np0.T @ np0.conj())
        ret = EVC[:,N0:].T
    return ret


def find_closest_vector_in_space(space, vec, field):
    # field==None: span_R(R)=R span_C(C)=C span_C(C)=R span_C(R)=C
    assert space.ndim>=2
    if space.ndim>2:
        space = space.reshape(space.shape[0], -1)
    vec = vec.reshape(-1)
    assert space.shape[1]==vec.shape[0]
    assert field in {'real','complex'}
    tag0 = np.iscomplexobj(space)
    tag1 = np.iscomplexobj(vec)
    key = ('R' if (field=='real') else 'C') + ('C' if tag0 else 'R') + ('C' if tag1 else 'R')
    if key in {'RRR', 'CCC', 'CRR', 'CRC', 'CCR'}:
        coeff,residuals,_,_ = np.linalg.lstsq(space.T, vec, rcond=None)
    elif key in {'RCR', 'RCC', 'RRC'}:
        tmp0 = np.concatenate([space.real, space.imag], axis=1)
        tmp1 = np.concatenate([vec.real, vec.imag], axis=0)
        coeff,residuals,_,_ = np.linalg.lstsq(tmp0.T, tmp1, rcond=None)
    ret = coeff, residuals.item()
    return ret


def get_matrix_orthogonal_basis(matrix_subspace, field, zero_eps=1e-10):
    r'''return matrix orthogonal basis

    Parameters:
        matrix_subspace (np.ndarray): (N0,N1,N2), `N0` matrices with each matrix of shape (N1,N2)
        field (str): 'real' or 'complex'
        zero_eps (float): zero threshold

    Returns:
        basis (np.ndarray): (N3,N1,N2), `N3` basis with each basis of shape (N1,N2). `N3` (<N0) is the rank of `matrix_subspace`
            Each basis is orthogonal to each other $Tr(A^\dag B)=0$ and normalized in Frobenius norm $Tr(A^\dag A)=1$
        basis_orth (np.ndarray): (N4,N1,N2), `N3` basis with each basis of shape (N1,N2)
        space_char (str): R_T C_T R C C_H R_cT R_c
            R_T: span_R(R_T^nn), real symmetric matrix over real field
            C_T: span_C(C_T) span_C(R_T^nn), real/complex symmetric matrix over complex field
            R: span_R(R^mn), real matrix over real field
            C: span_C(C^mn) span_C(R^mn), real/complex matrix over complex field
            C_H: span_R(C_H^nn), complex hermitian matrix over complex field
            R_cT: span_R(C_T^nn), complex symmetric matrix over real field
            R_c: span_R(C^mn), complex matrix over real field
    '''
    np0 = matrix_subspace #alias for short
    assert np0.ndim==3
    assert field in {'real','complex'}
    np.iscomplexobj(np0)
    N0,N1,N2 = np0.shape
    is_symmetric = (N1==N2) and (np.abs(np0-np0.transpose(0,2,1)).max() < zero_eps)
    is_anti_symmetric = (N1==N2) and (np.abs(np0+np0.transpose(0,2,1)).max() < zero_eps)
    is_hermitian = (N1==N2) and (np.abs(np0-np0.transpose(0,2,1).conj()).max() < zero_eps)
    if not np.iscomplexobj(np0):
        if is_symmetric: # span_R(R_T^nn) span_C(R_T^nn)
            N3 = (N1*(N1-1))//2
            tmp0 = matrix_to_gellmann_basis(np0).real
            tmp1 = [0,N3,2*N3,N1*N1]
            aS,aA,aDI = tmp0[:,:N3],tmp0[:,N3:(2*N3)],tmp0[:,(2*N3):]
            assert np.abs(aA).max() < zero_eps
            tmp0 = np.concatenate([aS,aDI], axis=1)
            tmp1 = reduce_vector_space(tmp0, zero_eps)
            tmp2 = get_vector_orthogonal_basis(tmp1, tag_reduce=False)
            ret = []
            for x in [tmp1,tmp2]:
                tmp3 = np.concatenate([x[:,:N3], np.zeros((x.shape[0],N3),dtype=x.dtype),x[:,N3:]], axis=1)
                ret.append(gellmann_basis_to_matrix(tmp3).real)
            ret = ret[0],ret[1], ('R_T' if (field=='real') else 'C_T')
        elif is_anti_symmetric:
            assert False, 'not implemented yet'
        else: # span_R(R^mn) span_C(R^mn)
            basis = reduce_vector_space(np0.reshape(N0,N1*N2), zero_eps)
            basis_orth = get_vector_orthogonal_basis(basis, tag_reduce=False)
            ret = basis.reshape(-1, N1, N2), basis_orth.reshape(-1, N1, N2), ('R' if (field=='real') else 'C')
    else:
        if (field=='real') and is_hermitian: #span_R(C_H^nn)
            tmp0 = matrix_to_gellmann_basis(np0).real
            tmp1 = reduce_vector_space(tmp0, zero_eps)
            tmp2 = get_vector_orthogonal_basis(tmp1, tag_reduce=False)
            basis = gellmann_basis_to_matrix(tmp1)
            basis_orth = gellmann_basis_to_matrix(tmp2)
            ret = basis,basis_orth,'C_H'
        elif (field=='real') and is_symmetric: #span_R(C_T^nn)
            N3 = (N1*(N1-1))//2
            tmp0 = matrix_to_gellmann_basis(np0)
            tmp1 = [0,N3,2*N3,N1*N1]
            aS,aA,aDI = tmp0[:,:N3],tmp0[:,N3:(2*N3)],tmp0[:,(2*N3):]
            assert np.abs(aA).max() < zero_eps
            tmp0 = np.concatenate([aS.real,aDI.real,aS.imag,aDI.imag], axis=1)
            tmp1 = reduce_vector_space(tmp0, zero_eps)
            tmp2 = get_vector_orthogonal_basis(tmp1, tag_reduce=False)
            ret = []
            for x in [tmp1,tmp2]:
                tmp3 = x[:,:((N1*N1+N1)//2)] + 1j*x[:,((N1*N1+N1)//2):]
                tmp3 = np.concatenate([tmp3[:,:N3], np.zeros((x.shape[0],N3),dtype=x.dtype),tmp3[:,N3:]], axis=1)
                tmp3 = gellmann_basis_to_matrix(tmp3)
                ret.append(np.block([[tmp3.real,-tmp3.imag],[tmp3.imag,tmp3.real]]))
            ret = ret[0],ret[1],'R_cT'
        elif field=='real': #span_R(C^mn)
            tmp0 = np.concatenate([np0.real,np0.imag], axis=2).reshape(N0, 2*N1*N2)
            tmp1 = reduce_vector_space(tmp0, zero_eps)
            tmp2 = get_vector_orthogonal_basis(tmp1, tag_reduce=False)
            ret = []
            for x in [tmp1,tmp2]:
                tmp3 = x.reshape(-1, N1, 2*N2)
                tmp3r,tmp3i = tmp3[:,:,:N2],tmp3[:,:,N2:]
                ret.append(np.block([[tmp3r,-tmp3i],[tmp3i,tmp3r]]))
            ret = ret[0],ret[1],'R_c'
        elif (field=='complex') and is_symmetric: #span_C(C_T)
            N3 = (N1*(N1-1))//2
            tmp0 = matrix_to_gellmann_basis(np0)
            tmp1 = [0,N3,2*N3,N1*N1]
            aS,aA,aDI = tmp0[:,:N3],tmp0[:,N3:(2*N3)],tmp0[:,(2*N3):]
            assert np.abs(aA).max() < zero_eps
            tmp0 = np.concatenate([aS,aDI], axis=1)
            tmp1 = reduce_vector_space(tmp0, zero_eps)
            tmp2 = get_vector_orthogonal_basis(tmp1, tag_reduce=False)
            ret = []
            for x in [tmp1,tmp2]:
                tmp3 = np.concatenate([x[:,:N3], np.zeros((x.shape[0],N3),dtype=x.dtype),x[:,N3:]], axis=1)
                ret.append(gellmann_basis_to_matrix(tmp3))
            ret = ret[0],ret[1],'C_T'
        else: #span_C(C^mn)
            tmp0 = reduce_vector_space(np0.reshape(N0,-1), zero_eps)
            tmp1 = get_vector_orthogonal_basis(tmp0, tag_reduce=False)
            ret = tmp0.reshape(-1,N1,N2), tmp1.reshape(-1,N1,N2), 'C'
    return ret


def get_matrix_subspace_example(key, arg=None):
    # <X,Z>
    assert key in {'XZ_R','XZ_C','0error-eq537','0error-eq524','hierarchy-ex1','hierarchy-ex3','hierarchy-ex5'}
    hf0 = lambda x,dim0,dim1: np.stack([build_matrix_with_index_value(dim0,dim1,y) for y in x])
    if key=='XZ_R':
        tmp0 = np.stack([
            np.array([[0,1], [1,0]]),
            np.array([[1,0], [0,-1]]),
        ], axis=0)
        ret = tmp0,'real'
    elif key=='XZ_C':
        tmp0 = np.stack([
            np.array([[0,1], [1,0]]),
            np.array([[1,0], [0,-1]]),
        ], axis=0)
        ret = tmp0,'complex'
    elif key=='0error-eq524':
        # doi.org/10.1007/978-3-319-42794-2 quantum zero-error information theory eq-5.24
        #at least rank=2
        tmp0 = hf0([
            [(0,0,1), (1,1,-1)],
            [(2,2,1), (3,3,-1)],
            [(2,0,1), (3,1,-1)],
            [(0,2,1), (1,3,1)],
            [(3,0,1), (0,3,-1)],
            [(1,0,1), (2,1,-np.sqrt(2)), (3,2,1)],
            [(0,1,1), (1,2,np.sqrt(2)), (2,3,1)],
            [(1,0,1), (3,2,-1), (0,1,-1), (2,3,1)],
        ], 4, 4)
        ret = tmp0, 'real'
        # hierarchy(rank=2, k=1): True
        # DetectMatrixSpaceRank(rank=2): loss=0
    elif key in {'hierarchy-ex1','hierarchy-ex3'}:
        # https://arxiv.org/abs/2210.16389v1 example1 example3
        #at least rank=2
        tmp0 = [
            [(0,0,1), (1,1,1), (2,2,1), (3,3,1)],
            [(0,0,1), (1,1,-1), (2,2,1), (3,3,-1)],
            [(0,1,1), (1,2,1), (2,3,1)],
            [(1,0,1), (2,1,1), (3,2,1)],
            [(0,1,1), (1,2,2), (2,3,3)],
            [(1,0,1), (2,1,2), (3,2,3)],
            [(0,2,1), (1,3,1)],
            [(2,0,1), (3,1,1.0)],
            [(0,0,1/2), (1,1,1/2), (2,2,-1/2), (3,3,-1/2)],
        ]
        if key=='hierarchy-ex1':
            tmp0 = hf0(tmp0[:-1], 4, 4)
            ret = tmp0, 'complex'
            # hierarchy(rank=2, k=1): True
            # DetectMatrixSpaceRank(rank=2): loss=0
        else:
            tmp0 = hf0(tmp0, 4, 4)
            ret = tmp0, 'complex'
            # hierarchy(rank=2, k=2): False
            # hierarchy(rank=2, k=3): True
            # DetectMatrixSpaceRank(rank=2): loss=0
    elif key=='hierarchy-ex5':
        tmp0 = [
            [(0,0,1), (1,1,1), (2,2,1), (3,3,1)],
            [(0,1,1), (1,2,1), (2,3,1), (3,0,1)],
            [(0,2,1), (1,3,1), (2,0,1), (3,1,-1.0)],
        ]
        tmp0 = hf0(tmp0, 4, 4)
        ret = tmp0, 'complex'
    elif key=='0error-eq537':
        # https://arxiv.org/abs/0906.2527v1 lemma3
        # doi.org/10.1007/978-3-319-42794-2 quantum error-error information theory eq-5.37
        if arg is None:
            theta = np.random.default_rng().uniform(0, np.pi/2)
        else:
            theta = float(arg)
        ct = np.cos(theta)
        st = np.sin(theta)
        npA = hf0([
            [(0,0,1),(1,1,1)],
            [(2,2,1),(3,3,1)],
            [(2,0,1),(0,2,-1)],
            [(3,0,1),(0,3,1)],
            [(1,3,1),(3,1,1)],
            [(0,1,ct),(2,3,st),(1,2,-1)],
            [(1,0,ct),(3,2,st),(2,1,-1)],
            [(0,1,st),(2,3,-ct),(1,0,st),(3,2,-ct)],
        ], 4, 4)
        npB = hf0([
            [(0,0,1),(1,1,1)],
            [(2,2,1),(3,3,1)],
            [(2,0,1),(0,2,1)],
            [(3,0,1),(0,3,1)],
            [(1,3,1),(3,1,-1)],
            [(0,1,ct),(2,3,st),(1,2,-1)],
            [(1,0,ct),(3,2,st),(2,1,-1)],
            [(0,1,st),(2,3,-ct),(1,0,st),(3,2,-ct)],
        ], 4, 4)
        npAB = np.stack([np.kron(x,y) for x in npA for y in npB], axis=0)
        npL = (np.eye(4)/2).reshape(-1)
        npR = (np.diag([1,-1,1,-1])/2).reshape(-1)
        ret = npA,npB,npAB,npL,npR,'real'
        # no rank-1 in orthogonal(A)
        # no rank-1 in orthogonal(B)
        # has rank-1 in orthogonal(AB)
        # npL @ npAB @ npR == 0
    return ret


def get_hermite_channel_matrix_subspace(matrix_space, zero_eps=1e-10):
    matrix_space = reduce_vector_space(matrix_space.reshape(matrix_space.shape[0], -1), zero_eps).reshape(-1, matrix_space.shape[1], matrix_space.shape[2])
    N0,dim,_ = matrix_space.shape
    # assert is_linear_independent(matrix_space.reshape(N0,-1))
    tmp0 = matrix_space.reshape(N0, -1).T
    tmp1 = np.eye(dim).reshape(-1)
    z0 = np.linalg.lstsq(tmp0, tmp1, rcond=None)[0]
    assert np.abs(tmp0@z0-tmp1).max() < zero_eps
    tmp1 = matrix_space.transpose(0,2,1).conj().reshape(N0,-1).T
    z0 = np.linalg.lstsq(tmp0, tmp1, rcond=None)[0]
    assert np.abs(tmp0 @ z0 - tmp1).max() < zero_eps

    tmp0 = matrix_space + matrix_space.transpose(0,2,1).conj()
    tmp1 = 1j*(matrix_space - matrix_space.transpose(0,2,1).conj())
    tmp2 = matrix_to_gellmann_basis(np.concatenate([tmp0,tmp1], axis=0))
    matrix_space_q = scipy.linalg.qr(tmp2.real.T, pivoting=True, mode='economic')[0].T[:N0]
    matrix_space_hermite = gellmann_basis_to_matrix(matrix_space_q)
    return matrix_space_hermite


# TODO maybe add "kwargs"
def get_completed_entangled_subspace(dim_tuple, kind, seed=None):
    kind = kind.lower()
    assert kind in {'quant-ph/0409032', 'quant-ph/0405077'}
    dim_tuple = tuple(int(x) for x in dim_tuple)
    assert (len(dim_tuple)>=2) and all(x>=1 for x in dim_tuple)
    if kind=='quant-ph/0409032':
        # A completely entangled subspace of maximal dimension
        # https://arxiv.org/abs/quant-ph/0409032
        assert len(dim_tuple)==3, 'check arxiv paper for more general implementation' #TODO
        dimA,dimB,dimC = dim_tuple
        tmp0 = [(a,b,x-a-b) for x in range(dimA+dimB+dimC-2)
                    for a in range(max(0,x+2-dimB-dimC),min(dimA,x+1))
                    for b in range(max(0,x-a-dimC+1),min(dimB,x-a+1))]
        tmp1 = ((x[0]+x[1]+x[2],x) for x in tmp0)
        hf0 = lambda x: x[0]
        abc_list = [[z[1] for z in y] for x,y in itertools.groupby(sorted(tmp1, key=hf0), key=hf0)]
        tmp0 = sorted({((y,z) if y<z else (z,y)) for x in abc_list for y in x for z in x if y!=z})
        z0 = np.zeros((len(tmp0),dimA,dimB,dimC), dtype=np.int8)
        tmp1 = np.array([x[0] for x in tmp0])
        z0[np.arange(len(tmp1)), tmp1[:,0], tmp1[:,1], tmp1[:,2]] = 1
        tmp1 = np.array([x[1] for x in tmp0])
        z0[np.arange(len(tmp1)), tmp1[:,0], tmp1[:,1], tmp1[:,2]] = -1
        tmp2 = z0.reshape(-1,dimA,dimB*dimC).astype(np.float64)
        matrix_subspace,matrix_subspace_orth,space_char = get_matrix_orthogonal_basis(tmp2, field='complex')
        ret_other = z0
    elif kind=='quant-ph/0405077':
        # On the maximal dimension of a completely entangled subspace for finite level quantum systems
        # https://arxiv.org/abs/quant-ph/0405077
        np_rng = get_numpy_rng(seed)
        dim_tuple = tuple(int(x) for x in dim_tuple)
        N0 = sum(dim_tuple) - len(dim_tuple) + 1
        assert all(x>1 for x in dim_tuple) and len(dim_tuple)>=2
        tmp0 = np_rng.normal(0,1,size=N0) + np_rng.normal(0,1,size=N0)*1j
        tmp1 = np.vander(tmp0, N=max(dim_tuple), increasing=True)
        tmp2 = [tmp1[:,:x] for x in dim_tuple]
        tmp3 = [(0,x+1) for x in range(len(dim_tuple))]
        tmp4 = [y for x in zip(tmp2,tmp3) for y in x]
        tmp5 = np.einsum(*tmp4, tuple(range(len(dim_tuple)+1)), optimize=True)
        # the orth(tmp5) is the required entangled matrix subspace
        matrix_subspace_orth,matrix_subspace,space_char = get_matrix_orthogonal_basis(tmp5.reshape(N0,dim_tuple[0],-1), field='complex')
        ret_other = None
    matrix_subspace = matrix_subspace.reshape(-1, *dim_tuple)
    matrix_subspace_orth = matrix_subspace_orth.reshape(-1, *dim_tuple)
    return matrix_subspace,matrix_subspace_orth,space_char,ret_other


def matrix_subspace_to_kraus_op(matrix_space, is_hermite=False, zero_eps=1e-10):
    if not is_hermite:
        matrix_space = get_hermite_channel_matrix_subspace(matrix_space, zero_eps)
    N0,dim,_ = matrix_space.shape
    EVL = np.linalg.eigvalsh(matrix_space)
    EVL_max = EVL.max()
    s = -1/EVL_max if (EVL_max>0) else (-1/EVL.min())
    z0 = matrix_space*s + np.eye(dim)
    tmp0 = z0.sum(axis=0)
    t = 1/np.linalg.eigvalsh(tmp0).max()
    matrix_space_PSD_sum1 = np.concatenate([(np.eye(dim) - t*tmp0)[np.newaxis], t*z0], axis=0)
    EVL,EVC = np.linalg.eigh(matrix_space_PSD_sum1)
    tmp0 = np.sqrt(np.maximum(0,EVL))
    tmp1 = np.arange(N0+1)
    tmp2 = np.zeros((N0+1,N0+1,dim,dim), dtype=matrix_space_PSD_sum1.dtype)
    tmp2[tmp1,tmp1] = (EVC*tmp0[:,np.newaxis]) @ EVC.transpose(0,2,1).conj()
    kraus_op = tmp2.reshape(N0+1,(N0+1)*dim,dim)
    return kraus_op


def kraus_op_to_matrix_subspace(op, reduce=True, zero_eps=1e-10):
    assert op.ndim==3
    dim_in = op.shape[2]
    ret = np.einsum(op.conj(), [0,1,2], op, [3,1,4], [0,3,2,4], optimize=True).reshape(-1, dim_in, dim_in)
    if reduce:
        tmp0 = reduce_vector_space(ret.reshape(ret.shape[0], -1), zero_eps).reshape(-1, ret.shape[1], ret.shape[2])
        # _,S,V = np.linalg.svd(ret.reshape(-1, dim_in*dim_in), full_matrices=False)
        # tmp0 = V[:(S>zero_eps).sum()].reshape(-1, dim_in, dim_in)
        ret = get_hermite_channel_matrix_subspace(tmp0, zero_eps)
    return ret


def detect_commute_matrix(np0, tag_real, zero_eps=1e-7):
    assert np0.dtype.type in {np.float32,np.float64}, f'not support complex matrix {np0.dtype}'
    assert (np0.ndim==3) and (np0.shape[1]==np0.shape[2])
    N0,N1,_ = np0.shape
    tmp0 = np.eye(N1)
    tmp1 = np.einsum(np0, [0,4,2], tmp0, [1,3], [0,1,2,3,4], optimize=True)
    tmp2 = np.einsum(np0, [0,1,3], tmp0, [2,4], [0,1,2,3,4], optimize=True)
    tmp3 = matrix_to_gellmann_basis((tmp1 - tmp2).reshape(N0*N1*N1,N1,N1))
    # aS,aA,aD,aI
    tmp3 = tmp3[:,:-1]
    N2 = (N1*(N1-1))//2
    if tag_real:
        tmp3 = np.concatenate([tmp3[:,:N2],tmp3[:,(2*N2):]], axis=1)
    else:
        tmp3[:,N2:(2*N2)] *= -1
    tmp3 = np.concatenate([tmp3.real,tmp3.imag], axis=0)
    EVL,EVC = np.linalg.eigh(tmp3.T @ tmp3)
    ret = []
    for ind0 in range((EVL<=zero_eps).sum()):
        if tag_real:
            tmp0 = np.zeros(N1*N1, dtype=np.float64)
            tmp0[:N2] = EVC[:N2,ind0]
            tmp0[(2*N2):-1] = EVC[N2:,ind0]
            ret.append(gellmann_basis_to_matrix(tmp0).real)
        else:
            tmp0 = np.concatenate([EVC[:,ind0], np.zeros([0])])
            ret.append(gellmann_basis_to_matrix(tmp0))
    return ret


def get_vector_plane(vec0, vec1):
    vec0 = vec0 / np.linalg.norm(vec0)
    vec1 = vec1 / np.linalg.norm(vec1)
    angle = np.arccos(np.dot(vec0, vec1))
    tmp0 = vec1 - vec0*np.dot(vec0,vec1)
    vec1_orth = tmp0 / np.linalg.norm(tmp0)
    def hf0(theta):
        ret = vec0*np.cos(theta) + vec1_orth*np.sin(theta)
        return ret
    return angle,hf0


def detect_antisym_y_Ux(np0, a=1):
    assert (np0.ndim==3) and np0.shape[1]==np0.shape[2]
    tmp0 = np0 - a*np0.transpose(0,2,1)
    tmp1 = np.eye(16)
    tmp2 = np.einsum(tmp0, [0,1,3], tmp1, [2,4], [0,1,2,3,4], optimize=True)
    tmp3 = np.einsum(tmp0, [0,2,3], tmp1, [1,4], [0,1,2,3,4], optimize=True)
    z0 = (tmp2+tmp3).reshape(-1, 256)
    z1 = z0.T @ z0
    ret = np.linalg.eigvalsh(z1), z1
    return ret


def matrix_subspace_to_biquadratic_form(np0):
    matZ = np.einsum(np0.conj(), [0,1,2], np0, [0,3,4], [1,2,3,4], optimize=True)
    return matZ


def divide_into_sym_antisym_space(np0):
    assert (np0.ndim==3) and (np0.shape[1]==np0.shape[2])
    N0,N1,_ = np0.shape
    tmp0 = np0 + np0.transpose(0,2,1)
    ret0 = reduce_vector_space(tmp0.reshape(N0, N1*N1)).reshape(-1,N1,N1)
    tmp0 = np0 - np0.transpose(0,2,1)
    ret1 = reduce_vector_space(tmp0.reshape(N0, N1*N1)).reshape(-1,N1,N1)
    return ret0,ret1

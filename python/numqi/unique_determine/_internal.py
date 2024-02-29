import os
import json
import itertools
import numpy as np

def get_matrix_list_indexing(mat_list:np.ndarray|list, index:list[int]):
    r'''get a list of matrices by index. Necessary as both 3d-array and list of sparse matrices are used in this module

    Parameters:
        mat_list (np.ndarray,list): list of matrices
        index (list[int]): index of the matrices

    Returns:
        ret (np.ndarray,list): list of matrices
    '''
    if isinstance(mat_list, np.ndarray):
        index = np.asarray(index)
        assert (mat_list.ndim==3) and (index.ndim==1)
        ret = mat_list[index]
    else:
        ret = [mat_list[x] for x in index]
    return ret


def get_qutrit_projector_basis(num_qutrit:int=1):
    r'''get projection basis of qutrit projector (over-complete)

    Parameters:
        num_qutrit (int): number of qutrits

    Returns:
        ret (np.ndarray): `ndim=3`, `shape=(15**num_qutrit+1, 3**num_qutrit, 3**num_qutrit)` the first element is identity
    '''
    tmp0 = [
        [(0,1)], [(1,1)], [(2,1)],
        [(0,1),(1,1)], [(0,1),(1,-1)], [(0,1),(1,1j)], [(0,1),(1,-1j)],
        [(0,1),(2,1)], [(0,1),(2,-1)], [(0,1),(2,1j)], [(0,1),(2,-1j)],
        [(1,1),(2,1)], [(1,1),(2,-1)], [(1,1),(2,1j)], [(1,1),(2,-1j)],
    ]
    matrix_subspace = []
    for x in tmp0:
        tmp1 = np.zeros(3, dtype=np.complex128)
        tmp1[[y[0] for y in x]] = [y[1] for y in x]
        matrix_subspace.append(tmp1[:,np.newaxis] * tmp1.conj())
    matrix_subspace = np.stack(matrix_subspace)

    if num_qutrit>=1:
        tmp0 = matrix_subspace
        for _ in range(num_qutrit-1):
            tmp1 = np.einsum(tmp0, [0,1,2], matrix_subspace, [3,4,5], [0,3,1,4,2,5], optimize=True)
            tmp2 = [x*y for x,y in zip(tmp0.shape, matrix_subspace.shape)]
            tmp0 = tmp1.reshape(tmp2)
        matrix_subspace = tmp0
    tmp0 = np.eye(matrix_subspace.shape[1])[np.newaxis]
    matrix_subspace = np.concatenate([tmp0,matrix_subspace], axis=0)
    return matrix_subspace


hf_chebval_n = lambda x, n: np.polynomial.chebyshev.chebval(x, np.array([0]*n+[1]))*(1 if n==0 else np.sqrt(2))

def get_chebshev_orthonormal(dim_qudit:int, alpha:float, with_computational_basis:bool=False, return_basis:bool=False):
    r'''get Chebyshev polynomials basis (PB)

    How many orthonormal bases are needed to distinguish all pure quantum states?
    [doi-link](https://doi.org/10.1140/epjd/e2015-60230-5)

    Parameters:
        dim_qudit (int): dimension of the qudit
        alpha (float): phase
        with_computational_basis (bool): if True, include computational basis (5PB), otherwise 4PB
        return_basis (bool): if True, also return the basis

    Returns:
        ret (np.ndarray): list of projection operator, `ndim=3`, `shape=(dim_qudit*4, dim_qudit, dim_qudit)`, or `shape=(dim_qudit*5, dim_qudit, dim_qudit)` if `with_computational_basis=True`
        ret (list[np.ndarray]): optional, list of length 4 or 5, each element is np.ndarray with `shape=(dim_qudit,dim_qudit)`
    '''
    rootd = np.cos(np.pi*(np.arange(dim_qudit)+0.5)/dim_qudit)
    basis0 = np.stack([hf_chebval_n(rootd, x) for x in range(dim_qudit)], axis=1)/np.sqrt(dim_qudit)

    rootd1 = np.cos(np.pi*(np.arange(dim_qudit-1)+0.5)/(dim_qudit-1))
    tmp1 = np.stack([hf_chebval_n(rootd1, x) for x in range(dim_qudit)], axis=1)/np.sqrt(dim_qudit-1)
    tmp2 = np.array([0]*(dim_qudit-1)+[1])
    basis1 = np.concatenate([tmp1,tmp2[np.newaxis]], axis=0)

    basis2 = np.stack([hf_chebval_n(rootd, x)*np.exp(1j*alpha*x) for x in range(dim_qudit)], axis=1)/np.sqrt(dim_qudit)

    tmp1 = np.stack([hf_chebval_n(rootd1, x)*np.exp(1j*alpha*x) for x in range(dim_qudit)], axis=1)/np.sqrt(dim_qudit-1)
    tmp2 = np.array([0]*(dim_qudit-1)+[1])
    basis3 = np.concatenate([tmp1,tmp2[np.newaxis]], axis=0)

    basis_list = [basis0,basis1,basis2,basis3]
    if with_computational_basis:
        basis_list.append(np.eye(dim_qudit))
        tmp0 = np.eye(dim_qudit)

    tmp0 = np.concatenate(basis_list, axis=0)
    ret = tmp0[:,:,np.newaxis]*(tmp0[:,np.newaxis].conj())
    if return_basis:
        ret = ret,basis_list
    return ret


def save_index_to_file(file:str, key:str|None=None, index:None|list[int]|list[str]|list[list[int]]=None):
    r'''save index to file and read index from file, mainly used for Pauli UD measurement schemes

    Parameters:
        file (str): file path
        key (str,None): key string, if None, return all saved index
        index (None,list[int],list[str],list[list[int]]): index to be saved, allowed types

            None: read index from file

            list[int]: single index, `[2,3,4]`

            list[str]: single index, `["2 3 4"]`

            list[list[int]]: multiple index, `[[2,3,4]]`

    Returns:
        ret (dict,list): the return type depends on (key), if `key=None`, return `dict[str,list[tuple[int]]]`, otherwise `list[tuple[int]]`
    '''
    if os.path.exists(file):
        with open(file, 'r', encoding='utf-8') as fid:
            all_data = json.load(fid)
    else:
        all_data = dict()
    if (index is not None) and len(index)>0:
        if isinstance(index[0], int): #[2,3,4]
            index_batch = [[int(x) for x in index]]
        elif isinstance(index[0], str): #["2 3 4"]
            index_batch = [[int(y) for y in x.split(' ')] for x in index]
        else: #[[2,3,4]]
            index_batch = [[int(y) for y in x] for x in index]
        data_i = [[int(y) for y in x.split(' ')] for x in all_data.get(key, [])] + index_batch
        hf1 = lambda x: (len(x),)+x
        tmp0 = sorted(set([tuple(sorted(set(x))) for x in data_i]), key=hf1)
        all_data[key] = [' '.join(str(y) for y in x) for x in tmp0]
        with open(file, 'w', encoding='utf-8') as fid:
            json.dump(all_data, fid, indent=2)
    if key is None:
        ret = {k:[[int(y) for y in x.split(' ')] for x in v] for k,v in all_data.items()}
    else:
        ret = [[int(y) for y in x.split(' ')] for x in all_data.get(key,[])]
    return ret


def remove_index_from_file(file:str, key_str:str, index:list[int]|list[str]|list[list[int]]):
    r'''remove index from file

    Parameters:
        file (str): file path
        key_str (str): key string
        index (list[int],list[str],list[list[int]]): index to be removed, see `numqi.unique_determine.save_index_to_file` for allowed types
    '''
    assert os.path.exists(file)
    assert len(index)>0
    with open(file, 'r', encoding='utf-8') as fid:
        all_data = json.load(fid)
    if isinstance(index[0], int): #[2,3,4]
        index_batch = [[int(x) for x in index]]
    elif isinstance(index[0], str): #["2 3 4"]
        index_batch = [[int(y) for y in x.split(' ')] for x in index]
    else: #[[2,3,4]]
        index_batch = [[int(y) for y in x] for x in index]
    index_set = {tuple(sorted(set(x))) for x in index_batch}
    data_i = {tuple(sorted(set([int(y) for y in x.split(' ')]))) for x in all_data.get(key_str, [])}
    hf1 = lambda x: (len(x),)+x
    tmp0 = sorted(data_i-index_set, key=hf1)
    all_data[key_str] = [' '.join(str(y) for y in x) for x in tmp0]
    with open(file, 'w', encoding='utf-8') as fid:
        json.dump(all_data, fid, indent=2)


def load_pauli_ud_example(num_qubit:int|None=None, tag_group_by_size:bool=False):
    r'''load Pauli UD measurement schemes from built-in data

    Parameters:
        num_qubit (int,None): number of qubits
        tag_group_by_size (bool): if True, return a dict of data grouped by size

    Returns:
        ret (dict,list): the return type depends on (num_qubit,tag_group_by_size)

            (None,False): dict[int,list[tuple[int]]]

            (None,True): dict[int,dict[int,list[tuple[int]]]]

            (int,False): list[tuple[int]]

            (int,True): dict[int,list[tuple[int]]]
    '''
    path = os.path.join(os.path.dirname(__file__), '..', '_data', 'pauli_ud_core.json')
    assert os.path.exists(path), f'installation error, file "{path}" missing'
    with open(path, 'r', encoding='utf-8') as fid:
        all_data = {int(k):v for k,v in json.load(fid).items()}
    hf_groupby = lambda x: {y0:list(y1) for y0,y1 in itertools.groupby(x,key=len)}
    if num_qubit is None:
        ret = {k:[tuple(int(y) for y in x.split(' ')) for x in v] for k,v in all_data.items()}
        if tag_group_by_size:
            ret = {k:hf_groupby(v) for k,v in ret.items()}
    else:
        if num_qubit not in all_data:
            raise ValueError(f'"num_qubit={num_qubit}" not found in {path}')
        ret = [tuple(int(y) for y in x.split(' ')) for x in all_data[num_qubit]]
        if tag_group_by_size:
            ret = hf_groupby(ret)
    return ret


def get_element_probing_POVM(kind:str, dim:int):
    r'''element probing POVM

    Strictly-complete measurements for bounded-rank quantum-state tomography
    [doi-link](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.93.052105)

    Parameters:
        kind (str): 'eq8' or 'eq9'
        dim (int): dimension of the system

    Returns:
        ret (np.ndarray): 3D array
        ret (dict,list): the return type depends on (num_qubit,tag_group_by_size)
    '''
    assert kind in ('eq8', 'eq9')
    if kind == 'eq8':
        dim = int(dim)
        assert dim>=2
        ret = np.zeros((2*dim, dim, dim), dtype=np.complex128)
        ret[0] = np.eye(dim)
        ret[1,0,0] = 1
        ind0 = np.arange(1,dim)
        ret[ind0+1, 0, ind0] = 1
        ret[ind0+1, ind0, 0] = 1
        ret[ind0+dim, 0, ind0] = -1j
        ret[ind0+dim, ind0, 0] = 1j
    else: #eq9
        # faiure set
        #       |00> + |02> + |20> + |22> - (|11> + |13> + |31> + |33>)
        #       (I+sx) \otimes sz
        assert dim>=4
        assert (dim%2)==0
        s12 = 1/np.sqrt(2)
        ind0 = np.arange(dim, dtype=np.int64)
        tmp0 = np.zeros((dim,dim), dtype=np.complex128) #B1
        tmp0[ind0,(ind0//2)*2] = s12
        tmp0[ind0,(ind0//2)*2+1] = s12 * (1-2*(ind0%2))
        tmp1 = np.zeros((dim,dim), dtype=np.complex128) #B2
        tmp1[ind0,(ind0//2)*2+1] = s12
        tmp1[ind0,((ind0//2)*2+2)%dim] = s12 * (1-2*(ind0%2))
        tmp2 = np.zeros((dim,dim), dtype=np.complex128) #B3
        tmp2[ind0,(ind0//2)*2] = s12
        tmp2[ind0,(ind0//2)*2+1] = (1j*s12) * (1-2*(ind0%2))
        tmp3 = np.zeros((dim,dim), dtype=np.complex128) #B4
        tmp3[ind0,(ind0//2)*2+1] = s12
        tmp3[ind0,((ind0//2)*2+2)%dim] = (1j*s12) * (1-2*(ind0%2))
        tmp4 = np.concatenate([tmp0,tmp1,tmp2,tmp3], axis=0)
        ret = tmp4[:,:,np.newaxis] * tmp4[:,np.newaxis].conj()
    return ret

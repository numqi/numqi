import os
import platformdirs
import h5py
import numpy as np
import scipy.sparse

# use hdf5 file to store data

_appname = 'numqi'
_appauthor = 'husisy'
_hdf5_filename = 'data.hdf5'

def get_savepath():
    _appdir = platformdirs.user_data_dir(_appname, _appauthor)
    if not os.path.exists(_appdir):
        os.makedirs(_appdir)
    ret = os.path.join(_appdir, _hdf5_filename)
    return ret


def _save_to_hdf5_enter(_save_func):
    def hf0(key, data, overwrite:bool=False):
        with h5py.File(get_savepath(), 'a', libver='latest') as fid:
            if (key not in fid.keys()) or overwrite:
                if key in fid.keys():
                    del fid[key]
                _save_func(fid, key, data)
    return hf0


@_save_to_hdf5_enter
def _save_to_hdf5_numpy(fid:h5py.File, key:str, data:np.ndarray):
    tmp0 = fid.create_dataset(key, data=data)
    tmp0.attrs['data_format'] = 'numpy'

@_save_to_hdf5_enter
def _save_to_hdf5_scipy_csr(fid:h5py.File, key:str, data:scipy.sparse.csr_array):
    grp = fid.create_group(key)
    grp.attrs['data_format'] = 'scipy_sparse_csr'
    grp.create_dataset('data', data=data.data)
    grp.create_dataset('indices', data=data.indices)
    grp.create_dataset('indptr', data=data.indptr)
    grp.create_dataset('shape', data=data.shape)

def _load_from_hdf5_scipy_csr(grp):
    data = grp['data'][:]
    indices = grp['indices'][:]
    indptr = grp['indptr'][:]
    shape = tuple(grp['shape'][:])
    ret = scipy.sparse.csr_array((data, indices, indptr), shape=shape)
    return ret

def is_key_in_disk(key:str|None=None):
    filepath = get_savepath()
    if not os.path.exists(filepath):
        ret = [] if (key is None) else False #not exists
    else:
        with h5py.File(filepath, 'r', libver='latest') as fid:
            ret = list(fid.keys()) if (key is None) else (key in fid.keys())
    return ret


def save_to_disk(key:str, data, overwrite:bool=False):
    import scipy.sparse
    if isinstance(data, scipy.sparse.sparray):
        assert data.format=='csr', 'only csr sparse matrix is supported'
        _save_to_hdf5_scipy_csr(key, data, overwrite=overwrite)
    elif isinstance(data, np.ndarray):
        _save_to_hdf5_numpy(key, data, overwrite=overwrite)
    else:
        raise TypeError(f'Unsupported data type: {type(data)}. Only numpy arrays and scipy sparse csr matrices are supported.')


def load_from_disk(key:str):
    assert isinstance(key, str), 'key must be a string'
    with h5py.File(get_savepath(), 'r', libver='latest') as fid:
        if key not in fid.keys():
            raise KeyError(f'Key {key} not found in disk')
        tmp0 = fid[key].attrs['data_format']
        if tmp0=='numpy':
            ret = fid[key][:]
        elif tmp0=='scipy_sparse_csr':
            ret = _load_from_hdf5_scipy_csr(fid[key])
        else:
            raise ValueError(f'Unsupported data format: {tmp0}')
    return ret


def delete_from_disk(key:str):
    assert isinstance(key, str), 'key must be a string'
    with h5py.File(get_savepath(), 'a', libver='latest') as fid:
        if key in fid.keys():
            del fid[key]
        else:
            raise KeyError(f'Key {key} not found in disk')

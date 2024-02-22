import numpy as np

import numqi

np_rng = np.random.default_rng()
hf_randc = lambda *size: np_rng.normal(size=size) + 1j*np_rng.normal(size=size)


# for test only
_finite_group_list = [numqi.group.get_symmetric_group_cayley_table(n) for n in range(2,6)]
_finite_group_list += [numqi.group.get_symmetric_group_cayley_table(n, alternating=True) for n in range(2,6)]
_finite_group_list += [numqi.group.get_klein_four_group_cayley_table()]
_finite_group_list += [numqi.group.get_dihedral_group_cayley_table(n) for n in range(3,8)]


def test_cayley_table_to_left_regular_form():
    for cayley_table in _finite_group_list:
        if len(cayley_table)>20:
            continue
        left_regular_form = numqi.group.cayley_table_to_left_regular_form(cayley_table)
        dim = len(cayley_table)
        for ind0 in range(dim):
            for ind1 in range(dim):
                # g_i g_j = L_i g_j
                tmp0 = np.zeros(dim)
                tmp0[ind1] = 1
                tmp1 = np.zeros(dim)
                tmp1[cayley_table[ind0,ind1]] = 1
                assert np.abs(left_regular_form[ind0] @ tmp0 - tmp1).max() < 1e-10

                # (g_i g_j = g_k) simeq (L_i L_j = L_k)
                assert np.abs(left_regular_form[ind0] @ left_regular_form[ind1] - left_regular_form[cayley_table[ind0,ind1]]).max() < 1e-10


def test_reduce_group_representation():
    for cayley_table in _finite_group_list:
        # cayley_table #(np,int64,(N0,N0))
        N0 = len(cayley_table) #n_element
        left_regular_form = numqi.group.cayley_table_to_left_regular_form(cayley_table) #(np,int,(N0,N0,N0))
        irrep_list = numqi.group.reduce_group_representation(left_regular_form) #(list,(np,complex,(N0,N1,N1)))
        character,class_list,character_table = numqi.group.get_character_and_class(irrep_list)
        # print('#element={} dim(rep)={}, sum(dim**2)={}'.format(N0, [x.shape[1] for x in irrep_list], sum(x.shape[1]**2 for x in irrep_list)))
        assert sum(x.shape[1]**2 for x in irrep_list)==N0

        tmp0 = np.concatenate([x.reshape(x.shape[0],-1).T*(np.sqrt(x.shape[1]/N0)) for x in irrep_list])
        assert np.abs(tmp0 @ tmp0.T.conj() - np.eye(N0)).max() < 1e-10

        assert np.abs(character @ character.T.conj()/N0 - np.eye(len(irrep_list))).max() < 1e-10

        tmp0 = character.T.conj() @ character
        for x in [np.array(x) for x in class_list]:
            assert np.abs(tmp0[x[:,np.newaxis],x]-N0/len(x)).max() < 1e-7

        tmp0 = character[:, [x[0] for x in class_list]] * np.sqrt(np.array([len(x)/N0 for x in class_list]))
        assert np.abs(tmp0 @ tmp0.T.conj() - np.eye(len(class_list))).max() < 1e-7


def test_to_unitary_representation():
    N0 = 4
    z0 = numqi.group.get_symmetric_group_cayley_table(N0)
    z1 = numqi.group.cayley_table_to_left_regular_form(z0)
    z2 = numqi.group.reduce_group_representation(z1)

    tmp0 = numqi.group.matrix_block_diagonal(z2[-2], z2[-1])
    tmp1 = hf_randc(*tmp0.shape[1:])
    z3 = tmp1 @ tmp0 @ np.linalg.inv(tmp1)

    tmp0 = numqi.group.to_unitary_representation(z3)
    assert np.abs(tmp0 @ tmp0.transpose(0,2,1).conj() - np.eye(z3.shape[1])).max() < 1e-10


def test_group_algebra_product():
    for cayley_table in _finite_group_list:
        dim = cayley_table.shape[0]
        vec0 = np_rng.uniform(0, 1, size=dim)
        vec1 = np_rng.uniform(0, 1, size=dim)

        ret_ = np.zeros(dim, dtype=np.float64)
        for ind0 in range(dim):
            ret_[cayley_table[ind0]] += vec0[ind0] * vec1
        index = numqi.group.get_index_cayley_table(cayley_table)
        ret0 = numqi.group.group_algebra_product(vec0, vec1, index, use_index=True)
        assert np.abs(ret_-ret0).max() < 1e-10


def test_projection_op():
    for cayley_table in _finite_group_list:
        left_regular_form = numqi.group.cayley_table_to_left_regular_form(cayley_table)
        irrep_list = numqi.group.reduce_group_representation(left_regular_form)
        op_list = [x.transpose(1,2,0).conj()*(x.shape[1]/x.shape[0]) for x in irrep_list]
        index_product = numqi.group.get_index_cayley_table(cayley_table)
        dim_group = len(cayley_table)
        # eq4.2 @XinzhengLi
        for ind0 in range(len(irrep_list)):
            dim_irrep = op_list[ind0].shape[0]
            x0 = op_list[ind0].reshape(dim_irrep**2, dim_group)
            tmp0 = numqi.group.group_algebra_product(x0[:,np.newaxis], x0, index_product, use_index=True).reshape([dim_irrep]*4+[dim_group])
            tmp1 = tmp0 * (1-np.eye(dim_irrep)).reshape(dim_irrep,dim_irrep,1,1)
            assert np.abs(tmp1).max() < 1e-10
            tmp1 = np.diagonal(tmp0, axis1=1, axis2=2).transpose(3,0,1,2).reshape(dim_irrep, -1, dim_group)
            assert np.abs(x0-tmp1).max() < 1e-10

            for ind1 in range(ind0+1,len(op_list)):
                dim_irrep1 = op_list[ind1].shape[0]
                x1 = op_list[ind1].reshape(dim_irrep1**2, dim_group)
                tmp0 = numqi.group.group_algebra_product(x0[:,np.newaxis], x1, index_product, use_index=True)
                assert np.abs(tmp0).max() < 1e-10
        projection_op = [np.diagonal(x, axis1=0, axis2=1).transpose(1,0) for x in op_list]
        tmp0 = sum(x.sum(axis=0)for x in projection_op)
        assert abs(tmp0[0]-1)<1e-10
        if len(tmp0)>1:
            assert np.abs(tmp0[1:]).max() < 1e-10

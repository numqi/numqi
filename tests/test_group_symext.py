import numpy as np
import scipy.linalg

import numqi


def test_get_B1B2_basis():
    basis_B_part = numqi.group.symext.get_B1B2_basis()
    basis_B = np.concatenate(basis_B_part, axis=0)
    assert np.abs(basis_B @ basis_B.T.conj() - np.eye(len(basis_B))).max() < 1e-7
    for dimA in [2,3,4,5]:
        dm0 = numqi.random.rand_ABk_density_matrix(dimA, dimB=3, kext=2)
        basis_AB_part = [np.kron(np.eye(dimA),x) for x in basis_B_part]
        basis_AB = np.concatenate(basis_AB_part)
        dm1 = basis_AB.conj() @ dm0 @ basis_AB.T
        tmp0 = [(x.conj() @ dm0 @ x.T) for x in basis_AB_part]
        assert np.abs(scipy.linalg.block_diag(*tmp0) - dm1).max() < 1e-10


def test_get_B1B2B3_basis():
    # 10 8 8 1
    basis_B_part = numqi.group.symext.get_B1B2B3_basis()
    basis_B = np.concatenate(basis_B_part, axis=0)
    assert np.abs(basis_B @ basis_B.T.conj() - np.eye(len(basis_B))).max() < 1e-7
    for dimA in [2,3,4,5]:
        dm0 = numqi.random.rand_ABk_density_matrix(dimA, dimB=3, kext=3)
        basis_AB_part = [np.kron(np.eye(dimA),x) for x in basis_B_part]
        basis_AB = np.concatenate(basis_AB_part, axis=0)
        dm1 = basis_AB.conj() @ dm0 @ basis_AB.T
        tmp0 = [(x.conj() @ dm0 @ x.T) for x in basis_AB_part]
        assert np.abs(tmp0[1]-tmp0[2]).max() < 1e-10
        assert np.abs(scipy.linalg.block_diag(*tmp0) - dm1).max() < 1e-10


def test_get_sud_symmetric_irrep_basis():
    dimA = 2
    dimB_kext_list = (
        [(x,2) for x in range(2,6)]
        + [(x,3) for x in range(2,6)]
        + [(x,4) for x in range(2,6)]
    )
    for dimB,kext in dimB_kext_list:
        basis_B_list = numqi.group.symext.get_sud_symmetric_irrep_basis(dimB, kext)
        basis_B = np.concatenate([y for x in basis_B_list for y in x], axis=0)
        assert np.abs(basis_B @ basis_B.T.conj() - np.eye(len(basis_B))).max() < 1e-7
        basis_AB_list = [[np.kron(np.eye(dimA),y) for y in x] for x in basis_B_list]

        dm0 = numqi.random.rand_ABk_density_matrix(dimA, dimB, kext)
        basis_AB = np.concatenate([y for x in basis_AB_list for y in x], axis=0)
        dm1 = basis_AB.conj() @ dm0 @ basis_AB.T
        tmp0 = [[(y.conj() @ dm0 @ y.T) for y in x] for x in basis_AB_list]
        for x in tmp0:
            if len(x)>1:
                assert all(np.abs(y-x[0]).max()<1e-10 for y in x[1:])
        tmp1 = [y for x in tmp0 for y in x]
        assert np.abs(scipy.linalg.block_diag(*tmp1) - dm1).max() < 1e-10


def test_get_B3_irrep_basis():
    dimA = 2
    for dimB in [2,3,4,5]:
        basis_B_part = numqi.group.symext.get_B3_irrep_basis(dimB)
        basis_B = np.concatenate(basis_B_part, axis=0)
        assert np.abs(basis_B @ basis_B.T.conj() - np.eye(len(basis_B))).max() < 1e-7

        dm0 = numqi.random.rand_ABk_density_matrix(dimA, dimB, kext=3)
        basis_AB_part = [np.kron(np.eye(dimA),x) for x in basis_B_part]
        basis_AB = np.concatenate(basis_AB_part, axis=0)
        dm1 = basis_AB.conj() @ dm0 @ basis_AB.T
        tmp0 = [(x.conj() @ dm0 @ x.T) for x in basis_AB_part]
        assert np.abs(tmp0[1]-tmp0[2]).max() < 1e-10
        assert np.abs(scipy.linalg.block_diag(*tmp0) - dm1).max() < 1e-10


def test_get_B4_irrep_basis():
    dimA = 2
    for dimB in [2,3,4,5]:
        basis_B_part = numqi.group.symext.get_B4_irrep_basis(dimB)
        # 1,3,2,3,1
        basis_B = np.concatenate(basis_B_part, axis=0)
        assert np.abs(basis_B @ basis_B.T.conj() - np.eye(len(basis_B))).max() < 1e-7

        dm0 = numqi.random.rand_ABk_density_matrix(dimA, dimB, kext=4)
        basis_AB_part = [np.kron(np.eye(dimA),x) for x in basis_B_part]
        basis_AB = np.concatenate(basis_AB_part, axis=0)
        dm1 = basis_AB.conj() @ dm0 @ basis_AB.T
        tmp0 = [(x.conj() @ dm0 @ x.T) for x in basis_AB_part]
        assert np.abs(scipy.linalg.block_diag(*tmp0) - dm1).max() < 1e-10

        assert np.abs(tmp0[1]-tmp0[2]).max() < 1e-10
        assert np.abs(tmp0[1]-tmp0[3]).max() < 1e-10
        assert np.abs(tmp0[4]-tmp0[5]).max() < 1e-10
        if dimB>=3:
            assert np.abs(tmp0[6]-tmp0[7]).max() < 1e-10
            assert np.abs(tmp0[6]-tmp0[8]).max() < 1e-10

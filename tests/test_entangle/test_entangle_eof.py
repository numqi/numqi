import numpy as np

import numqi

np_rng = np.random.default_rng()

def test_eof_A_B():
    dimA = 4
    dimB = 3
    q0 = numqi.random.rand_state(dimA*dimB).reshape(dimA, dimB)
    rhoA = np.einsum(q0.conj(), [0,1], q0, [0,2], [1,2], optimize=True)
    rhoB = np.einsum(q0.conj(), [0,1], q0, [2,1], [0,2], optimize=True)
    ret0 = numqi.utils.get_von_neumann_entropy(rhoA)
    ret1 = numqi.utils.get_von_neumann_entropy(rhoB)
    assert abs(ret0-ret1) < 1e-10


def test_EntanglementFormationModel_separable():
    num_sample = 5
    for dimA,dimB in [(2,2),(3,4),(4,3)]:
        num_term = 2*dimA*dimB
        for _ in range(num_sample):
            model = numqi.entangle.EntanglementFormationModel(dimA, dimB, num_term)
            dm0 = numqi.random.rand_separable_dm(dimA, dimB, k=dimA*dimB)
            model.set_density_matrix(dm0)
            theta_optim = numqi.optimize.minimize(model, num_repeat=3, print_freq=0, tol=1e-10)
            assert theta_optim.fun < 1e-7

def test_EntanglementFormationModel_isotropic():
    num_sample = 10
    for dim in [2,3]:
        # alpha_list = np.sort(np_rng.uniform(-1/(dim*dim-1), 1, size=num_sample))
        alpha_list = np.linspace(-1/(dim*dim-1), 1, num_sample)
        ret_ = numqi.state.get_Isotropic_eof(dim, alpha_list)

        model = numqi.entangle.EntanglementFormationModel(dim, dim, 2*dim*dim)
        ret0 = []
        kwargs = dict(num_repeat=3, print_freq=0, tol=1e-10, print_every_round=0)
        for alpha_i in alpha_list:
            model.set_density_matrix(numqi.state.Isotropic(dim, alpha_i))
            theta_optim = numqi.optimize.minimize(model, **kwargs)
            ret0.append(theta_optim.fun)
        ret0 = np.array(ret0)
        assert np.abs(ret_-ret0).max() < 1e-7


def test_EntanglementFormationModel_werner():
    num_sample = 10
    for dim in [2,3]:
        alpha_list = np.linspace(-1, 1, num_sample)
        ret_ = numqi.state.get_Werner_eof(dim, alpha_list)

        model = numqi.entangle.EntanglementFormationModel(dim, dim, 2*dim*dim)
        ret0 = []
        kwargs = dict(num_repeat=3, print_freq=0, tol=1e-10, print_every_round=0)
        for alpha_i in alpha_list:
            model.set_density_matrix(numqi.state.Werner(dim, alpha_i))
            theta_optim = numqi.optimize.minimize(model, **kwargs)
            ret0.append(theta_optim.fun)
        ret0 = np.array(ret0)
        assert np.abs(ret_-ret0).max() < 1e-7


def test_2qubits_Concurrence_EntanglementFormation():
    dimA = 2
    dimB = 2

    for _ in range(5):
        while True:
            rho = numqi.random.rand_density_matrix(dimA*dimB)
            ret_ = numqi.entangle.get_concurrence_2qubit(rho)
            if ret_>1e-5: #otherwise, optimization issue due to sqrt(0)
                break
        model = numqi.entangle.ConcurrenceModel(dimA, dimB, num_term=2*dimA*dimB, rank=dimA*dimB, zero_eps=1e-14)
        model.set_density_matrix(rho)
        theta_optim = numqi.optimize.minimize(model, theta0='uniform', num_repeat=3, tol=1e-10, print_every_round=0)
        assert abs(theta_optim.fun-ret_) < 1e-8 #fail sometimes, gradient at sqrt(0) might be bad

        ret_ = numqi.entangle.get_eof_2qubit(rho)
        model = numqi.entangle.EntanglementFormationModel(dimA, dimB, num_term=2*dimA*dimB, rank=dimA*dimB)
        model.set_density_matrix(rho)
        theta_optim = numqi.optimize.minimize(model, theta0='uniform', num_repeat=3, tol=1e-10, print_every_round=0)
        assert abs(theta_optim.fun-ret_) < 1e-8


def test_Monogamy_of_entanglement():
    # https://en.wikipedia.org/wiki/Monogamy_of_entanglement
    num_qubit = 4
    rank = 2
    assert num_qubit >= 3
    np_rng = np.random.default_rng()

    while True:
        tmp0 = [numqi.random.rand_haar_state(2**num_qubit, np_rng) for _ in range(rank)]
        prob = numqi.manifold.to_discrete_probability_softmax(np_rng.uniform(-1,1,size=rank))
        rho = sum(y*x[:,np.newaxis]*x.conj() for x,y in zip(tmp0,prob))
        rdm_concurrence_list = []
        for ind0 in range(1, num_qubit):
            if ind0==1:
                rdm = np.trace(rho.reshape(4,2**(num_qubit-2), 4,2**(num_qubit-2)), axis1=1, axis2=3)
            elif ind0==num_qubit-1:
                rdm = np.trace(rho.reshape(2,2**(num_qubit-2),2, 2,2**(num_qubit-2),2), axis1=1, axis2=4).reshape(4,4)
            else:
                shape = 2**(ind0+1), 2**(num_qubit-ind0-1)
                tmp0 = np.trace(rho.reshape(*shape, *shape), axis1=1, axis2=3)
                shape = 2, 2**(ind0-1), 2
                rdm = np.trace(tmp0.reshape(*shape, *shape), axis1=1, axis2=4).reshape(4,4)
            rdm_concurrence_list.append(numqi.entangle.get_concurrence_2qubit(rdm))
        print(rdm_concurrence_list)
        if sum(x>1e-5 for x in rdm_concurrence_list)>=2:
            break

    model = numqi.entangle.EntanglementFormationModel(2, 2**(num_qubit-1), num_term=2**(num_qubit+1), rank=rank, zero_eps=1e-14)
    model.set_density_matrix(rho)
    theta_optim = numqi.optimize.minimize(model, theta0='uniform', num_repeat=3, tol=1e-10, print_freq=500)
    print(sum(rdm_concurrence_list), theta_optim.fun)
    assert sum(rdm_concurrence_list) < theta_optim.fun

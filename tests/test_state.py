import numpy as np

import numqi

def test_get_2qutrit_Antoine2022():
    qlist = np.linspace(0, 2.5, 50)
    is_ppt = np.array([numqi.entangle.is_ppt(numqi.state.get_2qutrit_Antoine2022(x), (3,3)) for x in qlist])
    assert np.all((qlist <= 1.5)==is_ppt)

    # about 15 seconds
    qlist = np.linspace(0, 2.5, 10)
    model = numqi.entangle.AutodiffCHAREE(dim=(3,3), num_state=27, distance_kind='gellmann')
    ree_cha_list = []
    for qi in qlist:
        model.set_dm_target(numqi.state.get_2qutrit_Antoine2022(qi))
        tmp0 = numqi.optimize.minimize(model, theta0='uniform', num_repeat=3, tol=1e-12, print_every_round=0, early_stop_threshold=1e-8)
        ree_cha_list.append(tmp0.fun)
    ree_cha_list = np.array(ree_cha_list)
    assert ree_cha_list[qlist<=0.5].max() < 1e-7
    assert ree_cha_list[qlist>0.5].min() > 1e-7


def test_get_bes2x4_Horodecki1997():
    blist = np.linspace(0, 1, 50)
    rho_list = [numqi.state.get_bes2x4_Horodecki1997(x) for x in blist]
    assert all(np.abs(x-x.T).max() < 1e-10 for x in rho_list)
    assert all(np.linalg.eigvalsh(x)[0] > -1e-7 for x in rho_list)
    assert all(abs(np.trace(x)-1)<1e-10 for x in rho_list)

    assert all(numqi.entangle.is_ppt(x, (2,4)) for x in rho_list)

    blist = np.linspace(0, 1, 8)[1:-1] # exclude b0=0 and b0=1
    model = numqi.entangle.AutodiffCHAREE(dim=(2,4), num_state=16, distance_kind='gellmann')
    ree_cha_list = []
    for x in blist: #about 20 seconds
        model.set_dm_target(numqi.state.get_bes2x4_Horodecki1997(x))
        tmp0 = numqi.optimize.minimize(model, theta0='uniform', num_repeat=3, tol=1e-10, print_every_round=0)
        ree_cha_list.append(tmp0.fun)
    ree_cha_list = np.array(ree_cha_list)
    assert ree_cha_list.min() > 1e-8 #all are entangled


def test_get_bes3x3_Horodecki1997():
    alist = np.linspace(0, 1, 50)
    rho_list = [numqi.state.get_bes3x3_Horodecki1997(x) for x in alist]
    assert all(np.abs(x-x.T).max() < 1e-10 for x in rho_list)
    assert all(np.linalg.eigvalsh(x)[0] > -1e-7 for x in rho_list)
    assert all(abs(np.trace(x)-1)<1e-10 for x in rho_list)

    assert all(numqi.entangle.is_ppt(x, (3,3)) for x in rho_list)

    alist = np.linspace(0, 1, 8)[1:-1] # exclude b0=0 and b0=1
    model = numqi.entangle.AutodiffCHAREE(dim=(3,3), num_state=18, distance_kind='gellmann')
    ree_cha_list = []
    for x in alist: #about 20 seconds
        model.set_dm_target(numqi.state.get_bes3x3_Horodecki1997(x))
        tmp0 = numqi.optimize.minimize(model, theta0='uniform', num_repeat=3, tol=1e-10, print_every_round=0)
        ree_cha_list.append(tmp0.fun)
    ree_cha_list = np.array(ree_cha_list)
    assert ree_cha_list.min() > 1e-8 #all are entangled

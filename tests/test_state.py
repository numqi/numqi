import numpy as np

import numqi

def test_get_2qutrit_Antoine2022():
    qlist = np.linspace(0, 2.5, 50)
    rho_list = [numqi.state.get_2qutrit_Antoine2022(x) for x in qlist]

    is_ppt = np.array([numqi.entangle.is_ppt(x) for x in rho_list])
    assert np.all((qlist <= 1.5)==is_ppt)

    model = numqi.entangle.AutodiffCHAREE(dim0=3, dim1=3, num_state=27, distance_kind='gellmann')
    ree_cha_list = []
    # about 1 minute
    for rho in rho_list:
        model.set_dm_target(rho)
        tmp0 = numqi.optimize.minimize(model, theta0='uniform', num_repeat=3, tol=1e-12, print_every_round=0, early_stop_threshold=1e-8)
        ree_cha_list.append(tmp0.fun)
    ree_cha_list = np.array(ree_cha_list)
    assert ree_cha_list[qlist<=0.5].max() < 1e-7
    assert ree_cha_list[qlist>0.5].min() > 1e-7

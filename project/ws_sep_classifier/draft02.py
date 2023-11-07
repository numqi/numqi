import numpy as np
import scipy.stats

import numqi

np_rng = np.random.default_rng()

def is_positive_semi_definite(np0):
    # https://math.stackexchange.com/a/13311
    # https://math.stackexchange.com/a/87538
    # Sylvester's criterion
    try:
        np.linalg.cholesky(np0)
        ret = True
    except np.linalg.LinAlgError:
        ret = False
    return ret


dimA = 3
dimB = 3
num_sample = 1000

alpha = np.ones(dimA*dimB)/2

dm_list = []
ret = []
sampler = scipy.stats.dirichlet(alpha)
for _ in range(num_sample):
    matU = numqi.random.rand_haar_unitary(dimA*dimB)
    tmp0 = sampler.rvs()
    rho = (matU*tmp0) @ matU.T.conj()
    dm_list.append(rho)
    rho_pt = rho.reshape(dimA,dimB,dimA,dimB).transpose(0,3,2,1).reshape(dimA*dimB,-1)
    ret.append(is_positive_semi_definite(rho_pt))
ret = np.array(ret)


np_rng = np.random.default_rng()

dimA = 2
dimB = 4

model = numqi.entangle.AutodiffCHAREE(dimA, dimB, num_state=3*dimA*dimB, distance_kind='gellmann')
z0_list = []
kwargs = dict(num_repeat=3, tol=1e-12, print_every_round=0)
for ind0 in range(1000):
    dm0 = numqi.random.rand_density_matrix(dimA*dimB, seed=np_rng)
    beta_l_ppt,beta_u_ppt = numqi.entangle.get_ppt_boundary(dm0, (dimA,dimB))

    dm1 = numqi.entangle.hf_interpolate_dm(dm0, beta=beta_u_ppt)
    model.set_dm_target(dm1)
    theta_optim0 = numqi.optimize.minimize(model, theta0='uniform', **kwargs)

    dm2 = numqi.entangle.hf_interpolate_dm(dm0, beta=beta_u_ppt)
    model.set_dm_target(dm2)
    theta_optim1 = numqi.optimize.minimize(model, theta0='uniform', **kwargs)

    print(ind0, theta_optim0.fun, theta_optim1.fun, max(y for x in z0_list for y in x[1:]))
    z0_list.append((dm0, theta_optim0.fun, theta_optim1.fun))

# how to estimate the volume of density matrix

def test_shift_ball_volume():
    from tqdm import tqdm
    np_rng = np.random.default_rng()
    dim = 35
    delta = np_rng.uniform(-0.8, 0.8)
    num_sample = 10000000
    z0 = []
    for ind0 in tqdm(range(num_sample)):
        tmp0 = np_rng.normal(size=dim)
        tmp0 /= np.linalg.norm(tmp0)
        # law of cosines https://en.wikipedia.org/wiki/Law_of_cosines
        ct = tmp0[0] / np.linalg.norm(tmp0)
        tmp1 = np.sqrt(1 - delta*delta*(1-ct*ct))
        z0.append(-delta*ct+tmp1)
        z0.append(delta*ct + tmp1)
    z0 = np.array(z0)
    tmp0 = z0**dim
    print(tmp0.mean(), tmp0.std()/np.sqrt(num_sample))



np_rng = np.random.default_rng()
dimA = 2
dimB = 3

num_sample = 1000

radius_dm = []
radius_ppt = []

for _ in range(num_sample):
    tmp0 = np_rng.normal(size=dimA*dimB*dimA*dimB-1)
    tmp0 /= np.linalg.norm(tmp0)

    pass

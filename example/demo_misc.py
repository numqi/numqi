import numpy as np
from tqdm import tqdm

np_rng = np.random.default_rng()

def demo_shift_ball_volume():
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
    # large variance make it hard to estimate the volume for a center-shifted ball

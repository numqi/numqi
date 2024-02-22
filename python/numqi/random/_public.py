import random
import numpy as np

def get_random_rng(rng_or_seed=None):
    r'''Get random.Random object

    Parameters:
        rng_or_seed ([None], int, random.Random): If int or Random, use it for RNG. If None, use default RNG.

    Returns:
        ret (random.Random): random.Random object
    '''
    if rng_or_seed is None:
        ret = random.Random()
    elif isinstance(rng_or_seed, random.Random):
        ret = rng_or_seed
    else:
        ret = random.Random(int(rng_or_seed))
    return ret


def get_numpy_rng(rng_or_seed=None):
    r'''Get numpy.random.Generator object

    Parameters:
        rng_or_seed ([None], int, numpy.random.Generator): If int or Generator, use it for RNG. If None, use default RNG.

    Returns:
        ret (numpy.random.Generator): numpy.random.Generator object
    '''
    if rng_or_seed is None:
        ret = np.random.default_rng()
    elif isinstance(rng_or_seed, np.random.Generator):
        ret = rng_or_seed
    else:
        seed = int(rng_or_seed)
        ret = np.random.default_rng(seed)
    return ret

import itertools

def _get_nontrivial_subset_list(n:int):
    """Generate a list of non-trivial subsets of a set of size `n`.

    A non-trivial subset is defined as any subset that is not the empty set or the full set.
    For even `n`, the function also includes subsets that are complements of each other.

    Parameters:
        n (int): The size of the set. Must be greater than or equal to 2.

    Returns:
        list: A list of tuples, where each tuple represents a non-trivial subset of the set {0, 1, ..., n-1}.
    """
    assert n>=2
    ret = [x for r in list(range(1, (n+1)//2)) for x in itertools.combinations(tuple(range(n)), r)]
    if n%2==0:
        tmp0 = set(range(n))
        ret += {(x if (0 in x) else tuple(sorted((tmp0-set(x))))) for x in itertools.combinations(tuple(range(n)), n//2)}
    return ret

import itertools
import numpy as np


def make_strange_orthogonal_mat(m:int, n:int):
    assert (m>=1) and (n>=1)
    tmp0 = np.array([(1 if (x%2==0) else -1) for x in range(n)], dtype=np.int64)
    ret = np.einsum(np.eye(m, dtype=np.int64), [0,1], np.diag(tmp0), [2,3], [0,2,3,1], optimize=True).reshape(m*n, m*n)
    return ret


def guess_eigenvalue(EVL):
    ret = []
    zero_eps = 1e-7
    EVLi = EVL.copy()
    while len(EVLi):
        for INDEX in range(len(EVLi), 0, -1):
            if INDEX%2==1:
                tmp0 = np.arange(1-INDEX, INDEX, 2) / INDEX
            else:
                tmp0 = np.arange(2-INDEX, INDEX+1, 2) / INDEX
            tag = np.abs(tmp0.reshape(-1,1) - EVLi) < zero_eps
            if np.all(np.any(tag, axis=1)):
                ind0 = np.ones(len(EVLi), dtype=np.bool_)
                ind0[[np.nonzero(x)[0][0] for x in tag]] = 0
                ret.append(INDEX)
                EVLi = EVLi[ind0]
                break
            tag = np.abs(tmp0.reshape(-1,1)-1/INDEX - EVLi) < zero_eps
            if np.all(np.any(tag, axis=1)):
                ind0 = np.ones(len(EVLi), dtype=np.bool_)
                ind0[[np.nonzero(x)[0][0] for x in tag]] = 0
                ret.append(-INDEX)
                EVLi = EVLi[ind0]
                break
        else:
            ret = None # no solution
            break
    ret_str = None
    if ret is not None:
        ret = [(len(list(vlist)),(k if (k>0) else -k*1j)) for k,vlist in itertools.groupby(ret)]
        ret_str = ' + '.join([('' if (x==1) else f'{x}x')+str(y) for x,y in ret])
    return ret,ret_str


mn_dict = {
    4: ((2,2),),
    6: ((2,3), (3,2)),
    8: ((2,4), (4,2),),
    9: ((3,3),),
    10: ((2,5), (5,2)),
    12: ((2,6), (3,4), (4,3), (6,2)),
    14: ((2,7), (7,2),),
    15: ((3,5), (5,3)),
    16: ((2,8), (4,4), (8,2)),
    18: ((2,9), (3,6), (6,3), (9,2)),
    20: ((2,10), (4,5), (5,4), (10,2)),
    21: ((3,7), (7,3)),
    22: ((2,11), (11,2)),
    24: ((2,12), (3,8), (4,6), (6,4), (8,3), (12,2)),
    25: ((5,5),),
    26: ((2,13), (13,2)),
    27: ((3,9), (9,3)),
    28: ((2,14), (4,7), (7,4), (14,2)),
    30: ((2,15), (3,10), (5,6), (6,5), (10,3), (15,2)),
    32: ((2,16), (4,8), (8,4), (16,2)),
    33: ((3,11), (11,3)),
    34: ((2,17), (17,2)),
    35: ((5,7), (7,5)),
    36: ((2,18), (3,12), (4,9), (6,6), (9,4), (12,3), (18,2)),
}

for key,value in mn_dict.items():
    for m,n in value:
        np0 = make_strange_orthogonal_mat(m, n).astype(np.float64)
        assert np.abs(np0 @ np0.T - np.eye(m*n)).max() < 1e-10
        EVL = np.sort(np.angle(np.linalg.eigvals(np0))) / np.pi
        ret,ret_str = guess_eigenvalue(EVL)
        print(f'{m}x{n} = {ret_str}')
    print()

'''
2x2 = 4

2x3 = 4 + 2x1
3x2 = 4 + 2

2x4 = 6 + 2
4x2 = 6 + 2

3x3 = 2x4 + 1

2x5 = 6 + 2 + 2x1
5x2 = 6j + 4

2x6 = 10j + 2
3x4 = 10 + 2
4x3 = 2x5 + 2x1
6x2 = 10j + 2

2x7 = 12 + 2x1
7x2 = 12 + 2

3x5 = 2x6 + 3x1
5x3 = 2x6 + 2 + 1

2x8 = 2x8
4x4 = 4x4
8x2 = 2x8

2x9 = 2x8 + 2x1
3x6 = 16 + 2
6x3 = 16 + 2x1
9x2 = 2x8 + 2

2x10 = 18j + 2
4x5 = 2x9 + 2x1
5x4 = 18 + 2
10x2 = 18j + 2

3x7 = 4x4 + 2x2 + 1
7x3 = 4x4 + 2x2 + 1

2x11 = 2x6 + 2x3 + 2 + 2x1
11x2 = 3x6 + 4

2x12 = 22 + 2
3x8 = 22 + 2
4x6 = 22 + 2
6x4 = 22 + 2
8x3 = 2x11 + 2x1
12x2 = 22 + 2

5x5 = 6x4 + 1

2x13 = 20 + 4 + 2x1
13x2 = 20 + 4 + 2

3x9 = 8x3 + 3x1
9x3 = 4x6 + 2 + 1

2x14 = 18j + 6j + 4
4x7 = 2x9 + 2x3 + 4x1
7x4 = 18 + 6 + 2x2
14x2 = 18j + 6j + 4

2x15 = 28 + 2x1
3x10 = 28 + 2
5x6 = 2x14j + 2
6x5 = 2x14 + 2x1
10x3 = 28 + 2x1
15x2 = 28 + 2

2x16 = 3x10 + 2
4x8 = 3x10 + 2
8x4 = 3x10 + 2
16x2 = 3x10 + 2

3x11 = 2x16 + 1
11x3 = 2x16 + 1

2x17 = 3x10 + 2 + 2x1
17x2 = 3x10j + 4

5x7 = 2x16 + 2 + 1
7x5 = 2x16 + 3x1

2x18 = 2x12j + 6 + 4 + 2
3x12 = 2x12j + 6j + 4 + 2
4x9 = 4x6 + 2x3 + 2x2 + 2x1
6x6 = 9x4
9x4 = 2x12 + 6 + 4 + 2j
12x3 = 2x12 + 6 + 4 + 2x1
18x2 = 2x12j + 6 + 4 + 2
'''

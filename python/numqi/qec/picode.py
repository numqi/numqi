import numpy as np

def get_bg_picode(b2:int, g:int):
    # https://arxiv.org/abs/2411.13142 eq(7)
    assert (b2>=1) and (b2>=g)
    coeff = np.zeros((b2+g+1, 2), dtype=np.float64)
    a = np.sqrt((b2-g) / (2*b2))
    b = np.sqrt((b2+g) / (2*b2))
    coeff[0,0] = a
    coeff[b2,0] = b
    coeff[b2+g,1] = a
    coeff[g,1]  = b
    return coeff

# def get_Aydin_picode():
#     # https://doi.org/10.22331/q-2024-04-30-1321
#     # https://arxiv.org/abs/2411.13142 appendix A
#     pass


# def get_picode_ABBp():
#     pass #TODO

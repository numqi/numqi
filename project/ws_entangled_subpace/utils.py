import numpy as np

def W_type_state(a,b,c):
    psi = np.zeros(2**3, dtype=np.float64)
    psi[4] = a
    psi[2] = b
    psi[1] = c
    return psi.reshape(2,2,2)

def W_type_state_gme(a,b,c):
    r1 = b**2 + c**2 - a**2
    r2 = a**2 + c**2 - b**2
    r3 = a**2 + b**2 - c**2
    w = 2*a*b
    if r1 > 0 and r2 > 0 and r3 > 0:
        gamma_square = (1 + (16*(a**2)*(b**2)*(c**2)-w**2+r3**2)/(w**2-r3**2))/4
    else:
        gamma_square = max(a**2, b**2, c**2)
    gme = 1 - gamma_square
    return gme



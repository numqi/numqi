import numpy as np
import matplotlib.pyplot as plt

np_rng = np.random.default_rng()

X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
I = np.eye(2)

def hf_pauli_iexp(n3):
    assert n3.shape==(3,)
    n = np.linalg.norm(n3)
    n3 = n3/n
    ret = I*np.cos(n) + 1j*np.sin(n)*(n3[0]*X + n3[1]*Y + n3[2]*Z)
    return ret

alpha = 0.233
beta = 0.456
kx = np_rng.uniform(0, 2*np.pi)


tmp0 = hf_pauli_iexp(-(alpha/4)*np.array([1,0,0]))
tmp1 = hf_pauli_iexp(-(beta/2)*np.array([np.cos(kx),-np.sin(kx),0]))
ret_ = tmp0 @ tmp1 @ tmp0

c2a = np.cos(alpha/2)
s2a = np.sin(alpha/2)
c2b = np.cos(beta/2)
s2b = np.sin(beta/2)
ck = np.cos(kx)
sk = np.sin(kx)
E = np.arccos(c2a*c2b - ck*s2a*s2b)
nx = (ck*c2a*s2b + c2b*s2a)/np.sin(E)
ny = -sk*s2b/np.sin(E)
ret0 = hf_pauli_iexp(-E*np.array([nx,ny,0]))
print(np.abs(ret_-ret0).max())

kx_list = np.linspace(-np.pi, np.pi, 100)

c2a = np.cos(alpha/2)
s2a = np.sin(alpha/2)
c2b = np.cos(beta/2)
s2b = np.sin(beta/2)
ck = np.cos(kx_list)
sk = np.sin(kx_list)
ret0 = np.arccos(c2a*c2b - ck*s2a*s2b)
nx = (ck*c2a*s2b + c2b*s2a)/np.sin(ret0)
ny = -sk*s2b/np.sin(ret0)
assert np.abs(nx*nx+ny*ny-1).max() < 1e-10
tmp0 = 0.5*(alpha + beta*ck)
tmp1 = -0.5*(beta*sk)
ret_ = np.sqrt(tmp0*tmp0 + tmp1*tmp1)
nx_ = tmp0 / ret_
ny_ = tmp1 / ret_

fig,ax = plt.subplots()
# ax.plot(kx_list, ret_, 'x', label='ham')
# ax.plot(kx_list, ret0, label='E')
ax.plot(kx_list, ret_/ret0, label='ratio')
ax.legend()
ax.set_xlabel('k')
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)




c2a = np.cos(alpha/2)
s2a = np.sin(alpha/2)
c2b = np.cos(beta/2)
s2b = np.sin(beta/2)
c4b = np.cos(beta/4)
s4b = np.sin(beta/4)
ck = np.cos(kx_list)
sk = np.sin(kx_list)
c2k = np.cos(2*kx_list)
s2k = np.sin(2*kx_list)
ret0 = np.arccos(c2a*c2b - ck*s2a*s2b)
nx = (c4b*c4b*s2a - c2k*s2a*s4b*s4b + ck*c2a*s2b) / np.sin(ret0)
ny = (s2k*s2a*s4b*s4b - ck*c2a*s2b) / np.sin(ret0)
assert np.abs(nx*nx+ny*ny-1).max() < 1e-10




kx = np.pi
alpha = 0.233
beta = 0.456

c2a = np.cos(alpha/2)
s2a = np.sin(alpha/2)
c2b = np.cos(beta/2)
s2b = np.sin(beta/2)
# c4b = np.cos(beta/4)
s4b = np.sin(beta/4)
ck = np.cos(kx)
# sk = np.sin(kx)
# c2k = np.cos(2*kx)
s2k = np.sin(2*kx)
E = np.arccos(c2a*c2b - ck*s2a*s2b)
ny = (s2k*s2a*s4b*s4b - ck*c2a*s2b) / np.sin(E) #2.017
print(ny)


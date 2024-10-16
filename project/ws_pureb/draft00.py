import numpy as np
import torch

import numqi

def hf_werner_beta2alpha(beta, dim):
    rho = numqi.state.Werner(dim, 1)
    tmp0 = numqi.utils.hf_interpolate_dm(rho, beta=beta)[0,0]
    ret = (tmp0*dim*dim - 1) / (tmp0*dim-1)
    return ret

def hf_isotropic_beta2alpha(beta, dim):
    rho = numqi.state.Isotropic(dim, 1)
    tmp0 = numqi.utils.hf_interpolate_dm(rho, beta=beta)[1,1]
    ret = 1-tmp0*dim*dim
    return ret

class PureBosonicExt(torch.nn.Module):
    def __init__(self, dimA, dimB, kext):
        super().__init__()
        Bij = numqi.dicke.get_partial_trace_ABk_to_AB_index(kext, dimB)
        num_dicke = numqi.dicke.get_dicke_number(kext, dimB)
        tmp0 = [torch.int64,torch.int64,torch.complex128]
        self.Bij = [[torch.tensor(y0,dtype=y1) for y0,y1 in zip(x,tmp0)] for x in Bij]
        self.manifold = numqi.manifold.Sphere(dimA*num_dicke, dtype=torch.float64, method='quotient')
        self.dimA = dimA
        self.dimB = dimB

        self.dm_torch = None
        self.dm_target = None

    def set_dm_target(self, target):
        assert target.ndim in {1,2}
        if target.ndim==1:
            target = target[:,np.newaxis] * target.conj()
        assert (target.shape[0]==target.shape[1])
        self.dm_target = torch.tensor(target, dtype=torch.complex128)

    def forward(self):
        tmp1 = self.manifold().reshape(self.dimA,-1).to(torch.complex128)
        dm_torch = numqi.dicke.partial_trace_ABk_to_AB(tmp1, self.Bij)
        self.dm_torch = dm_torch.detach()
        tmp0 = numqi.gellmann.dm_to_gellmann_basis(self.dm_target)
        tmp1 = numqi.gellmann.dm_to_gellmann_basis(dm_torch)
        loss = torch.sum((tmp0-tmp1)**2)
        return loss


dim = 3
use_boson = True
rho = numqi.state.Werner(dim, 1)
kext_list = np.array([2,3,4,5,6])
alpha_list = []
for kext in kext_list:
    print(kext)
    beta = numqi.entangle.get_ABk_symmetric_extension_boundary(rho, (dim,dim), kext=kext, use_boson=use_boson)
    alpha_list.append(hf_werner_beta2alpha(beta, dim))
alpha_list = np.array(alpha_list)
ret_ = (kext_list+dim*dim-dim) / (kext_list*dim+dim-1)

dim = 4
use_boson = True
rho = numqi.state.Isotropic(dim, 1)
kext_list = np.array([2,3,4])
alpha_list = []
for kext in kext_list:
    print(kext)
    beta = numqi.entangle.get_ABk_symmetric_extension_boundary(rho, (dim,dim), kext=kext, use_boson=use_boson)
    alpha_list.append(hf_isotropic_beta2alpha(beta, dim))
alpha_list = np.array(alpha_list)
ret_ = (kext_list*dim+dim*dim-dim-kext_list) / (kext_list*(dim*dim-1))

print(alpha_list)
print(np.abs(ret_-alpha_list).max())

werner_boson_dict = {(2,2):4/5, (2,3):5/7, (2,4):2/3, (2,5):7/11, (2,6):8/13, (2,7):9/15, (2,8):10/17, (2,9):11/19,
                (3,2):5/7, (3,3):3/5, (3,4):7/13, (3,5):1/2, (3,6):9/19, (3,7): 5/11,
                (4,2):2/3, (4,3):7/13, (4,4):8/17}
# isotropic boson-ext is symmetric-ext
isotropic_boson_dict = {(2,2):2/3, (2,3):5/9, (2,4):1/2, (2,5):7/15, (2,6):4/9, (2,7):3/7, (2,8):5/12, (2,9):11/27,
                (3,2):5/8, (3,3):1/2, (3,4):7/16, (3,5):2/5, (3,6):3/8, (3,7):5/14, (3,8):11/32, #11/32 is not verified
                (4,2):3/5, (4,3):7/15, (4,4):2/5}
# use_boson=True d=2 (k=2,3,4,5): (k+d^2-d)/(k*d+d-1) 4/5 5/7 6/9 7/11
# use_boson=True d=3 (k=2,3,4,5): 5/7 3/5 7/13 9/19
# use_boson=True d=4 (k=2,3,4): 2/3 7/13 8/17

dim = 4
kext_list = list(range(2,5))
z0 = dict()
for kext in kext_list:
    model = numqi.entangle.PureBosonicExt(dim, dim, kext=kext, distance_kind='gellmann')
    model.set_dm_target(numqi.state.Werner(dim, werner_boson_dict[(dim,kext)]))
    z0[(dim,kext)] = numqi.optimize.minimize(model, tol=1e-20, num_repeat=10, print_every_round=0).fun
    print(dim, kext, z0[(dim,kext)])
# werner(d=2) k=5(5e-6) k=4/6/7/8/9(1e-16)
# werner(d=3) k=2(0.067) k=3(0.016) k=4(0.001) k=5(1e-9) k=6/7(1e-16)
# werner(d=4) k=2(0.075) k=3(0.0083) k=4(3e-5)

dim = 4
kext_list = list(range(6,9))
z0 = dict()
for kext in kext_list:
    model = numqi.entangle.PureBosonicExt(dim, dim, kext=kext, distance_kind='gellmann')
    tmp0 = (kext*dim+dim*dim-dim-kext) / (kext*(dim*dim-1))
    model.set_dm_target(numqi.state.Werner(dim, tmp0))
    z0[(dim,kext)] = numqi.optimize.minimize(model, tol=1e-20, num_repeat=10, print_every_round=1).fun
    print(dim, kext, z0[(dim,kext)])
# isotropic(d=2) k=2(0.0355) k=3(0.016) k=4/5/6/7/8/9(1e-22)
# isotropic(d=3) k=2(0.068) k=3(0.016) k=4(7e-5) k=5/6/7/8(1e-21)
# isotropic(d=4) k=2(0.075) k=3(0.0086) k=4/5/6/7/8(1e-21)



dim = 3
for kext in [5,6,7,8]:
    model = PureBosonicExt(dim, dim, kext=kext) #restrict to real number
    # model = numqi.entangle.PureBosonicExt(dim, dim, kext=kext, distance_kind='gellmann')
    tmp0 = (kext*dim+dim*dim-dim-kext) / (kext*(dim*dim-1))
    model.set_dm_target(numqi.state.Werner(dim, tmp0))
    numqi.optimize.minimize(model, tol=1e-20, num_repeat=10, print_every_round=1).fun
    z0 = model.manifold().detach().numpy().copy()
    print(kext, 1/np.sqrt(dim), np.linalg.svd(z0.reshape(dim, -1))[1])
# d=4,k=4,real, not zero

hession = numqi.optimize.get_model_hessian(model)
EVL = np.linalg.eigvalsh(hession)
print(EVL.shape[0], (EVL < 1e-8).sum())
# d=3,k=5, 63/19
# d=3,k=6, 84/40
# d=3,k=7, 108/64
# d=4,k=5, 224/89
# d=5,k=6, 226/201

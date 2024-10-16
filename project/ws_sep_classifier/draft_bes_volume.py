import numpy as np
import torch

import numqi

# is there any volume for 3x3 BES

class BESVolumeModel(torch.nn.Module):
    def __init__(self, dimA, dimB, norm=1e-4):
        super().__init__()
        np_rng = np.random.default_rng()
        self.dimA = dimA
        self.dimB = dimB
        np_rng = np.random.default_rng()
        tmp0 = np_rng.uniform(-1,1,size=dimA*dimA*dimB*dimB-1)
        self.theta = torch.nn.Parameter(torch.tensor(tmp0, dtype=torch.float64))
        self.dm0 = None
        self.dm0_pt = None
        self.norm = norm

    def set_density_matrix(self, dm0):
        assert dm0.shape==(self.dimA*self.dimB, self.dimA*self.dimB)
        assert np.abs(dm0-dm0.T.conj()).max() < 1e-10
        assert np.linalg.eigvalsh(dm0)[0] > -1e-10
        self.dm0 = torch.tensor(dm0, dtype=torch.complex128)
        tmp0 = dm0.reshape(self.dimA,self.dimB,self.dimA,self.dimB).transpose(0,3,2,1).reshape(dm0.shape)
        self.dm0_pt = torch.tensor(tmp0, dtype=torch.complex128)

    def forward(self):
        tmp0 = self.theta * (self.norm / torch.linalg.norm(self.theta))
        tmp1 = torch.concat([tmp0, torch.zeros(1, dtype=torch.float64)], dim=0)
        mat = self.dm0_pt + numqi.gellmann.gellmann_basis_to_matrix(tmp1)
        ret = torch.linalg.eigvalsh(mat)[0]
        return ret

# three optimization problems
# find a Ball $B(x,r)$ where $x$ is a BES and $r$ is a radius
#     s.t. min_{x in B, y in CHA} d(x,y) > 0
#     s.t. min_{x in B} lambda(x)>0
#     s.t. min_{x in B} lambda(pt(x))>0

np_rng = np.random.default_rng()

dm_tiles = numqi.entangle.load_upb('tiles', return_bes=True)[1]

beta_dm = numqi.gellmann.dm_to_gellmann_norm(dm_tiles)
kwargs = dict(dim=(3,3), kext=3, use_ppt=True, use_boson=True, return_info=True, use_tqdm=True)
beta_kext,vecA,vecN = numqi.entangle.get_ABk_symmetric_extension_boundary(dm_tiles, **kwargs)
dm_inner = numqi.utils.hf_interpolate_dm(dm_tiles, beta=(beta_dm+beta_kext)/2)

model = BESVolumeModel(3,3)
model.set_density_matrix(dm_inner)
model().item()

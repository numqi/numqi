# Symmetric Extension

```Python
import numpy as np
import numqi

dimA = 3
dimB = 3
kext = 2
rho_AB = numqi.random.rand_density_matrix(dimA*dimB)
# rho_AB = numqi.random.rand_separable_dm(dimA, dimB)
has_kext = numqi.entangle.check_ABk_symmetric_extension(rho_AB, (dimA,dimB), kext)
```

qubit bosonic extension: all symmetric extendible qubits density matrix are bosonic exntendible, so we only need bosonic extension here

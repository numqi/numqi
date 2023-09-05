# PPT criteria

```Python
import numpy as np
import matplotlib.pyplot as plt
import numqi
```

## numerical range

```Python
sx = np.array([[0,1],[1,0]])
sy = np.array([[0,-1j],[1j,0]])
sz = np.array([[1,0],[0,-1]])
op0 = np.kron(np.array([[1/np.sqrt(2),0],[0,0]]), sz)
op1 = (np.kron(sy, sx) - np.kron(sx, sy)) / 2
z0 = numqi.entangle.get_ppt_numerical_range(op0, op1, dim=(2,2), num_theta=400)

fig,ax = plt.subplots()
ax.plot(z0[:,0], z0[:,1])
ax.set_title('PPT numerical range')
ax.grid()
ax.set_xlabel('$O_0$')
ax.set_ylabel('$O_1$')
fig.tight_layout()
```

![ppt-numerical-range](../../data/ppt_numerical_range.png)

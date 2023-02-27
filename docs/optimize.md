# numpyqi.optimize

A wrapper for calling `scipy.optimize.minimize` on torch Module

[wiki/test-functions-for-optimization](https://en.wikipedia.org/wiki/Test_functions_for_optimization)

```python
import torch
import numpyqi

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x = torch.nn.Parameter(torch.tensor([1], dtype=torch.float32))
        self.y = torch.nn.Parameter(torch.tensor([-1], dtype=torch.float32))
    def forward(self):
        loss = self.x**2 + self.y**2
        return loss

model = DummyModel()
theta_optim = numpyqi.optimize.minimize(model, tol=1e-7, num_repeat=1, method='L-BFGS-B')
```

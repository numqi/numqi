import numpy as np
import pytest

import numqi

try:
    import torch
    from _torch_model import Rosenbrock
except ImportError:
    torch = None
    Rosenbrock = None

@pytest.mark.skipif(torch is None, reason='pytorch is not installed')
def test_gradient_correct():
    model = Rosenbrock(num_parameter=5)
    numqi.optimize.check_model_gradient(model, zero_eps=1e-4)

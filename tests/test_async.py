from gt4py import gtscript
from gt4py.gtscript import PARALLEL, FORWARD, BACKWARD, computation, interval
import gt4py.storage as gt_storage
import numpy as np

try:
    import gtstencil_example
except ImportError:
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from gtstencil_example import BACKEND, REBUILD, DTYPE_FLOAT, FIELD_FLOAT
from gtstencil_example.artificial import add_one

def test_async_launch():
    n = 512
    shape = (128, 128, n)
    default_origin = (0, 0, 0)
    n_iter = 10
    x = gt_storage.from_array(np.zeros(shape), BACKEND, default_origin, dtype=DTYPE_FLOAT)
    for _ in range(n_iter):
        add_one(x, async_launch=True, streams=0)
    x.device_to_host()
    x_np = x.view(np.ndarray)
    x_ref = np.zeros(shape)+n_iter
    assert np.allclose(x_np, x_ref)

if __name__ == "__main__":
    test_async_launch()

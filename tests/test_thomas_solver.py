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
from gtstencil_example.thomas_solver import thomas_solver_outofplace, thomas_solver_inplace, thomas_solver_gt_inplace

@gtscript.stencil(backend = BACKEND, rebuild = REBUILD)
def matmul_v(
    a: FIELD_FLOAT,
    b: FIELD_FLOAT,
    c: FIELD_FLOAT,
    x: FIELD_FLOAT,
    d: FIELD_FLOAT
):
    """
    Tridiagonal matrix - vector multiplication

    [b0 c0            ] [x0]   [d0]
    [a1 b1 c1         ] [x1]   [d1]
    [   a2 b2 c2      ] [x2] = [d2]
    [      ...        ] [..]   [..]
    [         ... cn-1] [..]   [..]
    [            an bn] [xn]   [dn]

    All input arrays have the shape of (1, 1, n)
    Assume a[0] = c[n] = 0, even they are not zeros!
    """
    with computation(PARALLEL):
        with interval(0, 1):
            d = b*x + c*x[0, 0, 1]
        with interval(1, -1):
            d = a*x[0, 0, -1] + b*x + c*x[0, 0, 1]
        with interval(-1, None):
            d = a*x[0, 0, -1] + b*x

n = 1000

def get_storages(shape, origin):
    a = gt_storage.from_array(np.random.randn(*shape), BACKEND, origin, dtype=DTYPE_FLOAT)
    b = gt_storage.from_array(np.random.randn(*shape), BACKEND, origin, dtype=DTYPE_FLOAT)
    c = gt_storage.from_array(np.random.randn(*shape), BACKEND, origin, dtype=DTYPE_FLOAT)
    x = gt_storage.from_array(np.random.randn(*shape), BACKEND, origin, dtype=DTYPE_FLOAT)
    d = gt_storage.empty(BACKEND, origin, shape, dtype=DTYPE_FLOAT)
    a.host_to_device()
    b.host_to_device()
    c.host_to_device()
    x.host_to_device()
    matmul_v(a, b, c, x, d)
    return a, b, c, d, x

def test_thomas_solver_outofplace():
    shape = (1, 1, n)
    default_origin = (0, 0, 0)
    a, b, c, d, x = get_storages(shape, default_origin)
    c1 = gt_storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    d1 = gt_storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    x1 = gt_storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)

    thomas_solver_outofplace(a, b, c, d, c1, d1, x1)
    x1.device_to_host()
    x.device_to_host()
    x1_np = x1.view(np.ndarray)
    x_np = x.view(np.ndarray)
    assert np.allclose(x_np, x1_np)

def test_thomas_solver_inplace():
    shape = (1, 1, n)
    default_origin = (0, 0, 0)
    a, b, c, d, x = get_storages(shape, default_origin)
    x1 = gt_storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)

    exec_info = {}
    thomas_solver_inplace(a, b, c, d, x1, exec_info=exec_info)
    x1.device_to_host()
    x.device_to_host()
    x1_np = x1.view(np.ndarray)
    x_np = x.view(np.ndarray)
    assert np.allclose(x_np, x1_np)

def test_thomas_solver_gt_inplace():
    shape = (1, 1, n)
    default_origin = (0, 0, 0)
    a, b, c, d, x = get_storages(shape, default_origin)
    x1 = gt_storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)

    exec_info = {}
    thomas_solver_gt_inplace(a, b, c, d, x1, exec_info=exec_info)
    x1.device_to_host()
    x.device_to_host()
    x1_np = x1.view(np.ndarray)
    x_np = x.view(np.ndarray)
    assert np.allclose(x_np, x1_np)

if __name__ == "__main__":
    test_thomas_solver_outofplace()
    test_thomas_solver_inplace()
    test_thomas_solver_gt_inplace()
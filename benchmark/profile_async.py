from gt4py import gtscript
from gt4py.gtgraph import InsertAsync, get_stencil_in_context
from gt4py.gtscript import PARALLEL, FORWARD, BACKWARD, computation, interval
from gt4py.stencil_object import StencilObject
import gt4py.storage as gt_storage
import numpy as np
import cupy
import ast
from time import time

try:
    import gtstencil_example
except ImportError:
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from gtstencil_example import BACKEND, REBUILD, DTYPE_FLOAT, FIELD_FLOAT
from gtstencil_example.artificial import add_one, add_add, mul_add

def prof_add_add(blocking=False):
    from gt4py.gtgraph import AsyncContext
    n_iter = 10
    async_context = AsyncContext(20, blocking=blocking, sleep_time=0.001)
    n = 512
    shape = (128, 128, n)
    default_origin = (0, 0, 0)
    x = gt_storage.ones(BACKEND, default_origin, shape, DTYPE_FLOAT)
    y = gt_storage.ones(BACKEND, default_origin, shape, DTYPE_FLOAT)
    z = gt_storage.ones(BACKEND, default_origin, shape, DTYPE_FLOAT)
    w = gt_storage.ones(BACKEND, default_origin, shape, DTYPE_FLOAT)
    a = gt_storage.ones(BACKEND, default_origin, shape, DTYPE_FLOAT)
    cupy.cuda.nvtx.RangePush("prof_add_add", 0)
    t0 = time()
    for _ in range(n_iter):
        async_context.schedule(add_add, x, y, z, w)
        async_context.schedule(add_one, a)
    async_context.wait_finish()
    t1 = time()
    cupy.cuda.nvtx.RangePop()
    y.device_to_host()
    w.device_to_host()
    a.device_to_host()
    y_np = y.view(np.ndarray)
    w_np = w.view(np.ndarray)
    a_np = a.view(np.ndarray)
    ref = np.ones(shape)+n_iter
    assert np.allclose(y_np, ref)
    assert np.allclose(w_np, ref)
    assert np.allclose(a_np, ref)
    print(f"add_add & add_one validated, blocking: {blocking}, time: {t1 - t0} s")

def prof_mul_add(blocking=False):
    from gt4py.gtgraph import AsyncContext
    n_iter = 10
    async_context = AsyncContext(20, blocking=blocking, sleep_time=0.001)
    n = 512
    shape = (128, 128, n)
    default_origin = (0, 0, 0)
    x = gt_storage.ones(BACKEND, default_origin, shape, DTYPE_FLOAT)
    y = gt_storage.ones(BACKEND, default_origin, shape, DTYPE_FLOAT)
    z = gt_storage.ones(BACKEND, default_origin, shape, DTYPE_FLOAT)
    w = gt_storage.ones(BACKEND, default_origin, shape, DTYPE_FLOAT)
    cupy.cuda.nvtx.RangePush("prof_mul_add", 1)
    t0 = time()
    for _ in range(n_iter):
        async_context.schedule(mul_add, x, y, z, w)
        async_context.schedule(add_one, z)
    async_context.wait_finish()
    t1 = time()
    cupy.cuda.nvtx.RangePop()
    z.device_to_host()
    w.device_to_host()
    z_np = z.view(np.ndarray)
    w_np = w.view(np.ndarray)
    ref_w = np.ones(shape)+n_iter
    ref_z = np.ones(shape)+1
    try:
        assert np.allclose(z_np, ref_z)
        assert np.allclose(w_np, ref_w)
        print(f"mul_add & add_one validated, blocking: {blocking}, time: {t1 - t0} s")
    except AssertionError:
        print(f"mul_add & add_one failed validation!, blocking: {blocking}, z: {z_np[0,0,0]}, w: {w_np[0,0,0]}")


def prof_add_add_non_async():
    n_iter = 10
    n = 512
    shape = (128, 128, n)
    default_origin = (0, 0, 0)
    x = gt_storage.ones(BACKEND, default_origin, shape, DTYPE_FLOAT)
    y = gt_storage.ones(BACKEND, default_origin, shape, DTYPE_FLOAT)
    z = gt_storage.ones(BACKEND, default_origin, shape, DTYPE_FLOAT)
    w = gt_storage.ones(BACKEND, default_origin, shape, DTYPE_FLOAT)
    a = gt_storage.ones(BACKEND, default_origin, shape, DTYPE_FLOAT)
    t0 = time()
    for _ in range(n_iter):
        add_add(x, y, z, w)
        add_one(a)
    t1 = time()
    y.device_to_host()
    w.device_to_host()
    a.device_to_host()
    y_np = y.view(np.ndarray)
    w_np = w.view(np.ndarray)
    a_np = a.view(np.ndarray)
    ref = np.ones(shape)+n_iter
    assert np.allclose(y_np, ref)
    assert np.allclose(w_np, ref)
    assert np.allclose(a_np, ref)
    print(f"nonasync add_add & add_one validated, time: {t1 - t0} s")

if __name__ == "__main__":
    prof_add_add_non_async()
    prof_add_add(blocking=False)
    prof_add_add(blocking=True)
    prof_mul_add(blocking=False)
    prof_mul_add(blocking=True)

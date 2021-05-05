from gt4py import gtscript
from gt4py.gtgraph import InsertAsync, get_stencil_in_context
from gt4py.gtscript import PARALLEL, FORWARD, BACKWARD, computation, interval
from gt4py.stencil_object import StencilObject
import gt4py.storage as gt_storage
import numpy as np
import cupy
import ast

try:
    import gtstencil_example
except ImportError:
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from gtstencil_example import BACKEND, REBUILD, DTYPE_FLOAT, FIELD_FLOAT
from gtstencil_example.artificial import add_one, add_add, mul_add

def test_async_launch(mystream = 0):
    n = 512
    shape = (128, 128, n)
    default_origin = (0, 0, 0)
    n_iter = 10
    x = gt_storage.zeros(BACKEND, default_origin, shape, DTYPE_FLOAT)
    print(f"Number of Kernels: {add_one.pyext_module.num_kernels()}")
    print(f"Dependency row ind: {add_one.pyext_module.dependency_row_ind()}")
    print(f"Dependency col ind: {add_one.pyext_module.dependency_col_ind()}")
    for _ in range(n_iter):
        add_one(x, async_launch=True, streams=mystream)
    x.device_to_host()
    x_np = x.view(np.ndarray)
    x_ref = np.zeros(shape)+n_iter
    assert np.allclose(x_np, x_ref)

def test_multi_streams(mystream = 0):
    n = 512
    shape = (128, 128, n)
    default_origin = (0, 0, 0)
    n_iter = 10
    x = gt_storage.ones(BACKEND, default_origin, shape, DTYPE_FLOAT)
    y = gt_storage.ones(BACKEND, default_origin, shape, DTYPE_FLOAT)
    z = gt_storage.ones(BACKEND, default_origin, shape, DTYPE_FLOAT)
    w = gt_storage.ones(BACKEND, default_origin, shape, DTYPE_FLOAT)
    print("add_add: ")
    print(f"Number of Kernels: {add_add.pyext_module.num_kernels()}")
    print(f"Dependency row ind: {add_add.pyext_module.dependency_row_ind()}")
    print(f"Dependency col ind: {add_add.pyext_module.dependency_col_ind()}")
    print("mul_add: ")
    print(f"Number of Kernels: {mul_add.pyext_module.num_kernels()}")
    print(f"Dependency row ind: {mul_add.pyext_module.dependency_row_ind()}")
    print(f"Dependency col ind: {mul_add.pyext_module.dependency_col_ind()}")
    for _ in range(n_iter):
        add_add(x, y, z, w, async_launch=True, streams=mystream)
    y.device_to_host()
    w.device_to_host()
    y_np = y.view(np.ndarray)
    w_np = w.view(np.ndarray)
    ref = np.ones(shape)+n_iter
    assert np.allclose(y_np, ref)
    assert np.allclose(w_np, ref)


def test_gtgraph():
    n_iter = 10
    n = 512
    shape = (128, 128, n)
    default_origin = (0, 0, 0)
    x = gt_storage.ones(BACKEND, default_origin, shape, DTYPE_FLOAT)
    y = gt_storage.ones(BACKEND, default_origin, shape, DTYPE_FLOAT)
    z = gt_storage.ones(BACKEND, default_origin, shape, DTYPE_FLOAT)
    w = gt_storage.ones(BACKEND, default_origin, shape, DTYPE_FLOAT)
    add_add(x, y, z, w)
    add_add(x, y, z, w)
    add_add(x, y, z, w)

def test_gtgraph_generated():
    from gt4py.gtgraph import AsyncContext
    n_iter = 10
    async_context = AsyncContext(20)
    n = 512
    shape = 128, 128, n
    default_origin = 0, 0, 0
    x = gt_storage.ones(BACKEND, default_origin, shape, DTYPE_FLOAT)
    y = gt_storage.ones(BACKEND, default_origin, shape, DTYPE_FLOAT)
    z = gt_storage.ones(BACKEND, default_origin, shape, DTYPE_FLOAT)
    w = gt_storage.ones(BACKEND, default_origin, shape, DTYPE_FLOAT)
    for _ in range(n_iter):
        async_context.schedule(add_add, x, y, z, w)
    async_context.wait_finish()
    y.device_to_host()
    w.device_to_host()
    y_np = y.view(np.ndarray)
    w_np = w.view(np.ndarray)
    ref = np.ones(shape)+n_iter
    assert np.allclose(y_np, ref)
    assert np.allclose(w_np, ref)

if __name__ == "__main__":
    mystream = cupy.cuda.stream.Stream(non_blocking=True)
    with mystream:
        print(f"using CUDA stream: {mystream.ptr}")
        test_async_launch(mystream.ptr)
        test_multi_streams(mystream.ptr)
    stencil_ctx = get_stencil_in_context()
    print(stencil_ctx)
    func_code = InsertAsync.apply(test_gtgraph, globals())
    print(func_code)
    test_gtgraph_generated()

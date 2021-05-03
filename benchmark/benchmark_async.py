from gt4py import gtscript
from gt4py.gtscript import PARALLEL, FORWARD, BACKWARD, computation, interval
import gt4py.storage as gt_storage
import numpy as np
from time import time
import cupy

try:
    import gtstencil_example
except ImportError:
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from gtstencil_example import BACKEND, REBUILD, DTYPE_FLOAT, FIELD_FLOAT
from gtstencil_example.artificial import add_one

def benchmark_async_launch(n_iter, async_launch=True, validate=False):
    n = 128
    shape = (512, 512, n)
    default_origin = (0, 0, 0)
    x = gt_storage.from_array(np.zeros(shape), BACKEND, default_origin, dtype=DTYPE_FLOAT)
    t1 = time()
    build_info = {}
    for _ in range(n_iter):
        add_one(x, async_launch=async_launch, streams=0, build_info=build_info)
    if async_launch:
        cupy.cuda.Device(0).synchronize()
    dt = time() - t1
    if validate:
        x.device_to_host()
        x_np = x.view(np.ndarray)
        x_ref = np.zeros(shape)+n_iter
        assert np.allclose(x_np, x_ref)
    return dt

if __name__ == "__main__":
    n_iter = 10
    if BACKEND != "gtc:cuda":
        print(f"Wrong backend: {BACKEND}")
        exit(-1)
    print(f"Benchmarking Async launch, iter = {n_iter}")
    t1 = benchmark_async_launch(n_iter, True)
    t2 = benchmark_async_launch(n_iter, False)
    print(f"With async: {t1} s \nWithout async: {t2} s")
import gt4py.storage as gt_storage
import numpy as np

try:
    import gtstencil_example
except ImportError:
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from gtstencil_example import BACKEND, REBUILD, DTYPE_FLOAT, FIELD_FLOAT
from gtstencil_example.thomas_solver import thomas_solver_inplace, thomas_solver_gt_inplace

from tests.test_thomas_solver import matmul_v

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

def benchmark_thomas_solver_inplace():
    nk = 128
    ni = 500
    nj = 500
    shape = (ni, nj, nk)
    default_origin = (0, 0, 0)
    a, b, c, d, x = get_storages(shape, default_origin)
    x1 = gt_storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)

    exec_info = {}
    thomas_solver_inplace(a, b, c, d, x1, exec_info=exec_info)

    running_time = exec_info["run_end_time"] - exec_info["run_start_time"]
    calling_time = exec_info["call_end_time"] - exec_info["call_start_time"]
    print("Gridsize:(%d,%d,%d)"%(ni,nj,nk))
    print("Running time of inplace thomas solver: %f seconds, calling time: %f"%(running_time, calling_time))

def benchmark_thomas_solver_gt_inplace():
    nk = 128
    ni = 500
    nj = 500
    shape = (ni, nj, nk)
    default_origin = (0, 0, 0)
    a, b, c, d, x = get_storages(shape, default_origin)
    x1 = gt_storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)

    exec_info = {}
    thomas_solver_gt_inplace(a, b, c, d, x1, exec_info=exec_info)

    running_time = exec_info["run_end_time"] - exec_info["run_start_time"]
    calling_time = exec_info["call_end_time"] - exec_info["call_start_time"]
    print("Gridsize:(%d,%d,%d)"%(ni,nj,nk))
    print("Running time of GridTools' inplace thomas solver: %f seconds, calling time: %f"%(running_time, calling_time))

if __name__ == "__main__":
    benchmark_thomas_solver_inplace()
    benchmark_thomas_solver_gt_inplace()
try:
    import gtstencil_example
except ImportError:
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import gt4py.storage as gt_storage
from gtstencil_example import BACKEND, REBUILD, DTYPE_FLOAT, FIELD_FLOAT
from gtstencil_example.fillz import compute as fillz_compute
from gtstencil_example.riem_solver3 import compute as riem_solver3_compute
from gtstencil_example.riem_solver_c import compute as riem_solver_c_compute
from gtstencil_example.saturation_adjustment import compute as sat_adj_compute
from gtstencil_example.thomas_solver import thomas_solver_inplace, thomas_solver_gt_inplace

from tests.read_data import read_data
from tests.mockgrid import Grid
from tests.test_thomas_solver import matmul_v

from time import time
import sys

from copy import deepcopy
import numpy as np

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

def benchmark_thomas_solver_inplace(repeat):
    nk = 128
    ni = 500
    nj = 500
    shape = (ni, nj, nk)
    default_origin = (0, 0, 0)
    a, b, c, d, x = get_storages(shape, default_origin)
    x1 = gt_storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    etime = np.zeros(repeat)

    for i in range(repeat):
        exec_info = {}
        thomas_solver_inplace(a, b, c, d, x1, exec_info=exec_info)
        running_time = exec_info["run_end_time"] - exec_info["run_start_time"]
        etime[i] = running_time
    return etime


def benchmark_stencil(name, compute_func, repeat: int = 1):
    etime = np.zeros(repeat)
    for i in range(repeat):
        for j in range(6):
            data = read_data(f"{name}_{j}", True)
            grid = Grid(data["grid_dict"])
            del data["grid_dict"]
            data["grid"] = grid
            t1 = time()
            compute_func(**data)
            etime[i] += time() - t1
    return etime

def benchmark(repeat: int):
    names = ["Fillz","Riem_Solver3","Riem_Solver_C","SatAdjust3d"]
    funcs = [fillz_compute,riem_solver3_compute,riem_solver_c_compute,sat_adj_compute]
    print(f"#Repeat times: {repeat}, backend: {BACKEND}")
    for i in range(4):
        name = names[i]
        etime = 0.0
        for _ in range(repeat):
            etime += benchmark_stencil(name, funcs[i])
        print(f"Stencil: {name}, time: {etime}")

def benchmark_detailed(repeat: int):
    names = ["Fillz","Riem_Solver3","Riem_Solver_C","SatAdjust3d", "Thomas_inplace"]
    funcs = [fillz_compute,riem_solver3_compute,riem_solver_c_compute,sat_adj_compute]
    #print(f"#Repeat times: {repeat}, backend: {BACKEND}")
    print(",".join(names))
    etimes = [benchmark_stencil(names[i], funcs[i], repeat) for i in range(4)]
    etime_thomas = benchmark_thomas_solver_inplace(repeat)
    for line in range(repeat):
        print(",".join([str(etimes[i][line]) for i in range(4)]+[str(etime_thomas[line])]))
        
if __name__ == "__main__":
    benchmark_detailed(int(sys.argv[1]))
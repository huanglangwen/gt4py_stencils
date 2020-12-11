try:
    import gtstencil_example
except ImportError:
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from gtstencil_example.fillz import compute as fillz_compute
from gtstencil_example.riem_solver3 import compute as riem_solver3_compute
from gtstencil_example.riem_solver_c import compute as riem_solver_c_compute
from gtstencil_example.saturation_adjustment import compute as sat_adj_compute

from tests import read_data
from tests.mockgrid import Grid

from time import time

def benchmark_stencil(name, compute_func):
    etime = 0.0
    for i in range(6):
        data = read_data(f"{name}_{i}", True)
        grid = Grid(data["grid_dict"])
        del data["grid_dict"]
        data["grid"] = grid
        t1 = time()
        compute_func(**data)
        etime += time() - t1
    return etime

def benchmark():
    names = ["Fillz","Riem_Solver3","Riem_Solver_C","SatAdjust3d"]
    funcs = [fillz_compute,riem_solver3_compute,riem_solver_c_compute,sat_adj_compute]
    for i in range(4):
        name = names[i]
        etime = benchmark_stencil(name, funcs[i])
        print(f"Stencil: {name}, time: {etime}")

if __name__ == "__main__":
    benchmark()
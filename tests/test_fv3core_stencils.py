try:
    import gtstencil_example
except ImportError:
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from gtstencil_example import BACKEND, REBUILD, DTYPE_FLOAT, FIELD_FLOAT
from gtstencil_example.fillz import compute as fillz_compute
from gtstencil_example.riem_solver3 import compute as riem_solver3_compute
from gtstencil_example.riem_solver_c import compute as riem_solver_c_compute
from gtstencil_example.saturation_adjustment import compute as sat_adj_compute

from tests import read_data
from tests.mockgrid import Grid

def test_fillz():
    for i in range(6):
        data = read_data(f"Fillz_{i}")
        grid = Grid(data["grid_dict"])
        del data["grid_dict"]
        data["grid"] = grid
        fillz_compute(**data)

def test_riem_solver3():
    for i in range(6):
        data = read_data(f"Riem_Solver3_{i}")
        grid = Grid(data["grid_dict"])
        del data["grid_dict"]
        data["grid"] = grid
        riem_solver3_compute(**data)

def test_riem_solver_c():
    for i in range(6):
        data = read_data(f"Riem_Solver_C_{i}")
        grid = Grid(data["grid_dict"])
        del data["grid_dict"]
        data["grid"] = grid
        riem_solver_c_compute(**data)

def test_sat_adj():
    for i in range(6):
        data = read_data(f"SatAdjust3d_{i}")
        grid = Grid(data["grid_dict"])
        del data["grid_dict"]
        data["grid"] = grid
        sat_adj_compute(**data)

if __name__ == "__main__":
    test_fillz()
    test_riem_solver3()
    test_riem_solver_c()
    test_sat_adj()
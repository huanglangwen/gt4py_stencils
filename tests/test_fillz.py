try:
    import gtstencil_example
except ImportError:
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from gtstencil_example import BACKEND, REBUILD, DTYPE_FLOAT, FIELD_FLOAT
from gtstencil_example.fillz import compute

from tests import read_data
from tests.mockgrid import Grid

def test_fillz():
    data = read_data("Fillz_0")
    grid = Grid(data["grid_dict"])
    del data["grid_dict"]
    data["grid"] = grid
    compute(**data)

if __name__ == "__main__":
    test_fillz()
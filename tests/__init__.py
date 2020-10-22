import os
import pickle
import gt4py as gt
from gt4py.storage.storage import Storage
import numpy as np

try:
    import gtstencil_example
except ImportError:
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from gtstencil_example import BACKEND, REBUILD, DTYPE_FLOAT, FIELD_FLOAT

def fix_gt_storage(data_dict):
    for key in data_dict.keys():
        data = data_dict[key]
        if isinstance(data, Storage):
            storage = gt.storage.from_array(data=data.view(np.ndarray),
                                            dtype=data.dtype,
                                            backend=BACKEND,
                                            shape=data.shape,
                                            default_origin=(0, 0, 0))
            data_dict[key] = storage
        elif isinstance(data, dict):
            fix_gt_storage(data)


def read_data(name):
    with open(os.path.join(os.path.dirname(__file__), f"data/{name}.pickle"), "rb") as f:
        data = pickle.load(f)
        fix_gt_storage(data)
        return data
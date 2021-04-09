try:
    import serialbox as ser
except ImportError:
    import os, sys
    if "SERIALBOX_DIR" in os.environ:
        SERIALBOX_DIR = os.environ["SERIALBOX_DIR"]
    else:
        SERIALBOX_DIR = os.path.dirname(__file__)
    sys.path.append(SERIALBOX_DIR)
    import serialbox as ser

try:
    import gtstencil_example
except ImportError:
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from gtstencil_example import BACKEND, REBUILD, DTYPE_FLOAT, DTYPE_INT, FIELD_FLOAT, DEFAULT_ORIGIN

from copy import deepcopy
import numpy as np
from gt4py import gtscript
import gt4py.storage as Storage

# Names of the input variables
IN_VARS = [ "iie", "kke", "kbot",
            "qv", "ql", "qr", "qg", "qa", "qn",
            "pt", "uin", "vin", "dz", "delp",
            "area", "dt_in", "land",
            "seconds", "p", "lradar",
            "reset",
            "qi", "qs",
            "qv_dt", "ql_dt", "qr_dt", "qi_dt", "qs_dt", "qg_dt", "qa_dt",
            "pt_dt", "w", "udt", "vdt",
            "rain", "snow", "ice", "graupel",
            "refl_10cm" ]
            
# Names of the output variables
OUT_VARS = [ "qi", "qs",
             "qv_dt", "ql_dt", "qr_dt", "qi_dt", "qs_dt", "qg_dt", "qa_dt",
             "pt_dt", "w", "udt", "vdt",
             "rain", "snow", "ice", "graupel",
             "refl_10cm" ]

# Names of the integer and boolean variables
INT_VARS = [ "iie", "kke", "kbot", "seconds", 
             "lradar", "reset" ]

# Names of the float variables
FLT_VARS = [ "qv", "ql", "qr", "qg", "qa", "qn",
             "pt", "uin", "vin", "dz", "delp",
             "area", "dt_in", "land",
             "p",
             "qi", "qs",
             "qv_dt", "ql_dt", "qr_dt", "qi_dt", "qs_dt", "qg_dt", "qa_dt",
             "pt_dt", "w", "udt", "vdt",
             "rain", "snow", "ice", "graupel",
             "refl_10cm" ]

def read_data(path, tile, ser_count, is_in):
    mode_str = "in" if is_in else "out"
    vars_     = IN_VARS if is_in else OUT_VARS
    serializer = ser.Serializer(ser.OpenModeKind.Read, path, "Generator_rank" + str(tile))
    savepoint  = ser.Savepoint(f"cloud_mp-{mode_str}-{ser_count:0>6d}")
    return data_dict_from_var_list(vars_, serializer, savepoint)

# Read given variables from a specific savepoint in the given serializer
def data_dict_from_var_list(vars, serializer, savepoint):
    data_dict = {}
    for var in vars:
        data_dict[var] = serializer.read(var, savepoint)
    searr_to_scalar(data_dict)
    return data_dict
    

# Convert single element arrays (searr) to scalar values of the correct 
# type
def searr_to_scalar(data_dict):
    for var in data_dict:
        if data_dict[var].size == 1:
            if var in INT_VARS: data_dict[var] = DTYPE_INT(data_dict[var][0])
            if var in FLT_VARS: data_dict[var] = DTYPE_FLOAT(data_dict[var][0])
            

# Transform a dictionary of numpy arrays into a dictionary of gt4py 
# storages of shape (iie-iis+1, jje-jjs+1, kke-kks+1)
def numpy_dict_to_gt4py_dict(np_dict):
    shape      = np_dict["qi"].shape
    gt4py_dict = {}
    for var in np_dict:
        data = np_dict[var]
        ndim = data.ndim
        if (ndim > 0) and (ndim <= 3) and (data.size >= 2):
            reshaped_data = np.empty(shape)
            if ndim == 1:       # 1D array (i-dimension)
                reshaped_data[...] = data[:, np.newaxis, np.newaxis]
            elif ndim == 2:     # 2D array (i-dimension, j-dimension)
                reshaped_data[...] = data[:, :, np.newaxis]
            elif ndim == 3:     # 3D array (i-dimension, j-dimension, k-dimension)
                reshaped_data[...] = data[...]
            dtype           = DTYPE_INT if var in INT_VARS else DTYPE_FLOAT
            gt4py_dict[var] = Storage.from_array(reshaped_data, BACKEND, DEFAULT_ORIGIN, dtype=dtype)
        else:   # Scalars
            gt4py_dict[var] = deepcopy(data)
    return gt4py_dict

# Cast a dictionary of gt4py storages into dictionary of numpy arrays
def view_gt4py_storage(gt4py_dict):
    np_dict = {}
    for var in gt4py_dict:
        data = gt4py_dict[var]
        # ~ if not isinstance(data, np.ndarray): data.synchronize()
        data.synchronize()
        np_dict[var] = data.view(np.ndarray)
    return np_dict

# Compare two dictionaries of numpy arrays, raise error if one array in 
# data does not match the one in ref_data
def compare_data(data, ref_data, explicit=True, blocking=True):
    wrong = []
    flag  = True
    for var in data:
        if not np.allclose(data[var], ref_data[var], rtol=1e-5, atol=1.e-5, equal_nan=True):
            wrong.append(var)
            flag = False
        else:
            if explicit: print(f"Successfully validated {var}!")
    if blocking:
        assert flag, f"Output data does not match reference data for field {wrong}!"
    else:
        if not flag: print(f"Output data does not match reference data for field {wrong}!")

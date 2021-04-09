import os
import numpy as np
from gt4py import gtscript

if "GT4PY_BACKEND" in os.environ:
    BACKEND    = os.environ["GT4PY_BACKEND"]
else:
    BACKEND    = "numpy"#"numpy"#"gtx86"#debug#cuda

DTYPE_FLOAT = np.float64
DTYPE_INT = np.int_
FIELD_FLOAT = gtscript.Field[DTYPE_FLOAT]
FIELD_INT = gtscript.Field[DTYPE_INT]
DEFAULT_ORIGIN = (0, 0, 0)
REBUILD = True
import os
import numpy as np
from gt4py import gtscript

if "GT4PY_BACKEND" in os.environ:
    BACKEND    = os.environ["GT4PY_BACKEND"]
else:
    BACKEND    = "cuda"#"numpy"#"gtx86"#debug

DTYPE_FLOAT = np.float64
FIELD_FLOAT = gtscript.Field[DTYPE_FLOAT]
REBUILD = True
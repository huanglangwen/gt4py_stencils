import os
import numpy as np

if "GT4PY_BACKEND" in os.environ:
    BACKEND    = os.environ["GT4PY_BACKEND"]
else:
    BACKEND    = "cuda"#"numpy"#"gtx86"#debug

DTYPE_FLOAT = np.float_
FIELD_FLOAT = gtscript.Field[DTYPE_FLOAT]
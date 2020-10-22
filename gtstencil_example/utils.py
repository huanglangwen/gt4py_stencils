import gt4py as gt
import numpy as np

from gtstencil_example import BACKEND, DTYPE_FLOAT, FIELD_FLOAT

try:
    import cupy as cp
except ImportError:
    cp = None

tracer_variables = [
    "qvapor",
    "qliquid",
    "qrain",
    "qice",
    "qsnow",
    "qgraupel",
    "qo3mr",
    "qsgs_tke",
    "qcld",
]

managed_memory = True

def make_storage_from_shape(
    shape,
    origin,
    dtype = DTYPE_FLOAT,
    init: bool = True,
):
    """Create a new gt4py storage of a given shape.

    Args:
        shape: Size of the new storage
        origin: Default origin for gt4py stencil calls
        dtype: Data type
        init: If True, initializes the storage to the default value for the type

    Returns:
        gtscript.Field[dtype]: New storage
    """
    storage = gt.storage.from_array(
        data=np.empty(shape, dtype=dtype),
        dtype=dtype,
        backend=BACKEND,
        default_origin=origin,
        shape=shape,
        managed_memory=managed_memory,
    )
    if init:
        storage[:] = dtype()

    return storage

def zeros(shape, storage_type=np.ndarray, dtype=DTYPE_FLOAT, order="F"):
    xp = cp if cp and storage_type is cp.ndarray else np
    return xp.zeros(shape)


def sum(array, axis=None, dtype=None, out=None, keepdims=False):
    xp = cp if cp and type(array) is cp.ndarray else np
    return xp.sum(array, axis, dtype, out, keepdims)


def repeat(array, repeats, axis=None):
    xp = cp if cp and type(array) is cp.ndarray else np
    return xp.repeat(array.data, repeats, axis)
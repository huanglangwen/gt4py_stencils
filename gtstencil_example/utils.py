import gt4py as gt
import numpy as np

from gtstencil_example import BACKEND, REBUILD, DTYPE_FLOAT, FIELD_FLOAT
from gt4py.gtscript import BACKWARD, PARALLEL, computation, interval#, region
from gt4py import gtscript

try:
    import cupy as cp
except ImportError:
    cp = None

halo = 3
origin = (halo, halo, 0)

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

@gtscript.stencil(backend = BACKEND, rebuild = REBUILD)
def copy_stencil(q_in: FIELD_FLOAT, q_out: FIELD_FLOAT):
    """Copy q_in to q_out.

    Args:
        q_in: input field
        q_out: output field
    """
    with computation(PARALLEL), interval(...):
        q_out = q_in

def copy(q_in, origin=(0, 0, 0), domain=None):
    """Copy q_in inside the origin and domain, and zero outside.

    Args:
        q_in: input field
        origin: Origin of the copy and new field
        domain: Extent to copy

    Returns:
        gtscript.Field[float]: Copied field
    """
    q_out = make_storage_from_shape(q_in.shape, origin, init=True)
    copy_stencil(q_in, q_out, origin=origin, domain=domain)
    return q_out

"""No region def in gt4py
@gtscript.function
def fill_4corners_x(q: FIELD_FLOAT):
    from __splitters__ import i_end, i_start, j_end, j_start

    # copy field
    q_out = q

    # Southwest
    with parallel(region[i_start - 2, j_start - 1]):
        q_out = q[1, 2, 0]
    with parallel(region[i_start - 1, j_start - 1]):
        q_out = q[0, 1, 0]

    # Southeast
    with parallel(region[i_end + 2, j_start - 1]):
        q_out = q[-1, 2, 0]
    with parallel(region[i_end + 1, j_start - 1]):
        q_out = q[0, 1, 0]

    # Northwest
    with parallel(region[i_start - 1, j_end + 1]):
        q_out = q[0, -1, 0]
    with parallel(region[i_start - 2, j_end + 1]):
        q_out = q[1, -2, 0]

    # Northeast
    with parallel(region[i_end + 1, j_end + 1]):
        q_out = q[0, -1, 0]
    with parallel(region[i_end + 2, j_end + 1]):
        q_out = q[-1, -2, 0]

    return q_out


@gtscript.function
def fill_4corners_y(q: FIELD_FLOAT):
    from __splitters__ import i_end, i_start, j_end, j_start

    # copy field
    q_out = q

    # Southwest
    with parallel(region[i_start - 1, j_start - 1]):
        q_out = q[1, 0, 0]
    with parallel(region[i_start - 1, j_start - 2]):
        q_out = q[2, 1, 0]

    # Southeast
    with parallel(region[i_end + 1, j_start - 1]):
        q_out = q[-1, 0, 0]
    with parallel(region[i_end + 1, j_start - 2]):
        q_out = q[-2, 1, 0]

    # Northwest
    with parallel(region[i_start - 1, j_end + 1]):
        q_out = q[1, 0, 0]
    with parallel(region[i_start - 1, j_end + 2]):
        q_out = q[2, -1, 0]

    # Northeast
    with parallel(region[i_end + 1, j_end + 1]):
        q_out = q[-1, 0, 0]
    with parallel(region[i_end + 1, j_end + 2]):
        q_out = q[-2, -1, 0]

    return q_out
"""

def fill_4corners(q, direction, grid):
    def definition(q: FIELD_FLOAT):
        from __externals__ import func

        with computation(PARALLEL), interval(...):
            q = func(q)

    extent = 3
    origin = (grid.is_ - extent, grid.js - extent, 0)
    domain = (grid.nic + 2 * extent, grid.njc + 2 * extent, q.shape[2])

    kwargs = {
        "origin": origin,
        "domain": domain,
    }

    if direction == "x":
        stencil = gtscript.stencil(definition=definition, externals={"func": fill_4corners_x})
        stencil(q, **kwargs)
    elif direction == "y":
        stencil = gtscript.stencil(definition=definition, externals={"func": fill_4corners_y})
        stencil(q, **kwargs)
    else:
        raise ValueError("Direction not recognized. Specify either x or y")

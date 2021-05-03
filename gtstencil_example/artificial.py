from gtstencil_example import BACKEND, REBUILD, FIELD_FLOAT
from gt4py import gtscript
from gt4py.gtscript import PARALLEL, FORWARD, BACKWARD, computation, interval

@gtscript.stencil(backend = BACKEND, rebuild = REBUILD)
def add_one(x: FIELD_FLOAT):
    with computation(FORWARD):
        with interval(0, None):
            x = x + 1

@gtscript.stencil(backend = BACKEND, rebuild = REBUILD)
def mul_add(x: FIELD_FLOAT, 
            y: FIELD_FLOAT,
            z: FIELD_FLOAT,
            w: FIELD_FLOAT):
    """
    mul_add(x, y, z): z = x * y, w = w + z
    """
    with computation(FORWARD):
        with interval(0, None):
            z = x * y
    with computation(FORWARD):
        with interval(0, None):
            w = w + z

@gtscript.stencil(backend = BACKEND, rebuild = REBUILD)
def add_add(x: FIELD_FLOAT, 
            y: FIELD_FLOAT,
            z: FIELD_FLOAT,
            w: FIELD_FLOAT):
    """
    add_add(x, y, z): y = x + y, w = z + w
    """
    with computation(FORWARD):
        with interval(0, None):
            y = x + y
    with computation(FORWARD):
        with interval(0, None):
            w = w + z
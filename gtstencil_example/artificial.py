from gtstencil_example import BACKEND, REBUILD, FIELD_FLOAT
from gt4py import gtscript
from gt4py.gtscript import PARALLEL, FORWARD, BACKWARD, computation, interval

@gtscript.stencil(backend = BACKEND, rebuild = REBUILD)
def add_one(x: FIELD_FLOAT):
    with computation(FORWARD):
        with interval(0, None):
            x = x + 1
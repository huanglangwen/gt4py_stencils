from gtstencil_example import BACKEND, REBUILD, FIELD_FLOAT
from gt4py import gtscript
from gt4py.gtscript import PARALLEL, FORWARD, BACKWARD, computation, interval

@gtscript.stencil(backend = BACKEND, rebuild = REBUILD)
def thomas_solver_outofplace(
    a: FIELD_FLOAT,
    b: FIELD_FLOAT,
    c: FIELD_FLOAT,
    d: FIELD_FLOAT,
    c1: FIELD_FLOAT,
    d1: FIELD_FLOAT,
    x: FIELD_FLOAT
):
    """
    Non inplace version of Thomas solver for tridiagonal system of size n:

    [b0 c0            ] [x0]   [d0]
    [a1 b1 c1         ] [x1]   [d1]
    [   a2 b2 c2      ] [x2] = [d2]
    [      ...        ] [..]   [..]
    [         ... cn-1] [..]   [..]
    [            an bn] [xn]   [dn]

    All input arrays have the shape of (1, 1, n)
    Assume a[0] = c[n] = 0, even they are not zeros!
    c1 and d1 act as temporary buffer for storing intermediate results
    """
    with computation(FORWARD):
        with interval(0, 1):
            c1 = c/b
            d1 = d/b
        with interval(1, -1):
            c1 = c/(b - a*c1[0, 0, -1])
            d1 = (d - a*d1[0, 0, -1])/(b - a*c1[0, 0, -1])
        with interval(-1, None):
            d1 = (d - a*d1[0, 0, -1])/(b - a*c1[0, 0, -1])
    with computation(BACKWARD):
        with interval(-1, None):
            x = d1
        with interval(0, -1):
            x = d1 - c1*x[0, 0, 1]

@gtscript.stencil(backend = BACKEND, rebuild = REBUILD)
def thomas_solver_inplace(
    a: FIELD_FLOAT,
    b: FIELD_FLOAT,
    c: FIELD_FLOAT,
    d: FIELD_FLOAT,
    x: FIELD_FLOAT
):
    """
    Inplace version of Thomas solver for tridiagonal system of size n:

    [b0 c0            ] [x0]   [d0]
    [a1 b1 c1         ] [x1]   [d1]
    [   a2 b2 c2      ] [x2] = [d2]
    [      ...        ] [..]   [..]
    [         ... cn-1] [..]   [..]
    [            an bn] [xn]   [dn]

    All input arrays have the shape of (1, 1, n)
    Assume a[0] = c[n] = 0, even they are not zeros!
    !CAUTION!: b and d will be modified during the computation
    """
    with computation(FORWARD):
        with interval(1, None):
            w = a/b[0, 0, -1]
            b = b - w*c[0, 0, -1]
            d = d - w*d[0, 0, -1]
    with computation(BACKWARD):
        with interval(-1, None):
            x = d/b
        with interval(0, -1):
            x = (d - c*x[0, 0, 1])/b

@gtscript.stencil(backend = BACKEND, rebuild = REBUILD)
def thomas_solver_gt_inplace(
    a: FIELD_FLOAT,
    b: FIELD_FLOAT,
    c: FIELD_FLOAT,
    d: FIELD_FLOAT,
    x: FIELD_FLOAT
):
    """
    GridTools version of Inplace Thomas solver for tridiagonal system of size n:

    [b0 c0            ] [x0]   [d0]
    [a1 b1 c1         ] [x1]   [d1]
    [   a2 b2 c2      ] [x2] = [d2]
    [      ...        ] [..]   [..]
    [         ... cn-1] [..]   [..]
    [            an bn] [xn]   [dn]

    All input arrays have the shape of (1, 1, n)
    Assume a[0] = c[n] = 0, even they are not zeros!
    !CAUTION!: b and d will be modified during the computation
    """
    with computation(FORWARD):
        with interval(0, 1):
            w = 1/b
            c = c*w
            d = d*w
        with interval(1, -1):
            w = 1.0/(b - c[0, 0, -1]*a)
            c = c*w
            d = (d - d[0, 0, -1]*a)*w
        with interval(-1, None):
            w = 1.0/(b - c[0, 0, -1]*a)
            d = (d - d[0, 0, -1]*a)*w

    with computation(BACKWARD):
        with interval(-1, None):
            x = d
        with interval(0, -1):
            x = d - c*x[0, 0, 1]
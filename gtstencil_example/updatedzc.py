from gtstencil_example import BACKEND, REBUILD, FIELD_FLOAT

import gt4py.gtscript as gtscript
from gt4py.gtscript import BACKWARD, PARALLEL, computation, interval

import constants
import utils
from utils import copy, fill_4corners

origin = (1, 1, 0)
DZ_MIN = constants.DZ_MIN

@gtscript.function
def p_weighted_average_top(vel, dp0):
    # TODO: ratio is a constant, where should this be placed?
    ratio = dp0 / (dp0 + dp0[0, 0, 1])
    # return (1. + ratio) * vel - ratio * vel[0, 0, 1]
    return vel + (vel - vel[0, 0, 1]) * ratio


@gtscript.function
def p_weighted_average_bottom(vel, dp0):
    ratio = dp0[0, 0, -1] / (dp0[0, 0, -2] + dp0[0, 0, -1])
    # return (1. + ratio ) * vel[0, 0, -1] - ratio * vel[0, 0, -2]
    return vel[0, 0, -1] + (vel[0, 0, -1] - vel[0, 0, -2]) * ratio

@gtscript.function
def p_weighted_average_domain(vel, dp0):
    # ratio = dp0 / ( dp0[0, 0, -1] + dp0 )
    # return ratio * vel[0, 0, -1] + (1. - ratio) * vel
    int_ratio = 1.0 / (dp0[0, 0, -1] + dp0)
    return (dp0 * vel[0, 0, -1] + dp0[0, 0, -1] * vel) * int_ratio


@gtscript.function
def xy_flux(gz_x, gz_y, xfx, yfx):
    fx = xfx * (gz_x[-1, 0, 0] if xfx > 0.0 else gz_x)
    fy = yfx * (gz_y[0, -1, 0] if yfx > 0.0 else gz_y)
    return fx, fy

@gtscript.stencil(backend = BACKEND, rebuild = REBUILD)
def update_dz_c(
    dp_ref: FIELD_FLOAT,
    zs: FIELD_FLOAT,
    area: FIELD_FLOAT,
    ut: FIELD_FLOAT,
    vt: FIELD_FLOAT,
    gz: FIELD_FLOAT,
    gz_x: FIELD_FLOAT,
    gz_y: FIELD_FLOAT,
    ws3: FIELD_FLOAT,
    *,
    dt: float,
):
    with computation(PARALLEL):
        with interval(0, 1):
            xfx = p_weighted_average_top(ut, dp_ref)
            yfx = p_weighted_average_top(vt, dp_ref)
        with interval(1, -1):
            xfx = p_weighted_average_domain(ut, dp_ref)
            yfx = p_weighted_average_domain(vt, dp_ref)
        with interval(-1, None):
            xfx = p_weighted_average_bottom(ut, dp_ref)
            yfx = p_weighted_average_bottom(vt, dp_ref)
    with computation(PARALLEL), interval(...):
        fx, fy = xy_flux(gz_x, gz_y, xfx, yfx)
        # TODO: check if below gz is ok, or if we need gz_y to pass this
        gz = (gz_y * area + fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) / (
            area + xfx - xfx[1, 0, 0] + yfx - yfx[0, 1, 0]
        )
    with computation(PARALLEL), interval(-1, None):
        rdt = 1.0 / dt
        ws3 = (zs - gz) * rdt
    with computation(BACKWARD), interval(0, -1):
        gz_kp1 = gz[0, 0, 1] + DZ_MIN
        gz = gz if gz > gz_kp1 else gz_kp1


def compute(grid, dp_ref, zs, ut, vt, gz_in, ws3, dt2):
    # TODO: once we have a concept for corners, the following 4 lines should be refactored
    gz = copy(gz_in, origin=origin)
    gz_x = copy(gz, origin=origin)
    ws = copy(ws3, domain=grid.domain_shape_buffer_1cell())
    fill_4corners(gz_x, "x", grid)
    gz_y = copy(gz_x, origin=origin)
    fill_4corners(gz_y, "y", grid)
    update_dz_c(
        dp_ref,
        zs,
        grid.area,
        ut,
        vt,
        gz,
        gz_x,
        gz_y,
        ws3,
        dt=dt2,
        origin=origin,
        domain=(grid.nic + 3, grid.njc + 3, grid.npz + 1),
    )
    grid.overwrite_edges(gz, gz_in, 2, 2)
    grid.overwrite_edges(ws3, ws, 2, 2)
    return gz, ws3

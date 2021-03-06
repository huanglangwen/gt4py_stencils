import math

import numpy as np

from gtstencil_example import BACKEND, REBUILD, FIELD_FLOAT
from gt4py import gtscript
from gt4py.gtscript import FORWARD, PARALLEL, computation, interval, __INLINED

from gtstencil_example import sim1_solver
from gtstencil_example import constants
from gtstencil_example import utils
from gtstencil_example.utils import copy

@gtscript.stencil(backend = BACKEND, rebuild = REBUILD)
def precompute(
    cp3: FIELD_FLOAT,
    dm: FIELD_FLOAT,
    zh: FIELD_FLOAT,
    q_con: FIELD_FLOAT,
    pem: FIELD_FLOAT,
    peln: FIELD_FLOAT,
    pk3: FIELD_FLOAT,
    peg: FIELD_FLOAT,
    pelng: FIELD_FLOAT,
    gm: FIELD_FLOAT,
    dz: FIELD_FLOAT,
    pm: FIELD_FLOAT,
    ptop: float,
    peln1: float,
    ptk: float,
    rgrav: float,
    akap: float,
):
    with computation(FORWARD):
        with interval(0, 1):
            pem = ptop
            peln = peln1
            pk3 = ptk
            peg = ptop
            pelng = peln1
        with interval(1, None):
            # TODO consolidate with riem_solver_c, same functions, math functions
            pem = pem[0, 0, -1] + dm[0, 0, -1]
            peln = log(pem)
            peg = peg[0, 0, -1] + dm[0, 0, -1] * (1.0 - q_con[0, 0, -1])
            pelng = log(peg)
            pk3 = exp(akap * peln)
    with computation(PARALLEL), interval(...):
        gm = 1.0 / (1.0 - cp3)
        dm = dm * rgrav
    with computation(PARALLEL), interval(0, -1):
        pm = (peg[0, 0, 1] - peg) / (pelng[0, 0, 1] - pelng)
        dz = zh[0, 0, 1] - zh


@gtscript.stencil(backend = BACKEND, rebuild = REBUILD)
def last_call_copy(peln_run: FIELD_FLOAT, peln: FIELD_FLOAT, pk3: FIELD_FLOAT, pk: FIELD_FLOAT, pem: FIELD_FLOAT, pe: FIELD_FLOAT):
    with computation(PARALLEL), interval(...):
        peln = peln_run
        pk = pk3
        pe = pem

# https://github.com/VulcanClimateModeling/fv3gfs-fortran/blob/master/FV3/atmos_cubed_sphere/model/fv_arrays.F90
use_logp = False
beta = 0.0

@gtscript.stencil(backend = BACKEND, rebuild = REBUILD)
def finalize(
    zs: FIELD_FLOAT,
    dz: FIELD_FLOAT,
    zh: FIELD_FLOAT,
    peln_run: FIELD_FLOAT,
    peln: FIELD_FLOAT,
    pk3: FIELD_FLOAT,
    pk: FIELD_FLOAT,
    pem: FIELD_FLOAT,
    pe: FIELD_FLOAT,
    ppe: FIELD_FLOAT,
    pe_init: FIELD_FLOAT,
    last_call: bool,
):
    with computation(PARALLEL), interval(...):
        if __INLINED(use_logp):
            pk3 = peln_run
        if __INLINED(beta < -0.1):
            ppe = pe + pem
        else:
            ppe = pe
        if last_call:
            peln = peln_run
            pk = pk3
            pe = pem
        else:
            pe = pe_init
    with computation(BACKWARD):
        with interval(-1, None):
            zh = zs
        with interval(0, -1):
            zh = zh[0, 0, 1] - dz


def compute(
    grid,
    last_call,
    dt,
    akap,
    cappa,
    ptop,
    zs,
    w,
    delz,
    q_con,
    delp,
    pt,
    zh,
    pe,
    ppe,
    pk3,
    pk,
    peln,
    wsd,
):
    rgrav = 1.0 / constants.GRAV
    km = grid.npz - 1
    peln1 = math.log(ptop)
    ptk = math.exp(akap * peln1)
    islice = slice(grid.is_, grid.ie + 1)
    kslice = slice(0, km + 1)
    kslice_shift = slice(1, km + 2)
    shape = w.shape
    domain = (grid.nic, grid.njc, km + 2)
    riemorigin = (grid.is_, grid.js, 0)
    dm = copy(delp)
    cp3 = copy(cappa)
    pe_init = copy(pe)
    pm = utils.make_storage_from_shape(shape, riemorigin)
    pem = utils.make_storage_from_shape(shape, riemorigin)
    peln_run = utils.make_storage_from_shape(shape, riemorigin)
    peg = utils.make_storage_from_shape(shape, riemorigin)
    pelng = utils.make_storage_from_shape(shape, riemorigin)
    gm = utils.make_storage_from_shape(shape, riemorigin)
    precompute(
        cp3,
        dm,
        zh,
        q_con,
        pem,
        peln_run,
        pk3,
        peg,
        pelng,
        gm,
        delz,
        pm,
        ptop,
        peln1,
        ptk,
        rgrav,
        akap,
        origin=riemorigin,
        domain=domain,
    )
    sim1_solver.solve(
        grid,
        grid.is_,
        grid.ie,
        grid.js,
        grid.je,
        dt,
        gm,
        cp3,
        pe,
        dm,
        pm,
        pem,
        w,
        delz,
        pt,
        wsd,
    )

    finalize(
        zs,
        delz,
        zh,
        peln_run,
        peln,
        pk3,
        pk,
        pem,
        pe,
        ppe,
        pe_init,
        last_call,
        origin=riemorigin,
        domain=domain,
    )

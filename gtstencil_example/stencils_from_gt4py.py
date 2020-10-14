from gtstencil_example import BACKEND, REBUILD, FIELD_FLOAT
from gt4py import gtscript
from gt4py.gtscript import PARALLEL, FORWARD, BACKWARD, computation, interval

@gtscript.stencil(backend = BACKEND, rebuild = REBUILD)
def fix_interior(
    q: FIELD_FLOAT, dp: FIELD_FLOAT, zfix: FIELD_FLOAT, upper_fix: FIELD_FLOAT, lower_fix: FIELD_FLOAT, dm: FIELD_FLOAT, dm_pos: FIELD_FLOAT
):
    with computation(FORWARD), interval(1, -1):
        # if a higher layer borrowed from this one, account for that here
        if lower_fix[0, 0, -1] != 0.0:
            q = q - (lower_fix[0, 0, -1] / dp)
        dq = q * dp
        if q < 0.0:
            zfix = 1.0
            if q[0, 0, -1] > 0.0:
                # Borrow from the layer above
                dq = (
                    q[0, 0, -1] * dp[0, 0, -1]
                    if q[0, 0, -1] * dp[0, 0, -1] < -(q * dp)
                    else -(q * dp)
                )
                q = q + dq / dp
                upper_fix = dq
            if (q < 0.0) and (q[0, 0, 1] > 0.0):
                # borrow from the layer below
                dq = (
                    q[0, 0, 1] * dp[0, 0, 1]
                    if q[0, 0, 1] * dp[0, 0, 1] < -(q * dp)
                    else -(q * dp)
                )
                q = q + dq / dp
                lower_fix = dq
    with computation(PARALLEL), interval(...):
        if upper_fix[0, 0, 1] != 0.0:
            # If a lower layer borrowed from this one, account for that here
            q = q - upper_fix[0, 0, 1] / dp
        dm = q * dp
        dm_pos = dm if dm > 0.0 else 0.0

@gtscript.stencil(backend = BACKEND, rebuild = REBUILD)
def sim1_solver(
    w: FIELD_FLOAT,
    dm: FIELD_FLOAT,
    gm: FIELD_FLOAT,
    dz: FIELD_FLOAT,
    ptr: FIELD_FLOAT,
    pm: FIELD_FLOAT,
    pe: FIELD_FLOAT,
    pem: FIELD_FLOAT,
    wsr: FIELD_FLOAT,
    cp3: FIELD_FLOAT,
    dt: float,
    t1g: float,
    rdt: float,
    p_fac: float,
):
    # TODO: we only want to bottom level of wsr, so this could be removed once wsr_top is a 2d field
    with computation(FORWARD):
        with interval(0, 1):
            wsr_top = wsr
        with interval(1, None):
            wsr_top = wsr_top[0, 0, -1]

    with computation(PARALLEL), interval(0, -1):
        pe = exp(gm * log(-dm / dz * constants.RDGAS * ptr)) - pm
        w1 = w
    with computation(FORWARD):
        with interval(0, -2):
            g_rat = dm / dm[0, 0, 1]
            bb = 2.0 * (1.0 + g_rat)
            dd = 3.0 * (pe + g_rat * pe[0, 0, 1])
        with interval(-2, -1):
            bb = 2.0
            dd = 3.0 * pe
    # bet[i,j,k] = bb[i,j,0]
    with computation(FORWARD):
        with interval(0, 1):
            bet = bb
        with interval(1, -1):
            bet = bet[0, 0, -1]

    ### stencils: w_solver
    # {
    with computation(PARALLEL):
        with interval(0, 1):
            pp = 0.0
        with interval(1, 2):
            pp = dd[0, 0, -1] / bet
    with computation(FORWARD), interval(1, -1):
        gam = g_rat[0, 0, -1] / bet[0, 0, -1]
        bet = bb - gam
    with computation(FORWARD), interval(2, None):
        pp = (dd[0, 0, -1] - pp[0, 0, -1]) / bet[0, 0, -1]
    with computation(BACKWARD), interval(1, -1):
        pp = pp - gam * pp[0, 0, 1]
        # w solver
        aa = t1g * 0.5 * (gm[0, 0, -1] + gm) / (dz[0, 0, -1] + dz) * (pem + pp)
    # }
    ## updates on bet:
    with computation(FORWARD):
        with interval(0, 1):
            bet = dm[0, 0, 0] - aa[0, 0, 1]
        with interval(1, None):
            bet = bet[0, 0, -1]
    ### w_pe_dz_compute
    # {
    with computation(FORWARD):
        with interval(0, 1):
            w = (dm * w1 + dt * pp[0, 0, 1]) / bet
        with interval(1, -2):
            gam = aa / bet[0, 0, -1]
            bet = dm - (aa + aa[0, 0, 1] + aa * gam)
            w = (dm * w1 + dt * (pp[0, 0, 1] - pp) - aa * w[0, 0, -1]) / bet
        with interval(-2, -1):
            p1 = t1g * gm / dz * (pem[0, 0, 1] + pp[0, 0, 1])
            gam = aa / bet[0, 0, -1]
            bet = dm - (aa + p1 + aa * gam)
            w = (
                dm * w1 + dt * (pp[0, 0, 1] - pp) - p1 * wsr_top - aa * w[0, 0, -1]
            ) / bet
    with computation(BACKWARD), interval(0, -2):
        w = w - gam[0, 0, 1] * w[0, 0, 1]
    with computation(FORWARD):
        with interval(0, 1):
            pe = 0.0
        with interval(1, None):
            pe = pe[0, 0, -1] + dm[0, 0, -1] * (w[0, 0, -1] - w1[0, 0, -1]) * rdt
    with computation(BACKWARD):
        with interval(-2, -1):
            p1 = (pe + 2.0 * pe[0, 0, 1]) * 1.0 / 3.0
        with interval(0, -2):
            p1 = (pe + bb * pe[0, 0, 1] + g_rat * pe[0, 0, 2]) * 1.0 / 3.0 - g_rat * p1[
                0, 0, 1
            ]
    with computation(PARALLEL), interval(0, -1):
        maxp = p_fac * pm if p_fac * dm > p1 + pm else p1 + pm
        dz = -dm * constants.RDGAS * ptr * exp((cp3 - 1.0) * log(maxp))
    # }

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

@gtscript.stencil(backend = BACKEND, rebuild = REBUILD)
def satadjust_part1(
    wqsat: FIELD_FLOAT,
    dq2dt: FIELD_FLOAT,
    dpln: FIELD_FLOAT,
    den: FIELD_FLOAT,
    pt1: FIELD_FLOAT,
    cvm: FIELD_FLOAT,
    mc_air: FIELD_FLOAT,
    peln: FIELD_FLOAT,
    qv: FIELD_FLOAT,
    ql: FIELD_FLOAT,
    q_liq: FIELD_FLOAT,
    qi: FIELD_FLOAT,
    qr: FIELD_FLOAT,
    qs: FIELD_FLOAT,
    q_sol: FIELD_FLOAT,
    qg: FIELD_FLOAT,
    pt: FIELD_FLOAT,
    dp: FIELD_FLOAT,
    delz: FIELD_FLOAT,
    te0: FIELD_FLOAT,
    qpz: FIELD_FLOAT,
    lhl: FIELD_FLOAT,
    lhi: FIELD_FLOAT,
    lcp2: FIELD_FLOAT,
    icp2: FIELD_FLOAT,
    tcp3: FIELD_FLOAT,
    zvir: float,
    hydrostatic: bool,
    consv_te: bool,
    c_air: float,
    c_vap: float,
    fac_imlt: float,
    d0_vap: float,
    lv00: float,
    fac_v2l: float,
    fac_l2v: float,
):
    with computation(FORWARD), interval(1, None):
        if hydrostatic:
            delz = delz[0, 0, -1]
    with computation(PARALLEL), interval(...):
        dpln = peln[0, 0, 1] - peln
        q_liq = ql + qr
        q_sol = qi + qs + qg
        qpz = q_liq + q_sol
        pt1 = pt / ((1.0 + zvir * qv) * (1.0 - qpz))
        t0 = pt1  # true temperature
        qpz = qpz + qv  # total_wat conserved in this routine
        # define air density based on hydrostatical property
        den = (
            dp / (dpln * constants.RDGAS * pt)
            if hydrostatic
            else -dp / (constants.GRAV * delz)
        )
        # define heat capacity and latend heat coefficient
        mc_air = (1.0 - qpz) * c_air
        cvm = compute_cvm(mc_air, qv, c_vap, q_liq, q_sol)
        lhi, icp2 = update_latent_heat_coefficient_i(pt1, cvm)
        #  fix energy conservation
        if consv_te:
            if hydrostatic:
                te0 = -c_air * t0
            else:
                te0 = -cvm * t0
        # fix negative cloud ice with snow
        if qi < 0.0:
            qs = qs + qi
            qi = 0.0

        #  melting of cloud ice to cloud water and rain
        qi, ql, q_liq, q_sol, cvm, pt1 = melt_cloud_ice(
            qv, qi, ql, q_liq, q_sol, pt1, icp2, fac_imlt, mc_air, c_vap, lhi, cvm
        )
        # update latend heat coefficient
        lhi, icp2 = update_latent_heat_coefficient_i(pt1, cvm)
        # fix negative snow with graupel or graupel with available snow
        qs, qg = fix_negative_snow(qs, qg)
        # after this point cloud ice & snow are positive definite
        # fix negative cloud water with rain or rain with available cloud water
        ql, qr = fix_negative_cloud_water(ql, qr)
        # enforce complete freezing of cloud water to cloud ice below - 48 c
        ql, qi, q_liq, q_sol, cvm, pt1 = complete_freezing(
            qv, ql, qi, q_liq, q_sol, pt1, cvm, icp2, mc_air, lhi, c_vap
        )
        wqsat, dq2dt = wqs2_fn_w(pt1, den)
        # update latent heat coefficient
        lhl, lhi, lcp2, icp2 = update_latent_heat_coefficient(pt1, cvm, lv00, d0_vap)
        diff_ice = dim(TICE, pt1) / 48.0
        dimmin = min(1.0, diff_ice)
        tcp3 = lcp2 + icp2 * dimmin

        dq0 = (qv - wqsat) / (
            1.0 + tcp3 * dq2dt
        )  # compute_dq0(qv, wqsat, dq2dt, tcp3)  #(qv - wqsat) / (1.0 + tcp3 * dq2dt)
        # TODO might be able to get rid of these temporary allocations when not used?
        if dq0 > 0:  # whole grid - box saturated
            src = min(
                spec.namelist.sat_adj0 * dq0,
                max(spec.namelist.ql_gen - ql, fac_v2l * dq0),
            )
        else:
            # TODO -- we'd like to use this abstraction rather than duplicate code, but inside the if conditional complains 'not implemented'
            # factor, src = ql_evaporation(wqsat, qv, ql, dq0,fac_l2v)
            factor = -1.0 * min(1, fac_l2v * 10.0 * (1.0 - qv / wqsat))
            src = -1.0 * min(ql, factor * dq0)

        qv, ql, q_liq, cvm, pt1 = wqsat_correct(
            src, pt1, lhl, qv, ql, q_liq, q_sol, mc_air, c_vap
        )
        # update latent heat coefficient
        lhl, lhi, lcp2, icp2 = update_latent_heat_coefficient(pt1, cvm, lv00, d0_vap)
        # TODO remove duplicate
        diff_ice = dim(TICE, pt1) / 48.0
        dimmin = min(1.0, diff_ice)
        tcp3 = lcp2 + icp2 * dimmin
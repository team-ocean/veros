from veros.core.operators import numpy as npx
from veros import logger

from veros import veros_kernel, veros_routine, KernelOutput
from veros.variables import allocate
from veros.core import density, utilities
from veros.core.operators import update, update_add, at


@veros_kernel
def dm_taper(sx, iso_slopec, iso_dslope):
    """
    tapering function for isopycnal slopes
    """
    return 0.5 * (1.0 + npx.tanh((-npx.abs(sx) + iso_slopec) / iso_dslope))


@veros_kernel
def isoneutral_diffusion_pre(state):
    """
    Isopycnal diffusion for tracer
    following functional formulation by Griffies et al
    Code adopted from MOM2.1
    """
    vs = state.variables
    settings = state.settings

    epsln = 1e-20

    dTdx = allocate(state.dimensions, ("xt", "yt", "zt"))
    dSdx = allocate(state.dimensions, ("xt", "yt", "zt"))
    dTdy = allocate(state.dimensions, ("xt", "yt", "zt"))
    dSdy = allocate(state.dimensions, ("xt", "yt", "zt"))
    dTdz = allocate(state.dimensions, ("xt", "yt", "zt"))
    dSdz = allocate(state.dimensions, ("xt", "yt", "zt"))

    """
    drho_dt and drho_ds at centers of T cells
    """
    drdT = vs.maskT * density.get_drhodT(state, vs.salt[:, :, :, vs.tau], vs.temp[:, :, :, vs.tau], npx.abs(vs.zt))
    drdS = vs.maskT * density.get_drhodS(state, vs.salt[:, :, :, vs.tau], vs.temp[:, :, :, vs.tau], npx.abs(vs.zt))

    """
    gradients at top face of T cells
    """
    dTdz = update(
        dTdz,
        at[:, :, :-1],
        vs.maskW[:, :, :-1]
        * (vs.temp[:, :, 1:, vs.tau] - vs.temp[:, :, :-1, vs.tau])
        / vs.dzw[npx.newaxis, npx.newaxis, :-1],
    )
    dSdz = update(
        dSdz,
        at[:, :, :-1],
        vs.maskW[:, :, :-1]
        * (vs.salt[:, :, 1:, vs.tau] - vs.salt[:, :, :-1, vs.tau])
        / vs.dzw[npx.newaxis, npx.newaxis, :-1],
    )

    """
    gradients at eastern face of T cells
    """
    dTdx = update(
        dTdx,
        at[:-1, :, :],
        vs.maskU[:-1, :, :]
        * (vs.temp[1:, :, :, vs.tau] - vs.temp[:-1, :, :, vs.tau])
        / (vs.dxu[:-1, npx.newaxis, npx.newaxis] * vs.cost[npx.newaxis, :, npx.newaxis]),
    )
    dSdx = update(
        dSdx,
        at[:-1, :, :],
        vs.maskU[:-1, :, :]
        * (vs.salt[1:, :, :, vs.tau] - vs.salt[:-1, :, :, vs.tau])
        / (vs.dxu[:-1, npx.newaxis, npx.newaxis] * vs.cost[npx.newaxis, :, npx.newaxis]),
    )

    """
    gradients at northern face of T cells
    """
    dTdy = update(
        dTdy,
        at[:, :-1, :],
        vs.maskV[:, :-1, :]
        * (vs.temp[:, 1:, :, vs.tau] - vs.temp[:, :-1, :, vs.tau])
        / vs.dyu[npx.newaxis, :-1, npx.newaxis],
    )
    dSdy = update(
        dSdy,
        at[:, :-1, :],
        vs.maskV[:, :-1, :]
        * (vs.salt[:, 1:, :, vs.tau] - vs.salt[:, :-1, :, vs.tau])
        / vs.dyu[npx.newaxis, :-1, npx.newaxis],
    )

    """
    Compute Ai_ez and K11 on center of east face of T cell.
    """
    diffloc = allocate(state.dimensions, ("xt", "yt", "zt"))
    diffloc = update(
        diffloc,
        at[1:-2, 2:-2, 1:],
        0.25
        * (vs.K_iso[1:-2, 2:-2, 1:] + vs.K_iso[1:-2, 2:-2, :-1] + vs.K_iso[2:-1, 2:-2, 1:] + vs.K_iso[2:-1, 2:-2, :-1]),
    )
    diffloc = update(diffloc, at[1:-2, 2:-2, 0], 0.5 * (vs.K_iso[1:-2, 2:-2, 0] + vs.K_iso[2:-1, 2:-2, 0]))

    sumz = allocate(state.dimensions, ("xt", "yt", "zt"))[1:-2, 2:-2]
    for kr in range(2):
        ki = 0 if kr == 1 else 1
        for ip in range(2):
            drodxe = (
                drdT[1 + ip : -2 + ip, 2:-2, ki:] * dTdx[1:-2, 2:-2, ki:]
                + drdS[1 + ip : -2 + ip, 2:-2, ki:] * dSdx[1:-2, 2:-2, ki:]
            )
            drodze = (
                drdT[1 + ip : -2 + ip, 2:-2, ki:] * dTdz[1 + ip : -2 + ip, 2:-2, : -1 + kr or None]
                + drdS[1 + ip : -2 + ip, 2:-2, ki:] * dSdz[1 + ip : -2 + ip, 2:-2, : -1 + kr or None]
            )
            sxe = -drodxe / (npx.minimum(0.0, drodze) - epsln)
            taper = dm_taper(sxe, settings.iso_slopec, settings.iso_dslope)
            sumz = update_add(
                sumz,
                at[:, :, ki:],
                vs.dzw[npx.newaxis, npx.newaxis, : -1 + kr or None]
                * vs.maskU[1:-2, 2:-2, ki:]
                * npx.maximum(settings.K_iso_steep, diffloc[1:-2, 2:-2, ki:] * taper),
            )
            vs.Ai_ez = update(vs.Ai_ez, at[1:-2, 2:-2, ki:, ip, kr], taper * sxe * vs.maskU[1:-2, 2:-2, ki:])

    vs.K_11 = update(vs.K_11, at[1:-2, 2:-2, :], sumz / (4.0 * vs.dzt[npx.newaxis, npx.newaxis, :]))

    """
    Compute Ai_nz and K_22 on center of north face of T cell.
    """
    diffloc = update(diffloc, at[...], 0)
    diffloc = update(
        diffloc,
        at[2:-2, 1:-2, 1:],
        0.25
        * (vs.K_iso[2:-2, 1:-2, 1:] + vs.K_iso[2:-2, 1:-2, :-1] + vs.K_iso[2:-2, 2:-1, 1:] + vs.K_iso[2:-2, 2:-1, :-1]),
    )
    diffloc = update(diffloc, at[2:-2, 1:-2, 0], 0.5 * (vs.K_iso[2:-2, 1:-2, 0] + vs.K_iso[2:-2, 2:-1, 0]))

    sumz = allocate(state.dimensions, ("xt", "yt", "zt"))[2:-2, 1:-2]
    for kr in range(2):
        ki = 0 if kr == 1 else 1
        for jp in range(2):
            drodyn = (
                drdT[2:-2, 1 + jp : -2 + jp, ki:] * dTdy[2:-2, 1:-2, ki:]
                + drdS[2:-2, 1 + jp : -2 + jp, ki:] * dSdy[2:-2, 1:-2, ki:]
            )
            drodzn = (
                drdT[2:-2, 1 + jp : -2 + jp, ki:] * dTdz[2:-2, 1 + jp : -2 + jp, : -1 + kr or None]
                + drdS[2:-2, 1 + jp : -2 + jp, ki:] * dSdz[2:-2, 1 + jp : -2 + jp, : -1 + kr or None]
            )
            syn = -drodyn / (npx.minimum(0.0, drodzn) - epsln)
            taper = dm_taper(syn, settings.iso_slopec, settings.iso_dslope)
            sumz = update_add(
                sumz,
                at[:, :, ki:],
                vs.dzw[npx.newaxis, npx.newaxis, : -1 + kr or None]
                * vs.maskV[2:-2, 1:-2, ki:]
                * npx.maximum(settings.K_iso_steep, diffloc[2:-2, 1:-2, ki:] * taper),
            )
            vs.Ai_nz = update(vs.Ai_nz, at[2:-2, 1:-2, ki:, jp, kr], taper * syn * vs.maskV[2:-2, 1:-2, ki:])
    vs.K_22 = update(vs.K_22, at[2:-2, 1:-2, :], sumz / (4.0 * vs.dzt[npx.newaxis, npx.newaxis, :]))

    """
    compute Ai_bx, Ai_by and K33 on top face of T cell.
    """
    sumx = allocate(state.dimensions, ("xt", "yt", "zt"))[2:-2, 2:-2, :-1]
    sumy = allocate(state.dimensions, ("xt", "yt", "zt"))[2:-2, 2:-2, :-1]

    for kr in range(2):
        drodzb = (
            drdT[2:-2, 2:-2, kr : -1 + kr or None] * dTdz[2:-2, 2:-2, :-1]
            + drdS[2:-2, 2:-2, kr : -1 + kr or None] * dSdz[2:-2, 2:-2, :-1]
        )

        # eastward slopes at the top of T cells
        for ip in range(2):
            drodxb = (
                drdT[2:-2, 2:-2, kr : -1 + kr or None] * dTdx[1 + ip : -3 + ip, 2:-2, kr : -1 + kr or None]
                + drdS[2:-2, 2:-2, kr : -1 + kr or None] * dSdx[1 + ip : -3 + ip, 2:-2, kr : -1 + kr or None]
            )
            sxb = -drodxb / (npx.minimum(0.0, drodzb) - epsln)
            taper = dm_taper(sxb, settings.iso_slopec, settings.iso_dslope)
            sumx = (
                sumx
                + vs.dxu[1 + ip : -3 + ip, npx.newaxis, npx.newaxis]
                * vs.K_iso[2:-2, 2:-2, :-1]
                * taper
                * sxb ** 2
                * vs.maskW[2:-2, 2:-2, :-1]
            )
            vs.Ai_bx = update(vs.Ai_bx, at[2:-2, 2:-2, :-1, ip, kr], taper * sxb * vs.maskW[2:-2, 2:-2, :-1])

        # northward slopes at the top of T cells
        for jp in range(2):
            facty = vs.cosu[1 + jp : -3 + jp] * vs.dyu[1 + jp : -3 + jp]
            drodyb = (
                drdT[2:-2, 2:-2, kr : -1 + kr or None] * dTdy[2:-2, 1 + jp : -3 + jp, kr : -1 + kr or None]
                + drdS[2:-2, 2:-2, kr : -1 + kr or None] * dSdy[2:-2, 1 + jp : -3 + jp, kr : -1 + kr or None]
            )
            syb = -drodyb / (npx.minimum(0.0, drodzb) - epsln)
            taper = dm_taper(syb, settings.iso_slopec, settings.iso_dslope)
            sumy = (
                sumy
                + facty[npx.newaxis, :, npx.newaxis]
                * vs.K_iso[2:-2, 2:-2, :-1]
                * taper
                * syb ** 2
                * vs.maskW[2:-2, 2:-2, :-1]
            )
            vs.Ai_by = update(vs.Ai_by, at[2:-2, 2:-2, :-1, jp, kr], taper * syb * vs.maskW[2:-2, 2:-2, :-1])

    vs.K_33 = update(
        vs.K_33,
        at[2:-2, 2:-2, :-1],
        sumx / (4 * vs.dxt[2:-2, npx.newaxis, npx.newaxis])
        + sumy / (4 * vs.dyt[npx.newaxis, 2:-2, npx.newaxis] * vs.cost[npx.newaxis, 2:-2, npx.newaxis]),
    )
    vs.K_33 = update(vs.K_33, at[2:-2, 2:-2, -1], 0.0)

    return KernelOutput(
        Ai_ez=vs.Ai_ez, Ai_nz=vs.Ai_nz, Ai_bx=vs.Ai_bx, Ai_by=vs.Ai_by, K_11=vs.K_11, K_22=vs.K_22, K_33=vs.K_33
    )


@veros_kernel
def isoneutral_diag_streamfunction_kernel(state):
    vs = state.variables

    K_gm_pad = utilities.pad_z_edges(vs.K_gm)

    """
    meridional component at east face of 'T' cells
    """
    diffloc = 0.25 * (
        K_gm_pad[1:-2, 2:-2, 1:-1] + K_gm_pad[1:-2, 2:-2, :-2] + K_gm_pad[2:-1, 2:-2, 1:-1] + K_gm_pad[2:-1, 2:-2, :-2]
    )
    vs.B2_gm = update(vs.B2_gm, at[1:-2, 2:-2, :], 0.25 * diffloc * npx.sum(vs.Ai_ez[1:-2, 2:-2, ...], axis=(3, 4)))

    """
    zonal component at north face of 'T' cells
    """
    diffloc = 0.25 * (
        K_gm_pad[2:-2, 1:-2, 1:-1] + K_gm_pad[2:-2, 1:-2, :-2] + K_gm_pad[2:-2, 2:-1, 1:-1] + K_gm_pad[2:-2, 2:-1, :-2]
    )
    vs.B1_gm = update(vs.B1_gm, at[2:-2, 1:-2, :], -0.25 * diffloc * npx.sum(vs.Ai_nz[2:-2, 1:-2, ...], axis=(3, 4)))

    return KernelOutput(B1_gm=vs.B1_gm, B2_gm=vs.B2_gm)


@veros_routine
def isoneutral_diag_streamfunction(state):
    """
    calculate hor. components of streamfunction for eddy driven velocity
    for diagnostics purpose only
    """
    vs = state.variables
    settings = state.settings

    if not (settings.enable_neutral_diffusion and settings.enable_skew_diffusion):
        return

    vs.update(isoneutral_diag_streamfunction_kernel(state))


@veros_routine
def check_isoneutral_slope_crit(state):
    """
    check linear stability criterion from Griffies et al
    """
    vs = state.variables
    settings = state.settings

    epsln = 1e-20
    if settings.enable_neutral_diffusion:
        ft1 = 1.0 / (4.0 * settings.K_iso_0 * settings.dt_tracer + epsln)
        delta1a = npx.min(
            vs.dxt[2:-2, npx.newaxis, npx.newaxis]
            * npx.abs(vs.cost[npx.newaxis, 2:-2, npx.newaxis])
            * vs.dzt[npx.newaxis, npx.newaxis, :]
            * ft1
        )
        delta1b = npx.min(vs.dyt[npx.newaxis, 2:-2, npx.newaxis] * vs.dzt[npx.newaxis, npx.newaxis, :] * ft1)
        delta_iso1 = min(vs.dzt[0] * ft1 * vs.dxt[-1] * abs(vs.cost[-1]), min(delta1a, delta1b))

        logger.info("Diffusion grid factor delta_iso1 = {}", float(delta_iso1))
        if delta_iso1 < settings.iso_slopec:
            raise RuntimeError(
                "Without latitudinal filtering, delta_iso1 is the steepest "
                "isoneutral slope available for linear stability of "
                "Redi and GM. Maximum allowable isoneutral slope is "
                f"specified as iso_slopec = {settings.iso_slopec}."
            )

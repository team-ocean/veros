from veros.core.operators import numpy as np
from loguru import logger

from veros import veros_kernel, veros_routine, run_kernel
from veros.core import density, utilities
from veros.core.operators import update, update_add, at, tanh


@veros_kernel
def dm_taper(sx, iso_slopec, iso_dslope):
    """
    tapering function for isopycnal slopes
    """
    return 0.5 * (1. + tanh((-np.abs(sx) + iso_slopec) / iso_dslope))


@veros_kernel
def isoneutral_diffusion_pre(salt, temp, zt, dxt, dxu, dyt, dyu, dzt, dzw, maskT, maskU,
                             maskV, maskW, cost, cosu, iso_slopec, iso_dslope, K_iso,
                             K_iso_steep, K_11, K_22, K_33, Ai_ez, Ai_nz, Ai_bx, Ai_by,
                             tau, eq_of_state_type):
    """
    Isopycnal diffusion for tracer
    following functional formulation by Griffies et al
    Code adopted from MOM2.1
    """
    epsln = 1e-20

    dTdx = np.zeros_like(maskU)
    dSdx = np.zeros_like(maskU)
    dTdy = np.zeros_like(maskV)
    dSdy = np.zeros_like(maskV)
    dTdz = np.zeros_like(maskW)
    dSdz = np.zeros_like(maskW)

    """
    drho_dt and drho_ds at centers of T cells
    """
    drdT = maskT * density.get_drhodT(
        eq_of_state_type, salt[:, :, :, tau], temp[:, :, :, tau], np.abs(zt)
    )
    drdS = maskT * density.get_drhodS(
        eq_of_state_type, salt[:, :, :, tau], temp[:, :, :, tau], np.abs(zt)
    )

    """
    gradients at top face of T cells
    """
    dTdz = update(dTdz, at[:, :, :-1], maskW[:, :, :-1] * \
        (temp[:, :, 1:, tau] - temp[:, :, :-1, tau]) / \
        dzw[np.newaxis, np.newaxis, :-1])
    dSdz = update(dSdz, at[:, :, :-1], maskW[:, :, :-1] * \
        (salt[:, :, 1:, tau] - salt[:, :, :-1, tau]) / \
        dzw[np.newaxis, np.newaxis, :-1])

    """
    gradients at eastern face of T cells
    """
    dTdx = update(dTdx, at[:-1, :, :], maskU[:-1, :, :] * (temp[1:, :, :, tau] - temp[:-1, :, :, tau]) \
        / (dxu[:-1, np.newaxis, np.newaxis] * cost[np.newaxis, :, np.newaxis]))
    dSdx = update(dSdx, at[:-1, :, :], maskU[:-1, :, :] * (salt[1:, :, :, tau] - salt[:-1, :, :, tau]) \
        / (dxu[:-1, np.newaxis, np.newaxis] * cost[np.newaxis, :, np.newaxis]))

    """
    gradients at northern face of T cells
    """
    dTdy = update(dTdy, at[:, :-1, :], maskV[:, :-1, :] * \
        (temp[:, 1:, :, tau] - temp[:, :-1, :, tau]) \
        / dyu[np.newaxis, :-1, np.newaxis])
    dSdy = update(dSdy, at[:, :-1, :], maskV[:, :-1, :] * \
        (salt[:, 1:, :, tau] - salt[:, :-1, :, tau]) \
        / dyu[np.newaxis, :-1, np.newaxis])


    """
    Compute Ai_ez and K11 on center of east face of T cell.
    """
    diffloc = np.zeros_like(maskT)
    diffloc = update(diffloc, at[1:-2, 2:-2, 1:], 0.25 * (K_iso[1:-2, 2:-2, 1:] + K_iso[1:-2, 2:-2, :-1]
                                      + K_iso[2:-1, 2:-2, 1:] + K_iso[2:-1, 2:-2, :-1]))
    diffloc = update(diffloc, at[1:-2, 2:-2, 0], 0.5 * (K_iso[1:-2, 2:-2, 0] + K_iso[2:-1, 2:-2, 0]))

    sumz = np.zeros_like(maskU)[1:-2, 2:-2]
    for kr in range(2):
        ki = 0 if kr == 1 else 1
        for ip in range(2):
            drodxe = drdT[1 + ip:-2 + ip, 2:-2, ki:] * dTdx[1:-2, 2:-2, ki:] \
                + drdS[1 + ip:-2 + ip, 2:-2, ki:] * dSdx[1:-2, 2:-2, ki:]
            drodze = drdT[1 + ip:-2 + ip, 2:-2, ki:] * dTdz[1 + ip:-2 + ip, 2:-2, :-1 + kr or None] \
                + drdS[1 + ip:-2 + ip, 2:-2, ki:] * dSdz[1 + ip:-2 + ip, 2:-2, :-1 + kr or None]
            sxe = -drodxe / (np.minimum(0., drodze) - epsln)
            taper = dm_taper(sxe, iso_slopec, iso_dslope)
            sumz = update_add(sumz, at[:, :, ki:], dzw[np.newaxis, np.newaxis, :-1 + kr or None] * maskU[1:-2, 2:-2, ki:] \
                * np.maximum(K_iso_steep, diffloc[1:-2, 2:-2, ki:] * taper))
            Ai_ez = update(Ai_ez, at[1:-2, 2:-2, ki:, ip, kr], taper * sxe * maskU[1:-2, 2:-2, ki:])
    K_11 = update(K_11, at[1:-2, 2:-2, :], sumz / (4. * dzt[np.newaxis, np.newaxis, :]))

    """
    Compute Ai_nz and K_22 on center of north face of T cell.
    """
    diffloc = update(diffloc, at[...], 0)
    diffloc = update(diffloc, at[2:-2, 1:-2, 1:], 0.25 * (K_iso[2:-2, 1:-2, 1:] + K_iso[2:-2, 1:-2, :-1]
                                      + K_iso[2:-2, 2:-1, 1:] + K_iso[2:-2, 2:-1, :-1]))
    diffloc = update(diffloc, at[2:-2, 1:-2, 0], 0.5 * (K_iso[2:-2, 1:-2, 0] + K_iso[2:-2, 2:-1, 0]))

    sumz = np.zeros_like(maskU)[2:-2, 1:-2]
    for kr in range(2):
        ki = 0 if kr == 1 else 1
        for jp in range(2):
            drodyn = drdT[2:-2, 1 + jp:-2 + jp, ki:] * dTdy[2:-2, 1:-2, ki:] + \
                drdS[2:-2, 1 + jp:-2 + jp, ki:] * dSdy[2:-2, 1:-2, ki:]
            drodzn = drdT[2:-2, 1 + jp:-2 + jp, ki:] * dTdz[2:-2, 1 + jp:-2 + jp, :-1 + kr or None] \
                + drdS[2:-2, 1 + jp:-2 + jp, ki:] * dSdz[2:-2, 1 + jp:-2 + jp, :-1 + kr or None]
            syn = -drodyn / (np.minimum(0., drodzn) - epsln)
            taper = dm_taper(syn, iso_slopec, iso_dslope)
            sumz = update_add(sumz, at[:, :, ki:], dzw[np.newaxis, np.newaxis, :-1 + kr or None] \
                * maskV[2:-2, 1:-2, ki:] * np.maximum(K_iso_steep, diffloc[2:-2, 1:-2, ki:] * taper))
            Ai_nz = update(Ai_nz, at[2:-2, 1:-2, ki:, jp, kr], taper * syn * maskV[2:-2, 1:-2, ki:])
    K_22 = update(K_22, at[2:-2, 1:-2, :], sumz / (4. * dzt[np.newaxis, np.newaxis, :]))

    """
    compute Ai_bx, Ai_by and K33 on top face of T cell.
    """
    sumx = np.zeros_like(maskT)[2:-2, 2:-2, :-1]
    sumy = np.zeros_like(maskT)[2:-2, 2:-2, :-1]

    for kr in range(2):
        drodzb = drdT[2:-2, 2:-2, kr:-1 + kr or None] * dTdz[2:-2, 2:-2, :-1] \
            + drdS[2:-2, 2:-2, kr:-1 + kr or None] * dSdz[2:-2, 2:-2, :-1]

        # eastward slopes at the top of T cells
        for ip in range(2):
            drodxb = drdT[2:-2, 2:-2, kr:-1 + kr or None] * dTdx[1 + ip:-3 + ip, 2:-2, kr:-1 + kr or None] \
                + drdS[2:-2, 2:-2, kr:-1 + kr or None] * dSdx[1 + ip:-3 + ip, 2:-2, kr:-1 + kr or None]
            sxb = -drodxb / (np.minimum(0., drodzb) - epsln)
            taper = dm_taper(sxb, iso_slopec, iso_dslope)
            sumx += dxu[1 + ip:-3 + ip, np.newaxis, np.newaxis] * \
                K_iso[2:-2, 2:-2, :-1] * taper * sxb**2 * maskW[2:-2, 2:-2, :-1]
            Ai_bx = update(Ai_bx, at[2:-2, 2:-2, :-1, ip, kr], taper * sxb * maskW[2:-2, 2:-2, :-1])

        # northward slopes at the top of T cells
        for jp in range(2):
            facty = cosu[1 + jp:-3 + jp] * dyu[1 + jp:-3 + jp]
            drodyb = drdT[2:-2, 2:-2, kr:-1 + kr or None] * dTdy[2:-2, 1 + jp:-3 + jp, kr:-1 + kr or None] \
                + drdS[2:-2, 2:-2, kr:-1 + kr or None] * dSdy[2:-2, 1 + jp:-3 + jp, kr:-1 + kr or None]
            syb = -drodyb / (np.minimum(0., drodzb) - epsln)
            taper = dm_taper(syb, iso_slopec, iso_dslope)
            sumy += facty[np.newaxis, :, np.newaxis] * K_iso[2:-2, 2:-2, :-1] \
                * taper * syb**2 * maskW[2:-2, 2:-2, :-1]
            Ai_by = update(Ai_by, at[2:-2, 2:-2, :-1, jp, kr], taper * syb * maskW[2:-2, 2:-2, :-1])

    K_33 = update(K_33, at[2:-2, 2:-2, :-1], sumx / (4 * dxt[2:-2, np.newaxis, np.newaxis]) + \
        sumy / (4 * dyt[np.newaxis, 2:-2, np.newaxis] * cost[np.newaxis, 2:-2, np.newaxis]))
    K_33 = update(K_33, at[2:-2, 2:-2, -1], 0.)

    return Ai_ez, Ai_nz, Ai_bx, Ai_by, K_11, K_22, K_33


@veros_kernel
def isoneutral_diag_streamfunction_kernel(B1_gm, B2_gm, K_gm, Ai_ez, Ai_nz):
    K_gm_pad = utilities.pad_z_edges(K_gm)

    """
    meridional component at east face of 'T' cells
    """
    diffloc = 0.25 * (K_gm_pad[1:-2, 2:-2, 1:-1] + K_gm_pad[1:-2, 2:-2, :-2]
                      + K_gm_pad[2:-1, 2:-2, 1:-1] + K_gm_pad[2:-1, 2:-2, :-2])
    B2_gm = update(B2_gm, at[1:-2, 2:-2, :], 0.25 * diffloc * np.sum(Ai_ez[1:-2, 2:-2, ...], axis=(3, 4)))

    """
    zonal component at north face of 'T' cells
    """
    diffloc = 0.25 * (K_gm_pad[2:-2, 1:-2, 1:-1] + K_gm_pad[2:-2, 1:-2, :-2]
                      + K_gm_pad[2:-2, 2:-1, 1:-1] + K_gm_pad[2:-2, 2:-1, :-2])
    B1_gm = update(B1_gm, at[2:-2, 1:-2, :], -0.25 * diffloc * np.sum(Ai_nz[2:-2, 1:-2, ...], axis=(3, 4)))
    return B1_gm, B2_gm


@veros_routine
def isoneutral_diag_streamfunction(vs):
    """
    calculate hor. components of streamfunction for eddy driven velocity
    for diagnostics purpose only
    """
    vs = isoneutral_diag_streamfunction_kernel.run_with_state(vs)

    return dict(
        B1_gm=B1_gm,
        B2_gm=B2_gm
    )


@veros_routine
def check_isoneutral_slope_crit(vs):
    """
    check linear stability criterion from Griffies et al
    """
    epsln = 1e-20
    if vs.enable_neutral_diffusion:
        ft1 = 1.0 / (4.0 * vs.K_iso_0 * vs.dt_tracer + epsln)
        delta1a = np.min(vs.dxt[2:-2, np.newaxis, np.newaxis] * np.abs(vs.cost[np.newaxis, 2:-2, np.newaxis])
                  * vs.dzt[np.newaxis, np.newaxis, :] * ft1)
        delta1b = np.min(vs.dyt[np.newaxis, 2:-2, np.newaxis] *
                         vs.dzt[np.newaxis, np.newaxis, :] * ft1)
        delta_iso1 = min(
            vs.dzt[0] * ft1 * vs.dxt[-1] * abs(vs.cost[-1]),
            min(delta1a, delta1b)
        )

        logger.info('Diffusion grid factor delta_iso1 = {}', float(delta_iso1))
        if delta_iso1 < vs.iso_slopec:
            raise RuntimeError('Without latitudinal filtering, delta_iso1 is the steepest '
                               'isoneutral slope available for linear stability of '
                               'Redi and GM. Maximum allowable isoneutral slope is '
                               'specified as iso_slopec = {}.'
                               .format(vs.iso_slopec))

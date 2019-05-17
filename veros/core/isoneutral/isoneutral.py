from loguru import logger

from .. import density, utilities
from ... import veros_method
from ...variables import allocate


@veros_method
def isoneutral_diffusion_pre(vs):
    """
    Isopycnal diffusion for tracer
    following functional formulation by Griffies et al
    Code adopted from MOM2.1
    """
    epsln = 1e-20

    dTdx = allocate(vs, ('xu', 'yt', 'zt'))
    dSdx = allocate(vs, ('xu', 'yt', 'zt'))
    dTdy = allocate(vs, ('xt', 'yu', 'zt'))
    dSdy = allocate(vs, ('xt', 'yu', 'zt'))
    dTdz = allocate(vs, ('xt', 'yt', 'zw'))
    dSdz = allocate(vs, ('xt', 'yt', 'zw'))

    """
    drho_dt and drho_ds at centers of T cells
    """
    drdT = vs.maskT * density.get_drhodT(
        vs, vs.salt[:, :, :, vs.tau], vs.temp[:, :, :, vs.tau], np.abs(vs.zt)
    )
    drdS = vs.maskT * density.get_drhodS(
        vs, vs.salt[:, :, :, vs.tau], vs.temp[:, :, :, vs.tau], np.abs(vs.zt)
    )

    """
    gradients at top face of T cells
    """
    dTdz[:, :, :-1] = vs.maskW[:, :, :-1] * \
        (vs.temp[:, :, 1:, vs.tau] - vs.temp[:, :, :-1, vs.tau]) / \
        vs.dzw[np.newaxis, np.newaxis, :-1]
    dSdz[:, :, :-1] = vs.maskW[:, :, :-1] * \
        (vs.salt[:, :, 1:, vs.tau] - vs.salt[:, :, :-1, vs.tau]) / \
        vs.dzw[np.newaxis, np.newaxis, :-1]

    """
    gradients at eastern face of T cells
    """
    dTdx[:-1, :, :] = vs.maskU[:-1, :, :] * (vs.temp[1:, :, :, vs.tau] - vs.temp[:-1, :, :, vs.tau]) \
        / (vs.dxu[:-1, np.newaxis, np.newaxis] * vs.cost[np.newaxis, :, np.newaxis])
    dSdx[:-1, :, :] = vs.maskU[:-1, :, :] * (vs.salt[1:, :, :, vs.tau] - vs.salt[:-1, :, :, vs.tau]) \
        / (vs.dxu[:-1, np.newaxis, np.newaxis] * vs.cost[np.newaxis, :, np.newaxis])

    """
    gradients at northern face of T cells
    """
    dTdy[:, :-1, :] = vs.maskV[:, :-1, :] * \
        (vs.temp[:, 1:, :, vs.tau] - vs.temp[:, :-1, :, vs.tau]) \
        / vs.dyu[np.newaxis, :-1, np.newaxis]
    dSdy[:, :-1, :] = vs.maskV[:, :-1, :] * \
        (vs.salt[:, 1:, :, vs.tau] - vs.salt[:, :-1, :, vs.tau]) \
        / vs.dyu[np.newaxis, :-1, np.newaxis]

    def dm_taper(sx):
        """
        tapering function for isopycnal slopes
        """
        return 0.5 * (1. + np.tanh((-np.abs(sx) + vs.iso_slopec) / vs.iso_dslope))

    """
    Compute Ai_ez and K11 on center of east face of T cell.
    """
    diffloc = allocate(vs, ('xt', 'yt', 'zt'))
    diffloc[1:-2, 2:-2, 1:] = 0.25 * (vs.K_iso[1:-2, 2:-2, 1:] + vs.K_iso[1:-2, 2:-2, :-1]
                                      + vs.K_iso[2:-1, 2:-2, 1:] + vs.K_iso[2:-1, 2:-2, :-1])
    diffloc[1:-2, 2:-2, 0] = 0.5 * (vs.K_iso[1:-2, 2:-2, 0] + vs.K_iso[2:-1, 2:-2, 0])

    sumz = allocate(vs, ('xu', 'yt', 'zw'))[1:-2, 2:-2]
    for kr in range(2):
        ki = 0 if kr == 1 else 1
        for ip in range(2):
            drodxe = drdT[1 + ip:-2 + ip, 2:-2, ki:] * dTdx[1:-2, 2:-2, ki:] \
                + drdS[1 + ip:-2 + ip, 2:-2, ki:] * dSdx[1:-2, 2:-2, ki:]
            drodze = drdT[1 + ip:-2 + ip, 2:-2, ki:] * dTdz[1 + ip:-2 + ip, 2:-2, :-1 + kr or None] \
                + drdS[1 + ip:-2 + ip, 2:-2, ki:] * dSdz[1 + ip:-2 + ip, 2:-2, :-1 + kr or None]
            sxe = -drodxe / (np.minimum(0., drodze) - epsln)
            taper = dm_taper(sxe)
            sumz[:, :, ki:] += vs.dzw[np.newaxis, np.newaxis, :-1 + kr or None] * vs.maskU[1:-2, 2:-2, ki:] \
                * np.maximum(vs.K_iso_steep, diffloc[1:-2, 2:-2, ki:] * taper)
            vs.Ai_ez[1:-2, 2:-2, ki:, ip, kr] = taper * sxe * vs.maskU[1:-2, 2:-2, ki:]
    vs.K_11[1:-2, 2:-2, :] = sumz / (4. * vs.dzt[np.newaxis, np.newaxis, :])

    """
    Compute Ai_nz and K_22 on center of north face of T cell.
    """
    diffloc[...] = 0
    diffloc[2:-2, 1:-2, 1:] = 0.25 * (vs.K_iso[2:-2, 1:-2, 1:] + vs.K_iso[2:-2, 1:-2, :-1]
                                      + vs.K_iso[2:-2, 2:-1, 1:] + vs.K_iso[2:-2, 2:-1, :-1])
    diffloc[2:-2, 1:-2, 0] = 0.5 * (vs.K_iso[2:-2, 1:-2, 0] + vs.K_iso[2:-2, 2:-1, 0])

    sumz = allocate(vs, ('xt', 'yu', 'zw'))[2:-2, 1:-2]
    for kr in range(2):
        ki = 0 if kr == 1 else 1
        for jp in range(2):
            drodyn = drdT[2:-2, 1 + jp:-2 + jp, ki:] * dTdy[2:-2, 1:-2, ki:] + \
                drdS[2:-2, 1 + jp:-2 + jp, ki:] * dSdy[2:-2, 1:-2, ki:]
            drodzn = drdT[2:-2, 1 + jp:-2 + jp, ki:] * dTdz[2:-2, 1 + jp:-2 + jp, :-1 + kr or None] \
                + drdS[2:-2, 1 + jp:-2 + jp, ki:] * dSdz[2:-2, 1 + jp:-2 + jp, :-1 + kr or None]
            syn = -drodyn / (np.minimum(0., drodzn) - epsln)
            taper = dm_taper(syn)
            sumz[:, :, ki:] += vs.dzw[np.newaxis, np.newaxis, :-1 + kr or None] \
                * vs.maskV[2:-2, 1:-2, ki:] * np.maximum(vs.K_iso_steep, diffloc[2:-2, 1:-2, ki:] * taper)
            vs.Ai_nz[2:-2, 1:-2, ki:, jp, kr] = taper * syn * vs.maskV[2:-2, 1:-2, ki:]
    vs.K_22[2:-2, 1:-2, :] = sumz / (4. * vs.dzt[np.newaxis, np.newaxis, :])

    """
    compute Ai_bx, Ai_by and K33 on top face of T cell.
    """
    sumx = allocate(vs, ('xt', 'yt', 'zt'))[2:-2, 2:-2, :-1]
    sumy = allocate(vs, ('xt', 'yt', 'zt'))[2:-2, 2:-2, :-1]

    for kr in range(2):
        drodzb = drdT[2:-2, 2:-2, kr:-1 + kr or None] * dTdz[2:-2, 2:-2, :-1] \
            + drdS[2:-2, 2:-2, kr:-1 + kr or None] * dSdz[2:-2, 2:-2, :-1]

        # eastward slopes at the top of T cells
        for ip in range(2):
            drodxb = drdT[2:-2, 2:-2, kr:-1 + kr or None] * dTdx[1 + ip:-3 + ip, 2:-2, kr:-1 + kr or None] \
                + drdS[2:-2, 2:-2, kr:-1 + kr or None] * dSdx[1 + ip:-3 + ip, 2:-2, kr:-1 + kr or None]
            sxb = -drodxb / (np.minimum(0., drodzb) - epsln)
            taper = dm_taper(sxb)
            sumx += vs.dxu[1 + ip:-3 + ip, np.newaxis, np.newaxis] * \
                vs.K_iso[2:-2, 2:-2, :-1] * taper * sxb**2 * vs.maskW[2:-2, 2:-2, :-1]
            vs.Ai_bx[2:-2, 2:-2, :-1, ip, kr] = taper * sxb * vs.maskW[2:-2, 2:-2, :-1]

        # northward slopes at the top of T cells
        for jp in range(2):
            facty = vs.cosu[1 + jp:-3 + jp] * vs.dyu[1 + jp:-3 + jp]
            drodyb = drdT[2:-2, 2:-2, kr:-1 + kr or None] * dTdy[2:-2, 1 + jp:-3 + jp, kr:-1 + kr or None] \
                + drdS[2:-2, 2:-2, kr:-1 + kr or None] * dSdy[2:-2, 1 + jp:-3 + jp, kr:-1 + kr or None]
            syb = -drodyb / (np.minimum(0., drodzb) - epsln)
            taper = dm_taper(syb)
            sumy += facty[np.newaxis, :, np.newaxis] * vs.K_iso[2:-2, 2:-2, :-1] \
                * taper * syb**2 * vs.maskW[2:-2, 2:-2, :-1]
            vs.Ai_by[2:-2, 2:-2, :-1, jp, kr] = taper * syb * vs.maskW[2:-2, 2:-2, :-1]

    vs.K_33[2:-2, 2:-2, :-1] = sumx / (4 * vs.dxt[2:-2, np.newaxis, np.newaxis]) + \
        sumy / (4 * vs.dyt[np.newaxis, 2:-2, np.newaxis] * vs.cost[np.newaxis, 2:-2, np.newaxis])
    vs.K_33[2:-2, 2:-2, -1] = 0.


@veros_method
def isoneutral_diag_streamfunction(vs):
    """
    calculate hor. components of streamfunction for eddy driven velocity
    for diagnostics purpose only
    """

    """
    meridional component at east face of 'T' cells
    """
    K_gm_pad = utilities.pad_z_edges(vs, vs.K_gm)

    diffloc = 0.25 * (K_gm_pad[1:-2, 2:-2, 1:-1] + K_gm_pad[1:-2, 2:-2, :-2] +
                      K_gm_pad[2:-1, 2:-2, 1:-1] + K_gm_pad[2:-1, 2:-2, :-2])
    sumz = np.sum(diffloc[..., np.newaxis, np.newaxis] * vs.Ai_ez[1:-2, 2:-2, ...], axis=(3, 4))
    vs.B2_gm[1:-2, 2:-2, :] = 0.25 * sumz

    """
    zonal component at north face of 'T' cells
    """
    diffloc = 0.25 * (K_gm_pad[2:-2, 1:-2, 1:-1] + K_gm_pad[2:-2, 1:-2, :-2] +
                      K_gm_pad[2:-2, 2:-1, 1:-1] + K_gm_pad[2:-2, 2:-1, :-2])
    sumz = np.sum(diffloc[..., np.newaxis, np.newaxis] * vs.Ai_nz[2:-2, 1:-2, ...], axis=(3, 4))
    vs.B1_gm[2:-2, 1:-2, :] = -0.25 * sumz


@veros_method(dist_safe=False, local_variables=[
    'dxt', 'dyt', 'dzt', 'cost'
])
def check_isoneutral_slope_crit(vs):
    """
    check linear stability criterion from Griffies et al
    """
    epsln = 1e-20
    if vs.enable_neutral_diffusion:
        ft1 = 1.0 / (4.0 * vs.K_iso_0 * vs.dt_tracer + epsln)
        delta1a = np.min(vs.dxt[2:-2, np.newaxis, np.newaxis] * np.abs(vs.cost[np.newaxis, 2:-2, np.newaxis]) \
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

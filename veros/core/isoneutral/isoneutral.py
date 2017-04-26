import logging

from .. import density, utilities
from ... import veros_method, veros_inline_method


@veros_method
def isoneutral_diffusion_pre(veros):
    """
    Isopycnal diffusion for tracer
    following functional formulation by Griffies et al
    Code adopted from MOM2.1
    """
    drdTS = np.zeros((veros.nx + 4, veros.ny + 4, veros.nz, 2))
    ddzt = np.zeros((veros.nx + 4, veros.ny + 4, veros.nz, 2))
    ddxt = np.zeros((veros.nx + 4, veros.ny + 4, veros.nz, 2))
    ddyt = np.zeros((veros.nx + 4, veros.ny + 4, veros.nz, 2))
    epsln = 1.e-20  # for double precision

    """
    drho_dt and drho_ds at centers of T cells
    """
    drdTS[:, :, :, 0] = density.get_drhodT(
        veros, veros.salt[:, :, :, veros.tau], veros.temp[:, :, :, veros.tau], np.abs(veros.zt)) * veros.maskT
    drdTS[:, :, :, 1] = density.get_drhodS(
        veros, veros.salt[:, :, :, veros.tau], veros.temp[:, :, :, veros.tau], np.abs(veros.zt)) * veros.maskT

    """
    gradients at top face of T cells
    """
    ddzt[:, :, :-1, 0] = veros.maskW[:, :, :-1] * \
        (veros.temp[:, :, 1:, veros.tau] - veros.temp[:, :, :-1, veros.tau]) / \
        veros.dzw[np.newaxis, np.newaxis, :-1]
    ddzt[:, :, :-1, 1] = veros.maskW[:, :, :-1] * \
        (veros.salt[:, :, 1:, veros.tau] - veros.salt[:, :, :-1, veros.tau]) / \
        veros.dzw[np.newaxis, np.newaxis, :-1]
    ddzt[..., -1, :] = 0.

    """
    gradients at eastern face of T cells
    """
    ddxt[:-1, :, :, 0] = veros.maskU[:-1, :, :] * (veros.temp[1:, :, :, veros.tau] - veros.temp[:-1, :, :, veros.tau]) \
        / (veros.dxu[:-1, np.newaxis, np.newaxis] * veros.cost[np.newaxis, :, np.newaxis])
    ddxt[:-1, :, :, 1] = veros.maskU[:-1, :, :] * (veros.salt[1:, :, :, veros.tau] - veros.salt[:-1, :, :, veros.tau]) \
        / (veros.dxu[:-1, np.newaxis, np.newaxis] * veros.cost[np.newaxis, :, np.newaxis])

    """
    gradients at northern face of T cells
    """
    ddyt[:, :-1, :, 0] = veros.maskV[:, :-1, :] * \
        (veros.temp[:, 1:, :, veros.tau] - veros.temp[:, :-1, :, veros.tau]) \
        / veros.dyu[np.newaxis, :-1, np.newaxis]
    ddyt[:, :-1, :, 1] = veros.maskV[:, :-1, :] * \
        (veros.salt[:, 1:, :, veros.tau] - veros.salt[:, :-1, :, veros.tau]) \
        / veros.dyu[np.newaxis, :-1, np.newaxis]

    """
    Compute Ai_ez and K11 on center of east face of T cell.
    """
    diffloc = np.zeros((veros.nx + 4, veros.ny + 4, veros.nz))
    diffloc[1:-2, 2:-2, 1:] = 0.25 * (veros.K_iso[1:-2, 2:-2, 1:] + veros.K_iso[1:-2, 2:-2, :-1]
                                      + veros.K_iso[2:-1, 2:-2, 1:] + veros.K_iso[2:-1, 2:-2, :-1])
    diffloc[1:-2, 2:-2, 0] = 0.5 * (veros.K_iso[1:-2, 2:-2, 0] + veros.K_iso[2:-1, 2:-2, 0])

    sumz = np.zeros((veros.nx + 1, veros.ny, veros.nz))
    for kr in xrange(2):
        ki = 0 if kr == 1 else 1
        for ip in xrange(2):
            drodxe = drdTS[1 + ip:-2 + ip, 2:-2, ki:, 0] * ddxt[1:-2, 2:-2, ki:, 0] \
                + drdTS[1 + ip:-2 + ip, 2:-2, ki:, 1] * ddxt[1:-2, 2:-2, ki:, 1]
            drodze = drdTS[1 + ip:-2 + ip, 2:-2, ki:, 0] * ddzt[1 + ip:-2 + ip, 2:-2, :-1 + kr or None, 0] \
                + drdTS[1 + ip:-2 + ip, 2:-2, ki:, 1] * \
                ddzt[1 + ip:-2 + ip, 2:-2, :-1 + kr or None, 1]
            sxe = -drodxe / (np.minimum(0., drodze) - epsln)
            taper = dm_taper(veros, sxe, veros.iso_slopec, veros.iso_dslope)
            sumz[:, :, ki:] += veros.dzw[np.newaxis, np.newaxis, :-1 + kr or None] * veros.maskU[1:-2, 2:-2, ki:] \
                * np.maximum(veros.K_iso_steep, diffloc[1:-2, 2:-2, ki:] * taper)
            veros.Ai_ez[1:-2, 2:-2, ki:, ip, kr] = taper * sxe * veros.maskU[1:-2, 2:-2, ki:]
    veros.K_11[1:-2, 2:-2, :] = sumz / (4. * veros.dzt[np.newaxis, np.newaxis, :])

    """
    Compute Ai_nz and K_22 on center of north face of T cell.
    """
    diffloc = np.zeros((veros.nx + 4, veros.ny + 4, veros.nz))
    diffloc[2:-2, 1:-2, 1:] = 0.25 * (veros.K_iso[2:-2, 1:-2, 1:] + veros.K_iso[2:-2, 1:-2, :-1]
                                      + veros.K_iso[2:-2, 2:-1, 1:] + veros.K_iso[2:-2, 2:-1, :-1])
    diffloc[2:-2, 1:-2, 0] = 0.5 * (veros.K_iso[2:-2, 1:-2, 0] + veros.K_iso[2:-2, 2:-1, 0])

    sumz = np.zeros((veros.nx, veros.ny + 1, veros.nz))
    for kr in xrange(2):
        ki = 0 if kr == 1 else 1
        for jp in xrange(2):
            drodyn = drdTS[2:-2, 1 + jp:-2 + jp, ki:, 0] * ddyt[2:-2, 1:-2, ki:, 0] + \
                drdTS[2:-2, 1 + jp:-2 + jp, ki:, 1] * ddyt[2:-2, 1:-2, ki:, 1]
            drodzn = drdTS[2:-2, 1 + jp:-2 + jp, ki:, 0] * ddzt[2:-2, 1 + jp:-2 + jp, :-1 + kr or None, 0] \
                + drdTS[2:-2, 1 + jp:-2 + jp, ki:, 1] * \
                ddzt[2:-2, 1 + jp:-2 + jp, :-1 + kr or None, 1]
            syn = -drodyn / (np.minimum(0., drodzn) - epsln)
            taper = dm_taper(veros, syn, veros.iso_slopec, veros.iso_dslope)
            sumz[:, :, ki:] += veros.dzw[np.newaxis, np.newaxis, :-1 + kr or None] \
                * veros.maskV[2:-2, 1:-2, ki:] * np.maximum(veros.K_iso_steep, diffloc[2:-2, 1:-2, ki:] * taper)
            veros.Ai_nz[2:-2, 1:-2, ki:, jp, kr] = taper * syn * veros.maskV[2:-2, 1:-2, ki:]
    veros.K_22[2:-2, 1:-2, :] = sumz / (4. * veros.dzt[np.newaxis, np.newaxis, :])

    """
    compute Ai_bx, Ai_by and K33 on top face of T cell.
    """
    # eastward slopes at the top of T cells
    sumx = np.zeros((veros.nx, veros.ny, veros.nz - 1))
    for ip in xrange(2):
        for kr in xrange(2):
            drodxb = drdTS[2:-2, 2:-2, kr:-1 + kr or None, 0] * ddxt[1 + ip:-3 + ip, 2:-2, kr:-1 + kr or None, 0] \
                + drdTS[2:-2, 2:-2, kr:-1 + kr or None, 1] * \
                ddxt[1 + ip:-3 + ip, 2:-2, kr:-1 + kr or None, 1]
            drodzb = drdTS[2:-2, 2:-2, kr:-1 + kr or None, 0] * ddzt[2:-2, 2:-2, :-1, 0] \
                + drdTS[2:-2, 2:-2, kr:-1 + kr or None, 1] * ddzt[2:-2, 2:-2, :-1, 1]
            sxb = -drodxb / (np.minimum(0., drodzb) - epsln)
            taper = dm_taper(veros, sxb, veros.iso_slopec, veros.iso_dslope)
            sumx += veros.dxu[1 + ip:-3 + ip, np.newaxis, np.newaxis] * \
                veros.K_iso[2:-2, 2:-2, :-1] * taper * sxb**2 * veros.maskW[2:-2, 2:-2, :-1]
            veros.Ai_bx[2:-2, 2:-2, :-1, ip, kr] = taper * sxb * veros.maskW[2:-2, 2:-2, :-1]

    # northward slopes at the top of T cells
    sumy = np.zeros((veros.nx, veros.ny, veros.nz - 1))
    for jp in xrange(2):  # jp=0,1
        facty = veros.cosu[1 + jp:-3 + jp] * veros.dyu[1 + jp:-3 + jp]
        for kr in xrange(2):  # kr=0,1
            drodyb = drdTS[2:-2, 2:-2, kr:-1 + kr or None, 0] * ddyt[2:-2, 1 + jp:-3 + jp, kr:-1 + kr or None, 0] \
                + drdTS[2:-2, 2:-2, kr:-1 + kr or None, 1] * \
                ddyt[2:-2, 1 + jp:-3 + jp, kr:-1 + kr or None, 1]
            drodzb = drdTS[2:-2, 2:-2, kr:-1 + kr or None, 0] * ddzt[2:-2, 2:-2, :-1, 0] \
                + drdTS[2:-2, 2:-2, kr:-1 + kr or None, 1] * ddzt[2:-2, 2:-2, :-1, 1]
            syb = -drodyb / (np.minimum(0., drodzb) - epsln)
            taper = dm_taper(veros, syb, veros.iso_slopec, veros.iso_dslope)
            sumy += facty[np.newaxis, :, np.newaxis] * veros.K_iso[2:-2, 2:-2, :-1] \
                * taper * syb**2 * veros.maskW[2:-2, 2:-2, :-1]
            veros.Ai_by[2:-2, 2:-2, :-1, jp, kr] = taper * syb * veros.maskW[2:-2, 2:-2, :-1]

    veros.K_33[2:-2, 2:-2, :-1] = sumx / (4 * veros.dxt[2:-2, np.newaxis, np.newaxis]) + sumy \
        / (4 * veros.dyt[np.newaxis, 2:-2, np.newaxis] * veros.cost[np.newaxis, 2:-2, np.newaxis])
    veros.K_33[2:-2, 2:-2, -1] = 0.


@veros_method
def isoneutral_diag_streamfunction(veros):
    """
    calculate hor. components of streamfunction for eddy driven velocity
    for diagnostics purpose only
    """

    """
    meridional component at east face of "T" cells
    """
    K_gm_pad = utilities.pad_z_edges(veros, veros.K_gm)

    diffloc = 0.25 * (K_gm_pad[1:-2, 2:-2, 1:-1] + K_gm_pad[1:-2, 2:-2, :-2] +
                      K_gm_pad[2:-1, 2:-2, 1:-1] + K_gm_pad[2:-1, 2:-2, :-2])
    sumz = np.sum(diffloc[..., np.newaxis, np.newaxis] * veros.Ai_ez[1:-2, 2:-2, ...], axis=(3, 4))
    veros.B2_gm[1:-2, 2:-2, :] = 0.25 * sumz

    """
    zonal component at north face of "T" cells
    """
    diffloc = 0.25 * (K_gm_pad[2:-2, 1:-2, 1:-1] + K_gm_pad[2:-2, 1:-2, :-2] +
                      K_gm_pad[2:-2, 2:-1, 1:-1] + K_gm_pad[2:-2, 2:-1, :-2])
    sumz = np.sum(diffloc[..., np.newaxis, np.newaxis] * veros.Ai_nz[2:-2, 1:-2, ...], axis=(3, 4))
    veros.B1_gm[2:-2, 1:-2, :] = -0.25 * sumz


@veros_inline_method
def dm_taper(veros, sx, iso_slopec, iso_dslope):
    """
    tapering function for isopycnal slopes
    """
    return 0.5 * (1. + np.tanh((-np.abs(sx) + iso_slopec) / iso_dslope))


@veros_method
def check_isoneutral_slope_crit(veros):
    """
    check linear stability criterion from Griffies et al
    """
    epsln = 1e-20
    if veros.enable_neutral_diffusion:
        ft1 = 1.0 / (4.0 * veros.K_iso_0 * veros.dt_tracer + epsln)
        delta1a = np.min(veros.dxt[2:-2, np.newaxis, np.newaxis] * np.abs(
            veros.cost[np.newaxis, 2:-2, np.newaxis]) * veros.dzt[np.newaxis, np.newaxis, :] * ft1)
        delta1b = np.min(veros.dyt[np.newaxis, 2:-2, np.newaxis] *
                         veros.dzt[np.newaxis, np.newaxis, :] * ft1)
        delta_iso1 = np.minimum(veros.dzt[0] * ft1 * veros.dxt[-1]
                                * np.abs(veros.cost[-1]), np.minimum(delta1a, delta1b))

        logging.info("diffusion grid factor delta_iso1 = {}".format(delta_iso1))
        if delta_iso1 < veros.iso_slopec:
            raise RuntimeError("""
                   Without latitudinal filtering, delta_iso1 is the steepest
                   isoneutral slope available for linear stab of Redi and GM.
                   Maximum allowable isoneutral slope is specified as {}
                   integration will be unstable
                   """.format(veros.iso_slopec))

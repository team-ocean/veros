import logging

from .. import density, utilities
from ... import veros_method, veros_inline_method


@veros_method
def isoneutral_diffusion_pre(vs):
    """
    Isopycnal diffusion for tracer
    following functional formulation by Griffies et al
    Code adopted from MOM2.1
    """
    drdTS = np.zeros((vs.nx + 4, vs.ny + 4, vs.nz, 2), dtype=vs.default_float_type)
    ddzt = np.zeros((vs.nx + 4, vs.ny + 4, vs.nz, 2), dtype=vs.default_float_type)
    ddxt = np.zeros((vs.nx + 4, vs.ny + 4, vs.nz, 2), dtype=vs.default_float_type)
    ddyt = np.zeros((vs.nx + 4, vs.ny + 4, vs.nz, 2), dtype=vs.default_float_type)
    epsln = 1e-20

    """
    drho_dt and drho_ds at centers of T cells
    """
    drdTS[:, :, :, 0] = vs.maskT * density.get_drhodT(
        vs, vs.salt[:, :, :, vs.tau], vs.temp[:, :, :, vs.tau], np.abs(vs.zt)
    )
    drdTS[:, :, :, 1] = vs.maskT * density.get_drhodS(
        vs, vs.salt[:, :, :, vs.tau], vs.temp[:, :, :, vs.tau], np.abs(vs.zt)
    )

    """
    gradients at top face of T cells
    """
    ddzt[:, :, :-1, 0] = vs.maskW[:, :, :-1] * \
        (vs.temp[:, :, 1:, vs.tau] - vs.temp[:, :, :-1, vs.tau]) / \
        vs.dzw[np.newaxis, np.newaxis, :-1]
    ddzt[:, :, :-1, 1] = vs.maskW[:, :, :-1] * \
        (vs.salt[:, :, 1:, vs.tau] - vs.salt[:, :, :-1, vs.tau]) / \
        vs.dzw[np.newaxis, np.newaxis, :-1]
    ddzt[..., -1, :] = 0.

    """
    gradients at eastern face of T cells
    """
    ddxt[:-1, :, :, 0] = vs.maskU[:-1, :, :] * (vs.temp[1:, :, :, vs.tau] - vs.temp[:-1, :, :, vs.tau]) \
        / (vs.dxu[:-1, np.newaxis, np.newaxis] * vs.cost[np.newaxis, :, np.newaxis])
    ddxt[:-1, :, :, 1] = vs.maskU[:-1, :, :] * (vs.salt[1:, :, :, vs.tau] - vs.salt[:-1, :, :, vs.tau]) \
        / (vs.dxu[:-1, np.newaxis, np.newaxis] * vs.cost[np.newaxis, :, np.newaxis])

    """
    gradients at northern face of T cells
    """
    ddyt[:, :-1, :, 0] = vs.maskV[:, :-1, :] * \
        (vs.temp[:, 1:, :, vs.tau] - vs.temp[:, :-1, :, vs.tau]) \
        / vs.dyu[np.newaxis, :-1, np.newaxis]
    ddyt[:, :-1, :, 1] = vs.maskV[:, :-1, :] * \
        (vs.salt[:, 1:, :, vs.tau] - vs.salt[:, :-1, :, vs.tau]) \
        / vs.dyu[np.newaxis, :-1, np.newaxis]

    """
    Compute Ai_ez and K11 on center of east face of T cell.
    """
    diffloc = np.zeros((vs.nx + 4, vs.ny + 4, vs.nz), dtype=vs.default_float_type)
    diffloc[1:-2, 2:-2, 1:] = 0.25 * (vs.K_iso[1:-2, 2:-2, 1:] + vs.K_iso[1:-2, 2:-2, :-1]
                                      + vs.K_iso[2:-1, 2:-2, 1:] + vs.K_iso[2:-1, 2:-2, :-1])
    diffloc[1:-2, 2:-2, 0] = 0.5 * (vs.K_iso[1:-2, 2:-2, 0] + vs.K_iso[2:-1, 2:-2, 0])

    sumz = np.zeros((vs.nx + 1, vs.ny, vs.nz), dtype=vs.default_float_type)
    for kr in range(2):
        ki = 0 if kr == 1 else 1
        for ip in range(2):
            drodxe = drdTS[1 + ip:-2 + ip, 2:-2, ki:, 0] * ddxt[1:-2, 2:-2, ki:, 0] \
                + drdTS[1 + ip:-2 + ip, 2:-2, ki:, 1] * ddxt[1:-2, 2:-2, ki:, 1]
            drodze = drdTS[1 + ip:-2 + ip, 2:-2, ki:, 0] * ddzt[1 + ip:-2 + ip, 2:-2, :-1 + kr or None, 0] \
                + drdTS[1 + ip:-2 + ip, 2:-2, ki:, 1] * \
                ddzt[1 + ip:-2 + ip, 2:-2, :-1 + kr or None, 1]
            sxe = -drodxe / (np.minimum(0., drodze) - epsln)
            taper = dm_taper(vs, sxe, vs.iso_slopec, vs.iso_dslope)
            sumz[:, :, ki:] += vs.dzw[np.newaxis, np.newaxis, :-1 + kr or None] * vs.maskU[1:-2, 2:-2, ki:] \
                * np.maximum(vs.K_iso_steep, diffloc[1:-2, 2:-2, ki:] * taper)
            vs.Ai_ez[1:-2, 2:-2, ki:, ip, kr] = taper * sxe * vs.maskU[1:-2, 2:-2, ki:]
    vs.K_11[1:-2, 2:-2, :] = sumz / (4. * vs.dzt[np.newaxis, np.newaxis, :])

    """
    Compute Ai_nz and K_22 on center of north face of T cell.
    """
    diffloc = np.zeros((vs.nx + 4, vs.ny + 4, vs.nz), dtype=vs.default_float_type)
    diffloc[2:-2, 1:-2, 1:] = 0.25 * (vs.K_iso[2:-2, 1:-2, 1:] + vs.K_iso[2:-2, 1:-2, :-1]
                                      + vs.K_iso[2:-2, 2:-1, 1:] + vs.K_iso[2:-2, 2:-1, :-1])
    diffloc[2:-2, 1:-2, 0] = 0.5 * (vs.K_iso[2:-2, 1:-2, 0] + vs.K_iso[2:-2, 2:-1, 0])

    sumz = np.zeros((vs.nx, vs.ny + 1, vs.nz), dtype=vs.default_float_type)
    for kr in range(2):
        ki = 0 if kr == 1 else 1
        for jp in range(2):
            drodyn = drdTS[2:-2, 1 + jp:-2 + jp, ki:, 0] * ddyt[2:-2, 1:-2, ki:, 0] + \
                drdTS[2:-2, 1 + jp:-2 + jp, ki:, 1] * ddyt[2:-2, 1:-2, ki:, 1]
            drodzn = drdTS[2:-2, 1 + jp:-2 + jp, ki:, 0] * ddzt[2:-2, 1 + jp:-2 + jp, :-1 + kr or None, 0] \
                + drdTS[2:-2, 1 + jp:-2 + jp, ki:, 1] * \
                ddzt[2:-2, 1 + jp:-2 + jp, :-1 + kr or None, 1]
            syn = -drodyn / (np.minimum(0., drodzn) - epsln)
            taper = dm_taper(vs, syn, vs.iso_slopec, vs.iso_dslope)
            sumz[:, :, ki:] += vs.dzw[np.newaxis, np.newaxis, :-1 + kr or None] \
                * vs.maskV[2:-2, 1:-2, ki:] * np.maximum(vs.K_iso_steep, diffloc[2:-2, 1:-2, ki:] * taper)
            vs.Ai_nz[2:-2, 1:-2, ki:, jp, kr] = taper * syn * vs.maskV[2:-2, 1:-2, ki:]
    vs.K_22[2:-2, 1:-2, :] = sumz / (4. * vs.dzt[np.newaxis, np.newaxis, :])

    """
    compute Ai_bx, Ai_by and K33 on top face of T cell.
    """
    # eastward slopes at the top of T cells
    sumx = np.zeros((vs.nx, vs.ny, vs.nz - 1), dtype=vs.default_float_type)
    for ip in range(2):
        for kr in range(2):
            drodxb = drdTS[2:-2, 2:-2, kr:-1 + kr or None, 0] * ddxt[1 + ip:-3 + ip, 2:-2, kr:-1 + kr or None, 0] \
                + drdTS[2:-2, 2:-2, kr:-1 + kr or None, 1] * \
                ddxt[1 + ip:-3 + ip, 2:-2, kr:-1 + kr or None, 1]
            drodzb = drdTS[2:-2, 2:-2, kr:-1 + kr or None, 0] * ddzt[2:-2, 2:-2, :-1, 0] \
                + drdTS[2:-2, 2:-2, kr:-1 + kr or None, 1] * ddzt[2:-2, 2:-2, :-1, 1]
            sxb = -drodxb / (np.minimum(0., drodzb) - epsln)
            taper = dm_taper(vs, sxb, vs.iso_slopec, vs.iso_dslope)
            sumx += vs.dxu[1 + ip:-3 + ip, np.newaxis, np.newaxis] * \
                vs.K_iso[2:-2, 2:-2, :-1] * taper * sxb**2 * vs.maskW[2:-2, 2:-2, :-1]
            vs.Ai_bx[2:-2, 2:-2, :-1, ip, kr] = taper * sxb * vs.maskW[2:-2, 2:-2, :-1]

    # northward slopes at the top of T cells
    sumy = np.zeros((vs.nx, vs.ny, vs.nz - 1), dtype=vs.default_float_type)
    for jp in range(2):
        facty = vs.cosu[1 + jp:-3 + jp] * vs.dyu[1 + jp:-3 + jp]
        for kr in range(2):
            drodyb = drdTS[2:-2, 2:-2, kr:-1 + kr or None, 0] * ddyt[2:-2, 1 + jp:-3 + jp, kr:-1 + kr or None, 0] \
                + drdTS[2:-2, 2:-2, kr:-1 + kr or None, 1] * \
                ddyt[2:-2, 1 + jp:-3 + jp, kr:-1 + kr or None, 1]
            drodzb = drdTS[2:-2, 2:-2, kr:-1 + kr or None, 0] * ddzt[2:-2, 2:-2, :-1, 0] \
                + drdTS[2:-2, 2:-2, kr:-1 + kr or None, 1] * ddzt[2:-2, 2:-2, :-1, 1]
            syb = -drodyb / (np.minimum(0., drodzb) - epsln)
            taper = dm_taper(vs, syb, vs.iso_slopec, vs.iso_dslope)
            sumy += facty[np.newaxis, :, np.newaxis] * vs.K_iso[2:-2, 2:-2, :-1] \
                * taper * syb**2 * vs.maskW[2:-2, 2:-2, :-1]
            vs.Ai_by[2:-2, 2:-2, :-1, jp, kr] = taper * syb * vs.maskW[2:-2, 2:-2, :-1]

    vs.K_33[2:-2, 2:-2, :-1] = sumx / (4 * vs.dxt[2:-2, np.newaxis, np.newaxis]) + sumy \
        / (4 * vs.dyt[np.newaxis, 2:-2, np.newaxis] * vs.cost[np.newaxis, 2:-2, np.newaxis])
    vs.K_33[2:-2, 2:-2, -1] = 0.


@veros_method
def isoneutral_diag_streamfunction(vs):
    """
    calculate hor. components of streamfunction for eddy driven velocity
    for diagnostics purpose only
    """

    """
    meridional component at east face of "T" cells
    """
    K_gm_pad = utilities.pad_z_edges(vs, vs.K_gm)

    diffloc = 0.25 * (K_gm_pad[1:-2, 2:-2, 1:-1] + K_gm_pad[1:-2, 2:-2, :-2] +
                      K_gm_pad[2:-1, 2:-2, 1:-1] + K_gm_pad[2:-1, 2:-2, :-2])
    sumz = np.sum(diffloc[..., np.newaxis, np.newaxis] * vs.Ai_ez[1:-2, 2:-2, ...], axis=(3, 4))
    vs.B2_gm[1:-2, 2:-2, :] = 0.25 * sumz

    """
    zonal component at north face of "T" cells
    """
    diffloc = 0.25 * (K_gm_pad[2:-2, 1:-2, 1:-1] + K_gm_pad[2:-2, 1:-2, :-2] +
                      K_gm_pad[2:-2, 2:-1, 1:-1] + K_gm_pad[2:-2, 2:-1, :-2])
    sumz = np.sum(diffloc[..., np.newaxis, np.newaxis] * vs.Ai_nz[2:-2, 1:-2, ...], axis=(3, 4))
    vs.B1_gm[2:-2, 1:-2, :] = -0.25 * sumz


@veros_inline_method
def dm_taper(vs, sx, iso_slopec, iso_dslope):
    """
    tapering function for isopycnal slopes
    """
    return 0.5 * (1. + np.tanh((-np.abs(sx) + iso_slopec) / iso_dslope))


@veros_method
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
        delta_iso1 = np.minimum(
                        vs.dzt[0] * ft1 * vs.dxt[-1] * np.abs(vs.cost[-1]),
                        np.minimum(delta1a, delta1b)
                    )

        logging.info("diffusion grid factor delta_iso1 = {}".format(delta_iso1))
        if delta_iso1 < vs.iso_slopec:
            raise RuntimeError("Without latitudinal filtering, delta_iso1 is the steepest "
                               "isoneutral slope available for linear stability of "
                               "Redi and GM. Maximum allowable isoneutral slope is "
                               "specified as iso_slopec = {}."
                               .format(vs.iso_slopec))

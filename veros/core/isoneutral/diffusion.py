from veros.core.operators import numpy as np

from veros import veros_kernel
from veros.core import utilities, diffusion
from veros.core.operators import update, update_add, at


@veros_kernel
def _calc_tracer_fluxes(tr, K_iso, K_skew, K_11, K_22, Ai_ez, Ai_nz, Ai_bx, Ai_by,
                        flux_east, flux_north, flux_top, dxt, dxu, dyt, dyu, dzt,
                        cost, cosu, tau):
    tr_pad = utilities.pad_z_edges(tr[..., tau])

    K1 = K_iso - K_skew
    K2 = K_iso + K_skew

    """
    construct total isoneutral tracer flux at east face of 'T' cells
    """
    diffloc = np.zeros_like(K1)[1:-2, 2:-2]
    diffloc = update(diffloc, at[:, :, 1:], 0.25 * (K1[1:-2, 2:-2, 1:] + K1[1:-2, 2:-2, :-1] +
                                K1[2:-1, 2:-2, 1:] + K1[2:-1, 2:-2, :-1]))
    diffloc = update(diffloc, at[:, :, 0], 0.5 * (K1[1:-2, 2:-2, 0] + K1[2:-1, 2:-2, 0]))
    sumz = np.zeros_like(K1)[1:-2, 2:-2]
    for kr in range(2):
        for ip in range(2):
            sumz += diffloc * Ai_ez[1:-2, 2:-2, :, ip, kr] * (
                tr_pad[1 + ip:-2 + ip, 2:-2, 1 + kr:-1 + kr or None] - tr_pad[1 + ip:-2 + ip, 2:-2, kr:-2 + kr])

    flux_east = update(flux_east, at[1:-2, 2:-2, :], sumz / (4. * dzt[np.newaxis, np.newaxis, :]) + (tr[2:-1, 2:-2, :, tau] - tr[1:-2, 2:-2, :, tau]) \
                             / (cost[np.newaxis, 2:-2, np.newaxis] * dxu[1:-2, np.newaxis, np.newaxis]) * K_11[1:-2, 2:-2, :])

    """
    construct total isoneutral tracer flux at north face of 'T' cells
    """
    diffloc = np.zeros_like(K1)[2:-2, 1:-2]
    diffloc = update(diffloc, at[:, :, 1:], 0.25 * (K1[2:-2, 1:-2, 1:] + K1[2:-2, 1:-2, :-1] +
                                K1[2:-2, 2:-1, 1:] + K1[2:-2, 2:-1, :-1]))
    diffloc = update(diffloc, at[:, :, 0], 0.5 * (K1[2:-2, 1:-2, 0] + K1[2:-2, 2:-1, 0]))
    sumz = np.zeros_like(K1)[2:-2, 1:-2]
    for kr in range(2):
        for jp in range(2):
            sumz += diffloc * Ai_nz[2:-2, 1:-2, :, jp, kr] * (
                tr_pad[2:-2, 1 + jp:-2 + jp, 1 + kr:-1 + kr or None] - tr_pad[2:-2, 1 + jp:-2 + jp, kr:-2 + kr])
    flux_north = update(flux_north, at[2:-2, 1:-2, :], cosu[np.newaxis, 1:-2, np.newaxis] * (sumz / (4. * dzt[np.newaxis, np.newaxis, :])
                                + (tr[2:-2, 2:-1, :, tau] - tr[2:-2, 1:-2, :, tau]) / dyu[np.newaxis, 1:-2, np.newaxis] * K_22[2:-2, 1:-2, :]))

    """
    compute the vertical tracer flux 'flux_top' containing the K31
    and K32 components which are to be solved explicitly. The K33
    component will be treated implicitly. Note that there are some
    cancellations of dxu(i-1+ip) and dyu(jrow-1+jp)
    """
    diffloc = K2[2:-2, 2:-2, :-1]
    sumx = 0.
    for ip in range(2):
        for kr in range(2):
            sumx += diffloc * Ai_bx[2:-2, 2:-2, :-1, ip, kr] / cost[np.newaxis, 2:-2, np.newaxis] \
                * (tr[2 + ip:-2 + ip, 2:-2, kr:-1 + kr or None, tau] - tr[1 + ip:-3 + ip, 2:-2, kr:-1 + kr or None, tau])
    sumy = 0.
    for jp in range(2):
        for kr in range(2):
            sumy += diffloc * Ai_by[2:-2, 2:-2, :-1, jp, kr] * cosu[np.newaxis, 1 + jp:-3 + jp, np.newaxis] \
                * (tr[2:-2, 2 + jp:-2 + jp, kr:-1 + kr or None, tau] - tr[2:-2, 1 + jp:-3 + jp, kr:-1 + kr or None, tau])
    flux_top = update(flux_top, at[2:-2, 2:-2, :-1], sumx / (4 * dxt[2:-2, np.newaxis, np.newaxis]) \
                              + sumy / (4 * dyt[np.newaxis, 2:-2, np.newaxis] * cost[np.newaxis, 2:-2, np.newaxis]))
    flux_top = update(flux_top, at[:, :, -1], 0.)

    return flux_east, flux_north, flux_top


@veros_kernel
def _calc_explicit_part(flux_east, flux_north, flux_top, cost, dxt, dyt, dzt, maskT):
    aloc = np.zeros_like(maskT)
    aloc = update(aloc, at[2:-2, 2:-2, :], maskT[2:-2, 2:-2, :] * ((flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :]) / (cost[np.newaxis, 2:-2, np.newaxis] * dxt[2:-2, np.newaxis, np.newaxis])
                                                  + (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :]) / (cost[np.newaxis, 2:-2, np.newaxis] * dyt[np.newaxis, 2:-2, np.newaxis])))
    aloc = update_add(aloc, at[:, :, 0], maskT[:, :, 0] * flux_top[:, :, 0] / dzt[0])
    aloc = update_add(aloc, at[:, :, 1:], maskT[:, :, 1:] * \
        (flux_top[:, :, 1:] - flux_top[:, :, :-1]) / \
        dzt[np.newaxis, np.newaxis, 1:])
    return aloc


@veros_kernel
def _calc_implicit_part(tr, kbot, K_33, dzt, dzw, dt_tracer, taup1):
    nz = dzw.shape[0]
    _, water_mask, edge_mask = utilities.create_water_masks(kbot[2:-2, 2:-2], nz)

    a_tri = np.zeros_like(K_33)[2:-2, 2:-2]
    b_tri = np.zeros_like(K_33)[2:-2, 2:-2]
    c_tri = np.zeros_like(K_33)[2:-2, 2:-2]
    delta = np.zeros_like(K_33)[2:-2, 2:-2]

    delta = update(delta, at[:, :, :-1], dt_tracer / dzw[np.newaxis, np.newaxis, :-1] * K_33[2:-2, 2:-2, :-1])
    delta = update(delta, at[:, :, -1], 0.)
    a_tri = update(a_tri, at[:, :, 1:], -delta[:, :, :-1] / dzt[np.newaxis, np.newaxis, 1:])
    b_tri = update(b_tri, at[:, :, 1:-1], 1 + (delta[:, :, 1:-1] + delta[:, :, :-2]) \
                        / dzt[np.newaxis, np.newaxis, 1:-1])
    b_tri = update(b_tri, at[:, :, -1], 1 + delta[:, :, -2] / dzt[np.newaxis, np.newaxis, -1])
    b_tri_edge = 1 + (delta[:, :, :] / dzt[np.newaxis, np.newaxis, :])
    c_tri = update(c_tri, at[:, :, :-1], -delta[:, :, :-1] / dzt[np.newaxis, np.newaxis, :-1])
    sol = utilities.solve_implicit(
        a_tri, b_tri, c_tri, tr[2:-2, 2:-2, :, taup1], water_mask, b_edge=b_tri_edge, edge_mask=edge_mask
    )
    return np.where(water_mask, sol, tr[2:-2, 2:-2, :, taup1])


@veros_kernel
def isoneutral_diffusion_tracer(tr, dtracer_iso, K_iso, K_gm, K_11, K_22, K_33,
                                Ai_ez, Ai_nz, Ai_bx, Ai_by, dxt, dxu, dyt, dyu, dzt,
                                dzw, cost, cosu, tau, taup1, dt_tracer, maskT, kbot,
                                flux_east, flux_north, flux_top, iso=True, skew=False):
    """
    Isoneutral diffusion for general tracers
    """
    if iso:
        K_iso = K_iso
    else:
        K_iso = np.zeros_like(K_iso)
    if skew:
        K_skew = K_gm
    else:
        K_skew = np.zeros_like(K_gm)

    flux_east, flux_north, flux_top = _calc_tracer_fluxes(
        tr, K_iso, K_skew, K_11, K_22, Ai_ez, Ai_nz, Ai_bx, Ai_by,
        flux_east, flux_north, flux_top, dxt, dxu, dyt, dyu, dzt,
        cost, cosu, tau
    )

    """
    add explicit part
    """
    aloc = _calc_explicit_part(flux_east, flux_north, flux_top, cost, dxt, dyt, dzt, maskT)
    dtracer_iso = update_add(dtracer_iso, at[...], aloc[...])
    tr = update_add(tr, at[2:-2, 2:-2, :, taup1], dt_tracer * aloc[2:-2, 2:-2, :])

    """
    add implicit part
    """
    if iso:
        aloc = update(aloc, at[...], tr[:, :, :, taup1])
        tr = update(tr, at[2:-2, 2:-2, :, taup1], _calc_implicit_part(tr, kbot, K_33, dzt, dzw, dt_tracer, taup1))
        dtracer_iso = update_add(dtracer_iso, at[...], (tr[:, :, :, taup1] - aloc) / dt_tracer)

    return tr, dtracer_iso, flux_east, flux_north, flux_top


@veros_kernel
def isoneutral_diffusion(tr, istemp, dtemp_iso, dsalt_iso, K_iso, K_gm, K_11, K_22, K_33,
                         Ai_ez, Ai_nz, Ai_bx, Ai_by, dxt, dxu, dyt, dyu, dzt,
                         dzw, cost, cosu, tau, taup1, dt_tracer, nz, maskT, maskW, kbot,
                         flux_east, flux_north, flux_top, int_drhodT, int_drhodS,
                         P_diss_skew, P_diss_iso, temp, salt, grav, rho_0,
                         enable_conserve_energy, iso=True, skew=False):
    """
    Isopycnal diffusion for tracer,
    following functional formulation by Griffies et al
    Dissipation is calculated and stored in P_diss_iso
    T/S changes are added to dtemp_iso/dsalt_iso
    """
    if istemp:
        dtracer_iso = dtemp_iso
    else:
        dtracer_iso = dsalt_iso

    tr, dtracer_iso, flux_east, flux_north, flux_top = isoneutral_diffusion_tracer(
        tr, dtracer_iso, K_iso, K_gm, K_11, K_22, K_33,
        Ai_ez, Ai_nz, Ai_bx, Ai_by, dxt, dxu, dyt, dyu, dzt,
        dzw, cost, cosu, tau, taup1, dt_tracer, maskT, kbot,
        flux_east, flux_north, flux_top, iso, skew
    )

    """
    dissipation by isopycnal mixing
    """
    if enable_conserve_energy:
        if istemp:
            int_drhodX = int_drhodT[:, :, :, tau]
        else:
            int_drhodX = int_drhodS[:, :, :, tau]

        """
        dissipation interpolated on W-grid
        """
        diss = diffusion.compute_dissipation(grav, rho_0, dxt, dyt, cost, int_drhodX, flux_east, flux_north)
        diss_wgrid = diffusion.dissipation_on_wgrid(diss, nz, dzw, kbot)

        if not iso:
            P_diss_skew += diss_wgrid
        else:
            P_diss_iso += diss_wgrid

        """
        diagnose dissipation of dynamic enthalpy by explicit and implicit vertical mixing
        """
        fxa = (-int_drhodX[2:-2, 2:-2, 1:] + int_drhodX[2:-2, 2:-2, :-1]) / \
            dzw[np.newaxis, np.newaxis, :-1]
        tracer = temp if istemp else salt
        if not iso:
            P_diss_skew = update_add(P_diss_skew, at[2:-2, 2:-2, :-1], - grav / rho_0 * \
                fxa * flux_top[2:-2, 2:-2, :-1] * maskW[2:-2, 2:-2, :-1])
            return tr, dtracer_iso, P_diss_skew
        else:
            P_diss_iso = update_add(P_diss_iso, at[2:-2, 2:-2, :-1], - grav / rho_0 * fxa * flux_top[2:-2, 2:-2, :-1] * maskW[2:-2, 2:-2, :-1] \
                - grav / rho_0 * fxa * K_33[2:-2, 2:-2, :-1] * (tracer[2:-2, 2:-2, 1:, taup1]
                                                                - tracer[2:-2, 2:-2, :-1, taup1]) \
                / dzw[np.newaxis, np.newaxis, :-1] * maskW[2:-2, 2:-2, :-1])
            return tr, dtracer_iso, P_diss_iso


@veros_kernel
def isoneutral_skew_diffusion(tr, istemp, dtemp_iso, dsalt_iso, K_iso,
                              K_gm, K_11, K_22, K_33, Ai_ez, Ai_nz, Ai_bx, Ai_by,
                              dxt, dxu, dyt, dyu, dzt, dzw, cost, cosu, tau, taup1,
                              dt_tracer, nz, maskT, maskW, kbot,
                              flux_east, flux_north, flux_top, int_drhodT, int_drhodS,
                              P_diss_skew, P_diss_iso, temp, salt, grav, rho_0,
                              enable_conserve_energy):
    """
    Isopycnal skew diffusion for tracer,
    following functional formulation by Griffies et al
    Dissipation is calculated and stored in P_diss_skew
    T/S changes are added to dtemp_iso/dsalt_iso
    """
    return isoneutral_diffusion(tr, istemp, dtemp_iso, dsalt_iso, K_iso, K_gm, K_11,
                                K_22, K_33, Ai_ez, Ai_nz, Ai_bx, Ai_by, dxt, dxu, dyt, dyu, dzt,
                                dzw, cost, cosu, tau, taup1, dt_tracer, nz, maskT, maskW, kbot,
                                flux_east, flux_north, flux_top, int_drhodT, int_drhodS,
                                P_diss_skew, P_diss_iso, temp, salt, grav, rho_0,
                                enable_conserve_energy, False, True)

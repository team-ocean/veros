from ... import veros_method
from .. import utilities, diffusion
from ...variables import allocate


@veros_method
def _calc_tracer_fluxes(vs, tr, K_iso, K_skew):
    tr_pad = utilities.pad_z_edges(vs, tr[..., vs.tau])

    K1 = K_iso - K_skew
    K2 = K_iso + K_skew

    """
    construct total isoneutral tracer flux at east face of 'T' cells
    """
    diffloc = allocate(vs, ('xt', 'yt', 'zt'))[1:-2, 2:-2]
    diffloc[:, :, 1:] = 0.25 * (K1[1:-2, 2:-2, 1:] + K1[1:-2, 2:-2, :-1] +
                                K1[2:-1, 2:-2, 1:] + K1[2:-1, 2:-2, :-1])
    diffloc[:, :, 0] = 0.5 * (K1[1:-2, 2:-2, 0] + K1[2:-1, 2:-2, 0])
    sumz = allocate(vs, ('xt', 'yt', 'zt'))[1:-2, 2:-2]
    for kr in range(2):
        for ip in range(2):
            sumz += diffloc * vs.Ai_ez[1:-2, 2:-2, :, ip, kr] * (
                tr_pad[1 + ip:-2 + ip, 2:-2, 1 + kr:-1 + kr or None] - tr_pad[1 + ip:-2 + ip, 2:-2, kr:-2 + kr])
    vs.flux_east[1:-2, 2:-2, :] = sumz / (4. * vs.dzt[np.newaxis, np.newaxis, :]) + (tr[2:-1, 2:-2, :, vs.tau] - tr[1:-2, 2:-2, :, vs.tau]) \
                                / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxu[1:-2, np.newaxis, np.newaxis]) * vs.K_11[1:-2, 2:-2, :]

    """
    construct total isoneutral tracer flux at north face of 'T' cells
    """
    diffloc = allocate(vs, ('xt', 'yt', 'zt'))[2:-2, 1:-2]
    diffloc[:, :, 1:] = 0.25 * (K1[2:-2, 1:-2, 1:] + K1[2:-2, 1:-2, :-1] +
                                K1[2:-2, 2:-1, 1:] + K1[2:-2, 2:-1, :-1])
    diffloc[:, :, 0] = 0.5 * (K1[2:-2, 1:-2, 0] + K1[2:-2, 2:-1, 0])
    sumz = allocate(vs, ('xt', 'yt', 'zt'))[2:-2, 1:-2]
    for kr in range(2):
        for jp in range(2):
            sumz += diffloc * vs.Ai_nz[2:-2, 1:-2, :, jp, kr] * (
                tr_pad[2:-2, 1 + jp:-2 + jp, 1 + kr:-1 + kr or None] - tr_pad[2:-2, 1 + jp:-2 + jp, kr:-2 + kr])
    vs.flux_north[2:-2, 1:-2, :] = vs.cosu[np.newaxis, 1:-2, np.newaxis] * (sumz / (4. * vs.dzt[np.newaxis, np.newaxis, :]) \
                                + (tr[2:-2, 2:-1, :, vs.tau] - tr[2:-2, 1:-2, :, vs.tau]) / vs.dyu[np.newaxis, 1:-2, np.newaxis] * vs.K_22[2:-2, 1:-2, :])

    """
    compute the vertical tracer flux 'vs.flux_top' containing the K31
    and K32 components which are to be solved explicitly. The K33
    component will be treated implicitly. Note that there are some
    cancellations of dxu(i-1+ip) and dyu(jrow-1+jp)
    """
    diffloc = K2[2:-2, 2:-2, :-1]
    sumx = 0.
    for ip in range(2):
        for kr in range(2):
            sumx += diffloc * vs.Ai_bx[2:-2, 2:-2, :-1, ip, kr] / vs.cost[np.newaxis, 2:-2, np.newaxis] \
                * (tr[2 + ip:-2 + ip, 2:-2, kr:-1 + kr or None, vs.tau] - tr[1 + ip:-3 + ip, 2:-2, kr:-1 + kr or None, vs.tau])
    sumy = 0.
    for jp in range(2):
        for kr in range(2):
            sumy += diffloc * vs.Ai_by[2:-2, 2:-2, :-1, jp, kr] * vs.cosu[np.newaxis, 1 + jp:-3 + jp, np.newaxis] \
                * (tr[2:-2, 2 + jp:-2 + jp, kr:-1 + kr or None, vs.tau] - tr[2:-2, 1 + jp:-3 + jp, kr:-1 + kr or None, vs.tau])
    vs.flux_top[2:-2, 2:-2, :-1] = sumx / (4 * vs.dxt[2:-2, np.newaxis, np.newaxis]) \
                                 + sumy / (4 * vs.dyt[np.newaxis, 2:-2, np.newaxis] * vs.cost[np.newaxis, 2:-2, np.newaxis])
    vs.flux_top[:, :, -1] = 0.


@veros_method
def _calc_explicit_part(vs):
    aloc = allocate(vs, ('xt', 'yt', 'zt'))
    aloc[2:-2, 2:-2, :] = vs.maskT[2:-2, 2:-2, :] * ((vs.flux_east[2:-2, 2:-2, :] - vs.flux_east[1:-3, 2:-2, :]) / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxt[2:-2, np.newaxis, np.newaxis])
                                                   + (vs.flux_north[2:-2, 2:-2, :] - vs.flux_north[2:-2, 1:-3, :]) / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dyt[np.newaxis, 2:-2, np.newaxis]))
    aloc[:, :, 0] += vs.maskT[:, :, 0] * vs.flux_top[:, :, 0] / vs.dzt[0]
    aloc[:, :, 1:] += vs.maskT[:, :, 1:] * \
        (vs.flux_top[:, :, 1:] - vs.flux_top[:, :, :-1]) / \
        vs.dzt[np.newaxis, np.newaxis, 1:]
    return aloc


@veros_method
def _calc_implicit_part(vs, tr):
    ks = vs.kbot[2:-2, 2:-2] - 1

    a_tri = allocate(vs, ('xt', 'yt', 'zt'), include_ghosts=False)
    b_tri = allocate(vs, ('xt', 'yt', 'zt'), include_ghosts=False)
    c_tri = allocate(vs, ('xt', 'yt', 'zt'), include_ghosts=False)
    delta = allocate(vs, ('xt', 'yt', 'zt'), include_ghosts=False)

    delta[:, :, :-1] = vs.dt_tracer / vs.dzw[np.newaxis, np.newaxis, :-1] * vs.K_33[2:-2, 2:-2, :-1]
    delta[:, :, -1] = 0.
    a_tri[:, :, 1:] = -delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, 1:]
    b_tri[:, :, 1:-1] = 1 + (delta[:, :, 1:-1] + delta[:, :, :-2]) \
                        / vs.dzt[np.newaxis, np.newaxis, 1:-1]
    b_tri[:, :, -1] = 1 + delta[:, :, -2] / vs.dzt[np.newaxis, np.newaxis, -1]
    b_tri_edge = 1 + (delta[:, :, :] / vs.dzt[np.newaxis, np.newaxis, :])
    c_tri[:, :, :-1] = -delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, :-1]
    sol, water_mask = utilities.solve_implicit(
        vs, ks, a_tri, b_tri, c_tri, tr[2:-2, 2:-2, :, vs.taup1], b_edge=b_tri_edge
    )
    tr[2:-2, 2:-2, :, vs.taup1] = utilities.where(vs, water_mask, sol, tr[2:-2, 2:-2, :, vs.taup1])


@veros_method
def isoneutral_diffusion(vs, tr, istemp, iso=True, skew=False):
    """
    Isopycnal diffusion for tracer,
    following functional formulation by Griffies et al
    Dissipation is calculated and stored in P_diss_iso
    T/S changes are added to dtemp_iso/dsalt_iso
    """
    if iso:
        K_iso = vs.K_iso
    else:
        K_iso = 0
    if skew:
        K_skew = vs.K_gm
    else:
        K_skew = 0

    _calc_tracer_fluxes(vs, tr, K_iso, K_skew)

    """
    add explicit part
    """
    aloc = _calc_explicit_part(vs)
    if istemp:
        vs.dtemp_iso[...] += aloc[...]
    else:
        vs.dsalt_iso[...] += aloc[...]
    tr[2:-2, 2:-2, :, vs.taup1] += vs.dt_tracer * aloc[2:-2, 2:-2, :]

    """
    add implicit part
    """
    if iso:
        aloc[...] = tr[:, :, :, vs.taup1]
        _calc_implicit_part(vs, tr)
        if istemp:
            vs.dtemp_iso += (tr[:, :, :, vs.taup1] - aloc) / vs.dt_tracer
        else:
            vs.dsalt_iso += (tr[:, :, :, vs.taup1] - aloc) / vs.dt_tracer

    """
    dissipation by isopycnal mixing
    """
    if vs.enable_conserve_energy:
        if istemp:
            int_drhodX = vs.int_drhodT[:, :, :, vs.tau]
        else:
            int_drhodX = vs.int_drhodS[:, :, :, vs.tau]

        """
        dissipation interpolated on W-grid
        """
        if not iso:
            diffusion.dissipation_on_wgrid(vs, vs.P_diss_skew, int_drhodX=int_drhodX)
        else:
            diffusion.dissipation_on_wgrid(vs, vs.P_diss_iso, int_drhodX=int_drhodX)

        """
        diagnose dissipation of dynamic enthalpy by explicit and implicit vertical mixing
        """
        fxa = (-int_drhodX[2:-2, 2:-2, 1:] + int_drhodX[2:-2, 2:-2, :-1]) / \
            vs.dzw[np.newaxis, np.newaxis, :-1]
        tracer = vs.temp if istemp else vs.salt
        if not iso:
            vs.P_diss_skew[2:-2, 2:-2, :-1] += - vs.grav / vs.rho_0 * \
                fxa * vs.flux_top[2:-2, 2:-2, :-1] * vs.maskW[2:-2, 2:-2, :-1]
        else:
            vs.P_diss_iso[2:-2, 2:-2, :-1] += - vs.grav / vs.rho_0 * fxa * vs.flux_top[2:-2, 2:-2, :-1] * vs.maskW[2:-2, 2:-2, :-1] \
                - vs.grav / vs.rho_0 * fxa * vs.K_33[2:-2, 2:-2, :-1] * (tracer[2:-2, 2:-2, 1:, vs.taup1]
                                                                                  - tracer[2:-2, 2:-2, :-1, vs.taup1]) \
                / vs.dzw[np.newaxis, np.newaxis, :-1] * vs.maskW[2:-2, 2:-2, :-1]


@veros_method
def isoneutral_skew_diffusion(vs, tr, istemp):
    """
    Isopycnal skew diffusion for tracer,
    following functional formulation by Griffies et al
    Dissipation is calculated and stored in vs.P_diss_skew
    T/S changes are added to dtemp_iso/dsalt_iso
    """
    isoneutral_diffusion(vs, tr, istemp, skew=True, iso=False)


@veros_method
def isoneutral_diffusion_all(vs, tr, istemp):
    """
    Isopycnal diffusion plus skew diffusion for tracer,
    following functional formulation by Griffies et al
    Dissipation is calculated and stored in P_diss_iso
    """
    isoneutral_diffusion(vs, tr, istemp, skew=True, iso=True)

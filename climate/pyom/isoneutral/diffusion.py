import numpy as np

from climate.pyom import numerics, utilities


def _calc_tracer_fluxes(tr, K_iso, K_skew, pyom):
    tr_pad = np.empty((pyom.nx+4,pyom.ny+4,pyom.nz+2))
    tr_pad[:,:,1:-1] = tr[...,pyom.tau]
    tr_pad[:,:,0] = tr[:,:,1,pyom.tau]
    tr_pad[:,:,-1] = tr[:,:,-2,pyom.tau]

    K1 = K_iso - K_skew
    K2 = K_iso + K_skew

    """
    construct total isoneutral tracer flux at east face of "T" cells
    """
    diffloc = np.empty((pyom.nx+1,pyom.ny,pyom.nz))
    diffloc[:,:,1:] = 0.25 * (K1[1:-2,2:-2,1:] + K1[1:-2,2:-2,:-1] + K1[2:-1,2:-2,1:] + K1[2:-1,2:-2,:-1])
    diffloc[:,:,0] = 0.5 * (K1[1:-2,2:-2,0] + K1[2:-1,2:-2,0])
    sumz = np.zeros((pyom.nx+1,pyom.ny,pyom.nz))
    for kr in xrange(2):
        for ip in xrange(2):
            sumz += diffloc * pyom.Ai_ez[1:-2,2:-2,:,ip,kr] * (tr_pad[1+ip:-2+ip,2:-2,1+kr:-1+kr or None] - tr_pad[1+ip:-2+ip,2:-2,kr:-2+kr])
    pyom.flux_east[1:-2,2:-2,:] = sumz / (4.*pyom.dzt[None,None,:]) + (tr[2:-1,2:-2,:,pyom.tau] - tr[1:-2,2:-2,:,pyom.tau]) / (pyom.cost[None,2:-2,None] * pyom.dxu[1:-2,None,None]) * pyom.K_11[1:-2,2:-2,:]

    """
    construct total isoneutral tracer flux at north face of "T" cells
    """
    diffloc = np.empty((pyom.nx,pyom.ny+1,pyom.nz))
    diffloc[:,:,1:] = 0.25 * (K1[2:-2,1:-2,1:] + K1[2:-2,1:-2,:-1] + K1[2:-2,2:-1,1:] + K1[2:-2,2:-1,:-1])
    diffloc[:,:,0] = 0.5 * (K1[2:-2,1:-2,0] + K1[2:-2,2:-1,0])
    sumz = np.zeros((pyom.nx,pyom.ny+1,pyom.nz))
    for kr in xrange(2):
        for jp in xrange(2):
            sumz += diffloc * pyom.Ai_nz[2:-2,1:-2,:,jp,kr] * (tr_pad[2:-2,1+jp:-2+jp,1+kr:-1+kr or None] - tr_pad[2:-2,1+jp:-2+jp,kr:-2+kr])
    pyom.flux_north[2:-2,1:-2,:] = pyom.cosu[None,1:-2,None] * (sumz / (4.*pyom.dzt[None,None,:]) + (tr[2:-2,2:-1,:,pyom.tau] - tr[2:-2,1:-2,:,pyom.tau]) / pyom.dyu[None,1:-2,None] * pyom.K_22[2:-2,1:-2,:])

    """
    compute the vertical tracer flux "pyom.flux_top" containing the K31
    and K32 components which are to be solved explicitly. The K33
    component will be treated implicitly. Note that there are some
    cancellations of dxu(i-1+ip) and dyu(jrow-1+jp)
    """
    diffloc = K2[2:-2,2:-2,:-1]
    sumx = 0.
    for ip in xrange(2):
        for kr in xrange(2):
            sumx += diffloc * pyom.Ai_bx[2:-2,2:-2,:-1,ip,kr] / pyom.cost[None,2:-2,None] \
                    * (tr[2+ip:-2+ip,2:-2,kr:-1+kr or None,pyom.tau] - tr[1+ip:-3+ip,2:-2,kr:-1+kr or None,pyom.tau])
    sumy = 0.
    for jp in xrange(2):
        for kr in xrange(2):
            sumy += diffloc * pyom.Ai_by[2:-2,2:-2,:-1,jp,kr] * pyom.cosu[None,1+jp:-3+jp,None] \
                    * (tr[2:-2,2+jp:-2+jp,kr:-1+kr or None,pyom.tau] - tr[2:-2,1+jp:-3+jp,kr:-1+kr or None, pyom.tau])
    pyom.flux_top[2:-2,2:-2,:-1] = sumx / (4*pyom.dxt[2:-2,None,None]) + sumy / (4*pyom.dyt[None,2:-2,None] * pyom.cost[None,2:-2,None])
    pyom.flux_top[:,:,-1] = 0.


def _calc_explicit_part(pyom):
    aloc = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    aloc[2:-2,2:-2,:] = pyom.maskT[2:-2,2:-2,:] * ((pyom.flux_east[2:-2,2:-2,:] - pyom.flux_east[1:-3,2:-2,:]) / (pyom.cost[None,2:-2,None] * pyom.dxt[2:-2,None,None]) \
                                                 + (pyom.flux_north[2:-2,2:-2,:] - pyom.flux_north[2:-2,1:-3,:]) / (pyom.cost[None,2:-2,None] * pyom.dyt[None,2:-2,None]))
    aloc[:,:,0] += pyom.maskT[:,:,0] * pyom.flux_top[:,:,0] / pyom.dzt[0]
    aloc[:,:,1:] += pyom.maskT[:,:,1:] * (pyom.flux_top[:,:,1:] - pyom.flux_top[:,:,:-1]) / pyom.dzt[None,None,1:]
    return aloc


def _calc_implicit_part(tr, pyom):
    ks = pyom.kbot[2:-2,2:-2] - 1

    a_tri = np.zeros((pyom.nx,pyom.ny,pyom.nz))
    b_tri = np.zeros((pyom.nx,pyom.ny,pyom.nz))
    c_tri = np.zeros((pyom.nx,pyom.ny,pyom.nz))
    delta = np.zeros((pyom.nx,pyom.ny,pyom.nz))

    delta[:,:,:-1] = pyom.dt_tracer / pyom.dzw[None,None,:-1] * pyom.K_33[2:-2,2:-2,:-1]
    delta[:,:,-1] = 0.
    a_tri[:,:,1:] = -delta[:,:,:-1] / pyom.dzt[None,None,1:]
    b_tri[:,:,1:-1] = 1 + (delta[:,:,1:-1] + delta[:,:,:-2]) / pyom.dzt[None,None,1:-1]
    b_tri[:,:,-1] = 1 + delta[:,:,-2] / pyom.dzt[None,None,-1]
    b_tri_edge = 1 + (delta[:,:,:] / pyom.dzt[None,None,:])
    c_tri[:,:,:-1] = -delta[:,:,:-1] / pyom.dzt[None,None,:-1]
    sol, water_mask = utilities.solve_implicit(ks, a_tri, b_tri, c_tri, tr[2:-2, 2:-2, :, pyom.taup1], pyom, b_edge=b_tri_edge)
    tr[2:-2,2:-2,:,pyom.taup1][water_mask] = sol


def _dissipation_on_wgrid(P, int_drhodX, pyom):
    aloc = np.zeros_like(P)
    aloc[1:-1,1:-1,:] = 0.5 * pyom.grav / pyom.rho_0 * ((int_drhodX[2:,1:-1,:]-int_drhodX[1:-1,1:-1,:]) * pyom.flux_east[1:-1,1:-1,:] \
                                                       +(int_drhodX[1:-1,1:-1,:]-int_drhodX[:-2,1:-1,:]) * pyom.flux_east[:-2,1:-1,:]) \
                                                     / (pyom.dxt[1:-1,None,None] * pyom.cost[None,1:-1,None]) \
                      + 0.5 * pyom.grav / pyom.rho_0 * ((int_drhodX[1:-1,2:,:]-int_drhodX[1:-1,1:-1,:]) * pyom.flux_north[1:-1,1:-1,:] \
                                                       +(int_drhodX[1:-1,1:-1,:]-int_drhodX[1:-1,:-2,:]) * pyom.flux_north[1:-1,:-2,:]) \
                                                     / (pyom.dyt[None,1:-1,None] * pyom.cost[None,1:-1,None])

    ks = pyom.kbot[:,:] - 1
    land_mask = (ks >= 0)
    edge_mask = land_mask[:, :, None] & (np.indices((pyom.nx+4, pyom.ny+4, pyom.nz-1))[2] == ks[:,:,None])
    water_mask = land_mask[:, :, None] & (np.indices((pyom.nx+4, pyom.ny+4, pyom.nz-1))[2] > ks[:,:,None])
    if np.count_nonzero(land_mask):
        dzw_pad = utilities.pad_z_edges(pyom.dzw)
        P[:, :, :-1][edge_mask] += (0.5 * (aloc[:,:,:-1] + aloc[:,:,1:]) + 0.5 * (aloc[:, :, :-1] * dzw_pad[None, None, :-3] / pyom.dzw[None, None, :-1]))[edge_mask]
        P[:, :, :-1][water_mask] += 0.5 * (aloc[:,:,:-1] + aloc[:,:,1:])[water_mask]
        P[:, :, -1][land_mask] += aloc[:,:,-1][land_mask]


def isoneutral_diffusion(tr, istemp, pyom, iso=True, skew=False):
    """
    Isopycnal diffusion for tracer,
    following functional formulation by Griffies et al
    Dissipation is calculated and stored in P_diss_iso
    T/S changes are added to dtemp_iso/dsalt_iso
    """
    if iso:
        K_iso = pyom.K_iso
    else:
        K_iso = np.zeros_like(pyom.K_iso)
    if skew:
        K_skew = pyom.K_gm
    else:
        K_skew = np.zeros_like(pyom.K_gm)

    _calc_tracer_fluxes(tr, K_iso, K_skew, pyom)

    """
    add explicit part
    """
    aloc = _calc_explicit_part(pyom)
    if istemp:
        pyom.dtemp_iso[...] += aloc[...]
    else:
        pyom.dsalt_iso[...] += aloc[...]
    tr[2:-2, 2:-2, :, pyom.taup1] += pyom.dt_tracer * aloc[2:-2, 2:-2, :]

    """
    add implicit part
    """
    if iso:
        aloc[...] = tr[:,:,:,pyom.taup1]
        _calc_implicit_part(tr,pyom)
        if istemp:
            pyom.dtemp_iso += (tr[:,:,:,pyom.taup1] - aloc) / pyom.dt_tracer
        else:
            pyom.dsalt_iso += (tr[:,:,:,pyom.taup1] - aloc) / pyom.dt_tracer

    """
    dissipation by isopycnal mixing
    """
    if pyom.enable_conserve_energy:
        if istemp:
            int_drhodX = pyom.int_drhodT[:,:,:,pyom.tau]
        else:
            int_drhodX = pyom.int_drhodS[:,:,:,pyom.tau]

        """
        dissipation interpolated on W-grid
        """
        if not iso:
            _dissipation_on_wgrid(pyom.P_diss_skew, int_drhodX, pyom)
        else:
            _dissipation_on_wgrid(pyom.P_diss_iso, int_drhodX, pyom)

        """
        diagnose dissipation of dynamic enthalpy by explicit and implicit vertical mixing
        """
        fxa = (-int_drhodX[2:-2,2:-2,1:] + int_drhodX[2:-2,2:-2,:-1]) / pyom.dzw[None,None,:-1]
        tracer = pyom.temp if istemp else pyom.salt
        if not iso:
            pyom.P_diss_skew[2:-2,2:-2,:-1] += - pyom.grav / pyom.rho_0 * fxa * pyom.flux_top[2:-2,2:-2,:-1] * pyom.maskW[2:-2,2:-2,:-1]
        else:
            pyom.P_diss_iso[2:-2,2:-2,:-1] += - pyom.grav / pyom.rho_0 * fxa * pyom.flux_top[2:-2,2:-2,:-1] * pyom.maskW[2:-2,2:-2,:-1] \
                                              - pyom.grav / pyom.rho_0 * fxa * pyom.K_33[2:-2,2:-2,:-1] * (tracer[2:-2,2:-2,1:,pyom.taup1] \
                                                                                                         - tracer[2:-2,2:-2,:-1,pyom.taup1]) \
                                                                                        / pyom.dzw[None,None,:-1] * pyom.maskW[2:-2,2:-2,:-1]


def isoneutral_skew_diffusion(tr,istemp,pyom):
    """
    Isopycnal skew diffusion for tracer,
    following functional formulation by Griffies et al
    Dissipation is calculated and stored in pyom.P_diss_skew
    T/S changes are added to dtemp_iso/dsalt_iso
    """
    isoneutral_diffusion(tr,istemp,pyom,skew=True,iso=False)


def isoneutral_diffusion_all(tr,istemp,pyom):
    """
    Isopycnal diffusion plus skew diffusion for tracer,
    following functional formulation by Griffies et al
    Dissipation is calculated and stored in P_diss_iso
    """
    isoneutral_diffusion(tr,istemp,pyom,skew=True,iso=True)

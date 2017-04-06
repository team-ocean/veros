from ... import veros_method
from .. import numerics, utilities, diffusion

@veros_method
def _calc_tracer_fluxes(veros, tr, K_iso, K_skew):
    tr_pad = np.empty((veros.nx+4,veros.ny+4,veros.nz+2))
    tr_pad[:,:,1:-1] = tr[...,veros.tau]
    tr_pad[:,:,0] = tr[:,:,1,veros.tau]
    tr_pad[:,:,-1] = tr[:,:,-2,veros.tau]

    K1 = K_iso - K_skew
    K2 = K_iso + K_skew

    """
    construct total isoneutral tracer flux at east face of "T" cells
    """
    diffloc = np.empty((veros.nx+1,veros.ny,veros.nz))
    diffloc[:,:,1:] = 0.25 * (K1[1:-2,2:-2,1:] + K1[1:-2,2:-2,:-1] + K1[2:-1,2:-2,1:] + K1[2:-1,2:-2,:-1])
    diffloc[:,:,0] = 0.5 * (K1[1:-2,2:-2,0] + K1[2:-1,2:-2,0])
    sumz = np.zeros((veros.nx+1,veros.ny,veros.nz))
    for kr in xrange(2):
        for ip in xrange(2):
            sumz += diffloc * veros.Ai_ez[1:-2,2:-2,:,ip,kr] * (tr_pad[1+ip:-2+ip,2:-2,1+kr:-1+kr or None] - tr_pad[1+ip:-2+ip,2:-2,kr:-2+kr])
    veros.flux_east[1:-2,2:-2,:] = sumz / (4.*veros.dzt[np.newaxis,np.newaxis,:]) + (tr[2:-1,2:-2,:,veros.tau] - tr[1:-2,2:-2,:,veros.tau]) / (veros.cost[np.newaxis,2:-2,np.newaxis] * veros.dxu[1:-2,np.newaxis,np.newaxis]) * veros.K_11[1:-2,2:-2,:]

    """
    construct total isoneutral tracer flux at north face of "T" cells
    """
    diffloc = np.empty((veros.nx,veros.ny+1,veros.nz))
    diffloc[:,:,1:] = 0.25 * (K1[2:-2,1:-2,1:] + K1[2:-2,1:-2,:-1] + K1[2:-2,2:-1,1:] + K1[2:-2,2:-1,:-1])
    diffloc[:,:,0] = 0.5 * (K1[2:-2,1:-2,0] + K1[2:-2,2:-1,0])
    sumz = np.zeros((veros.nx,veros.ny+1,veros.nz))
    for kr in xrange(2):
        for jp in xrange(2):
            sumz += diffloc * veros.Ai_nz[2:-2,1:-2,:,jp,kr] * (tr_pad[2:-2,1+jp:-2+jp,1+kr:-1+kr or None] - tr_pad[2:-2,1+jp:-2+jp,kr:-2+kr])
    veros.flux_north[2:-2,1:-2,:] = veros.cosu[np.newaxis,1:-2,np.newaxis] * (sumz / (4.*veros.dzt[np.newaxis,np.newaxis,:]) + (tr[2:-2,2:-1,:,veros.tau] - tr[2:-2,1:-2,:,veros.tau]) / veros.dyu[np.newaxis,1:-2,np.newaxis] * veros.K_22[2:-2,1:-2,:])

    """
    compute the vertical tracer flux "veros.flux_top" containing the K31
    and K32 components which are to be solved explicitly. The K33
    component will be treated implicitly. Note that there are some
    cancellations of dxu(i-1+ip) and dyu(jrow-1+jp)
    """
    diffloc = K2[2:-2,2:-2,:-1]
    sumx = 0.
    for ip in xrange(2):
        for kr in xrange(2):
            sumx += diffloc * veros.Ai_bx[2:-2,2:-2,:-1,ip,kr] / veros.cost[np.newaxis,2:-2,np.newaxis] \
                    * (tr[2+ip:-2+ip,2:-2,kr:-1+kr or None,veros.tau] - tr[1+ip:-3+ip,2:-2,kr:-1+kr or None,veros.tau])
    sumy = 0.
    for jp in xrange(2):
        for kr in xrange(2):
            sumy += diffloc * veros.Ai_by[2:-2,2:-2,:-1,jp,kr] * veros.cosu[np.newaxis,1+jp:-3+jp,np.newaxis] \
                    * (tr[2:-2,2+jp:-2+jp,kr:-1+kr or None,veros.tau] - tr[2:-2,1+jp:-3+jp,kr:-1+kr or None, veros.tau])
    veros.flux_top[2:-2,2:-2,:-1] = sumx / (4*veros.dxt[2:-2,np.newaxis,np.newaxis]) + sumy / (4*veros.dyt[np.newaxis,2:-2,np.newaxis] * veros.cost[np.newaxis,2:-2,np.newaxis])
    veros.flux_top[:,:,-1] = 0.

@veros_method
def _calc_explicit_part(veros):
    aloc = np.zeros((veros.nx+4, veros.ny+4, veros.nz))
    aloc[2:-2,2:-2,:] = veros.maskT[2:-2,2:-2,:] * ((veros.flux_east[2:-2,2:-2,:] - veros.flux_east[1:-3,2:-2,:]) / (veros.cost[np.newaxis,2:-2,np.newaxis] * veros.dxt[2:-2,np.newaxis,np.newaxis]) \
                                                 + (veros.flux_north[2:-2,2:-2,:] - veros.flux_north[2:-2,1:-3,:]) / (veros.cost[np.newaxis,2:-2,np.newaxis] * veros.dyt[np.newaxis,2:-2,np.newaxis]))
    aloc[:,:,0] += veros.maskT[:,:,0] * veros.flux_top[:,:,0] / veros.dzt[0]
    aloc[:,:,1:] += veros.maskT[:,:,1:] * (veros.flux_top[:,:,1:] - veros.flux_top[:,:,:-1]) / veros.dzt[np.newaxis,np.newaxis,1:]
    return aloc

@veros_method
def _calc_implicit_part(veros, tr):
    ks = veros.kbot[2:-2,2:-2] - 1

    a_tri = np.zeros((veros.nx,veros.ny,veros.nz))
    b_tri = np.zeros((veros.nx,veros.ny,veros.nz))
    c_tri = np.zeros((veros.nx,veros.ny,veros.nz))
    delta = np.zeros((veros.nx,veros.ny,veros.nz))

    delta[:,:,:-1] = veros.dt_tracer / veros.dzw[np.newaxis,np.newaxis,:-1] * veros.K_33[2:-2,2:-2,:-1]
    delta[:,:,-1] = 0.
    a_tri[:,:,1:] = -delta[:,:,:-1] / veros.dzt[np.newaxis,np.newaxis,1:]
    b_tri[:,:,1:-1] = 1 + (delta[:,:,1:-1] + delta[:,:,:-2]) / veros.dzt[np.newaxis,np.newaxis,1:-1]
    b_tri[:,:,-1] = 1 + delta[:,:,-2] / veros.dzt[np.newaxis,np.newaxis,-1]
    b_tri_edge = 1 + (delta[:,:,:] / veros.dzt[np.newaxis,np.newaxis,:])
    c_tri[:,:,:-1] = -delta[:,:,:-1] / veros.dzt[np.newaxis,np.newaxis,:-1]
    sol, water_mask = utilities.solve_implicit(veros, ks, a_tri, b_tri, c_tri, tr[2:-2, 2:-2, :, veros.taup1], b_edge=b_tri_edge)
    tr[2:-2,2:-2,:,veros.taup1] = np.where(water_mask, sol, tr[2:-2,2:-2,:,veros.taup1])

@veros_method
def isoneutral_diffusion(veros, tr, istemp, iso=True, skew=False):
    """
    Isopycnal diffusion for tracer,
    following functional formulation by Griffies et al
    Dissipation is calculated and stored in P_diss_iso
    T/S changes are added to dtemp_iso/dsalt_iso
    """
    if iso:
        K_iso = veros.K_iso
    else:
        K_iso = np.zeros_like(veros.K_iso)
    if skew:
        K_skew = veros.K_gm
    else:
        K_skew = np.zeros_like(veros.K_gm)

    _calc_tracer_fluxes(veros, tr, K_iso, K_skew)

    """
    add explicit part
    """
    aloc = _calc_explicit_part(veros)
    if istemp:
        veros.dtemp_iso[...] += aloc[...]
    else:
        veros.dsalt_iso[...] += aloc[...]
    tr[2:-2, 2:-2, :, veros.taup1] += veros.dt_tracer * aloc[2:-2, 2:-2, :]

    """
    add implicit part
    """
    if iso:
        aloc[...] = tr[:,:,:,veros.taup1]
        _calc_implicit_part(veros,tr)
        if istemp:
            veros.dtemp_iso += (tr[:,:,:,veros.taup1] - aloc) / veros.dt_tracer
        else:
            veros.dsalt_iso += (tr[:,:,:,veros.taup1] - aloc) / veros.dt_tracer

    """
    dissipation by isopycnal mixing
    """
    if veros.enable_conserve_energy:
        if istemp:
            int_drhodX = veros.int_drhodT[:,:,:,veros.tau]
        else:
            int_drhodX = veros.int_drhodS[:,:,:,veros.tau]

        """
        dissipation interpolated on W-grid
        """
        if not iso:
            diffusion.dissipation_on_wgrid(veros, veros.P_diss_skew, int_drhodX=int_drhodX)
        else:
            diffusion.dissipation_on_wgrid(veros, veros.P_diss_iso, int_drhodX=int_drhodX)

        """
        diagnose dissipation of dynamic enthalpy by explicit and implicit vertical mixing
        """
        fxa = (-int_drhodX[2:-2,2:-2,1:] + int_drhodX[2:-2,2:-2,:-1]) / veros.dzw[np.newaxis,np.newaxis,:-1]
        tracer = veros.temp if istemp else veros.salt
        if not iso:
            veros.P_diss_skew[2:-2,2:-2,:-1] += - veros.grav / veros.rho_0 * fxa * veros.flux_top[2:-2,2:-2,:-1] * veros.maskW[2:-2,2:-2,:-1]
        else:
            veros.P_diss_iso[2:-2,2:-2,:-1] += - veros.grav / veros.rho_0 * fxa * veros.flux_top[2:-2,2:-2,:-1] * veros.maskW[2:-2,2:-2,:-1] \
                                              - veros.grav / veros.rho_0 * fxa * veros.K_33[2:-2,2:-2,:-1] * (tracer[2:-2,2:-2,1:,veros.taup1] \
                                                                                                         - tracer[2:-2,2:-2,:-1,veros.taup1]) \
                                                                                        / veros.dzw[np.newaxis,np.newaxis,:-1] * veros.maskW[2:-2,2:-2,:-1]

@veros_method
def isoneutral_skew_diffusion(veros,tr,istemp):
    """
    Isopycnal skew diffusion for tracer,
    following functional formulation by Griffies et al
    Dissipation is calculated and stored in veros.P_diss_skew
    T/S changes are added to dtemp_iso/dsalt_iso
    """
    isoneutral_diffusion(veros,tr,istemp,skew=True,iso=False)

@veros_method
def isoneutral_diffusion_all(veros,tr,istemp):
    """
    Isopycnal diffusion plus skew diffusion for tracer,
    following functional formulation by Griffies et al
    Dissipation is calculated and stored in P_diss_iso
    """
    isoneutral_diffusion(veros,tr,istemp,skew=True,iso=True)

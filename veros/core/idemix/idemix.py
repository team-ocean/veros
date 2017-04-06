from .. import advection, numerics, utilities
from ... import veros_method

@veros_method
def set_idemix_parameter(veros):
    """
    set main IDEMIX parameter
    """
    bN0 = np.sum(np.sqrt(np.maximum(0., veros.Nsqr[:,:,:-1,veros.tau])) * veros.dzw[np.newaxis, np.newaxis, :-1] * veros.maskW[:,:,:-1], axis=2) \
        + np.sqrt(np.maximum(0., veros.Nsqr[:,:,-1,veros.tau])) * 0.5 * veros.dzw[-1:] * veros.maskW[:,:,-1]
    fxa = np.sqrt(np.maximum(0., veros.Nsqr[...,veros.tau])) / (1e-22 + np.abs(veros.coriolis_t[...,np.newaxis]))
    cstar = np.maximum(1e-2, bN0[:,:,np.newaxis] / (veros.pi * veros.jstar))
    veros.c0[...] = np.maximum(0., veros.gamma * cstar * gofx2(veros,fxa) * veros.maskW)
    veros.v0[...] = np.maximum(0., veros.gamma * cstar * hofx1(veros,fxa) * veros.maskW)
    veros.alpha_c[...] = np.maximum(1e-4, veros.mu0 * np.arccosh(np.maximum(1.,fxa)) * np.abs(veros.coriolis_t[...,np.newaxis]) / cstar**2) * veros.maskW

@veros_method
def integrate_idemix(veros):
    """
    integrate idemix on W grid
    """
    a_tri, b_tri, c_tri, d_tri, delta = (np.zeros((veros.nx, veros.ny, veros.nz)) for _ in range(5))
    forc = np.zeros((veros.nx+4, veros.ny+4, veros.nz))
    maxE_iw = np.zeros((veros.nx+4, veros.ny+4, veros.nz))

    """
    forcing by EKE dissipation
    """
    if veros.enable_eke:
        forc[...] = veros.eke_diss_iw
    else: # shortcut without EKE model
        if veros.enable_store_cabbeling_heat:
            forc[...] = veros.K_diss_gm + veros.K_diss_h - veros.P_diss_skew - veros.P_diss_hmix  - veros.P_diss_iso
        else:
            forc[...] = veros.K_diss_gm + veros.K_diss_h - veros.P_diss_skew

    if veros.enable_eke and (veros.enable_eke_diss_bottom or veros.enable_eke_diss_surfbot):
        """
        vertically integrate EKE dissipation and inject at bottom and/or surface
        """
        a_loc = np.sum(veros.dzw[np.newaxis, np.newaxis, :-1] * forc[:,:,:-1] * veros.maskW[:,:,:-1], axis=2)
        a_loc += 0.5 * forc[:,:,-1] * veros.maskW[:,:,-1] * veros.dzw[-1]
        forc[...] = 0.

        ks = np.maximum(0, veros.kbot[2:-2, 2:-2] - 1)
        mask = ks[:,:,np.newaxis] == np.arange(veros.nz)[np.newaxis, np.newaxis, :]
        if veros.enable_eke_diss_bottom:
            forc[2:-2, 2:-2, :] = np.where(mask, a_loc[2:-2, 2:-2, np.newaxis] / veros.dzw[np.newaxis, np.newaxis, :], forc[2:-2, 2:-2, :])
        else:
            forc[2:-2, 2:-2, :] = np.where(mask, veros.eke_diss_surfbot_frac * a_loc[2:-2, 2:-2, np.newaxis] / veros.dzw[np.newaxis, np.newaxis, :], forc[2:-2, 2:-2, :])
            forc[2:-2, 2:-2, -1] = (1. - veros.eke_diss_surfbot_frac) * a_loc[2:-2, 2:-2] / (0.5 * veros.dzw[-1])

    """
    forcing by bottom friction
    """
    if not veros.enable_store_bottom_friction_tke:
        forc += veros.K_diss_bot

    """
    prevent negative dissipation of IW energy
    """
    maxE_iw[...] = np.maximum(0., veros.E_iw[:,:,:,veros.tau])

    """
    vertical diffusion and dissipation is solved implicitly
    """
    ks = veros.kbot[2:-2, 2:-2] - 1
    delta[:,:,:-1] = veros.dt_tracer * veros.tau_v / veros.dzt[np.newaxis, np.newaxis, 1:] * 0.5 \
                     * (veros.c0[2:-2, 2:-2, :-1] + veros.c0[2:-2, 2:-2, 1:])
    delta[:,:,-1] = 0.
    a_tri[:,:,1:-1] = -delta[:,:,:-2] * veros.c0[2:-2,2:-2,:-2] / veros.dzw[np.newaxis, np.newaxis, 1:-1]
    a_tri[:,:,-1] = -delta[:,:,-2] / (0.5 * veros.dzw[-1:]) * veros.c0[2:-2,2:-2,-2]
    b_tri[:,:,1:-1] = 1 + delta[:,:,1:-1] * veros.c0[2:-2, 2:-2, 1:-1] / veros.dzw[np.newaxis, np.newaxis, 1:-1] \
                      + delta[:,:,:-2] * veros.c0[2:-2, 2:-2, 1:-1] / veros.dzw[np.newaxis, np.newaxis, 1:-1] \
                      + veros.dt_tracer * veros.alpha_c[2:-2, 2:-2, 1:-1] * maxE_iw[2:-2, 2:-2, 1:-1]
    b_tri[:,:,-1] = 1 + delta[:,:,-2] / (0.5 * veros.dzw[-1:]) * veros.c0[2:-2, 2:-2, -1] \
                    + veros.dt_tracer * veros.alpha_c[2:-2, 2:-2, -1] * maxE_iw[2:-2, 2:-2, -1]
    b_tri_edge = 1 + delta / veros.dzw * veros.c0[2:-2, 2:-2, :] \
                 + veros.dt_tracer * veros.alpha_c[2:-2, 2:-2, :] * maxE_iw[2:-2, 2:-2, :]
    c_tri[:,:,:-1] = -delta[:,:,:-1] / veros.dzw[np.newaxis, np.newaxis, :-1] * veros.c0[2:-2, 2:-2, 1:]
    d_tri[...] = veros.E_iw[2:-2, 2:-2, :, veros.tau] + veros.dt_tracer * forc[2:-2, 2:-2, :]
    d_tri_edge = d_tri + veros.dt_tracer * veros.forc_iw_bottom[2:-2, 2:-2, np.newaxis] / veros.dzw[np.newaxis, np.newaxis, :]
    d_tri[:,:,-1] += veros.dt_tracer * veros.forc_iw_surface[2:-2, 2:-2] / (0.5 * veros.dzw[-1:])
    sol, water_mask = utilities.solve_implicit(veros, ks, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge, d_edge=d_tri_edge)
    veros.E_iw[2:-2, 2:-2, :, veros.taup1] = np.where(water_mask, sol, veros.E_iw[2:-2, 2:-2, :, veros.taup1])

    """
    store IW dissipation
    """
    veros.iw_diss[...] = veros.alpha_c * maxE_iw * veros.E_iw[...,veros.taup1]

    """
    add tendency due to lateral diffusion
    """
    if veros.enable_idemix_hor_diffusion:
        veros.flux_east[:-1,:,:] = veros.tau_h * 0.5 * (veros.v0[1:,:,:] + veros.v0[:-1,:,:]) \
                                * (veros.v0[1:,:,:] * veros.E_iw[1:,:,:,veros.tau] - veros.v0[:-1,:,:] * veros.E_iw[:-1,:,:,veros.tau]) \
                                / (veros.cost[np.newaxis, :, np.newaxis] * veros.dxu[:-1, np.newaxis, np.newaxis]) * veros.maskU[:-1,:,:]
        veros.flux_east[-5,:,:] = 0. # NOTE: probably a mistake in the fortran code, first index should be -1
        veros.flux_north[:,:-1,:] = veros.tau_h * 0.5 * (veros.v0[:,1:,:] + veros.v0[:,:-1,:]) \
                                 * (veros.v0[:,1:,:] * veros.E_iw[:,1:,:,veros.tau] - veros.v0[:,:-1,:] * veros.E_iw[:,:-1,:,veros.tau]) \
                                 / veros.dyu[np.newaxis, :-1, np.newaxis] * veros.maskV[:,:-1,:] * veros.cosu[np.newaxis, :-1, np.newaxis]
        veros.flux_north[:,-1,:] = 0.
        veros.E_iw[2:-2, 2:-2, :, veros.taup1] += veros.dt_tracer * veros.maskW[2:-2,2:-2,:] \
                                * ((veros.flux_east[2:-2, 2:-2, :] - veros.flux_east[1:-3, 2:-2, :]) \
                                    / (veros.cost[np.newaxis, 2:-2, np.newaxis] * veros.dxt[2:-2, np.newaxis, np.newaxis]) \
                                    + (veros.flux_north[2:-2, 2:-2, :] - veros.flux_north[2:-2, 1:-3, :]) \
                                    / (veros.cost[np.newaxis, 2:-2, np.newaxis] * veros.dyt[np.newaxis, 2:-2, np.newaxis]))

    """
    add tendency due to advection
    """
    if veros.enable_idemix_superbee_advection:
        advection.adv_flux_superbee_wgrid(veros,veros.flux_east,veros.flux_north,veros.flux_top,veros.E_iw[:,:,:,veros.tau])

    if veros.enable_idemix_upwind_advection:
        advection.adv_flux_upwind_wgrid(veros,veros.flux_east,veros.flux_north,veros.flux_top,veros.E_iw[:,:,:,veros.tau])

    if veros.enable_idemix_superbee_advection or veros.enable_idemix_upwind_advection:
        veros.dE_iw[2:-2, 2:-2, :, veros.tau] = veros.maskW[2:-2, 2:-2, :] * (-(veros.flux_east[2:-2, 2:-2, :] - veros.flux_east[1:-3, 2:-2, :]) \
                                                                                / (veros.cost[np.newaxis, 2:-2, np.newaxis] * veros.dxt[2:-2, np.newaxis, np.newaxis]) \
                                                                           - (veros.flux_north[2:-2, 2:-2, :] - veros.flux_north[2:-2, 1:-3, :]) \
                                                                                / (veros.cost[np.newaxis, 2:-2, np.newaxis] * veros.dyt[np.newaxis, 2:-2, np.newaxis]))
        veros.dE_iw[:,:,0,veros.tau] += -veros.flux_top[:,:,0] / veros.dzw[0:1]
        veros.dE_iw[:,:,1:-1,veros.tau] += -(veros.flux_top[:,:,1:-1] - veros.flux_top[:,:,:-2]) / veros.dzw[np.newaxis, np.newaxis, 1:-1]
        veros.dE_iw[:,:,-1,veros.tau] += -(veros.flux_top[:,:,-1] - veros.flux_top[:,:,-2]) / (0.5 * veros.dzw[-1:])

        """
        Adam Bashforth time stepping
        """
        veros.E_iw[:,:,:,veros.taup1] += veros.dt_tracer * ((1.5 + veros.AB_eps) * veros.dE_iw[:,:,:,veros.tau] \
                                                       - (0.5 + veros.AB_eps) * veros.dE_iw[:,:,:,veros.taum1])

@veros_method
def gofx2(veros,x):
    """
    a function g(x)
    """
    x[x < 3.] = 3. # NOTE: probably a mistake in the fortran code, should just set x locally
    c = 1.-(2./veros.pi) * np.arcsin(1./x)
    return 2. / veros.pi / c * 0.9 * x**(-2./3.) * (1 - np.exp(-x/4.3))

@veros_method
def hofx1(veros,x):
    """
    a function h(x)
    """
    return (2. / veros.pi) / (1. - (2. / veros.pi) * np.arcsin(1./x)) * (x-1.) / (x+1.)

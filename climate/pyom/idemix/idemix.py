import numpy as np

from climate.pyom import advection, numerics, utilities

def set_idemix_parameter(pyom):
    """
    set main IDEMIX parameter
    """
    bN0 = np.sum(np.sqrt(np.maximum(0., pyom.Nsqr[:,:,:-1,pyom.tau])) * pyom.dzw[None, None, :-1] * pyom.maskW[:,:,:-1], axis=2) \
        + np.sqrt(np.maximum(0., pyom.Nsqr[:,:,-1,pyom.tau])) * 0.5 * pyom.dzw[-1:] * pyom.maskW[:,:,-1]
    fxa = np.sqrt(np.maximum(0., pyom.Nsqr[...,pyom.tau])) / (1e-22 + np.abs(pyom.coriolis_t[...,None]))
    cstar = np.maximum(1e-2, bN0[:,:,None] / (pyom.pi * pyom.jstar))
    pyom.c0[...] = np.maximum(0., pyom.gamma * cstar * gofx2(fxa,pyom) * pyom.maskW)
    pyom.v0[...] = np.maximum(0., pyom.gamma * cstar * hofx1(fxa,pyom) * pyom.maskW)
    pyom.alpha_c[...] = np.maximum(1e-4, pyom.mu0 * np.arccosh(np.maximum(1.,fxa)) * np.abs(pyom.coriolis_t[...,None]) / cstar**2) * pyom.maskW


def integrate_idemix(pyom):
    """
    integrate idemix on W grid
    """
    a_tri, b_tri, c_tri, d_tri, delta = (np.zeros((pyom.nx, pyom.ny, pyom.nz)) for _ in range(5))
    forc = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    maxE_iw = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    """
    forcing by EKE dissipation
    """
    if pyom.enable_eke:
        forc[...] = pyom.eke_diss_iw
    else: # shortcut without EKE model
        if pyom.enable_store_cabbeling_heat:
            forc[...] = pyom.K_diss_gm + pyom.K_diss_h - pyom.P_diss_skew - pyom.P_diss_hmix  - pyom.P_diss_iso
        else:
            forc[...] = pyom.K_diss_gm + pyom.K_diss_h - pyom.P_diss_skew

    if pyom.enable_eke and (pyom.enable_eke_diss_bottom or pyom.enable_eke_diss_surfbot):
        """
        vertically integrate EKE dissipation and inject at bottom and/or surface
        """
        a_loc = np.sum(pyom.dzw[None, None, :-1] * forc[:,:,:-1] * pyom.maskW[:,:,:-1], axis=2)
        a_loc += 0.5 * forc[:,:,-1] * pyom.maskW[:,:,-1] * pyom.dzw[-1]
        forc[...] = 0.

        ks = np.maximum(0, pyom.kbot[2:-2, 2:-2] - 1)
        mask = ks[:,:,None] == np.indices((pyom.nx, pyom.ny, pyom.nz))[2]
        if pyom.enable_eke_diss_bottom:
            forc[2:-2, 2:-2, :] = np.where(mask, a_loc[2:-2, 2:-2, None] / pyom.dzw[None, None, :], forc[2:-2, 2:-2, :])
        else:
            forc[2:-2, 2:-2, :] = np.where(mask, pyom.eke_diss_surfbot_frac * a_loc[2:-2, 2:-2, None] / pyom.dzw[None, None, :], forc[2:-2, 2:-2, :])
            forc[2:-2, 2:-2, -1] = (1. - pyom.eke_diss_surfbot_frac) * a_loc[2:-2, 2:-2] / (0.5 * pyom.dzw[-1])

    """
    forcing by bottom friction
    """
    if not pyom.enable_store_bottom_friction_tke:
        forc += pyom.K_diss_bot

    """
    prevent negative dissipation of IW energy
    """
    maxE_iw[...] = np.maximum(0., pyom.E_iw[:,:,:,pyom.tau])

    """
    vertical diffusion and dissipation is solved implicitly
    """
    ks = pyom.kbot[2:-2, 2:-2] - 1
    delta[:,:,:-1] = pyom.dt_tracer * pyom.tau_v / pyom.dzt[None, None, 1:] * 0.5 \
                     * (pyom.c0[2:-2, 2:-2, :-1] + pyom.c0[2:-2, 2:-2, 1:])
    delta[:,:,-1] = 0.
    a_tri[:,:,1:-1] = -delta[:,:,:-2] * pyom.c0[2:-2,2:-2,:-2] / pyom.dzw[None, None, 1:-1]
    a_tri[:,:,-1] = -delta[:,:,-2] / (0.5 * pyom.dzw[-1:]) * pyom.c0[2:-2,2:-2,-2]
    b_tri[:,:,1:-1] = 1 + delta[:,:,1:-1] * pyom.c0[2:-2, 2:-2, 1:-1] / pyom.dzw[None, None, 1:-1] \
                      + delta[:,:,:-2] * pyom.c0[2:-2, 2:-2, 1:-1] / pyom.dzw[None, None, 1:-1] \
                      + pyom.dt_tracer * pyom.alpha_c[2:-2, 2:-2, 1:-1] * maxE_iw[2:-2, 2:-2, 1:-1]
    b_tri[:,:,-1] = 1 + delta[:,:,-2] / (0.5 * pyom.dzw[-1:]) * pyom.c0[2:-2, 2:-2, -1] \
                    + pyom.dt_tracer * pyom.alpha_c[2:-2, 2:-2, -1] * maxE_iw[2:-2, 2:-2, -1]
    b_tri_edge = 1 + delta / pyom.dzw * pyom.c0[2:-2, 2:-2, :] \
                 + pyom.dt_tracer * pyom.alpha_c[2:-2, 2:-2, :] * maxE_iw[2:-2, 2:-2, :]
    c_tri[:,:,:-1] = -delta[:,:,:-1] / pyom.dzw[None, None, :-1] * pyom.c0[2:-2, 2:-2, 1:]
    d_tri[...] = pyom.E_iw[2:-2, 2:-2, :, pyom.tau] + pyom.dt_tracer * forc[2:-2, 2:-2, :]
    d_tri_edge = d_tri + pyom.dt_tracer * pyom.forc_iw_bottom[2:-2, 2:-2, None] / pyom.dzw[None, None, :]
    d_tri[:,:,-1] += pyom.dt_tracer * pyom.forc_iw_surface[2:-2, 2:-2] / (0.5 * pyom.dzw[-1:])
    sol, water_mask = utilities.solve_implicit(ks, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge, d_edge=d_tri_edge)
    pyom.E_iw[2:-2, 2:-2, :, pyom.taup1] = np.where(water_mask, sol, pyom.E_iw[2:-2, 2:-2, :, pyom.taup1])

    """
    store IW dissipation
    """
    pyom.iw_diss[...] = pyom.alpha_c * maxE_iw * pyom.E_iw[...,pyom.taup1]

    """
    add tendency due to lateral diffusion
    """
    if pyom.enable_idemix_hor_diffusion:
        pyom.flux_east[:-1,:,:] = pyom.tau_h * 0.5 * (pyom.v0[1:,:,:] + pyom.v0[:-1,:,:]) \
                                * (pyom.v0[1:,:,:] * pyom.E_iw[1:,:,:,pyom.tau] - pyom.v0[:-1,:,:] * pyom.E_iw[:-1,:,:,pyom.tau]) \
                                / (pyom.cost[None, :, None] * pyom.dxu[:-1, None, None]) * pyom.maskU[:-1,:,:]
        pyom.flux_east[-5,:,:] = 0. # NOTE: probably a mistake in the fortran code, first index should be -1
        pyom.flux_north[:,:-1,:] = pyom.tau_h * 0.5 * (pyom.v0[:,1:,:] + pyom.v0[:,:-1,:]) \
                                 * (pyom.v0[:,1:,:] * pyom.E_iw[:,1:,:,pyom.tau] - pyom.v0[:,:-1,:] * pyom.E_iw[:,:-1,:,pyom.tau]) \
                                 / pyom.dyu[None, :-1, None] * pyom.maskV[:,:-1,:] * pyom.cosu[None, :-1, None]
        pyom.flux_north[:,-1,:] = 0.
        pyom.E_iw[2:-2, 2:-2, :, pyom.taup1] += pyom.dt_tracer * pyom.maskW[2:-2,2:-2,:] \
                                * ((pyom.flux_east[2:-2, 2:-2, :] - pyom.flux_east[1:-3, 2:-2, :]) \
                                    / (pyom.cost[None, 2:-2, None] * pyom.dxt[2:-2, None, None]) \
                                    + (pyom.flux_north[2:-2, 2:-2, :] - pyom.flux_north[2:-2, 1:-3, :]) \
                                    / (pyom.cost[None, 2:-2, None] * pyom.dyt[None, 2:-2, None]))

    """
    add tendency due to advection
    """
    if pyom.enable_idemix_superbee_advection:
        advection.adv_flux_superbee_wgrid(pyom.flux_east,pyom.flux_north,pyom.flux_top,pyom.E_iw[:,:,:,pyom.tau],pyom)

    if pyom.enable_idemix_upwind_advection:
        advection.adv_flux_upwind_wgrid(pyom.flux_east,pyom.flux_north,pyom.flux_top,pyom.E_iw[:,:,:,pyom.tau],pyom)

    if pyom.enable_idemix_superbee_advection or pyom.enable_idemix_upwind_advection:
        pyom.dE_iw[2:-2, 2:-2, :, pyom.tau] = pyom.maskW[2:-2, 2:-2, :] * (-(pyom.flux_east[2:-2, 2:-2, :] - pyom.flux_east[1:-3, 2:-2, :]) \
                                                                                / (pyom.cost[None, 2:-2, None] * pyom.dxt[2:-2, None, None]) \
                                                                           - (pyom.flux_north[2:-2, 2:-2, :] - pyom.flux_north[2:-2, 1:-3, :]) \
                                                                                / (pyom.cost[None, 2:-2, None] * pyom.dyt[None, 2:-2, None]))
        pyom.dE_iw[:,:,0,pyom.tau] += -pyom.flux_top[:,:,0] / pyom.dzw[0:1]
        pyom.dE_iw[:,:,1:-1,pyom.tau] += -(pyom.flux_top[:,:,1:-1] - pyom.flux_top[:,:,:-2]) / pyom.dzw[None, None, 1:-1]
        pyom.dE_iw[:,:,-1,pyom.tau] += -(pyom.flux_top[:,:,-1] - pyom.flux_top[:,:,-2]) / (0.5 * pyom.dzw[-1:])

        """
        Adam Bashforth time stepping
        """
        pyom.E_iw[:,:,:,pyom.taup1] += pyom.dt_tracer * ((1.5 + pyom.AB_eps) * pyom.dE_iw[:,:,:,pyom.tau] \
                                                       - (0.5 + pyom.AB_eps) * pyom.dE_iw[:,:,:,pyom.taum1])


def gofx2(x,pyom):
    """
    a function g(x)
    """
    x[x < 3.] = 3. # NOTE: probably a mistake in the fortran code, should just set x locally
    c = 1.-(2./pyom.pi) * np.arcsin(1./x)
    return 2. / pyom.pi / c * 0.9 * x**(-2./3.) * (1 - np.exp(-x/4.3))


def hofx1(x,pyom):
    """
    a function h(x)
    """
    return (2. / pyom.pi) / (1. - (2. / pyom.pi) * np.arcsin(1./x)) * (x-1.) / (x+1.)

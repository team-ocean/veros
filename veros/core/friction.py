import math

from .. import veros_method
from . import numerics, utilities, cyclic

@veros_method
def explicit_vert_friction(veros):
    """
    explicit vertical friction
    dissipation is calculated and added to K_diss_v
    """
    diss = np.zeros((veros.nx+4, veros.ny+4, veros.nz))

    """
    vertical friction of zonal momentum
    """
    fxa = 0.5 * (veros.kappaM[1:-2, 1:-2, :-1] + veros.kappaM[2:-1, 1:-2, :-1])
    veros.flux_top[1:-2, 1:-2, :-1] = fxa * (veros.u[1:-2, 1:-2, 1:, veros.tau] - veros.u[1:-2, 1:-2, :-1, veros.tau]) \
                                     / veros.dzw[np.newaxis, np.newaxis, :-1] * veros.maskU[1:-2, 1:-2, 1:] * veros.maskU[1:-2, 1:-2, :-1]
    veros.flux_top[:,:,-1] = 0.0
    veros.du_mix[:,:,0] = veros.flux_top[:,:,0] / veros.dzt[0] * veros.maskU[:,:,0]
    veros.du_mix[:,:,1:] = (veros.flux_top[:,:,1:] - veros.flux_top[:,:,:-1]) / veros.dzt[1:] * veros.maskU[:,:,1:]

    """
    diagnose dissipation by vertical friction of zonal momentum
    """
    diss[1:-2, 1:-2, :-1] = (veros.u[1:-2, 1:-2, 1:, veros.tau] - veros.u[1:-2, 1:-2, :-1, veros.tau]) \
                            * veros.flux_top[1:-2, 1:-2, :-1] / veros.dzw[np.newaxis, np.newaxis, :-1]
    diss[:,:,veros.nz-1] = 0.0
    diss[...] = numerics.ugrid_to_tgrid(veros,diss)
    veros.K_diss_v += diss

    """
    vertical friction of meridional momentum
    """
    fxa = 0.5 * (veros.kappaM[1:-2, 1:-2, :-1] + veros.kappaM[1:-2, 2:-1, :-1])
    veros.flux_top[1:-2, 1:-2, :-1] = fxa * (veros.v[1:-2, 1:-2, 1:, veros.tau] - veros.v[1:-2, 1:-2, :-1, veros.tau]) \
                                     / veros.dzw[np.newaxis, np.newaxis, :-1] * veros.maskV[1:-2, 1:-2, 1:] \
                                     * veros.maskV[1:-2, 1:-2, :-1]
    veros.flux_top[:,:,-1] = 0.0
    veros.dv_mix[:,:,1:] = (veros.flux_top[:,:,1:] - veros.flux_top[:,:,:-1]) \
                          / veros.dzt[np.newaxis, np.newaxis, 1:] * veros.maskV[:, :, 1:]
    veros.dv_mix[:,:,0] = veros.flux_top[:,:,0] / veros.dzt[0] * veros.maskV[:,:,0]

    """
    diagnose dissipation by vertical friction of meridional momentum
    """
    diss[1:-2, 1:-2, :-1] = (veros.v[1:-2, 1:-2, 1:, veros.tau] - veros.v[1:-2, 1:-2, :-1, veros.tau]) \
                 * veros.flux_top[1:-2, 1:-2, :-1] / veros.dzw[np.newaxis, np.newaxis, :-1]
    diss[:,:,-1] = 0.0
    diss[...] = numerics.vgrid_to_tgrid(veros,diss)
    veros.K_diss_v += diss

    if not veros.enable_hydrostatic:
        """
        vertical friction of vertical momentum
        """
        fxa = 0.5 * (veros.kappaM[1:-2, 1:-2, :-1] + veros.kappaM[1:-2, 1:-2, 1:])
        veros.flux_top[1:-2, 1:-2, :-1] = fxa * (veros.w[1:-2, 1:-2, 1:, veros.tau] \
                                                - veros.w[1:-2, 1:-2, :-1, veros.tau]) \
                                         / veros.dzw[np.newaxis, np.newaxis, 1:] \
                                         * veros.maskW[1:-2, 1:-2, 1:] * veros.maskW[1:-2, 1:-2, :-1]
        veros.flux_top[:,:,-1] = 0.0
        veros.dw_mix[:,:,1:] = (veros.flux_top[:,:,1:] - veros.flux_top[:,:,:-1]) \
                              / veros.dzw[np.newaxis, np.newaxis, 1:] * veros.maskW[:,:,1:]
        veros.dw_mix[:,:,0] = veros.flux_top[:,:,0] / veros.dzw[0] * veros.maskW[:,:,0]

        """
        diagnose dissipation by vertical friction of vertical momentum
        """
        # to be implemented

@veros_method
def implicit_vert_friction(veros):
    """
    vertical friction
    dissipation is calculated and added to K_diss_v
    """
    a_tri = np.zeros((veros.nx+1, veros.ny+1, veros.nz))
    b_tri = np.zeros((veros.nx+1, veros.ny+1, veros.nz))
    c_tri = np.zeros((veros.nx+1, veros.ny+1, veros.nz))
    d_tri = np.zeros((veros.nx+1, veros.ny+1, veros.nz))
    delta = np.zeros((veros.nx+1, veros.ny+1, veros.nz))
    diss = np.zeros((veros.nx+4, veros.ny+4, veros.nz))

    """
    implicit vertical friction of zonal momentum
    """
    kss = np.maximum(veros.kbot[1:-2, 1:-2], veros.kbot[2:-1, 1:-2]) - 1
    fxa = 0.5 * (veros.kappaM[1:-2, 1:-2, :-1] + veros.kappaM[2:-1, 1:-2, :-1])
    delta[:,:,:-1] = veros.dt_mom / veros.dzw[:-1] * fxa * veros.maskU[1:-2,1:-2,1:] * veros.maskU[1:-2,1:-2,:-1]
    a_tri[:,:, 1:] = -delta[:,:,:-1] / veros.dzt[np.newaxis,np.newaxis,1:]
    b_tri[:,:, 1:] = 1 + delta[:,:,:-1] / veros.dzt[np.newaxis,np.newaxis,1:]
    b_tri[:,:, 1:-1] += delta[:,:,1:-1] / veros.dzt[np.newaxis,np.newaxis,1:-1]
    b_tri_edge = 1 + delta / veros.dzt[np.newaxis,np.newaxis,:]
    c_tri[...] = -delta / veros.dzt[np.newaxis,np.newaxis,:]
    d_tri[...] = veros.u[1:-2,1:-2,:,veros.tau]
    res, mask = utilities.solve_implicit(veros, kss, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)
    veros.u[1:-2,1:-2,:,veros.taup1] = np.where(mask, res, veros.u[1:-2,1:-2,:,veros.taup1])

    veros.du_mix[1:-2, 1:-2] = (veros.u[1:-2,1:-2,:,veros.taup1] - veros.u[1:-2,1:-2,:,veros.tau]) / veros.dt_mom

    """
    diagnose dissipation by vertical friction of zonal momentum
    """
    fxa = 0.5 * (veros.kappaM[1:-2, 1:-2, :-1] + veros.kappaM[2:-1, 1:-2, :-1])
    veros.flux_top[1:-2, 1:-2, :-1] = fxa * (veros.u[1:-2, 1:-2, 1:, veros.taup1] - veros.u[1:-2, 1:-2, :-1, veros.taup1]) \
                                    / veros.dzw[:-1] * veros.maskU[1:-2, 1:-2, 1:] * veros.maskU[1:-2, 1:-2, :-1]
    diss[1:-2, 1:-2, :-1] = (veros.u[1:-2, 1:-2, 1:, veros.tau] - veros.u[1:-2, 1:-2, :-1, veros.tau]) \
                            * veros.flux_top[1:-2, 1:-2, :-1] / veros.dzw[:-1]
    diss[:,:,-1] = 0.0
    diss[...] = numerics.ugrid_to_tgrid(veros,diss)
    veros.K_diss_v += diss

    """
    implicit vertical friction of meridional momentum
    """
    kss = np.maximum(veros.kbot[1:-2, 1:-2], veros.kbot[1:-2, 2:-1]) - 1
    fxa = 0.5 * (veros.kappaM[1:-2, 1:-2, :-1] + veros.kappaM[1:-2, 2:-1, :-1])
    delta[:,:,:-1] = veros.dt_mom / veros.dzw[np.newaxis,np.newaxis,:-1] * fxa * veros.maskV[1:-2,1:-2,1:] * veros.maskV[1:-2,1:-2,:-1]
    a_tri[:,:,1:] = -delta[:,:,:-1] / veros.dzt[np.newaxis,np.newaxis,1:]
    b_tri[:,:,1:] = 1 + delta[:,:,:-1] / veros.dzt[np.newaxis,np.newaxis,1:]
    b_tri[:,:,1:-1] += delta[:,:,1:-1] / veros.dzt[np.newaxis,np.newaxis,1:-1]
    b_tri_edge = 1 + delta / veros.dzt[np.newaxis,np.newaxis,:]
    c_tri[:,:,:-1] = -delta[:,:,:-1] / veros.dzt[np.newaxis,np.newaxis,:-1]
    c_tri[:,:,-1] = 0.
    d_tri[...] = veros.v[1:-2,1:-2,:,veros.tau]
    res, mask = utilities.solve_implicit(veros, kss, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)
    veros.v[1:-2,1:-2,:,veros.taup1] = np.where(mask, res, veros.v[1:-2,1:-2,:,veros.taup1])
    veros.dv_mix[1:-2, 1:-2] = (veros.v[1:-2, 1:-2, :, veros.taup1] - veros.v[1:-2, 1:-2, :, veros.tau]) / veros.dt_mom

    """
    diagnose dissipation by vertical friction of meridional momentum
    """
    fxa = 0.5*(veros.kappaM[1:-2, 1:-2, :-1] + veros.kappaM[1:-2, 2:-1, :-1])
    veros.flux_top[1:-2, 1:-2, :-1] = fxa * (veros.v[1:-2, 1:-2, 1:, veros.taup1] - veros.v[1:-2, 1:-2, :-1, veros.taup1]) \
            / veros.dzw[:-1] * veros.maskV[1:-2, 1:-2, 1:] * veros.maskV[1:-2, 1:-2, :-1]
    diss[1:-2, 1:-2, :-1] = (veros.v[1:-2, 1:-2, 1:, veros.tau] - veros.v[1:-2, 1:-2, :-1, veros.tau]) * veros.flux_top[1:-2, 1:-2, :-1] / veros.dzw[:-1]
    diss[:,:,-1] = 0.0
    diss = numerics.vgrid_to_tgrid(veros,diss)
    veros.K_diss_v += diss

    if not veros.enable_hydrostatic:
        kss = veros.kbot[2:-2, 2:-2] - 1
        delta[:-1,:-1,:-1] = veros.dt_mom / veros.dzt[np.newaxis,np.newaxis,:-1] * 0.5 * (veros.kappaM[2:-2,2:-2,:-1] + veros.kappaM[2:-2,2:-2,1:])
        delta[:-1,:-1,-1] = 0.
        a_tri[:-1,:-1,1:-1] = -delta[:-1,:-1,:-2] / veros.dzw[np.newaxis,np.newaxis,1:-1]
        a_tri[:-1,:-1,-1] = 0.
        b_tri_edge = 1 + delta[:-1,:-1] / veros.dzw[np.newaxis,np.newaxis,:]
        b_tri[:-1,:-1,1:] = 1 + delta[:-1,:-1,:-1] / veros.dzw[np.newaxis,np.newaxis,:-1]
        b_tri[:-1,:-1,1:-1] += delta[:-1,:-1,1:-1] / veros.dzw[np.newaxis,np.newaxis,1:-1]
        c_tri[:-1,:-1,:-1] = - delta[:-1,:-1,:-1] / veros.dzw[np.newaxis,np.newaxis,:-1]
        c_tri[:-1,:-1,-1] = 0.
        d_tri[:-1,:-1] = veros.w[2:-2,2:-2,:,veros.tau]
        res, mask = utilities.solve_implicit(veros, kss, a_tri[:-1,:-1], b_tri[:-1,:-1], c_tri[:-1,:-1], d_tri[:-1,:-1], b_edge=b_tri_edge)
        veros.w[2:-2,2:-2,:,veros.taup1] = np.where(mask, res, veros.w[2:-2,2:-2,:,veros.taup1])
        veros.dw_mix[2:-2, 2:-2] = (veros.w[2:-2,2:-2,:,veros.taup1] - veros.w[2:-2,2:-2,:,veros.tau]) / veros.dt_mom

        """
        diagnose dissipation by vertical friction of vertical momentum
        """
        fxa = 0.5 * (veros.kappaM[1:-2, 1:-2, :-1] + veros.kappaM[1:-2,1:-2,1:])
        veros.flux_top[1:-2,1:-2,:-1] = fxa * (veros.w[1:-2,1:-2,1:,veros.taup1] - veros.w[1:-2,1:-2,:-1,veros.taup1]) \
                / veros.dzt[1:] * veros.maskW[1:-2, 1:-2, 1:] * veros.maskW[1:-2, 1:-2, :-1]
        diss[1:-2, 1:-2, :-1] = (veros.w[1:-2,1:-2,1:,veros.tau] - veros.w[1:-2,1:-2,:-1,veros.tau]) * veros.flux_top[1:-2,1:-2,:-1] / veros.dzt[1:]
        diss[:,:,-1] = 0.0
        veros.K_diss_v += diss

@veros_method
def rayleigh_friction(veros):
    """
    interior Rayleigh friction
    dissipation is calculated and added to K_diss_bot
    """
    veros.du_mix[...] += -veros.maskU * veros.r_ray * veros.u[..., veros.tau]
    if veros.enable_conserve_energy:
        diss = veros.maskU * veros.r_ray * veros.u[...,veros.tau]**2
        veros.K_diss_bot[...] = numerics.calc_diss(veros,diss,veros.K_diss_bot,'U')
    veros.dv_mix[...] += -veros.maskV * veros.r_ray * veros.v[...,veros.tau]
    if veros.enable_conserve_energy:
        diss = veros.maskV * veros.r_ray * veros.v[...,veros.tau]**2
        veros.K_diss_bot[...] = numerics.calc_diss(veros,diss,veros.K_diss_bot,'V')
    if not veros.enable_hydrostatic:
        raise NotImplementedError("Rayleigh friction for vertical velocity not implemented")

@veros_method
def linear_bottom_friction(veros):
    """
    linear bottom friction
    dissipation is calculated and added to K_diss_bot
    """
    if veros.enable_bottom_friction_var:
        """
        with spatially varying coefficient
        """
        k = np.maximum(veros.kbot[1:-2,2:-2], veros.kbot[2:-1,2:-2]) - 1
        mask = np.arange(veros.nz) == k[:,:,np.newaxis]
        veros.du_mix[1:-2,2:-2] += -(veros.maskU[1:-2,2:-2] * veros.r_bot_var_u[1:-2,2:-2,np.newaxis]) * veros.u[1:-2,2:-2,:,veros.tau] * mask
        if veros.enable_conserve_energy:
            diss = np.zeros((veros.nx+4, veros.ny+4, veros.nz))
            diss[1:-2,2:-2] = veros.maskU[1:-2,2:-2] * veros.r_bot_var_u[1:-2,2:-2,np.newaxis] * veros.u[1:-2,2:-2,:,veros.tau]**2 * mask
            veros.K_diss_bot[...] = numerics.calc_diss(veros,diss,veros.K_diss_bot,'U')

        k = np.maximum(veros.kbot[2:-2, 2:-1], veros.kbot[2:-2, 1:-2]) - 1
        mask = np.arange(veros.nz) == k[:,:,np.newaxis]
        veros.dv_mix[2:-2,1:-2] += -(veros.maskV[2:-2,1:-2] * veros.r_bot_var_v[2:-2,1:-2,np.newaxis]) * veros.v[2:-2,1:-2,:,veros.tau] * mask
        if veros.enable_conserve_energy:
            diss = np.zeros((veros.nx+4, veros.ny+4, veros.nz))
            diss[2:-2,1:-2] = veros.maskV[2:-2,1:-2] * veros.r_bot_var_v[2:-2,1:-2,np.newaxis] * veros.v[2:-2,1:-2,:,veros.tau]**2 * mask
            veros.K_diss_bot[...] = numerics.calc_diss(veros,diss,veros.K_diss_bot,'V')
    else:
        """
        with constant coefficient
        """
        k = np.maximum(veros.kbot[1:-2,2:-2], veros.kbot[2:-1,2:-2]) - 1
        mask = np.arange(veros.nz) == k[:,:,np.newaxis]
        veros.du_mix[1:-2,2:-2] += -veros.maskU[1:-2,2:-2] * veros.r_bot * veros.u[1:-2,2:-2,:,veros.tau] * mask
        if veros.enable_conserve_energy:
            diss = np.zeros((veros.nx+4, veros.ny+4, veros.nz))
            diss[1:-2,2:-2] = veros.maskU[1:-2,2:-2] * veros.r_bot * veros.u[1:-2,2:-2,:,veros.tau]**2 * mask
            veros.K_diss_bot[...] = numerics.calc_diss(veros,diss,veros.K_diss_bot,'U')

        k = np.maximum(veros.kbot[2:-2, 2:-1], veros.kbot[2:-2, 1:-2]) - 1
        mask = np.arange(veros.nz) == k[:,:,np.newaxis]
        veros.dv_mix[2:-2,1:-2] += -veros.maskV[2:-2,1:-2] * veros.r_bot * veros.v[2:-2,1:-2,:,veros.tau] * mask
        if veros.enable_conserve_energy:
            diss = np.zeros((veros.nx+4, veros.ny+4, veros.nz))
            diss[2:-2,1:-2] = veros.maskV[2:-2,1:-2] * veros.r_bot * veros.v[2:-2,1:-2,:,veros.tau]**2 * mask
            veros.K_diss_bot[...] = numerics.calc_diss(veros,diss,veros.K_diss_bot,'V')

    if not veros.enable_hydrostatic:
        raise NotImplementedError("bottom friction for vertical velocity not implemented")

@veros_method
def quadratic_bottom_friction(veros):
    """
    quadratic bottom friction
    dissipation is calculated and added to K_diss_bot
    """
    # we might want to account for EKE in the drag, also a tidal residual
    k = np.maximum(veros.kbot[1:-2,2:-2], veros.kbot[2:-1,2:-2]) - 1
    mask = k[..., np.newaxis] == np.arange(veros.nz)[np.newaxis, np.newaxis, :]
    fxa = veros.maskV[1:-2,2:-2,:] * veros.v[1:-2,2:-2,:,veros.tau]**2 + veros.maskV[1:-2,1:-3,:] * veros.v[1:-2,1:-3,:,veros.tau]**2 \
        + veros.maskV[2:-1,2:-2,:] * veros.v[2:-1,2:-2,:,veros.tau]**2 + veros.maskV[2:-1,1:-3,:] * veros.v[2:-1,1:-3,:,veros.tau]**2
    fxa = np.sqrt(veros.u[1:-2,2:-2,:,veros.tau]**2 + 0.25 * fxa)
    aloc = veros.maskU[1:-2,2:-2,:] * veros.r_quad_bot * veros.u[1:-2,2:-2,:,veros.tau] \
                             * fxa / veros.dzt[np.newaxis, np.newaxis, :] * mask
    veros.du_mix[1:-2,2:-2,:] += -aloc

    if veros.enable_conserve_energy:
        diss = np.zeros((veros.nx+4, veros.ny+4, veros.nz))
        diss[1:-2,2:-2,:] = aloc * veros.u[1:-2,2:-2,:,veros.tau]
        veros.K_diss_bot[...] = numerics.calc_diss(veros,diss,veros.K_diss_bot,'U')

    k = np.maximum(veros.kbot[2:-2,1:-2], veros.kbot[2:-2,2:-1]) - 1
    mask = k[..., np.newaxis] == np.arange(veros.nz)[np.newaxis, np.newaxis, :]
    fxa = veros.maskU[2:-2,1:-2,:] * veros.u[2:-2,1:-2,:,veros.tau]**2 + veros.maskU[1:-3,1:-2,:] * veros.u[1:-3,1:-2,:,veros.tau]**2 \
        + veros.maskU[2:-2,2:-1,:] * veros.u[2:-2,2:-1,:,veros.tau]**2 + veros.maskU[1:-3,2:-1,:] * veros.u[1:-3,2:-1,:,veros.tau]**2
    fxa = np.sqrt(veros.v[2:-2,1:-2,:,veros.tau]**2 + 0.25 * fxa)
    aloc = veros.maskV[2:-2,1:-2,:] * veros.r_quad_bot * veros.v[2:-2,1:-2,:,veros.tau] \
                             * fxa / veros.dzt[np.newaxis, np.newaxis, :] * mask
    veros.dv_mix[2:-2,1:-2,:] += -aloc

    if veros.enable_conserve_energy:
        diss = np.zeros((veros.nx+4, veros.ny+4, veros.nz))
        diss[2:-2,1:-2,:] = aloc * veros.v[2:-2,1:-2,:,veros.tau]
        veros.K_diss_bot[...] = numerics.calc_diss(veros,diss,veros.K_diss_bot,'V')

    if not veros.enable_hydrostatic:
        raise NotImplementedError("bottom friction for vertical velocity not implemented")

@veros_method
def harmonic_friction(veros):
    """
    horizontal harmonic friction
    dissipation is calculated and added to K_diss_h
    """
    diss = np.zeros((veros.nx+4, veros.ny+4, veros.nz))

    """
    Zonal velocity
    """
    if veros.enable_hor_friction_cos_scaling:
        fxa = veros.cost**veros.hor_friction_cosPower
        veros.flux_east[:-1] = veros.A_h * fxa[np.newaxis,:,np.newaxis] * (veros.u[1:,:,:,veros.tau] - veros.u[:-1,:,:,veros.tau]) \
                / (veros.cost * veros.dxt[1:, np.newaxis])[:,:,np.newaxis] * veros.maskU[1:] * veros.maskU[:-1]
        fxa = veros.cosu**veros.hor_friction_cosPower
        veros.flux_north[:,:-1] = veros.A_h * fxa[np.newaxis,:-1,np.newaxis] * (veros.u[:,1:,:,veros.tau] - veros.u[:,:-1,:,veros.tau]) \
                / veros.dyu[np.newaxis,:-1,np.newaxis] * veros.maskU[:,1:] * veros.maskU[:,:-1] * veros.cosu[np.newaxis,:-1,np.newaxis]
    else:
        veros.flux_east[:-1,:,:] = veros.A_h * (veros.u[1:,:,:,veros.tau] - veros[:-1,:,:,veros.tau]) \
                / (veros.cost * veros.dxt[1:, np.newaxis])[:,:,np.newaxis] * veros.maskU[1:] * veros.maskU[:-1]
        veros.flux_north[:,:-1,:] = veros.A_h * (veros.u[:,1:,:,veros.tau] - veros.u[:,:-1,:,veros.tau]) \
                / veros.dyu[np.newaxis, :-1, np.newaxis] * veros.maskU[:,1:] * veros.maskU[:,:-1] * veros.cosu[np.newaxis,:-1,np.newaxis]
    veros.flux_east[-1,:,:] = 0.
    veros.flux_north[:,-1,:] = 0.

    """
    update tendency
    """
    veros.du_mix[2:-2, 2:-2, :] += veros.maskU[2:-2,2:-2] * ((veros.flux_east[2:-2,2:-2] - veros.flux_east[1:-3,2:-2]) \
                                                            / (veros.cost[2:-2] * veros.dxu[2:-2, np.newaxis])[:,:,np.newaxis] \
                                                        + (veros.flux_north[2:-2,2:-2] - veros.flux_north[2:-2,1:-3]) \
                                                            / (veros.cost[2:-2] * veros.dyt[2:-2])[np.newaxis, :, np.newaxis])

    if veros.enable_conserve_energy:
        """
        diagnose dissipation by lateral friction
        """
        diss[1:-2, 2:-2] = 0.5*((veros.u[2:-1,2:-2,:,veros.tau] - veros.u[1:-2,2:-2,:,veros.tau]) * veros.flux_east[1:-2,2:-2] \
                + (veros.u[1:-2,2:-2,:,veros.tau] - veros.u[:-3,2:-2,:,veros.tau]) * veros.flux_east[:-3,2:-2]) \
                    / (veros.cost[2:-2] * veros.dxu[1:-2,np.newaxis])[:,:,np.newaxis]\
                + 0.5*((veros.u[1:-2,3:-1,:,veros.tau] - veros.u[1:-2,2:-2,:,veros.tau]) * veros.flux_north[1:-2,2:-2] \
                + (veros.u[1:-2,2:-2,:,veros.tau] - veros.u[1:-2,1:-3,:,veros.tau]) * veros.flux_north[1:-2,1:-3]) \
                    / (veros.cost[2:-2] * veros.dyt[2:-2])[np.newaxis,:,np.newaxis]
        veros.K_diss_h[...] = 0.
        veros.K_diss_h[...] = numerics.calc_diss(veros,diss,veros.K_diss_h,'U')

    """
    Meridional velocity
    """
    if veros.enable_hor_friction_cos_scaling:
        fxa = (veros.cosu ** veros.hor_friction_cosPower) * np.ones(veros.nx+3)[:,np.newaxis]
        veros.flux_east[:-1] = veros.A_h * fxa[:, :, np.newaxis] * (veros.v[1:,:,:,veros.tau] - veros.v[:-1,:,:,veros.tau]) \
                / (veros.cosu * veros.dxu[:-1, np.newaxis])[:,:,np.newaxis] * veros.maskV[1:] * veros.maskV[:-1]
        fxa = (veros.cost[1:] ** veros.hor_friction_cosPower) * np.ones(veros.nx+4)[:, np.newaxis]
        veros.flux_north[:,:-1] = veros.A_h * fxa[:,:,np.newaxis] * (veros.v[:,1:,:,veros.tau] - veros.v[:,:-1,:,veros.tau]) \
                / veros.dyt[np.newaxis,1:,np.newaxis] * veros.cost[np.newaxis,1:,np.newaxis] * veros.maskV[:,:-1] * veros.maskV[:,1:]
    else:
        veros.flux_east[:-1] = veros.A_h * (veros.v[1:,:,:,veros.tau] - veros.v[:-1,:,:,veros.tau]) \
                / (veros.cosu * veros.dxu[:-1, np.newaxis])[:,:,np.newaxis] * veros.maskV[1:] * veros.maskV[:-1]
        veros.flux_north[:,:-1] = veros.A_h * (veros.v[:,1:,:,veros.tau] - veros.v[:,:-1,:,veros.tau]) \
                / veros.dyt[np.newaxis,1:,np.newaxis] * veros.cost[np.newaxis,1:,np.newaxis] * veros.maskV[:,:-1] * veros.maskV[:,1:]
    veros.flux_east[-1,:,:] = 0.
    veros.flux_north[:,-1,:] = 0.

    """
    update tendency
    """
    veros.dv_mix[2:-2,2:-2] += veros.maskV[2:-2,2:-2] * ((veros.flux_east[2:-2,2:-2] - veros.flux_east[1:-3,2:-2]) \
                                / (veros.cosu[2:-2] * veros.dxt[2:-2,np.newaxis])[:,:,np.newaxis] \
                            + (veros.flux_north[2:-2,2:-2] - veros.flux_north[2:-2,1:-3]) \
                                / (veros.dyu[2:-2] * veros.cosu[2:-2])[np.newaxis,:,np.newaxis])

    if veros.enable_conserve_energy:
        """
        diagnose dissipation by lateral friction
        """
        diss[2:-2,1:-2] = 0.5 * ((veros.v[3:-1,1:-2,:,veros.tau] - veros.v[2:-2,1:-2,:,veros.tau]) * veros.flux_east[2:-2,1:-2]\
                + (veros.v[2:-2,1:-2,:,veros.tau] - veros.v[1:-3,1:-2,:,veros.tau]) * veros.flux_east[1:-3,1:-2]) \
                / (veros.cosu[1:-2] * veros.dxt[2:-2,np.newaxis])[:,:,np.newaxis] \
                + 0.5*((veros.v[2:-2,2:-1,:,veros.tau] - veros.v[2:-2,1:-2,:,veros.tau]) * veros.flux_north[2:-2,1:-2] \
                + (veros.v[2:-2,1:-2,:,veros.tau] - veros.v[2:-2,:-3,:,veros.tau]) * veros.flux_north[2:-2,:-3]) \
                / (veros.cosu[1:-2] * veros.dyu[1:-2])[np.newaxis,:,np.newaxis]
        veros.K_diss_h[...] = numerics.calc_diss(veros,diss,veros.K_diss_h,'V')

    if not veros.enable_hydrostatic:
        if veros.enable_hor_friction_cos_scaling:
            raise NotImplementedError("scaling of lateral friction for vertical velocity not implemented")

        veros.flux_east[:-1] = veros.A_h * (veros.w[1:,:,:,veros.tau] - veros.w[:-1,:,:,veros.tau]) \
                / (veros.cost * veros.dxu[:,np.newaxis])[:,:,np.newaxis] * veros.maskW[1:] * veros.maskW[:-1]
        veros.flux_north[:,:-1] = veros.A_h * (veros.w[:,1:,:,veros.tau] - veros.w[:,:-1,:,veros.tau]) \
                / veros.dyu[np.newaxis,:-1,np.newaxis] * veros.maskW[:,1:] * veros.maskW[:,:-1] * veros.cosu[np.newaxis,:-1,np.newaxis]
        veros.flux_east[-1,:,:] = 0.
        veros.flux_north[:,-1,:] = 0.

        """
        update tendency
        """
        veros.dw_mix[2:-2,2:-2] += veros.maskW[2:-2,2:-2]*((veros.flux_east[2:-2,2:-2] - veros.flux_east[1:-3,2:-2]) \
                / (veros.cost[2:-2] * veros.dxt[2:-2,np.newaxis])[:,:,np.newaxis] \
                + (veros.flux_north[2:-2,2:-2] - veros.flux_north[2:-2,1:-3]) \
                / (veros.dyt[2:-2] * veros.cost[2:-2])[np.newaxis,:,np.newaxis])

        """
        diagnose dissipation by lateral friction
        """
        # to be implemented

@veros_method
def biharmonic_friction(veros):
    """
    horizontal biharmonic friction
    dissipation is calculated and added to K_diss_h
    """
    if not veros.enable_hydrostatic:
        raise NotImplementedError("biharmonic mixing for non-hydrostatic case not yet implemented")

    fxa = math.sqrt(abs(veros.A_hbi))

    """
    Zonal velocity
    """
    veros.flux_east[:-1,:,:] = fxa * (veros.u[1:,:,:,veros.tau] - veros.u[:-1,:,:,veros.tau]) \
                            / (veros.cost[np.newaxis, :, np.newaxis] * veros.dxt[1:, np.newaxis, np.newaxis]) \
                            * veros.maskU[1:,:,:] * veros.maskU[:-1,:,:]
    veros.flux_north[:,:-1,:] = fxa * (veros.u[:,1:,:,veros.tau] - veros.u[:,:-1,:,veros.tau]) \
                             / veros.dyu[np.newaxis, :-1, np.newaxis] * veros.maskU[:,1:,:] \
                             * veros.maskU[:,:-1,:] * veros.cosu[np.newaxis, :-1, np.newaxis]
    veros.flux_east[-1,:,:] = 0.
    veros.flux_north[:,-1,:] = 0.

    del2 = np.zeros((veros.nx+4, veros.ny+4, veros.nz))
    del2[1:,1:,:] = (veros.flux_east[1:,1:,:] - veros.flux_east[:-1,1:,:]) \
                        / (veros.cost[np.newaxis, 1:, np.newaxis] * veros.dxu[1:, np.newaxis, np.newaxis]) \
                  + (veros.flux_north[1:,1:,:] - veros.flux_north[1:,:-1,:]) \
                        / (veros.cost[np.newaxis, 1:, np.newaxis] * veros.dyt[np.newaxis, 1:, np.newaxis])

    veros.flux_east[:-1,:,:] = fxa * (del2[1:,:,:] - del2[:-1,:,:]) \
                            / (veros.cost[np.newaxis, :, np.newaxis] * veros.dxt[1:, np.newaxis, np.newaxis]) \
                            * veros.maskU[1:,:,:] * veros.maskU[:-1,:,:]
    veros.flux_north[:,:-1,:] = fxa * (del2[:,1:,:] - del2[:,:-1,:]) \
                             / veros.dyu[np.newaxis, :-1, np.newaxis] * veros.maskU[:,1:,:] \
                             * veros.maskU[:,:-1,:] * veros.cosu[np.newaxis, :-1, np.newaxis]
    veros.flux_east[-1,:,:] = 0.
    veros.flux_north[:,-1,:] = 0.

    """
    update tendency
    """
    veros.du_mix[2:-2,2:-2,:] += -veros.maskU[2:-2,2:-2,:] * ((veros.flux_east[2:-2,2:-2,:] - veros.flux_east[1:-3,2:-2,:]) \
                                    / (veros.cost[np.newaxis, 2:-2, np.newaxis] * veros.dxu[2:-2, np.newaxis, np.newaxis]) \
                                    + (veros.flux_north[2:-2,2:-2,:] - veros.flux_north[2:-2,1:-3,:]) \
                                    / (veros.cost[np.newaxis, 2:-2, np.newaxis] * veros.dyt[np.newaxis, 2:-2, np.newaxis]))
    if veros.enable_conserve_energy:
        """
        diagnose dissipation by lateral friction
        """
        if veros.enable_cyclic_x:
            cyclic.setcyclic_x(veros.flux_east)
            cyclic.setcyclic_x(veros.flux_north)
        diss = np.zeros((veros.nx+4, veros.ny+4, veros.nz))
        diss[1:-2, 2:-2, :] = -0.5 * ((veros.u[2:-1,2:-2,:,veros.tau] - veros.u[1:-2,2:-2,:,veros.tau]) * veros.flux_east[1:-2,2:-2,:] \
                                    + (veros.u[1:-2,2:-2,:,veros.tau] - veros.u[:-3,2:-2,:,veros.tau]) * veros.flux_east[:-3,2:-2,:]) \
                                    / (veros.cost[np.newaxis, 2:-2, np.newaxis] * veros.dxu[1:-2, np.newaxis, np.newaxis])  \
                              -0.5 * ((veros.u[1:-2,3:-1,:,veros.tau] - veros.u[1:-2,2:-2,:,veros.tau]) * veros.flux_north[1:-2,2:-2,:] \
                                    + (veros.u[1:-2,2:-2,:,veros.tau] - veros.u[1:-2,1:-3,:,veros.tau]) * veros.flux_north[1:-2,1:-3,:]) \
                                    / (veros.cost[np.newaxis, 2:-2, np.newaxis] * veros.dyt[np.newaxis, 2:-2, np.newaxis])
        veros.K_diss_h[...] = 0.
        veros.K_diss_h[...] = numerics.calc_diss(veros,diss,veros.K_diss_h,'U')

    """
    Meridional velocity
    """
    veros.flux_east[:-1, :, :] = fxa * (veros.v[1:,:,:,veros.tau] - veros.v[:-1,:,:,veros.tau]) \
                             / (veros.cosu[np.newaxis, :, np.newaxis] * veros.dxu[:-1, np.newaxis, np.newaxis]) \
                             * veros.maskV[1:,:,:] * veros.maskV[:-1,:,:]
    veros.flux_north[:,:-1,:] = fxa * (veros.v[:,1:,:,veros.tau] - veros.v[:,:-1,:,veros.tau]) \
                             / veros.dyt[np.newaxis, 1:, np.newaxis] * veros.cost[np.newaxis, 1:, np.newaxis] \
                             * veros.maskV[:,:-1,:] * veros.maskV[:,1:,:]
    veros.flux_east[-1,:,:] = 0.
    veros.flux_north[:,-1,:] = 0.

    del2[1:,1:,:] = (veros.flux_east[1:,1:,:] - veros.flux_east[:-1,1:,:]) \
                        / (veros.cosu[np.newaxis, 1:, np.newaxis] * veros.dxt[1:, np.newaxis, np.newaxis])  \
                  + (veros.flux_north[1:,1:,:] - veros.flux_north[1:,:-1,:]) \
                        / (veros.dyu[np.newaxis, 1:, np.newaxis] * veros.cosu[np.newaxis, 1:, np.newaxis])
    veros.flux_east[:-1,:,:] = fxa * (del2[1:,:,:] - del2[:-1,:,:]) \
                            / (veros.cosu[np.newaxis,:,np.newaxis] * veros.dxu[:-1,np.newaxis,np.newaxis]) \
                            * veros.maskV[1:,:,:] * veros.maskV[:-1,:,:]
    veros.flux_north[:,:-1,:] = fxa * (del2[:,1:,:] - del2[:,:-1,:]) \
                             / veros.dyt[np.newaxis,1:,np.newaxis] * veros.cost[np.newaxis, 1:, np.newaxis] \
                             * veros.maskV[:,:-1,:] * veros.maskV[:,1:,:]
    veros.flux_east[-1,:,:] = 0.
    veros.flux_north[:,-1,:] = 0.

    """
    update tendency
    """
    veros.dv_mix[2:-2, 2:-2, :] += -veros.maskV[2:-2,2:-2,:] * ((veros.flux_east[2:-2,2:-2,:] - veros.flux_east[1:-3,2:-2,:]) \
                                    / (veros.cosu[np.newaxis, 2:-2, np.newaxis] * veros.dxt[2:-2, np.newaxis, np.newaxis]) \
                                    + (veros.flux_north[2:-2,2:-2,:] - veros.flux_north[2:-2,1:-3,:]) \
                                    / (veros.dyu[np.newaxis, 2:-2, np.newaxis] * veros.cosu[np.newaxis, 2:-2, np.newaxis]))

    if veros.enable_conserve_energy:
        """
        diagnose dissipation by lateral friction
        """
        if veros.enable_cyclic_x:
            cyclic.setcyclic_x(veros.flux_east)
            cyclic.setcyclic_x(veros.flux_north)
        diss[2:-2, 1:-2, :] = -0.5*((veros.v[3:-1,1:-2,:,veros.tau] - veros.v[2:-2,1:-2,:,veros.tau]) * veros.flux_east[2:-2,1:-2,:] \
                                  + (veros.v[2:-2,1:-2,:,veros.tau] - veros.v[1:-3,1:-2,:,veros.tau]) * veros.flux_east[1:-3,1:-2,:]) \
                                  / (veros.cosu[np.newaxis, 1:-2, np.newaxis] * veros.dxt[2:-2, np.newaxis, np.newaxis]) \
                             - 0.5*((veros.v[2:-2,2:-1,:,veros.tau] - veros.v[2:-2,1:-2,:,veros.tau]) * veros.flux_north[2:-2,1:-2,:] \
                                  + (veros.v[2:-2,1:-2,:,veros.tau] - veros.v[2:-2,:-3,:,veros.tau]) * veros.flux_north[2:-2,:-3,:]) \
                                  / (veros.cosu[np.newaxis, 1:-2, np.newaxis] * veros.dyu[np.newaxis, 1:-2, np.newaxis])
        veros.K_diss_h[...] = numerics.calc_diss(veros,diss,veros.K_diss_h,'V')

@veros_method
def momentum_sources(veros):
    """
    other momentum sources
    dissipation is calculated and added to K_diss_bot
    """
    veros.du_mix[...] += veros.maskU * veros.u_source
    if veros.enable_conserve_energy:
        diss = -veros.maskU * veros.u[..., veros.tau] * veros.u_source
        veros.K_diss_bot[...] = numerics.calc_diss(veros,diss,veros.K_diss_bot,'U')
    veros.dv_mix[...] += veros.maskV * veros.v_source
    if veros.enable_conserve_energy:
        diss = -veros.maskV * veros.v[..., veros.tau] * veros.v_source
        veros.K_diss_bot[...] = numerics.calc_diss(veros,diss,veros.K_diss_bot,'V')

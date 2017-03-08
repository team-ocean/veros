import math

from climate.pyom import numerics, utilities, cyclic, pyom_method

@pyom_method
def explicit_vert_friction(pyom):
    """
    explicit vertical friction
    dissipation is calculated and added to K_diss_v
    """
    diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    """
    vertical friction of zonal momentum
    """
    fxa = 0.5 * (pyom.kappaM[1:-2, 1:-2, :-1] + pyom.kappaM[2:-1, 1:-2, :-1])
    pyom.flux_top[1:-2, 1:-2, :-1] = fxa * (pyom.u[1:-2, 1:-2, 1:, pyom.tau] - pyom.u[1:-2, 1:-2, :-1, pyom.tau]) \
                                     / pyom.dzw[None, None, :-1] * pyom.maskU[1:-2, 1:-2, 1:] * pyom.maskU[1:-2, 1:-2, :-1]
    pyom.flux_top[:,:,-1] = 0.0
    pyom.du_mix[:,:,0] = pyom.flux_top[:,:,0] / pyom.dzt[0] * pyom.maskU[:,:,0]
    pyom.du_mix[:,:,1:] = (pyom.flux_top[:,:,1:] - pyom.flux_top[:,:,:-1]) / pyom.dzt[1:] * pyom.maskU[:,:,1:]

    """
    diagnose dissipation by vertical friction of zonal momentum
    """
    diss[1:-2, 1:-2, :-1] = (pyom.u[1:-2, 1:-2, 1:, pyom.tau] - pyom.u[1:-2, 1:-2, :-1, pyom.tau]) \
                            * pyom.flux_top[1:-2, 1:-2, :-1] / pyom.dzw[np.newaxis, np.newaxis, :-1]
    diss[:,:,pyom.nz-1] = 0.0
    diss[...] = numerics.ugrid_to_tgrid(pyom,diss)
    pyom.K_diss_v += diss

    """
    vertical friction of meridional momentum
    """
    fxa = 0.5 * (pyom.kappaM[1:-2, 1:-2, :-1] + pyom.kappaM[1:-2, 2:-1, :-1])
    pyom.flux_top[1:-2, 1:-2, :-1] = fxa * (pyom.v[1:-2, 1:-2, 1:, pyom.tau] - pyom.v[1:-2, 1:-2, :-1, pyom.tau]) \
                                     / pyom.dzw[np.newaxis, np.newaxis, :-1] * pyom.maskV[1:-2, 1:-2, 1:] \
                                     * pyom.maskV[1:-2, 1:-2, :-1]
    pyom.flux_top[:,:,-1] = 0.0
    pyom.dv_mix[:,:,1:] = (pyom.flux_top[:,:,1:] - pyom.flux_top[:,:,:-1]) \
                          / pyom.dzt[np.newaxis, np.newaxis, 1:] * pyom.maskV[:, :, 1:]
    pyom.dv_mix[:,:,0] = pyom.flux_top[:,:,0] / pyom.dzt[0] * pyom.maskV[:,:,0]

    """
    diagnose dissipation by vertical friction of meridional momentum
    """
    diss[1:-2, 1:-2, :-1] = (pyom.v[1:-2, 1:-2, 1:, pyom.tau] - pyom.v[1:-2, 1:-2, :-1, pyom.tau]) \
                 * pyom.flux_top[1:-2, 1:-2, :-1] / pyom.dzw[np.newaxis, np.newaxis, :-1]
    diss[:,:,-1] = 0.0
    diss[...] = numerics.vgrid_to_tgrid(pyom,diss)
    pyom.K_diss_v += diss

    if not pyom.enable_hydrostatic:
        """
        vertical friction of vertical momentum
        """
        fxa = 0.5 * (pyom.kappaM[1:-2, 1:-2, :-1] + pyom.kappaM[1:-2, 1:-2, 1:])
        pyom.flux_top[1:-2, 1:-2, :-1] = fxa * (pyom.w[1:-2, 1:-2, 1:, pyom.tau] \
                                                - pyom.w[1:-2, 1:-2, :-1, pyom.tau]) \
                                         / pyom.dzw[np.newaxis, np.newaxis, 1:] \
                                         * pyom.maskW[1:-2, 1:-2, 1:] * pyom.maskW[1:-2, 1:-2, :-1]
        pyom.flux_top[:,:,-1] = 0.0
        pyom.dw_mix[:,:,1:] = (pyom.flux_top[:,:,1:] - pyom.flux_top[:,:,:-1]) \
                              / pyom.dzw[np.newaxis, np.newaxis, 1:] * pyom.maskW[:,:,1:]
        pyom.dw_mix[:,:,0] = pyom.flux_top[:,:,0] / pyom.dzw[0] * pyom.maskW[:,:,0]

        """
        diagnose dissipation by vertical friction of vertical momentum
        """
        # to be implemented

@pyom_method
def implicit_vert_friction(pyom):
    """
    vertical friction
    dissipation is calculated and added to K_diss_v
    """
    a_tri = np.zeros((pyom.nx+1, pyom.ny+1, pyom.nz))
    b_tri = np.zeros((pyom.nx+1, pyom.ny+1, pyom.nz))
    c_tri = np.zeros((pyom.nx+1, pyom.ny+1, pyom.nz))
    d_tri = np.zeros((pyom.nx+1, pyom.ny+1, pyom.nz))
    delta = np.zeros((pyom.nx+1, pyom.ny+1, pyom.nz))
    diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    """
    implicit vertical friction of zonal momentum
    """
    kss = np.maximum(pyom.kbot[1:-2, 1:-2], pyom.kbot[2:-1, 1:-2]) - 1
    fxa = 0.5 * (pyom.kappaM[1:-2, 1:-2, :-1] + pyom.kappaM[2:-1, 1:-2, :-1])
    delta[:,:,:-1] = pyom.dt_mom / pyom.dzw[:-1] * fxa * pyom.maskU[1:-2,1:-2,1:] * pyom.maskU[1:-2,1:-2,:-1]
    a_tri[:,:, 1:] = -delta[:,:,:-1] / pyom.dzt[None,None,1:]
    b_tri[:,:, 1:] = 1 + delta[:,:,:-1] / pyom.dzt[None,None,1:]
    b_tri[:,:, 1:-1] += delta[:,:,1:-1] / pyom.dzt[None,None,1:-1]
    b_tri_edge = 1 + delta / pyom.dzt[None,None,:]
    c_tri[...] = -delta / pyom.dzt[None,None,:]
    d_tri[...] = pyom.u[1:-2,1:-2,:,pyom.tau]
    res, mask = utilities.solve_implicit(pyom, kss, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)
    pyom.u[1:-2,1:-2,:,pyom.taup1] = np.where(mask, res, pyom.u[1:-2,1:-2,:,pyom.taup1])

    pyom.du_mix[1:-2, 1:-2] = (pyom.u[1:-2,1:-2,:,pyom.taup1] - pyom.u[1:-2,1:-2,:,pyom.tau]) / pyom.dt_mom

    """
    diagnose dissipation by vertical friction of zonal momentum
    """
    fxa = 0.5 * (pyom.kappaM[1:-2, 1:-2, :-1] + pyom.kappaM[2:-1, 1:-2, :-1])
    pyom.flux_top[1:-2, 1:-2, :-1] = fxa * (pyom.u[1:-2, 1:-2, 1:, pyom.taup1] - pyom.u[1:-2, 1:-2, :-1, pyom.taup1]) \
                                    / pyom.dzw[:-1] * pyom.maskU[1:-2, 1:-2, 1:] * pyom.maskU[1:-2, 1:-2, :-1]
    diss[1:-2, 1:-2, :-1] = (pyom.u[1:-2, 1:-2, 1:, pyom.tau] - pyom.u[1:-2, 1:-2, :-1, pyom.tau]) \
                            * pyom.flux_top[1:-2, 1:-2, :-1] / pyom.dzw[:-1]
    diss[:,:,-1] = 0.0
    diss[...] = numerics.ugrid_to_tgrid(pyom,diss)
    pyom.K_diss_v += diss

    """
    implicit vertical friction of meridional momentum
    """
    kss = np.maximum(pyom.kbot[1:-2, 1:-2], pyom.kbot[1:-2, 2:-1]) - 1
    fxa = 0.5 * (pyom.kappaM[1:-2, 1:-2, :-1] + pyom.kappaM[1:-2, 2:-1, :-1])
    delta[:,:,:-1] = pyom.dt_mom / pyom.dzw[None,None,:-1] * fxa * pyom.maskV[1:-2,1:-2,1:] * pyom.maskV[1:-2,1:-2,:-1]
    a_tri[:,:,1:] = -delta[:,:,:-1] / pyom.dzt[None,None,1:]
    b_tri[:,:,1:] = 1 + delta[:,:,:-1] / pyom.dzt[None,None,1:]
    b_tri[:,:,1:-1] += delta[:,:,1:-1] / pyom.dzt[None,None,1:-1]
    b_tri_edge = 1 + delta / pyom.dzt[None,None,:]
    c_tri[:,:,:-1] = -delta[:,:,:-1] / pyom.dzt[None,None,:-1]
    c_tri[:,:,-1] = 0.
    d_tri[...] = pyom.v[1:-2,1:-2,:,pyom.tau]
    res, mask = utilities.solve_implicit(pyom, kss, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)
    pyom.v[1:-2,1:-2,:,pyom.taup1] = np.where(mask, res, pyom.v[1:-2,1:-2,:,pyom.taup1])
    pyom.dv_mix[1:-2, 1:-2] = (pyom.v[1:-2, 1:-2, :, pyom.taup1] - pyom.v[1:-2, 1:-2, :, pyom.tau]) / pyom.dt_mom

    """
    diagnose dissipation by vertical friction of meridional momentum
    """
    fxa = 0.5*(pyom.kappaM[1:-2, 1:-2, :-1] + pyom.kappaM[1:-2, 2:-1, :-1])
    pyom.flux_top[1:-2, 1:-2, :-1] = fxa * (pyom.v[1:-2, 1:-2, 1:, pyom.taup1] - pyom.v[1:-2, 1:-2, :-1, pyom.taup1]) \
            / pyom.dzw[:-1] * pyom.maskV[1:-2, 1:-2, 1:] * pyom.maskV[1:-2, 1:-2, :-1]
    diss[1:-2, 1:-2, :-1] = (pyom.v[1:-2, 1:-2, 1:, pyom.tau] - pyom.v[1:-2, 1:-2, :-1, pyom.tau]) * pyom.flux_top[1:-2, 1:-2, :-1] / pyom.dzw[:-1]
    diss[:,:,-1] = 0.0
    diss = numerics.vgrid_to_tgrid(pyom,diss)
    pyom.K_diss_v += diss

    if not pyom.enable_hydrostatic:
        kss = pyom.kbot[2:-2, 2:-2] - 1
        delta[:-1,:-1,:-1] = pyom.dt_mom / pyom.dzt[None,None,:-1] * 0.5 * (pyom.kappaM[2:-2,2:-2,:-1] + pyom.kappaM[2:-2,2:-2,1:])
        delta[:-1,:-1,-1] = 0.
        a_tri[:-1,:-1,1:-1] = -delta[:-1,:-1,:-2] / pyom.dzw[None,None,1:-1]
        a_tri[:-1,:-1,-1] = 0.
        b_tri_edge = 1 + delta[:-1,:-1] / pyom.dzw[None,None,:]
        b_tri[:-1,:-1,1:] = 1 + delta[:-1,:-1,:-1] / pyom.dzw[None,None,:-1]
        b_tri[:-1,:-1,1:-1] += delta[:-1,:-1,1:-1] / pyom.dzw[None,None,1:-1]
        c_tri[:-1,:-1,:-1] = - delta[:-1,:-1,:-1] / pyom.dzw[None,None,:-1]
        c_tri[:-1,:-1,-1] = 0.
        d_tri[:-1,:-1] = pyom.w[2:-2,2:-2,:,pyom.tau]
        res, mask = utilities.solve_implicit(pyom, kss, a_tri[:-1,:-1], b_tri[:-1,:-1], c_tri[:-1,:-1], d_tri[:-1,:-1], b_edge=b_tri_edge)
        pyom.w[2:-2,2:-2,:,pyom.taup1] = np.where(mask, res, pyom.w[2:-2,2:-2,:,pyom.taup1])
        pyom.dw_mix[2:-2, 2:-2] = (pyom.w[2:-2,2:-2,:,pyom.taup1] - pyom.w[2:-2,2:-2,:,pyom.tau]) / pyom.dt_mom

        """
        diagnose dissipation by vertical friction of vertical momentum
        """
        fxa = 0.5 * (pyom.kappaM[1:-2, 1:-2, :-1] + pyom.kappaM[1:-2,1:-2,1:])
        pyom.flux_top[1:-2,1:-2,:-1] = fxa * (pyom.w[1:-2,1:-2,1:,pyom.taup1] - pyom.w[1:-2,1:-2,:-1,pyom.taup1]) \
                / pyom.dzt[1:] * pyom.maskW[1:-2, 1:-2, 1:] * pyom.maskW[1:-2, 1:-2, :-1]
        diss[1:-2, 1:-2, :-1] = (pyom.w[1:-2,1:-2,1:,pyom.tau] - pyom.w[1:-2,1:-2,:-1,pyom.tau]) * pyom.flux_top[1:-2,1:-2,:-1] / pyom.dzt[1:]
        diss[:,:,-1] = 0.0
        pyom.K_diss_v += diss

@pyom_method
def rayleigh_friction(pyom):
    """
    interior Rayleigh friction
    dissipation is calculated and added to K_diss_bot
    """
    pyom.du_mix[...] += -pyom.maskU * pyom.r_ray * pyom.u[..., pyom.tau]
    if pyom.enable_conserve_energy:
        diss = pyom.maskU * pyom.r_ray * pyom.u[...,pyom.tau]**2
        pyom.K_diss_bot[...] = numerics.calc_diss(pyom,diss,pyom.K_diss_bot,'U')
    pyom.dv_mix[...] += -pyom.maskV * pyom.r_ray * pyom.v[...,pyom.tau]
    if pyom.enable_conserve_energy:
        diss = pyom.maskV * pyom.r_ray * pyom.v[...,pyom.tau]**2
        pyom.K_diss_bot[...] = numerics.calc_diss(pyom,diss,pyom.K_diss_bot,'V')
    if not pyom.enable_hydrostatic:
        raise NotImplementedError("Rayleigh friction for vertical velocity not implemented")

@pyom_method
def linear_bottom_friction(pyom):
    """
    linear bottom friction
    dissipation is calculated and added to K_diss_bot
    """
    if pyom.enable_bottom_friction_var:
        """
        with spatially varying coefficient
        """
        k = np.maximum(pyom.kbot[1:-2,2:-2], pyom.kbot[2:-1,2:-2]) - 1
        mask = np.arange(pyom.nz) == k[:,:,None]
        pyom.du_mix[1:-2,2:-2] += -(pyom.maskU[1:-2,2:-2] * pyom.r_bot_var_u[1:-2,2:-2,None]) * pyom.u[1:-2,2:-2,:,pyom.tau] * mask
        if pyom.enable_conserve_energy:
            diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
            diss[1:-2,2:-2] = pyom.maskU[1:-2,2:-2] * pyom.r_bot_var_u[1:-2,2:-2,None] * pyom.u[1:-2,2:-2,:,pyom.tau]**2 * mask
            pyom.K_diss_bot[...] = numerics.calc_diss(pyom,diss,pyom.K_diss_bot,'U')

        k = np.maximum(pyom.kbot[2:-2, 2:-1], pyom.kbot[2:-2, 1:-2]) - 1
        mask = np.arange(pyom.nz) == k[:,:,None]
        pyom.dv_mix[2:-2,1:-2] += -(pyom.maskV[2:-2,1:-2] * pyom.r_bot_var_v[2:-2,1:-2,None]) * pyom.v[2:-2,1:-2,:,pyom.tau] * mask
        if pyom.enable_conserve_energy:
            diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
            diss[2:-2,1:-2] = pyom.maskV[2:-2,1:-2] * pyom.r_bot_var_v[2:-2,1:-2,None] * pyom.v[2:-2,1:-2,:,pyom.tau]**2 * mask
            pyom.K_diss_bot[...] = numerics.calc_diss(pyom,diss,pyom.K_diss_bot,'V')
    else:
        """
        with constant coefficient
        """
        k = np.maximum(pyom.kbot[1:-2,2:-2], pyom.kbot[2:-1,2:-2]) - 1
        mask = np.arange(pyom.nz) == k[:,:,None]
        pyom.du_mix[1:-2,2:-2] += -pyom.maskU[1:-2,2:-2] * pyom.r_bot * pyom.u[1:-2,2:-2,:,pyom.tau] * mask
        if pyom.enable_conserve_energy:
            diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
            diss[1:-2,2:-2] = pyom.maskU[1:-2,2:-2] * pyom.r_bot * pyom.u[1:-2,2:-2,:,pyom.tau]**2 * mask
            pyom.K_diss_bot[...] = numerics.calc_diss(pyom,diss,pyom.K_diss_bot,'U')

        k = np.maximum(pyom.kbot[2:-2, 2:-1], pyom.kbot[2:-2, 1:-2]) - 1
        mask = np.arange(pyom.nz) == k[:,:,None]
        pyom.dv_mix[2:-2,1:-2] += -pyom.maskV[2:-2,1:-2] * pyom.r_bot * pyom.v[2:-2,1:-2,:,pyom.tau] * mask
        if pyom.enable_conserve_energy:
            diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
            diss[2:-2,1:-2] = pyom.maskV[2:-2,1:-2] * pyom.r_bot * pyom.v[2:-2,1:-2,:,pyom.tau]**2 * mask
            pyom.K_diss_bot[...] = numerics.calc_diss(pyom,diss,pyom.K_diss_bot,'V')

    if not pyom.enable_hydrostatic:
        raise NotImplementedError("bottom friction for vertical velocity not implemented")

@pyom_method
def quadratic_bottom_friction(pyom):
    """
    quadratic bottom friction
    dissipation is calculated and added to K_diss_bot
    """
    # we might want to account for EKE in the drag, also a tidal residual
    k = np.maximum(pyom.kbot[1:-2,2:-2],pyom.kbot[2:-1,2:-2]) - 1
    mask = k[..., np.newaxis] == np.indices((pyom.nx+1, pyom.ny, pyom.nz))[2]
    fxa = pyom.maskV[1:-2,2:-2,:] * pyom.v[1:-2,2:-2,:,pyom.tau]**2 + pyom.maskV[1:-2,1:-3,:] * pyom.v[1:-2,1:-3,:,pyom.tau]**2 \
        + pyom.maskV[2:-1,2:-2,:] * pyom.v[2:-1,2:-2,:,pyom.tau]**2 + pyom.maskV[2:-1,1:-3,:] * pyom.v[2:-1,1:-3,:,pyom.tau]**2
    fxa = np.sqrt(pyom.u[1:-2,2:-2,:,pyom.tau]**2 + 0.25 * fxa)
    aloc = pyom.maskU[1:-2,2:-2,:] * pyom.r_quad_bot * pyom.u[1:-2,2:-2,:,pyom.tau] \
                             * fxa / pyom.dzt[np.newaxis, np.newaxis, :] * mask
    pyom.du_mix[1:-2,2:-2,:] += -aloc

    if pyom.enable_conserve_energy:
        diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
        diss[1:-2,2:-2,:] = aloc * pyom.u[1:-2,2:-2,:,pyom.tau]
        pyom.K_diss_bot[...] = numerics.calc_diss(pyom,diss,pyom.K_diss_bot,'U')

    k = np.maximum(pyom.kbot[2:-2,1:-2],pyom.kbot[2:-2,2:-1]) - 1
    mask = k[..., np.newaxis] == np.indices((pyom.nx, pyom.ny+1, pyom.nz))[2]
    fxa = pyom.maskU[2:-2,1:-2,:] * pyom.u[2:-2,1:-2,:,pyom.tau]**2 + pyom.maskU[1:-3,1:-2,:] * pyom.u[1:-3,1:-2,:,pyom.tau]**2 \
        + pyom.maskU[2:-2,2:-1,:] * pyom.u[2:-2,2:-1,:,pyom.tau]**2 + pyom.maskU[1:-3,2:-1,:] * pyom.u[1:-3,2:-1,:,pyom.tau]**2
    fxa = np.sqrt(pyom.v[2:-2,1:-2,:,pyom.tau]**2 + 0.25 * fxa)
    aloc = pyom.maskV[2:-2,1:-2,:] * pyom.r_quad_bot * pyom.v[2:-2,1:-2,:,pyom.tau] \
                             * fxa / pyom.dzt[np.newaxis, np.newaxis, :] * mask
    pyom.dv_mix[2:-2,1:-2,:] += -aloc

    if pyom.enable_conserve_energy:
        diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
        diss[2:-2,1:-2,:] = aloc * pyom.v[2:-2,1:-2,:,pyom.tau]
        pyom.K_diss_bot[...] = numerics.calc_diss(pyom,diss,pyom.K_diss_bot,'V')

    if not pyom.enable_hydrostatic:
        raise NotImplementedError("bottom friction for vertical velocity not implemented")

@pyom_method
def harmonic_friction(pyom):
    """
    horizontal harmonic friction
    dissipation is calculated and added to K_diss_h
    """
    diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    """
    Zonal velocity
    """
    if pyom.enable_hor_friction_cos_scaling:
        fxa = pyom.cost**pyom.hor_friction_cosPower
        pyom.flux_east[:-1] = pyom.A_h * fxa[None,:,None] * (pyom.u[1:,:,:,pyom.tau] - pyom.u[:-1,:,:,pyom.tau]) \
                / (pyom.cost * pyom.dxt[1:, None])[:,:,None] * pyom.maskU[1:] * pyom.maskU[:-1]
        fxa = pyom.cosu**pyom.hor_friction_cosPower
        pyom.flux_north[:,:-1] = pyom.A_h * fxa[None,:-1,None] * (pyom.u[:,1:,:,pyom.tau] - pyom.u[:,:-1,:,pyom.tau]) \
                / pyom.dyu[None,:-1,None] * pyom.maskU[:,1:] * pyom.maskU[:,:-1] * pyom.cosu[None,:-1,None]
    else:
        pyom.flux_east[:-1,:,:] = pyom.A_h * (pyom.u[1:,:,:,pyom.tau] - pyom[:-1,:,:,pyom.tau]) \
                / (pyom.cost * pyom.dxt[1:, None])[:,:,None] * pyom.maskU[1:] * pyom.maskU[:-1]
        pyom.flux_north[:,:-1,:] = pyom.A_h * (pyom.u[:,1:,:,pyom.tau] - pyom.u[:,:-1,:,pyom.tau]) \
                / pyom.dyu[None, :-1, None] * pyom.maskU[:,1:] * pyom.maskU[:,:-1] * pyom.cosu[None,:-1,None]
    pyom.flux_east[-1,:,:] = 0.
    pyom.flux_north[:,-1,:] = 0.

    """
    update tendency
    """
    pyom.du_mix[2:-2, 2:-2, :] += pyom.maskU[2:-2,2:-2] * ((pyom.flux_east[2:-2,2:-2] - pyom.flux_east[1:-3,2:-2]) \
                                                            / (pyom.cost[2:-2] * pyom.dxu[2:-2, None])[:,:,None] \
                                                        + (pyom.flux_north[2:-2,2:-2] - pyom.flux_north[2:-2,1:-3]) \
                                                            / (pyom.cost[2:-2] * pyom.dyt[2:-2])[None, :, None])

    if pyom.enable_conserve_energy:
        """
        diagnose dissipation by lateral friction
        """
        diss[1:-2, 2:-2] = 0.5*((pyom.u[2:-1,2:-2,:,pyom.tau] - pyom.u[1:-2,2:-2,:,pyom.tau]) * pyom.flux_east[1:-2,2:-2] \
                + (pyom.u[1:-2,2:-2,:,pyom.tau] - pyom.u[:-3,2:-2,:,pyom.tau]) * pyom.flux_east[:-3,2:-2]) \
                    / (pyom.cost[2:-2] * pyom.dxu[1:-2,None])[:,:,None]\
                + 0.5*((pyom.u[1:-2,3:-1,:,pyom.tau] - pyom.u[1:-2,2:-2,:,pyom.tau]) * pyom.flux_north[1:-2,2:-2] \
                + (pyom.u[1:-2,2:-2,:,pyom.tau] - pyom.u[1:-2,1:-3,:,pyom.tau]) * pyom.flux_north[1:-2,1:-3]) \
                    / (pyom.cost[2:-2] * pyom.dyt[2:-2])[None,:,None]
        pyom.K_diss_h[...] = 0.
        pyom.K_diss_h[...] = numerics.calc_diss(pyom,diss,pyom.K_diss_h,'U')

    """
    Meridional velocity
    """
    if pyom.enable_hor_friction_cos_scaling:
        fxa = (pyom.cosu ** pyom.hor_friction_cosPower) * np.ones(pyom.nx+3)[:,None]
        pyom.flux_east[:-1] = pyom.A_h * fxa[:, :, None] * (pyom.v[1:,:,:,pyom.tau] - pyom.v[:-1,:,:,pyom.tau]) \
                / (pyom.cosu * pyom.dxu[:-1, None])[:,:,None] * pyom.maskV[1:] * pyom.maskV[:-1]
        fxa = (pyom.cost[1:] ** pyom.hor_friction_cosPower) * np.ones(pyom.nx+4)[:, None]
        pyom.flux_north[:,:-1] = pyom.A_h * fxa[:,:,None] * (pyom.v[:,1:,:,pyom.tau] - pyom.v[:,:-1,:,pyom.tau]) \
                / pyom.dyt[None,1:,None] * pyom.cost[None,1:,None] * pyom.maskV[:,:-1] * pyom.maskV[:,1:]
    else:
        pyom.flux_east[:-1] = pyom.A_h * (pyom.v[1:,:,:,pyom.tau] - pyom.v[:-1,:,:,pyom.tau]) \
                / (pyom.cosu * pyom.dxu[:-1, None])[:,:,None] * pyom.maskV[1:] * pyom.maskV[:-1]
        pyom.flux_north[:,:-1] = pyom.A_h * (pyom.v[:,1:,:,pyom.tau] - pyom.v[:,:-1,:,pyom.tau]) \
                / pyom.dyt[None,1:,None] * pyom.cost[None,1:,None] * pyom.maskV[:,:-1] * pyom.maskV[:,1:]
    pyom.flux_east[-1,:,:] = 0.
    pyom.flux_north[:,-1,:] = 0.

    """
    update tendency
    """
    pyom.dv_mix[2:-2,2:-2] += pyom.maskV[2:-2,2:-2] * ((pyom.flux_east[2:-2,2:-2] - pyom.flux_east[1:-3,2:-2]) \
                                / (pyom.cosu[2:-2] * pyom.dxt[2:-2,None])[:,:,None] \
                            + (pyom.flux_north[2:-2,2:-2] - pyom.flux_north[2:-2,1:-3]) \
                                / (pyom.dyu[2:-2] * pyom.cosu[2:-2])[None,:,None])

    if pyom.enable_conserve_energy:
        """
        diagnose dissipation by lateral friction
        """
        diss[2:-2,1:-2] = 0.5 * ((pyom.v[3:-1,1:-2,:,pyom.tau] - pyom.v[2:-2,1:-2,:,pyom.tau]) * pyom.flux_east[2:-2,1:-2]\
                + (pyom.v[2:-2,1:-2,:,pyom.tau] - pyom.v[1:-3,1:-2,:,pyom.tau]) * pyom.flux_east[1:-3,1:-2]) \
                / (pyom.cosu[1:-2] * pyom.dxt[2:-2,None])[:,:,None] \
                + 0.5*((pyom.v[2:-2,2:-1,:,pyom.tau] - pyom.v[2:-2,1:-2,:,pyom.tau]) * pyom.flux_north[2:-2,1:-2] \
                + (pyom.v[2:-2,1:-2,:,pyom.tau] - pyom.v[2:-2,:-3,:,pyom.tau]) * pyom.flux_north[2:-2,:-3]) \
                / (pyom.cosu[1:-2] * pyom.dyu[1:-2])[None,:,None]
        pyom.K_diss_h[...] = numerics.calc_diss(pyom,diss,pyom.K_diss_h,'V')

    if not pyom.enable_hydrostatic:
        if pyom.enable_hor_friction_cos_scaling:
            raise NotImplementedError("scaling of lateral friction for vertical velocity not implemented")

        pyom.flux_east[:-1] = pyom.A_h * (pyom.w[1:,:,:,pyom.tau] - pyom.w[:-1,:,:,pyom.tau]) \
                / (pyom.cost * pyom.dxu[:,None])[:,:,None] * pyom.maskW[1:] * pyom.maskW[:-1]
        pyom.flux_north[:,:-1] = pyom.A_h * (pyom.w[:,1:,:,pyom.tau] - pyom.w[:,:-1,:,pyom.tau]) \
                / pyom.dyu[None,:-1,None] * pyom.maskW[:,1:] * pyom.maskW[:,:-1] * pyom.cosu[None,:-1,None]
        pyom.flux_east[-1,:,:] = 0.
        pyom.flux_north[:,-1,:] = 0.

        """
        update tendency
        """
        pyom.dw_mix[2:-2,2:-2] += pyom.maskW[2:-2,2:-2]*((pyom.flux_east[2:-2,2:-2] - pyom.flux_east[1:-3,2:-2]) \
                / (pyom.cost[2:-2] * pyom.dxt[2:-2,None])[:,:,None] \
                + (pyom.flux_north[2:-2,2:-2] - pyom.flux_north[2:-2,1:-3]) \
                / (pyom.dyt[2:-2] * pyom.cost[2:-2])[None,:,None])

        """
        diagnose dissipation by lateral friction
        """
        # to be implemented

@pyom_method
def biharmonic_friction(pyom):
    """
    horizontal biharmonic friction
    dissipation is calculated and added to K_diss_h
    """
    if not pyom.enable_hydrostatic:
        raise NotImplementedError("biharmonic mixing for non-hydrostatic case not yet implemented")

    fxa = math.sqrt(abs(pyom.A_hbi))

    """
    Zonal velocity
    """
    pyom.flux_east[:-1,:,:] = fxa * (pyom.u[1:,:,:,pyom.tau] - pyom.u[:-1,:,:,pyom.tau]) \
                            / (pyom.cost[np.newaxis, :, np.newaxis] * pyom.dxt[1:, np.newaxis, np.newaxis]) \
                            * pyom.maskU[1:,:,:] * pyom.maskU[:-1,:,:]
    pyom.flux_north[:,:-1,:] = fxa * (pyom.u[:,1:,:,pyom.tau] - pyom.u[:,:-1,:,pyom.tau]) \
                             / pyom.dyu[np.newaxis, :-1, np.newaxis] * pyom.maskU[:,1:,:] \
                             * pyom.maskU[:,:-1,:] * pyom.cosu[np.newaxis, :-1, np.newaxis]
    pyom.flux_east[-1,:,:] = 0.
    pyom.flux_north[:,-1,:] = 0.

    del2 = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    del2[1:,1:,:] = (pyom.flux_east[1:,1:,:] - pyom.flux_east[:-1,1:,:]) \
                        / (pyom.cost[np.newaxis, 1:, np.newaxis] * pyom.dxu[1:, np.newaxis, np.newaxis]) \
                  + (pyom.flux_north[1:,1:,:] - pyom.flux_north[1:,:-1,:]) \
                        / (pyom.cost[np.newaxis, 1:, np.newaxis] * pyom.dyt[np.newaxis, 1:, np.newaxis])

    pyom.flux_east[:-1,:,:] = fxa * (del2[1:,:,:] - del2[:-1,:,:]) \
                            / (pyom.cost[np.newaxis, :, np.newaxis] * pyom.dxt[1:, np.newaxis, np.newaxis]) \
                            * pyom.maskU[1:,:,:] * pyom.maskU[:-1,:,:]
    pyom.flux_north[:,:-1,:] = fxa * (del2[:,1:,:] - del2[:,:-1,:]) \
                             / pyom.dyu[np.newaxis, :-1, np.newaxis] * pyom.maskU[:,1:,:] \
                             * pyom.maskU[:,:-1,:] * pyom.cosu[np.newaxis, :-1, np.newaxis]
    pyom.flux_east[-1,:,:] = 0.
    pyom.flux_north[:,-1,:] = 0.

    """
    update tendency
    """
    pyom.du_mix[2:-2,2:-2,:] += -pyom.maskU[2:-2,2:-2,:] * ((pyom.flux_east[2:-2,2:-2,:] - pyom.flux_east[1:-3,2:-2,:]) \
                                    / (pyom.cost[np.newaxis, 2:-2, np.newaxis] * pyom.dxu[2:-2, np.newaxis, np.newaxis]) \
                                    + (pyom.flux_north[2:-2,2:-2,:] - pyom.flux_north[2:-2,1:-3,:]) \
                                    / (pyom.cost[np.newaxis, 2:-2, np.newaxis] * pyom.dyt[np.newaxis, 2:-2, np.newaxis]))
    if pyom.enable_conserve_energy:
        """
        diagnose dissipation by lateral friction
        """
        if pyom.enable_cyclic_x:
            cyclic.setcyclic_x(pyom.flux_east)
            cyclic.setcyclic_x(pyom.flux_north)
        diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
        diss[1:-2, 2:-2, :] = -0.5 * ((pyom.u[2:-1,2:-2,:,pyom.tau] - pyom.u[1:-2,2:-2,:,pyom.tau]) * pyom.flux_east[1:-2,2:-2,:] \
                                    + (pyom.u[1:-2,2:-2,:,pyom.tau] - pyom.u[:-3,2:-2,:,pyom.tau]) * pyom.flux_east[:-3,2:-2,:]) \
                                    / (pyom.cost[np.newaxis, 2:-2, np.newaxis] * pyom.dxu[1:-2, np.newaxis, np.newaxis])  \
                              -0.5 * ((pyom.u[1:-2,3:-1,:,pyom.tau] - pyom.u[1:-2,2:-2,:,pyom.tau]) * pyom.flux_north[1:-2,2:-2,:] \
                                    + (pyom.u[1:-2,2:-2,:,pyom.tau] - pyom.u[1:-2,1:-3,:,pyom.tau]) * pyom.flux_north[1:-2,1:-3,:]) \
                                    / (pyom.cost[np.newaxis, 2:-2, np.newaxis] * pyom.dyt[np.newaxis, 2:-2, np.newaxis])
        pyom.K_diss_h[...] = 0.
        pyom.K_diss_h[...] = numerics.calc_diss(pyom,diss,pyom.K_diss_h,'U')

    """
    Meridional velocity
    """
    pyom.flux_east[:-1, :, :] = fxa * (pyom.v[1:,:,:,pyom.tau] - pyom.v[:-1,:,:,pyom.tau]) \
                             / (pyom.cosu[np.newaxis, :, np.newaxis] * pyom.dxu[:-1, np.newaxis, np.newaxis]) \
                             * pyom.maskV[1:,:,:] * pyom.maskV[:-1,:,:]
    pyom.flux_north[:,:-1,:] = fxa * (pyom.v[:,1:,:,pyom.tau] - pyom.v[:,:-1,:,pyom.tau]) \
                             / pyom.dyt[np.newaxis, 1:, np.newaxis] * pyom.cost[np.newaxis, 1:, np.newaxis] \
                             * pyom.maskV[:,:-1,:] * pyom.maskV[:,1:,:]
    pyom.flux_east[-1,:,:] = 0.
    pyom.flux_north[:,-1,:] = 0.

    del2[1:,1:,:] = (pyom.flux_east[1:,1:,:] - pyom.flux_east[:-1,1:,:]) \
                        / (pyom.cosu[np.newaxis, 1:, np.newaxis] * pyom.dxt[1:, np.newaxis, np.newaxis])  \
                  + (pyom.flux_north[1:,1:,:] - pyom.flux_north[1:,:-1,:]) \
                        / (pyom.dyu[np.newaxis, 1:, np.newaxis] * pyom.cosu[np.newaxis, 1:, np.newaxis])
    pyom.flux_east[:-1,:,:] = fxa * (del2[1:,:,:] - del2[:-1,:,:]) \
                            / (pyom.cosu[np.newaxis,:,np.newaxis] * pyom.dxu[:-1,np.newaxis,np.newaxis]) \
                            * pyom.maskV[1:,:,:] * pyom.maskV[:-1,:,:]
    pyom.flux_north[:,:-1,:] = fxa * (del2[:,1:,:] - del2[:,:-1,:]) \
                             / pyom.dyt[np.newaxis,1:,np.newaxis] * pyom.cost[np.newaxis, 1:, np.newaxis] \
                             * pyom.maskV[:,:-1,:] * pyom.maskV[:,1:,:]
    pyom.flux_east[-1,:,:] = 0.
    pyom.flux_north[:,-1,:] = 0.

    """
    update tendency
    """
    pyom.dv_mix[2:-2, 2:-2, :] += -pyom.maskV[2:-2,2:-2,:] * ((pyom.flux_east[2:-2,2:-2,:] - pyom.flux_east[1:-3,2:-2,:]) \
                                    / (pyom.cosu[np.newaxis, 2:-2, np.newaxis] * pyom.dxt[2:-2, np.newaxis, np.newaxis]) \
                                    + (pyom.flux_north[2:-2,2:-2,:] - pyom.flux_north[2:-2,1:-3,:]) \
                                    / (pyom.dyu[np.newaxis, 2:-2, np.newaxis] * pyom.cosu[np.newaxis, 2:-2, np.newaxis]))

    if pyom.enable_conserve_energy:
        """
        diagnose dissipation by lateral friction
        """
        if pyom.enable_cyclic_x:
            cyclic.setcyclic_x(pyom.flux_east)
            cyclic.setcyclic_x(pyom.flux_north)
        diss[2:-2, 1:-2, :] = -0.5*((pyom.v[3:-1,1:-2,:,pyom.tau] - pyom.v[2:-2,1:-2,:,pyom.tau]) * pyom.flux_east[2:-2,1:-2,:] \
                                  + (pyom.v[2:-2,1:-2,:,pyom.tau] - pyom.v[1:-3,1:-2,:,pyom.tau]) * pyom.flux_east[1:-3,1:-2,:]) \
                                  / (pyom.cosu[np.newaxis, 1:-2, np.newaxis] * pyom.dxt[2:-2, np.newaxis, np.newaxis]) \
                             - 0.5*((pyom.v[2:-2,2:-1,:,pyom.tau] - pyom.v[2:-2,1:-2,:,pyom.tau]) * pyom.flux_north[2:-2,1:-2,:] \
                                  + (pyom.v[2:-2,1:-2,:,pyom.tau] - pyom.v[2:-2,:-3,:,pyom.tau]) * pyom.flux_north[2:-2,:-3,:]) \
                                  / (pyom.cosu[np.newaxis, 1:-2, np.newaxis] * pyom.dyu[np.newaxis, 1:-2, np.newaxis])
        pyom.K_diss_h[...] = numerics.calc_diss(pyom,diss,pyom.K_diss_h,'V')

@pyom_method
def momentum_sources(pyom):
    """
    other momentum sources
    dissipation is calculated and added to K_diss_bot
    """
    pyom.du_mix[...] += pyom.maskU * pyom.u_source
    if pyom.enable_conserve_energy:
        diss = -pyom.maskU * pyom.u[..., pyom.tau] * pyom.u_source
        pyom.K_diss_bot[...] = numerics.calc_diss(pyom,diss,pyom.K_diss_bot,'U')
    pyom.dv_mix[...] += pyom.maskV * pyom.v_source
    if pyom.enable_conserve_energy:
        diss = -pyom.maskV * pyom.v[..., pyom.tau] * pyom.v_source
        pyom.K_diss_bot[...] = numerics.calc_diss(pyom,diss,pyom.K_diss_bot,'V')

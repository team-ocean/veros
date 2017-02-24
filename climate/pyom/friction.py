import math
import numpy as np

from climate.pyom import numerics, utilities, cyclic


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
    diss[...] = numerics.ugrid_to_tgrid(diss,pyom)
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
    diss[...] = numerics.vgrid_to_tgrid(diss,pyom)
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
    res, mask = utilities.solve_implicit(kss, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)
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
    diss[...] = numerics.ugrid_to_tgrid(diss,pyom)
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
    res, mask = utilities.solve_implicit(kss, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)
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
    diss = numerics.vgrid_to_tgrid(diss,pyom)
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
        res, mask = utilities.solve_implicit(kss, a_tri[:-1,:-1], b_tri[:-1,:-1], c_tri[:-1,:-1], d_tri[:-1,:-1], b_edge=b_tri_edge)
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


def rayleigh_friction(pyom):
    """
    interior Rayleigh friction
    dissipation is calculated and added to K_diss_bot
    """
    pyom.du_mix[...] += -pyom.maskU * pyom.r_ray * pyom.u[..., pyom.tau]
    if pyom.enable_conserve_energy:
        diss = pyom.maskU * pyom.r_ray * pyom.u[...,pyom.tau]**2
        pyom.K_diss_bot[...] = numerics.calc_diss(diss,pyom.K_diss_bot,'U',pyom)
    pyom.dv_mix[...] += -pyom.maskV * pyom.r_ray * pyom.v[...,pyom.tau]
    if pyom.enable_conserve_energy:
        diss = pyom.maskV * pyom.r_ray * pyom.v[...,pyom.tau]**2
        pyom.K_diss_bot[...] = numerics.calc_diss(diss,pyom.K_diss_bot,'V',pyom)
    if not pyom.enable_hydrostatic:
        raise NotImplementedError("Rayleigh friction for vertical velocity not implemented")


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
            pyom.K_diss_bot[...] = numerics.calc_diss(diss,pyom.K_diss_bot,'U',pyom)

        k = np.maximum(pyom.kbot[2:-2, 2:-1], pyom.kbot[2:-2, 1:-2]) - 1
        mask = np.arange(pyom.nz) == k[:,:,None]
        pyom.dv_mix[2:-2,1:-2] += -(pyom.maskV[2:-2,1:-2] * pyom.r_bot_var_v[2:-2,1:-2,None]) * pyom.v[2:-2,1:-2,:,pyom.tau] * mask
        if pyom.enable_conserve_energy:
            diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
            diss[2:-2,1:-2] = pyom.maskV[2:-2,1:-2] * pyom.r_bot_var_v[2:-2,1:-2,None] * pyom.v[2:-2,1:-2,:,pyom.tau]**2 * mask
            pyom.K_diss_bot[...] = numerics.calc_diss(diss,pyom.K_diss_bot,'V',pyom)
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
            pyom.K_diss_bot[...] = numerics.calc_diss(diss,pyom.K_diss_bot,'U',pyom)

        k = np.maximum(pyom.kbot[2:-2, 2:-1], pyom.kbot[2:-2, 1:-2]) - 1
        mask = np.arange(pyom.nz) == k[:,:,None]
        pyom.dv_mix[2:-2,1:-2] += -pyom.maskV[2:-2,1:-2] * pyom.r_bot * pyom.v[2:-2,1:-2,:,pyom.tau] * mask
        if pyom.enable_conserve_energy:
            diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
            diss[2:-2,1:-2] = pyom.maskV[2:-2,1:-2] * pyom.r_bot * pyom.v[2:-2,1:-2,:,pyom.tau]**2 * mask
            pyom.K_diss_bot[...] = numerics.calc_diss(diss,pyom.K_diss_bot,'V',pyom)

    if not pyom.enable_hydrostatic:
        raise NotImplementedError("bottom friction for vertical velocity not implemented")


def quadratic_bottom_friction(pyom):
    """
    quadratic bottom friction
    dissipation is calculated and added to K_diss_bot
    """
    diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    aloc = np.zeros((pyom.nx+4, pyom.ny+4))

    # we might want to account for EKE in the drag, also a tidal residual
    aloc[...] = 0.0
    for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
        for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
            k = max(pyom.kbot[i,j],pyom.kbot[i+1,j]) - 1
            if k >= 0:
                fxa = pyom.maskV[i,j,k] * pyom.v[i,j,k,pyom.tau]**2 + pyom.maskV[i,j-1,k] * pyom.v[i,j-1,k,pyom.tau]**2 \
                      + pyom.maskV[i+1,j,k] * pyom.v[i+1,j,k,pyom.tau]**2 + pyom.maskV[i+1,j-1,k] * pyom.v[i+1,j-1,k,pyom.tau]**2
                fxa = np.sqrt(pyom.u[i,j,k,pyom.tau]**2 + 0.25*fxa)
                aloc[i,j] = pyom.maskU[i,j,k] * pyom.r_quad_bot * pyom.u[i,j,k,pyom.tau] * fxa / pyom.dzt[k]
                pyom.du_mix[i,j,k] -= aloc[i,j]

    if pyom.enable_conserve_energy:
        diss[...] = 0.0
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                k = max(pyom.kbot[i,j],pyom.kbot[i+1,j]) - 1
                if k >= 0:
                    diss[i,j,k] = aloc[i,j] * pyom.u[i,j,k,pyom.tau]
        pyom.K_diss_bot[...] = numerics.calc_diss(diss,pyom.K_diss_bot,'U',pyom)

    aloc[...] = 0.0
    for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
        for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
            k = max(pyom.kbot[i,j+1],pyom.kbot[i,j]) - 1
            if k >= 0:
                fxa = pyom.maskU[i,j,k] * pyom.u[i,j,k,pyom.tau]**2 + pyom.maskU[i-1,j,k] * pyom.u[i-1,j,k,pyom.tau]**2 \
                      + pyom.maskU[i,j+1,k] * pyom.u[i,j+1,k,pyom.tau]**2 + pyom.maskU[i-1,j+1,k] * pyom.u[i-1,j+1,k,pyom.tau]**2
                fxa = np.sqrt(pyom.v[i,j,k,pyom.tau]**2 + 0.25*fxa)
                aloc[i,j] = pyom.maskV[i,j,k] * pyom.r_quad_bot * pyom.v[i,j,k,pyom.tau] * fxa / pyom.dzt[k]
                pyom.dv_mix[i,j,k] -= aloc[i,j]

    if pyom.enable_conserve_energy:
        diss[...] = 0.0
        for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                k = max(pyom.kbot[i,j+1],pyom.kbot[i,j]) - 1
                if k >= 0:
                    diss[i,j,k] = aloc[i,j] * pyom.v[i,j,k,pyom.tau]
        pyom.K_diss_bot[...] = numerics.calc_diss(diss,pyom.K_diss_bot,'V',pyom)

    if not pyom.enable_hydrostatic:
        raise NotImplementedError("bottom friction for vertical velocity not implemented")


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
        pyom.K_diss_h[...] = numerics.calc_diss(diss,pyom.K_diss_h,'U',pyom)

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
        pyom.K_diss_h[...] = numerics.calc_diss(diss,pyom.K_diss_h,'V',pyom)

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


def biharmonic_friction(pyom):
    """
    horizontal biharmonic friction
    dissipation is calculated and added to K_diss_h
    """
    del2 = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    if not pyom.enable_hydrostatic:
        raise NotImplementedError("biharmonic mixing for non-hydrostatic not yet implemented")

    is_ = pyom.is_pe - pyom.onx
    ie_ = pyom.ie_pe + pyom.onx
    js_ = pyom.js_pe - pyom.onx
    je_ = pyom.je_pe + pyom.onx
    fxa = math.sqrt(abs(pyom.A_hbi))

    """
    Zonal velocity
    """
    for j in xrange(js_,je_): # j = js,je
        for i in xrange(is_,ie_-1): # i = is,ie-1
            pyom.flux_east[i,j,:] = fxa * (pyom.u[i+1,j,:,pyom.tau] - pyom.u[i,j,:,pyom.tau]) \
                                    / (pyom.cost[j] * pyom.dxt[i+1]) * pyom.maskU[i+1,j,:] * pyom.maskU[i,j,:]
    for j in xrange(js_,je_-1): # j = js,je-1
        pyom.flux_north[:,j,:] = fxa * (pyom.u[:,j+1,:,pyom.tau] - pyom.u[:,j,:,pyom.tau]) \
                                 / pyom.dyu[j] * pyom.maskU[:,j+1,:] * pyom.maskU[:,j,:] * pyom.cosu[j]
    pyom.flux_east[ie_-1,:,:] = 0.
    pyom.flux_north[:,je_-1,:] = 0.

    for j in xrange(js_+1,je_): # j = js+1,je
        for i in xrange(is_+1,ie_): # i = is+1,ie
            del2[i,j,:] = (pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:]) / (pyom.cost[j] * pyom.dxu[i]) \
                        + (pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:]) / (pyom.cost[j] * pyom.dyt[j])

    for j in xrange(js_,je_): # j = js,je
        for i in xrange(is_,ie_-1): # i = is,ie-1
            pyom.flux_east[i,j,:] = fxa * (del2[i+1,j,:] - del2[i,j,:]) \
                                    / (pyom.cost[j] * pyom.dxt[i+1]) * pyom.maskU[i+1,j,:] * pyom.maskU[i,j,:]
    for j in xrange(js_,je_-1): # j = js,je-1
        pyom.flux_north[:,j,:] = fxa * (del2[:,j+1,:] - del2[:,j,:]) \
                                 / pyom.dyu[j] * pyom.maskU[:,j+1,:] * pyom.maskU[:,j,:] * pyom.cosu[j]
    pyom.flux_east[ie_-1,:,:] = 0.
    pyom.flux_north[:,je_-1,:] = 0.

    """
    update tendency
    """
    for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
        for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
            pyom.du_mix[i,j,:] += -pyom.maskU[i,j,:] * ((pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:]) \
                                                    / (pyom.cost[j] * pyom.dxu[i]) \
                                                 +(pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:]) \
                                                    / (pyom.cost[j] * pyom.dyt[j]))
    if pyom.enable_conserve_energy:
        """
        diagnose dissipation by lateral friction
        """
        if pyom.enable_cyclic_x:
            cyclic.setcyclic_x(pyom.flux_east)
            cyclic.setcyclic_x(pyom.flux_north)
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                diss[i,j,:] = -0.5*((pyom.u[i+1,j,:,pyom.tau] - pyom.u[i,j,:,pyom.tau]) * pyom.flux_east[i,j,:] \
                                   +(pyom.u[i,j,:,pyom.tau] - pyom.u[i-1,j,:,pyom.tau]) * pyom.flux_east[i-1,j,:]) \
                                  / (pyom.cost[j] * pyom.dxu[i])  \
                              -0.5*((pyom.u[i,j+1,:,pyom.tau] - pyom.u[i,j,:,pyom.tau]) * pyom.flux_north[i,j,:] \
                                   +(pyom.u[i,j,:,pyom.tau] - pyom.u[i,j-1,:,pyom.tau]) * pyom.flux_north[i,j-1,:]) \
                                  / (pyom.cost[j] * pyom.dyt[j])
        pyom.K_diss_h[...] = 0.
        pyom.K_diss_h[...] = numerics.calc_diss(diss,pyom.K_diss_h,'U',pyom)

    """
    Meridional velocity
    """
    for j in xrange(js_,je_): # j = js,je
        for i in xrange(is_,ie_-1): # i = is,ie-1
            pyom.flux_east[i,j,:] = fxa * (pyom.v[i+1,j,:,pyom.tau] - pyom.v[i,j,:,pyom.tau]) \
                                     / (pyom.cosu[j] * pyom.dxu[i]) \
                                     * pyom.maskV[i+1,j,:] * pyom.maskV[i,j,:]
    for j in xrange(js_,je_-1): # j = js,je-1
        pyom.flux_north[:,j,:] = fxa * (pyom.v[:,j+1,:,pyom.tau] - pyom.v[:,j,:,pyom.tau]) \
                                 / pyom.dyt[j+1] * pyom.cost[j+1] * pyom.maskV[:,j,:] * pyom.maskV[:,j+1,:]
    pyom.flux_east[ie_-1,:,:] = 0.
    pyom.flux_north[:,je_-1,:] = 0.

    for j in xrange(js_+1,je_): # j = js+1,je
        for i in xrange(is_+1,ie_): # i = is+1,ie
            del2[i,j,:] = (pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:]) / (pyom.cosu[j] * pyom.dxt[i])  \
                         +(pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:]) / (pyom.dyu[j] * pyom.cosu[j])
    for j in xrange(js_,je_): # j = js,je
        for i in xrange(is_,ie_-1): # i = is,ie-1
            pyom.flux_east[i,j,:] = fxa * (del2[i+1,j,:] - del2[i,j,:]) \
                                    / (pyom.cosu[j] * pyom.dxu[i]) \
                                    * pyom.maskV[i+1,j,:] * pyom.maskV[i,j,:]
    for j in xrange(js_,je_-1): # j = js,je-1
        pyom.flux_north[:,j,:] = fxa * (del2[:,j+1,:] - del2[:,j,:]) \
                                 / pyom.dyt[j+1] * pyom.cost[j+1] * pyom.maskV[:,j,:] * pyom.maskV[:,j+1,:]
    pyom.flux_east[ie_-1,:,:] = 0.
    pyom.flux_north[:,je_-1,:] = 0.

    """
    update tendency
    """
    for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
        for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
            pyom.dv_mix[i,j,:] += -pyom.maskV[i,j,:] * ((pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:]) \
                                                        / (pyom.cosu[j] * pyom.dxt[i]) \
                                                     + (pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:]) \
                                                        / (pyom.dyu[j] * pyom.cosu[j]))

    if pyom.enable_conserve_energy:
        """
        diagnose dissipation by lateral friction
        """
        if pyom.enable_cyclic_x:
            cyclic.setcyclic_x(pyom.flux_east)
            cyclic.setcyclic_x(pyom.flux_north)
        for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                diss[i,j,:] = -0.5*((pyom.v[i+1,j,:,pyom.tau] - pyom.v[i,j,:,pyom.tau]) * pyom.flux_east[i,j,:] \
                                  + (pyom.v[i,j,:,pyom.tau] - pyom.v[i-1,j,:,pyom.tau]) * pyom.flux_east[i-1,j,:]) \
                                 / (pyom.cosu[j]*pyom.dxt[i]) \
                             - 0.5*((pyom.v[i,j+1,:,pyom.tau] - pyom.v[i,j,:,pyom.tau]) * pyom.flux_north[i,j,:] \
                                  + (pyom.v[i,j,:,pyom.tau] - pyom.v[i,j-1,:,pyom.tau]) * pyom.flux_north[i,j-1,:]) \
                                 / (pyom.cosu[j] * pyom.dyu[j])
        pyom.K_diss_h[...] = numerics.calc_diss(diss,pyom.K_diss_h,'V',pyom)


def momentum_sources(pyom):
    """
    other momentum sources
    dissipation is calculated and added to K_diss_bot
    """
    diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    for k in xrange(pyom.nz): # k = 1,nz
        pyom.du_mix[:,:,k] += pyom.maskU[:,:,k] * pyom.u_source[:,:,k]
    if pyom.enable_conserve_energy:
        for k in xrange(pyom.nz): # k = 1,nz
            diss[:,:,k] = -pyom.maskU[:,:,k] * pyom.u[:,:,k,pyom.tau] * pyom.u_source[:,:,k]
        pyom.K_diss_bot[...] = numerics.calc_diss(diss,pyom.K_diss_bot,'U',pyom)
    for k in xrange(pyom.nz): # k = 1,nz
        pyom.dv_mix[:,:,k] += pyom.maskV[:,:,k] * pyom.v_source[:,:,k]
    if pyom.enable_conserve_energy:
        for k in xrange(pyom.nz): # k = 1,nz
            diss[:,:,k] = -pyom.maskV[:,:,k] * pyom.v[:,:,k,pyom.tau] * pyom.v_source[:,:,k]
        pyom.K_diss_bot[...] = numerics.calc_diss(diss,pyom.K_diss_bot,'V',pyom)

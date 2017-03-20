from .. import pyom_method
from . import friction, isoneutral, external, non_hydrostatic

@pyom_method
def momentum(pyom):
    """
    solve for momentum for taup1
    """

    """
    time tendency due to Coriolis force
    """
    pyom.du_cor[2:-2, 2:-2] = pyom.maskU[2:-2, 2:-2] \
            * (pyom.coriolis_t[2:-2, 2:-2,np.newaxis] * (pyom.v[2:-2, 2:-2,:,pyom.tau] + pyom.v[2:-2, 1:-3,:,pyom.tau]) \
            * pyom.dxt[2:-2,np.newaxis,np.newaxis] / pyom.dxu[2:-2,np.newaxis,np.newaxis] \
            + pyom.coriolis_t[3:-1,2:-2,np.newaxis] * (pyom.v[3:-1,2:-2,:,pyom.tau] + pyom.v[3:-1,1:-3,:,pyom.tau]) \
            * pyom.dxt[3:-1,np.newaxis,np.newaxis] / pyom.dxu[2:-2,np.newaxis,np.newaxis]) * 0.25
    pyom.dv_cor[2:-2, 2:-2] = -pyom.maskV[2:-2, 2:-2]\
            * (pyom.coriolis_t[2:-2, 2:-2,np.newaxis] * (pyom.u[1:-3, 2:-2,:,pyom.tau] + pyom.u[2:-2, 2:-2,:,pyom.tau]) \
            * pyom.dyt[np.newaxis,2:-2,np.newaxis] * pyom.cost[np.newaxis,2:-2,np.newaxis] \
            / (pyom.dyu[np.newaxis,2:-2,np.newaxis] * pyom.cosu[np.newaxis,2:-2,np.newaxis]) \
            + pyom.coriolis_t[2:-2,3:-1,np.newaxis] * (pyom.u[1:-3, 3:-1,:,pyom.tau] + pyom.u[2:-2,3:-1,:,pyom.tau]) \
            * pyom.dyt[np.newaxis,3:-1,np.newaxis] * pyom.cost[np.newaxis,3:-1,np.newaxis] \
            / (pyom.dyu[np.newaxis,2:-2,np.newaxis] * pyom.cosu[np.newaxis,2:-2,np.newaxis])) * 0.25

    """
    time tendency due to metric terms
    """
    if pyom.coord_degree:
        pyom.du_cor[2:-2, 2:-2] += pyom.maskU[2:-2, 2:-2] * 0.125 * pyom.tantr[np.newaxis,2:-2,np.newaxis] \
                * ((pyom.u[2:-2, 2:-2,:,pyom.tau] + pyom.u[1:-3, 2:-2,:,pyom.tau]) \
                 * (pyom.v[2:-2, 2:-2,:,pyom.tau] + pyom.v[2:-2, 1:-3,:,pyom.tau]) \
                 * pyom.dxt[2:-2,np.newaxis,np.newaxis] / pyom.dxu[2:-2,np.newaxis,np.newaxis] \
                 + (pyom.u[3:-1, 2:-2,:,pyom.tau] + pyom.u[2:-2, 2:-2,:,pyom.tau]) \
                 * (pyom.v[3:-1, 2:-2,:,pyom.tau] + pyom.v[3:-1, 1:-3,:,pyom.tau]) \
                 * pyom.dxt[3:-1,np.newaxis,np.newaxis] / pyom.dxu[2:-2,np.newaxis,np.newaxis])
        pyom.dv_cor[2:-2, 2:-2] += -pyom.maskV[2:-2, 2:-2] * 0.125 \
                * (pyom.tantr[np.newaxis,2:-2,np.newaxis] * (pyom.u[2:-2,2:-2,:,pyom.tau] + pyom.u[1:-3,2:-2,:,pyom.tau])**2 \
                * pyom.dyt[np.newaxis,2:-2,np.newaxis] * pyom.cost[np.newaxis,2:-2,np.newaxis] \
                / (pyom.dyu[np.newaxis,2:-2,np.newaxis] * pyom.cosu[np.newaxis,2:-2,np.newaxis]) \
                + pyom.tantr[np.newaxis,3:-1,np.newaxis] * (pyom.u[2:-2,3:-1,:,pyom.tau] + pyom.u[1:-3,3:-1,:,pyom.tau])**2 \
                * pyom.dyt[np.newaxis,3:-1,np.newaxis] * pyom.cost[np.newaxis,3:-1,np.newaxis] \
                / (pyom.dyu[np.newaxis,2:-2,np.newaxis] * pyom.cosu[np.newaxis,2:-2,np.newaxis]))

    """
    non hydrostatic Coriolis terms, metric terms are neglected
    """
    if not pyom.enable_hydrostatic:
        pyom.du_cor[2:-2, :, 1:] += -pyom.maskU[2:-2, :, 1:] * 0.25 \
                * (pyom.coriolis_h[2:-2,:,np.newaxis] * pyom.area_t[2:-2,:,np.newaxis] \
                * (pyom.w[2:-2,:,1:,pyom.tau] + pyom.w[2:-2,:,:-1,pyom.tau]) \
                + pyom.coriolis_h[3:-1,:,np.newaxis] * pyom.area_t[3:-1,:,np.newaxis] \
                * (pyom.w[3:-1,:,1:,pyom.tau] + pyom.w[3:-1,:,:-1,pyom.tau])) \
                / pyom.area_u[2:-2,:,np.newaxis]
        pyom.du_cor[2:-2,:,0] += -pyom.maskU[2:-2,:,0] * 0.25 \
                * (pyom.coriolis_h[2:-2] * pyom.area_t[2:-2] * (pyom.w[2:-2,:,0,pyom.tau]) \
                 + pyom.coriolis_h[3:-1] * pyom.area_t[3:-1] * (pyom.w[3:-1,:,0,pyom.tau])) \
                 / pyom.area_u[2:-2]
        pyom.dw_cor[2:-2,:,:-1] = pyom.maskW[2:-2,:,:-1] * 0.25 \
                * (pyom.coriolis_h[2:-2,:,np.newaxis] * pyom.dzt[np.newaxis,np.newaxis,:-1] \
                * (pyom.u[2:-2,:,:-1,pyom.tau] + pyom.u[1:-3,:,:-1,pyom.tau]) \
                +  pyom.coriolis_h[2:-2,:,np.newaxis] * pyom.dzt[np.newaxis,np.newaxis,1:] \
                * (pyom.u[2:-2,:,1:,pyom.tau] + pyom.u[1:-3,:,1:,pyom.tau])) \
                / pyom.dzw[np.newaxis,np.newaxis,:-1]

    """
    transfer to time tendencies
    """
    pyom.du[2:-2,2:-2,:,pyom.tau] = pyom.du_cor[2:-2,2:-2]
    pyom.dv[2:-2,2:-2,:,pyom.tau] = pyom.dv_cor[2:-2,2:-2]

    if not pyom.enable_hydrostatic:
        pyom.dw[2:-2,2:-2,:,pyom.tau] = pyom.dw_cor[2:-2,2:-2]
        #for j in xrange(pyom.js_pe, pyom.je_pe): #j = js_pe,je_pe
        #    for i in xrange(pyom.is_pe, pyom.ie_pe): #i = is_pe,ie_pe
        #        pyom.dw[i,j,:,pyom.tau] = pyom.dw_cor[i,j,:]

    """
    wind stress forcing
    """
    pyom.du[2:-2,2:-2,-1,pyom.tau] += pyom.maskU[2:-2,2:-2,-1] * pyom.surface_taux[2:-2,2:-2] / pyom.dzt[-1]
    pyom.dv[2:-2,2:-2,-1,pyom.tau] += pyom.maskV[2:-2,2:-2,-1] * pyom.surface_tauy[2:-2,2:-2] / pyom.dzt[-1]

    """
    advection
    """
    momentum_advection(pyom)
    pyom.du[:,:,:,pyom.tau] += pyom.du_adv
    pyom.dv[:,:,:,pyom.tau] += pyom.dv_adv
    if not pyom.enable_hydrostatic:
        pyom.dw[:,:,:,pyom.tau] += pyom.dw_adv

    with pyom.timers["friction"]:
        """
        vertical friction
        """
        pyom.K_diss_v[...] = 0.0
        if pyom.enable_implicit_vert_friction:
            friction.implicit_vert_friction(pyom)
        if pyom.enable_explicit_vert_friction:
            friction.explicit_vert_friction(pyom)

        """
        TEM formalism for eddy-driven velocity
        """
        if pyom.enable_TEM_friction:
            isoneutral.isoneutral_friction(pyom)

        """
        horizontal friction
        """
        if pyom.enable_hor_friction:
            friction.harmonic_friction(pyom)
        if pyom.enable_biharmonic_friction:
            friction.biharmonic_friction(pyom)

        """
        Rayleigh and bottom friction
        """
        pyom.K_diss_bot[...] = 0.0
        if pyom.enable_ray_friction:
            friction.rayleigh_friction(pyom)
        if pyom.enable_bottom_friction:
            friction.linear_bottom_friction(pyom)
        if pyom.enable_quadratic_bottom_friction:
            friction.quadratic_bottom_friction(pyom)

        """
        add user defined forcing
        """
        if pyom.enable_momentum_sources:
            friction.momentum_sources(pyom)

    """
    external mode
    """
    with pyom.timers["pressure"]:
        if pyom.enable_streamfunction:
            external.solve_streamfunction(pyom)
        else:
            external.solve_pressure(pyom)
            if pyom.itt == 0:
                pyom.psi[:,:,pyom.tau] = pyom.psi[:,:,pyom.taup1]
                pyom.psi[:,:,pyom.taum1] = pyom.psi[:,:,pyom.taup1]
        if not pyom.enable_hydrostatic:
            non_hydrostytic.solve_non_hydrostatic(pyom)

@pyom_method
def vertical_velocity(pyom):
    """
           vertical velocity from continuity :
           \int_0^z w_z dz = w(z)-w(0) = - \int dz (u_x + v_y)
           w(z) = -int dz u_x + v_y
    """
    fxa = np.empty((pyom.nx+3, pyom.ny+3, pyom.nz))
    # integrate from bottom to surface to see error in w
    fxa[:,:,0] = -pyom.maskW[1:,1:,0] * pyom.dzt[0] * \
          ((pyom.u[1:,1:,0,pyom.taup1]-pyom.u[:-1,1:,0,pyom.taup1])/(pyom.cost[np.newaxis,1:]*pyom.dxt[1:,np.newaxis]) \
          + (pyom.cosu[np.newaxis,1:] * pyom.v[1:,1:,0,pyom.taup1] - pyom.cosu[np.newaxis,:-1] * pyom.v[1:,:-1,0,pyom.taup1]) / (pyom.cost[np.newaxis, 1:] * pyom.dyt[np.newaxis, 1:]))
    fxa[:,:,1:] = -pyom.maskW[1:,1:,1:] * pyom.dzt[np.newaxis, np.newaxis, 1:] \
                    * ((pyom.u[1:,1:,1:,pyom.taup1] - pyom.u[:-1,1:,1:,pyom.taup1]) \
                    / (pyom.cost[np.newaxis, 1:, np.newaxis] * pyom.dxt[1:, np.newaxis, np.newaxis]) \
                    + (pyom.cosu[np.newaxis, 1:, np.newaxis] * pyom.v[1:,1:,1:,pyom.taup1] \
                    - pyom.cosu[np.newaxis, :-1, np.newaxis] * pyom.v[1:,:-1,1:,pyom.taup1]) \
                    / (pyom.cost[np.newaxis, 1:, np.newaxis] * pyom.dyt[np.newaxis, 1:, np.newaxis]))
    pyom.w[1:,1:,:,pyom.taup1] = np.cumsum(fxa, axis=2)

@pyom_method
def momentum_advection(pyom):
    """
    Advection of momentum with second order which is energy conserving
    """

    """
    Code from MITgcm
    """
    utr = pyom.dzt[np.newaxis, np.newaxis, :] * pyom.dyt[np.newaxis, :, np.newaxis] * pyom.u[...,pyom.tau] * pyom.maskU
    vtr = pyom.dzt[np.newaxis, np.newaxis, :] * pyom.cosu[np.newaxis, :, np.newaxis] * pyom.dxt[:, np.newaxis, np.newaxis] * pyom.v[...,pyom.tau] * pyom.maskV
    wtr = pyom.w[...,pyom.tau] * pyom.maskW * pyom.area_t[:,:,np.newaxis]

    """
    for zonal momentum
    """
    pyom.flux_east[1:-2, 2:-2] = 0.25*(pyom.u[1:-2,2:-2,:,pyom.tau]+pyom.u[2:-1,2:-2,:,pyom.tau])*(utr[2:-1,2:-2]+utr[1:-2,2:-2])
    pyom.flux_north[2:-2,1:-2] = 0.25*(pyom.u[2:-2,1:-2,:,pyom.tau]+pyom.u[2:-2,2:-1,:,pyom.tau])*(vtr[3:-1,1:-2]+vtr[2:-2,1:-2])
    pyom.flux_top[2:-2,2:-2,:-1] = 0.25*(pyom.u[2:-2,2:-2,1:,pyom.tau]+pyom.u[2:-2,2:-2,:-1,pyom.tau])*(wtr[2:-2,2:-2,:-1]+wtr[3:-1,2:-2,:-1])
    pyom.flux_top[:,:,-1] = 0.0
    pyom.du_adv[2:-2,2:-2] = -pyom.maskU[2:-2,2:-2] * (pyom.flux_east[2:-2,2:-2] - pyom.flux_east[1:-3,2:-2] \
            + pyom.flux_north[2:-2,2:-2]-pyom.flux_north[2:-2,1:-3])/(pyom.dzt[np.newaxis,np.newaxis,:]*pyom.area_u[2:-2,2:-2,np.newaxis])

    pyom.du_adv[:,:,0] += -pyom.maskU[:,:,0] * pyom.flux_top[:,:,0] / (pyom.area_u[:,:] * pyom.dzt[0])
    pyom.du_adv[:,:,1:] += -pyom.maskU[:,:,1:] * (pyom.flux_top[:,:,1:] - pyom.flux_top[:,:,:-1]) /(pyom.dzt[1:] * pyom.area_u[:,:,np.newaxis])

    """
    for meridional momentum
    """
    pyom.flux_east[1:-2, 2:-2] = 0.25 *(pyom.v[1:-2,2:-2,:,pyom.tau]+pyom.v[2:-1,2:-2,:,pyom.tau])*(utr[1:-2,3:-1]+utr[1:-2,2:-2])
    pyom.flux_north[2:-2, 1:-2] = 0.25 * (pyom.v[2:-2,1:-2,:,pyom.tau]+pyom.v[2:-2,2:-1,:,pyom.tau])*(vtr[2:-2,2:-1]+vtr[2:-2,1:-2])
    pyom.flux_top[2:-2,2:-2,:-1] = 0.25*(pyom.v[2:-2,2:-2,1:,pyom.tau]+pyom.v[2:-2,2:-2,:-1,pyom.tau])*(wtr[2:-2,2:-2,:-1]+wtr[2:-2,3:-1,:-1])
    pyom.flux_top[:,:,pyom.nz-1] = 0.0
    pyom.dv_adv[2:-2,2:-2] = -pyom.maskV[2:-2,2:-2]*(pyom.flux_east[2:-2,2:-2] - pyom.flux_east[1:-3,2:-2] \
            + pyom.flux_north[2:-2,2:-2] - pyom.flux_north[2:-2,1:-3])/(pyom.dzt*pyom.area_v[2:-2,2:-2,np.newaxis])
    tmp = pyom.dzt * pyom.area_v[:,:,np.newaxis]
    pyom.dv_adv[:,:,0] += -pyom.maskV[:,:,0]*pyom.flux_top[:,:,0]/tmp[:,:,0]
    pyom.dv_adv[:,:,1:] += -pyom.maskV[:,:,1:]*(pyom.flux_top[:,:,1:]-pyom.flux_top[:,:,:-1])/tmp[:,:,1:]

    if not pyom.enable_hydrostatic:
        """
        for vertical momentum
        """
        pyom.flux_east[2:-2, 2:-2, :-1] = 0.5 * (pyom.w[2:-2, 2:-2, :-1, pyom.tau] + pyom.w[3:-1, 2:-2, :-1, pyom.tau])\
                * (pyom.u[2:-2, 2:-2, :-1,pyom.tau] + pyom.u[2:-2,2:-2,1:,pyom.tau])*0.5*pyom.maskW[3:-1,2:-2,:-1]*pyom.maskW[2:-2,2:-2,:-1]
        pyom.flux_east[2:-2, 2:-2, -1] = 0.5 * (pyom.w[2:-2, 2:-2, -1,pyom.tau] + pyom.w[3:-1, 2:-2, -1, pyom.tau])\
                * (pyom.u[2:-2, 2:-2, -1, pyom.tau] + pyom.u[2:-2, 2:-2, -1, pyom.tau]) * 0.5 * pyom.maskW[3:-1, 2:-2, -1] * pyom.maskW[2:-2, 2:-2, -1]
        pyom.flux_north[2:-2,2:-2,:-1] = 0.5*(pyom.w[2:-2,2:-2,:-1,pyom.tau] + pyom.w[2:-2,3:-1,:-1,pyom.tau]) * \
                (pyom.v[2:-2,2:-2,:-1,pyom.tau]+pyom.v[2:-2,2:-2,1:,pyom.tau]) * \
                0.5*pyom.maskW[2:-2,3:-1,:-1]*pyom.maskW[2:-2,2:-2,:-1]*pyom.cosu[np.newaxis,2:-2,np.newaxis]
        pyom.flux_north[2:-2,2:-2,-1] = 0.5*(pyom.w[2:-2,2:-2,-1,pyom.tau] + pyom.w[2:-2,3:-1,-1,pyom.tau]) * \
                (pyom.v[2:-2,2:-2,-1,pyom.tau]+pyom.v[2:-2,2:-2,-1,pyom.tau]) * \
                0.5*pyom.maskW[2:-2,3:-1,-1]*pyom.maskW[2:-2,2:-2,-1]*pyom.cosu[2:-2]
        pyom.flux_top[2:-2,2:-2,:-1] = 0.5*(pyom.w[2:-2,2:-2,1:,pyom.tau]+pyom.w[2:-2,2:-2,:-1,pyom.tau])\
                *(pyom.w[2:-2,2:-2,1:,pyom.tau]+pyom.w[2:-2,2:-2,:-1,pyom.tau])\
                *0.5*pyom.maskW[2:-2,2:-2,1:]*pyom.maskW[2:-2,2:-2,:-1]
        pyom.flux_top[:,:,-1] = 0.
        pyom.dw_adv[2:-2,2:-2] = pyom.maskW[2:-2,2:-2] * (-(pyom.flux_east[2:-2,2:-2] - pyom.flux_east[1:-3,2:-2])/(pyom.cost[np.newaxis,2:-2,np.newaxis]*pyom.dxt[2:-2,np.newaxis,np.newaxis])\
                -(pyom.flux_north[2:-2,2:-2] - pyom.flux_north[2:-2,1:-3])/(pyom.cost[np.newaxis,2:-2,np.newaxis]*pyom.dyt[np.newaxis,2:-2,np.newaxis]))
        tmp = pyom.maskW / pyom.dzw
        pyom.dw_adv[:,:,0] -= tmp[:,:,0] * pyom.flux_top[:,:,0]
        pyom.dw_adv[:,:,1:] -= tmp[:,:,1:] * (pyom.flux_top[:,:,1:] - pyom.flux_top[:,:,:-1])

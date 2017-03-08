from climate.pyom import friction, isoneutral, external, non_hydrostatic, pyom_method

@pyom_method
def momentum(pyom):
    """
    solve for momentum for taup1
    """

    """
    time tendency due to Coriolis force
    """
    pyom.du_cor[2:-2, 2:-2] = pyom.maskU[2:-2, 2:-2] \
            * (pyom.coriolis_t[2:-2, 2:-2,None] * (pyom.v[2:-2, 2:-2,:,pyom.tau] + pyom.v[2:-2, 1:-3,:,pyom.tau]) \
            * pyom.dxt[2:-2,None,None] / pyom.dxu[2:-2,None,None] \
            + pyom.coriolis_t[3:-1,2:-2,None] * (pyom.v[3:-1,2:-2,:,pyom.tau] + pyom.v[3:-1,1:-3,:,pyom.tau]) \
            * pyom.dxt[3:-1,None,None] / pyom.dxu[2:-2,None,None]) * 0.25
    pyom.dv_cor[2:-2, 2:-2] = -pyom.maskV[2:-2, 2:-2]\
            * (pyom.coriolis_t[2:-2, 2:-2,None] * (pyom.u[1:-3, 2:-2,:,pyom.tau] + pyom.u[2:-2, 2:-2,:,pyom.tau]) \
            * pyom.dyt[None,2:-2,None] * pyom.cost[None,2:-2,None] \
            / (pyom.dyu[None,2:-2,None] * pyom.cosu[None,2:-2,None]) \
            + pyom.coriolis_t[2:-2,3:-1,None] * (pyom.u[1:-3, 3:-1,:,pyom.tau] + pyom.u[2:-2,3:-1,:,pyom.tau]) \
            * pyom.dyt[None,3:-1,None] * pyom.cost[None,3:-1,None] \
            / (pyom.dyu[None,2:-2,None] * pyom.cosu[None,2:-2,None])) * 0.25

    """
    time tendency due to metric terms
    """
    if pyom.coord_degree:
        pyom.du_cor[2:-2, 2:-2] += pyom.maskU[2:-2, 2:-2] * 0.125 * pyom.tantr[None,2:-2,None] \
                * ((pyom.u[2:-2, 2:-2,:,pyom.tau] + pyom.u[1:-3, 2:-2,:,pyom.tau]) \
                 * (pyom.v[2:-2, 2:-2,:,pyom.tau] + pyom.v[2:-2, 1:-3,:,pyom.tau]) \
                 * pyom.dxt[2:-2,None,None] / pyom.dxu[2:-2,None,None] \
                 + (pyom.u[3:-1, 2:-2,:,pyom.tau] + pyom.u[2:-2, 2:-2,:,pyom.tau]) \
                 * (pyom.v[3:-1, 2:-2,:,pyom.tau] + pyom.v[3:-1, 1:-3,:,pyom.tau]) \
                 * pyom.dxt[3:-1,None,None] / pyom.dxu[2:-2,None,None])
        pyom.dv_cor[2:-2, 2:-2] += -pyom.maskV[2:-2, 2:-2] * 0.125 \
                * (pyom.tantr[None,2:-2,None] * (pyom.u[2:-2,2:-2,:,pyom.tau] + pyom.u[1:-3,2:-2,:,pyom.tau])**2 \
                * pyom.dyt[None,2:-2,None] * pyom.cost[None,2:-2,None] \
                / (pyom.dyu[None,2:-2,None] * pyom.cosu[None,2:-2,None]) \
                + pyom.tantr[None,3:-1,None] * (pyom.u[2:-2,3:-1,:,pyom.tau] + pyom.u[1:-3,3:-1,:,pyom.tau])**2 \
                * pyom.dyt[None,3:-1,None] * pyom.cost[None,3:-1,None] \
                / (pyom.dyu[None,2:-2,None] * pyom.cosu[None,2:-2,None]))

    """
    non hydrostatic Coriolis terms, metric terms are neglected
    """
    if not pyom.enable_hydrostatic:
        pyom.du_cor[2:-2, :, 1:] += -pyom.maskU[2:-2, :, 1:] * 0.25 \
                * (pyom.coriolis_h[2:-2,:,None] * pyom.area_t[2:-2,:,None] \
                * (pyom.w[2:-2,:,1:,pyom.tau] + pyom.w[2:-2,:,:-1,pyom.tau]) \
                + pyom.coriolis_h[3:-1,:,None] * pyom.area_t[3:-1,:,None] \
                * (pyom.w[3:-1,:,1:,pyom.tau] + pyom.w[3:-1,:,:-1,pyom.tau])) \
                / pyom.area_u[2:-2,:,None]
        pyom.du_cor[2:-2,:,0] += -pyom.maskU[2:-2,:,0] * 0.25 \
                * (pyom.coriolis_h[2:-2] * pyom.area_t[2:-2] * (pyom.w[2:-2,:,0,pyom.tau]) \
                 + pyom.coriolis_h[3:-1] * pyom.area_t[3:-1] * (pyom.w[3:-1,:,0,pyom.tau])) \
                 / pyom.area_u[2:-2]
        pyom.dw_cor[2:-2,:,:-1] = pyom.maskW[2:-2,:,:-1] * 0.25 \
                * (pyom.coriolis_h[2:-2,:,None] * pyom.dzt[None,None,:-1] \
                * (pyom.u[2:-2,:,:-1,pyom.tau] + pyom.u[1:-3,:,:-1,pyom.tau]) \
                +  pyom.coriolis_h[2:-2,:,None] * pyom.dzt[None,None,1:] \
                * (pyom.u[2:-2,:,1:,pyom.tau] + pyom.u[1:-3,:,1:,pyom.tau])) \
                / pyom.dzw[None,None,:-1]

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
          ((pyom.u[1:,1:,0,pyom.taup1]-pyom.u[:-1,1:,0,pyom.taup1])/(pyom.cost[None,1:]*pyom.dxt[1:,None]) \
          + (pyom.cosu[None,1:] * pyom.v[1:,1:,0,pyom.taup1] - pyom.cosu[None,:-1] * pyom.v[1:,:-1,0,pyom.taup1]) / (pyom.cost[None, 1:] * pyom.dyt[None, 1:]))
    fxa[:,:,1:] = -pyom.maskW[1:,1:,1:] * pyom.dzt[None, None, 1:] \
                    * ((pyom.u[1:,1:,1:,pyom.taup1] - pyom.u[:-1,1:,1:,pyom.taup1]) \
                    / (pyom.cost[None, 1:, None] * pyom.dxt[1:, None, None]) \
                    + (pyom.cosu[None, 1:, None] * pyom.v[1:,1:,1:,pyom.taup1] \
                    - pyom.cosu[None, :-1, None] * pyom.v[1:,:-1,1:,pyom.taup1]) \
                    / (pyom.cost[None, 1:, None] * pyom.dyt[None, 1:, None]))
    pyom.w[1:,1:,:,pyom.taup1] = np.cumsum(fxa, axis=2)

@pyom_method
def momentum_advection(pyom):
    """
    Advection of momentum with second order which is energy conserving
    """

    """
    Code from MITgcm
    """
    #        uTrans(i,j) = u(i,j)*dyG(i,j)*drF(k)
    #        vTrans(i,j) = v(i,j)*dxG(i,j)*drF(k)

    #        fZon(i,j) = 0.25*(uTrans(i,j) + uTrans(i+1,j)) *(u(i,j) + u(i+1,j))
    #        fMer(i,j) = 0.25*(vTrans(i,j) + vTrans(i-1,j)) *(u(i,j) + u(i,j-1))
    #
    #          gU(i,j,k,bi,bj) = -
    #     &     *((fZon(i,j)  - fZon(i-1,j))
    #     &       +(fMer(i,j+1)  - fMer(i,  j))
    #     &       +(fVerUkp(i,j) - fVerUkm(i,j))
    #     &) /drF(k)   / rAw(i,j)


    #        fZon(i,j) = 0.25*(uTrans(i,j) + uTrans(i,j-1))  *(v(i,j) + v(i-1,j))
    #        fMer(i,j) = 0.25*(vTrans(i,j) + vTrans(i,j+1))  *(v(i,j) +  v(i,j+1))

    #          gV(i,j,k,bi,bj) = -recip_drF(k)*recip_rAs(i,j,bi,bj)
    #     &     *((fZon(i+1,j)  - fZon(i,j))
    #     &       +(fMer(i,  j)  - fMer(i,j-1))
    #     &       +(fVerVkp(i,j) - fVerVkm(i,j))
    #     &)

    utr = pyom.dzt[None, None, :] * pyom.dyt[None, :, None] * pyom.u[...,pyom.tau] * pyom.maskU
    vtr = pyom.dzt[None, None, :] * pyom.cosu[None, :, None] * pyom.dxt[:, None, None] * pyom.v[...,pyom.tau] * pyom.maskV
    wtr = pyom.w[...,pyom.tau] * pyom.maskW * pyom.area_t[:,:,None]

    """
    for zonal momentum
    """
    pyom.flux_east[1:-2, 2:-2] = 0.25*(pyom.u[1:-2,2:-2,:,pyom.tau]+pyom.u[2:-1,2:-2,:,pyom.tau])*(utr[2:-1,2:-2]+utr[1:-2,2:-2])
    pyom.flux_north[2:-2,1:-2] = 0.25*(pyom.u[2:-2,1:-2,:,pyom.tau]+pyom.u[2:-2,2:-1,:,pyom.tau])*(vtr[3:-1,1:-2]+vtr[2:-2,1:-2])
    pyom.flux_top[2:-2,2:-2,:-1] = 0.25*(pyom.u[2:-2,2:-2,1:,pyom.tau]+pyom.u[2:-2,2:-2,:-1,pyom.tau])*(wtr[2:-2,2:-2,:-1]+wtr[3:-1,2:-2,:-1])
    pyom.flux_top[:,:,-1] = 0.0
    pyom.du_adv[2:-2,2:-2] = -pyom.maskU[2:-2,2:-2] * (pyom.flux_east[2:-2,2:-2] - pyom.flux_east[1:-3,2:-2] \
            + pyom.flux_north[2:-2,2:-2]-pyom.flux_north[2:-2,1:-3])/(pyom.dzt[None,None,:]*pyom.area_u[2:-2,2:-2,None])

    pyom.du_adv[:,:,0] += -pyom.maskU[:,:,0] * pyom.flux_top[:,:,0] / (pyom.area_u[:,:] * pyom.dzt[0])
    pyom.du_adv[:,:,1:] += -pyom.maskU[:,:,1:] * (pyom.flux_top[:,:,1:] - pyom.flux_top[:,:,:-1]) /(pyom.dzt[1:] * pyom.area_u[:,:,None])

    """
    for meridional momentum
    """
    pyom.flux_east[1:-2, 2:-2] = 0.25 *(pyom.v[1:-2,2:-2,:,pyom.tau]+pyom.v[2:-1,2:-2,:,pyom.tau])*(utr[1:-2,3:-1]+utr[1:-2,2:-2])
    pyom.flux_north[2:-2, 1:-2] = 0.25 * (pyom.v[2:-2,1:-2,:,pyom.tau]+pyom.v[2:-2,2:-1,:,pyom.tau])*(vtr[2:-2,2:-1]+vtr[2:-2,1:-2])
    pyom.flux_top[2:-2,2:-2,:-1] = 0.25*(pyom.v[2:-2,2:-2,1:,pyom.tau]+pyom.v[2:-2,2:-2,:-1,pyom.tau])*(wtr[2:-2,2:-2,:-1]+wtr[2:-2,3:-1,:-1])
    pyom.flux_top[:,:,pyom.nz-1] = 0.0
    pyom.dv_adv[2:-2,2:-2] = -pyom.maskV[2:-2,2:-2]*(pyom.flux_east[2:-2,2:-2] - pyom.flux_east[1:-3,2:-2] \
            + pyom.flux_north[2:-2,2:-2] - pyom.flux_north[2:-2,1:-3])/(pyom.dzt*pyom.area_v[2:-2,2:-2,None])
    tmp = pyom.dzt * pyom.area_v[:,:,None]
    pyom.dv_adv[:,:,0] += -pyom.maskV[:,:,0]*pyom.flux_top[:,:,0]/tmp[:,:,0]
    pyom.dv_adv[:,:,1:] += -pyom.maskV[:,:,1:]*(pyom.flux_top[:,:,1:]-pyom.flux_top[:,:,:-1])/tmp[:,:,1:]

    if not pyom.enable_hydrostatic:
        """
        for vertical momentum
        """
        pyom.flux_east[2:-2,2:-2,:-1] = 0.5*(pyom.w[2:-2,2:-2,:-1,pyom.tau]+pyom.w[3:-1,2:-2,:-1,pyom.tau])\
                *(pyom.u[2:-2,2:-2,:-1,pyom.tau] + pyom.u[2:-2,2:-2,1:,pyom.tau])*0.5*pyom.maskW[3:-1,2:-2,:-1]*pyom.maskW[2:-2,2:-2,:-1]
        pyom.flux_east[2:-2,2:-2,-1] = 0.5*(pyom.w[2:-2,2:-2,-1,pyom.tau]+pyom.w[3:-1,2:-2,-1,pyom.tau])\
                *(pyom.u[2:-2,2:-2,-1,pyom.tau] + pyom.u[2:-2,2:-2,-1,pyom.tau])*0.5*pyom.maskW[3:-1,2:-2,-1]*pyom.maskW[2:-2,2:-2,-1]
        pyom.flux_north[2:-2,2:-2,:-1] = 0.5*(pyom.w[2:-2,2:-2,:-1,pyom.tau] + pyom.w[2:-2,3:-1,:-1,pyom.tau]) * \
                (pyom.v[2:-2,2:-2,:-1,pyom.tau]+pyom.v[2:-2,2:-2,1:,pyom.tau]) * \
                0.5*pyom.maskW[2:-2,3:-1,:-1]*pyom.maskW[2:-2,2:-2,:-1]*pyom.cosu[None,2:-2,None]
        pyom.flux_north[2:-2,2:-2,-1] = 0.5*(pyom.w[2:-2,2:-2,-1,pyom.tau] + pyom.w[2:-2,3:-1,-1,pyom.tau]) * \
                (pyom.v[2:-2,2:-2,-1,pyom.tau]+pyom.v[2:-2,2:-2,-1,pyom.tau]) * \
                0.5*pyom.maskW[2:-2,3:-1,-1]*pyom.maskW[2:-2,2:-2,-1]*pyom.cosu[2:-2]
        pyom.flux_top[2:-2,2:-2,:-1] = 0.5*(pyom.w[2:-2,2:-2,1:,pyom.tau]+pyom.w[2:-2,2:-2,:-1,pyom.tau])\
                *(pyom.w[2:-2,2:-2,1:,pyom.tau]+pyom.w[2:-2,2:-2,:-1,pyom.tau])\
                *0.5*pyom.maskW[2:-2,2:-2,1:]*pyom.maskW[2:-2,2:-2,:-1]
        pyom.flux_top[:,:,-1] = 0.
        pyom.dw_adv[2:-2,2:-2] = pyom.maskW[2:-2,2:-2] * (-(pyom.flux_east[2:-2,2:-2] - pyom.flux_east[1:-3,2:-2])/(pyom.cost[None,2:-2,None]*pyom.dxt[2:-2,None,None])\
                -(pyom.flux_north[2:-2,2:-2] - pyom.flux_north[2:-2,1:-3])/(pyom.cost[None,2:-2,None]*pyom.dyt[None,2:-2,None]))
        tmp = pyom.maskW / pyom.dzw
        pyom.dw_adv[:,:,0] -= tmp[:,:,0] * pyom.flux_top[:,:,0]
        pyom.dw_adv[:,:,1:] -= tmp[:,:,1:] * (pyom.flux_top[:,:,1:] - pyom.flux_top[:,:,:-1])

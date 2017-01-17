import numpy as np

from climate.pyom import friction, isoneutral, external

def momentum(pyom):
    """
    solve for momentum for taup1
    """
    #implicit none
    #integer :: i,j,k

    """
    time tendency due to Coriolis force
    """
    for j in xrange(pyom.js_pe, pyom.je_pe): #j = js_pe,je_pe
        for i in xrange(pyom.is_pe, pyom.ie_pe): #i = is_pe,ie_pe
            pyom.du_cor[i,j,:] = pyom.maskU[i,j,:]*(pyom.coriolis_t[i,j]*(pyom.v[i,j,:,pyom.tau]+pyom.v[i,j-1,:,pyom.tau])*pyom.dxt[i]/pyom.dxu[i] \
                                          +pyom.coriolis_t[i+1,j]*(pyom.v[i+1,j,:,pyom.tau]+pyom.v[i+1,j-1,:,pyom.tau])*pyom.dxt[i+1]/pyom.dxu[i])*0.25
            pyom.dv_cor[i,j,:] = -pyom.maskV[i,j,:]*(pyom.coriolis_t[i,j]*(pyom.u[i-1,j,:,pyom.tau]+pyom.u[i,j,:,pyom.tau])*pyom.dyt[j]*pyom.cost[j]/(pyom.dyu[j]*pyom.cosu[j]) \
                                        + pyom.coriolis_t[i,j+1]*(pyom.u[i-1,j+1,:,pyom.tau]+pyom.u[i,j+1,:,pyom.tau])*pyom.dyt[j+1]*pyom.cost[j+1]/(pyom.dyu[j]*pyom.cosu[j]))*0.25

    """
    time tendency due to metric terms
    """
    if pyom.coord_degree:
        for j in xrange(pyom.js_pe, pyom.je_pe): #j = js_pe,je_pe
            for i in xrange(pyom.is_pe, pyom.ie_pe): #i = is_pe,ie_pe
                pyom.du_cor[i,j,:] += pyom.maskU[i,j,:]*0.125*pyom.tantr[j]*(\
                                  (pyom.u[i,j,:,pyom.tau]+pyom.u[i-1,j,:,pyom.tau])*(pyom.v[i,j,:,pyom.tau]+pyom.v[i,j-1,:,pyom.tau])*pyom.dxt[i]/pyom.dxu[i] \
                                + (pyom.u[i+1,j,:,pyom.tau]+pyom.u[i,j,:,pyom.tau])*(pyom.v[i+1,j,:,pyom.tau]+pyom.v[i+1,j-1,:,pyom.tau])*pyom.dxt[i+1]/pyom.dxu[i])
                pyom.dv_cor[i,j,:] -= pyom.maskV[i,j,:]*0.125*(\
                                       pyom.tantr[j]*(pyom.u[i,j,:,pyom.tau]+pyom.u[i-1,j,:,pyom.tau])**2*pyom.dyt[j]*pyom.cost[j]/(pyom.dyu[j]*pyom.cosu[j]) \
                                     + pyom.tantr[j+1]*(pyom.u[i,j+1,:,pyom.tau]+pyom.u[i-1,j+1,:,pyom.tau])**2*pyom.dyt[j+1]*pyom.cost[j+1]/(pyom.dyu[j]*pyom.cosu[j]))

    """
    non hydrostatic Coriolis terms, metric terms are neglected
    """
    if not pyom.enable_hydrostatic:
        for k in xrange(1, pyom.nz): #k = 2,nz
            for i in xrange(pyom.is_pe, pyom.ie_pe): #i = is_pe,ie_pe
                pyom.du_cor[i,:,k] -= pyom.maskU[i,:,k]*0.25*(pyom.coriolis_h[i,:]*pyom.area_t[i,:]*(pyom.w[i,:,k,pyom.tau]+pyom.w[i,:,k-1,pyom.tau]) \
                                                   + pyom.coriolis_h[i+1,:]*pyom.area_t[i+1,:]*(pyom.w[i+1,:,k,pyom.tau]+pyom.w[i+1,:,k-1,pyom.tau])) \
                                                     / pyom.area_u[i,:]
        k = 0
        for i in xrange(pyom.is_pe, pyom.ie_pe): #i = is_pe,ie_pe
            pyom.du_cor[i,:,k] -= pyom.maskU[i,:,k]*0.25*(pyom.coriolis_h[i,:]*pyom.area_t[i,:]*(pyom.w[i,:,k,pyom.tau]) \
                                               + pyom.coriolis_h[i+1,:]*pyom.area_t[i+1,:]*(pyom.w[i+1,:,k,pyom.tau]))/pyom.area_u[i,:]
        for k in xrange(nz-1): #k = 1,nz-1
            for i in xrange(pyom.is_pe, pyom.ie_pe): #i = is_pe,ie_pe
                pyom.dw_cor[i,:,k] = pyom.maskW[i,:,k]*0.25*(pyom.coriolis_h[i,:]*pyom.dzt[k]*(pyom.u[i,:,k,pyom.tau]+pyom.u[i-1,:,k,pyom.tau]) \
                                                  + pyom.coriolis_h[i,:]*pyom.dzt[k+1]*(pyom.u[i,:,k+1,pyom.tau]+pyom.u[i-1,:,k+1,pyom.tau]))/pyom.dzw[k]

    """
    transfer to time tendencies
    """
    for j in xrange(pyom.js_pe, pyom.je_pe): #j = js_pe,je_pe
        for i in xrange(pyom.is_pe, pyom.ie_pe): #i = is_pe,ie_pe
            pyom.du[i,j,:,pyom.tau] = pyom.du_cor[i,j,:]
            pyom.dv[i,j,:,pyom.tau] = pyom.dv_cor[i,j,:]

    if not pyom.enable_hydrostatic:
        for j in xrange(pyom.js_pe, pyom.je_pe): #j = js_pe,je_pe
            for i in xrange(pyom.is_pe, pyom.ie_pe): #i = is_pe,ie_pe
                pyom.dw[i,j,:,pyom.tau] = pyom.dw_cor[i,j,:]

    """
    wind stress forcing
    """
    for j in xrange(pyom.js_pe, pyom.je_pe): #j = js_pe,je_pe
        for i in xrange(pyom.is_pe, pyom.ie_pe): #i = is_pe,ie_pe
            pyom.du[i,j,pyom.nz-1,pyom.tau] += pyom.maskU[i,j,pyom.nz-1]*pyom.surface_taux[i,j]/pyom.dzt[pyom.nz-1]
            pyom.dv[i,j,pyom.nz-1,pyom.tau] += pyom.maskV[i,j,pyom.nz-1]*pyom.surface_tauy[i,j]/pyom.dzt[pyom.nz-1]

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
    ---------------------------------------------------------------------------------
     external mode
    ---------------------------------------------------------------------------------
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
            #TODO: not implemented yet
            #call solve_non_hydrostatic
            raise NotImplementedError()

def vertical_velocity(pyom):
    """
           vertical velocity from continuity :
           \int_0^z w_z dz = w(z)-w(0) = - \int dz (u_x +v_y)
            w(z) = -int dz u_x + v_y
    """
    #integer :: i,j,k
    # integrate from bottom to surface to see error in w
    k = 0
    for j in xrange(pyom.js_pe-pyom.onx+1, pyom.je_pe+pyom.onx): #j = js_pe-onx+1,je_pe+onx
        for i in xrange(pyom.is_pe-pyom.onx+1, pyom.ie_pe+pyom.onx): #i = is_pe-onx+1,ie_pe+onx
            pyom.w[i,j,k,pyom.taup1] = -pyom.maskW[i,j,k]*pyom.dzt[k]* \
                  ((pyom.u[i,j,k,pyom.taup1]-pyom.u[i-1,j,k,pyom.taup1])/(pyom.cost[j]*pyom.dxt[i]) \
                  +(pyom.cosu[j]*pyom.v[i,j,k,pyom.taup1]-pyom.cosu[j-1]*pyom.v[i,j-1,k,pyom.taup1])/(pyom.cost[j]*pyom.dyt[j]))
    for k in xrange(1, pyom.nz): #k = 2,nz
        for j in xrange(pyom.js_pe-pyom.onx+1, pyom.je_pe+pyom.onx): #j = js_pe-onx+1,je_pe+onx
            for i in xrange(pyom.is_pe-onx+1, pyom.ie_pe+pyom.onx): #i = is_pe-onx+1,ie_pe+onx
                pyom.w[i,j,k,pyom.taup1] -= pyom.maskW[i,j,k]*pyom.dzt[k]* \
                                            ((u[i,j,k,pyom.taup1] - pyom.u[i-1,j,k,pyom.taup1])/(pyom.cost[j]*pyom.dxt[i]) \
                                            +(pyom.cosu[j]*pyom.v[i,j,k,pyom.taup1]-pyom.cosu[j-1]*pyom.v[i,j-1,k,pyom.taup1])/(pyom.cost[j]*pyom.dyt[j]))


def momentum_advection(pyom):
    """
    Advection of momentum with second order which is energy conserving
    """
    #integer :: i,j,k
    #real*8 :: utr(pyom.is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    #real*8 :: vtr(pyom.is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    #real*8 :: wtr(pyom.is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    utr = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    vtr = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    wtr = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

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


    for j in xrange(pyom.js_pe-pyom.onx, pyom.je_pe+pyom.onx): #j = js_pe-onx,je_pe+onx
        for i in xrange(pyom.is_pe-pyom.onx, pyom.ie_pe+pyom.onx): #i = is_pe-onx,ie_pe+onx
            utr[i,j,:] = pyom.dzt[:]*pyom.dyt[j]*pyom.u[i,j,:,pyom.tau]*pyom.maskU[i,j,:]
            vtr[i,j,:] = pyom.dzt[:]*pyom.cosu[j]*pyom.dxt[i]*pyom.v[i,j,:,pyom.tau]*pyom.maskV[i,j,:]
            wtr[i,j,:] = pyom.area_t[i,j]*pyom.w[i,j,:,pyom.tau]*pyom.maskW[i,j,:]

    """
    for zonal momentum
    """
    for j in xrange(pyom.js_pe, pyom.je_pe): #j = js_pe,je_pe
        for i in xrange(pyom.is_pe-1, pyom.ie_pe): #i = is_pe-1,ie_pe
            pyom.flux_east[i,j,:] = 0.25*(pyom.u[i,j,:,pyom.tau]+pyom.u[i+1,j,:,pyom.tau])*(utr[i+1,j,:]+utr[i,j,:])
    for j in xrange(pyom.js_pe-1, pyom.je_pe): #j = js_pe-1,je_pe
        for i in xrange(pyom.is_pe, pyom.ie_pe): #i = is_pe,ie_pe
            pyom.flux_north[i,j,:] = 0.25*(pyom.u[i,j,:,pyom.tau]+pyom.u[i,j+1,:,pyom.tau])*(vtr[i+1,j,:]+vtr[i,j,:])
    for k in xrange(pyom.nz-1): #k = 1,nz-1
        for j in xrange(pyom.js_pe, pyom.je_pe): #j = js_pe,je_pe
            for i in xrange(pyom.is_pe, pyom.ie_pe): #i = is_pe,ie_pe
                pyom.flux_top[i,j,k] = 0.25*(pyom.u[i,j,k+1,pyom.tau]+pyom.u[i,j,k,pyom.tau])*(wtr[i,j,k]+wtr[i+1,j,k])
    pyom.flux_top[:,:,pyom.nz-1] = 0.0
    for j in xrange(pyom.js_pe, pyom.je_pe): #j = js_pe,je_pe
        for i in xrange(pyom.is_pe, pyom.ie_pe): #i = is_pe,ie_pe
            pyom.du_adv[i,j,:] = -pyom.maskU[i,j,:] * (pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:] \
                                           + pyom.flux_north[i,j,:]-pyom.flux_north[i,j-1,:])/(pyom.area_u[i,j]*pyom.dzt[:])
    k = 0
    pyom.du_adv[:,:,k] -= pyom.maskU[:,:,k] * pyom.flux_top[:,:,k] / (pyom.area_u[:,:] * pyom.dzt[k])
    for k in xrange(1, pyom.nz): #k = 2,nz
        pyom.du_adv[:,:,k] -= pyom.maskU[:,:,k] * (pyom.flux_top[:,:,k] - pyom.flux_top[:,:,k-1]) / (pyom.area_u[:,:] * pyom.dzt[k])
    """
    for meridional momentum
    """
    for j in xrange(pyom.js_pe, pyom.je_pe): #j = js_pe,je_pe
        for i in xrange(pyom.is_pe-1, pyom.ie_pe): #i = is_pe-1,ie_pe
            pyom.flux_east[i,j,:] = 0.25*(pyom.v[i,j,:,pyom.tau]+pyom.v[i+1,j,:,pyom.tau])*(utr[i,j+1,:]+utr[i,j,:])
    for j in xrange(pyom.js_pe-1, pyom.je_pe): #j = js_pe-1,je_pe
        for i in xrange(pyom.is_pe, pyom.ie_pe): #i = is_pe,ie_pe
            pyom.flux_north[i,j,:] = 0.25*(pyom.v[i,j,:,pyom.tau]+pyom.v[i,j+1,:,pyom.tau])*(vtr[i,j+1,:]+vtr[i,j,:])
    for k in xrange(pyom.nz-1): #k = 1,nz-1
        for j in xrange(pyom.js_pe, pyom.je_pe): #j = js_pe,je_pe
            for i in xrange(pyom.is_pe, pyom.ie_pe): #i = is_pe,ie_pe
                pyom.flux_top[i,j,k] = 0.25*(pyom.v[i,j,k+1,pyom.tau]+pyom.v[i,j,k,pyom.tau])*(wtr[i,j,k]+wtr[i,j+1,k])
    pyom.flux_top[:,:,pyom.nz-1] = 0.0
    for j in xrange(pyom.js_pe, pyom.je_pe): #j = js_pe,je_pe
        for i in xrange(pyom.is_pe, pyom.ie_pe): #i = is_pe,ie_pe
            pyom.dv_adv[i,j,:] = -pyom.maskV[i,j,:]*(pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:] \
                                          + pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:])/(pyom.area_v[i,j]*pyom.dzt[:])
    k = 0
    pyom.dv_adv[:,:,k] -= pyom.maskV[:,:,k]*pyom.flux_top[:,:,k]/(pyom.area_v[:,:]*pyom.dzt[k])
    for k in xrange(1, pyom.nz): #k = 2,nz
        pyom.dv_adv[:,:,k] -= pyom.maskV[:,:,k]*(pyom.flux_top[:,:,k]-pyom.flux_top[:,:,k-1])/(pyom.area_v[:,:]*pyom.dzt[k])

    if not pyom.enable_hydrostatic:
        """
        for vertical momentum
        """
        for k in xrange(pyom.nz): #k = 1,nz
            for j in xrange(pyom.js_pe, pyom.je_pe): #j = js_pe,je_pe
                for i in xrange(pyom.is_pe-1, pyom.ie_pe): #i = is_pe-1,ie_pe
                    pyom.flux_east[i,j,k] = 0.5*(pyom.w[i,j,k,pyom.tau]+pyom.w[i+1,j,k,pyom.tau])*(pyom.u[i,j,k,pyom.tau] \
                                            + pyom.u[i,j,min(pyom.nz-1,k+1),pyom.tau])*0.5*pyom.maskW[i+1,j,k]*pyom.maskW[i,j,k]
        for k in xrange(pyom.nz): #k = 1,nz
            for j in xrange(pyom.js_pe-1, pyom.je_pe): #j = js_pe-1,je_pe
                for i in xrange(pyom.is_pe, pyom.ie_pe): #i = is_pe,ie_pe
                    pyom.flux_north[i,j,k] = 0.5*(pyom.w[i,j,k,pyom.tau] + pyom.w[i,j+1,k,pyom.tau]) * \
                                            (pyom.v[i,j,k,pyom.tau]+pyom.v[i,j,min(nz-1,k+1),pyom.tau]) * \
                                            0.5*pyom.maskW[i,j+1,k]*pyom.maskW[i,j,k]*pyom.cosu[j]
        for k in xrange(pyom.nz-1): #k = 1,nz-1
            for j in xrange(pyom.js_pe, pyom.je_pe): #j = js_pe,je_pe
                for i in xrange(pyom.is_pe, pyom.ie_pe): #i = is_pe,ie_pe
                    pyom.flux_top[i,j,k] = 0.5*(pyom.w[i,j,k+1,pyom.tau]+pyom.w[i,j,k,pyom.tau])*(pyom.w[i,j,k,pyom.tau] + \
                                           pyom.w[i,j,k+1,pyom.tau])*0.5*pyom.maskW[i,j,k+1]*pyom.maskW[i,j,k]
        pyom.flux_top[:,:,pyom.nz-1] = 0.0
        for j in xrange(pyom.js_pe, pyom.je_pe): #j = js_pe,je_pe
            for i in xrange(pyom.is_pe, pyom.ie_pe): #i = is_pe,ie_pe
                pyom.dw_adv[i,j,:] = pyom.maskW[i,j,:] * (-(pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:])/(pyom.cost[j]*pyom.dxt[i]) \
                                               -(pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:])/(pyom.cost[j]*pyom.dyt[j]))
        k = 0
        pyom.dw_adv[:,:,k] -= pyom.maskW[:,:,k]*pyom.flux_top[:,:,k]/pyom.dzw[k]
        for k in xrange(1, pyom.nz): #k = 2,nz
            pyom.dw_adv[:,:,k] -= pyom.maskW[:,:,k]*(pyom.flux_top[:,:,k]-pyom.flux_top[:,:,k-1])/pyom.dzw[k]

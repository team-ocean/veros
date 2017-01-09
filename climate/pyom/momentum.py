from climate.pyom.external import solve_stream, solve_pressure

def momentum(
        pyom,
        fricTimer,
        pressTimer,
        ):
    """
    =======================================================================
     solve for momentum for taup1
    =======================================================================
    """
    #implicit none
    #integer :: i,j,k

    """
    ---------------------------------------------------------------------------------
      time tendency due to Coriolis force
    ---------------------------------------------------------------------------------
    """
    for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
        for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
            du_cor[i,j,:]= maskU[i,j,:]*(  coriolis_t[i  ,j]*(v[i  ,j,:,tau]+v[i  ,j-1,:,tau])*dxt[i  ]/dxu[i] \
                                          +coriolis_t[i+1,j]*(v[i+1,j,:,tau]+v[i+1,j-1,:,tau])*dxt[i+1]/dxu[i] )*0.25
            dv_cor[i,j,:]=-maskV[i,j,:]*(coriolis_t[i,j  ]*(u[i-1,j  ,:,tau]+u[i,j  ,:,tau])*dyt[j  ]*cost[j  ]/( dyu[j]*cosu[j] ) \
                                        +coriolis_t[i,j+1]*(u[i-1,j+1,:,tau]+u[i,j+1,:,tau])*dyt[j+1]*cost[j+1]/( dyu[j]*cosu[j] ) )*0.25

    """
    ---------------------------------------------------------------------------------
      time tendency due to metric terms
    ---------------------------------------------------------------------------------
    """
    if coord_degree:
        for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
            for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
                du_cor[i,j,:] += maskU[i,j,:]*0.125*tantr[j]*( \
                                  (u[i  ,j,:,tau]+u[i-1,j,:,tau])*(v[i  ,j,:,tau]+v[i  ,j-1,:,tau])*dxt[i  ]/dxu[i] \
                                + (u[i+1,j,:,tau]+u[i  ,j,:,tau])*(v[i+1,j,:,tau]+v[i+1,j-1,:,tau])*dxt[i+1]/dxu[i] )
                dv_cor[i,j,:] -= maskV[i,j,:]*0.125*( \
                                       tantr[j  ]*(u[i,j  ,:,tau]+u[i-1,j  ,:,tau])**2*dyt[j  ]*cost[j  ]/( dyu[j]*cosu[j] ) \
                                     + tantr[j+1]*(u[i,j+1,:,tau]+u[i-1,j+1,:,tau])**2*dyt[j+1]*cost[j+1]/( dyu[j]*cosu[j] ) )

    """
    ---------------------------------------------------------------------------------
     non hydrostatic Coriolis terms, metric terms are neglected
    ---------------------------------------------------------------------------------
    """
    if not enable_hydrostatic:
        for k in xrange(2, nz+1): #k=2,nz
            for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
                du_cor[i,:,k] -= maskU[i,:,k]*0.25*(coriolis_h[i  ,:]*area_t[i  ,:]*(w[i  ,:,k,tau]+w[i  ,:,k-1,tau]) \
                                                   +coriolis_h[i+1,:]*area_t[i+1,:]*(w[i+1,:,k,tau]+w[i+1,:,k-1,tau]) ) \
                                                     /area_u[i,:]
        k = 1
        for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
            du_cor[i,:,k] -= maskU[i,:,k]*0.25*(coriolis_h[i  ,:]*area_t[i  ,:]*(w[i  ,:,k,tau]) \
                                               +coriolis_h[i+1,:]*area_t[i+1,:]*(w[i+1,:,k,tau]) )/area_u[i,:]
        for k in xrange(1, nz): #k=1,nz-1
            for i in xrange(is_pe, ie_pe): #i=is_pe,ie_pe
                dw_cor[i,:,k] = maskW[i,:,k]*0.25*(coriolis_h[i,:]*dzt[k  ]*(u[i,:,k  ,tau]+u[i-1,:,k  ,tau]) \
                                                  +coriolis_h[i,:]*dzt[k+1]*(u[i,:,k+1,tau]+u[i-1,:,k+1,tau]) )/dzw[k]

    """
    ---------------------------------------------------------------------------------
     transfer to time tendencies
    ---------------------------------------------------------------------------------
    """
    for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
        for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
            du[i,j,:,tau] = du_cor[i,j,:]
            dv[i,j,:,tau] = dv_cor[i,j,:]

    if not enable_hydrostatic:
        for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
            for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
                dw[i,j,:,tau] = dw_cor[i,j,:]

    """
    ---------------------------------------------------------------------------------
     wind stress forcing
    ---------------------------------------------------------------------------------
    """
    for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
        for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
            du[i,j,nz,tau] += maskU[i,j,nz]*surface_taux[i,j]/dzt[nz]
            dv[i,j,nz,tau] += maskV[i,j,nz]*surface_tauy[i,j]/dzt[nz]

    """
    ---------------------------------------------------------------------------------
     advection
    ---------------------------------------------------------------------------------
    """
    momentum_advection()
    du[:,:,:,tau] += du_adv
    dv[:,:,:,tau] += dv_adv
    if not enable_hydrostatic:
        dw[:,:,:,tau] += dw_adv

    with fricTimer:
        """
        ---------------------------------------------------------------------------------
         vertical friction
        ---------------------------------------------------------------------------------
        """
        K_diss_v[...] = 0.0
        if enable_implicit_vert_friction:
            #TODO: not implemented yet
            #implicit_vert_friction()
            raise NotImplementedError()
        if enable_explicit_vert_friction:
            #TODO: not implemented yet
            #explicit_vert_friction()
            raise NotImplementedError()

        """
        ---------------------------------------------------------------------------------
         TEM formalism for eddy-driven velocity
        ---------------------------------------------------------------------------------
        """
        if enable_TEM_friction:
            #TODO: not implemented yet
            #call isoneutral_friction
            raise NotImplementedError()

        """
        ---------------------------------------------------------------------------------
        horizontal friction
        ---------------------------------------------------------------------------------
        """
        if enable_hor_friction:
            #TODO: not implemented yet
            #call harmonic_friction
            raise NotImplementedError()
        if enable_biharmonic_friction:
            #TODO: not implemented yet
            #call biharmonic_friction
            raise NotImplementedError()

        """
        ---------------------------------------------------------------------------------
         Rayleigh and bottom friction
        ---------------------------------------------------------------------------------
        """
        K_diss_bot[...] = 0.0
        if enable_ray_friction:
            #TODO: not implemented yet
            #call rayleigh_friction
            raise NotImplementedError()
        if enable_bottom_friction:
            #TODO: not implemented yet
            #call linear_bottom_friction
            raise NotImplementedError()
        if enable_quadratic_bottom_friction:
            #TODO: not implemented yet
            #call quadratic_bottom_friction
            raise NotImplementedError()

        """
        ---------------------------------------------------------------------------------
         add user defined forcing
        ---------------------------------------------------------------------------------
        """
        if enable_momentum_sources:
            #TODO: Not implemented yet
            #call momentum_sources
            raise NotImplementedError()

    """
    ---------------------------------------------------------------------------------
     external mode
    ---------------------------------------------------------------------------------
    """
    with pressTimer:
        if enable_streamfunction:
            solve_stream.solve_streamfunction(pyom)
        else:
            solve_pressure.solve_pressure(pyom)
            if itt == 0:
                psi[:,:,tau] = psi[:,:,taup1]
                psi[:,:,taum1] = psi[:,:,taup1]
        if not enable_hydrostatic:
            #TODO: not implemented yet
            #call solve_non_hydrostatic
            raise NotImplementedError()

def vertical_velocity(u, v, w, maskW, dyt, dxt, dzt, taup, cost, cosu):
    """
    =======================================================================
           vertical velocity from continuity :
           \int_0^z w_z dz =w(z)-w(0) = - \int dz (u_x +v_y)
            w(z)=-int dz u_x + v_y
    =======================================================================
    """
    #integer :: i,j,k
    # integrate from bottom to surface to see error in w
    k = 1
    for j in xrange(js_pe-onx+1, je_pe+onx+1): #j=js_pe-onx+1,je_pe+onx
        for i in xrange(is_pe-onx+1, ie_pe+onx+1): #i=is_pe-onx+1,ie_pe+onx
            w[i,j,k,taup1] = -maskW[i,j,k]*dzt[k]* \
                  ((        u[i,j,k,taup1]-          u[i-1,j,k,taup1])/(cost[j]*dxt[i]) \
                  +(cosu[j]*v[i,j,k,taup1]-cosu[j-1]*v[i,j-1,k,taup1])/(cost[j]*dyt[j]) )
    for k in xrange(2, nz+1): #k=2,nz
        for j in xrange(js_pe-onx+1, je_pe+onx+1): #j=js_pe-onx+1,je_pe+onx
            for i in xrange(is_pe-onx+1, ie_pe+onx+1): #i=is_pe-onx+1,ie_pe+onx
                w[i,j,k,taup1] -= maskW[i,j,k]*dzt[k]* \
                     ((        u[i,j,k,taup1]          -u[i-1,j,k,taup1])/(cost[j]*dxt[i]) \
                     +(cosu[j]*v[i,j,k,taup1]-cosu[j-1]*v[i,j-1,k,taup1])/(cost[j]*dyt[j]) )


def momentum_advection():
    """
    =======================================================================
     Advection of momentum with second order which is energy conserving
    =======================================================================
    """
    #integer :: i,j,k
    #real*8 :: utr(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    #real*8 :: vtr(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    #real*8 :: wtr(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)

    """
    ---------------------------------------------------------------------------------
      Code from MITgcm
    ---------------------------------------------------------------------------------
    """

    #        uTrans(i,j) = u(i,j)*dyG(i,j)*drF(k)
    #        vTrans(i,j) = v(i,j)*dxG(i,j)*drF(k)

    #        fZon(i,j) = 0.25*( uTrans(i,j) + uTrans(i+1,j) ) *( u(i,j) + u(i+1,j) )
    #        fMer(i,j) = 0.25*( vTrans(i,j) + vTrans(i-1,j) ) *( u(i,j) + u(i,j-1) )
    #
    #          gU(i,j,k,bi,bj) =  -
    #     &     *( ( fZon(i,j  )  - fZon(i-1,j)  )
    #     &       +( fMer(i,j+1)  - fMer(i,  j)  )
    #     &       +( fVerUkp(i,j) - fVerUkm(i,j) )
    #     &     ) /drF(k)   / rAw(i,j)


    #        fZon(i,j) = 0.25*( uTrans(i,j) + uTrans(i,j-1) )  *(v(i,j) + v(i-1,j) )
    #        fMer(i,j) = 0.25*( vTrans(i,j) + vTrans(i,j+1) )  *(v(i,j) +  v(i,j+1) )

    #          gV(i,j,k,bi,bj) =  -recip_drF(k)*recip_rAs(i,j,bi,bj)
    #     &     *( ( fZon(i+1,j)  - fZon(i,j  )  )
    #     &       +( fMer(i,  j)  - fMer(i,j-1)  )
    #     &       +( fVerVkp(i,j) - fVerVkm(i,j) )
    #     &     )


    for j in xrange(js_pe-onx, je_pe+onx+1): #j=js_pe-onx,je_pe+onx
        for i in xrange(is_pe-onx, ie_pe+onx+1): #i=is_pe-onx,ie_pe+onx
            utr[i,j,:] = dzt[:]*dyt[j]*u[i,j,:,tau]*maskU[i,j,:]
            vtr[i,j,:] = dzt[:]*cosu[j]*dxt[i]*v[i,j,:,tau]*maskV[i,j,:]
            wtr[i,j,:] = area_t[i,j]*w[i,j,:,tau]*maskW[i,j,:]

    """
    ---------------------------------------------------------------------------------
     for zonal momentum
    ---------------------------------------------------------------------------------
    """
    for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
        for i in xrange(is_pe-1, ie_pe+1): #i=is_pe-1,ie_pe
            flux_east[i,j,:] = 0.25*(u[i,j,:,tau]+u[i+1,j,:,tau])*(utr[i+1,j,:]+utr[i,j,:])
    for j in xrange(js_pe-1, je_pe+1): #j=js_pe-1,je_pe
        for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
            flux_north[i,j,:] = 0.25*(u[i,j,:,tau]+u[i,j+1,:,tau])*(vtr[i+1,j,:]+vtr[i,j,:])
    for k in xrange(1, nz): #k=1,nz-1
        for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
            for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
                flux_top[i,j,k] = 0.25*(u[i,j,k+1,tau]+u[i,j,k,tau])*(wtr[i,j,k]+wtr[i+1,j,k])
    flux_top[:,:,nz] = 0.0
    for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
        for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
            du_adv[i,j,:] = - maskU[i,j,:]*( flux_east[i,j,:] -flux_east[i-1,j,:] \
                                           +flux_north[i,j,:]-flux_north[i,j-1,:])/(area_u[i,j]*dzt[:])
    k=1
    du_adv[:,:,k] -= maskU[:,:,k]*flux_top[:,:,k]/(area_u[:,:]*dzt[k])
    for k in xrange(2, nz+1): #k=2,nz
        du_adv[:,:,k] -= maskU[:,:,k]*(flux_top[:,:,k]-flux_top[:,:,k-1])/(area_u[:,:]*dzt[k])
    """
    ---------------------------------------------------------------------------------
     for meridional momentum
    ---------------------------------------------------------------------------------
    """
    for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
        for i in xrange(is_pe-1, ie_pe+1): #i=is_pe-1,ie_pe
            flux_east[i,j,:] = 0.25*(v[i,j,:,tau]+v[i+1,j,:,tau])*(utr[i,j+1,:]+utr[i,j,:])
    for j in xrange(js_pe-1, je_pe+1): #j=js_pe-1,je_pe
        for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
            flux_north[i,j,:] = 0.25*(v[i,j,:,tau]+v[i,j+1,:,tau])*(vtr[i,j+1,:]+vtr[i,j,:])
    for k in xrange(1, nz): #k=1,nz-1
        for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
            for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
                flux_top[i,j,k] = 0.25*(v[i,j,k+1,tau]+v[i,j,k,tau])*(wtr[i,j,k]+wtr[i,j+1,k])
    flux_top[:,:,nz] = 0.0
    for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
        for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
            dv_adv[i,j,:] = -maskV[i,j,:]*( flux_east[i,j,:] -flux_east[i-1,j,:] \
                                          +flux_north[i,j,:]-flux_north[i,j-1,:])/(area_v[i,j]*dzt[:])
    k=1
    dv_adv[:,:,k] -= maskV[:,:,k]*flux_top[:,:,k]/(area_v[:,:]*dzt[k])
    for k in xrange(2, nz+1): #k=2,nz
        dv_adv[:,:,k] -= maskV[:,:,k]*(flux_top[:,:,k]-flux_top[:,:,k-1])/(area_v[:,:]*dzt[k])

    if not enable_hydrostatic:
        """
        ---------------------------------------------------------------------------------
         for vertical momentum
        ---------------------------------------------------------------------------------
        """
        for k in xrange(1, nz+1): #k=1,nz
            for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
                for i in xrange(is_pe-1, ie_pe+1): #i=is_pe-1,ie_pe
                    flux_east[i,j,k] = 0.5*(w[i,j,k,tau]+w[i+1,j,k,tau])*(u[i,j,k,tau]+u[i,j,min(nz,k+1),tau])*0.5*maskW[i+1,j,k]*maskW[i,j,k]
        for k in xrange(1, nz+1): #k=1,nz
            for j in xrange(js_pe-1, je_pe+1): #j=js_pe-1,je_pe
                for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
                    flux_north[i,j,k] = 0.5*(w[i,j,k,tau]+w[i,j+1,k,tau])* \
                                        (v[i,j,k,tau]+v[i,j,min(nz,k+1),tau])*0.5*maskW[i,j+1,k]*maskW[i,j,k]*cosu[j]
        for k in xrange(1, nz): #k=1,nz-1
            for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
                for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
                    flux_top[i,j,k] = 0.5*(w[i,j,k+1,tau]+w[i,j,k,tau])*(w[i,j,k,tau]+w[i,j,k+1,tau])*0.5*maskW[i,j,k+1]*maskW[i,j,k]
        flux_top[:,:,nz] = 0.0
        for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
            for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
                dw_adv[i,j,:] = maskW[i,j,:]* (-( flux_east[i,j,:]-  flux_east[i-1,j,:])/(cost[j]*dxt[i]) \
                                               -(flux_north[i,j,:]- flux_north[i,j-1,:])/(cost[j]*dyt[j]) )
        k=1
        dw_adv[:,:,k] -= maskW[:,:,k]*flux_top[:,:,k]/dzw[k]
        for k in xrange(2, nz+1): #k=2,nz
            dw_adv[:,:,k] -= maskW[:,:,k]*(flux_top[:,:,k]-flux_top[:,:,k-1])/dzw[k]

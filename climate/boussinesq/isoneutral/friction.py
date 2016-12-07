import numpy as np

def isoneutral_friction(kbot, nz):
    """
    =======================================================================
      vertical friction using TEM formalism for eddy driven velocity
    =======================================================================
    """
    integer :: i,j,k,ks
    real*8 :: a_tri(nz),b_tri(nz),c_tri(nz),d_tri(nz),delta(nz),fxa
    real*8 :: diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    real*8 :: aloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)

    a_tri = np.zeros(nz)
    b_tri = np.zeros(nz)
    c_tri = np.zeros(nz)
    d_tri = np.zeros(nz)
    delta = np.zeros(nz)

    if enable_implicit_vert_friction:
        aloc = u[:,:,:,taup1]
    else:
        aloc = u[:,:,:,tau]

    # implicit vertical friction of zonal momentum by GM
    for j in xrange(js_pe-1, je_pe+1): #j=js_pe-1,je_pe
        for i in xrange(is_pe-1, ie_pe+1): #i=is_pe-1,ie_pe
            ks = max(kbot[i,j], kbot[i+1,j])
            if ks > 0:
                for k in xrange(ks, nz): #k=ks,nz-1
                    fxa = 0.5*(kappa_gm[i,j,k]+kappa_gm[i+1,j,k])
                    delta[k] = dt_mom/dzw[k]*fxa*maskU[i,j,k+1]*maskU[i,j,k]
                delta[nz] = 0.0
                a_tri[ks] = 0.0
                for k in xrange(ks+1, nz+1): #k=ks+1,nz
                    a_tri[k] = - delta[k-1]/dzt[k]
                b_tri[ks] = 1+ delta[ks]/dzt[ks]
                for k in xrange(ks+1, nz): #k=ks+1,nz-1
                    b_tri[k] = 1+ delta[k]/dzt[k] + delta[k-1]/dzt[k]
                b_tri[nz] = 1+ delta[nz-1]/dzt[nz]
                for k in xrange(ks, nz): #k=ks,nz-1
                    c_tri[k] = - delta[k]/dzt[k]
                c_tri[nz] = 0.0
                d_tri[ks:nz+1] = aloc[i,j,ks:nz+1] #  A u = d
                numerics.solve_tridiag(a_tri[ks:nz+1],b_tri[ks:nz+1],c_tri[ks:nz+1],d_tri[ks:nz+1],u[i,j,ks:nz,taup1],nz-ks+1)
                du_mix[i,j,ks:nz+1] = du_mix[i,j,ks:nz+1] + (u[i,j,ks:nz+1,taup1]-aloc[i,j,ks:nz+1])/dt_mom

    if enable_conserve_energy:
        # diagnose dissipation
        for k in xrange(1, nz): #k=1,nz-1
            for j in xrange(js_pe-1, je_pe+1): #j=js_pe-1,je_pe
                for i in xrange(is_pe-1, ie_pe+1): #i=is_pe-1,ie_pe
                    fxa = 0.5*(kappa_gm[i,j,k]+kappa_gm[i+1,j,k])
                    flux_top[i,j,k] = fxa*(u[i,j,k+1,taup1]-u[i,j,k,taup1])/dzw[k]*maskU[i,j,k+1]*maskU[i,j,k]
        for k in xrange(1, nz): # k=1,nz-1
            for j in xrange(js_pe-1, je_pe+1): # j=js_pe-1,je_pe
                for i in xrange(is_pe-1, ie_pe+1): # i=is_pe-1,ie_pe
                    diss[i,j,k] = (u[i,j,k+1,tau]-u[i,j,k,tau])*flux_top[i,j,k]/dzw[k]
        diss[:,:,nz] = 0.0
        numerics.ugrid_to_tgrid(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,diss)
        K_diss_gm = diss

    if enable_implicit_vert_friction:
        aloc = v[:,:,:,taup1]
    else:
        aloc = v[:,:,:,tau]

    # implicit vertical friction of meridional momentum by GM
    for j in xrange(js_pe-1, je_pe+1): # j=js_pe-1,je_pe
        for i in xrange(is_pe-1, ie_pe+1): # i=is_pe-1,ie_pe
            ks=max(kbot(i,j),kbot(i,j+1))
            if ks > 0:
                for k in xrange(ks, nz): # k=ks,nz-1
                    fxa = 0.5*(kappa_gm[i,j,k]+kappa_gm[i,j+1,k])
                    delta[k] = dt_mom/dzw[k]*fxa*maskV[i,j,k+1]*maskV[i,j,k]
                delta[nz] = 0.0
                a_tri[ks] = 0.0
                for k in xrange(ks+1, nz+1): # k=ks+1,nz
                    a_tri[k] = - delta[k-1]/dzt[k]
                b_tri[ks] = 1+ delta[ks]/dzt[ks]
                for k in xrange(ks+1, nz): # k=ks+1,nz-1
                    b_tri[k] = 1+ delta[k]/dzt[k] + delta[k-1]/dzt[k]
                b_tri[nz] = 1+ delta[nz-1]/dzt[nz]
                for k in xrange(ks, nz-1): # k=ks,nz-1
                    c_tri[k] = - delta[k]/dzt[k]
                c_tri[nz] = 0.0
                d_tri[ks:nz+1] = aloc[i,j,ks:nz+1]
                numerics.solve_tridiag(a_tri[ks:nz+1],b_tri[ks:nz+1],c_tri[ks:nz+1],d_tri[ks:nz+1],v[i,j,ks:nz+1,taup1],nz-ks+1)
                dv_mix[i,j,ks:nz+1] = dv_mix[i,j,ks:nz+1] + (v[i,j,ks:nz,taup1]-aloc[i,j,ks:nz+1])/dt_mom

    if enable_conserve_energy:
        # diagnose dissipation
        for k in xrange(1, nz): # k=1,nz-1
            for j in xrange(js_pe-1, je_pe+1): # j=js_pe-1,je_pe
                for i in xrange(is_pe-1, ie_pe+1): # i=is_pe-1,ie_pe
                    fxa = 0.5*(kappa_gm[i,j,k]+kappa_gm[i,j+1,k])
                    flux_top[i,j,k] = fxa*(v[i,j,k+1,taup1]-v[i,j,k,taup1])/dzw[k]*maskV[i,j,k+1]*maskV[i,j,k]
        for k in xrange(1, nz): #k=1,nz-1
            for j in xrange(js_pe-1, je_pe+1): # j=js_pe-1,je_pe
                for i in xrange(is_pe-1, ie_pe+1): # i=is_pe-1,ie_pe
                    diss[i,j,k] =(v[i,j  ,k+1,tau]-v[i,j  ,k,tau])*flux_top[i,j  ,k]/dzw[k]
        diss[:,:,nz] = 0.0
        numerics.vgrid_to_tgrid(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,diss)
        K_diss_gm = K_diss_gm + diss

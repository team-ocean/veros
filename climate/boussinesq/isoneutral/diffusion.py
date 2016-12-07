import numpy as np
import climate.boussinesq.numerics as numerics

def isoneutral_diffusion(is_,ie_,js_,je_,nz_,tr,istemp):
    """
    =======================================================================
       Isopycnal diffusion for tracer,
       following functional formulation by Griffies et al
       Dissipation is calculated and stored in P_diss_iso
       T/S changes are added to dtemp_iso/dsalt_iso
    =======================================================================
    """

    """
    -----------------------------------------------------------------------
         construct total isoneutral tracer flux at east face of "T" cells
    -----------------------------------------------------------------------
    """
    for k in xrange(nz): #k=1, nz
        for j in xrange(js_pe, je_pe+1): #j=js_pe, je_pe
            for i in xrange(is_pe-1, ie_pe+1): #i=is_pe-1,ie_pe
                diffloc = 0.25*(K_iso[i,j,k+1]+K_iso[i,j,max(1,k)] + K_iso[i+1,j,k+1]+K_iso[i+1,j,max(1,k)])
                sumz = 0.
                for kr in xrange(2): #kr=0,1
                    km1kr = max(k+kr,1)
                    kpkr  = min(k+1+kr,nz)
                    for ip in xrange(2):#ip=0,1
                        sumz = sumz + diffloc*Ai_ez[i,j,k+1,ip,kr] *(tr[i+ip,j,kpkr,tau]-tr[i+ip,j,km1kr,tau])
                flux_east[i,j,k+1] = sumz/(4*dzt[k+1]) + (tr(i+1,j,k+1,tau)-tr(i,j,k+1,tau))/(cost[j]*dxu[i]) *K_11[i,j,k+1]
    """
    -----------------------------------------------------------------------
         construct total isoneutral tracer flux at north face of "T" cells
    -----------------------------------------------------------------------
    """
    for k in xrange(nz): #k=1, nz
        for j in xrange(js_pe-1, je_pe+1): #j=js_pe-1, je_pe
            for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
                diffloc = 0.25*(K_iso[i,j,k+1]+K_iso[i,j,max(1,k)] + K_iso[i,j+1,k+1]+K_iso[i,j+1,max(1,k)] )
                sumz    = 0.
                for kr in xrange(2): # kr=0,1
                    km1kr = max(k+kr,1)
                    kpkr  = min(k+1+kr,nz)
                    for jp in xrange(2): #do jp=0,1
                        sumz = sumz + diffloc*Ai_nz[i,j,k+1,jp,kr] *(tr[i,j+jp,kpkr,tau]-tr[i,j+jp,km1kr,tau])
                flux_north(i,j,k+1) = cosu[j]*( sumz/(4*dzt[k+1])+ (tr[i,j+1,k+1,tau]-tr[i,j,k+1,tau])/dyu[j]*K_22[i,j,k+1])
    """
    -----------------------------------------------------------------------
         compute the vertical tracer flux "flux_top" containing the K31
         and K32 components which are to be solved explicitly. The K33
         component will be treated implicitly. Note that there are some
         cancellations of dxu(i-1+ip) and dyu(jrow-1+jp)
    -----------------------------------------------------------------------
    """
    for k in xrange(nz-1): #k=1,nz-1
        for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
            for i in xrange(is_pe, ie_pe): #i=is_pe,ie_pe
                diffloc = K_iso[i,j,k+1]
                sumx = 0.
                for ip in xrange(2): #ip=0,1
                    for kr in xrange(2): #kr=0,1
                        sumx = sumx + diffloc*Ai_bx[i,j,k+1,ip,kr]/cost[j]*(tr[i+ip,j,k+1+kr,tau] - tr[i-1+ip,j,k+1+kr,tau])
                sumy    = 0.
                for jp in xrange(2): #jp=0,1
                    for kr in xrange(2): #kr=0,1
                        sumy = sumy + diffloc*Ai_by[i,j,k+1,jp,kr]*cosu[j-1+jp]* (tr[i,j+jp,k+1+kr,tau]-tr[i,j-1+jp,k+1+kr,tau])
                flux_top[i,j,k+1] = sumx/(4*dxt[i]) +sumy/(4*dyt[j]*cost[j] )
    flux_top[:,:,nz]=0.0
    """
    ---------------------------------------------------------------------------------
         add explicit part
    ---------------------------------------------------------------------------------
    """
    for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
        for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
            aloc[i,j,:] = maskT[i,j,:]*( (flux_east[i,j,:] - flux_east[i-1,j,:])/(cost[j]*dxt[i]) \
                                    +(flux_north[i,j,:] - flux_north[i,j-1,:])/(cost[j]*dyt[j]) )
     k=1
     aloc[:,:,k] = aloc[:,:,k]+maskT[:,:,k]*flux_top[:,:,k]/dzt[k]
     for k in xrange(2, nz+1):
        aloc[:,:,k]=aloc[:,:,k]+maskT[:,:,k]*(flux_top[:,:,k]- flux_top[:,:,k-1)]/dzt[k]

     if (istemp):
          dtemp_iso = dtemp_iso + aloc
     else:
          dsalt_iso = dsalt_iso + aloc

    for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
        for i xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
            tr[i,j,:,taup1] = tr[i,j,:,taup1]+dt_tracer*aloc[i,j,:]
    """
    ---------------------------------------------------------------------------------
         add implicit part
    ---------------------------------------------------------------------------------
    """
    aloc = tr(:,:,:,taup1)
    a_tri = np.zeroes(nz)
    b_tri = np.zeroes(nz)
    c_tri = np.zeroes(nz)
    d_tri = np.zeroes(nz)
    delta = np.zeroes(nz)
    for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
        for i xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
            ks=kbot[i,j]
            if (ks>0):
                for k in xrange(ks, nz-1): #k=ks,nz-1
                    delta[k] = dt_tracer/dzw[k]*K_33[i,j,k]
                delta[nz] = 0.0
                for k in xrange(ks+1,nz+1): #k=ks+1,nz
                    a_tri[k] = - delta[k-1]/dzt[k]
                a_tri(ks)=0.0
                for k in xrange(ks, nz): #k=ks+1,nz-1
                    b_tri[k] = 1+ delta[k]/dzt[k] + delta[k-1]/dzt[k]
                b_tri[nz] = 1+ delta[nz-1]/dzt[nz]
                b_tri[ks] = 1+ delta[ks]/dzt[ks]
                for k in xrange(ks, nz): #k=ks,nz-1
                    c_tri[k] = - delta[k]/dzt[k]
                c_tri[nz] = 0.0
                d_tri[ks:nz] = tr[i,j,ks:nz,taup1]
                call solve_tridiag(a_tri[ks:nz],b_tri[ks:nz],c_tri[ks:nz],d_tri[ks:nz],sol[ks:nz],nz-ks+1)
                tr(i,j,ks:nz,taup1) = sol[ks:nz]
    if (istemp):
         dtemp_iso = dtemp_iso + (tr[:,:,:,taup1]-aloc)/dt_tracer
    else:
         dsalt_iso = dsalt_iso + (tr[:,:,:,taup1]-aloc)/dt_tracer

    """
    ---------------------------------------------------------------------------------
     dissipation by isopycnal mixing
    ---------------------------------------------------------------------------------
    """
    if enable_conserve_energy:

        if istemp:
            bloc[:,:,:] = int_drhodT[:,:,:,tau]
        else:
            bloc[:,:,:] = int_drhodS[:,:,:,tau]

        for k in xrange(1, nz+1): #k=1,nz
            for j in xrange(js_pe-onx+1,je_pe+onx): #j=js_pe-onx+1,je_pe+onx-1
                for i in xrange(is_pe-onx+1,ie_pe+onx): #i=is_pe-onx+1,ie_pe+onx-1
                    fxa = bloc[i,j,k]
                    aloc[i,j,k] =+0.5*grav/rho_0*( (bloc[i+1,j,k]-fxa)*flux_east[i,j,k] \
                                                  +(fxa-bloc[i-1,j,k])*flux_east[i-1,j,k] ) /(dxt[i]*cost[j]) \
                                 +0.5*grav/rho_0*( (bloc[i,j+1,k]-fxa)*flux_north[i,j,k] \
                                                  +(fxa-bloc[i,j-1,k])*flux_north[i,j-1,k] ) /(dyt[j]*cost[j])
        """
        ---------------------------------------------------------------------------------
         dissipation interpolated on W-grid
        ---------------------------------------------------------------------------------
        """
        for j in xrange(js_pe-onx, je_pe+onx+1): #j=js_pe-onx,je_pe+onx
            for i in xrange(is_pe-onx, ie_pe+onx+1): #i=is_pe-onx,ie_pe+onx
                ks = kbot[i,j]
                if ks > 0:
                    k=ks
                    P_diss_iso[i,j,k] = P_diss_iso[i,j,k]+ \
                                       0.5*(aloc[i,j,k]+aloc[i,j,k+1]) + 0.5*aloc[i,j,k]*dzw[max(1,k-1)]/dzw[k]
                    for k in xrange(ks+1, nz): #k=ks+1,nz-1
                        P_diss_iso[i,j,k] = P_diss_iso[i,j,k]+ 0.5*(aloc[i,j,k] +aloc[i,j,k+1])
                    k = nz
                    P_diss_iso[i,j,k] = P_diss_iso[i,j,k]+ aloc[i,j,k]
        """
        ---------------------------------------------------------------------------------
         diagnose dissipation of dynamic enthalpy by explicit and implicit vertical mixing
        ---------------------------------------------------------------------------------
        """
        if (istemp):
            for k in xrange(1, nz): #k=1,nz-1
                for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
                    for i in xrange(is_pe, ie_pe): #i=is_pe,ie_pe
                        fxa = (-bloc[i,j,k+1] +bloc[i,j,k])/dzw[k]
                        P_diss_iso[i,j,k] = P_diss_iso[i,j,k]  -grav/rho_0*fxa*flux_top[i,j,k]*maskW[i,j,k] \
                                       -grav/rho_0*fxa*K_33[i,j,k]*(temp[i,j,k+1,taup1]-temp[i,j,k,taup1])/dzw[k]*maskW[i,j,k]
        else:
            for k in xrange(1, nz): #k=1,nz-1
                for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
                    for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
                        fxa = (-bloc[i,j,k+1] +bloc[i,j,k])/dzw[k]
                        P_diss_iso[i,j,k] = P_diss_iso[i,j,k] - grav/rho_0*fxa*flux_top[i,j,k]*maskW[i,j,k] \
                                       -grav/rho_0*fxa*K_33[i,j,k]*(salt[i,j,k+1,taup1]-salt[i,j,k,taup1])/dzw[k]*maskW[i,j,k]








def isoneutral_skew_diffusion(is_,ie_,js_,je_,nz_,tr,istemp):
    """
    =======================================================================
       Isopycnal skew diffusion for tracer, 
       following functional formulation by Griffies et al 
       Dissipation is calculated and stored in P_diss_skew
       T/S changes are added to dtemp_iso/dsalt_iso
    =======================================================================
    """

    """
    -----------------------------------------------------------------------
         construct total isoneutral tracer flux at east face of "T" cells
    -----------------------------------------------------------------------
    """
    for k in xrange(1, nz+1): #k=1,nz
        for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
            for i in xrange(is_pe-1, ie_pe+1): #i=is_pe-1,ie_pe
                diffloc =-0.25*(K_gm[i,j,k]+K_gm[i,j,max(1,k-1)] + K_gm[i+1,j,k]+K_gm[i+1,j,max(1,k-1)])
                sumz = 0.
                for kr in xrange(2): #kr=0,1
                    km1kr = max(k-1+kr,1)
                    kpkr  = min(k+kr,nz)
                    for ip in xrange(2): #ip=0,1
                        sumz = sumz + diffloc*Ai_ez[i,j,k,ip,kr] *(tr[i+ip,j,kpkr,tau]-tr[i+ip,j,km1kr,tau])
                flux_east[i,j,k] = sumz/(4*dzt[k]) + (tr[i+1,j,k,tau]-tr[i,j,k,tau])/(cost[j]*dxu[i]) *K_11[i,j,k]
    """
    -----------------------------------------------------------------------
         construct total isoneutral tracer flux at north face of "T" cells
    -----------------------------------------------------------------------
    """
    for k in xrange(1, nz+1): #k=1,nz
        for j in xrange(js_pe-1, je_pe+1): #j=js_pe-1,je_pe
            for i in xrange(is_pe,ie_pe+1): #i=is_pe,ie_pe
                diffloc =-0.25*(K_gm[i,j,k]+K_gm[i,j,max(1,k-1)] + K_gm[i,j+1,k]+K_gm[i,j+1,max(1,k-1)])
                sumz    = 0.
                for kr in xrange(2): #kr=0,1
                    km1kr = max(k-1+kr,1)
                    kpkr  = min(k+kr,nz)
                    for jp in xrange(2): #jp=0,1
                        sumz = sumz + diffloc*Ai_nz[i,j,k,jp,kr] *(tr[i,j+jp,kpkr,tau]-tr[i,j+jp,km1kr,tau])
                flux_north[i,j,k] = cosu[j]*( sumz/(4*dzt[k])+ (tr[i,j+1,k,tau]-tr[i,j,k,tau])/dyu[j]*K_22[i,j,k] )
    """
    -----------------------------------------------------------------------
         compute the vertical tracer flux "flux_top" containing the K31
         and K32 components which are to be solved explicitly.
    -----------------------------------------------------------------------
    """
    for k in xrange(1, nz): #k=1,nz-1
        for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
            for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
                diffloc = K_gm[i,j,k]
                sumx = 0.
                for ip in xrange(2): #ip=0,1
                    for kr in xrange(2): #kr=0,1
                        sumx = sumx + diffloc*Ai_bx[i,j,k,ip,kr]/cost[j]*(tr[i+ip,j,k+kr,tau] - tr[i-1+ip,j,k+kr,tau])
                sumy    = 0.
                for jp in xrange(2):
                    for kr in xrange(2):
                        sumy = sumy + diffloc*Ai_by[i,j,k,jp,kr]*cosu[j-1+jp]* (tr[i,j+jp,k+kr,tau]-tr[i,j-1+jp,k+kr,tau])
                flux_top[i,j,k] = sumx/(4*dxt[i]) +sumy/(4*dyt(j)*cost[j] )
    flux_top[:,:,nz] = 0.0
    """
    ---------------------------------------------------------------------------------
         add explicit part
    ---------------------------------------------------------------------------------
    """
    for j in xrange(js_pe, je_pe+1): # j=js_pe,je_pe
        for i in xrange(is_pe, ie_pe+1): # i=is_pe,ie_pe
            aloc[i,j,:] = maskT[i,j,:]*( (flux_east[i,j,:] - flux_east[i-1,j,:])/(cost[j]*dxt[i]) \
                                        +(flux_north[i,j,:] - flux_north[i,j-1,:])/(cost[j]*dyt[j]) )
    k=1
    aloc[:,:,k] = aloc[:,:,k]+maskT[:,:,k]*flux_top[:,:,k]/dzt[k]
    for k in xrange(2, nz+1): # k=2,nz
        aloc[:,:,k] = aloc[:,:,k]+maskT[:,:,k]*(flux_top[:,:,k] - flux_top[:,:,k-1])/dzt[k]

    if istemp:
         dtemp_iso = dtemp_iso + aloc
    else:
         dsalt_iso = dsalt_iso + aloc

    for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
        for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
            tr[i,j,:,taup1] = tr[i,j,:,taup1]+dt_tracer*aloc[i,j,:]

    """
    ---------------------------------------------------------------------------------
     dissipation by isopycnal mixing
    ---------------------------------------------------------------------------------
    """
    if enable_conserve_energy:

        if istemp:
            bloc[:,:,:] = int_drhodT[:,:,:,tau]
        else:
            bloc[:,:,:] = int_drhodS[:,:,:,tau]

        for k in xrange(1, nz+1): #k=1,nz
            for j in xrange(js_pe-onx+1,je_pe+onx): #j=js_pe-onx+1,je_pe+onx-1
                for i in xrange(is_pe-onx+1,ie_pe+onx): #i=is_pe-onx+1,ie_pe+onx-1
                    fxa = bloc[i,j,k]
                    aloc[i,j,k] =+0.5*grav/rho_0*( (bloc[i+1,j,k]-fxa)*flux_east[i  ,j,k] \
                                                  +(fxa-bloc[i-1,j,k])*flux_east[i-1,j,k]) /(dxt[i]*cost[j]) \
                                 +0.5*grav/rho_0*( (bloc[i,j+1,k]-fxa)*flux_north[i,j  ,k] \
                                                  +(fxa-bloc[i,j-1,k])*flux_north[i,j-1,k]) /(dyt[j]*cost[j])
        """
        ---------------------------------------------------------------------------------
         dissipation interpolated on W-grid
        ---------------------------------------------------------------------------------
        """
        for j in xrange(js_pe-onx,je_pe+onx+1): #j=js_pe-onx,je_pe+onx
            for i in xrange(is_pe-onx,ie_pe+onx+1): #i=is_pe-onx,ie_pe+onx
                ks = kbot[i,j]
                if ks > 0:
                    k=ks
                    P_diss_skew[i,j,k] = P_diss_skew[i,j,k]+ \
                                    0.5*(aloc[i,j,k]+aloc[i,j,k+1]) + 0.5*aloc[i,j,k]*dzw[max(1,k-1)]/dzw[k]
                    for k in xrange(ks+1, nz): #k=ks+1,nz-1
                        P_diss_skew[i,j,k] = P_diss_skew[i,j,k]+ 0.5*(aloc[i,j,k] +aloc[i,j,k+1])
                    k = nz
                    P_diss_skew[i,j,k] = P_diss_skew[i,j,k]+ aloc[i,j,k]
        """
        ---------------------------------------------------------------------------------
         dissipation by vertical component of skew mixing
        ---------------------------------------------------------------------------------
        """
        for k in xrange(1, nz): #k=1,nz-1
            for j in xrange(js_pe, je_pe): #j=js_pe,je_pe
                for i in xrange(is_pe, ie_pe): #i=is_pe,ie_pe
                    fxa = (-bloc[i,j,k+1] +bloc[i,j,k])/dzw[k]
                    P_diss_skew[i,j,k] = P_diss_skew[i,j,k]  -grav/rho_0*fxa*flux_top[i,j,k]*maskW[i,j,k]





def isoneutral_diffusion_all(is_,ie_,js_,je_,nz_,tr,istemp):
    """
    =======================================================================
       Isopycnal diffusion plus skew diffusion for tracer,
       following functional formulation by Griffies et al
       Dissipation is calculated and stored in P_diss_iso
    =======================================================================
    """

    if enable_skew_diffusion:
        aloc = K_gm
    else:
        aloc = np.zeros(ie_pe+onx-(is_pe-onx),je_pe+onx-(js_pe-onx),nz)

    """
    -----------------------------------------------------------------------
         construct total isoneutral tracer flux at east face of "T" cells
    -----------------------------------------------------------------------
    """
    for k in xrange(1, nz+1): #k=1,nz
        for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
            for i in xrange(is_pe-1, ie_pe+1): #i=is_pe-1,ie_pe
                diffloc = 0.25*(K_iso[i,j,k]+K_iso[i,j,max(1,k-1)] + K_iso[i+1,j,k]+K_iso[i+1,j,max(1,k-1)] ) \
                        - 0.25*(aloc[i,j,k]+aloc[i,j,max(1,k-1)] + aloc[i+1,j,k]+aloc[i+1,j,max(1,k-1)] )
                sumz = 0.
                for kr in xrange(2): #kr=0,1
                    km1kr = max(k-1+kr,1)
                    kpkr  = min(k+kr,nz)
                    for ip in xrange(2): #ip=0,1
                        sumz = sumz + diffloc*Ai_ez[i,j,k,ip,kr] *(tr[i+ip,j,kpkr,tau]-tr[i+ip,j,km1kr,tau])
                flux_east[i,j,k] = sumz/(4*dzt[k]) + (tr[i+1,j,k,tau]-tr[i,j,k,tau])/(cost[j]*dxu[i]) *K_11[i,j,k]
    """
    -----------------------------------------------------------------------
         construct total isoneutral tracer flux at north face of "T" cells
    -----------------------------------------------------------------------
    """
    for k in xrange(1, nz+1): #k=1,nz
        for j in xrange(js_pe-1,je_pe+1): #j=js_pe-1,je_pe
            for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
                diffloc = 0.25*(K_iso[i,j,k]+K_iso[i,j,max(1,k-1)] + K_iso[i,j+1,k]+K_iso[i,j+1,max(1,k-1)] ) \
                        - 0.25*(aloc[i,j,k]+aloc[i,j,max(1,k-1)] + aloc[i,j+1,k]+aloc[i,j+1,max(1,k-1)] )
                sumz    = 0.
                for kr in xrange(2): #kr=0,1
                    km1kr = max(k-1+kr,1)
                    kpkr  = min(k+kr,nz)
                    for jp in xrange(2): #jp=0,1
                        sumz = sumz + diffloc*Ai_nz[i,j,k,jp,kr] *(tr[i,j+jp,kpkr,tau]-tr[i,j+jp,km1kr,tau])
                flux_north[i,j,k] = cosu[j]*( sumz/(4*dzt[k])+ (tr[i,j+1,k,tau]-tr[i,j,k,tau])/dyu[j]*K_22[i,j,k])
    """
    -----------------------------------------------------------------------
         compute the vertical tracer flux "flux_top" containing the K31
         and K32 components which are to be solved explicitly. The K33
         component will be treated implicitly. Note that there are some
         cancellations of dxu(i-1+ip) and dyu(jrow-1+jp)
    -----------------------------------------------------------------------
    """
    for k in xrange(1, nz): #k=1,nz-1
        for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
            for i in xrange(is_pe, ie_pe): #i=is_pe,ie_pe
                diffloc = K_iso[i,j,k] + aloc[i,j,k]
                sumx = 0.
                for ip in xrange(2): #ip=0,1
                    for kr in xrange(2): #kr=0,1
                        sumx = sumx + diffloc*Ai_bx[i,j,k,ip,kr]/cost[j]*(tr[i+ip,j,k+kr,tau] - tr[i-1+ip,j,k+kr,tau])
                sumy    = 0.
                for jp in xrange(2): #jp=0,1
                    for kr in xrange(2): #kr=0,1
                        sumy = sumy + diffloc*Ai_by[i,j,k,jp,kr]*cosu[j-1+jp]* (tr[i,j+jp,k+kr,tau]-tr[i,j-1+jp,k+kr,tau])
                flux_top[i,j,k] = sumx/(4*dxt[i]) +sumy/(4*dyt[j]*cost[j] )
    flux_top[:,:,nz] = 0.0
    """
    ---------------------------------------------------------------------------------
         add explicit part
    ---------------------------------------------------------------------------------
    """
    for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
        for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
            aloc[i,j,:] = maskT[i,j,:]*( (flux_east[i,j,:]-  flux_east[i-1,j,:])/(cost[j]*dxt[i]) \
                                       +(flux_north[i,j,:]- flux_north[i,j-1,:])/(cost[j]*dyt[j]) )
    k=1
    aloc[:,:,k] = aloc[:,:,k]+maskT[:,:,k]*flux_top[:,:,k]/dzt[k]
    for k in xrange(2, nz+1): #k=2,nz
        aloc[:,:,k] = aloc[:,:,k]+maskT[:,:,k]*(flux_top[:,:,k]- flux_top[:,:,k-1])/dzt[k]

    if istemp:
         dtemp_iso = aloc
    else
         dsalt_iso = aloc

    for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
        for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
            tr[i,j,:,taup1] = tr[i,j,:,taup1]+dt_tracer*aloc[i,j,:]

    """
    ---------------------------------------------------------------------------------
         add implicit part
    ---------------------------------------------------------------------------------
    """
    aloc = tr[:,:,:,taup1]
    a_tri=np.zeros(nz)
    b_tri=np.zeros(nz)
    c_tri=np.zeros(nz)
    d_tri=np.zeros(nz)
    delta=np.zeros(nz)
    for j in xrange(js_pe, je_pe+1): # j=js_pe,je_pe
        for i in xrange(is_pe, ie_pe+1): # i=is_pe,ie_pe
            ks = kbot[i,j]
            if ks > 0:
                for k in xrange(ks, nz): # k=ks,nz-1
                    delta(k) = dt_tracer/dzw(k)*K_33(i,j,k)
                delta[nz] = 0.0
                for k in xrange(ks+1, nz+1): # k=ks+1,nz
                    a_tri[k] = - delta[k-1]/dzt[k]
                a_tri[ks] = 0.0
                for k in xrange(ks+1, nz): # k=ks+1,nz-1
                    b_tri[k] = 1+ delta[k]/dzt[k] + delta[k-1]/dzt[k]
                b_tri[nz] = 1+ delta[nz-1]/dzt[nz]
                b_tri[ks] = 1+ delta[ks]/dzt[ks]
                for k in xrange(ks, nz): #k=ks,nz-1
                    c_tri[k] = - delta[k]/dzt[k]
                c_tri[nz] = 0.0
                d_tri[ks:nz+1] = tr(i,j,ks:nz+1,taup1)
                numerics.solve_tridiag(a_tri[ks:nz+1],b_tri[ks:nz+1],c_tri[ks:nz+1],d_tri[ks:nz+1],sol[ks:nz+1],nz-ks+1)
                tr[i,j,ks:nz+1,taup1] = sol[ks:nz+1]

    if istemp:
         dtemp_iso = dtemp_iso + (tr[:,:,:,taup1]-aloc)/dt_tracer
    else:
         dsalt_iso = dsalt_iso + (tr[:,:,:,taup1]-aloc)/dt_tracer
    """
    ---------------------------------------------------------------------------------
     dissipation by isopycnal mixing
    ---------------------------------------------------------------------------------
    """
    if enable_conserve_energy:

        if istemp:
            bloc[:,:,:] = int_drhodT[:,:,:,tau]
        else:
            bloc[:,:,:] = int_drhodS[:,:,:,tau]

        for k in xrange(1, nz+1): #k=1,nz
            for j in xrange(js_pe-onx+1,je_pe+onx): #j=js_pe-onx+1,je_pe+onx-1
                for i in xrange(is_pe-onx+1,ie_pe+onx): #i=is_pe-onx+1,ie_pe+onx-1
                    fxa = bloc[i,j,k]
                    aloc[i,j,k] =+0.5*grav/rho_0*( (bloc[i+1,j,k]-fxa)*flux_east[i  ,j,k] \
                                                  +(fxa-bloc[i-1,j,k])*flux_east[i-1,j,k] ) /(dxt[i]*cost[j]) \
                                 +0.5*grav/rho_0*( (bloc[i,j+1,k]-fxa)*flux_north[i,j  ,k] \
                                                  +(fxa-bloc[i,j-1,k])*flux_north[i,j-1,k] ) /(dyt[j]*cost[j])
        """
        ---------------------------------------------------------------------------------
         dissipation interpolated on W-grid
        ---------------------------------------------------------------------------------
        """
        for j in xrange(js_pe-onx, je_pe+onx+1): # j=js_pe-onx,je_pe+onx
            for i in xrange(is_pe-onx, ie_pe+onx+1): #i=is_pe-onx,ie_pe+onx
                ks = kbot[i,j]
                if ks > 0:
                    k = ks
                    P_diss_iso[i,j,k] = P_diss_iso[i,j,k]+ \
                                 0.5*(aloc[i,j,k]+aloc[i,j,k+1]) + 0.5*aloc[i,j,k]*dzw[max(1,k-1)]/dzw[k]
                    for k in xrange(ks+1, nz): #k=ks+1,nz-1
                        P_diss_iso[i,j,k] = P_diss_iso[i,j,k]+ 0.5*(aloc[i,j,k] +aloc[i,j,k+1])
                    k = nz
                    P_diss_iso[i,j,k] = P_diss_iso[i,j,k] + aloc[i,j,k]
        """
        ---------------------------------------------------------------------------------
         diagnose dissipation of dynamic enthalpy by explicit and implicit vertical mixing
        ---------------------------------------------------------------------------------
        """
        if istemp:
            for k in xrange(1, nz): #k=1,nz-1
                for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
                    for i in xrange(is_pe, ie_pe): # i=is_pe,ie_pe
                        fxa = (-bloc[i,j,k+1] +bloc[i,j,k])/dzw[k]
                        P_diss_iso[i,j,k] = P_diss_iso[i,j,k]  -grav/rho_0*fxa*flux_top[i,j,k]*maskW[i,j,k] \
                                       -grav/rho_0*fxa*K_33[i,j,k]*(temp[i,j,k+1,taup1]-temp[i,j,k,taup1])/dzw[k]*maskW[i,j,k]
        else:
            for k in xrange(1, nz): # k=1,nz-1
                for j in xrange(js_pe, je_pe+1): # j=js_pe,je_pe
                    for i in xrange(is_pe, ie_pe+1): # i=is_pe,ie_pe
                        fxa = (-bloc[i,j,k+1] +bloc[i,j,k])/dzw[k]
                        P_diss_iso[i,j,k] = P_diss_iso[i,j,k]  -grav/rho_0*fxa*flux_top[i,j,k]*maskW[i,j,k] \
                                       -grav/rho_0*fxa*K_33[i,j,k]*(salt[i,j,k+1,taup1]-salt[i,j,k,taup1])/dzw[k]*maskW[i,j,k]

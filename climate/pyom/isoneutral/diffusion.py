import numpy as np

from climate.pyom import numerics

def isoneutral_diffusion(tr, istemp, pyom):
    """
    Isopycnal diffusion for tracer,
    following functional formulation by Griffies et al
    Dissipation is calculated and stored in P_diss_iso
    T/S changes are added to dtemp_iso/dsalt_iso
    """

    #  real*8 :: tr(is_:ie_,js_:je_,nz_,3)
    #  logical, intent(in) :: istemp
    #  integer :: i,j,k,kr,ip,jp,km1kr,kpkr,ks
    #  real*8 :: sumz,sumx,sumy
    #  real*8 :: a_tri(nz),b_tri(nz),c_tri(nz),d_tri(nz),delta(nz),sol(nz)
    #  real*8 :: aloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    #  real*8 :: bloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    #  real*8 :: fxa,diffloc

    a_tri = np.zeros(pyom.nz)
    b_tri = np.zeros(pyom.nz)
    c_tri = np.zeros(pyom.nz)
    d_tri = np.zeros(pyom.nz)
    delta = np.zeros(pyom.nz)
    sol = np.zeros(pyom.nz)
    aloc = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    bloc = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    """
    construct total isoneutral tracer flux at east face of "T" cells
    """
    for k in xrange(pyom.nz): #k=1, nz
        for j in xrange(pyom.js_pe, pyom.je_pe+1): #j=js_pe, je_pe
            for i in xrange(pyom.is_pe-1, pyom.ie_pe+1): #i=is_pe-1,ie_pe
                diffloc = 0.25 * (pyom.K_iso[i,j,k] + pyom.K_iso[i,j,max(1,k-1)] + pyom.K_iso[i+1,j,k] + pyom.K_iso[i+1,j,max(1,k-1)])
                sumz = 0.
                for kr in xrange(2): #kr=0,1
                    km1kr = max(k-1+kr,1)
                    kpkr = min(k+kr,pyom.nz-1)
                    for ip in xrange(2):#ip=0,1
                        sumz = sumz + diffloc * pyom.Ai_ez[i,j,k,ip,kr] * (tr[i+ip,j,kpkr,pyom.tau]-tr[i+ip,j,km1kr,pyom.tau])
                pyom.flux_east[i,j,k] = sumz/(4*pyom.dzt[k]) + (tr[i+1,j,k,pyom.tau]-tr[i,j,k,pyom.tau])/(pyom.cost[j]*pyom.dxu[i])*pyom.K_11[i,j,k]
    """
    construct total isoneutral tracer flux at north face of "T" cells
    """
    for k in xrange(pyom.nz): #k=1, nz
        for j in xrange(pyom.js_pe-1, pyom.je_pe+1): #j=js_pe-1, je_pe
            for i in xrange(pyom.is_pe, pyom.ie_pe+1): #i=is_pe,ie_pe
                diffloc = 0.25*(pyom.K_iso[i,j,k]+pyom.K_iso[i,j,max(1,k-1)] + pyom.K_iso[i,j+1,k]+pyom.K_iso[i,j+1,max(1,k-1)])
                sumz = 0.
                for kr in xrange(2): # kr=0,1
                    km1kr = max(k-1+kr,1)
                    kpkr = min(k+kr,pyom.nz-1)
                    for jp in xrange(2): #do jp=0,1
                        sumz = sumz + diffloc * pyom.Ai_nz[i,j,k,jp,kr] * (tr[i,j+jp,kpkr,pyom.tau]-tr[i,j+jp,km1kr,pyom.tau])
                pyom.flux_north[i,j,k] = pyom.cosu[j]*(sumz/(4*pyom.dzt[k]) + (tr[i,j+1,k,pyom.tau]-tr[i,j,k,pyom.tau])/pyom.dyu[j]*pyom.K_22[i,j,k])
    """
    compute the vertical tracer flux "pyom.flux_top" containing the K31
    and K32 components which are to be solved explicitly. The K33
    component will be treated implicitly. Note that there are some
    cancellations of dxu(i-1+ip) and dyu(jrow-1+jp)
    """
    for k in xrange(pyom.nz-1): #k=1,nz-1
        for j in xrange(pyom.js_pe, pyom.je_pe+1): #j=js_pe,je_pe
            for i in xrange(pyom.is_pe, pyom.ie_pe): #i=is_pe,ie_pe
                diffloc = pyom.K_iso[i,j,k]
                sumx = 0.
                for ip in xrange(2): #ip=0,1
                    for kr in xrange(2): #kr=0,1
                        sumx = sumx + diffloc*pyom.Ai_bx[i,j,k,ip,kr]/pyom.cost[j]*(tr[i+ip,j,k+kr,pyom.tau] - tr[i-1+ip,j,k+kr,pyom.tau])
                sumy = 0.
                for jp in xrange(2): #jp=0,1
                    for kr in xrange(2): #kr=0,1
                        sumy = sumy + diffloc*pyom.Ai_by[i,j,k,jp,kr]*pyom.cosu[j-1+jp]*(tr[i,j+jp,k+kr,pyom.tau] - tr[i,j-1+jp,k+kr,pyom.tau])
                pyom.flux_top[i,j,k] = sumx/(4*pyom.dxt[i]) + sumy/(4*pyom.dyt[j]*pyom.cost[j])
    pyom.flux_top[:,:,pyom.nz-1] = 0.0
    """
    add explicit part
    """
    for j in xrange(pyom.js_pe, pyom.je_pe+1): #j=js_pe,je_pe
        for i in xrange(pyom.is_pe, pyom.ie_pe+1): #i=is_pe,ie_pe
            aloc[i,j,:] = pyom.maskT[i,j,:]*( (pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:])/(pyom.cost[j]*pyom.dxt[i]) \
                                    +(pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:])/(pyom.cost[j]*pyom.dyt[j]))
    k = 0
    aloc[:,:,k] = aloc[:,:,k]+pyom.maskT[:,:,k]*pyom.flux_top[:,:,k]/pyom.dzt[k]
    for k in xrange(1, pyom.nz):
        aloc[:,:,k] = aloc[:,:,k] + pyom.maskT[:,:,k] * (pyom.flux_top[:,:,k]- pyom.flux_top[:,:,k-1]) / pyom.dzt[k]

    if istemp:
        pyom.dtemp_iso = pyom.dtemp_iso + aloc
    else:
        pyom.dsalt_iso = pyom.dsalt_iso + aloc

    for j in xrange(pyom.js_pe, pyom.je_pe+1): #j=js_pe,je_pe
        for i in xrange(pyom.is_pe, pyom.ie_pe+1): #i=is_pe,ie_pe
            tr[i,j,:,pyom.taup1] = tr[i,j,:,pyom.taup1]+pyom.dt_tracer*aloc[i,j,:]
    """
    add implicit part
    """
    aloc = tr[:,:,:,pyom.taup1]
    for j in xrange(pyom.js_pe, pyom.je_pe+1): #j=js_pe,je_pe
        for i in xrange(pyom.is_pe, pyom.ie_pe+1): #i=is_pe,ie_pe
            ks = pyom.kbot[i,j] - 1
            if ks >= 0:
                for k in xrange(ks, pyom.nz-1): #k=ks,nz-1
                    delta[k] = pyom.dt_tracer/pyom.dzw[k]*pyom.K_33[i,j,k]
                delta[pyom.nz-1] = 0.0
                for k in xrange(ks+1,pyom.nz): #k=ks+1,nz
                    a_tri[k] = - delta[k-1]/pyom.dzt[k]
                a_tri[ks] = 0.0
                for k in xrange(ks+1, pyom.nz-1): #k=ks+1,nz-1
                    b_tri[k] = 1 + delta[k]/pyom.dzt[k] + delta[k-1]/pyom.dzt[k]
                b_tri[pyom.nz-1] = 1 + delta[pyom.nz-2]/pyom.dzt[pyom.nz-1]
                b_tri[ks] = 1 + delta[ks]/pyom.dzt[ks]
                for k in xrange(ks, pyom.nz-1): #k=ks,nz-1
                    c_tri[k] = -delta[k]/pyom.dzt[k]
                c_tri[pyom.nz-1] = 0.0
                d_tri[ks:] = tr[i,j,ks:,pyom.taup1]
                sol[ks:pyom.nz] = numerics.solve_tridiag(a_tri[ks:],b_tri[ks:],c_tri[ks:],d_tri[ks:],pyom.nz-ks)
                tr[i,j,ks:,pyom.taup1] = sol[ks:]
    if istemp:
         pyom.dtemp_iso = pyom.dtemp_iso + (tr[:,:,:,pyom.taup1]-aloc) / pyom.dt_tracer
    else:
         pyom.dsalt_iso = pyom.dsalt_iso + (tr[:,:,:,pyom.taup1]-aloc) / pyom.dt_tracer
    """
    dissipation by isopycnal mixing
    """
    if pyom.enable_conserve_energy:
        if istemp:
            bloc[:,:,:] = pyom.int_drhodT[:,:,:,pyom.tau]
        else:
            bloc[:,:,:] = pyom.int_drhodS[:,:,:,pyom.tau]
        for k in xrange(pyom.nz): #k=1,nz
            for j in xrange(pyom.js_pe-pyom.onx+1,pyom.je_pe+pyom.onx-1): #j=js_pe-onx+1,je_pe+onx-1
                for i in xrange(pyom.is_pe-pyom.onx+1,pyom.ie_pe+pyom.onx-1): #i=is_pe-onx+1,ie_pe+onx-1
                    fxa = bloc[i,j,k]
                    aloc[i,j,k] = 0.5*pyom.grav/pyom.rho_0*((bloc[i+1,j,k]-fxa)*pyom.flux_east[i,j,k] \
                                                  +(fxa-bloc[i-1,j,k])*pyom.flux_east[i-1,j,k]) / (pyom.dxt[i]*pyom.cost[j]) \
                                + 0.5*pyom.grav/pyom.rho_0*((bloc[i,j+1,k]-fxa)*pyom.flux_north[i,j,k] \
                                                  +(fxa-bloc[i,j-1,k])*pyom.flux_north[i,j-1,k]) / (pyom.dyt[j]*pyom.cost[j])
        """
        dissipation interpolated on W-grid
        """
        for j in xrange(pyom.js_pe-pyom.onx, pyom.je_pe+pyom.onx): #j=js_pe-onx,je_pe+onx
            for i in xrange(pyom.is_pe-pyom.onx, pyom.ie_pe+pyom.onx): #i=is_pe-onx,ie_pe+onx
                ks = pyom.kbot[i,j] - 1
                if ks >= 0:
                    k = ks
                    pyom.P_diss_iso[i,j,k] = pyom.P_diss_iso[i,j,k] + \
                                       0.5*(aloc[i,j,k]+aloc[i,j,k+1]) + 0.5*aloc[i,j,k]*pyom.dzw[max(0,k-1)] / pyom.dzw[k]
                    for k in xrange(ks+1, pyom.nz-2): #k=ks+1,nz-1
                        pyom.P_diss_iso[i,j,k] = pyom.P_diss_iso[i,j,k] + 0.5*(aloc[i,j,k] + aloc[i,j,k+1])
                    k = pyom.nz-1
                    pyom.P_diss_iso[i,j,k] = pyom.P_diss_iso[i,j,k]+ aloc[i,j,k]
        """
        diagnose dissipation of dynamic enthalpy by explicit and implicit vertical mixing
        """
        if istemp:
            for k in xrange(pyom.nz-1): #k=1,nz-1
                for j in xrange(pyom.js_pe, pyom.je_pe+1): #j=js_pe,je_pe
                    for i in xrange(pyom.is_pe, pyom.ie_pe): #i=is_pe,ie_pe
                        fxa = (-bloc[i,j,k+1] + bloc[i,j,k]) / pyom.dzw[k]
                        pyom.P_diss_iso[i,j,k] = pyom.P_diss_iso[i,j,k]  -pyom.grav/pyom.rho_0*fxa*pyom.flux_top[i,j,k]*pyom.maskW[i,j,k] \
                                       -pyom.grav/pyom.rho_0*fxa*pyom.K_33[i,j,k]*(pyom.temp[i,j,k+1,pyom.taup1]-pyom.temp[i,j,k,pyom.taup1])/pyom.dzw[k]*pyom.maskW[i,j,k]
        else:
            for k in xrange(pyom.nz-1): #k=1,nz-1
                for j in xrange(pyom.js_pe, pyom.je_pe+1): #j=js_pe,je_pe
                    for i in xrange(pyom.is_pe, pyom.ie_pe+1): #i=is_pe,ie_pe
                        fxa = (-bloc[i,j,k+1] + bloc[i,j,k]) / pyom.dzw[k]
                        pyom.P_diss_iso[i,j,k] = pyom.P_diss_iso[i,j,k] - pyom.grav/pyom.rho_0*fxa*pyom.flux_top[i,j,k]*pyom.maskW[i,j,k] \
                                       -pyom.grav/pyom.rho_0*fxa*pyom.K_33[i,j,k]*(pyom.salt[i,j,k+1,pyom.taup1]-pyom.salt[i,j,k,pyom.taup1])/pyom.dzw[k]*pyom.maskW[i,j,k]

        return pyom.P_diss_iso, pyom.dtemp_iso, pyom.dsalt_iso




def isoneutral_skew_diffusion(tr,istemp,pyom):
    """
    Isopycnal skew diffusion for tracer,
    following functional formulation by Griffies et al
    Dissipation is calculated and stored in pyom.P_diss_skew
    T/S changes are added to dtemp_iso/dsalt_iso
    """

    aloc = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    bloc = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    """
    construct total isoneutral tracer flux at east face of "T" cells
    """
    for k in xrange(pyom.nz): #k=1,nz
        for j in xrange(pyom.js_pe, pyom.je_pe+1): #j=js_pe,je_pe
            for i in xrange(pyom.is_pe-1, pyom.ie_pe+1): #i=is_pe-1,ie_pe
                diffloc =-0.25*(pyom.K_gm[i,j,k]+pyom.K_gm[i,j,max(1,k-1)] + pyom.K_gm[i+1,j,k]+pyom.K_gm[i+1,j,max(0,k-1)])
                sumz = 0.
                for kr in xrange(2): #kr=0,1
                    km1kr = max(k-1+kr,0)
                    kpkr = min(k+kr,pyom.nz-1)
                    for ip in xrange(2): #ip=0,1
                        sumz = sumz + diffloc*pyom.Ai_ez[i,j,k,ip,kr] *(tr[i+ip,j,kpkr,pyom.tau]-tr[i+ip,j,km1kr,pyom.tau])
                pyom.flux_east[i,j,k] = sumz/(4*pyom.dzt[k]) + (tr[i+1,j,k,pyom.tau]-tr[i,j,k,pyom.tau])/(pyom.cost[j]*pyom.dxu[i]) * pyom.K_11[i,j,k]

    """
    construct total isoneutral tracer flux at north face of "T" cells
    """
    for k in xrange(pyom.nz): #k=1,nz
        for j in xrange(pyom.js_pe-1, pyom.je_pe+1): #j=js_pe-1,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe+1): #i=is_pe,ie_pe
                diffloc = -0.25*(pyom.K_gm[i,j,k]+pyom.K_gm[i,j,max(1,k-1)] + pyom.K_gm[i,j+1,k]+pyom.K_gm[i,j+1,max(1,k-1)])
                sumz = 0.
                for kr in xrange(2): #kr=0,1
                    km1kr = max(k-1+kr,0)
                    kpkr = min(k+kr,pyom.nz-1)
                    for jp in xrange(2): #jp=0,1
                        sumz = sumz + diffloc*pyom.Ai_nz[i,j,k,jp,kr] *(tr[i,j+jp,kpkr,pyom.tau]-tr[i,j+jp,km1kr,pyom.tau])
                pyom.flux_north[i,j,k] = pyom.cosu[j]*(sumz/(4*pyom.dzt[k])+ (tr[i,j+1,k,pyom.tau]-tr[i,j,k,pyom.tau])/pyom.dyu[j]*pyom.K_22[i,j,k])

    """
    compute the vertical tracer flux "flux_top" containing the K31
    and K32 components which are to be solved explicitly.
    """
    for k in xrange(pyom.nz-1): #k=1,nz-1
        for j in xrange(pyom.js_pe, pyom.je_pe+1): #j=js_pe,je_pe
            for i in xrange(pyom.is_pe, pyom.ie_pe+1): #i=is_pe,ie_pe
                diffloc = pyom.K_gm[i,j,k]
                sumx = 0.
                for ip in xrange(2): #ip=0,1
                    for kr in xrange(2): #kr=0,1
                        sumx = sumx + diffloc*pyom.Ai_bx[i,j,k,ip,kr]/pyom.cost[j]*(tr[i+ip,j,k+kr,pyom.tau] - tr[i-1+ip,j,k+kr,pyom.tau])
                sumy    = 0.
                for jp in xrange(2):
                    for kr in xrange(2):
                        sumy = sumy + diffloc*pyom.Ai_by[i,j,k,jp,kr]*pyom.cosu[j-1+jp] * (tr[i,j+jp,k+kr,pyom.tau]-tr[i,j-1+jp,k+kr,pyom.tau])
                pyom.flux_top[i,j,k] = sumx/(4*pyom.dxt[i]) + sumy/(4*pyom.dyt[j]*pyom.cost[j])
    pyom.flux_top[:,:,pyom.nz-1] = 0.0
    """
    add explicit part
    """
    for j in xrange(pyom.js_pe, pyom.je_pe+1): # j=js_pe,je_pe
        for i in xrange(pyom.is_pe, pyom.ie_pe+1): # i=is_pe,ie_pe
            aloc[i,j,:] = pyom.maskT[i,j,:]*( (pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:])/(pyom.cost[j]*pyom.dxt[i]) \
                                        +(pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:])/(pyom.cost[j]*pyom.dyt[j]) )
    k = 0
    aloc[:,:,k] = aloc[:,:,k]+pyom.maskT[:,:,k]*pyom.flux_top[:,:,k]/pyom.dzt[k]
    for k in xrange(1, pyom.nz): # k=2,nz
        aloc[:,:,k] = aloc[:,:,k] + pyom.maskT[:,:,k]*(pyom.flux_top[:,:,k] - pyom.flux_top[:,:,k-1])/pyom.dzt[k]

    if istemp:
         pyom.dtemp_iso = pyom.dtemp_iso + aloc
    else:
         pyom.dsalt_iso = pyom.dsalt_iso + aloc

    for j in xrange(pyom.js_pe, pyom.je_pe+1): #j=js_pe,je_pe
        for i in xrange(pyom.is_pe, pyom.ie_pe+1): #i=is_pe,ie_pe
            tr[i,j,:,pyom.taup1] = tr[i,j,:,pyom.taup1]+pyom.dt_tracer*aloc[i,j,:]

    """
    dissipation by isopycnal mixing
    """
    if pyom.enable_conserve_energy:
        if istemp:
            bloc[:,:,:] = pyom.int_drhodT[:,:,:,pyom.tau]
        else:
            bloc[:,:,:] = pyom.int_drhodS[:,:,:,pyom.tau]

        for k in xrange(pyom.nz): #k=1,nz
            for j in xrange(pyom.js_pe-pyom.onx+1,pyom.je_pe+pyom.onx-1): #j=js_pe-onx+1,je_pe+onx-1
                for i in xrange(pyom.is_pe-pyom.onx+1,pyom.ie_pe+pyom.onx-1): #i=is_pe-onx+1,ie_pe+onx-1
                    fxa = bloc[i,j,k]
                    aloc[i,j,k] = 0.5*pyom.grav/pyom.rho_0*((bloc[i+1,j,k]-fxa)*pyom.flux_east[i,j,k] \
                                                  +(fxa-bloc[i-1,j,k])*pyom.flux_east[i-1,j,k]) / (pyom.dxt[i]*pyom.cost[j]) \
                                  +0.5*pyom.grav/pyom.rho_0*((bloc[i,j+1,k]-fxa)*pyom.flux_north[i,j,k] \
                                                  +(fxa-bloc[i,j-1,k])*pyom.flux_north[i,j-1,k]) / (pyom.dyt[j]*pyom.cost[j])
        """
        dissipation interpolated on W-grid
        """
        for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): #j=js_pe-onx,je_pe+onx
            for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx): #i=is_pe-onx,ie_pe+onx
                ks = pyom.kbot[i,j] - 1
                if ks >= 0:
                    k = ks
                    pyom.P_diss_skew[i,j,k] = pyom.P_diss_skew[i,j,k]+ \
                                    0.5*(aloc[i,j,k]+aloc[i,j,k+1]) + 0.5*aloc[i,j,k]*pyom.dzw[max(1,k-1)]/pyom.dzw[k]
                    for k in xrange(ks+1, pyom.nz-2): #k=ks+1,nz-1
                        pyom.P_diss_skew[i,j,k] = pyom.P_diss_skew[i,j,k] + 0.5*(aloc[i,j,k]  +aloc[i,j,k+1])
                    k = pyom.nz - 1
                    pyom.P_diss_skew[i,j,k] = pyom.P_diss_skew[i,j,k] + aloc[i,j,k]
        """
        dissipation by vertical component of skew mixing
        """
        for k in xrange(pyom.nz-1): #k=1,nz-1
            for j in xrange(pyom.js_pe, pyom.je_pe): #j=js_pe,je_pe
                for i in xrange(pyom.is_pe, pyom.ie_pe): #i=is_pe,ie_pe
                    fxa = (-bloc[i,j,k+1] + bloc[i,j,k])/pyom.dzw[k]
                    pyom.P_diss_skew[i,j,k] = pyom.P_diss_skew[i,j,k] - pyom.grav/pyom.rho_0 * fxa * pyom.flux_top[i,j,k] * pyom.maskW[i,j,k]


def isoneutral_diffusion_all(istemp,pyom):
    """
    Isopycnal diffusion plus skew diffusion for tracer,
    following functional formulation by Griffies et al
    Dissipation is calculated and stored in P_diss_iso
    """

    if enable_skew_diffusion:
        aloc = pyom.K_gm
    else:
        aloc = np.zeros(pyom.ie_pe+pyom.onx-(pyom.is_pe-pyom.onx),pyom.je_pe+pyom.onx-(pyom.js_pe-pyom.onx),pyom.nz)

    """
    construct total isoneutral tracer flux at east face of "T" cells
    """
    for k in xrange(1, pyom.nz+1): #k=1,nz
        for j in xrange(pyom.js_pe, pyom.je_pe+1): #j=js_pe,je_pe
            for i in xrange(pyom.is_pe-1, pyom.ie_pe+1): #i=is_pe-1,ie_pe
                diffloc = 0.25*(pyom.K_iso[i,j,k]+pyom.K_iso[i,j,max(1,k-1)] + pyom.K_iso[i+1,j,k]+pyom.K_iso[i+1,j,max(1,k-1)] ) \
                        - 0.25*(aloc[i,j,k]+aloc[i,j,max(1,k-1)] + aloc[i+1,j,k]+aloc[i+1,j,max(1,k-1)] )
                sumz = 0.
                for kr in xrange(2): #kr=0,1
                    km1kr = max(k-1+kr,1)
                    kpkr  = min(k+kr,pyom.nz)
                    for ip in xrange(2): #ip=0,1
                        sumz = sumz + diffloc*pyom.Ai_ez[i,j,k,ip,kr] *(tr[i+ip,j,kpkr,pyom.tau]-tr[i+ip,j,km1kr,pyom.tau])
                pyom.flux_east[i,j,k] = sumz/(4*pyom.dzt[k]) + (tr[i+1,j,k,pyom.tau]-tr[i,j,k,pyom.tau])/(pyom.cost[j]*pyom.dxu[i]) *pyom.K_11[i,j,k]
    """
    construct total isoneutral tracer flux at north face of "T" cells
    """
    for k in xrange(1, pyom.nz+1): #k=1,nz
        for j in xrange(pyom.js_pe-1,pyom.je_pe+1): #j=js_pe-1,je_pe
            for i in xrange(pyom.is_pe, pyom.ie_pe+1): #i=is_pe,ie_pe
                diffloc = 0.25*(pyom.K_iso[i,j,k]+pyom.K_iso[i,j,max(1,k-1)] + pyom.K_iso[i,j+1,k]+pyom.K_iso[i,j+1,max(1,k-1)] ) \
                        - 0.25*(aloc[i,j,k]+aloc[i,j,max(1,k-1)] + aloc[i,j+1,k]+aloc[i,j+1,max(1,k-1)] )
                sumz    = 0.
                for kr in xrange(2): #kr=0,1
                    km1kr = max(k-1+kr,1)
                    kpkr  = min(k+kr,pyom.nz)
                    for jp in xrange(2): #jp=0,1
                        sumz = sumz + diffloc*pyom.Ai_nz[i,j,k,jp,kr] *(tr[i,j+jp,kpkr,pyom.tau]-tr[i,j+jp,km1kr,pyom.tau])
                pyom.flux_north[i,j,k] = pyom.cosu[j]*( sumz/(4*pyom.dzt[k])+ (tr[i,j+1,k,pyom.tau]-tr[i,j,k,pyom.tau])/pyom.dyu[j]*pyom.K_22[i,j,k])
    """
    compute the vertical tracer flux "flux_top" containing the K31
    and K32 components which are to be solved explicitly. The K33
    component will be treated implicitly. Note that there are some
    cancellations of dxu(i-1+ip) and dyu(jrow-1+jp)
    """
    for k in xrange(1, pyom.nz): #k=1,nz-1
        for j in xrange(pyom.js_pe, pyom.je_pe+1): #j=js_pe,je_pe
            for i in xrange(pyom.is_pe, pyom.ie_pe): #i=is_pe,ie_pe
                diffloc = pyom.K_iso[i,j,k] + aloc[i,j,k]
                sumx = 0.
                for ip in xrange(2): #ip=0,1
                    for kr in xrange(2): #kr=0,1
                        sumx = sumx + diffloc*pyom.Ai_bx[i,j,k,ip,kr]/pyom.cost[j]*(tr[i+ip,j,k+kr,pyom.tau] - tr[i-1+ip,j,k+kr,pyom.tau])
                sumy    = 0.
                for jp in xrange(2): #jp=0,1
                    for kr in xrange(2): #kr=0,1
                        sumy = sumy + diffloc*pyom.Ai_by[i,j,k,jp,kr]*pyom.cosu[j-1+jp]* (tr[i,j+jp,k+kr,pyom.tau]-tr[i,j-1+jp,k+kr,pyom.tau])
                pyom.flux_top[i,j,k] = sumx/(4*pyom.dxt[i]) +sumy/(4*pyom.dyt[j]*pyom.cost[j] )
    pyom.flux_top[:,:,pyom.nz] = 0.0
    """
    add explicit part
    """
    for j in xrange(pyom.js_pe, pyom.je_pe+1): #j=js_pe,je_pe
        for i in xrange(pyom.is_pe, pyom.ie_pe+1): #i=is_pe,ie_pe
            aloc[i,j,:] = pyom.maskT[i,j,:]*( (pyom.flux_east[i,j,:]-  pyom.flux_east[i-1,j,:])/(pyom.cost[j]*pyom.dxt[i]) \
                                       +(pyom.flux_north[i,j,:]- pyom.flux_north[i,j-1,:])/(pyom.cost[j]*pyom.dyt[j]) )
    k=1
    aloc[:,:,k] = aloc[:,:,k]+pyom.maskT[:,:,k]*pyom.flux_top[:,:,k]/pyom.dzt[k]
    for k in xrange(2, pyom.nz+1): #k=2,nz
        aloc[:,:,k] = aloc[:,:,k]+pyom.maskT[:,:,k]*(pyom.flux_top[:,:,k]- pyom.flux_top[:,:,k-1])/pyom.dzt[k]

    if istemp:
         pyom.dtemp_iso = aloc
    else:
         pyom.dsalt_iso = aloc

    for j in xrange(pyom.js_pe, pyom.je_pe+1): #j=js_pe,je_pe
        for i in xrange(pyom.is_pe, pyom.ie_pe+1): #i=is_pe,ie_pe
            tr[i,j,:,pyom.taup1] = tr[i,j,:,pyom.taup1]+pyom.dt_tracer*aloc[i,j,:]

    """
    add implicit part
    """
    aloc = tr[:,:,:,pyom.taup1]
    a_tri=np.zeros(pyom.nz)
    b_tri=np.zeros(pyom.nz)
    c_tri=np.zeros(pyom.nz)
    d_tri=np.zeros(pyom.nz)
    delta=np.zeros(pyom.nz)
    for j in xrange(pyom.js_pe, pyom.je_pe+1): # j=js_pe,je_pe
        for i in xrange(pyom.is_pe, pyom.ie_pe+1): # i=is_pe,ie_pe
            ks = pyom.kbot[i,j]
            if ks > 0:
                for k in xrange(ks, pyom.nz): # k=ks,nz-1
                    delta[k] = pyom.dt_tracer/pyom.dzw(k)*pyom.K_33(i,j,k)
                delta[pyom.nz] = 0.0
                for k in xrange(ks+1, pyom.nz+1): # k=ks+1,nz
                    a_tri[k] = - delta[k-1]/pyom.dzt[k]
                a_tri[ks] = 0.0
                for k in xrange(ks+1, pyom.nz): # k=ks+1,nz-1
                    b_tri[k] = 1+ delta[k]/pyom.dzt[k] + delta[k-1]/pyom.dzt[k]
                b_tri[pyom.nz] = 1+ delta[pyom.nz-1]/pyom.dzt[pyom.nz]
                b_tri[ks] = 1+ delta[ks]/pyom.dzt[ks]
                for k in xrange(ks, pyom.nz): #k=ks,nz-1
                    c_tri[k] = - delta[k]/pyom.dzt[k]
                c_tri[pyom.nz] = 0.0
                d_tri[ks:pyom.nz+1] = tr[i,j,ks:pyom.nz+1,pyom.taup1]
                numerics.solve_tridiag(a_tri[ks:pyom.nz+1],b_tri[ks:pyom.nz+1],c_tri[ks:pyom.nz+1],d_tri[ks:pyom.nz+1],sol[ks:pyom.nz+1],pyom.nz-ks+1)
                tr[i,j,ks:pyom.nz+1,pyom.taup1] = sol[ks:pyom.nz+1]

    if istemp:
         pyom.dtemp_iso = pyom.dtemp_iso + (tr[:,:,:,pyom.taup1]-aloc)/pyom.dt_tracer
    else:
         pyom.dsalt_iso = pyom.dsalt_iso + (tr[:,:,:,pyom.taup1]-aloc)/pyom.dt_tracer
    """
    dissipation by isopycnal mixing
    """
    if pyom.enable_conserve_energy:
        if istemp:
            bloc[:,:,:] = pyom.int_drhodT[:,:,:,pyom.tau]
        else:
            bloc[:,:,:] = pyom.int_drhodS[:,:,:,pyom.tau]

        for k in xrange(1, pyom.nz+1): #k=1,nz
            for j in xrange(pyom.js_pe-pyom.onx+1,pyom.je_pe+pyom.onx): #j=js_pe-onx+1,je_pe+onx-1
                for i in xrange(pyom.is_pe-pyom.onx+1,pyom.ie_pe+pyom.onx): #i=is_pe-onx+1,ie_pe+onx-1
                    fxa = bloc[i,j,k]
                    aloc[i,j,k] = 0.5*pyom.grav/pyom.rho_0*((bloc[i+1,j,k]-fxa)*pyom.flux_east[i,j,k] \
                                                  + (fxa-bloc[i-1,j,k])*pyom.flux_east[i-1,j,k]) /(pyom.dxt[i]*pyom.cost[j]) \
                                  + 0.5*pyom.grav/pyom.rho_0*((bloc[i,j+1,k]-fxa)*pyom.flux_north[i,j,k] \
                                                  + (fxa-bloc[i,j-1,k])*pyom.flux_north[i,j-1,k]) /(pyom.dyt[j]*pyom.cost[j])
        """
        dissipation interpolated on W-grid
        """
        for j in xrange(pyom.js_pe-pyom.onx, pyom.je_pe+pyom.onx+1): # j=js_pe-onx,je_pe+onx
            for i in xrange(pyom.is_pe-pyom.onx, pyom.ie_pe+pyom.onx+1): #i=is_pe-onx,ie_pe+onx
                ks = pyom.kbot[i,j]
                if ks > 0:
                    k = ks
                    pyom.P_diss_iso[i,j,k] = pyom.P_diss_iso[i,j,k]+ \
                                 0.5*(aloc[i,j,k]+aloc[i,j,k+1]) + 0.5*aloc[i,j,k]*pyom.dzw[max(1,k-1)]/pyom.dzw[k]
                    for k in xrange(ks+1, pyom.nz): #k=ks+1,nz-1
                        pyom.P_diss_iso[i,j,k] = pyom.P_diss_iso[i,j,k]+ 0.5*(aloc[i,j,k] +aloc[i,j,k+1])
                    k = pyom.nz
                    pyom.P_diss_iso[i,j,k] = pyom.P_diss_iso[i,j,k] + aloc[i,j,k]
        """
        diagnose dissipation of dynamic enthalpy by explicit and implicit vertical mixing
        """
        if istemp:
            for k in xrange(1, pyom.nz): #k=1,nz-1
                for j in xrange(pyom.js_pe, pyom.je_pe+1): #j=js_pe,je_pe
                    for i in xrange(pyom.is_pe, pyom.ie_pe): # i=is_pe,ie_pe
                        fxa = (-bloc[i,j,k+1] +bloc[i,j,k])/pyom.dzw[k]
                        pyom.P_diss_iso[i,j,k] = pyom.P_diss_iso[i,j,k] - pyom.grav/pyom.rho_0*fxa*pyom.flux_top[i,j,k]*pyom.maskW[i,j,k] \
                                       -pyom.grav/pyom.rho_0*fxa*pyom.K_33[i,j,k]*(pyom.temp[i,j,k+1,pyom.taup1]-pyom.temp[i,j,k,pyom.taup1])/pyom.dzw[k]*pyom.maskW[i,j,k]
        else:
            for k in xrange(1, pyom.nz): # k=1,nz-1
                for j in xrange(pyom.js_pe, pyom.je_pe+1): # j=js_pe,je_pe
                    for i in xrange(pyom.is_pe, pyom.ie_pe+1): # i=is_pe,ie_pe
                        fxa = (-bloc[i,j,k+1] +bloc[i,j,k])/pyom.dzw[k]
                        pyom.P_diss_iso[i,j,k] = pyom.P_diss_iso[i,j,k] - pyom.grav/pyom.rho_0*fxa*pyom.flux_top[i,j,k]*pyom.maskW[i,j,k] \
                                       -pyom.grav/pyom.rho_0*fxa*pyom.K_33[i,j,k]*(pyom.salt[i,j,k+1,pyom.taup1]-pyom.salt[i,j,k,pyom.taup1])/pyom.dzw[k]*pyom.maskW[i,j,k]

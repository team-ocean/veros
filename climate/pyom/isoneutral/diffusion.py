import numpy as np

from climate.pyom import numerics

#@profile
def isoneutral_diffusion(tr, istemp, pyom):
    """
    Isopycnal diffusion for tracer,
    following functional formulation by Griffies et al
    Dissipation is calculated and stored in P_diss_iso
    T/S changes are added to dtemp_iso/dsalt_iso
    """

    #  real*8 :: tr(is_:ie_,js_:je_,nz_,3)
    #  real*8 :: a_tri(nz),b_tri(nz),c_tri(nz),d_tri(nz),delta(nz),sol(nz)
    #  real*8 :: aloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    #  real*8 :: bloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    #  real*8 :: fxa,diffloc

    a_tri = np.zeros((pyom.nx,pyom.ny,pyom.nz))
    b_tri = np.zeros((pyom.nx,pyom.ny,pyom.nz))
    c_tri = np.zeros((pyom.nx,pyom.ny,pyom.nz))
    d_tri = np.zeros((pyom.nx,pyom.ny,pyom.nz))
    delta = np.zeros((pyom.nx,pyom.ny,pyom.nz))
    aloc = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    bloc = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    tr_pad = np.empty((pyom.nx+4,pyom.ny+4,pyom.nz+2))
    tr_pad[:,:,1:-1] = tr[...,pyom.tau]
    tr_pad[:,:,0] = tr[:,:,1,pyom.tau]
    tr_pad[:,:,-1] = tr[:,:,-2,pyom.tau]

    """
    construct total isoneutral tracer flux at east face of "T" cells
    """
    diffloc = np.empty((pyom.nx+1,pyom.ny,pyom.nz))
    diffloc[:,:,1:] = 0.25 * (pyom.K_iso[1:-2,2:-2,1:] + pyom.K_iso[1:-2,2:-2,:-1] + pyom.K_iso[2:-1,2:-2,1:] + pyom.K_iso[2:-1,2:-2,:-1])
    diffloc[:,:,0] = 0.5 * (pyom.K_iso[1:-2,2:-2,0] + pyom.K_iso[2:-1,2:-2,0])
    sumz = np.zeros((pyom.nx+1,pyom.ny,pyom.nz))
    for kr in xrange(2):
        for ip in xrange(2):
            sumz += diffloc * pyom.Ai_ez[1:-2,2:-2,:,ip,kr] * (tr_pad[1+ip:-2+ip,2:-2,1+kr:-1+kr or None] - tr_pad[1+ip:-2+ip,2:-2,kr:-2+kr])
    pyom.flux_east[1:-2,2:-2,:] = sumz / (4.*pyom.dzt[None,None,:]) + (tr[2:-1,2:-2,:,pyom.tau] - tr[1:-2,2:-2,:,pyom.tau]) / (pyom.cost[None,2:-2,None] * pyom.dxu[1:-2,None,None]) * pyom.K_11[1:-2,2:-2,:]

    #for k in xrange(pyom.nz): #k=1, nz
    #    for j in xrange(pyom.js_pe, pyom.je_pe): #j=js_pe, je_pe
    #        for i in xrange(pyom.is_pe-1, pyom.ie_pe): #i=is_pe-1,ie_pe
    #            diffloc = 0.25 * (pyom.K_iso[i,j,k] + pyom.K_iso[i,j,max(0,k-1)] + pyom.K_iso[i+1,j,k] + pyom.K_iso[i+1,j,max(0,k-1)])
    #            sumz = 0.
    #            for kr in xrange(2): #kr=0,1
    #                km1kr = max(k-1+kr,0)
    #                kpkr = min(k+kr,pyom.nz-1)
    #                for ip in xrange(2): #ip=0,1
    #                    sumz += diffloc * pyom.Ai_ez[i,j,k,ip,kr] * (tr[i+ip,j,kpkr,pyom.tau]-tr[i+ip,j,km1kr,pyom.tau])
    #            pyom.flux_east[i,j,k] = sumz/(4*pyom.dzt[k]) + (tr[i+1,j,k,pyom.tau]-tr[i,j,k,pyom.tau])/(pyom.cost[j]*pyom.dxu[i])*pyom.K_11[i,j,k]

    """
    construct total isoneutral tracer flux at north face of "T" cells
    """
    diffloc = np.empty((pyom.nx,pyom.ny+1,pyom.nz))
    diffloc[:,:,1:] = 0.25 * (pyom.K_iso[2:-2,1:-2,1:] + pyom.K_iso[2:-2,1:-2,:-1] + pyom.K_iso[2:-2,2:-1,1:] + pyom.K_iso[2:-2,2:-1,:-1])
    diffloc[:,:,0] = 0.5 * (pyom.K_iso[2:-2,1:-2,0] + pyom.K_iso[2:-2,2:-1,0])
    sumz = np.zeros((pyom.nx,pyom.ny+1,pyom.nz))
    for kr in xrange(2):
        for jp in xrange(2):
            sumz += diffloc * pyom.Ai_nz[2:-2,1:-2,:,jp,kr] * (tr_pad[2:-2,1+jp:-2+jp,1+kr:-1+kr or None] - tr_pad[2:-2,1+jp:-2+jp,kr:-2+kr])
    pyom.flux_north[2:-2,1:-2,:] = pyom.cosu[None,1:-2,None] * (sumz / (4.*pyom.dzt[None,None,:]) + (tr[2:-2,2:-1,:,pyom.tau] - tr[2:-2,1:-2,:,pyom.tau]) / pyom.dyu[None,1:-2,None] * pyom.K_22[2:-2,1:-2,:])

    #for k in xrange(pyom.nz): #k=1, nz
    #    for j in xrange(pyom.js_pe-1, pyom.je_pe): #j=js_pe-1, je_pe
    #        for i in xrange(pyom.is_pe, pyom.ie_pe): #i=is_pe,ie_pe
    #            diffloc = 0.25*(pyom.K_iso[i,j,k]+pyom.K_iso[i,j,max(0,k-1)] + pyom.K_iso[i,j+1,k]+pyom.K_iso[i,j+1,max(0,k-1)])
    #            sumz = 0.
    #            for kr in xrange(2): # kr=0,1
    #                km1kr = max(k-1+kr,0)
    #                kpkr = min(k+kr,pyom.nz-1)
    #                for jp in xrange(2): #do jp=0,1
    #                    sumz += diffloc * pyom.Ai_nz[i,j,k,jp,kr] * (tr[i,j+jp,kpkr,pyom.tau] - tr[i,j+jp,km1kr,pyom.tau])
    #            pyom.flux_north[i,j,k] = pyom.cosu[j] * (sumz / (4*pyom.dzt[k]) + (tr[i,j+1,k,pyom.tau] - tr[i,j,k,pyom.tau]) / pyom.dyu[j] * pyom.K_22[i,j,k])

    """
    compute the vertical tracer flux "pyom.flux_top" containing the K31
    and K32 components which are to be solved explicitly. The K33
    component will be treated implicitly. Note that there are some
    cancellations of dxu(i-1+ip) and dyu(jrow-1+jp)
    """
    diffloc = pyom.K_iso[2:-2,2:-2,:-1]
    sumx = 0.
    for ip in xrange(2):
        for kr in xrange(2):
            sumx += diffloc * pyom.Ai_bx[2:-2,2:-2,:-1,ip,kr] / pyom.cost[None,2:-2,None] \
                    * (tr[2+ip:-2+ip,2:-2,kr:-1+kr or None,pyom.tau] - tr[1+ip:-3+ip,2:-2,kr:-1+kr or None,pyom.tau])
    sumy = 0.
    for jp in xrange(2):
        for kr in xrange(2):
            sumy += diffloc * pyom.Ai_by[2:-2,2:-2,:-1,jp,kr] / pyom.cosu[None,1+jp:-3+jp,None] \
                    * (tr[2:-2,2+jp:-2+jp,kr:-1+kr or None,pyom.tau] - tr[2:-2,1+jp:-3+jp,kr:-1+kr or None, pyom.tau])
    pyom.flux_top[2:-2,2:-2,:-1] = sumx / (4*pyom.dxt[2:-2,None,None]) + sumy / (4*pyom.dyt[None,2:-2,None] * pyom.cost[None,2:-2,None])
    pyom.flux_top[:,:,-1] = 0.

    #for k in xrange(pyom.nz-1): #k=1,nz-1
    #    for j in xrange(pyom.js_pe, pyom.je_pe): #j=js_pe,je_pe
    #        for i in xrange(pyom.is_pe, pyom.ie_pe): #i=is_pe,ie_pe
    #            diffloc = pyom.K_iso[i,j,k]
    #            sumx = 0.
    #            for ip in xrange(2): #ip=0,1
    #                for kr in xrange(2): #kr=0,1
    #                    sumx += diffloc*pyom.Ai_bx[i,j,k,ip,kr]/pyom.cost[j]*(tr[i+ip,j,k+kr,pyom.tau] - tr[i-1+ip,j,k+kr,pyom.tau])
    #            sumy = 0.
    #            for jp in xrange(2): #jp=0,1
    #                for kr in xrange(2): #kr=0,1
    #                    sumy += diffloc*pyom.Ai_by[i,j,k,jp,kr]*pyom.cosu[j-1+jp]*(tr[i,j+jp,k+kr,pyom.tau] - tr[i,j-1+jp,k+kr,pyom.tau])
    #            pyom.flux_top[i,j,k] = sumx/(4*pyom.dxt[i]) + sumy/(4*pyom.dyt[j]*pyom.cost[j])
    #pyom.flux_top[:,:,pyom.nz-1] = 0.0

    """
    add explicit part
    """
    aloc[2:-2,2:-2,:] = pyom.maskT[2:-2,2:-2,:] * ((pyom.flux_east[2:-2,2:-2,:] - pyom.flux_east[1:-3,2:-2,:]) / (pyom.cost[None,2:-2,None] * pyom.dxt[2:-2,None,None]) \
                                                 + (pyom.flux_north[2:-2,2:-2,:] - pyom.flux_north[2:-2,1:-3,:]) / (pyom.cost[None,2:-2,None] * pyom.dyt[None,2:-2,None]))
    aloc[:,:,0] += pyom.maskT[:,:,0] * pyom.flux_top[:,:,0] / pyom.dzt[0]
    aloc[:,:,1:] += pyom.maskT[:,:,1:] * (pyom.flux_top[:,:,1:] - pyom.flux_top[:,:,:-1]) / pyom.dzt[None,None,1:]
    #for j in xrange(pyom.js_pe, pyom.je_pe): #j=js_pe,je_pe
    #    for i in xrange(pyom.is_pe, pyom.ie_pe): #i=is_pe,ie_pe
    #        aloc[i,j,:] = pyom.maskT[i,j,:]*((pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:])/(pyom.cost[j]*pyom.dxt[i]) \
    #                                + (pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:])/(pyom.cost[j]*pyom.dyt[j]))
    #k = 0
    #aloc[:,:,k] += pyom.maskT[:,:,k] * pyom.flux_top[:,:,k] / pyom.dzt[k]
    #for k in xrange(1, pyom.nz):
    #    aloc[:,:,k] += pyom.maskT[:,:,k] * (pyom.flux_top[:,:,k] - pyom.flux_top[:,:,k-1]) / pyom.dzt[k]

    if istemp:
        pyom.dtemp_iso[...] += aloc[...]
    else:
        pyom.dsalt_iso[...] += aloc[...]

    tr[2:-2,2:-2,:,pyom.taup1] += pyom.dt_tracer * aloc[2:-2,2:-2,:]

    #for j in xrange(pyom.js_pe, pyom.je_pe): #j=js_pe,je_pe
    #    for i in xrange(pyom.is_pe, pyom.ie_pe): #i=is_pe,ie_pe
    #        tr[i,j,:,pyom.taup1] = tr[i,j,:,pyom.taup1] + pyom.dt_tracer * aloc[i,j,:]

    """
    add implicit part
    """
    # NOTE: might be wrong
    aloc[...] = tr[:,:,:,pyom.taup1]
    ks = pyom.kbot[2:-2,2:-2] - 1
    ks_mask = np.logical_and(np.indices(delta.shape)[2] == ks[:,:,None], (ks >= 0)[:,:,None])
    land_mask = np.logical_and(np.indices(delta.shape)[2] >= ks[:,:,None], (ks >= 0)[:,:,None])
    if np.count_nonzero(land_mask):
        delta[land_mask] = (pyom.dt_tracer / pyom.dzw[None,None,:] * pyom.K_33[2:-2,2:-2,:])[land_mask]
        delta[:,:,-1] = 0.
        a_tri[:,:,1:] = -delta[:,:,:-1] / pyom.dzt[None,None,1:]
        a_tri[ks_mask] = 0.
        b_tri[:,:,1:-1] = 1 + (delta[:,:,1:-1] + delta[:,:,:-2]) / pyom.dzt[None,None,1:-1]
        b_tri[:,:,-1] = 1 + delta[:,:,-2] / pyom.dzt[None,None,-1]
        b_tri[ks_mask] = 1 + (delta[:,:,:] / pyom.dzt[None,None,0])[ks_mask]
        c_tri[:,:,:-1] = -delta[:,:,:-1] / pyom.dzt[None,None,:-1]
        c_tri[:,:,-1] = 0.
        d_tri[land_mask] = tr[2:-2,2:-2,:,pyom.taup1][land_mask]
        tr[2:-2,2:-2,:,pyom.taup1][land_mask] = numerics.solve_tridiag(a_tri[land_mask],b_tri[land_mask],c_tri[land_mask],d_tri[land_mask])


    #delta = np.zeros(pyom.nz)
    #a_tri = np.zeros(pyom.nz)
    #b_tri = np.zeros(pyom.nz)
    #c_tri = np.zeros(pyom.nz)
    #d_tri = np.zeros(pyom.nz)
    #sol = np.zeros(pyom.nz)
    #aloc[...] = tr[:,:,:,pyom.taup1]
    #for j in xrange(pyom.js_pe, pyom.je_pe): #j=js_pe,je_pe
    #    for i in xrange(pyom.is_pe, pyom.ie_pe): #i=is_pe,ie_pe
    #        ks = pyom.kbot[i,j] - 1
    #        if ks >= 0:
    #            for k in xrange(ks, pyom.nz-1): #k=ks,nz-1
    #                delta[k] = pyom.dt_tracer/pyom.dzw[k]*pyom.K_33[i,j,k]
    #            delta[pyom.nz-1] = 0.0
    #            for k in xrange(ks+1, pyom.nz): #k=ks+1,nz
    #                a_tri[k] = -delta[k-1]/pyom.dzt[k]
    #            a_tri[ks] = 0.0
    #            for k in xrange(ks+1, pyom.nz-1): #k=ks+1,nz-1
    #                b_tri[k] = 1 + delta[k]/pyom.dzt[k] + delta[k-1]/pyom.dzt[k]
    #            b_tri[pyom.nz-1] = 1 + delta[pyom.nz-2]/pyom.dzt[pyom.nz-1]
    #            b_tri[ks] = 1 + delta[ks]/pyom.dzt[ks]
    #            for k in xrange(ks, pyom.nz-1): #k=ks,nz-1
    #                c_tri[k] = -delta[k]/pyom.dzt[k]
    #            c_tri[pyom.nz-1] = 0.0
    #            d_tri[ks:] = tr[i,j,ks:,pyom.taup1]
    #            sol[ks:] = numerics.solve_tridiag(a_tri[ks:],b_tri[ks:],c_tri[ks:],d_tri[ks:])
    #            tr[i,j,ks:,pyom.taup1] = sol[ks:]

    if istemp:
        pyom.dtemp_iso += (tr[:,:,:,pyom.taup1] - aloc) / pyom.dt_tracer
    else:
        pyom.dsalt_iso += (tr[:,:,:,pyom.taup1] - aloc) / pyom.dt_tracer

    """
    dissipation by isopycnal mixing
    """
    if pyom.enable_conserve_energy:
        if istemp:
            bloc[:,:,:] = pyom.int_drhodT[:,:,:,pyom.tau]
        else:
            bloc[:,:,:] = pyom.int_drhodS[:,:,:,pyom.tau]
        aloc[1:-1,1:-1,:] = 0.5 * pyom.grav / pyom.rho_0 * ((bloc[2:,1:-1,:]-bloc[1:-1,1:-1,:]) * pyom.flux_east[1:-1,1:-1,:] \
                                                           +(bloc[1:-1,1:-1,:]-bloc[:-2,1:-1,:]) * pyom.flux_east[:-2,1:-1,:]) \
                                                         / (pyom.dxt[1:-1,None,None] * pyom.cost[None,1:-1,None]) \
                          + 0.5 * pyom.grav / pyom.rho_0 * ((bloc[1:-1,2:,:]-bloc[1:-1,1:-1,:]) * pyom.flux_north[1:-1,1:-1,:] \
                                                           +(bloc[1:-1,1:-1,:]-bloc[1:-1,:-2,:]) * pyom.flux_north[1:-1,:-2,:]) \
                                                         / (pyom.dyt[None,1:-1,None] * pyom.cost[None,1:-1,None])

        #for k in xrange(pyom.nz): #k=1,nz
        #    for j in xrange(pyom.js_pe-pyom.onx+1,pyom.je_pe+pyom.onx-1): #j=js_pe-onx+1,je_pe+onx-1
        #        for i in xrange(pyom.is_pe-pyom.onx+1,pyom.ie_pe+pyom.onx-1): #i=is_pe-onx+1,ie_pe+onx-1
        #            fxa = bloc[i,j,k]
        #            aloc[i,j,k] = 0.5*pyom.grav/pyom.rho_0*((bloc[i+1,j,k]-fxa)*pyom.flux_east[i,j,k] \
        #                                          +(fxa-bloc[i-1,j,k])*pyom.flux_east[i-1,j,k]) / (pyom.dxt[i]*pyom.cost[j]) \
        #                        + 0.5*pyom.grav/pyom.rho_0*((bloc[i,j+1,k]-fxa)*pyom.flux_north[i,j,k] \
        #                                          +(fxa-bloc[i,j-1,k])*pyom.flux_north[i,j-1,k]) / (pyom.dyt[j]*pyom.cost[j])

        """
        dissipation interpolated on W-grid
        """
        ks = pyom.kbot[:,:] - 1
        ks_mask = lambda shift: np.logical_and(np.indices(pyom.P_diss_iso.shape)[2] == np.clip(ks[:,:,None] + shift, 0, pyom.nz-1), (ks >= 0)[:,:,None])
        land_mask = np.logical_and(
                            np.logical_and(np.indices(pyom.P_diss_iso.shape)[2] >= ks[:,:,None]+1, np.indices(pyom.P_diss_iso.shape)[2] < pyom.nz-1)
                            , (ks >= 0)[:,:,None])
        # NOTE: probably wrong at the edges
        pyom.P_diss_iso[ks_mask(0)] += 0.5 * (aloc[ks_mask(0)] + aloc[ks_mask(1)]) + 0.5 * (aloc[:,:,1:] * pyom.dzw[None,None,:-1] / pyom.dzw[None,None,1:])[ks_mask(0)]
        pyom.P_diss_iso[land_mask] += 0.5 * (aloc[:,:,:][land_mask] + aloc[:,:,1:][land_mask])
        pyom.P_diss_iso[:,:,-1] += aloc[:,:,-1]

        #for j in xrange(pyom.js_pe-pyom.onx, pyom.je_pe+pyom.onx): #j=js_pe-onx,je_pe+onx
        #    for i in xrange(pyom.is_pe-pyom.onx, pyom.ie_pe+pyom.onx): #i=is_pe-onx,ie_pe+onx
        #        ks = pyom.kbot[i,j] - 1
        #        if ks >= 0:
        #            k = ks
        #            pyom.P_diss_iso[i,j,k] = pyom.P_diss_iso[i,j,k] + \
        #                               0.5*(aloc[i,j,k]+aloc[i,j,k+1]) + 0.5*aloc[i,j,k]*pyom.dzw[max(0,k-1)] / pyom.dzw[k]
        #            for k in xrange(ks+1, pyom.nz-1): #k=ks+1,nz-1
        #                pyom.P_diss_iso[i,j,k] = pyom.P_diss_iso[i,j,k] + 0.5*(aloc[i,j,k] + aloc[i,j,k+1])
        #            k = pyom.nz-1
        #            pyom.P_diss_iso[i,j,k] = pyom.P_diss_iso[i,j,k]+ aloc[i,j,k]

        """
        diagnose dissipation of dynamic enthalpy by explicit and implicit vertical mixing
        """
        fxa = (-bloc[2:-2,2:-2,1:] + bloc[2:-2,2:-2,:-1]) / pyom.dzw[None,None,:-1]
        if istemp:
            pyom.P_diss_iso[2:-2,2:-2,:-1] += - pyom.grav / pyom.rho_0 * fxa * pyom.flux_top[2:-2,2:-2,:-1] * pyom.maskW[2:-2,2:-2,:-1] \
                                              - pyom.grav / pyom.rho_0 * fxa * pyom.K_33[2:-2,2:-2,:-1] * (pyom.temp[2:-2,2:-2,1:,pyom.taup1] \
                                                                                                         - pyom.temp[2:-2,2:-2,:-1,pyom.taup1]) \
                                                                                        / pyom.dzw[None,None,:-1] * pyom.maskW[2:-2,2:-2,:-1]
            #for k in xrange(pyom.nz-1): #k=1,nz-1
            #    for j in xrange(pyom.js_pe, pyom.je_pe): #j=js_pe,je_pe
            #        for i in xrange(pyom.is_pe, pyom.ie_pe): #i=is_pe,ie_pe
            #            fxa = (-bloc[i,j,k+1] + bloc[i,j,k]) / pyom.dzw[k]
            #            pyom.P_diss_iso[i,j,k] = pyom.P_diss_iso[i,j,k] - pyom.grav/pyom.rho_0*fxa*pyom.flux_top[i,j,k]*pyom.maskW[i,j,k] \
            #                           - pyom.grav/pyom.rho_0*fxa*pyom.K_33[i,j,k]*(pyom.temp[i,j,k+1,pyom.taup1]-pyom.temp[i,j,k,pyom.taup1])/pyom.dzw[k]*pyom.maskW[i,j,k]
        else:
            pyom.P_diss_iso[2:-2,2:-2,:-1] += - pyom.grav / pyom.rho_0 * fxa * pyom.flux_top[2:-2,2:-2,:-1] * pyom.maskW[2:-2,2:-2,:-1] \
                                              - pyom.grav / pyom.rho_0 * fxa * pyom.K_33[2:-2,2:-2,:-1] * (pyom.salt[2:-2,2:-2,1:,pyom.taup1] \
                                                                                                         - pyom.salt[2:-2,2:-2,:-1,pyom.taup1]) \
                                                                                        / pyom.dzw[None,None,:-1] * pyom.maskW[2:-2,2:-2,:-1]
            #for k in xrange(pyom.nz-1): #k=1,nz-1
            #    for j in xrange(pyom.js_pe, pyom.je_pe): #j=js_pe,je_pe
            #        for i in xrange(pyom.is_pe, pyom.ie_pe): #i=is_pe,ie_pe
            #            fxa = (-bloc[i,j,k+1] + bloc[i,j,k]) / pyom.dzw[k]
            #            pyom.P_diss_iso[i,j,k] = pyom.P_diss_iso[i,j,k] - pyom.grav/pyom.rho_0*fxa*pyom.flux_top[i,j,k]*pyom.maskW[i,j,k] \
            #                           - pyom.grav/pyom.rho_0*fxa*pyom.K_33[i,j,k]*(pyom.salt[i,j,k+1,pyom.taup1]-pyom.salt[i,j,k,pyom.taup1])/pyom.dzw[k]*pyom.maskW[i,j,k]


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
        for j in xrange(pyom.js_pe, pyom.je_pe): #j=js_pe,je_pe
            for i in xrange(pyom.is_pe-1, pyom.ie_pe): #i=is_pe-1,ie_pe
                diffloc =-0.25*(pyom.K_gm[i,j,k]+pyom.K_gm[i,j,max(0,k-1)] + pyom.K_gm[i+1,j,k]+pyom.K_gm[i+1,j,max(0,k-1)])
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
        for j in xrange(pyom.js_pe-1, pyom.je_pe): #j=js_pe-1,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): #i=is_pe,ie_pe
                diffloc = -0.25*(pyom.K_gm[i,j,k]+pyom.K_gm[i,j,max(0,k-1)] + pyom.K_gm[i,j+1,k]+pyom.K_gm[i,j+1,max(0,k-1)])
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
        for j in xrange(pyom.js_pe, pyom.je_pe): #j=js_pe,je_pe
            for i in xrange(pyom.is_pe, pyom.ie_pe): #i=is_pe,ie_pe
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
    for j in xrange(pyom.js_pe, pyom.je_pe): # j=js_pe,je_pe
        for i in xrange(pyom.is_pe, pyom.ie_pe): # i=is_pe,ie_pe
            aloc[i,j,:] = pyom.maskT[i,j,:]*((pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:])/(pyom.cost[j]*pyom.dxt[i]) \
                                        + (pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:])/(pyom.cost[j]*pyom.dyt[j]) )
    k = 0
    aloc[:,:,k] = aloc[:,:,k]+pyom.maskT[:,:,k]*pyom.flux_top[:,:,k]/pyom.dzt[k]
    for k in xrange(1, pyom.nz): # k=2,nz
        aloc[:,:,k] = aloc[:,:,k] + pyom.maskT[:,:,k]*(pyom.flux_top[:,:,k] - pyom.flux_top[:,:,k-1])/pyom.dzt[k]

    if istemp:
         pyom.dtemp_iso = pyom.dtemp_iso + aloc
    else:
         pyom.dsalt_iso = pyom.dsalt_iso + aloc

    for j in xrange(pyom.js_pe, pyom.je_pe): #j=js_pe,je_pe
        for i in xrange(pyom.is_pe, pyom.ie_pe): #i=is_pe,ie_pe
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
                    for k in xrange(ks+1, pyom.nz-1): #k=ks+1,nz-1
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

    if pyom.enable_skew_diffusion:
        aloc = pyom.K_gm * np.ones((pyom.nx+4, pyom.ny+4, pyom.nz))
    else:
        aloc = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    bloc = np.zeros((pyom.ny+4, pyom.ny+4, pyom.nz))

    a_tri = np.zeros(pyom.nz)
    b_tri = np.zeros(pyom.nz)
    c_tri = np.zeros(pyom.nz)
    d_tri = np.zeros(pyom.nz)
    delta = np.zeros(pyom.nz)

    """
    construct total isoneutral tracer flux at east face of "T" cells
    """
    for k in xrange(pyom.nz): #k=1,nz
        for j in xrange(pyom.js_pe, pyom.je_pe): #j=js_pe,je_pe
            for i in xrange(pyom.is_pe-1, pyom.ie_pe): #i=is_pe-1,ie_pe
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
    for k in xrange(pyom.nz): #k=1,nz
        for j in xrange(pyom.js_pe-1,pyom.je_pe): #j=js_pe-1,je_pe
            for i in xrange(pyom.is_pe, pyom.ie_pe): #i=is_pe,ie_pe
                diffloc = 0.25*(pyom.K_iso[i,j,k]+pyom.K_iso[i,j,max(1,k-1)] + pyom.K_iso[i,j+1,k]+pyom.K_iso[i,j+1,max(1,k-1)] ) \
                        - 0.25*(aloc[i,j,k]+aloc[i,j,max(1,k-1)] + aloc[i,j+1,k]+aloc[i,j+1,max(1,k-1)] )
                sumz = 0.
                for kr in xrange(2): #kr=0,1
                    km1kr = max(k-1+kr,0)
                    kpkr = min(k+kr,pyom.nz-1)
                    for jp in xrange(2): #jp=0,1
                        sumz = sumz + diffloc*pyom.Ai_nz[i,j,k,jp,kr] *(tr[i,j+jp,kpkr,pyom.tau]-tr[i,j+jp,km1kr,pyom.tau])
                pyom.flux_north[i,j,k] = pyom.cosu[j]*( sumz/(4*pyom.dzt[k])+ (tr[i,j+1,k,pyom.tau]-tr[i,j,k,pyom.tau])/pyom.dyu[j]*pyom.K_22[i,j,k])
    """
    compute the vertical tracer flux "flux_top" containing the K31
    and K32 components which are to be solved explicitly. The K33
    component will be treated implicitly. Note that there are some
    cancellations of dxu(i-1+ip) and dyu(jrow-1+jp)
    """
    for k in xrange(pyom.nz-1): #k=1,nz-1
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
    pyom.flux_top[:,:,pyom.nz-1] = 0.0
    """
    add explicit part
    """
    for j in xrange(pyom.js_pe, pyom.je_pe): #j=js_pe,je_pe
        for i in xrange(pyom.is_pe, pyom.ie_pe): #i=is_pe,ie_pe
            aloc[i,j,:] = pyom.maskT[i,j,:]*( (pyom.flux_east[i,j,:]-  pyom.flux_east[i-1,j,:])/(pyom.cost[j]*pyom.dxt[i]) \
                                       +(pyom.flux_north[i,j,:]- pyom.flux_north[i,j-1,:])/(pyom.cost[j]*pyom.dyt[j]) )
    k=1
    aloc[:,:,k] = aloc[:,:,k]+pyom.maskT[:,:,k]*pyom.flux_top[:,:,k]/pyom.dzt[k]
    for k in xrange(1, pyom.nz): #k=2,nz
        aloc[:,:,k] = aloc[:,:,k]+pyom.maskT[:,:,k]*(pyom.flux_top[:,:,k]- pyom.flux_top[:,:,k-1])/pyom.dzt[k]

    if istemp:
         pyom.dtemp_iso[...] = aloc
    else:
         pyom.dsalt_iso[...] = aloc

    for j in xrange(pyom.js_pe, pyom.je_pe+1): #j=js_pe,je_pe
        for i in xrange(pyom.is_pe, pyom.ie_pe+1): #i=is_pe,ie_pe
            tr[i,j,:,pyom.taup1] = tr[i,j,:,pyom.taup1]+pyom.dt_tracer*aloc[i,j,:]

    """
    add implicit part
    """
    aloc = tr[:,:,:,pyom.taup1]
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
         pyom.dtemp_iso = pyom.dtemp_iso + (tr[:,:,:,pyom.taup1] - aloc) / pyom.dt_tracer
    else:
         pyom.dsalt_iso = pyom.dsalt_iso + (tr[:,:,:,pyom.taup1] - aloc) / pyom.dt_tracer
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
            for k in xrange(1, pyom.nz-1): #k=1,nz-1
                for j in xrange(pyom.js_pe, pyom.je_pe+1): #j=js_pe,je_pe
                    for i in xrange(pyom.is_pe, pyom.ie_pe): # i=is_pe,ie_pe
                        fxa = (-bloc[i,j,k+1] +bloc[i,j,k])/pyom.dzw[k]
                        pyom.P_diss_iso[i,j,k] = pyom.P_diss_iso[i,j,k] - pyom.grav/pyom.rho_0*fxa*pyom.flux_top[i,j,k]*pyom.maskW[i,j,k] \
                                       -pyom.grav/pyom.rho_0*fxa*pyom.K_33[i,j,k]*(pyom.temp[i,j,k+1,pyom.taup1]-pyom.temp[i,j,k,pyom.taup1])/pyom.dzw[k]*pyom.maskW[i,j,k]
        else:
            for k in xrange(1, pyom.nz-1): # k=1,nz-1
                for j in xrange(pyom.js_pe, pyom.je_pe+1): # j=js_pe,je_pe
                    for i in xrange(pyom.is_pe, pyom.ie_pe+1): # i=is_pe,ie_pe
                        fxa = (-bloc[i,j,k+1] +bloc[i,j,k])/pyom.dzw[k]
                        pyom.P_diss_iso[i,j,k] = pyom.P_diss_iso[i,j,k] - pyom.grav/pyom.rho_0*fxa*pyom.flux_top[i,j,k]*pyom.maskW[i,j,k] \
                                       -pyom.grav/pyom.rho_0*fxa*pyom.K_33[i,j,k]*(pyom.salt[i,j,k+1,pyom.taup1]-pyom.salt[i,j,k,pyom.taup1])/pyom.dzw[k]*pyom.maskW[i,j,k]

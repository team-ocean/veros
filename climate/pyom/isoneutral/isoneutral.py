import numpy as np

from climate.pyom import density

def isoneutral_diffusion_pre(pyom):
    """
    Isopycnal diffusion for tracer
    following functional formulation by Griffies et al
    Code adopted from MOM2.1
    """
    # real*8 :: drdTS(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,2)
    # real*8 :: ddzt(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,2)
    # real*8 :: ddxt(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,2)
    # real*8 :: ddyt(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,2)
    # real*8 :: drodxe,drodze,drodyn,drodzn,drodxb,drodyb,drodzb
    # real*8 :: taper,sxe,syn,facty,sumz,sumx,sumy,sxb,syb,dm_taper,diffloc
    # real*8,parameter :: epsln=1.D-20  ! for double precision

    drdTS = np.zeros((pyom.nx+4,pyom.ny+4,pyom.nz,2))
    ddzt = np.zeros((pyom.nx+4,pyom.ny+4,pyom.nz,2))
    ddxt = np.zeros((pyom.nx+4,pyom.ny+4,pyom.nz,2))
    ddyt = np.zeros((pyom.nx+4,pyom.ny+4,pyom.nz,2))
    epsln = 1.e-20  # for double precision

    """
    drho_dt and drho_ds at centers of T cells
    """
    drdTS[:,:,:,0] = density.get_drhodT(pyom.salt[:,:,:,pyom.tau],pyom.temp[:,:,:,pyom.tau],np.abs(pyom.zt),pyom)*pyom.maskT
    drdTS[:,:,:,1] = density.get_drhodS(pyom.salt[:,:,:,pyom.tau],pyom.temp[:,:,:,pyom.tau],np.abs(pyom.zt),pyom)*pyom.maskT
    #for k in xrange(pyom.nz): # k=1,nz
    #    for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx):
    #        for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx):
    #            drdTS[i,j,k,0] = density.get_drhodT(pyom.salt[i,j,k,pyom.tau],pyom.temp[i,j,k,pyom.tau],abs(pyom.zt[k]),pyom)*pyom.maskT[i,j,k]
    #            drdTS[i,j,k,1] = density.get_drhodS(pyom.salt[i,j,k,pyom.tau],pyom.temp[i,j,k,pyom.tau],abs(pyom.zt[k]),pyom)*pyom.maskT[i,j,k]

    """
    gradients at top face of T cells
    """
    ddzt[:,:,:-1,0] = pyom.maskW[:,:,:-1] * (pyom.temp[:,:,1:,pyom.tau] - pyom.temp[:,:,:-1,pyom.tau]) / pyom.dzw[None,None,:-1]
    ddzt[:,:,:-1,1] = pyom.maskW[:,:,:-1] * (pyom.salt[:,:,1:,pyom.tau] - pyom.salt[:,:,:-1,pyom.tau]) / pyom.dzw[None,None,:-1]
    ddzt[...,-1,:] = 0.
    #for k in xrange(pyom.nz-1): # k=1,nz-1
    #    for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx):
    #        for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx):
    #            ddzt[i,j,k,0] = pyom.maskW[i,j,k] * (pyom.temp[i,j,k+1,pyom.tau] - pyom.temp[i,j,k,pyom.tau])/pyom.dzw[k]
    #            ddzt[i,j,k,1] = pyom.maskW[i,j,k] * (pyom.salt[i,j,k+1,pyom.tau] - pyom.salt[i,j,k,pyom.tau])/pyom.dzw[k]
    #ddzt[:,:,pyom.nz-1,:] = 0.

    """
    gradients at eastern face of T cells
    """
    ddxt[:-1,:,:,0] = pyom.maskU[:-1,:,:] * (pyom.temp[1:,:,:,pyom.tau] - pyom.temp[:-1,:,:,pyom.tau]) / (pyom.dxu[:-1,None,None] * pyom.cost[None,:,None])
    ddxt[:-1,:,:,1] = pyom.maskU[:-1,:,:] * (pyom.salt[1:,:,:,pyom.tau] - pyom.salt[:-1,:,:,pyom.tau]) / (pyom.dxu[:-1,None,None] * pyom.cost[None,:,None])
    #for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx):
    #    for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx-1):
    #        ddxt[i,j,:,0] = pyom.maskU[i,j,:]*(pyom.temp[i+1,j,:,pyom.tau]-pyom.temp[i,j,:,pyom.tau]) / (pyom.dxu[i]*pyom.cost[j])
    #        ddxt[i,j,:,1] = pyom.maskU[i,j,:]*(pyom.salt[i+1,j,:,pyom.tau]-pyom.salt[i,j,:,pyom.tau]) / (pyom.dxu[i]*pyom.cost[j])

    """
    gradients at northern face of T cells
    """
    ddyt[:,:-1,:,0] = pyom.maskV[:,:-1,:] * (pyom.temp[:,1:,:,pyom.tau] - pyom.temp[:,:-1,:,pyom.tau]) / pyom.dyu[None,:-1,None]
    ddyt[:,:-1,:,1] = pyom.maskV[:,:-1,:] * (pyom.salt[:,1:,:,pyom.tau] - pyom.salt[:,:-1,:,pyom.tau]) / pyom.dyu[None,:-1,None]
    #for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx-1):
    #    for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx):
    #        ddyt[i,j,:,0] = pyom.maskV[i,j,:]*(pyom.temp[i,j+1,:,pyom.tau] - pyom.temp[i,j,:,pyom.tau]) / pyom.dyu[j]
    #        ddyt[i,j,:,1] = pyom.maskV[i,j,:]*(pyom.salt[i,j+1,:,pyom.tau] - pyom.salt[i,j,:,pyom.tau]) / pyom.dyu[j]

    """
    Compute Ai_ez and K11 on center of east face of T cell.
    """
    diffloc = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    diffloc[1:-2,2:-2,1:] = 0.25 * (pyom.K_iso[1:-2,2:-2,1:] + pyom.K_iso[1:-2,2:-2,:-1] + pyom.K_iso[2:-1,2:-2,1:] + pyom.K_iso[2:-1,2:-2,:-1])
    diffloc[1:-2,2:-2,0] = 0.5 * (pyom.K_iso[1:-2,2:-2,0] + pyom.K_iso[2:-1,2:-2,0])

    sumz = np.zeros((pyom.nx+1, pyom.ny, pyom.nz))
    for kr in xrange(2):
        ki = 0 if kr == 1 else 1
        for ip in xrange(2):
            # drodxe = lambda i,j,k,ip: drdTS[i+ip,j,k,0]*ddxt[i,j,k,0] + drdTS[i+ip,j,k,1]*ddxt[i,j,k,1]
            drodxe = drdTS[1+ip:-2+ip,2:-2,ki:,0]*ddxt[1:-2,2:-2,ki:,0] + drdTS[1+ip:-2+ip,2:-2,ki:,1]*ddxt[1:-2,2:-2,ki:,1]
            # drodze = lambda i,j,k,ip,kr: drdTS[i+ip,j,k,0]*ddzt[i+ip,j,k+kr-1,0] + drdTS[i+ip,j,k,1]*ddzt[i+ip,j,k+kr-1,1]
            drodze = drdTS[1+ip:-2+ip,2:-2,ki:,0]*ddzt[1+ip:-2+ip,2:-2,:-1+kr or None,0] + drdTS[1+ip:-2+ip,2:-2,ki:,1]*ddzt[1+ip:-2+ip,2:-2,:-1+kr or None,1]
            sxe = -drodxe / (np.minimum(0.,drodze)-epsln)
            taper = dm_taper(sxe,pyom.iso_slopec,pyom.iso_dslope)
            sumz[:,:,ki:] += pyom.dzw[None,None,:-1+kr or None] * pyom.maskU[1:-2,2:-2,ki:] * np.maximum(pyom.K_iso_steep, diffloc[1:-2,2:-2,ki:]*taper)
            pyom.Ai_ez[1:-2,2:-2,ki:,ip,kr] = taper * sxe * pyom.maskU[1:-2,2:-2,ki:]
    pyom.K_11[1:-2,2:-2,:] = sumz / (4. * pyom.dzt[None,None,:])

    #for k in xrange(1,pyom.nz): # k=2,nz
    #    for j in xrange(pyom.js_pe,pyom.je_pe): # j=js_pe,je_pe
    #        for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i=is_pe-1,ie_pe
    #            diffloc = 0.25*(pyom.K_iso[i,j,k]+pyom.K_iso[i,j,k-1] + pyom.K_iso[i+1,j,k]+pyom.K_iso[i+1,j,k-1])
    #            sumz = 0.
    #            for kr in xrange(2): # kr=0,1
    #                for ip in xrange(2): # ip=0,1
    #                    sxe = -drodxe(i,j,k,ip)/(min(0.,drodze(i,j,k,ip,kr))-epsln)  #! i+1, k-1
    #                    taper = dm_taper(sxe,pyom.iso_slopec,pyom.iso_dslope)
    #                    sumz = sumz + pyom.dzw[k+kr-1]*pyom.maskU[i,j,k]*max(pyom.K_iso_steep,diffloc*taper)
    #                    pyom.Ai_ez[i,j,k,ip,kr] = taper*sxe*pyom.maskU[i,j,k]
    #            pyom.K_11[i,j,k] = sumz/(4.*pyom.dzt[k])
    #k = 0
    #for j in xrange(pyom.js_pe,pyom.je_pe):
    #    for i in xrange(pyom.is_pe-1,pyom.ie_pe):
    #        diffloc = 0.5*(pyom.K_iso[i,j,k]+ pyom.K_iso[i+1,j,k])
    #        sumz = 0.
    #        kr = 1
    #        for ip in xrange(2): # ip=0,1
    #            sxe  = -drodxe(i,j,k,ip)/(min(0.,drodze(i,j,k,ip,kr))-epsln)
    #            taper = dm_taper(sxe,pyom.iso_slopec,pyom.iso_dslope)
    #            sumz = sumz + pyom.dzw[k+kr-1]*pyom.maskU[i,j,k]*max(pyom.K_iso_steep,diffloc*taper)
    #            pyom.Ai_ez[i,j,k,ip,kr] = taper*sxe*pyom.maskU[i,j,k]
    #        pyom.K_11[i,j,k] = sumz/(4*pyom.dzt[k])

    """
    Compute Ai_nz and K_22 on center of north face of T cell.
    """
    sumz = np.zeros((pyom.nx, pyom.ny+1, pyom.nz))
    for kr in xrange(2):
        ki = 0 if kr == 1 else 1
        for jp in xrange(2):
            # drodyn = lambda i,j,k,jp: drdTS[i,j+jp,k,0]*ddyt[i,j,k,0] + drdTS[i,j+jp,k,1]*ddyt[i,j,k,1]
            drodyn = drdTS[2:-2,1+jp:-2+jp,ki:,0] * ddyt[2:-2,1:-2,ki:,0] + drdTS[2:-2,1+jp:-2+jp,ki:,1] * ddyt[2:-2,1:-2,ki:,1]
            # drodzn = lambda i,j,k,jp,kr: drdTS[i,j+jp,k,0]*ddzt[i,j+jp,k+kr-1,0] + drdTS[i,j+jp,k,1]*ddzt[i,j+jp,k+kr-1,1]
            drodzn = drdTS[2:-2,1+jp:-2+jp,ki:,0] * ddzt[2:-2,1+jp:-2+jp,:-1+kr or None,0] + drdTS[2:-2,1+jp:-2+jp,ki:,1] * ddzt[2:-2,1+jp:-2+jp,:-1+kr or None,1]
            syn = -drodyn / (np.minimum(0.,drodzn) - epsln)
            taper = dm_taper(syn,pyom.iso_slopec,pyom.iso_dslope)
            sumz[:,:,ki:] += pyom.dzw[None,None,:-1+kr or None] * pyom.maskV[2:-2,1:-2,ki:] * np.maximum(pyom.K_iso_steep, diffloc[2:-2,1:-2,ki:]*taper)
            pyom.Ai_nz[2:-2,1:-2,ki:,jp,kr] = taper * syn * pyom.maskV[2:-2,1:-2,ki:]
    pyom.K_22[2:-2,1:-2,:] = sumz / (4.*pyom.dzt[None,None,:])

    #for k in xrange(1,pyom.nz): # k=2,nz
    #    for j in xrange(pyom.js_pe-1,pyom.je_pe):
    #        for i in xrange(pyom.is_pe,pyom.ie_pe):
    #            diffloc = 0.25*(pyom.K_iso[i,j,k]+pyom.K_iso[i,j,k-1] + pyom.K_iso[i,j+1,k]+pyom.K_iso[i,j+1,k-1])
    #            sumz = 0.
    #            for kr in xrange(2): # kr=0,1
    #                for jp in xrange(2): # jp=0,1
    #                    syn = -drodyn(i,j,k,jp)/(min(0.,drodzn(i,j,k,jp,kr))-epsln)
    #                    taper = dm_taper(syn,pyom.iso_slopec,pyom.iso_dslope)
    #                    sumz = sumz + pyom.dzw[k+kr-1] * pyom.maskV[i,j,k]*max(pyom.K_iso_steep,diffloc*taper)
    #                    pyom.Ai_nz[i,j,k,jp,kr] = taper*syn*pyom.maskV[i,j,k]
    #            pyom.K_22[i,j,k] = sumz/(4*pyom.dzt[k])
    #k = 0 # k=1
    #for j in xrange(pyom.js_pe-1,pyom.je_pe):
    #    for i in xrange(pyom.is_pe,pyom.ie_pe):
    #        diffloc = 0.5*(pyom.K_iso[i,j,k] + pyom.K_iso[i,j+1,k])
    #        sumz = 0.
    #        kr=1
    #        for jp in xrange(2): # jp=0,1
    #            syn = -drodyn(i,j,k,jp)/(min(0.,drodzn(i,j,k,jp,kr))-epsln)
    #            taper = dm_taper(syn,pyom.iso_slopec,pyom.iso_dslope)
    #            sumz = sumz + pyom.dzw[k+kr-1] * pyom.maskV[i,j,k] * max(pyom.K_iso_steep,diffloc*taper)
    #            pyom.Ai_nz[i,j,k,jp,kr] = taper*syn*pyom.maskV[i,j,k]
    #        pyom.K_22[i,j,k] = sumz/(4*pyom.dzt[k])

    """
    compute Ai_bx, Ai_by and K33 on top face of T cell.
    """
    # eastward slopes at the top of T cells
    sumx = np.zeros((pyom.nx,pyom.ny,pyom.nz-1))
    for ip in xrange(2):
        for kr in xrange(2):
            #drodxb = lambda i,j,k,ip,kr: drdTS[i,j,k+kr,0]*ddxt[i-1+ip,j,k+kr,0] + drdTS[i,j,k+kr,1]*ddxt[i-1+ip,j,k+kr,1]
            drodxb = drdTS[2:-2,2:-2,kr:-1+kr or None,0] * ddxt[1+ip:-3+ip,2:-2,kr:-1+kr or None,0] + drdTS[2:-2,2:-2,kr:-1+kr or None,1] * ddxt[1+ip:-3+ip,2:-2,kr:-1+kr or None,1]
            #drodzb = lambda i,j,k,kr: drdTS[i,j,k+kr,0]*ddzt[i,j,k,0] + drdTS[i,j,k+kr,1]*ddzt[i,j,k,1]
            drodzb = drdTS[2:-2,2:-2,kr:-1+kr or None,0] * ddzt[2:-2,2:-2,:-1,0] + drdTS[2:-2,2:-2,kr:-1+kr or None,1] * ddzt[2:-2,2:-2,:-1,1]
            sxb = -drodxb / (np.minimum(0.,drodzb) - epsln)
            taper = dm_taper(sxb,pyom.iso_slopec,pyom.iso_dslope)
            sumx += pyom.dxu[1+ip:-3+ip,None,None] * pyom.K_iso[2:-2,2:-2,:-1] * taper * sxb**2 * pyom.maskW[2:-2,2:-2,:-1]
            pyom.Ai_bx[2:-2,2:-2,:-1,ip,kr] = taper * sxb * pyom.maskW[2:-2,2:-2,:-1]

    # northward slopes at the top of T cells
    sumy = np.zeros((pyom.nx,pyom.ny,pyom.nz-1))
    for jp in xrange(2): # jp=0,1
        facty = pyom.cosu[1+jp:-3+jp]*pyom.dyu[1+jp:-3+jp]
        for kr in xrange(2): # kr=0,1
            #drodyb = lambda i,j,k,jp,kr: drdTS[i,j,k+kr,0]*ddyt[i,j-1+jp,k+kr,0] + drdTS[i,j,k+kr,1]*ddyt[i,j-1+jp,k+kr,1]
            drodyb = drdTS[2:-2,2:-2,kr:-1+kr or None,0] * ddyt[2:-2,1+jp:-3+jp,kr:-1+kr or None,0] + drdTS[2:-2,2:-2,kr:-1+kr or None,1] * ddyt[2:-2,1+jp:-3+jp,kr:-1+kr or None,1]
            drodzb = drdTS[2:-2,2:-2,kr:-1+kr or None,0] * ddzt[2:-2,2:-2,:-1,0] + drdTS[2:-2,2:-2,kr:-1+kr or None,1] * ddzt[2:-2,2:-2,:-1,1]
            syb = -drodyb / (np.minimum(0.,drodzb) - epsln)
            taper = dm_taper(syb,pyom.iso_slopec,pyom.iso_dslope)
            sumy += facty[None,:,None] * pyom.K_iso[2:-2,2:-2,:-1] * taper * syb**2 * pyom.maskW[2:-2,2:-2,:-1]
            pyom.Ai_by[2:-2,2:-2,:-1,jp,kr] = taper * syb * pyom.maskW[2:-2,2:-2,:-1]

    pyom.K_33[2:-2,2:-2,:-1] = sumx / (4*pyom.dxt[2:-2,None,None]) + sumy / (4*pyom.dyt[None,2:-2,None] * pyom.cost[None,2:-2,None])
    pyom.K_33[2:-2,2:-2,-1] = 0.

    #for k in xrange(pyom.nz-1): # k=1,nz-1
    #    for j in xrange(pyom.js_pe,pyom.je_pe):
    #        for i in xrange(pyom.is_pe,pyom.ie_pe):
    #            # eastward slopes at the top of T cells
    #            sumx = 0.
    #            for ip in xrange(2): # ip=0,1
    #                for kr in xrange(2): # kr=0,1
    #                    sxb = -drodxb(i,j,k,ip,kr)/(min(0.,drodzb(i,j,k,kr))-epsln) # i-1,k+1
    #                    taper = dm_taper(sxb,pyom.iso_slopec,pyom.iso_dslope)
    #                    sumx = sumx + pyom.dxu[i-1+ip]*pyom.K_iso[i,j,k]*taper*sxb**2  *pyom.maskW[i,j,k]
    #                    pyom.Ai_bx[i,j,k,ip,kr] = taper * sxb * pyom.maskW[i,j,k]
    #            # northward slopes at the top of T cells
    #            sumy = 0.
    #            for jp in xrange(2): # jp=0,1
    #                facty = pyom.cosu[j-1+jp]*pyom.dyu[j-1+jp]
    #                for kr in xrange(2): # kr=0,1
    #                    syb = -drodyb(i,j,k,jp,kr)/(min(0.,drodzb(i,j,k,kr))-epsln)
    #                    taper = dm_taper(syb,pyom.iso_slopec,pyom.iso_dslope)
    #                    sumy = sumy + facty*pyom.K_iso[i,j,k]*taper*syb**2 *pyom.maskW[i,j,k]
    #                    pyom.Ai_by[i,j,k,jp,kr] = taper * syb * pyom.maskW[i,j,k]
    #            pyom.K_33[i,j,k] = sumx/(4*pyom.dxt[i]) + sumy/(4*pyom.dyt[j]*pyom.cost[j])
    #pyom.K_33[:,:,pyom.nz-1] = 0


def isoneutral_diag_streamfunction(pyom):
    """
    calculate hor. components of streamfunction for eddy driven velocity
    for diagnostics purpose only
    """

    #integer :: i,j,k,kr,ip,jp,km1kr,kpkr
    #real*8 :: sumz, diffloc

    """
    meridional component at east face of "T" cells
    """
    for k in xrange(pyom.nz): # k=1,nz
        for j in xrange(pyom.js_pe,pyom.je_pe):
            for i in xrange(pyom.is_pe-1,pyom.ie_pe):
                diffloc = 0.25*(K_gm[i,j,k]+K_gm[i,j,max(0,k-1)] + K_gm[i+1,j,k]+K_gm[i+1,j,max(0,k-1)])
                sumz = 0.
                for kr in xrange(2): # kr=0,1
                    km1kr = max(k-1+kr,0)
                    kpkr  = min(k+kr,pyom.nz-1)
                    for ip in xrange(2): # ip=0,1
                        sumz = sumz + diffloc * pyom.Ai_ez[i,j,k,ip,kr]
                B2_gm[i,j,k] = 0.25*sumz

    """
    zonal component at north face of "T" cells
    """
    for k in xrange(pyom.nz): # k=1,nz
        for j in xrange(pyom.js_pe-1,pyom.je_pe):
            for i in xrange(pyom.is_pe,pyom.ie_pe):
                diffloc = 0.25*(K_gm[i,j,k]+K_gm[i,j,max(0,k-1)] + K_gm[i,j+1,k]+K_gm[i,j+1,max(0,k-1)])
                sumz = 0.
                for kr in xrange(2): # kr=0,1
                    km1kr = max(k-1+kr,0)
                    kpkr = min(k+kr,pyom.nz-1)
                    for jp in xrange(2): # jp=0,1
                        sumz = sumz + diffloc * pyom.Ai_nz[i,j,k,jp,kr]
            B1_gm[i,j,k] = -0.25*sumz


def dm_taper(sx,iso_slopec,iso_dslope):
    """
    tapering function for isopycnal slopes
    """
    return 0.5 * (1. + np.tanh((-np.abs(sx) + iso_slopec) / iso_dslope))


def check_isoneutral_slope_crit(pyom):
    """
    check linear stability criterion from Griffies et al
    """
    epsln = 1e-20
    if pyom.enable_neutral_diffusion:
        ft1 = 1.0/(4.0*pyom.K_iso_0*pyom.dt_tracer + epsln)
        i = pyom.is_pe+pyom.onx
        j = pyom.js_pe+pyom.onx
        k = 0 # k=1
        delta_iso1 = pyom.dzt[k]*ft1*pyom.dxt[i]*np.abs(pyom.cost[j])
        for k in xrange(pyom.nz): # k=1,nz
            for j in xrange(pyom.js_pe,pyom.je_pe):
                for i in xrange(pyom.is_pe,pyom.ie_pe):
                    delta1a = pyom.dxt[i]*np.abs(pyom.cost[j])*pyom.dzt[k]*ft1
                    delta1b = pyom.dyt[j]*pyom.dzt[k]*ft1
                    if delta_iso1 > delta1a or delta_iso1 > delta1b:
                        delta_iso1 = min(delta1a,delta1b)

        print ("diffusion grid factor delta_iso1 = {:.5e}".format(delta_iso1))
        if delta_iso1 < pyom.iso_slopec:
            raise RuntimeError("""
                   Without latitudinal filtering, delta_iso1 is the steepest
                   isoneutral slope available for linear stab of Redi and GM.
                   Maximum allowable isoneutral slope is specified as {}
                   integration will be unstable
                   """.format(pyom.iso_slopec))

import numpy as np

def isoneutral_diffusion_pre(i,j,k,ip,jp,kr):
    """
    Isopycnal diffusion for tracer
    following functional formulation by Griffies et al
    Code adopted from MOM2.1
    """

    epsln = 1.e-20  # for double precision

    """
    statement functions for density triads
    """
    drodxe[i,j,k,ip]    = drdTS[i+ip,j,k,1]*ddxt[i,j,k,1]         + drdTS[i+ip,j,k,2]*ddxt[i,j,k,2]
    drodze[i,j,k,ip,kr] = drdTS[i+ip,j,k,1]*ddzt[i+ip,j,k+kr-1,1] + drdTS[i+ip,j,k,2]*ddzt[i+ip,j,k+kr-1,2]
    drodyn[i,j,k,jp]    = drdTS[i,j+jp,k,1]*ddyt[i,j,k,1]         + drdTS[i,j+jp,k,2]*ddyt[i,j,k,2]
    drodzn[i,j,k,jp,kr] = drdTS[i,j+jp,k,1]*ddzt[i,j+jp,k+kr-1,1] + drdTS[i,j+jp,k,2]*ddzt[i,j+jp,k+kr-1,2]

    drodxb[i,j,k,ip,kr] = drdTS[i,j,k+kr,1]*ddxt[i-1+ip,j,k+kr,1] + drdTS[i,j,k+kr,2]*ddxt[i-1+ip,j,k+kr,2]
    drodyb[i,j,k,jp,kr] = drdTS[i,j,k+kr,1]*ddyt[i,j-1+jp,k+kr,1] + drdTS[i,j,k+kr,2]*ddyt[i,j-1+jp,k+kr,2]
    drodzb[i,j,k,kr]    = drdTS[i,j,k+kr,1]*ddzt[i,j,k,1]         + drdTS[i,j,k+kr,2]*ddzt[i,j,k,2]

    """
    drho_dt and drho_ds at centers of T cells
    """
    for k in xrange(1,nz): # k=1,nz
        for j in xrange(js_pe-onx,je_pe+onx):
            for i in xrange(is_pe-onx,ie_pe+onx):
                drdTS[i,j,k,1] = get_drhodT(salt[i,j,k,tau],temp[i,j,k,tau],abs(zt[k]))*maskT[i,j,k]
                drdTS[i,j,k,2] = get_drhodS(salt[i,j,k,tau],temp[i,j,k,tau],abs(zt[k]))*maskT[i,j,k]

    """
    gradients at top face of T cells
    """
    for k in xrange(0,nz-1): # k=1,nz-1
        for j in xrange(je_pe-onx,je_pe+onx):
            for i in xrange(is_pe-onx,ie_pe+onx):
                ddzt[i,j,k,1] = maskW[i,j,k]* (temp[i,j,k+1,tau] - temp[i,j,k,tau])/dzw[k]
                ddzt[i,j,k,2] = maskW[i,j,k]* (salt[i,j,k+1,tau] - salt[i,j,k,tau])/dzw[k]
    ddzt[:,:,nz,:] = 0

    """
    gradients at eastern face of T cells
    """
    for j in xrange(js_pe-onx,je_pe+onx):
        for i in xrange(is_pe-onx,ie_pe+onx-1):
            ddxt[i,j,:,1] = maskU[i,j,:]*(temp[i+1,j,:,tau]-temp[i,j,:,tau]) / (dxu[i]*cost[j])
            ddxt[i,j,:,2] = maskU[i,j,:]*(salt[i+1,j,:,tau]-salt[i,j,:,tau]) / (dxu[i]*cost[j])

    """
    gradients at northern face of T cells
    """
    for j in xrange(js_pe-onx,je_pe+onx-1):
        for i in xrange(is_pe-onx,ie_pe+onx):
            ddyt[i,j,:,1] = maskV[i,j,:]*(temp[i,j+1,:,tau] - temp[i,j,:,tau]) / dyu[j]
            ddyt[i,j,:,2] = maskV[i,j,:]*(salt[i,j+1,:,tau] - salt[i,j,:,tau]) / dyu[j]

    """
    Compute Ai_ez and K11 on center of east face of T cell.
    """
    for k in xrange(1,nz): # k=2,nz
        for j in xrange(js_pe,je_pe):
            for i in xrange(is_pe-1,ie_pe):
                diffloc = 0.25*(K_iso[i,j,k]+K_iso[i,j,k-1] + K_iso[i+1,j,k]+K_iso[i+1,j,k-1])
            sumz = 0.
            for kr in xrange(2): # kr=0,1
                for ip in xrange(2): # ip=0,1
                    sxe = -drodxe[i,j,k,ip]/(min(zerod0,drodze[i,j,k,ip,kr])-epsln)  # i+1, k-1
                    taper = dm_taper(sxe)
                    sumz = sumz + dzw[k+kr-1]*maskU[i,j,k]*max(K_iso_steep,diffloc*taper)
                    Ai_ez[i,j,k,ip,kr] =  taper*sxe*maskU[i,j,k]
            K_11[i,j,k] = sumz/(4*dzt[k])
    k = 0 # k=1
    for j in xrange(js_pe,je_pe):
        for i in xrange(is_pe-1,ie_pe):
            diffloc = 0.5*(K_iso[i,j,k]+ K_iso[i+1,j,k])
            sumz = 0.
            kr = 1
            for ip in xrange(2): # ip=0,1
                sxe  = -drodxe[i,j,k,ip]/(min(zerod0,drodze[i,j,k,ip,kr])-epsln)
                taper = dm_taper(sxe)
                sumz = sumz + dzw[k+kr-1]*maskU[i,j,k]*max(K_iso_steep,diffloc*taper)
                Ai_ez[i,j,k,ip,kr] = taper*sxe*maskU[i,j,k]
            K_11[i,j,k] = sumz/(4*dzt[k])

    """
    Compute Ai_nz and K_22 on center of north face of T cell.
    """
    for k in xrange(1,nz): # k=2,nz
        for j in xrange(js_pe-1,je_pe):
            for i in xrange(is_pe,ie_pe):
                diffloc = 0.25*(K_iso[i,j,k]+K_iso[i,j,k-1] + K_iso[i,j+1,k]+K_iso[i,j+1,k-1])
                sumz = 0.
                for kr in xrange(2): # kr=0,1
                    for jp in xrange(2): # jp=0,1
                        syn = -drodyn[i,j,k,jp]/(min(zerod0,drodzn[i,j,k,jp,kr])-epsln)
                        taper = dm_taper(syn)
                        sumz = sumz + dzw[k+kr-1] *maskV[i,j,k]*max(K_iso_steep,diffloc*taper)
                        Ai_nz[i,j,k,jp,kr] = taper*syn*maskV[i,j,k]
                K_22[i,j,k] = sumz/(4*dzt[k])
    k = 1
    for j in xrange(js_pe-1,je_pe):
        for i in xrange(is_pe,ie_pe):
            diffloc = 0.5*(K_iso[i,j,k] + K_iso[i,j+1,k])
            sumz = 0.
            kr=1
            for jp in xrange(2): # jp=0,1
                syn = -drodyn[i,j,k,jp]/(min(zerod0,drodzn[i,j,k,jp,kr])-epsln)
                taper = dm_taper(syn)
                sumz = sumz + dzw[k+kr-1] *maskV[i,j,k]*max(K_iso_steep,diffloc*taper)
                Ai_nz[i,j,k,jp,kr] = taper*syn*maskV[i,j,k]
            K_22[i,j,k] = sumz/(4*dzt[k])

    """
    compute Ai_bx, Ai_by and K33 on top face of T cell.
    """
    for k in xrange(0,nz-1): # k=1,nz-1
        for j in xrange(js_pe,je_pe):
            for i in xrange(is_pe,ie_pe):
                # eastward slopes at the top of T cells
                sumx = 0.
                for ip in xrange(2): # ip=0,1
                    for kr in xrange(2): # kr=0,1
                        sxb = -drodxb[i,j,k,ip,kr]/(min(zerod0,drodzb[i,j,k,kr])-epsln) # i-1,k+1
                        taper = dm_taper(sxb)
                        sumx = sumx + dxu[i-1+ip]*K_iso[i,j,k]*taper*sxb**2  *maskW[i,j,k]
                        Ai_bx[i,j,k,ip,kr] =  taper*sxb*maskW[i,j,k]
                # northward slopes at the top of T cells
                sumy = 0.
                for jp in xrange(2): # jp=0,1
                    facty = cosu[j-1+jp]*dyu[j-1+jp]
                    for kr in xrange(2): # kr=0,1
                        syb = -drodyb[i,j,k,jp,kr]/(min(zerod0,drodzb[i,j,k,kr])-epsln)
                        taper = dm_taper(syb)
                        sumy = sumy + facty*K_iso[i,j,k]*taper*syb**2 *maskW[i,j,k]
                        Ai_by[i,j,k,jp,kr] = taper*syb  *maskW[i,j,k]
                K_33[i,j,k] = sumx/(4*dxt[i]) + sumy/(4*dyt[j]*cost[j])
    K_33[:,:,nz] = 0


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
    for k in xrange(0,nz): # k=1,nz
        for j in xrange(js_pe,je_pe):
            for i in xrange(is_pe-1,ie_pe):
                diffloc = 0.25*(K_gm[i,j,k]+K_gm[i,j,max(1,k-1)] + K_gm[i+1,j,k]+K_gm[i+1,j,max(1,k-1)])
                sumz = 0.
                for kr in xrange(2): # kr=0,1
                    km1kr = max(k-1+kr,1)
                    kpkr  = min(k+kr,nz)
                    for ip in xrange(2): # ip=0,1
                        sumz = sumz + diffloc*Ai_ez[i,j,k,ip,kr]
                B2_gm[i,j,k] = 0.25*sumz

    """
    zonal component at north face of "T" cells
    """
    for k in xrange(0,nz): # k=1,nz
        for j in xrange(js_pe-1,je_pe):
            for i in xrange(is_pe,ie_pe):
                diffloc = 0.25*(K_gm[i,j,k]+K_gm[i,j,max(1,k-1)] + K_gm[i,j+1,k]+K_gm[i,j+1,max(1,k-1)])
                sumz = 0.
                for kr in xrange(2): # kr=0,1
                    km1kr = max(k-1+kr,1)
                    kpkr = min(k+kr,nz)
                    for jp in xrange(2): # jp=0,1
                        sumz = sumz + diffloc*Ai_nz[i,j,k,jp,kr]
            B1_gm[i,j,k] = -0.25*sumz


def dm_taper(sx,pyom):
    """
    tapering function for isopycnal slopes
    """
    return 0.5*(1.+tanh((-abs(sx)+iso_slopec)/iso_dslope))


def check_isoneutral_slope_crit(pyom):
    """
    check linear stability criterion from Griffies et al
    """

    #real*8 :: ft1,epsln = 1D-20, delta_iso1,delta1a,delta1b
    #integer :: i,j,k

    if pyom.enable_neutral_diffusion:
        ft1 = 1.0/(4.0*K_iso_0*dt_tracer + epsln)
        i = is_pe+onx; j= js_pe+onx; k = 1
        delta_iso1  = dzt[k]*ft1*dxt[i]*np.abs(cost[j])
        for k in xrange(0,nz): # k=1,nz
            for j in xrange(js_pe,je_pe):
                for i in xrange(is_pe,ie_pe):
                    delta1a = dxt[i]*np.abs(cost[j])*dzt[k]*ft1
                    delta1b = dyt[j]*dzt[k]*ft1
                    if delta_iso1 > delta1a or delta_iso1 > delta1b:
                        delta_iso1 = min(delta1a,delta1b)
    global_min(delta_iso1)

    if my_pe == 0:
        print ("diffusion grid factor delta_iso1 = {}".format(delta_iso1))
        if delta_iso1 < iso_slopec:
            print ("""
                   ERROR:
                   Without latitudinal filtering, delta_iso1 is the steepest
                   isoneutral slope available for linear stab of Redi and GM.
                   Maximum allowable isoneutral slope is specified as {}
                   integration will be unstable
                   """.format(iso_slopec))
            halt_stop(" in check_slop_crit")

from climate.pyom import pyom_method

def adv_flux_superbee_spectral(is_,ie_,js_,je_,np_,adv_fe,adv_fn,adv_ft,var,uvel,vvel,wvel,pyom):
    """
    Calculates advection of a tracer in spectral space
    """
    # integer, intent(in) :: is_,ie_,js_,je_,np_
    # real*8, intent(inout) :: adv_fe(is_:ie_,js_:je_,np_), adv_fn(is_:ie_,js_:je_,np_)
    # real*8, intent(inout) :: adv_ft(is_:ie_,js_:je_,np_),    var(is_:ie_,js_:je_,np_,3)
    # real*8, intent(in) :: uvel(is_:ie_,js_:je_,np_),vvel(is_:ie_,js_:je_,np_),wvel(is_:ie_,js_:je_,np_)
    # integer :: i,j,k,km1,kp2
    # real*8 :: Rjp,Rj,Rjm,uCFL = 0.5,Cr
    # real*8 :: Limiter
    Limiter = lambda Cr: max(0.,max(min(1.,2.*Cr), min(2.,Cr)))

    for k in xrange(2,np-1): # k = 2,np-1
        for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
            for i in xrange(is_pe-1,ie_pe): # i = is_pe-1,ie_pe
                uCFL = abs(uvel[i,j,k]*dt_tracer/(cost[j]*dxt[min(nx,max(1,i))]))
                Rjp = (var[i+2,j,k]-var[i+1,j,k])*maskUp[i+1,j,k]
                Rj = (var[i+1,j,k]-var[i,j,k])*maskUp[i,j,k]
                Rjm = (var[i,j,k]-var[i-1,j,k])*maskUp[i-1,j,k]
                if Rj != 0.:
                    if uvel[i,j,k] > 0:
                        Cr = Rjm/Rj
                    else:
                        Cr = Rjp/Rj
                else:
                    if uvel[i,j,k] > 0:
                        Cr = Rjm*1.E20
                    else:
                        Cr = Rjp*1.E20
                Cr = Limiter(Cr)
                adv_fe[i,j,k] = uvel[i,j,k]*(var[i+1,j,k]+var[i,j,k])*0.5   \
                                            -abs(uvel[i,j,k])*((1.-Cr)+uCFL*Cr)*Rj*0.5

    for k in xrange(2,np-1): # k = 2,np-1
        for j in xrange(js_pe-1,je_pe): # j = js_pe-1,je_pe
            for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
                Rjp = (var[i,j+2,k]-var[i,j+1,k])*maskVp[i,j+1,k]
                Rj = (var[i,j+1,k]-var[i,j,k])*maskVp[i,j,k]
                Rjm = (var[i,j,k]-var[i,j-1,k])*maskVp[i,j-1,k]
                uCFL = abs(vvel[i,j,k]*dt_tracer/dyt[min(ny,max(1,j))])
                if Rj != 0.:
                    if vvel[i,j,k] > 0:
                        Cr = Rjm/Rj
                    else:
                        Cr = Rjp/Rj
                else:
                    if vvel[i,j,k] > 0:
                        Cr = Rjm*1.E20
                    else:
                        Cr = Rjp*1.E20
                Cr = Limiter(Cr)
                adv_fn[i,j,k] = vvel[i,j,k]*(var[i,j+1,k]+var[i,j,k])*0.5   \
                                            -abs(vvel[i,j,k])*((1.-Cr)+uCFL*Cr)*Rj*0.5

    for k in xrange(1,np-1): # k = 1,np-1
        kp2 = k+2
        if kp2 > np:
            kp2 = 3
        km1 = k-1
        if km1 < 1:
            km1 = np-2
        for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
            for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
                Rjp = (var[i,j,kp2,tau]-var[i,j,k+1])*maskWp[i,j,k+1]
                Rj = (var[i,j,k+1]-var[i,j,k])*maskWp[i,j,k]
                Rjm = (var[i,j,k]-var[i,j,km1,tau])*maskWp[i,j,km1]
                uCFL = abs(wvel[i,j,k]*dt_tracer/dphit[k])
                if Rj != 0.:
                    if wvel[i,j,k] > 0:
                        Cr = Rjm/Rj
                    else:
                        Cr = Rjp/Rj
                else:
                    if wvel[i,j,k] > 0:
                        Cr = Rjm*1.E20
                    else:
                        Cr = Rjp*1.E20
                Cr = Limiter(Cr)
                adv_ft[i,j,k] = wvel[i,j,k]*(var[i,j,k+1]+var[i,j,k])*0.5   \
                                            -abs(wvel[i,j,k])*((1.-Cr)+uCFL*Cr)*Rj*0.5


def reflect_flux(is_,ie_,js_,je_,np_,adv_fe,adv_fn,pyom):
    """
    refection boundary condition for advective flux in spectral space
    """
    # integer, intent(in) :: is_,ie_,js_,je_,np_
    # real*8, intent(inout) :: adv_fe(is_:ie_,js_:je_,np_),adv_fn(is_:ie_,js_:je_,np_)
    # integer :: i,j,k,kk
    # real*8 :: flux

    for k in xrange(2,np-1): # k = 2,np-1
        # reflexion at southern boundary
        for j in xrange(js_pe-1,je_pe): # j = js_pe-1,je_pe
            for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
                kk = bc_south[i,j,k]
                if kk > 0:
                    flux = adv_fn[i,j+1,k]
                    adv_fn[i,j,k] = adv_fn[i,j,k]  + flux
                    adv_fn[i,j,kk] = adv_fn[i,j,kk] - flux

        # reflexion at northern boundary
        for j in xrange(js_pe-1,je_pe): # j = js_pe-1,je_pe
            for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
                kk = bc_north[i,j,k]
                if kk > 0:
                    flux = adv_fn[i,j-1,k]
                    adv_fn[i,j,k] = adv_fn[i,j,k]  + flux
                    adv_fn[i,j,kk] = adv_fn[i,j,kk] - flux

        # reflexion at western boundary
        for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
            for i in xrange(is_pe-1,ie_pe): # i = is_pe-1,ie_pe
                kk = bc_west[i,j,k]
                if kk > 0:
                    flux = adv_fe[i+1,j,k]
                    adv_fe[i,j,k] = adv_fe[i,j,k]  + flux
                    adv_fe[i,j,kk] = adv_fe[i,j,kk] - flux

        # reflexion at eastern boundary
        for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
            for i in xrange(is_pe-1,ie_pe): # i = is_pe-1,ie_pe
                kk = bc_east[i,j,k]
                if kk > 0:
                    flux = adv_fe[i-1,j,k]
                    adv_fe[i,j,k] = adv_fe[i,j,k]  + flux
                    adv_fe[i,j,kk] = adv_fe[i,j,kk] - flux

def reflect_ini(pyom):
    """
    initialize indexing for reflection boundary conditions
    """
    # integer :: i,j,k,kk
    # real*8 :: fxa

    if my_pe == 0:
        print("preparing reflection boundary conditions")

    for k in xrange(2,np-1): # k = 2,np-1
        # southern boundary from pi to 2 pi
        if phit[k] >= pyom.pi and phit[k] < 2*pyom.pi:
            fxa = 2*pyom.pi - phit[k]
            if xa < 0.:
                fxa = fxa + 2*pi
            if fxa > 2*pi:
                fxa = fxa - 2*pi
            kk = np.argmin(np.abs(phit - fxa),axis=1) #kk = minloc((phit  - fxa)**2,1)
            for j in xrange(js_pe-1,je_pe): # j = js_pe-1,je_pe
                # where (maskTp(is_pe:ie_pe,j,k) == 0.0 .and. maskTp(is_pe:ie_pe,j+1,k)== 1.0) bc_south(is_pe:ie_pe,j,k) = kk
                bc_south_mask = np.logical_and(maskTp[is_pe:ie_pe,j,k] == 0.0, maskTp[is_pe:ie_pe,j+1,k] == 1.0)
                bc_south[is_pe:ie_pe,j,k][bc_south_mask] = kk

        # northern boundary from 0 to pi
        else:
            fxa = 2*pyom.pi - phit[k]
            if fxa < 0:
                fxa = fxa + 2*pyom.pi
            if fxa > 2*pi:
                fxa = fxa - 2*pyom.pi
            kk = np.argmin(np.abs(phit - fxa),axis=1) #kk = minloc((phit  - fxa)**2,1)
            for j in xrange(js_pe-1,je_pe): # j = js_pe-1,je_pe
                # where (maskTp(is_pe:ie_pe,j,k) == 1.0 .and. maskTp(is_pe:ie_pe,j+1,k)== 0.0) bc_north(is_pe:ie_pe,j,k) = kk
                bc_north_mask = np.logical_and(maskTp[is_pe:ie_pe,j,k] == 1.0, maskTp[is_pe:ie_pe,j+1,k] == 0.0)
                bc_north[is_pe:ie_pe,j,k][bc_north_mask] = kk

        # western boundary from 0.5 pi to 0.75 pi
        if phit[k] >= pyom.pi/2 and phit[k] < 3*pyom.pi/2.:
            fxa = pyom.pi - phit[k]
            if fxa < 0.:
                fxa = fxa + 2*pi
            if fxa > 2*pi:
                fxa = fxa - 2*pi
            kk = np.argmin(np.abs(phit - fxa),axis=1) # kk = minloc((phit  - fxa)**2,1)
            for i in xrange(is_pe-1,ie_pe): # i = is_pe-1,ie_pe
                # where (maskTp(i,js_pe:je_pe,k) == 0.0 .and. maskTp(i+1,js_pe:je_pe,k)== 1.0) bc_west(i,js_pe:je_pe,k) = kk
                bc_west_mask = np.logical_and(maskTp[i,js_pe:je_pe,k] == 0.0, maskTp[i+1,js_pe:je_pe,k] == 1.0)
                bc_west[i,js_pe:je_pe,k][bc_west_mask] = kk

        # eastern boundary from 0 to 0.5 pi and from 0.75 pi to 2 pi
        else:
            fxa = pyom.pi - phit[k]
            if fxa < 0:
                fxa = fxa + 2*pyom.pi
            if fxa > 2*pi:
                fxa = fxa - 2*pyom.pi
            kk = np.argmin(np.abs(phit - fxa),axis=1) # kk = minloc((phit  - fxa)**2,1)
            for i in xrange(is_pe-1,ie_pe): # i = is_pe-1,ie_pe
                # where (maskTp(i,js_pe:je_pe,k) == 1.0 .and. maskTp(i+1,js_pe:je_pe,k)== 0.0) bc_east(i,js_pe:je_pe,k) = kk
                bc_east_mask = np.logical_and(maskTp[i,js_pe:je_pe,k] == 1.0, maskTp[i+1,js_pe:je_pe,k] == 0.0)
                bc_east[i,js_pe:je_pe,k][bc_east_mask] = kk


@pyom_method
def calc_spectral_topo(pyom):
    """
    spectral stuff related to topography
    """
    if pyom.enable_idemix_M2 or pyom.enable_idemix_niw:
        # wavenumber grid
        dphit = 2.*pyom.pi/(np-2)
        dphiu = dphit
        phit[1] = 0.0 - dphit[1]
        phiu[1] = phit[1]+dphit[1]/2.
        for i in xrange(2,np): # i = 2,np
            phit[i] = phit[i-1]+dphit[i]; phiu[i] = phiu[i-1]+dphiu[i]
        # topographic mask for waves
        maskTp = 0.0
        for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
            for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
                if (kbot[i,j] != 0):
                    maskTp[i,j,:] = 1.0
        if pyom.enable_cyclic_x:
            cyclic.setcyclic_xp(maskTp)
        else:
            cyclic.setcyclic_p(maskTp)
        maskUp = maskTp
        for i in xrange(is_pe-onx,ie_pe+onx-1): # i = is_pe-onx,ie_pe+onx-1
            maskUp[i,:,:] = min(maskTp[i,:,:],maskTp[i+1,:,:])
        if pyom.enable_cyclic_x:
            cyclic.setcyclic_xp(maskUp)
        else:
            cyclic.setcyclic_p(maskUp)
        maskVp = maskTp
        for j in xrange(js_pe-onx,je_pe+onx-1): # j = js_pe-onx,je_pe+onx-1
            maskVp[:,j,:] = min(maskTp[:,j,:],maskTp[:,j+1,:])
        if pyom.enable_cyclic_x:
            cyclic.setcyclic_xp(maskVp)
        else:
            cyclic.setcyclic_p(maskVp)
        maskWp = maskTp
        for k in xrange(1,np-1): # k = 1,np-1
            maskWp[:,:,k] = min(maskTp[:,:,k],maskTp[:,:,k+1])
        # precalculate mirrow boundary conditions
        reflect_ini()
        # mark shelf for wave interaction
        get_shelf()

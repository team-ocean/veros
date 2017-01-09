def thermodynamics(pyom):
    """
    integrate temperature and salinity and diagnose sources of dynamic enthalpy
    """

    # integer :: i,j,k,ks
    # real*8 :: a_tri(nz),b_tri(nz),c_tri(nz),d_tri(nz),delta(nz)
    # real*8 :: aloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz),fxa,fxb
    # real*8 :: get_drhodT,get_drhodS

    advect_temperature()
    advect_salinity()

    if pyom.enable_conserve_energy:
        """
        advection of dynamic enthalpy
        """
        if pyom.enable_superbee_advection:
            adv_flux_superbee(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_east,flux_north,flux_top,Hd[:,:,:,tau])
        else:
            adv_flux_2nd(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_east,flux_north,flux_top,Hd[:,:,:,tau])

        for j in xrange(js_pe,je_pe):
            for i in xrange(is_pe,ie_pe):
                dHd[i,j,:,tau]=maskT[i,j,:] * (-(flux_east[i,j,:] - flux_east[i-1,j,:])/(cost(j)*dxt(i)) \
                                               -(flux_north[i,j,:] - flux_north[i,j-1,:])/(cost(j)*dyt(j)))

        k = 0 # k=1
        dHd[:,:,k,tau] = dHd[:,:,k,tau]-maskT[:,:,k]*flux_top[:,:,k]/dzt(k)
        for k in xrange(1,nz): # k=2,nz
            dHd[:,:,k,tau] = dHd[:,:,k,tau] - maskT[:,:,k] * (flux_top[:,:,k] - flux_top[:,:,k-1])/dzt(k)

        """
        changes in dyn. Enthalpy due to advection
        """
        for k in xrange(0,nz): # k=1,nz
            for j in xrange(js_pe,je_pe):
                for i in xrange(is_pe,ie_pe):
                    fxa = grav/rho_0 * (-int_drhodT[i,j,k,tau]*dtemp[i,j,k,tau] - int_drhodS[i,j,k,tau]*dsalt[i,j,k,tau])
                    aloc[i,j,k] = fxa - dHd[i,j,k,tau]

        """
        contribution by vertical advection is - g rho w /rho0, substract this also
        """
        for k in xrange(0,nz-1): # k=1,nz-1
            aloc[:,:,k] = aloc[:,:,k] - 0.25*grav/rho_0*w[:,:,k,tau]*(rho[:,:,k,tau]+rho[:,:,k+1,tau])*dzw(k)/dzt(k)
        for k in xrange(1,nz): # k=2,nz
            aloc[:,:,k] = aloc[:,:,k] - 0.25*grav/rho_0*w[:,:,k-1,tau]*(rho[:,:,k,tau]+rho[:,:,k-1,tau])*dzw(k-1)/dzt(k)

    if pyom.enable_conserve_energy and pyom.enable_tke:
        """
        dissipation by advection interpolated on W-grid
        """
        for j in xrange(js_pe,je_pe):
            for i in xrange(is_pe,ie_pe):
                ks = kbot[i,j]
                if ks > 0:
                    k = ks
                    P_diss_adv[i,j,k] = 0.5*(aloc[i,j,k]+aloc[i,j,k+1]) + 0.5*aloc[i,j,k]*dzw[max(1,k-1)]/dzw[k] # max(1,k-1) probably not correct
                    for k in xrange(ks+1,nz-1): # k=ks+1,nz-1
                        P_diss_adv[i,j,k] = 0.5 * (aloc[i,j,k] + aloc[i,j,k+1])
                    k = nz
                    P_diss_adv[i,j,k] = aloc[i,j,k]

        """
        distribute P_diss_adv over domain, prevent draining of TKE
        """
        fxa = 0
        fxb = 0
        for k in xrange(0,nz-1): # k=1,nz-1
            for j in xrange(js_pe,je_pe):
                for i in xrange(is_pe,ie_pe):
                    fxa = fxa + area_t[i,j]*P_diss_adv[i,j,k]*dzw[k]*maskW[i,j,k]
                    if tke[i,j,k,tau] > .0:
                        fxb = fxb + area_t[i,j]*dzw[k]*maskW[i,j,k]
        k = nz
        for j in xrange(js_pe,je_pe):
            for i in xrange(is_pe,ie_pe):
                fxa = fxa + 0.5*area_t[i,j] * P_diss_adv[i,j,k] * dzw[k] * maskW[i,j,k]
                fxb = fxb + 0.5*area_t[i,j] * dzw[k] * maskW[i,j,k]
        global_sum(fxa)
        global_sum(fxb)
        P_diss_adv = 0.0
        for k in xrange(0,nz): # k=1,nz
            for j in xrange(js_pe,je_pe):
                for i in xrange(is_pe,ie_pe):
                    if tke[i,j,k,tau] > .0 or k == nz:
                        P_diss_adv[i,j,k] = fxa / fxb

    """
    Adam Bashforth time stepping for advection
    """
    temp[:,:,:,taup1] = temp[:,:,:,tau] + dt_tracer * ((1.5+AB_eps)*dtemp[:,:,:,tau] - (0.5+AB_eps)*dtemp[:,:,:,taum1]) * maskT
    salt[:,:,:,taup1] = salt[:,:,:,tau] + dt_tracer * ((1.5+AB_eps)*dsalt[:,:,:,tau] - (0.5+AB_eps)*dsalt[:,:,:,taum1]) * maskT

    """
    horizontal diffusion
    """
    tic("iso")
    if pyom.enable_hor_diffusion:
        tempsalt_diffusion()
    else:
        tempsalt_biharmonic()

    """
    sources like restoring zones, etc
    """
    if pyom.enable_tempsalt_sources:
        tempsalt_sources()

    """
    isopycnal diffusion
    """
    if pyom.enable_neutral_diffusion:
        P_diss_iso = 0.0
        dtemp_iso = 0.0
        dsalt_iso = 0.0
        isoneutral_diffusion_pre()
        isoneutral_diffusion(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,temp,True)
        isoneutral_diffusion(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,salt,False)
        if pyom.enable_skew_diffusion:
            P_diss_skew = 0.0
            isoneutral_skew_diffusion(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,temp,True)
            isoneutral_skew_diffusion(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,salt,False)
    toc("iso")

    tic("vmix")
    """
    vertical mixing of temperature and salinity
    """
    dtemp_vmix = temp[:,:,:,taup1]
    dsalt_vmix = salt[:,:,:,taup1]
    a_tri = 0.0
    b_tri = 0.0
    c_tri = 0.0
    d_tri = 0.0
    delta = 0.0
    for j in xrange(js_pe,je_pe):
        for i in xrange(is_pe,ie_pe):
            ks = kbot[i,j]
            if ks > 0:
                for k in xrange(ks,nz-1):
                    delta[k] = dt_tracer/dzw[k] * kappaH[i,j,k]
                delta[nz] = 0.0
                for k in xrange(ks+1,nz):
                    a_tri[k] = -delta[k-1]/dzt[k]
                a_tr[ks] = 0.0
                for k in xrange(ks+1,nz-1):
                    b_tri[k] = 1 + delta[k]/dzt[k] + delta[k-1]/dzt[k]
                b_tri[nz] = 1 + delta[nz-1]/dzt[nz]
                b_tri[ks] = 1 + delta[ks]/dzt[ks]
                for k in xrange(ks,nz-1):
                    c_tri[k] = -delta[k]/dzt[k]
                c_tri[nz] = 0.0
                d_tri[ks:nz] = temp[i,j,ks:nz,taup1]
                d_tri[nz] = d_tri[nz] + dt_tracer * forc_temp_surface[i,j] / dzt[nz]
                solve_tridiag(a_tri[ks:nz],b_tri[ks:nz],c_tri[ks:nz],d_tri[ks:nz],temp[i,j,ks:nz,taup1],nz-ks+1)
                d_tri[ks:nz] = salt[i,j,ks:nz,taup1]
                d_tri[nz] = d_tri[nz] + dt_tracer * forc_salt_surface[i,j] / dzt[nz]
                solve_tridiag(a_tri[ks:nz],b_tri[ks:nz],c_tri[ks:nz],d_tri[ks:nz],salt[i,j,ks:nz,taup1],nz-ks+1)
    dtemp_vmix = (temp[:,:,:,taup1] - dtemp_vmix) / dt_tracer
    dsalt_vmix = (salt[:,:,:,taup1] - dsalt_vmix) / dt_tracer
    toc("vmix")

    """
    boundary exchange
    """
    border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,temp[:,:,:,taup1])
    setcyclic_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,temp[:,:,:,taup1])
    border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,salt[:,:,:,taup1])
    setcyclic_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,salt[:,:,:,taup1])

    tic("eq_of_state")
    calc_eq_of_state(taup1)
    toc("eq_of_state")

    """
    surface density flux
    """
    for j in xrange(js_pe-onx,je_pe+onx):
        for i in xrange(is_pe-onx,ie_pe+onx):
            forc_rho_surface[i,j] = (
                                        get_drhodT(salt[i,j,nz,taup1],temp[i,j,nz,taup1],abs(zt[nz])) * forc_temp_surface[i,j] \
                                      + get_drhodS(salt[i,j,nz,taup1],temp[i,j,nz,taup1],abs(zt[nz])) * forc_salt_surface[i,j] \
                                    ) * maskT[i,j,nz]

    tic("vmix")
    P_diss_v = 0.0
    if pyom.enable_conserve_energy:
        """
        diagnose dissipation of dynamic enthalpy by vertical mixing
        """
        for k in xrange(0,nz-1): # k=1,nz-1
            for j in xrange(js_pe,je_pe):
                for i in xrange(is_pe,ie_pe):
                    fxa = (-int_drhodT[i,j,k+1,taup1] + int_drhodT[i,j,k,taup1]) / dzw[k]
                    P_diss_v[i,j,k] = P_diss_v[i,j,k] \
                                      - grav/rho_0 * fxa * kappaH[i,j,k] * (temp[i,j,k+1,taup1] - temp[i,j,k,taup1]) / dzw[k] * maskW[i,j,k]
                    fxa = (-int_drhodS[i,j,k+1,taup1] + int_drhodS[i,j,k,taup1]) / dzw[k]
                    P_diss_v[i,j,k] = P_diss_v[i,j,k] &
                                      - grav/rho_0 * fxa * kappaH[i,j,k] * (salt[i,j,k+1,taup1] - salt[i,j,k,taup1]) / dzw[k] * maskW[i,j,k]
        k = nz
        for j in xrange(js_pe,je_pe):
            for i in xrange(is_pe, ie_pe):
                fxa = 2 * int_drhodT[i,j,k,taup1] / dzw[k]
                P_diss_v[i,j,k] = P_diss_v[i,j,k] - grav/rho_0 * fxa * forc_temp_surface[i,j] * maskW[i,j,k]
                fxa = 2 * int_drhodS[i,j,k,taup1] / dzw[k]
                P_diss_v[i,j,k]= P_diss_v[i,j,k] - grav/rho_0 * fxa * forc_salt_surface[i,j] * maskW[i,j,k]


    if pyom.enable_conserve_energy:
        """
        determine effect due to nonlinear equation of state
        """
        aloc[:,:,1:nz-1] = kappaH[:,:,1:nz-1] * Nsqr[:,:,1:nz-1,taup1]
        P_diss_nonlin[:,:,1:nz-1] = P_diss_v[:,:,1:nz-1] - aloc[:,:,1:nz-1]
        P_diss_v[:,:,1:nz-1] = aloc[:,:,1:nz-1]
    else:
        """
        diagnose N^2 kappaH, i.e. exchange of pot. energy with TKE
        """
        P_diss_v[:,:,1:nz-1] = kappaH[:,:,1:nz-1] * Nsqr[:,:,1:nz-1,taup1]
        P_diss_v[:,:,nz] = -forc_rho_surface[:,:] * maskT[:,:,nz] * grav/rho_0
    toc("vmix")


def advect_tracer(is_,ie_,js_,je_,nz_,tr,dtr,pyom):
    """
    calculate time tendency of a tracer due to advection
    """
    # integer, intent(in) :: is_,ie_,js_,je_,nz_
    # real*8, intent(inout) :: dtr(is_:ie_,js_:je_,nz_),tr(is_:ie_,js_:je_,nz_)
    # integer :: i,j,k

    if pyom.enable_superbee_advection:
        adv_flux_superbee(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_east,flux_north,flux_top,tr)
    else:
        adv_flux_2nd(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_east,flux_north,flux_top,tr)
    for j in xrange(js_pe,je_pe):
        for i in xrange(is_pe,ie_pe):
            dtr[i,j,:] = maskT[i,j,:] * (-(flux_east[i,j,:] -  flux_east[i-1,j,:]) / (cost[j]*dxt[i]) \
                                         -(flux_north[i,j,:] - flux_north[i,j-1,:]) / (cost[j]*dyt[j]))
    k = 0 # k=1
    dtr[:,:,k] = dtr[:,:,k] - maskT[:,:,k] * flux_top[:,:,k] / dzt[k]
    for k in xrange(1,nz): # k=2,nz
        dtr[:,:,k] = dtr[:,:,k] - maskT[:,:,k] * (flux_top[:,:,k] - flux_top[:,:,k-1]) / dzt[k]


def advect_temperature(pyom):
    """
    integrate temperature
    """
    if pyom.enable_superbee_advection:
        adv_flux_superbee(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_east,flux_north,flux_top,temp[:,:,:,tau])
    else:
        adv_flux_2nd(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_east,flux_north,flux_top,temp[:,:,:,tau])
    for j in xrange(js_pe,je_pe):
        for i in xrange(is_pe,ie_pe):
            dtemp[i,j,:,tau] = maskT[i,j,:] * (-(flux_east[i,j,:] - flux_east[i-1,j,:]) / (cost[j]*dxt[i]) \
                                               -(flux_north[i,j,:] - flux_north[i,j-1,:]) / (cost[j]*dyt[j]))
    k = 0 # k=1
    dtemp[:,:,k,tau] = dtemp[:,:,k,tau] - maskT[:,:,k] * flux_top[:,:,k] / dzt[k]
    for k in xrange(1,nz):
        dtemp[:,:,k,tau] = dtemp[:,:,k,tau] - maskT[:,:,k] * (flux_top[:,:,k] - flux_top[:,:,k-1]) / dzt[k]


def advect_salinity(pyom):
    """
    integrate salinity
    """
    if pyom.enable_superbee_advection:
        adv_flux_superbee(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_east,flux_north,flux_top,salt[:,:,:,tau])
    else:
        adv_flux_2nd(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_east,flux_north,flux_top,salt[:,:,:,tau])
    for j in xrange(js_pe,je_pe):
        for i in xrange(is_pe,ie_pe):
            dsalt[i,j,:,tau] = maskT[i,j,:] * (-(flux_east[i,j,:] - flux_east[i-1,j,:]) / (cost[j]*dxt[i]) \
                                               -(flux_north[i,j,:] - flux_north[i,j-1,:]) / (cost[j]*dyt[j]))
    k = 0 # k=1
    dsalt[:,:,k,tau] = dsalt[:,:,k,tau] - maskT[:,:,k] * flux_top[:,:,k] / dzt[k]
    for k in xrange(1,nz):
        dsalt[:,:,k,tau] = dsalt[:,:,k,tau] - maskT[:,:,k] * (flux_top[:,:,k] - flux_top[:,:,k-1]) / dzt[k]


def calc_eq_of_state(pyom,n):
    """
    calculate density, stability frequency, dynamic enthalpy and derivatives
    for time level n from temperature and salinity
    """

    # integer, intent(in) :: n
    # integer :: i,j,k
    # real*8 :: get_rho,get_int_drhodT,get_int_drhodS, get_dyn_enthalpy
    # real*8 :: fxa

    """
    calculate new density
    """
    for k in xrange(0,nz): # k=1,nz
        for j in xrange(js_pe-onx,je_pe+onx):
            for i in xrange(is_pe-onx,ie_pe+onx):
                rho[i,j,k,n] = get_rho(salt[i,j,k,n],temp[i,j,k,n],abs(zt[k])) * maskT[i,j,k]

    if pyom.enable_conserve_energy:
        """
        calculate new dynamic enthalpy and derivatives
        """
        for k in xrange(0,nz): # k=1,nz
            for j in xrange(js_pe-onx,je_pe+onx):
                for i in xrange(is_pe-onx,ie_pe+onx):
                    Hd[i,j,k,n] = get_dyn_enthalpy(salt[i,j,k,n],temp[i,j,k,n],abs(zt[k])) * maskT[i,j,k]
                    int_drhodT[i,j,k,n] = get_int_drhodT(salt[i,j,k,n],temp[i,j,k,n],abs(zt[k]))
                    int_drhodS[i,j,k,n] = get_int_drhodS(salt[i,j,k,n],temp[i,j,k,n],abs(zt[k]))

    """
    new stability frequency
    """
    for k in xrange(0,nz-1): # k=1,nz-1
        for j in xrange(js_pe-onx,je_pe+onx):
            for i in xrange(is_pe-onx,ie_pe+onx):
                fxa = -grav / rho_0 / dzw[k] * maskW[i,j,k]
                Nsqr[i,j,k,n] = fxa * (get_rho(salt[i,j,k+1,n],temp[i,j,k+1,n],abs(zt[k]))-rho[i,j,k,n])
    Nsqr[:,:,nz,n] = Nsqr[:,:,nz-1,n]

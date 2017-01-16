import numpy as np

from climate.pyom import advection, diffusion, isoneutral, cyclic
from climate.pyom import numerics, density

def thermodynamics(pyom):
    """
    integrate temperature and salinity and diagnose sources of dynamic enthalpy
    """

    # integer :: i,j,k,ks
    # real*8 :: a_tri(nz),b_tri(nz),c_tri(nz),d_tri(nz),delta(nz)
    # real*8 :: aloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz),fxa,fxb
    # real*8 :: get_drhodT,get_drhodS

    a_tri = np.zeros(pyom.nz)
    b_tri = np.zeros(pyom.nz)
    c_tri = np.zeros(pyom.nz)
    d_tri = np.zeros(pyom.nz)
    delta = np.zeros(pyom.nz)
    aloc = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    bloc = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    advect_temperature(pyom)
    advect_salinity(pyom)

    if pyom.enable_conserve_energy:
        """
        advection of dynamic enthalpy
        """
        if pyom.enable_superbee_advection:
            advection.adv_flux_superbee(pyom.flux_east,pyom.flux_north,pyom.flux_top,pyom.Hd[:,:,:,pyom.tau],pyom)
        else:
            advection.adv_flux_2nd(pyom.flux_east,pyom.flux_north,pyom.flux_top,pyom.Hd[:,:,:,pyom.tau],pyom)

        for j in xrange(pyom.js_pe,pyom.je_pe):
            for i in xrange(pyom.is_pe,pyom.ie_pe):
                pyom.dHd[i,j,:,pyom.tau] = pyom.maskT[i,j,:] * (-(pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:])/(pyom.cost[j]*pyom.dxt[i]) \
                                               -(pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:])/(pyom.cost[j]*pyom.dyt[j]))

        k = 0 # k=1
        pyom.dHd[:,:,k,pyom.tau] = pyom.dHd[:,:,k,pyom.tau] - pyom.maskT[:,:,k] * pyom.flux_top[:,:,k] / pyom.dzt[k]
        for k in xrange(1,pyom.nz): # k=2,nz
            pyom.dHd[:,:,k,pyom.tau] = pyom.dHd[:,:,k,pyom.tau] - pyom.maskT[:,:,k] * (pyom.flux_top[:,:,k] - pyom.flux_top[:,:,k-1])/pyom.dzt[k]

        """
        changes in dyn. Enthalpy due to advection
        """
        for k in xrange(pyom.nz): # k=1,nz
            for j in xrange(pyom.js_pe,pyom.je_pe):
                for i in xrange(pyom.is_pe,pyom.ie_pe):
                    fxa = pyom.grav/pyom.rho_0 * (-pyom.int_drhodT[i,j,k,pyom.tau]*pyom.dtemp[i,j,k,pyom.tau] - pyom.int_drhodS[i,j,k,pyom.tau]*pyom.dsalt[i,j,k,pyom.tau])
                    aloc[i,j,k] = fxa - pyom.dHd[i,j,k,pyom.tau]

        """
        contribution by vertical advection is - g rho w /rho0, substract this also
        """
        for k in xrange(pyom.nz-1): # k=1,nz-1
            aloc[:,:,k] = aloc[:,:,k] - 0.25*pyom.grav/pyom.rho_0*pyom.w[:,:,k,pyom.tau]*(pyom.rho[:,:,k,pyom.tau]+pyom.rho[:,:,k+1,pyom.tau])*pyom.dzw[k]/pyom.dzt[k]
        for k in xrange(1,pyom.nz): # k=2,nz
            aloc[:,:,k] = aloc[:,:,k] - 0.25*pyom.grav/pyom.rho_0*pyom.w[:,:,k-1,pyom.tau]*(pyom.rho[:,:,k,pyom.tau]+pyom.rho[:,:,k-1,pyom.tau])*pyom.dzw[k-1]/pyom.dzt[k]

    if pyom.enable_conserve_energy and pyom.enable_tke:
        """
        dissipation by advection interpolated on W-grid
        """
        for j in xrange(pyom.js_pe,pyom.je_pe):
            for i in xrange(pyom.is_pe,pyom.ie_pe):
                ks = pyom.kbot[i,j] - 1
                if ks >= 0:
                    k = ks
                    pyom.P_diss_adv[i,j,k] = 0.5*(aloc[i,j,k]+aloc[i,j,k+1]) + 0.5*aloc[i,j,k]*pyom.dzw[max(0,k-1)]/pyom.dzw[k] # max(1,k-1)
                    for k in xrange(ks,pyom.nz-1): # k=ks+1,nz-1
                        pyom.P_diss_adv[i,j,k] = 0.5 * (aloc[i,j,k] + aloc[i,j,k+1])
                    k = pyom.nz-1
                    pyom.P_diss_adv[i,j,k] = aloc[i,j,k]

        """
        distribute pyom.P_diss_adv over domain, prevent draining of TKE
        """
        fxa = 0
        fxb = 0
        for k in xrange(pyom.nz-1): # k=1,nz-1
            for j in xrange(pyom.js_pe,pyom.je_pe):
                for i in xrange(pyom.is_pe,pyom.ie_pe):
                    fxa = fxa + pyom.area_t[i,j]*pyom.P_diss_adv[i,j,k]*pyom.dzw[k]*pyom.maskW[i,j,k]
                    if pyom.tke[i,j,k,pyom.tau] > .0:
                        fxb = fxb + pyom.area_t[i,j]*pyom.dzw[k]*pyom.maskW[i,j,k]
        k = pyom.nz-1
        for j in xrange(pyom.js_pe,pyom.je_pe):
            for i in xrange(pyom.is_pe,pyom.ie_pe):
                fxa = fxa + 0.5*pyom.area_t[i,j] * pyom.P_diss_adv[i,j,k] * pyom.dzw[k] * pyom.maskW[i,j,k]
                fxb = fxb + 0.5*pyom.area_t[i,j] * pyom.dzw[k] * pyom.maskW[i,j,k]
        #global_sum(fxa)
        #global_sum(fxb)
        pyom.P_diss_adv = 0.0
        for k in xrange(pyom.nz): # k=1,nz
            for j in xrange(pyom.js_pe,pyom.je_pe):
                for i in xrange(pyom.is_pe,pyom.ie_pe):
                    if pyom.tke[i,j,k,pyom.tau] > .0 or k == pyom.nz:
                        pyom.P_diss_adv[i,j,k] = fxa / fxb

    """
    Adam Bashforth time stepping for advection
    """
    pyom.temp[:,:,:,pyom.taup1] = pyom.temp[:,:,:,pyom.tau] + pyom.dt_tracer * ((1.5+pyom.AB_eps)*pyom.dtemp[:,:,:,pyom.tau] - (0.5+pyom.AB_eps)*pyom.dtemp[:,:,:,pyom.taum1]) * pyom.maskT
    pyom.salt[:,:,:,pyom.taup1] = pyom.salt[:,:,:,pyom.tau] + pyom.dt_tracer * ((1.5+pyom.AB_eps)*pyom.dsalt[:,:,:,pyom.tau] - (0.5+pyom.AB_eps)*pyom.dsalt[:,:,:,pyom.taum1]) * pyom.maskT

    """
    horizontal diffusion
    """
    with pyom.timers["isoneutral"]:
        if pyom.enable_hor_diffusion:
            diffusion.tempsalt_diffusion(pyom)
        else:
            diffusion.tempsalt_biharmonic(pyom)

        """
        sources like restoring zones, etc
        """
        if pyom.enable_tempsalt_sources:
            diffusion.tempsalt_sources(pyom)

        """
        isopycnal diffusion
        """
        if pyom.enable_neutral_diffusion:
            pyom.P_diss_iso[...] = 0.0
            pyom.dtemp_iso[...] = 0.0
            pyom.dsalt_iso[...] = 0.0
            isoneutral.isoneutral_diffusion_pre(pyom)
            isoneutral.isoneutral_diffusion(pyom.temp,True,pyom)
            isoneutral.isoneutral_diffusion(pyom.salt,False,pyom)
            if pyom.enable_skew_diffusion:
                pyom.P_diss_skew[...] = 0.0
                isoneutral.isoneutral_skew_diffusion(pyom.temp,True,pyom)
                isoneutral.isoneutral_skew_diffusion(pyom.salt,False,pyom)

    with pyom.timers["vmix"]:
        """
        vertical mixing of temperature and salinity
        """
        dtemp_vmix = pyom.temp[:,:,:,pyom.taup1]
        dsalt_vmix = pyom.salt[:,:,:,pyom.taup1]
        for j in xrange(pyom.js_pe,pyom.je_pe):
            for i in xrange(pyom.is_pe,pyom.ie_pe):
                ks = pyom.kbot[i,j] - 1
                if ks >= 0:
                    for k in xrange(ks,pyom.nz-1): # k=ks,nz-1
                        delta[k] = pyom.dt_tracer/pyom.dzw[k] * pyom.kappaH[i,j,k]
                    delta[pyom.nz-1] = 0.0
                    for k in xrange(ks+1,pyom.nz): # k=ks+1,nz
                        a_tri[k] = -delta[k-1]/pyom.dzt[k]
                    a_tri[ks] = 0.0
                    for k in xrange(ks+1,pyom.nz-1): # k=ks+1,nz-1
                        b_tri[k] = 1 + delta[k]/pyom.dzt[k] + delta[k-1]/pyom.dzt[k]
                    b_tri[pyom.nz-1] = 1 + delta[pyom.nz-2]/pyom.dzt[pyom.nz-1]
                    b_tri[ks] = 1 + delta[ks]/pyom.dzt[ks]
                    for k in xrange(ks,pyom.nz-1): # k=ks,nz-1
                        c_tri[k] = -delta[k]/pyom.dzt[k]
                    c_tri[pyom.nz-1] = 0.0
                    d_tri[ks:] = pyom.temp[i,j,ks:,pyom.taup1]
                    d_tri[pyom.nz-1] = d_tri[pyom.nz-1] + pyom.dt_tracer * pyom.forc_temp_surface[i,j] / pyom.dzt[pyom.nz-1]
                    pyom.temp[i,j,ks:,pyom.taup1] = numerics.solve_tridiag(a_tri[ks:],b_tri[ks:],c_tri[ks:],d_tri[ks:pyom.nz],pyom.nz-ks)
                    d_tri[ks:] = pyom.salt[i,j,ks:,pyom.taup1]
                    d_tri[pyom.nz-1] = d_tri[pyom.nz-1] + pyom.dt_tracer * pyom.forc_salt_surface[i,j] / pyom.dzt[pyom.nz-1]
                    pyom.salt[i,j,ks:,pyom.taup1] = numerics.solve_tridiag(a_tri[ks:],b_tri[ks:],c_tri[ks:],d_tri[ks:pyom.nz],pyom.nz-ks)
        dtemp_vmix = (pyom.temp[:,:,:,pyom.taup1] - dtemp_vmix) / pyom.dt_tracer
        dsalt_vmix = (pyom.salt[:,:,:,pyom.taup1] - dsalt_vmix) / pyom.dt_tracer

    """
    boundary exchange
    """
    cyclic.setcyclic_xyz(pyom.temp[:,:,:,pyom.taup1],pyom.enable_cyclic_x,pyom.nx,pyom.nz)
    cyclic.setcyclic_xyz(pyom.salt[:,:,:,pyom.taup1],pyom.enable_cyclic_x,pyom.nx,pyom.nz)

    with pyom.timers["eq_of_state"]:
        calc_eq_of_state(pyom,pyom.taup1)

    """
    surface density flux
    """
    for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx):
        for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx):
            pyom.forc_rho_surface[i,j] = (
                                        density.get_drhodT(pyom.salt[i,j,pyom.nz-1,pyom.taup1],pyom.temp[i,j,pyom.nz-1,pyom.taup1],abs(pyom.zt[pyom.nz-1]),pyom) * pyom.forc_temp_surface[i,j] \
                                      + density.get_drhodS(pyom.salt[i,j,pyom.nz-1,pyom.taup1],pyom.temp[i,j,pyom.nz-1,pyom.taup1],abs(pyom.zt[pyom.nz-1]),pyom) * pyom.forc_salt_surface[i,j] \
                                    ) * pyom.maskT[i,j,pyom.nz-1]

    with pyom.timers["vmix"]:
        pyom.P_diss_v[...] = 0.0
        if pyom.enable_conserve_energy:
            """
            diagnose dissipation of dynamic enthalpy by vertical mixing
            """
            for k in xrange(pyom.nz-1): # k=1,nz-1
                for j in xrange(pyom.js_pe,pyom.je_pe):
                    for i in xrange(pyom.is_pe,pyom.ie_pe):
                        fxa = (-pyom.int_drhodT[i,j,k+1,pyom.taup1] + pyom.int_drhodT[i,j,k,pyom.taup1]) / pyom.dzw[k]
                        pyom.P_diss_v[i,j,k] = pyom.P_diss_v[i,j,k] \
                                          - pyom.grav/pyom.rho_0 * fxa * pyom.kappaH[i,j,k] * (pyom.temp[i,j,k+1,pyom.taup1] - pyom.temp[i,j,k,pyom.taup1]) / pyom.dzw[k] * pyom.maskW[i,j,k]
                        fxa = (-pyom.int_drhodS[i,j,k+1,pyom.taup1] + pyom.int_drhodS[i,j,k,pyom.taup1]) / pyom.dzw[k]
                        pyom.P_diss_v[i,j,k] = pyom.P_diss_v[i,j,k] \
                                          - pyom.grav/pyom.rho_0 * fxa * pyom.kappaH[i,j,k] * (pyom.salt[i,j,k+1,pyom.taup1] - pyom.salt[i,j,k,pyom.taup1]) / pyom.dzw[k] * pyom.maskW[i,j,k]
            k = pyom.nz-1
            for j in xrange(pyom.js_pe,pyom.je_pe):
                for i in xrange(pyom.is_pe, pyom.ie_pe):
                    fxa = 2 * pyom.int_drhodT[i,j,k,pyom.taup1] / pyom.dzw[k]
                    pyom.P_diss_v[i,j,k] = pyom.P_diss_v[i,j,k] - pyom.grav/pyom.rho_0 * fxa * pyom.forc_temp_surface[i,j] * pyom.maskW[i,j,k]
                    fxa = 2 * pyom.int_drhodS[i,j,k,pyom.taup1] / pyom.dzw[k]
                    pyom.P_diss_v[i,j,k]= pyom.P_diss_v[i,j,k] - pyom.grav/pyom.rho_0 * fxa * pyom.forc_salt_surface[i,j] * pyom.maskW[i,j,k]

        if pyom.enable_conserve_energy:
            """
            determine effect due to nonlinear equation of state
            """
            aloc[:,:,:pyom.nz-1] = pyom.kappaH[:,:,:pyom.nz-1] * pyom.Nsqr[:,:,:pyom.nz-1,pyom.taup1]
            pyom.P_diss_nonlin[:,:,:pyom.nz-1] = pyom.P_diss_v[:,:,:pyom.nz-1] - aloc[:,:,:pyom.nz-1]
            pyom.P_diss_v[:,:,:pyom.nz-1] = aloc[:,:,:pyom.nz-1]
        else:
            """
            diagnose N^2 pyom.kappaH, i.e. exchange of pot. energy with TKE
            """
            pyom.P_diss_v[:,:,:pyom.nz-1] = pyom.kappaH[:,:,:pyom.nz-1] * pyom.Nsqr[:,:,:pyom.nz-1,pyom.taup1]
            pyom.P_diss_v[:,:,pyom.nz-1] = -pyom.forc_rho_surface[:,:] * pyom.maskT[:,:,pyom.nz-1] * pyom.grav/pyom.rho_0


def advect_tracer(tr,dtr,pyom):
    """
    calculate time tendency of a tracer due to advection
    """
    # integer, intent(in) :: is_,ie_,js_,je_,nz_
    # real*8, intent(inout) :: dtr(is_:ie_,js_:je_,nz_),tr(is_:ie_,js_:je_,nz_)
    # integer :: i,j,k

    if pyom.enable_superbee_advection:
        advection.adv_flux_superbee(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx,pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx,pyom.nz,pyom.flux_east,pyom.flux_north,pyom.flux_top,tr)
    else:
        advection.adv_flux_2nd(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx,pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx,pyom.nz,pyom.flux_east,pyom.flux_north,pyom.flux_top,tr)
    for j in xrange(pyom.js_pe,pyom.je_pe):
        for i in xrange(pyom.is_pe,pyom.ie_pe):
            dtr[i,j,:] = pyom.maskT[i,j,:] * (-(pyom.flux_east[i,j,:] -  pyom.flux_east[i-1,j,:]) / (pyom.cost[j]*pyom.dxt[i]) \
                                         -(pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:]) / (pyom.cost[j]*pyom.dyt[j]))
    k = 0 # k=1
    dtr[:,:,k] = dtr[:,:,k] - pyom.maskT[:,:,k] * pyom.flux_top[:,:,k] / pyom.dzt[k]
    for k in xrange(1,pyom.nz): # k=2,nz
        dtr[:,:,k] = dtr[:,:,k] - pyom.maskT[:,:,k] * (pyom.flux_top[:,:,k] - pyom.flux_top[:,:,k-1]) / pyom.dzt[k]


def advect_temperature(pyom):
    """
    integrate temperature
    """
    if pyom.enable_superbee_advection:
        advection.adv_flux_superbee(pyom.flux_east,pyom.flux_north,pyom.flux_top,pyom.temp[:,:,:,pyom.tau],pyom)
    else:
        advection.adv_flux_2nd(pyom.flux_east,pyom.flux_north,pyom.flux_top,pyom.temp[:,:,:,pyom.tau],pyom)
    for j in xrange(pyom.js_pe,pyom.je_pe):
        for i in xrange(pyom.is_pe,pyom.ie_pe):
            pyom.dtemp[i,j,:,pyom.tau] = pyom.maskT[i,j,:] * (-(pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:]) / (pyom.cost[j]*pyom.dxt[i]) \
                                               -(pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:]) / (pyom.cost[j]*pyom.dyt[j]))
    k = 0 # k=1
    pyom.dtemp[:,:,k,pyom.tau] = pyom.dtemp[:,:,k,pyom.tau] - pyom.maskT[:,:,k] * pyom.flux_top[:,:,k] / pyom.dzt[k]
    for k in xrange(1,pyom.nz):
        pyom.dtemp[:,:,k,pyom.tau] = pyom.dtemp[:,:,k,pyom.tau] - pyom.maskT[:,:,k] * (pyom.flux_top[:,:,k] - pyom.flux_top[:,:,k-1]) / pyom.dzt[k]


def advect_salinity(pyom):
    """
    integrate salinity
    """
    if pyom.enable_superbee_advection:
        advection.adv_flux_superbee(pyom.flux_east,pyom.flux_north,pyom.flux_top,pyom.salt[:,:,:,pyom.tau],pyom)
    else:
        advection.adv_flux_2nd(pyom.flux_east,pyom.flux_north,pyom.flux_top,pyom.salt[:,:,:,pyom.tau],pyom)
    for j in xrange(pyom.js_pe,pyom.je_pe):
        for i in xrange(pyom.is_pe,pyom.ie_pe):
            pyom.dsalt[i,j,:,pyom.tau] = pyom.maskT[i,j,:] * (-(pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:]) / (pyom.cost[j]*pyom.dxt[i]) \
                                               -(pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:]) / (pyom.cost[j]*pyom.dyt[j]))
    k = 0 # k=1
    pyom.dsalt[:,:,k,pyom.tau] = pyom.dsalt[:,:,k,pyom.tau] - pyom.maskT[:,:,k] * pyom.flux_top[:,:,k] / pyom.dzt[k]
    for k in xrange(1,pyom.nz):
        pyom.dsalt[:,:,k,pyom.tau] = pyom.dsalt[:,:,k,pyom.tau] - pyom.maskT[:,:,k] * (pyom.flux_top[:,:,k] - pyom.flux_top[:,:,k-1]) / pyom.dzt[k]


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
    for k in xrange(pyom.nz): # k=1,nz
        for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx):
            for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx):
                pyom.rho[i,j,k,n] = density.get_rho(pyom.salt[i,j,k,n],pyom.temp[i,j,k,n],abs(pyom.zt[k]),pyom) * pyom.maskT[i,j,k]

    if pyom.enable_conserve_energy:
        """
        calculate new dynamic enthalpy and derivatives
        """
        for k in xrange(pyom.nz): # k=1,nz
            for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx):
                for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx):
                    pyom.Hd[i,j,k,n] = density.get_dyn_enthalpy(pyom.salt[i,j,k,n],pyom.temp[i,j,k,n],abs(pyom.zt[k]),pyom) * pyom.maskT[i,j,k]
                    pyom.int_drhodT[i,j,k,n] = density.get_int_drhodT(pyom.salt[i,j,k,n],pyom.temp[i,j,k,n],abs(pyom.zt[k]),pyom)
                    pyom.int_drhodS[i,j,k,n] = density.get_int_drhodS(pyom.salt[i,j,k,n],pyom.temp[i,j,k,n],abs(pyom.zt[k]),pyom)

    """
    new stability frequency
    """
    for k in xrange(pyom.nz-1): # k=1,nz-1
        for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx):
            for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx):
                fxa = -pyom.grav / pyom.rho_0 / pyom.dzw[k] * pyom.maskW[i,j,k]
                pyom.Nsqr[i,j,k,n] = fxa * (density.get_rho(pyom.salt[i,j,k+1,n],pyom.temp[i,j,k+1,n],abs(pyom.zt[k]),pyom)-pyom.rho[i,j,k,n])
    pyom.Nsqr[:,:,pyom.nz-1,n] = pyom.Nsqr[:,:,pyom.nz-2,n]

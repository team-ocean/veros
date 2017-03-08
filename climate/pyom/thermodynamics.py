from climate.pyom import advection, diffusion, isoneutral, cyclic
from climate.pyom import numerics, density, utilities, pyom_method

@pyom_method
def thermodynamics(pyom):
    """
    integrate temperature and salinity and diagnose sources of dynamic enthalpy
    """
    advect_temperature(pyom)
    advect_salinity(pyom)

    if pyom.enable_conserve_energy:
        """
        advection of dynamic enthalpy
        """
        if pyom.enable_superbee_advection:
            advection.adv_flux_superbee(pyom,pyom.flux_east,pyom.flux_north,pyom.flux_top,pyom.Hd[:,:,:,pyom.tau])
        else:
            advection.adv_flux_2nd(pyom,pyom.flux_east,pyom.flux_north,pyom.flux_top,pyom.Hd[:,:,:,pyom.tau])

        pyom.dHd[2:-2, 2:-2, :, pyom.tau] = pyom.maskT[2:-2, 2:-2, :] * (-(pyom.flux_east[2:-2, 2:-2, :] - pyom.flux_east[1:-3, 2:-2, :]) \
                                                                            / (pyom.cost[None, 2:-2, None] * pyom.dxt[2:-2, None, None]) \
                                                                        - (pyom.flux_north[2:-2, 2:-2,:] - pyom.flux_north[2:-2, 1:-3, :]) \
                                                                            / (pyom.cost[None, 2:-2, None] * pyom.dyt[None, 2:-2, None]))
        pyom.dHd[:,:,0,pyom.tau] += -pyom.maskT[:,:,0] * pyom.flux_top[:,:,0] / pyom.dzt[0]
        pyom.dHd[:,:,1:,pyom.tau] += -pyom.maskT[:,:,1:] * (pyom.flux_top[:,:,1:] - pyom.flux_top[:,:,:-1]) / pyom.dzt[None, None, 1:]

        """
        changes in dyn. Enthalpy due to advection
        """
        aloc = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
        aloc[2:-2, 2:-2, :] = pyom.grav / pyom.rho_0 * (-pyom.int_drhodT[2:-2, 2:-2, :, pyom.tau] * pyom.dtemp[2:-2, 2:-2, :, pyom.tau] \
                                                       - pyom.int_drhodS[2:-2, 2:-2, :, pyom.tau] * pyom.dsalt[2:-2, 2:-2, :, pyom.tau]) \
                            - pyom.dHd[2:-2, 2:-2, :, pyom.tau]

        """
        contribution by vertical advection is - g rho w / rho0, substract this also
        """
        aloc[:, :, :-1] += -0.25 * pyom.grav / pyom.rho_0 * pyom.w[:, :, :-1, pyom.tau] \
                           * (pyom.rho[:, :, :-1, pyom.tau] + pyom.rho[:, :, 1:, pyom.tau]) \
                           * pyom.dzw[None, None, :-1] / pyom.dzt[None, None, :-1]
        aloc[:, :, 1:] += -0.25 * pyom.grav / pyom.rho_0 * pyom.w[:, :, :-1, pyom.tau] \
                          * (pyom.rho[:, :, 1:, pyom.tau] + pyom.rho[:, :, :-1, pyom.tau]) \
                          * pyom.dzw[None, None, :-1] / pyom.dzt[None, None, 1:]

    if pyom.enable_conserve_energy and pyom.enable_tke:
        """
        dissipation by advection interpolated on W-grid
        """
        pyom.P_diss_adv[...] = 0.
        diffusion.dissipation_on_wgrid(pyom, pyom.P_diss_adv, aloc=aloc)

        """
        distribute pyom.P_diss_adv over domain, prevent draining of TKE
        """
        fxa = np.sum(pyom.area_t[2:-2, 2:-2, None] * pyom.P_diss_adv[2:-2, 2:-2, :-1] \
                            * pyom.dzw[None, None, :-1] * pyom.maskW[2:-2, 2:-2, :-1]) \
            + np.sum(0.5 * pyom.area_t[2:-2, 2:-2] * pyom.P_diss_adv[2:-2, 2:-2, -1] \
                            * pyom.dzw[-1] * pyom.maskW[2:-2, 2:-2, -1])
        tke_mask = pyom.tke[2:-2, 2:-2, :-1, pyom.tau] > 0.
        fxb = np.sum(pyom.area_t[2:-2, 2:-2, None] * pyom.dzw[None, None, :-1] * pyom.maskW[2:-2, 2:-2, :-1] * tke_mask) \
            + np.sum(0.5 * pyom.area_t[2:-2, 2:-2] * pyom.dzw[-1] * pyom.maskW[2:-2, 2:-2, -1])
        pyom.P_diss_adv[...] = 0.
        pyom.P_diss_adv[2:-2, 2:-2, :-1] = fxa / fxb * tke_mask
        pyom.P_diss_adv[2:-2, 2:-2, -1] = fxa / fxb

    """
    Adam Bashforth time stepping for advection
    """
    pyom.temp[:,:,:,pyom.taup1] = pyom.temp[:,:,:,pyom.tau] + pyom.dt_tracer * \
                    ((1.5+pyom.AB_eps)*pyom.dtemp[:,:,:,pyom.tau] - (0.5+pyom.AB_eps)*pyom.dtemp[:,:,:,pyom.taum1]) * pyom.maskT
    pyom.salt[:,:,:,pyom.taup1] = pyom.salt[:,:,:,pyom.tau] + pyom.dt_tracer * \
                    ((1.5+pyom.AB_eps)*pyom.dsalt[:,:,:,pyom.tau] - (0.5+pyom.AB_eps)*pyom.dsalt[:,:,:,pyom.taum1]) * pyom.maskT

    """
    horizontal diffusion
    """
    with pyom.timers["isoneutral"]:
        if pyom.enable_hor_diffusion:
            diffusion.tempsalt_diffusion(pyom)
        if pyom.enable_biharmonic_mixing:
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
            isoneutral.isoneutral_diffusion(pyom,pyom.temp,True)
            isoneutral.isoneutral_diffusion(pyom,pyom.salt,False)
            if pyom.enable_skew_diffusion:
                pyom.P_diss_skew[...] = 0.0
                isoneutral.isoneutral_skew_diffusion(pyom,pyom.temp,True)
                isoneutral.isoneutral_skew_diffusion(pyom,pyom.salt,False)

    with pyom.timers["vmix"]:
        """
        vertical mixing of temperature and salinity
        """
        pyom.dtemp_vmix[...] = pyom.temp[:,:,:,pyom.taup1]
        pyom.dsalt_vmix[...] = pyom.salt[:,:,:,pyom.taup1]

        a_tri = np.zeros((pyom.nx, pyom.ny, pyom.nz))
        b_tri = np.zeros((pyom.nx, pyom.ny, pyom.nz))
        c_tri = np.zeros((pyom.nx, pyom.ny, pyom.nz))
        d_tri = np.zeros((pyom.nx, pyom.ny, pyom.nz))
        delta = np.zeros((pyom.nx, pyom.ny, pyom.nz))

        ks = pyom.kbot[2:-2, 2:-2] - 1
        delta[:, :, :-1] = pyom.dt_tracer / pyom.dzw[None, None, :-1] * pyom.kappaH[2:-2, 2:-2, :-1]
        delta[:, :, -1] = 0.
        a_tri[:, :, 1:] = -delta[:,:,:-1] / pyom.dzt[None, None, 1:]
        b_tri[:, :, 1:] = 1 + (delta[:, :, 1:] + delta[:, :, :-1]) / pyom.dzt[None, None, 1:]
        b_tri_edge = 1 + delta / pyom.dzt[None, None, :]
        c_tri[:, :, :-1] = -delta[:, :, :-1] / pyom.dzt[None, None, :-1]
        d_tri[...] = pyom.temp[2:-2, 2:-2, :, pyom.taup1]
        d_tri[:, :, -1] += pyom.dt_tracer * pyom.forc_temp_surface[2:-2, 2:-2] / pyom.dzt[-1]
        sol, mask = utilities.solve_implicit(pyom, ks, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)
        pyom.temp[2:-2, 2:-2, :, pyom.taup1] = np.where(mask, sol, pyom.temp[2:-2, 2:-2, :, pyom.taup1])
        d_tri[...] = pyom.salt[2:-2, 2:-2, :, pyom.taup1]
        d_tri[:, :, -1] += pyom.dt_tracer * pyom.forc_salt_surface[2:-2, 2:-2] / pyom.dzt[-1]
        sol, mask = utilities.solve_implicit(pyom, ks, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)
        pyom.salt[2:-2, 2:-2, :, pyom.taup1] = np.where(mask, sol, pyom.salt[2:-2, 2:-2, :, pyom.taup1])

        pyom.dtemp_vmix[...] = (pyom.temp[:,:,:,pyom.taup1] - pyom.dtemp_vmix) / pyom.dt_tracer
        pyom.dsalt_vmix[...] = (pyom.salt[:,:,:,pyom.taup1] - pyom.dsalt_vmix) / pyom.dt_tracer

    """
    boundary exchange
    """
    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(pyom.temp[..., pyom.taup1])
        cyclic.setcyclic_x(pyom.salt[..., pyom.taup1])

    with pyom.timers["eq_of_state"]:
        calc_eq_of_state(pyom, pyom.taup1)

    """
    surface density flux
    """
    pyom.forc_rho_surface[...] = (
                                density.get_drhodT(pyom,pyom.salt[:,:,-1,pyom.taup1],pyom.temp[:,:,-1,pyom.taup1],np.abs(pyom.zt[-1])) * pyom.forc_temp_surface \
                              + density.get_drhodS(pyom,pyom.salt[:,:,-1,pyom.taup1],pyom.temp[:,:,-1,pyom.taup1],np.abs(pyom.zt[-1])) * pyom.forc_salt_surface \
                            ) * pyom.maskT[:,:,-1]

    with pyom.timers["vmix"]:
        pyom.P_diss_v[...] = 0.0
        if pyom.enable_conserve_energy:
            """
            diagnose dissipation of dynamic enthalpy by vertical mixing
            """
            fxa = (-pyom.int_drhodT[2:-2, 2:-2, 1:, pyom.taup1] + pyom.int_drhodT[2:-2, 2:-2, :-1,pyom.taup1]) / pyom.dzw[None, None, :-1]
            pyom.P_diss_v[2:-2, 2:-2, :-1] += -pyom.grav / pyom.rho_0 * fxa * pyom.kappaH[2:-2, 2:-2, :-1] \
                                              * (pyom.temp[2:-2, 2:-2, 1:, pyom.taup1] - pyom.temp[2:-2, 2:-2, :-1,pyom.taup1]) \
                                              / pyom.dzw[None, None, :-1] * pyom.maskW[2:-2, 2:-2, :-1]
            fxa = (-pyom.int_drhodS[2:-2, 2:-2, 1:, pyom.taup1] + pyom.int_drhodS[2:-2, 2:-2, :-1,pyom.taup1]) / pyom.dzw[None, None, :-1]
            pyom.P_diss_v[2:-2, 2:-2, :-1] += -pyom.grav / pyom.rho_0 * fxa * pyom.kappaH[2:-2, 2:-2, :-1] \
                                              * (pyom.salt[2:-2, 2:-2, 1:, pyom.taup1] - pyom.salt[2:-2, 2:-2, :-1,pyom.taup1]) \
                                              / pyom.dzw[None, None, :-1] * pyom.maskW[2:-2, 2:-2, :-1]

            fxa = 2 * pyom.int_drhodT[2:-2, 2:-2, -1, pyom.taup1] / pyom.dzw[-1]
            pyom.P_diss_v[2:-2, 2:-2, -1] += - pyom.grav / pyom.rho_0 * fxa * pyom.forc_temp_surface[2:-2 ,2:-2] * pyom.maskW[2:-2, 2:-2, -1]
            fxa = 2 * pyom.int_drhodS[2:-2, 2:-2, -1, pyom.taup1] / pyom.dzw[-1]
            pyom.P_diss_v[2:-2, 2:-2, -1] += - pyom.grav / pyom.rho_0 * fxa * pyom.forc_salt_surface[2:-2 ,2:-2] * pyom.maskW[2:-2, 2:-2, -1]

        if pyom.enable_conserve_energy:
            """
            determine effect due to nonlinear equation of state
            """
            aloc[:,:,:-1] = pyom.kappaH[:,:,:-1] * pyom.Nsqr[:,:,:-1,pyom.taup1]
            pyom.P_diss_nonlin[:,:,:-1] = pyom.P_diss_v[:,:,:-1] - aloc[:,:,:-1]
            pyom.P_diss_v[:,:,:-1] = aloc[:,:,:-1]
        else:
            """
            diagnose N^2 pyom.kappaH, i.e. exchange of pot. energy with TKE
            """
            pyom.P_diss_v[:,:,:-1] = pyom.kappaH[:,:,:-1] * pyom.Nsqr[:,:,:-1,pyom.taup1]
            pyom.P_diss_v[:,:,-1] = -pyom.forc_rho_surface * pyom.maskT[:,:,-1] * pyom.grav / pyom.rho_0

@pyom_method
def advect_tracer(pyom, tr, dtr):
    """
    calculate time tendency of a tracer due to advection
    """
    if pyom.enable_superbee_advection:
        advection.adv_flux_superbee(pyom,pyom.flux_east,pyom.flux_north,pyom.flux_top,tr)
    else:
        advection.adv_flux_2nd(pyom,pyom.flux_east,pyom.flux_north,pyom.flux_top,tr)
    dtr[2:-2, 2:-2, :] = pyom.maskT[2:-2, 2:-2, :] * (-(pyom.flux_east[2:-2, 2:-2, :] - pyom.flux_east[1:-3, 2:-2, :]) \
                                                        / (pyom.cost[None, 2:-2, None] * pyom.dxt[2:-2, None, None]) \
                                                     - (pyom.flux_north[2:-2, 2:-2, :] - pyom.flux_north[2:-2, 1:-3, :]) \
                                                        / (pyom.cost[None, 2:-2, None] * pyom.dyt[None, 2:-2, None]))
    dtr[:, :, 0] += -pyom.maskT[:, :, 0] * pyom.flux_top[:, :, 0] / pyom.dzt[0]
    dtr[:, :, 1:] += -pyom.maskT[:, :, 1:] * (pyom.flux_top[:, :, 1:] - pyom.flux_top[:, :, :-1]) / pyom.dzt[1:]

@pyom_method
def advect_temperature(pyom):
    """
    integrate temperature
    """
    return advect_tracer(pyom, pyom.temp[..., pyom.tau], pyom.dtemp[..., pyom.tau])

@pyom_method
def advect_salinity(pyom):
    """
    integrate salinity
    """
    return advect_tracer(pyom, pyom.salt[..., pyom.tau], pyom.dsalt[..., pyom.tau])

@pyom_method
def calc_eq_of_state(pyom, n):
    """
    calculate density, stability frequency, dynamic enthalpy and derivatives
    for time level n from temperature and salinity
    """
    density_args = (pyom, pyom.salt[..., n], pyom.temp[..., n], np.abs(pyom.zt))

    """
    calculate new density
    """
    pyom.rho[..., n] = density.get_rho(*density_args) * pyom.maskT

    if pyom.enable_conserve_energy:
        """
        calculate new dynamic enthalpy and derivatives
        """
        pyom.Hd[..., n] = density.get_dyn_enthalpy(*density_args) * pyom.maskT
        pyom.int_drhodT[..., n] = density.get_int_drhodT(*density_args)
        pyom.int_drhodS[..., n] = density.get_int_drhodS(*density_args)

    """
    new stability frequency
    """
    fxa = -pyom.grav / pyom.rho_0 / pyom.dzw[None, None, :-1] * pyom.maskW[:, :, :-1]
    pyom.Nsqr[:, :, :-1, n] = fxa * (density.get_rho(pyom, pyom.salt[:,:,1:,n], pyom.temp[:,:,1:,n], np.abs(pyom.zt[:-1])) - pyom.rho[:,:,:-1,n])
    pyom.Nsqr[:, :, -1, n] = pyom.Nsqr[:,:,-2,n]

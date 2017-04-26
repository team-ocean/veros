from .. import veros_method
from . import advection, diffusion, isoneutral, cyclic, numerics, density, utilities


@veros_method
def thermodynamics(veros):
    """
    integrate temperature and salinity and diagnose sources of dynamic enthalpy
    """
    advect_temperature(veros)
    advect_salinity(veros)

    if veros.enable_conserve_energy:
        """
        advection of dynamic enthalpy
        """
        if veros.enable_superbee_advection:
            advection.adv_flux_superbee(veros, veros.flux_east, veros.flux_north,
                                        veros.flux_top, veros.Hd[:, :, :, veros.tau])
        else:
            advection.adv_flux_2nd(veros, veros.flux_east, veros.flux_north,
                                   veros.flux_top, veros.Hd[:, :, :, veros.tau])

        veros.dHd[2:-2, 2:-2, :, veros.tau] = veros.maskT[2:-2, 2:-2, :] * (-(veros.flux_east[2:-2, 2:-2, :] - veros.flux_east[1:-3, 2:-2, :])
                                                                            / (veros.cost[np.newaxis, 2:-2, np.newaxis] * veros.dxt[2:-2, np.newaxis, np.newaxis])
                                                                            - (veros.flux_north[2:-2, 2:-2, :] - veros.flux_north[2:-2, 1:-3, :])
                                                                            / (veros.cost[np.newaxis, 2:-2, np.newaxis] * veros.dyt[np.newaxis, 2:-2, np.newaxis]))
        veros.dHd[:, :, 0, veros.tau] += -veros.maskT[:, :, 0] \
            * veros.flux_top[:, :, 0] / veros.dzt[0]
        veros.dHd[:, :, 1:, veros.tau] += -veros.maskT[:, :, 1:] \
            * (veros.flux_top[:, :, 1:] - veros.flux_top[:, :, :-1]) \
            / veros.dzt[np.newaxis, np.newaxis, 1:]

        """
        changes in dyn. Enthalpy due to advection
        """
        aloc = np.zeros((veros.nx + 4, veros.ny + 4, veros.nz))
        aloc[2:-2, 2:-2, :] = veros.grav / veros.rho_0 * (-veros.int_drhodT[2:-2, 2:-2, :, veros.tau] * veros.dtemp[2:-2, 2:-2, :, veros.tau]
                                                          - veros.int_drhodS[2:-2, 2:-2, :, veros.tau] * veros.dsalt[2:-2, 2:-2, :, veros.tau]) \
            - veros.dHd[2:-2, 2:-2, :, veros.tau]

        """
        contribution by vertical advection is - g rho w / rho0, substract this also
        """
        aloc[:, :, :-1] += -0.25 * veros.grav / veros.rho_0 * veros.w[:, :, :-1, veros.tau] \
            * (veros.rho[:, :, :-1, veros.tau] + veros.rho[:, :, 1:, veros.tau]) \
            * veros.dzw[np.newaxis, np.newaxis, :-1] / veros.dzt[np.newaxis, np.newaxis, :-1]
        aloc[:, :, 1:] += -0.25 * veros.grav / veros.rho_0 * veros.w[:, :, :-1, veros.tau] \
            * (veros.rho[:, :, 1:, veros.tau] + veros.rho[:, :, :-1, veros.tau]) \
            * veros.dzw[np.newaxis, np.newaxis, :-1] / veros.dzt[np.newaxis, np.newaxis, 1:]

    if veros.enable_conserve_energy and veros.enable_tke:
        """
        dissipation by advection interpolated on W-grid
        """
        veros.P_diss_adv[...] = 0.
        diffusion.dissipation_on_wgrid(veros, veros.P_diss_adv, aloc=aloc)

        """
        distribute veros.P_diss_adv over domain, prevent draining of TKE
        """
        fxa = np.sum(veros.area_t[2:-2, 2:-2, np.newaxis] * veros.P_diss_adv[2:-2, 2:-2, :-1]
                     * veros.dzw[np.newaxis, np.newaxis, :-1] * veros.maskW[2:-2, 2:-2, :-1]) \
            + np.sum(0.5 * veros.area_t[2:-2, 2:-2] * veros.P_diss_adv[2:-2, 2:-2, -1]
                     * veros.dzw[-1] * veros.maskW[2:-2, 2:-2, -1])
        tke_mask = veros.tke[2:-2, 2:-2, :-1, veros.tau] > 0.
        fxb = np.sum(veros.area_t[2:-2, 2:-2, np.newaxis] * veros.dzw[np.newaxis, np.newaxis, :-1] * veros.maskW[2:-2, 2:-2, :-1] * tke_mask) \
            + np.sum(0.5 * veros.area_t[2:-2, 2:-2] * veros.dzw[-1] * veros.maskW[2:-2, 2:-2, -1])
        veros.P_diss_adv[...] = 0.
        veros.P_diss_adv[2:-2, 2:-2, :-1] = fxa / fxb * tke_mask
        veros.P_diss_adv[2:-2, 2:-2, -1] = fxa / fxb

    """
    Adam Bashforth time stepping for advection
    """
    veros.temp[:, :, :, veros.taup1] = veros.temp[:, :, :, veros.tau] + veros.dt_tracer \
        * ((1.5 + veros.AB_eps) * veros.dtemp[:, :, :, veros.tau]
           - (0.5 + veros.AB_eps) * veros.dtemp[:, :, :, veros.taum1]) * veros.maskT
    veros.salt[:, :, :, veros.taup1] = veros.salt[:, :, :, veros.tau] + veros.dt_tracer \
        * ((1.5 + veros.AB_eps) * veros.dsalt[:, :, :, veros.tau]
           - (0.5 + veros.AB_eps) * veros.dsalt[:, :, :, veros.taum1]) * veros.maskT

    """
    horizontal diffusion
    """
    with veros.timers["isoneutral"]:
        if veros.enable_hor_diffusion:
            diffusion.tempsalt_diffusion(veros)
        if veros.enable_biharmonic_mixing:
            diffusion.tempsalt_biharmonic(veros)

        """
        sources like restoring zones, etc
        """
        if veros.enable_tempsalt_sources:
            diffusion.tempsalt_sources(veros)

        """
        isopycnal diffusion
        """
        if veros.enable_neutral_diffusion:
            veros.P_diss_iso[...] = 0.0
            veros.dtemp_iso[...] = 0.0
            veros.dsalt_iso[...] = 0.0
            isoneutral.isoneutral_diffusion_pre(veros)
            isoneutral.isoneutral_diffusion(veros, veros.temp, True)
            isoneutral.isoneutral_diffusion(veros, veros.salt, False)
            if veros.enable_skew_diffusion:
                veros.P_diss_skew[...] = 0.0
                isoneutral.isoneutral_skew_diffusion(veros, veros.temp, True)
                isoneutral.isoneutral_skew_diffusion(veros, veros.salt, False)

    with veros.timers["vmix"]:
        """
        vertical mixing of temperature and salinity
        """
        veros.dtemp_vmix[...] = veros.temp[:, :, :, veros.taup1]
        veros.dsalt_vmix[...] = veros.salt[:, :, :, veros.taup1]

        a_tri = np.zeros((veros.nx, veros.ny, veros.nz))
        b_tri = np.zeros((veros.nx, veros.ny, veros.nz))
        c_tri = np.zeros((veros.nx, veros.ny, veros.nz))
        d_tri = np.zeros((veros.nx, veros.ny, veros.nz))
        delta = np.zeros((veros.nx, veros.ny, veros.nz))

        ks = veros.kbot[2:-2, 2:-2] - 1
        delta[:, :, :-1] = veros.dt_tracer / veros.dzw[np.newaxis, np.newaxis, :-1] \
            * veros.kappaH[2:-2, 2:-2, :-1]
        delta[:, :, -1] = 0.
        a_tri[:, :, 1:] = -delta[:, :, :-1] / veros.dzt[np.newaxis, np.newaxis, 1:]
        b_tri[:, :, 1:] = 1 + (delta[:, :, 1:] + delta[:, :, :-1]) \
            / veros.dzt[np.newaxis, np.newaxis, 1:]
        b_tri_edge = 1 + delta / veros.dzt[np.newaxis, np.newaxis, :]
        c_tri[:, :, :-1] = -delta[:, :, :-1] / veros.dzt[np.newaxis, np.newaxis, :-1]
        d_tri[...] = veros.temp[2:-2, 2:-2, :, veros.taup1]
        d_tri[:, :, -1] += veros.dt_tracer * veros.forc_temp_surface[2:-2, 2:-2] / veros.dzt[-1]
        sol, mask = utilities.solve_implicit(veros, ks, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)
        veros.temp[2:-2, 2:-2, :, veros.taup1] = np.where(mask, sol, veros.temp[2:-2, 2:-2, :, veros.taup1])
        d_tri[...] = veros.salt[2:-2, 2:-2, :, veros.taup1]
        d_tri[:, :, -1] += veros.dt_tracer * veros.forc_salt_surface[2:-2, 2:-2] / veros.dzt[-1]
        sol, mask = utilities.solve_implicit(
            veros, ks, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)
        veros.salt[2:-2, 2:-2, :,
                   veros.taup1] = np.where(mask, sol, veros.salt[2:-2, 2:-2, :, veros.taup1])

        veros.dtemp_vmix[...] = (veros.temp[:, :, :, veros.taup1] -
                                 veros.dtemp_vmix) / veros.dt_tracer
        veros.dsalt_vmix[...] = (veros.salt[:, :, :, veros.taup1] -
                                 veros.dsalt_vmix) / veros.dt_tracer

    """
    boundary exchange
    """
    if veros.enable_cyclic_x:
        cyclic.setcyclic_x(veros.temp[..., veros.taup1])
        cyclic.setcyclic_x(veros.salt[..., veros.taup1])

    with veros.timers["eq_of_state"]:
        calc_eq_of_state(veros, veros.taup1)

    """
    surface density flux
    """
    veros.forc_rho_surface[...] = (
        density.get_drhodT(veros, veros.salt[:, :, -1, veros.taup1],
                           veros.temp[:, :, -1, veros.taup1],
                           np.abs(veros.zt[-1])) * veros.forc_temp_surface
        + density.get_drhodS(veros, veros.salt[:, :, -1, veros.taup1],
                             veros.temp[:, :, -1, veros.taup1],
                             np.abs(veros.zt[-1])) * veros.forc_salt_surface
    ) * veros.maskT[:, :, -1]

    with veros.timers["vmix"]:
        veros.P_diss_v[...] = 0.0
        if veros.enable_conserve_energy:
            """
            diagnose dissipation of dynamic enthalpy by vertical mixing
            """
            fxa = (-veros.int_drhodT[2:-2, 2:-2, 1:, veros.taup1] + veros.int_drhodT[2:-2, 2:-2, :-1, veros.taup1]) \
                / veros.dzw[np.newaxis, np.newaxis, :-1]
            veros.P_diss_v[2:-2, 2:-2, :-1] += -veros.grav / veros.rho_0 * fxa * veros.kappaH[2:-2, 2:-2, :-1] \
                * (veros.temp[2:-2, 2:-2, 1:, veros.taup1] - veros.temp[2:-2, 2:-2, :-1, veros.taup1]) \
                / veros.dzw[np.newaxis, np.newaxis, :-1] * veros.maskW[2:-2, 2:-2, :-1]
            fxa = (-veros.int_drhodS[2:-2, 2:-2, 1:, veros.taup1] + veros.int_drhodS[2:-2, 2:-2, :-1, veros.taup1]) \
                / veros.dzw[np.newaxis, np.newaxis, :-1]
            veros.P_diss_v[2:-2, 2:-2, :-1] += -veros.grav / veros.rho_0 * fxa * veros.kappaH[2:-2, 2:-2, :-1] \
                * (veros.salt[2:-2, 2:-2, 1:, veros.taup1] - veros.salt[2:-2, 2:-2, :-1, veros.taup1]) \
                / veros.dzw[np.newaxis, np.newaxis, :-1] * veros.maskW[2:-2, 2:-2, :-1]

            fxa = 2 * veros.int_drhodT[2:-2, 2:-2, -1, veros.taup1] / veros.dzw[-1]
            veros.P_diss_v[2:-2, 2:-2, -1] += - veros.grav / veros.rho_0 * fxa * \
                veros.forc_temp_surface[2:-2, 2:-2] * veros.maskW[2:-2, 2:-2, -1]
            fxa = 2 * veros.int_drhodS[2:-2, 2:-2, -1, veros.taup1] / veros.dzw[-1]
            veros.P_diss_v[2:-2, 2:-2, -1] += - veros.grav / veros.rho_0 * fxa * \
                veros.forc_salt_surface[2:-2, 2:-2] * veros.maskW[2:-2, 2:-2, -1]

        if veros.enable_conserve_energy:
            """
            determine effect due to nonlinear equation of state
            """
            aloc[:, :, :-1] = veros.kappaH[:, :, :-1] * veros.Nsqr[:, :, :-1, veros.taup1]
            veros.P_diss_nonlin[:, :, :-1] = veros.P_diss_v[:, :, :-1] - aloc[:, :, :-1]
            veros.P_diss_v[:, :, :-1] = aloc[:, :, :-1]
        else:
            """
            diagnose N^2 veros.kappaH, i.e. exchange of pot. energy with TKE
            """
            veros.P_diss_v[:, :, :-1] = veros.kappaH[:, :, :-1] * veros.Nsqr[:, :, :-1, veros.taup1]
            veros.P_diss_v[:, :, -1] = -veros.forc_rho_surface * \
                veros.maskT[:, :, -1] * veros.grav / veros.rho_0


@veros_method
def advect_tracer(veros, tr, dtr):
    """
    calculate time tendency of a tracer due to advection
    """
    if veros.enable_superbee_advection:
        advection.adv_flux_superbee(veros, veros.flux_east, veros.flux_north, veros.flux_top, tr)
    else:
        advection.adv_flux_2nd(veros, veros.flux_east, veros.flux_north, veros.flux_top, tr)
    dtr[2:-2, 2:-2, :] = veros.maskT[2:-2, 2:-2, :] * (-(veros.flux_east[2:-2, 2:-2, :] - veros.flux_east[1:-3, 2:-2, :])
                                                       / (veros.cost[np.newaxis, 2:-2, np.newaxis] * veros.dxt[2:-2, np.newaxis, np.newaxis])
                                                       - (veros.flux_north[2:-2, 2:-2, :] - veros.flux_north[2:-2, 1:-3, :])
                                                       / (veros.cost[np.newaxis, 2:-2, np.newaxis] * veros.dyt[np.newaxis, 2:-2, np.newaxis]))
    dtr[:, :, 0] += -veros.maskT[:, :, 0] * veros.flux_top[:, :, 0] / veros.dzt[0]
    dtr[:, :, 1:] += -veros.maskT[:, :, 1:] * \
        (veros.flux_top[:, :, 1:] - veros.flux_top[:, :, :-1]) / veros.dzt[1:]


@veros_method
def advect_temperature(veros):
    """
    integrate temperature
    """
    return advect_tracer(veros, veros.temp[..., veros.tau], veros.dtemp[..., veros.tau])


@veros_method
def advect_salinity(veros):
    """
    integrate salinity
    """
    return advect_tracer(veros, veros.salt[..., veros.tau], veros.dsalt[..., veros.tau])


@veros_method
def calc_eq_of_state(veros, n):
    """
    calculate density, stability frequency, dynamic enthalpy and derivatives
    for time level n from temperature and salinity
    """
    density_args = (veros, veros.salt[..., n], veros.temp[..., n], np.abs(veros.zt))

    """
    calculate new density
    """
    veros.rho[..., n] = density.get_rho(*density_args) * veros.maskT

    if veros.enable_conserve_energy:
        """
        calculate new dynamic enthalpy and derivatives
        """
        veros.Hd[..., n] = density.get_dyn_enthalpy(*density_args) * veros.maskT
        veros.int_drhodT[..., n] = density.get_int_drhodT(*density_args)
        veros.int_drhodS[..., n] = density.get_int_drhodS(*density_args)

    """
    new stability frequency
    """
    fxa = -veros.grav / veros.rho_0 / \
        veros.dzw[np.newaxis, np.newaxis, :-1] * veros.maskW[:, :, :-1]
    veros.Nsqr[:, :, :-1, n] = fxa * (density.get_rho(veros, veros.salt[:, :, 1:, n],
                                                      veros.temp[:, :, 1:, n], np.abs(veros.zt[:-1])) - veros.rho[:, :, :-1, n])
    veros.Nsqr[:, :, -1, n] = veros.Nsqr[:, :, -2, n]

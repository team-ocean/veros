from .. import veros_method
from ..distributed import global_sum
from ..variables import allocate
from . import advection, diffusion, isoneutral, density, utilities


@veros_method
def thermodynamics(vs):
    """
    integrate temperature and salinity and diagnose sources of dynamic enthalpy
    """
    advect_temperature(vs)
    advect_salinity(vs)

    if vs.enable_conserve_energy:
        """
        advection of dynamic enthalpy
        """
        if vs.enable_superbee_advection:
            advection.adv_flux_superbee(vs, vs.flux_east, vs.flux_north,
                                        vs.flux_top, vs.Hd[:, :, :, vs.tau])
        else:
            advection.adv_flux_2nd(vs, vs.flux_east, vs.flux_north,
                                vs.flux_top, vs.Hd[:, :, :, vs.tau])

        vs.dHd[2:-2, 2:-2, :, vs.tau] = vs.maskT[2:-2, 2:-2, :] * (-(vs.flux_east[2:-2, 2:-2, :] - vs.flux_east[1:-3, 2:-2, :])
                                                                    / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxt[2:-2, np.newaxis, np.newaxis])
                                                                - (vs.flux_north[2:-2, 2:-2, :] - vs.flux_north[2:-2, 1:-3, :])
                                                                    / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dyt[np.newaxis, 2:-2, np.newaxis]))
        vs.dHd[:, :, 0, vs.tau] += -vs.maskT[:, :, 0] \
            * vs.flux_top[:, :, 0] / vs.dzt[0]
        vs.dHd[:, :, 1:, vs.tau] += -vs.maskT[:, :, 1:] \
            * (vs.flux_top[:, :, 1:] - vs.flux_top[:, :, :-1]) \
            / vs.dzt[np.newaxis, np.newaxis, 1:]

        """
        changes in dyn. Enthalpy due to advection
        """
        aloc = allocate(vs, ('xt', 'yt', 'zt'))
        aloc[2:-2, 2:-2, :] = vs.grav / vs.rho_0 * (-vs.int_drhodT[2:-2, 2:-2, :, vs.tau] * vs.dtemp[2:-2, 2:-2, :, vs.tau]
                                                - vs.int_drhodS[2:-2, 2:-2, :, vs.tau] * vs.dsalt[2:-2, 2:-2, :, vs.tau]) \
                            - vs.dHd[2:-2, 2:-2, :, vs.tau]

        """
        contribution by vertical advection is - g rho w / rho0, substract this also
        """
        aloc[:, :, :-1] += -0.25 * vs.grav / vs.rho_0 * vs.w[:, :, :-1, vs.tau] \
            * (vs.rho[:, :, :-1, vs.tau] + vs.rho[:, :, 1:, vs.tau]) \
            * vs.dzw[np.newaxis, np.newaxis, :-1] / vs.dzt[np.newaxis, np.newaxis, :-1]
        aloc[:, :, 1:] += -0.25 * vs.grav / vs.rho_0 * vs.w[:, :, :-1, vs.tau] \
            * (vs.rho[:, :, 1:, vs.tau] + vs.rho[:, :, :-1, vs.tau]) \
            * vs.dzw[np.newaxis, np.newaxis, :-1] / vs.dzt[np.newaxis, np.newaxis, 1:]

    if vs.enable_conserve_energy and vs.enable_tke:
        """
        dissipation by advection interpolated on W-grid
        """
        vs.P_diss_adv[...] = 0.
        diffusion.dissipation_on_wgrid(vs, vs.P_diss_adv, aloc=aloc)

        """
        distribute vs.P_diss_adv over domain, prevent draining of TKE
        """
        fxa = np.sum(vs.area_t[2:-2, 2:-2, np.newaxis] * vs.P_diss_adv[2:-2, 2:-2, :-1]
                    * vs.dzw[np.newaxis, np.newaxis, :-1] * vs.maskW[2:-2, 2:-2, :-1]) \
            + np.sum(0.5 * vs.area_t[2:-2, 2:-2] * vs.P_diss_adv[2:-2, 2:-2, -1]
                    * vs.dzw[-1] * vs.maskW[2:-2, 2:-2, -1])
        tke_mask = vs.tke[2:-2, 2:-2, :-1, vs.tau] > 0.
        fxb = np.sum(vs.area_t[2:-2, 2:-2, np.newaxis] * vs.dzw[np.newaxis, np.newaxis, :-1] * vs.maskW[2:-2, 2:-2, :-1] * tke_mask) \
            + np.sum(0.5 * vs.area_t[2:-2, 2:-2] * vs.dzw[-1] * vs.maskW[2:-2, 2:-2, -1])

        fxa = global_sum(vs, fxa)
        fxb = global_sum(vs, fxb)

        vs.P_diss_adv[...] = 0.
        vs.P_diss_adv[2:-2, 2:-2, :-1] = fxa / fxb * tke_mask
        vs.P_diss_adv[2:-2, 2:-2, -1] = fxa / fxb

    """
    Adam Bashforth time stepping for advection
    """
    vs.temp[:, :, :, vs.taup1] = vs.temp[:, :, :, vs.tau] + vs.dt_tracer \
        * ((1.5 + vs.AB_eps) * vs.dtemp[:, :, :, vs.tau]
        - (0.5 + vs.AB_eps) * vs.dtemp[:, :, :, vs.taum1]) * vs.maskT
    vs.salt[:, :, :, vs.taup1] = vs.salt[:, :, :, vs.tau] + vs.dt_tracer \
        * ((1.5 + vs.AB_eps) * vs.dsalt[:, :, :, vs.tau]
        - (0.5 + vs.AB_eps) * vs.dsalt[:, :, :, vs.taum1]) * vs.maskT

    """
    horizontal diffusion
    """
    with vs.timers['isoneutral']:
        if vs.enable_hor_diffusion:
            diffusion.tempsalt_diffusion(vs)
        if vs.enable_biharmonic_mixing:
            diffusion.tempsalt_biharmonic(vs)

        """
        sources like restoring zones, etc
        """
        if vs.enable_tempsalt_sources:
            diffusion.tempsalt_sources(vs)

        """
        isopycnal diffusion
        """
        if vs.enable_neutral_diffusion:
            vs.P_diss_iso[...] = 0.0
            vs.dtemp_iso[...] = 0.0
            vs.dsalt_iso[...] = 0.0
            isoneutral.isoneutral_diffusion_pre(vs)
            isoneutral.isoneutral_diffusion(vs, vs.temp, True)
            isoneutral.isoneutral_diffusion(vs, vs.salt, False)
            if vs.enable_skew_diffusion:
                vs.P_diss_skew[...] = 0.0
                isoneutral.isoneutral_skew_diffusion(vs, vs.temp, True)
                isoneutral.isoneutral_skew_diffusion(vs, vs.salt, False)

    with vs.timers['vmix']:
        """
        vertical mixing of temperature and salinity
        """
        vs.dtemp_vmix[...] = vs.temp[:, :, :, vs.taup1]
        vs.dsalt_vmix[...] = vs.salt[:, :, :, vs.taup1]

        a_tri = allocate(vs, ('xt', 'yt', 'zt'), include_ghosts=False)
        b_tri = allocate(vs, ('xt', 'yt', 'zt'), include_ghosts=False)
        c_tri = allocate(vs, ('xt', 'yt', 'zt'), include_ghosts=False)
        d_tri = allocate(vs, ('xt', 'yt', 'zt'), include_ghosts=False)
        delta = allocate(vs, ('xt', 'yt', 'zw'), include_ghosts=False)

        ks = vs.kbot[2:-2, 2:-2] - 1
        delta[:, :, :-1] = vs.dt_tracer / vs.dzw[np.newaxis, np.newaxis, :-1] \
            * vs.kappaH[2:-2, 2:-2, :-1]
        delta[:, :, -1] = 0.
        a_tri[:, :, 1:] = -delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, 1:]
        b_tri[:, :, 1:] = 1 + (delta[:, :, 1:] + delta[:, :, :-1]) \
            / vs.dzt[np.newaxis, np.newaxis, 1:]
        b_tri_edge = 1 + delta / vs.dzt[np.newaxis, np.newaxis, :]
        c_tri[:, :, :-1] = -delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, :-1]
        d_tri[...] = vs.temp[2:-2, 2:-2, :, vs.taup1]
        d_tri[:, :, -1] += vs.dt_tracer * vs.forc_temp_surface[2:-2, 2:-2] / vs.dzt[-1]
        sol, mask = utilities.solve_implicit(vs, ks, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)
        vs.temp[2:-2, 2:-2, :, vs.taup1] = utilities.where(vs, mask, sol, vs.temp[2:-2, 2:-2, :, vs.taup1])
        d_tri[...] = vs.salt[2:-2, 2:-2, :, vs.taup1]
        d_tri[:, :, -1] += vs.dt_tracer * vs.forc_salt_surface[2:-2, 2:-2] / vs.dzt[-1]
        sol, mask = utilities.solve_implicit(
            vs, ks, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge
        )
        vs.salt[2:-2, 2:-2, :, vs.taup1] = utilities.where(vs, mask, sol, vs.salt[2:-2, 2:-2, :, vs.taup1])

        vs.dtemp_vmix[...] = (vs.temp[:, :, :, vs.taup1] -
                            vs.dtemp_vmix) / vs.dt_tracer
        vs.dsalt_vmix[...] = (vs.salt[:, :, :, vs.taup1] -
                            vs.dsalt_vmix) / vs.dt_tracer

    """
    boundary exchange
    """
    utilities.enforce_boundaries(vs, vs.temp[..., vs.taup1])
    utilities.enforce_boundaries(vs, vs.salt[..., vs.taup1])

    with vs.timers['eq_of_state']:
        calc_eq_of_state(vs, vs.taup1)

    """
    surface density flux
    """
    vs.forc_rho_surface[...] = vs.maskT[:, :, -1] * (
        density.get_drhodT(vs, vs.salt[:, :, -1, vs.taup1],
                        vs.temp[:, :, -1, vs.taup1],
                        np.abs(vs.zt[-1])) * vs.forc_temp_surface
        + density.get_drhodS(vs, vs.salt[:, :, -1, vs.taup1],
                            vs.temp[:, :, -1, vs.taup1],
                            np.abs(vs.zt[-1])) * vs.forc_salt_surface
        )

    with vs.timers['vmix']:
        vs.P_diss_v[...] = 0.0
        if vs.enable_conserve_energy:
            """
            diagnose dissipation of dynamic enthalpy by vertical mixing
            """
            fxa = (-vs.int_drhodT[2:-2, 2:-2, 1:, vs.taup1] + vs.int_drhodT[2:-2, 2:-2, :-1, vs.taup1]) \
                / vs.dzw[np.newaxis, np.newaxis, :-1]
            vs.P_diss_v[2:-2, 2:-2, :-1] += -vs.grav / vs.rho_0 * fxa * vs.kappaH[2:-2, 2:-2, :-1] \
                * (vs.temp[2:-2, 2:-2, 1:, vs.taup1] - vs.temp[2:-2, 2:-2, :-1, vs.taup1]) \
                / vs.dzw[np.newaxis, np.newaxis, :-1] * vs.maskW[2:-2, 2:-2, :-1]
            fxa = (-vs.int_drhodS[2:-2, 2:-2, 1:, vs.taup1] + vs.int_drhodS[2:-2, 2:-2, :-1, vs.taup1]) \
                / vs.dzw[np.newaxis, np.newaxis, :-1]
            vs.P_diss_v[2:-2, 2:-2, :-1] += -vs.grav / vs.rho_0 * fxa * vs.kappaH[2:-2, 2:-2, :-1] \
                * (vs.salt[2:-2, 2:-2, 1:, vs.taup1] - vs.salt[2:-2, 2:-2, :-1, vs.taup1]) \
                / vs.dzw[np.newaxis, np.newaxis, :-1] * vs.maskW[2:-2, 2:-2, :-1]

            fxa = 2 * vs.int_drhodT[2:-2, 2:-2, -1, vs.taup1] / vs.dzw[-1]
            vs.P_diss_v[2:-2, 2:-2, -1] += - vs.grav / vs.rho_0 * fxa * \
                vs.forc_temp_surface[2:-2, 2:-2] * vs.maskW[2:-2, 2:-2, -1]
            fxa = 2 * vs.int_drhodS[2:-2, 2:-2, -1, vs.taup1] / vs.dzw[-1]
            vs.P_diss_v[2:-2, 2:-2, -1] += - vs.grav / vs.rho_0 * fxa * \
                vs.forc_salt_surface[2:-2, 2:-2] * vs.maskW[2:-2, 2:-2, -1]

        if vs.enable_conserve_energy:
            """
            determine effect due to nonlinear equation of state
            """
            aloc[:, :, :-1] = vs.kappaH[:, :, :-1] * vs.Nsqr[:, :, :-1, vs.taup1]
            vs.P_diss_nonlin[:, :, :-1] = vs.P_diss_v[:, :, :-1] - aloc[:, :, :-1]
            vs.P_diss_v[:, :, :-1] = aloc[:, :, :-1]
        else:
            """
            diagnose N^2 vs.kappaH, i.e. exchange of pot. energy with TKE
            """
            vs.P_diss_v[:, :, :-1] = vs.kappaH[:, :, :-1] * vs.Nsqr[:, :, :-1, vs.taup1]
            vs.P_diss_v[:, :, -1] = -vs.forc_rho_surface * vs.maskT[:, :, -1] * vs.grav / vs.rho_0


@veros_method
def advect_tracer(vs, tr, dtr):
    """
    calculate time tendency of a tracer due to advection
    """
    if vs.enable_superbee_advection:
        advection.adv_flux_superbee(vs, vs.flux_east, vs.flux_north, vs.flux_top, tr)
    else:
        advection.adv_flux_2nd(vs, vs.flux_east, vs.flux_north, vs.flux_top, tr)
    dtr[2:-2, 2:-2, :] = vs.maskT[2:-2, 2:-2, :] * (-(vs.flux_east[2:-2, 2:-2, :] - vs.flux_east[1:-3, 2:-2, :])
                                                    / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxt[2:-2, np.newaxis, np.newaxis])
                                                   - (vs.flux_north[2:-2, 2:-2, :] - vs.flux_north[2:-2, 1:-3, :])
                                                    / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dyt[np.newaxis, 2:-2, np.newaxis]))
    dtr[:, :, 0] += -vs.maskT[:, :, 0] * vs.flux_top[:, :, 0] / vs.dzt[0]
    dtr[:, :, 1:] += -vs.maskT[:, :, 1:] * (vs.flux_top[:, :, 1:] - vs.flux_top[:, :, :-1]) / vs.dzt[1:]


@veros_method
def advect_temperature(vs):
    """
    integrate temperature
    """
    return advect_tracer(vs, vs.temp[..., vs.tau], vs.dtemp[..., vs.tau])


@veros_method
def advect_salinity(vs):
    """
    integrate salinity
    """
    return advect_tracer(vs, vs.salt[..., vs.tau], vs.dsalt[..., vs.tau])


@veros_method
def calc_eq_of_state(vs, n):
    """
    calculate density, stability frequency, dynamic enthalpy and derivatives
    for time level n from temperature and salinity
    """
    density_args = (vs, vs.salt[..., n], vs.temp[..., n], np.abs(vs.zt))

    """
    calculate new density
    """
    vs.rho[..., n] = density.get_rho(*density_args) * vs.maskT

    """
    calculate new potential density
    """
    vs.prho[...] = density.get_potential_rho(*density_args) * vs.maskT

    if vs.enable_conserve_energy:
        """
        calculate new dynamic enthalpy and derivatives
        """
        vs.Hd[..., n] = density.get_dyn_enthalpy(*density_args) * vs.maskT
        vs.int_drhodT[..., n] = density.get_int_drhodT(*density_args)
        vs.int_drhodS[..., n] = density.get_int_drhodS(*density_args)

    """
    new stability frequency
    """
    fxa = -vs.grav / vs.rho_0 / vs.dzw[np.newaxis, np.newaxis, :-1] * vs.maskW[:, :, :-1]
    vs.Nsqr[:, :, :-1, n] = fxa * (density.get_rho(
                                        vs, vs.salt[:, :, 1:, n], vs.temp[:, :, 1:, n], np.abs(vs.zt[:-1])
                                    ) - vs.rho[:, :, :-1, n]
                                  )
    vs.Nsqr[:, :, -1, n] = vs.Nsqr[:, :, -2, n]

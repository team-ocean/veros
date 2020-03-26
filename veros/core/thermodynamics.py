from veros.core.operators import numpy as np

from veros import veros_routine, veros_kernel, run_kernel
from veros.distributed import global_sum
from veros.core import advection, diffusion, isoneutral, density, utilities
from veros.core.operators import update, update_add, at


@veros_kernel(static_args=('enable_superbee_advection',))
def advect_tracer(tr, dtr, flux_east, flux_north, flux_top, u_wgrid, v_wgrid, w_wgrid,
                  dxt, dyt, dzt, dzw, maskT, maskU, maskV, maskW, cost, cosu, dt_tracer,
                  tau, enable_superbee_advection):
    """
    calculate time tendency of a tracer due to advection
    """
    if enable_superbee_advection:
        flux_east, flux_north, flux_top = advection.adv_flux_superbee(
            flux_east, flux_north, flux_top, tr, u_wgrid, v_wgrid, w_wgrid,
            dxt, dyt, dzw, maskU, maskV, maskW, cost, cosu, dt_tracer, tau
        )
    else:
        flux_east, flux_north, flux_top = advection.adv_flux_2nd(
            flux_east, flux_north, flux_top, tr, u_wgrid, v_wgrid, w_wgrid,
            cosu, maskU, maskV, maskW, tau
        )
    dtr = update(dtr, at[2:-2, 2:-2, :], maskT[2:-2, 2:-2, :] * (-(flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
                                                 / (cost[np.newaxis, 2:-2, np.newaxis] * dxt[2:-2, np.newaxis, np.newaxis])
                                                 - (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
                                                 / (cost[np.newaxis, 2:-2, np.newaxis] * dyt[np.newaxis, 2:-2, np.newaxis])))
    dtr = update_add(dtr, at[:, :, 0], -maskT[:, :, 0] * flux_top[:, :, 0] / dzt[0])
    dtr = update_add(dtr, at[:, :, 1:], -maskT[:, :, 1:] * (flux_top[:, :, 1:] - flux_top[:, :, :-1]) / dzt[1:])

    return dtr


@veros_kernel(static_args=('enable_superbee_advection',))
def advect_temperature(temp, dtemp, tau, flux_east, flux_north, flux_top, u,
                       v, w, dxt, dyt, dzt, dzw, maskT, maskU, maskV,
                       maskW, cost, cosu, dt_tracer, enable_superbee_advection):
    """
    integrate temperature
    """
    return advect_tracer(temp[..., tau], dtemp[..., tau], flux_east, flux_north,
                         flux_top, u, v, w, dxt, dyt, dzt, dzw,
                         maskT, maskU, maskV, maskW, cost, cosu, dt_tracer,
                         tau, enable_superbee_advection)


@veros_kernel(static_args=('enable_superbee_advection',))
def advect_salinity(salt, dsalt, tau, flux_east, flux_north, flux_top, u,
                    v, w, dxt, dyt, dzt, dzw, maskT, maskU, maskV,
                    maskW, cost, cosu, dt_tracer, enable_superbee_advection):
    """
    integrate salinity
    """
    return advect_tracer(salt[..., tau], dsalt[..., tau], flux_east, flux_north,
                         flux_top, u, v, w, dxt, dyt, dzt, dzw,
                         maskT, maskU, maskV, maskW, cost, cosu, dt_tracer,
                         tau, enable_superbee_advection)


@veros_kernel(static_args=('enable_conserve_energy', 'eq_of_state_type'))
def calc_eq_of_state(n, salt, temp, rho, prho, Hd, int_drhodT, int_drhodS, Nsqr,
                     zt, dzw, maskT, maskW, grav, rho_0, enable_conserve_energy, eq_of_state_type):
    """
    calculate density, stability frequency, dynamic enthalpy and derivatives
    for time level n from temperature and salinity
    """
    salt = salt[..., n]
    temp = temp[..., n]
    press = np.abs(zt)

    """
    calculate new density
    """
    rho = update(rho, at[..., n], density.get_rho(eq_of_state_type, salt, temp, press) * maskT)

    """
    calculate new potential density
    """
    prho = update(prho, at[...], density.get_potential_rho(eq_of_state_type, salt, temp) * maskT)

    """
    calculate new dynamic enthalpy and derivatives
    """
    if enable_conserve_energy:
        Hd = update(Hd, at[..., n], density.get_dyn_enthalpy(eq_of_state_type, salt, temp, press) * maskT)
        int_drhodT = update(int_drhodT, at[..., n], density.get_int_drhodT(eq_of_state_type, salt, temp, press))
        int_drhodS = update(int_drhodS, at[..., n], density.get_int_drhodS(eq_of_state_type, salt, temp, press))

    """
    new stability frequency
    """
    fxa = -grav / rho_0 / dzw[np.newaxis, np.newaxis, :-1] * maskW[:, :, :-1]
    Nsqr = update(Nsqr, at[:, :, :-1, n], fxa * (density.get_rho(
                                    eq_of_state_type, salt[:, :, 1:], temp[:, :, 1:], press[:-1]
                                ) - rho[:, :, :-1, n]))
    Nsqr = update(Nsqr, at[:, :, -1, n], Nsqr[:, :, -2, n])

    return rho, prho, Hd, int_drhodT, int_drhodS, Nsqr


@veros_kernel(static_args=('enable_conserve_energy', 'enable_superbee_advection', 'enable_tke', 'nz'))
def advect_temp_salt_enthalpy(temp, dtemp, salt, dsalt, u, v, w, nz, tau,
                              taup1, taum1, dt_tracer, dxt, dyt, dzt, dzw, maskT, maskU, maskV, maskW, cost, cosu,
                              area_t, Hd, dHd, grav, rho_0, AB_eps, int_drhodT, int_drhodS, rho, P_diss_adv,
                              tke, enable_superbee_advection, enable_conserve_energy, enable_tke, kbot):
    """
    integrate temperature and salinity and diagnose sources of dynamic enthalpy
    """
    flux_east = np.zeros_like(maskU)
    flux_north = np.zeros_like(maskV)
    flux_top = np.zeros_like(maskW)

    dtemp = update(dtemp, at[..., tau], advect_temperature(temp, dtemp, tau, flux_east, flux_north, flux_top,
                               u, v, w, dxt, dyt, dzt, dzw,
                               maskT, maskU, maskV, maskW, cost, cosu, dt_tracer,
                               enable_superbee_advection))

    dsalt = update(dsalt, at[..., tau], advect_salinity(salt, dsalt, tau, flux_east, flux_north, flux_top,
                            u, v, w, dxt, dyt, dzt, dzw,
                            maskT, maskU, maskV, maskW, cost, cosu, dt_tracer,
                            enable_superbee_advection))

    if enable_conserve_energy:
        """
        advection of dynamic enthalpy
        """
        if enable_superbee_advection:
            flux_east, flux_north, flux_top = \
                advection.adv_flux_superbee(flux_east, flux_north, flux_top,
                                            Hd[:, :, :, tau], u, v, w, dxt, dyt, dzt,
                                            maskU, maskV, maskW, cost, cosu, dt_tracer, tau)
        else:
            flux_east, flux_north, flux_top = \
                advection.adv_flux_2nd(flux_east, flux_north, flux_top,
                                       Hd[:, :, :, tau], u, v, w, cosu,
                                       maskU, maskV, maskW, tau)

        dHd = update(dHd, at[2:-2, 2:-2, :, tau], maskT[2:-2, 2:-2, :] * (-(flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
                                                          / (cost[np.newaxis, 2:-2, np.newaxis] * dxt[2:-2, np.newaxis, np.newaxis])
                                                          - (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
                                                          / (cost[np.newaxis, 2:-2, np.newaxis] * dyt[np.newaxis, 2:-2, np.newaxis])))
        dHd = update_add(dHd, at[:, :, 0, tau], -maskT[:, :, 0] \
            * flux_top[:, :, 0] / dzt[0])
        dHd = update_add(dHd, at[:, :, 1:, tau], -maskT[:, :, 1:] \
            * (flux_top[:, :, 1:] - flux_top[:, :, :-1]) \
            / dzt[np.newaxis, np.newaxis, 1:])

        """
        changes in dyn. Enthalpy due to advection
        """
        diss = np.zeros_like(u[..., 0])
        diss = update(diss, at[2:-2, 2:-2, :], grav / rho_0 * (-int_drhodT[2:-2, 2:-2, :, tau] * dtemp[2:-2, 2:-2, :, tau]
                                              - int_drhodS[2:-2, 2:-2, :, tau] * dsalt[2:-2, 2:-2, :, tau]) \
                                           - dHd[2:-2, 2:-2, :, tau])

        """
        contribution by vertical advection is - g rho w / rho0, substract this also
        """
        diss = update_add(diss, at[:, :, :-1], -0.25 * grav / rho_0 * w[:, :, :-1, tau] \
            * (rho[:, :, :-1, tau] + rho[:, :, 1:, tau]) \
            * dzw[np.newaxis, np.newaxis, :-1] / dzt[np.newaxis, np.newaxis, :-1])
        diss = update_add(diss, at[:, :, 1:], -0.25 * grav / rho_0 * w[:, :, :-1, tau] \
            * (rho[:, :, 1:, tau] + rho[:, :, :-1, tau]) \
            * dzw[np.newaxis, np.newaxis, :-1] / dzt[np.newaxis, np.newaxis, 1:])

    if enable_conserve_energy and enable_tke:
        """
        dissipation by advection interpolated on W-grid
        """
        P_diss_adv_w = diffusion.dissipation_on_wgrid(diss, nz, dzw, kbot)

        """
        distribute P_diss_adv_w over domain, prevent draining of TKE
        """
        fxa = np.sum(area_t[2:-2, 2:-2, np.newaxis] * P_diss_adv_w[2:-2, 2:-2, :-1]
                     * dzw[np.newaxis, np.newaxis, :-1] * maskW[2:-2, 2:-2, :-1]) \
            + np.sum(0.5 * area_t[2:-2, 2:-2] * P_diss_adv[2:-2, 2:-2, -1]
                     * dzw[-1] * maskW[2:-2, 2:-2, -1])
        tke_mask = tke[2:-2, 2:-2, :-1, tau] > 0.
        fxb = np.sum(area_t[2:-2, 2:-2, np.newaxis] * dzw[np.newaxis, np.newaxis, :-1] * maskW[2:-2, 2:-2, :-1] * tke_mask) \
            + np.sum(0.5 * area_t[2:-2, 2:-2] * dzw[-1] * maskW[2:-2, 2:-2, -1])

        fxa = global_sum(fxa)
        fxb = global_sum(fxb)

        P_diss_adv = update(P_diss_adv, at[2:-2, 2:-2, :-1], fxa / fxb * tke_mask)
        P_diss_adv = update(P_diss_adv, at[2:-2, 2:-2, -1], fxa / fxb)

    """
    Adam Bashforth time stepping for advection
    """
    temp = update(temp, at[:, :, :, taup1], temp[:, :, :, tau] + dt_tracer \
        * ((1.5 + AB_eps) * dtemp[:, :, :, tau]
           - (0.5 + AB_eps) * dtemp[:, :, :, taum1]) * maskT)
    salt = update(salt, at[:, :, :, taup1], salt[:, :, :, tau] + dt_tracer \
        * ((1.5 + AB_eps) * dsalt[:, :, :, tau]
           - (0.5 + AB_eps) * dsalt[:, :, :, taum1]) * maskT)

    return temp, salt, dtemp, dsalt, dHd, P_diss_adv


@veros_kernel(static_args=('enable_cyclic_x'))
def vertmix_tempsalt(temp, salt, dtemp_vmix, dsalt_vmix, kappaH, kbot, dzt, dzw, taup1,
                     dt_tracer, forc_temp_surface, forc_salt_surface, enable_cyclic_x):
    """
    vertical mixing of temperature and salinity
    """
    dtemp_vmix = update(dtemp_vmix, at[...], temp[:, :, :, taup1])
    dsalt_vmix = update(dsalt_vmix, at[...], salt[:, :, :, taup1])

    a_tri = np.zeros_like(salt[2:-2, 2:-2, :, taup1])
    b_tri = np.zeros_like(salt[2:-2, 2:-2, :, taup1])
    c_tri = np.zeros_like(salt[2:-2, 2:-2, :, taup1])
    d_tri = np.zeros_like(salt[2:-2, 2:-2, :, taup1])
    delta = np.zeros_like(salt[2:-2, 2:-2, :, taup1])

    ks = kbot[2:-2, 2:-2] - 1
    delta = update(delta, at[:, :, :-1], dt_tracer / dzw[np.newaxis, np.newaxis, :-1] \
        * kappaH[2:-2, 2:-2, :-1])
    delta = update(delta, at[:, :, -1], 0.)
    a_tri = update(a_tri, at[:, :, 1:], -delta[:, :, :-1] / dzt[np.newaxis, np.newaxis, 1:])
    b_tri = update(b_tri, at[:, :, 1:], 1 + (delta[:, :, 1:] + delta[:, :, :-1]) \
        / dzt[np.newaxis, np.newaxis, 1:])
    b_tri_edge = 1 + delta / dzt[np.newaxis, np.newaxis, :]
    c_tri = update(c_tri, at[:, :, :-1], -delta[:, :, :-1] / dzt[np.newaxis, np.newaxis, :-1])
    d_tri = temp[2:-2, 2:-2, :, taup1]
    d_tri = update_add(d_tri, at[:, :, -1], dt_tracer * forc_temp_surface[2:-2, 2:-2] / dzt[-1])
    sol, mask = utilities.solve_implicit(ks, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)
    temp = update(temp, at[2:-2, 2:-2, :, taup1], utilities.where(mask, sol, temp[2:-2, 2:-2, :, taup1]))
    d_tri = salt[2:-2, 2:-2, :, taup1]
    d_tri = update_add(d_tri, at[:, :, -1], dt_tracer * forc_salt_surface[2:-2, 2:-2] / dzt[-1])
    sol, mask = utilities.solve_implicit(
        ks, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge
    )
    salt = update(salt, at[2:-2, 2:-2, :, taup1], utilities.where(mask, sol, salt[2:-2, 2:-2, :, taup1]))

    dtemp_vmix = update(dtemp_vmix, at[...], (temp[:, :, :, taup1] - dtemp_vmix) / dt_tracer)
    dsalt_vmix = update(dsalt_vmix, at[...], (salt[:, :, :, taup1] - dsalt_vmix) / dt_tracer)

    """
    boundary exchange
    """
    temp = update(temp, at[..., taup1], utilities.enforce_boundaries(temp[..., taup1], enable_cyclic_x))
    salt = update(salt, at[..., taup1], utilities.enforce_boundaries(salt[..., taup1], enable_cyclic_x))

    return dtemp_vmix, temp, dsalt_vmix, salt


@veros_kernel(static_args=('eq_of_state_type'))
def surf_densityf(salt, temp, zt, maskT, taup1, forc_temp_surface, forc_salt_surface, eq_of_state_type):
    """
    surface density flux
    """
    return maskT[:, :, -1] * (
        density.get_drhodT(eq_of_state_type,
                           salt[:, :, -1, taup1],
                           temp[:, :, -1, taup1],
                           np.abs(zt[-1])) * forc_temp_surface
        + density.get_drhodS(eq_of_state_type,
                             salt[:, :, -1, taup1],
                             temp[:, :, -1, taup1],
                             np.abs(zt[-1])) * forc_salt_surface
    )


@veros_kernel(static_args=('enable_conserve_energy'))
def diag_P_diss_v(P_diss_v, P_diss_nonlin, int_drhodT, int_drhodS, temp, salt, kappaH,
                  grav, rho_0, dzw, maskT, maskW, taup1, forc_temp_surface,
                  forc_salt_surface, forc_rho_surface, Nsqr, enable_conserve_energy):
    P_diss_v = update(P_diss_v, at[...], 0.0)
    aloc = np.zeros_like(kappaH)

    if enable_conserve_energy:
        """
        diagnose dissipation of dynamic enthalpy by vertical mixing
        """
        fxa = (-int_drhodT[2:-2, 2:-2, 1:, taup1] + int_drhodT[2:-2, 2:-2, :-1, taup1]) \
            / dzw[np.newaxis, np.newaxis, :-1]
        P_diss_v = update_add(P_diss_v, at[2:-2, 2:-2, :-1], -grav / rho_0 * fxa * kappaH[2:-2, 2:-2, :-1] \
            * (temp[2:-2, 2:-2, 1:, taup1] - temp[2:-2, 2:-2, :-1, taup1]) \
            / dzw[np.newaxis, np.newaxis, :-1] * maskW[2:-2, 2:-2, :-1])
        fxa = (-int_drhodS[2:-2, 2:-2, 1:, taup1] + int_drhodS[2:-2, 2:-2, :-1, taup1]) \
            / dzw[np.newaxis, np.newaxis, :-1]
        P_diss_v = update_add(P_diss_v, at[2:-2, 2:-2, :-1], -grav / rho_0 * fxa * kappaH[2:-2, 2:-2, :-1] \
            * (salt[2:-2, 2:-2, 1:, taup1] - salt[2:-2, 2:-2, :-1, taup1]) \
            / dzw[np.newaxis, np.newaxis, :-1] * maskW[2:-2, 2:-2, :-1])

        fxa = 2 * int_drhodT[2:-2, 2:-2, -1, taup1] / dzw[-1]
        P_diss_v = update_add(P_diss_v, at[2:-2, 2:-2, -1], - grav / rho_0 * fxa * \
            forc_temp_surface[2:-2, 2:-2] * maskW[2:-2, 2:-2, -1])
        fxa = 2 * int_drhodS[2:-2, 2:-2, -1, taup1] / dzw[-1]
        P_diss_v = update_add(P_diss_v, at[2:-2, 2:-2, -1], - grav / rho_0 * fxa * \
            forc_salt_surface[2:-2, 2:-2] * maskW[2:-2, 2:-2, -1])

    if enable_conserve_energy:
        """
        determine effect due to nonlinear equation of state
        """
        aloc = update(aloc, at[:, :, :-1], kappaH[:, :, :-1] * Nsqr[:, :, :-1, taup1])
        P_diss_nonlin = update(P_diss_nonlin, at[:, :, :-1], P_diss_v[:, :, :-1] - aloc[:, :, :-1])
        P_diss_v = update(P_diss_v, at[:, :, :-1], aloc[:, :, :-1])
    else:
        """
        diagnose N^2 vs.kappaH, i.e. exchange of pot. energy with TKE
        """
        P_diss_v = update(P_diss_v, at[:, :, :-1], kappaH[:, :, :-1] * Nsqr[:, :, :-1, taup1])
        P_diss_v = update(P_diss_v, at[:, :, -1], -forc_rho_surface * maskT[:, :, -1] * grav / rho_0)

    return P_diss_v, P_diss_nonlin


@veros_routine(
    inputs=(
        'temp', 'salt', 'dtemp', 'dsalt',
        'dtemp_vmix', 'dsalt_vmix', 'dtemp_iso', 'dsalt_iso',
        'u', 'v', 'w', 'u_wgrid', 'v_wgrid', 'w_wgrid',
        'tau', 'taup1', 'taum1', 'dt_tracer', 'kbot',
        'dxt', 'dxu', 'dyt', 'dyu', 'dzt', 'zt', 'dzw',
        'maskT', 'maskU', 'maskV', 'maskW', 'cost', 'cosu',
        'area_t', 'Hd', 'dHd', 'temp_source', 'salt_source',
        'P_diss_skew', 'P_diss_iso', 'grav', 'rho_0',
        'int_drhodT', 'int_drhodS',
        'rho', 'prho', 'P_diss_adv', 'tke',
        'P_diss_v', 'P_diss_nonlin', 'kappaH',
        'forc_temp_surface', 'forc_salt_surface',
        'forc_rho_surface', 'Nsqr',
        'K_11', 'K_22', 'K_33', 'K_gm', 'K_iso',
        'Ai_ez', 'Ai_nz', 'Ai_bx', 'Ai_by',
        'flux_east', 'flux_north', 'flux_top',
    ),
    outputs=(
        'temp', 'salt',
        'dtemp', 'dsalt',
        'dtemp_hmix', 'dsalt_hmix',
        'dtemp_vmix', 'dsalt_vmix',
        'dtemp_iso', 'dsalt_iso',
        'P_diss_iso',
        'P_diss_skew',
        'P_diss_adv',
        'P_diss_hmix',
        'P_diss_sources',
        'rho', 'prho',
        'Hd', 'dHd',
        'int_drhodT',
        'int_drhodS',
        'Nsqr',
        'P_diss_v',
        'P_diss_nonlin',
        'forc_rho_surface',
        'K_11', 'K_22', 'K_33',
        'Ai_ez', 'Ai_nz', 'Ai_bx', 'Ai_by'
    ),
    settings=(
        'K_h',
        'K_hbi',
        'K_iso_steep',
        'nz',
        'dt_tracer',
        'AB_eps',
        'iso_slopec',
        'iso_dslope',
        'enable_superbee_advection',
        'enable_conserve_energy',
        'enable_tke',
        'enable_hor_diffusion',
        'enable_biharmonic_mixing',
        'enable_tempsalt_sources',
        'enable_skew_diffusion',
        'enable_neutral_diffusion',
        'hor_friction_cosPower',
        'enable_hor_friction_cos_scaling',
        'enable_cyclic_x',
        'pyom_compatibility_mode',
    ),
)
def thermodynamics(vs):
    """
    integrate temperature and salinity and diagnose sources of dynamic enthalpy
    """
    """
    Advection tendencies for temperature, salinity and dynamic enthalpy
    """
    temp, salt, dtemp, dsalt, dHd, P_diss_adv = run_kernel(advect_temp_salt_enthalpy, vs)
    """
    horizontal diffusion
    """
    with vs.timers['isoneutral']:
        dtemp_hmix = dsalt_hmix = P_diss_hmix = 0
        if vs.enable_hor_diffusion:
            temp, salt, dtemp_hmix, dsalt_hmix, P_diss_hmix = run_kernel(
                diffusion.tempsalt_diffusion, vs, temp=temp, salt=salt
            )
        if vs.enable_biharmonic_mixing:
            temp, salt, dtemp_hmix, dsalt_hmix, P_diss_hmix = run_kernel(
                diffusion.tempsalt_biharmonic, vs, temp=temp, salt=salt
            )

        """
        sources like restoring zones, etc
        """
        P_diss_sources = None
        if vs.enable_tempsalt_sources:
            temp, salt, P_diss_sources = run_kernel(
                diffusion.tempsalt_sources, vs, temp=temp, salt=salt
            )

        """
        isopycnal diffusion
        """
        if vs.enable_neutral_diffusion:
            P_diss_iso = update(vs.P_diss_iso, at[...], 0.0)
            dtemp_iso = update(vs.dtemp_iso, at[...], 0.0)
            dsalt_iso = update(vs.dsalt_iso, at[...], 0.0)

            Ai_ez, Ai_nz, Ai_bx, Ai_by, K_11, K_22, K_33 = run_kernel(
                isoneutral.isoneutral_diffusion_pre, vs, salt=salt, temp=temp
            )

            temp, dtemp_iso, P_diss_iso = run_kernel(
                isoneutral.isoneutral_diffusion, vs,
                tr=temp, istemp=True, K_11=K_11, K_22=K_22, K_33=K_33,
                Ai_ez=Ai_ez, Ai_nz=Ai_nz, Ai_bx=Ai_bx, Ai_by=Ai_by,
                temp=temp, salt=salt
            )

            salt, dsalt_iso, P_diss_iso = run_kernel(
                isoneutral.isoneutral_diffusion, vs,
                tr=salt, istemp=False, K_11=K_11, K_22=K_22, K_33=K_33,
                Ai_ez=Ai_ez, Ai_nz=Ai_nz, Ai_bx=Ai_bx, Ai_by=Ai_by,
                P_diss_iso=P_diss_iso, temp=temp, salt=salt
            )

            P_diss_skew = None
            if vs.enable_skew_diffusion:
                P_diss_skew = update(vs.P_diss_skew, at[...], 0.0)
                temp, dtemp_iso, P_diss_skew = run_kernel(
                    isoneutral.isoneutral_skew_diffusion, vs,
                    tr=temp, istemp=True, dtemp_iso=dtemp_iso,
                    dsalt_iso=dsalt_iso, K_11=K_11, K_22=K_22, K_33=K_33,
                    Ai_ez=Ai_ez, Ai_nz=Ai_nz, Ai_bx=Ai_bx, Ai_by=Ai_by,
                    P_diss_iso=P_diss_iso, temp=temp, salt=salt
                )

                salt, dsalt_iso, P_diss_skew = run_kernel(
                    isoneutral.isoneutral_skew_diffusion, vs,
                    tr=salt, istemp=False, dtemp_iso=dtemp_iso,
                    dsalt_iso=dsalt_iso, K_11=K_11, K_22=K_22, K_33=K_33,
                    Ai_ez=Ai_ez, Ai_nz=Ai_nz, Ai_bx=Ai_bx, Ai_by=Ai_by,
                    P_diss_iso=P_diss_iso, P_diss_skew=P_diss_skew,
                    temp=temp, salt=salt
                )

    with vs.timers['vmix']:
        dtemp_vmix, temp, dsalt_vmix, salt = run_kernel(vertmix_tempsalt, vs, temp=temp, salt=salt)

    with vs.timers['eq_of_state']:
        rho, prho, Hd, int_drhodT, int_drhodS, Nsqr = run_kernel(
            calc_eq_of_state, vs, n=vs.taup1, salt=salt, temp=temp
        )

    """
    surface density flux
    """
    forc_rho_surface = run_kernel(surf_densityf, vs, salt=salt, temp=temp)

    with vs.timers['vmix']:
        P_diss_v, P_diss_nonlin = run_kernel(
            diag_P_diss_v, vs,
            int_drhodT=int_drhodT, int_drhodS=int_drhodS,
            temp=temp, salt=salt, forc_rho_surface=forc_rho_surface, Nsqr=Nsqr
        )

    return dict(
        temp=temp, salt=salt,
        dtemp=dtemp, dsalt=dsalt,
        dtemp_hmix=dtemp_hmix,
        dsalt_hmix=dsalt_hmix,
        dtemp_vmix=dtemp_vmix,
        dsalt_vmix=dsalt_vmix,
        dtemp_iso=dtemp_iso,
        dsalt_iso=dsalt_iso,
        P_diss_adv=P_diss_adv,
        P_diss_hmix=P_diss_hmix,
        P_diss_skew=P_diss_skew,
        P_diss_sources=P_diss_sources,
        rho=rho, prho=prho,
        Hd=Hd, dHd=dHd,
        int_drhodT=int_drhodT,
        int_drhodS=int_drhodS,
        Nsqr=Nsqr,
        P_diss_v=P_diss_v,
        P_diss_iso=P_diss_iso,
        P_diss_nonlin=P_diss_nonlin,
        forc_rho_surface=forc_rho_surface,
        K_11=K_11, K_22=K_22, K_33=K_33,
        Ai_ez=Ai_ez, Ai_nz=Ai_nz,
        Ai_bx=Ai_bx, Ai_by=Ai_by
    )

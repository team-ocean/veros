from veros.state import KernelOutput
from veros.core.operators import numpy as np

from veros import veros_kernel, veros_routine, runtime_settings
from veros.variables import allocate
from veros.core import advection, utilities
from veros.core.operators import update, update_add, at, for_loop


@veros_routine
def set_tke_diffusivities(state):
    vs = state.variables
    settings = state.settings

    if settings.enable_tke:
        tke_diff_out = set_tke_diffusivities_kernel(state)
        vs.update(tke_diff_out)
    else:
        vs.kappaM = vs.kappaM_0 * np.ones_like(vs.kappaM)
        vs.kappaH = vs.kappaH_0 * np.ones_like(vs.kappaH)


@veros_kernel
def set_tke_diffusivities_kernel(state):
    """
    set vertical diffusivities based on TKE model
    """
    vs = state.variables
    settings = state.settings

    Rinumber = allocate(state.dimensions, ("xt", "yt", "zt"))

    sqrttke = np.sqrt(np.maximum(0., vs.tke[:, :, :, vs.tau]))
    """
    calculate buoyancy length scale
    """
    mxl = np.sqrt(2) * sqrttke \
        / np.sqrt(np.maximum(1e-12, vs.Nsqr[:, :, :, vs.tau])) * vs.maskW

    """
    apply limits for mixing length
    """
    if settings.tke_mxl_choice == 1:
        """
        bounded by the distance to surface/bottom
        """
        mxl = np.minimum(
            np.minimum(mxl, -vs.zw[np.newaxis, np.newaxis, :]
                        + vs.dzw[np.newaxis, np.newaxis, :] * 0.5
                        ), vs.ht[:, :, np.newaxis] + vs.zw[np.newaxis, np.newaxis, :]
        )
        mxl = np.maximum(mxl, settings.mxl_min)
    elif settings.tke_mxl_choice == 2:
        """
        bound length scale as in mitgcm/OPA code
        """
        nz = state.dimensions["zt"]

        def backwards_pass(k, mxl):
            return update(mxl, at[:, :, k], np.minimum(mxl[:, :, k], mxl[:, :, k + 1] + vs.dzt[k + 1]))

        mxl = for_loop(nz - 2, -1, backwards_pass, mxl)
        mxl = update(mxl, at[:, :, -1], np.minimum(mxl[:, :, -1], settings.mxl_min + vs.dzt[-1]))

        def forwards_pass(k, mxl):
            return update(mxl, at[:, :, k], np.minimum(mxl[:, :, k], mxl[:, :, k - 1] + vs.dzt[k]))

        mxl = for_loop(1, nz, forwards_pass, mxl)
        mxl = np.maximum(mxl, settings.mxl_min)
    else:
        raise ValueError('unknown mixing length choice in tke_mxl_choice')

    """
    calculate viscosity and diffusivity based on Prandtl number
    """
    K_diss_v = utilities.enforce_boundaries(vs.K_diss_v, settings.enable_cyclic_x)
    kappaM = update(vs.kappaM, at[...], np.minimum(settings.kappaM_max, settings.c_k * mxl * sqrttke))
    Rinumber = update(Rinumber, at[...], vs.Nsqr[:, :, :, vs.tau] / \
        np.maximum(K_diss_v / np.maximum(1e-12, kappaM), 1e-12))
    if settings.enable_idemix:
        Rinumber = update(Rinumber, at[...], np.minimum(Rinumber, kappaM * vs.Nsqr[:, :, :, vs.tau]
                                    / np.maximum(1e-12, vs.alpha_c * vs.E_iw[:, :, :, vs.tau]**2)))
    if settings.enable_Prandtl_tke:
        Prandtlnumber = np.maximum(1., np.minimum(10, 6.6 * Rinumber))
    else:
        Prandtlnumber = update(vs.Prandtlnumber, at[...], settings.Prandtl_tke0)
    kappaH = np.maximum(settings.kappaH_min, kappaM / Prandtlnumber)

    if settings.enable_kappaH_profile:
        # Correct diffusivity according to
        # Bryan, K., and L. J. Lewis, 1979:
        # A water mass model of the world ocean. J. Geophys. Res., 84, 2503–2517.
        # It mainly modifies kappaH within 20S - 20N deg. belt
        kappaH = np.maximum(kappaH, (0.8 + 1.05 / settings.pi
                                            * np.arctan((-vs.zw[np.newaxis, np.newaxis, :] - 2500.)
                                                        / 222.2)) * 1e-4)
    kappaM = np.maximum(settings.kappaM_min, kappaM)

    return KernelOutput(sqrttke=sqrttke, mxl=mxl, kappaM=kappaM, kappaH=kappaH, Prandtlnumber=Prandtlnumber)


@veros_routine
def integrate_tke(state):
    vs = state.variables
    tke_out = integrate_tke_kernel(state)
    vs.update(tke_out)


@veros_kernel
def integrate_tke_kernel(state):
    """
    integrate Tke equation on W grid with surface flux boundary condition
    """
    vs = state.variables
    settings = state.settings

    conditional_outputs = {}

    tke = vs.tke

    flux_east = allocate(state.dimensions, ("xt", "yt", "zt"))
    flux_north = allocate(state.dimensions, ("xt", "yt", "zt"))
    flux_top = allocate(state.dimensions, ("xt", "yt", "zt"))

    dt_tke = settings.dt_mom  # use momentum time step to prevent spurious oscillations

    """
    Sources and sinks by vertical friction, vertical mixing, and non-conservative advection
    """
    forc = vs.K_diss_v - vs.P_diss_v - vs.P_diss_adv

    """
    store transfer due to vertical mixing from dyn. enthalpy by non-linear eq.of
    state either to TKE or to heat
    """
    if not settings.enable_store_cabbeling_heat:
        forc = forc - vs.P_diss_nonlin

    """
    transfer part of dissipation of EKE to TKE
    """
    if settings.enable_eke:
        forc = forc + vs.eke_diss_tke

    if settings.enable_idemix:
        """
        transfer dissipation of internal waves to TKE
        """
        forc = forc + vs.iw_diss
        """
        store bottom friction either in TKE or internal waves
        """
        if settings.enable_store_bottom_friction_tke:
            forc = forc + vs.K_diss_bot

    else:  # short-cut without idemix
        if settings.enable_eke:
            forc = forc + vs.eke_diss_iw

        else:  # and without EKE model
            if settings.enable_store_cabbeling_heat:
                forc = forc + vs.K_diss_h - vs.P_diss_skew - vs.P_diss_hmix - vs.P_diss_iso
            else:
                forc = forc + vs.K_diss_h - vs.P_diss_skew

            if settings.enable_TEM_friction:
                forc = forc + vs.K_diss_gm

        forc = forc + vs.K_diss_bot

    """
    vertical mixing and dissipation of TKE
    """
    _, water_mask, edge_mask = utilities.create_water_masks(vs.kbot[2:-2, 2:-2], settings.nz)

    a_tri, b_tri, c_tri, d_tri, delta = (np.zeros_like(vs.kappaM[2:-2, 2:-2, :]) for _ in range(5))

    delta = update(delta, at[:, :, :-1], dt_tke / vs.dzt[np.newaxis, np.newaxis, 1:] * settings.alpha_tke * 0.5 \
        * (vs.kappaM[2:-2, 2:-2, :-1] + vs.kappaM[2:-2, 2:-2, 1:]))

    a_tri = update(a_tri, at[:, :, 1:-1], -delta[:, :, :-2] / vs.dzw[np.newaxis, np.newaxis, 1:-1])
    a_tri = update(a_tri, at[:, :, -1], -delta[:, :, -2] / (0.5 * vs.dzw[-1]))

    b_tri = update(b_tri, at[:, :, 1:-1], 1 + (delta[:, :, 1:-1] + delta[:, :, :-2]) / vs.dzw[np.newaxis, np.newaxis, 1:-1] \
        + dt_tke * settings.c_eps \
        * vs.sqrttke[2:-2, 2:-2, 1:-1] / vs.mxl[2:-2, 2:-2, 1:-1])
    b_tri = update(b_tri, at[:, :, -1], 1 + delta[:, :, -2] / (0.5 * vs.dzw[-1]) \
        + dt_tke * settings.c_eps / vs.mxl[2:-2, 2:-2, -1] * vs.sqrttke[2:-2, 2:-2, -1])
    b_tri_edge = 1 + delta / vs.dzw[np.newaxis, np.newaxis, :] \
        + dt_tke * settings.c_eps / vs.mxl[2:-2, 2:-2, :] * vs.sqrttke[2:-2, 2:-2, :]

    c_tri = update(c_tri, at[:, :, :-1], -delta[:, :, :-1] / vs.dzw[np.newaxis, np.newaxis, :-1])

    d_tri = update(d_tri, at[...], tke[2:-2, 2:-2, :, vs.tau] + dt_tke * forc[2:-2, 2:-2, :])
    d_tri = update_add(d_tri, at[:, :, -1], dt_tke * vs.forc_tke_surface[2:-2, 2:-2] / (0.5 * vs.dzw[-1]))

    sol = utilities.solve_implicit(a_tri, b_tri, c_tri, d_tri, water_mask, b_edge=b_tri_edge, edge_mask=edge_mask)
    tke = update(tke, at[2:-2, 2:-2, :, vs.taup1], np.where(water_mask, sol, tke[2:-2, 2:-2, :, vs.taup1]))

    """
    store tke dissipation for diagnostics
    """
    tke_diss = settings.c_eps / vs.mxl * vs.sqrttke * tke[:, :, :, vs.taup1]

    """
    Add TKE if surface density flux drains TKE in uppermost box
    """
    mask = tke[2:-2, 2:-2, -1, vs.taup1] < 0.0
    tke_surf_corr = np.zeros_like(vs.tke_surf_corr)
    tke_surf_corr = update(tke_surf_corr, at[2:-2, 2:-2], np.where(mask,
                                                -tke[2:-2, 2:-2, -1, vs.taup1] * 0.5
                                                * vs.dzw[-1] / dt_tke, 0.))
    tke = update(tke, at[2:-2, 2:-2, -1, vs.taup1], np.maximum(0., tke[2:-2, 2:-2, -1, vs.taup1]))

    if settings.enable_tke_hor_diffusion:
        """
        add tendency due to lateral diffusion
        """
        flux_east = update(flux_east, at[:-1, :, :], settings.K_h_tke * (tke[1:, :, :, vs.tau] - tke[:-1, :, :, vs.tau]) \
            / (vs.cost[np.newaxis, :, np.newaxis] * vs.dxu[:-1, np.newaxis, np.newaxis]) * vs.maskU[:-1, :, :])

        if runtime_settings.pyom_compatibility_mode:
            flux_east = update(flux_east, at[-5, :, :], 0.)
        else:
            flux_east = update(flux_east, at[-1, :, :], 0.)

        flux_north = update(flux_north, at[:, :-1, :], settings.K_h_tke * (tke[:, 1:, :, vs.tau] - tke[:, :-1, :, vs.tau]) \
            / vs.dyu[np.newaxis, :-1, np.newaxis] * vs.maskV[:, :-1, :] * vs.cosu[np.newaxis, :-1, np.newaxis])
        flux_north = update(flux_north, at[:, -1, :], 0.)

        tke = update_add(tke, at[2:-2, 2:-2, :, vs.taup1], dt_tke * vs.maskW[2:-2, 2:-2, :] * \
            ((flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
             / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxt[2:-2, np.newaxis, np.newaxis])
             + (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
             / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dyt[np.newaxis, 2:-2, np.newaxis])))

    """
    add tendency due to advection
    """
    if settings.enable_tke_superbee_advection:
        flux_east, flux_north, flux_top = advection.adv_flux_superbee_wgrid(state, tke[:, :, :, vs.tau])

    if settings.enable_tke_upwind_advection:
        flux_east, flux_north, flux_top = advection.adv_flux_upwind_wgrid(state, tke[:, :, :, vs.tau])

    if settings.enable_tke_superbee_advection or settings.enable_tke_upwind_advection:
        dtke = vs.dtke
        dtke = update(dtke, at[2:-2, 2:-2, :, vs.tau], vs.maskW[2:-2, 2:-2, :] * (-(flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
                                                           / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxt[2:-2, np.newaxis, np.newaxis])
                                                           - (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
                                                           / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dyt[np.newaxis, 2:-2, np.newaxis])))
        dtke = update_add(dtke, at[:, :, 0, vs.tau], -flux_top[:, :, 0] / vs.dzw[0])
        dtke = update_add(dtke, at[:, :, 1:-1, vs.tau], -(flux_top[:, :, 1:-1] - flux_top[:, :, :-2]) / vs.dzw[1:-1])
        dtke = update_add(dtke, at[:, :, -1, vs.tau], -(flux_top[:, :, -1] - flux_top[:, :, -2]) / (0.5 * vs.dzw[-1]))

        """
        Adam Bashforth time stepping
        """
        tke = update_add(tke, at[:, :, :, vs.taup1], settings.dt_tracer * ((1.5 + settings.AB_eps) * dtke[:, :, :, vs.tau]
                                            - (0.5 + settings.AB_eps) * dtke[:, :, :, vs.taum1]))

        conditional_outputs.update(dtke=dtke)

    return KernelOutput(tke=tke, tke_surf_corr=tke_surf_corr, tke_diss=tke_diss, **conditional_outputs)

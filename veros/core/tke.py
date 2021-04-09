from veros.core.operators import numpy as np

from veros import veros_kernel, veros_routine, run_kernel
from veros.core import advection, utilities
from veros.core.operators import update, update_add, at


@veros_routine
def set_tke_diffusivities(vs):
    if vs.enable_idemix:
        alpha_c = vs.alpha_c
        E_iw = vs.E_iw
    else:
        alpha_c = 0
        E_iw = 0

    if vs.enable_tke:
        (
            sqrttke, mxl, kappaM, kappaH, Prandtlnumber
        ) = run_kernel(set_tke_diffusivities_kernel, vs, alpha_c=alpha_c, E_iw=E_iw)
    else:
        kappaM = vs.kappaM_0 * np.ones_like(vs.kappaM)
        kappaH = vs.kappaH_0 * np.ones_like(vs.kappaH)
        sqrttke = mxl = Prandtlnumber = None
    return dict(
        sqrttke=sqrttke,
        mxl=mxl,
        kappaM=kappaM,
        kappaH=kappaH,
        Prandtlnumber=Prandtlnumber
    )


@veros_kernel
def set_tke_diffusivities_kernel(tke, mxl_min, Nsqr, ht, maskW, zw, dzt, dzw, tau,
                                 nz, K_diss_v, kappaM_max, kappaM_min, kappaH_min,
                                 kappaM_0, kappaH_0, c_k, alpha_c, E_iw, Prandtl_tke0,
                                 enable_cyclic_x, enable_Prandtl_tke, enable_kappaH_profile, enable_tke,
                                 tke_mxl_choice, enable_idemix,
                                 kappaM, kappaH, Prandtlnumber):
    """
    set vertical diffusivities based on TKE model
    """
    Rinumber = np.zeros_like(maskW)

    sqrttke = np.sqrt(np.maximum(0., tke[:, :, :, tau]))
    """
    calculate buoyancy length scale
    """
    mxl = np.sqrt(2) * sqrttke \
        / np.sqrt(np.maximum(1e-12, Nsqr[:, :, :, tau])) * maskW

    """
    apply limits for mixing length
    """
    if tke_mxl_choice == 1:
        """
        bounded by the distance to surface/bottom
        """
        mxl = np.minimum(
            np.minimum(mxl, -zw[np.newaxis, np.newaxis, :]
                        + dzw[np.newaxis, np.newaxis, :] * 0.5
                        ), ht[:, :, np.newaxis] + zw[np.newaxis, np.newaxis, :]
        )
        mxl = np.maximum(mxl, mxl_min)
    elif tke_mxl_choice == 2:
        """
        bound length scale as in mitgcm/OPA code

        Note that the following code doesn't vectorize. If critical for performance,
        consider re-implementing it in Cython.
        """
        # TODO: use scans
        for k in range(nz - 2, -1, -1):
            mxl = update(mxl, at[:, :, k], np.minimum(mxl[:, :, k], mxl[:, :, k + 1] + dzt[k + 1]))
        mxl = update(mxl, at[:, :, -1], np.minimum(mxl[:, :, -1], mxl_min + dzt[-1]))
        for k in range(1, nz):
            mxl = update(mxl, at[:, :, k], np.minimum(mxl[:, :, k], mxl[:, :, k - 1] + dzt[k]))
        mxl = update(mxl, at[...], np.maximum(mxl, mxl_min))
    else:
        raise ValueError('unknown mixing length choice in tke_mxl_choice')

    """
    calculate viscosity and diffusivity based on Prandtl number
    """
    K_diss_v = utilities.enforce_boundaries(K_diss_v, enable_cyclic_x)
    kappaM = update(kappaM, at[...], np.minimum(kappaM_max, c_k * mxl * sqrttke))
    Rinumber = update(Rinumber, at[...], Nsqr[:, :, :, tau] / \
        np.maximum(K_diss_v / np.maximum(1e-12, kappaM), 1e-12))
    if enable_idemix:
        Rinumber = update(Rinumber, at[...], np.minimum(Rinumber, kappaM * Nsqr[:, :, :, tau]
                                    / np.maximum(1e-12, alpha_c * E_iw[:, :, :, tau]**2)))
    if enable_Prandtl_tke:
        Prandtlnumber = np.maximum(1., np.minimum(10, 6.6 * Rinumber))
    else:
        Prandtlnumber = update(Prandtlnumber, at[...], Prandtl_tke0)
    kappaH = np.maximum(kappaH_min, kappaM / Prandtlnumber)
    if enable_kappaH_profile:
        # Correct diffusivity according to
        # Bryan, K., and L. J. Lewis, 1979:
        # A water mass model of the world ocean. J. Geophys. Res., 84, 2503–2517.
        # It mainly modifies kappaH within 20S - 20N deg. belt
        kappaH = np.maximum(kappaH, (0.8 + 1.05 / np.pi
                                            * np.arctan((-zw[np.newaxis, np.newaxis, :] - 2500.)
                                                        / 222.2)) * 1e-4)
    kappaM = np.maximum(kappaM_min, kappaM)

    return (sqrttke, mxl, kappaM, kappaH, Prandtlnumber)


@veros_routine
def integrate_tke(vs):
    vs = integrate_tke_kernel.run_with_state(vs, iw_diss=None)
    return dict(tke=tke, dtke=dtke, tke_surf_corr=tke_surf_corr)


@veros_kernel
def integrate_tke_kernel(K_diss_v, P_diss_v, P_diss_adv, P_diss_nonlin, eke_diss_tke, iw_diss,
                         eke_diss_iw, K_diss_bot, K_diss_gm, K_diss_h, P_diss_skew, P_diss_hmix,
                         P_diss_iso, tke, dtke, sqrttke, mxl, kbot, kappaM, dt_mom, alpha_tke,
                         c_eps, AB_eps, dxt, dxu, dyt, dyu, dzt, dzw, tau, taup1, taum1,
                         dt_tracer, maskU, maskV, maskW, forc_tke_surface, tke_diss,
                         tke_surf_corr, K_h_tke, cost, cosu, u_wgrid, v_wgrid, w_wgrid,
                         enable_tke_hor_diffusion, enable_eke, enable_idemix,
                         enable_store_cabbeling_heat, enable_tke_superbee_advection,
                         enable_tke_upwind_advection, enable_store_bottom_friction_tke,
                         pyom_compatibility_mode):
    """
    integrate Tke equation on W grid with surface flux boundary condition
    """
    nz = dzw.shape[0]

    flux_east = np.zeros_like(maskU)
    flux_north = np.zeros_like(maskV)
    flux_top = np.zeros_like(maskW)

    dt_tke = dt_mom  # use momentum time step to prevent spurious oscillations

    """
    Sources and sinks by vertical friction, vertical mixing, and non-conservative advection
    """
    forc = K_diss_v - P_diss_v - P_diss_adv

    """
    store transfer due to vertical mixing from dyn. enthalpy by non-linear eq.of
    state either to TKE or to heat
    """
    if not enable_store_cabbeling_heat:
        forc = update_add(forc, at[...], -P_diss_nonlin)

    """
    transfer part of dissipation of EKE to TKE
    """
    if enable_eke:
        forc = update_add(forc, at[...], eke_diss_tke)

    if enable_idemix:
        """
        transfer dissipation of internal waves to TKE
        """
        forc = update_add(forc, at[...], iw_diss)
        """
        store bottom friction either in TKE or internal waves
        """
        if enable_store_bottom_friction_tke:
            forc = update_add(forc, at[...], K_diss_bot)
    else:  # short-cut without idemix
        if enable_eke:
            forc = update_add(forc, at[...], eke_diss_iw)
        else:  # and without EKE model
            if enable_store_cabbeling_heat:
                forc = update_add(forc, at[...], K_diss_gm + K_diss_h - P_diss_skew \
                    - P_diss_hmix - P_diss_iso)
            else:
                forc = update_add(forc, at[...], K_diss_gm + K_diss_h - P_diss_skew)
        forc = update_add(forc, at[...], K_diss_bot)

    """
    vertical mixing and dissipation of TKE
    """
    _, water_mask, edge_mask = utilities.create_water_masks(kbot[2:-2, 2:-2], nz)

    a_tri, b_tri, c_tri, d_tri, delta = (np.zeros_like(kappaM[2:-2, 2:-2, :]) for _ in range(5))

    delta = update(delta, at[:, :, :-1], dt_tke / dzt[np.newaxis, np.newaxis, 1:] * alpha_tke * 0.5 \
        * (kappaM[2:-2, 2:-2, :-1] + kappaM[2:-2, 2:-2, 1:]))

    a_tri = update(a_tri, at[:, :, 1:-1], -delta[:, :, :-2] / dzw[np.newaxis, np.newaxis, 1:-1])
    a_tri = update(a_tri, at[:, :, -1], -delta[:, :, -2] / (0.5 * dzw[-1]))

    b_tri = update(b_tri, at[:, :, 1:-1], 1 + (delta[:, :, 1:-1] + delta[:, :, :-2]) / dzw[np.newaxis, np.newaxis, 1:-1] \
        + dt_tke * c_eps \
        * sqrttke[2:-2, 2:-2, 1:-1] / mxl[2:-2, 2:-2, 1:-1])
    b_tri = update(b_tri, at[:, :, -1], 1 + delta[:, :, -2] / (0.5 * dzw[-1]) \
        + dt_tke * c_eps / mxl[2:-2, 2:-2, -1] * sqrttke[2:-2, 2:-2, -1])
    b_tri_edge = 1 + delta / dzw[np.newaxis, np.newaxis, :] \
        + dt_tke * c_eps / mxl[2:-2, 2:-2, :] * sqrttke[2:-2, 2:-2, :]

    c_tri = update(c_tri, at[:, :, :-1], -delta[:, :, :-1] / dzw[np.newaxis, np.newaxis, :-1])

    d_tri = update(d_tri, at[...], tke[2:-2, 2:-2, :, tau] + dt_tke * forc[2:-2, 2:-2, :])
    d_tri = update_add(d_tri, at[:, :, -1], dt_tke * forc_tke_surface[2:-2, 2:-2] / (0.5 * dzw[-1]))

    sol = utilities.solve_implicit(a_tri, b_tri, c_tri, d_tri, water_mask, b_edge=b_tri_edge, edge_mask=edge_mask)
    tke = update(tke, at[2:-2, 2:-2, :, taup1], np.where(water_mask, sol, tke[2:-2, 2:-2, :, taup1]))

    """
    store tke dissipation for diagnostics
    """
    tke_diss = update(tke_diss, at[...], c_eps / mxl * sqrttke * tke[:, :, :, taup1])

    """
    Add TKE if surface density flux drains TKE in uppermost box
    """
    mask = tke[2:-2, 2:-2, -1, taup1] < 0.0
    tke_surf_corr = update(tke_surf_corr, at[...], 0.)
    tke_surf_corr = update(tke_surf_corr, at[2:-2, 2:-2], np.where(mask,
                                                -tke[2:-2, 2:-2, -1, taup1] * 0.5
                                                * dzw[-1] / dt_tke, 0.))
    tke = update(tke, at[2:-2, 2:-2, -1, taup1], np.maximum(0., tke[2:-2, 2:-2, -1, taup1]))

    if enable_tke_hor_diffusion:
        """
        add tendency due to lateral diffusion
        """
        flux_east = update(flux_east, at[:-1, :, :], K_h_tke * (tke[1:, :, :, tau] - tke[:-1, :, :, tau]) \
            / (cost[np.newaxis, :, np.newaxis] * dxu[:-1, np.newaxis, np.newaxis]) * maskU[:-1, :, :])
        if pyom_compatibility_mode:
            flux_east = update(flux_east, at[-5, :, :], 0.)
        else:
            flux_east = update(flux_east, at[-1, :, :], 0.)
        flux_north = update(flux_north, at[:, :-1, :], K_h_tke * (tke[:, 1:, :, tau] - tke[:, :-1, :, tau]) \
            / dyu[np.newaxis, :-1, np.newaxis] * maskV[:, :-1, :] * cosu[np.newaxis, :-1, np.newaxis])
        flux_north = update(flux_north, at[:, -1, :], 0.)
        tke = update_add(tke, at[2:-2, 2:-2, :, taup1], dt_tke * maskW[2:-2, 2:-2, :] * \
            ((flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
             / (cost[np.newaxis, 2:-2, np.newaxis] * dxt[2:-2, np.newaxis, np.newaxis])
             + (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
             / (cost[np.newaxis, 2:-2, np.newaxis] * dyt[np.newaxis, 2:-2, np.newaxis])))

    """
    add tendency due to advection
    """
    if enable_tke_superbee_advection:
        flux_east, flux_north, flux_top = advection.adv_flux_superbee_wgrid(
            flux_east, flux_north, flux_top, u_wgrid, v_wgrid, w_wgrid,
            tke[:, :, :, tau], maskW, dxt, dyt, dzw, cost, cosu, dt_tracer
        )
    if enable_tke_upwind_advection:
        flux_east, flux_north, flux_top = advection.adv_flux_upwind_wgrid(
            flux_east, flux_north, flux_top, u_wgrid, v_wgrid, w_wgrid,
            tke[:, :, :, tau], maskW, cosu
        )
    if enable_tke_superbee_advection or enable_tke_upwind_advection:
        dtke = update(dtke, at[2:-2, 2:-2, :, tau], maskW[2:-2, 2:-2, :] * (-(flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
                                                           / (cost[np.newaxis, 2:-2, np.newaxis] * dxt[2:-2, np.newaxis, np.newaxis])
                                                           - (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
                                                           / (cost[np.newaxis, 2:-2, np.newaxis] * dyt[np.newaxis, 2:-2, np.newaxis])))
        dtke = update_add(dtke, at[:, :, 0, tau], -flux_top[:, :, 0] / dzw[0])
        dtke = update_add(dtke, at[:, :, 1:-1, tau], -(flux_top[:, :, 1:-1] - flux_top[:, :, :-2]) / dzw[1:-1])
        dtke = update_add(dtke, at[:, :, -1, tau], -(flux_top[:, :, -1] - flux_top[:, :, -2]) / (0.5 * dzw[-1]))
        """
        Adam Bashforth time stepping
        """
        tke = update_add(tke, at[:, :, :, taup1], dt_tracer * ((1.5 + AB_eps) * dtke[:, :, :, tau]
                                            - (0.5 + AB_eps) * dtke[:, :, :, taum1]))

    return tke, dtke, tke_surf_corr

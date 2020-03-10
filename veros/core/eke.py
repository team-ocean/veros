import math

import numpy as np

from veros import veros_kernel, veros_routine, run_kernel
from veros.core import utilities, advection


@veros_routine(
    inputs=(),
    outputs=('hrms_k0'),
    settings=(
        'enable_eke_leewave_dissipation', 'eke_hrms_k0_min',
        'pi', 'eke_topo_hrms', 'eke_topo_lam'
    )
)
def init_eke(vs):
    """
    Initialize EKE
    """
    if vs.enable_eke_leewave_dissipation:
        hrms_k0 = np.maximum(vs.eke_hrms_k0_min, 2 / vs.pi * vs.eke_topo_hrms**2
                             / np.maximum(1e-12, vs.eke_topo_lam)**1.5)
    else:
        hrms_k0 = 0
    return dict(hrms_k0=hrms_k0)


@veros_routine(
    inputs=('eke', 'Nsqr', 'coriolis_t', 'beta', 'dzw', 'maskW', 'tau', 'pi', 'K_gm', 'K_iso',
            'K_gm_0', 'K_iso_0', 'L_rossby', 'L_rhines', 'kappa_gm', 'sqrteke',),
    outputs=('L_rossby', 'L_rhines', 'eke_len', 'sqrteke', 'K_gm', 'K_iso', 'kappa_gm'),
    settings=('enable_eke', 'enable_eke_isopycnal_diffusion', 'enable_TEM_friction',
              'eke_len', 'eke_lmin', 'eke_cross', 'eke_crhin', 'eke_k_max', 'eke_c_k')
)
def set_eke_diffusivities(vs):
    L_rossby, L_rhines, eke_len, sqrteke, K_gm, K_iso = run_kernel(set_eke_diffusivities_kernel, vs)

    if vs.enable_TEM_friction:
        kappa_gm = run_kernel(update_kappa_gm, vs)
    else:
        kappa_gm = None

    return dict(
        L_rossby=L_rossby,
        L_rhines=L_rhines,
        eke_len=eke_len,
        sqrteke=sqrteke,
        K_gm=K_gm,
        K_iso=K_iso,
        kappa_gm=kappa_gm
    )


@veros_kernel
def update_kappa_gm(K_gm, coriolis_t, Nsqr, maskW, tau):
    return K_gm * np.minimum(0.01, coriolis_t[..., np.newaxis]**2 / np.maximum(1e-9, Nsqr[..., tau])) * maskW


@veros_kernel(static_args=('enable_eke', 'enable_eke_isopycnal_diffusion', 'enable_TEM_friction',))
def set_eke_diffusivities_kernel(eke, Nsqr, coriolis_t, beta, dzw, maskW, tau, pi, K_gm, K_iso,
                                 K_gm_0, K_iso_0, L_rossby, L_rhines, sqrteke,
                                 eke_len, eke_lmin, eke_cross, eke_crhin, eke_k_max, eke_c_k,
                                 enable_eke, enable_eke_isopycnal_diffusion, enable_TEM_friction):
    """
    set skew diffusivity K_gm and isopycnal diffusivity K_iso
    set also vertical viscosity if TEM formalism is chosen
    """
    C_rossby = np.zeros_like(coriolis_t)

    if enable_eke:
        """
        calculate Rossby radius as minimum of mid-latitude and equatorial R. rad.
        """
        C_rossby[...] = np.sum(np.sqrt(np.maximum(0., Nsqr[:, :, :, tau]))
                               * dzw[np.newaxis, np.newaxis, :] * maskW[:, :, :] / pi, axis=2)
        L_rossby[...] = np.minimum(C_rossby / np.maximum(np.abs(coriolis_t), 1e-16),
                                   np.sqrt(C_rossby / np.maximum(2 * beta, 1e-16)))
        """
        calculate vertical viscosity and skew diffusivity
        """
        sqrteke = np.sqrt(np.maximum(0., eke[:, :, :, tau]))
        L_rhines[...] = np.sqrt(
            sqrteke / np.maximum(beta[..., np.newaxis], 1e-16))
        eke_len[...] = np.maximum(eke_lmin, np.minimum(
            eke_cross * L_rossby[..., np.newaxis], eke_crhin * L_rhines))
        K_gm[...] = np.minimum(eke_k_max, eke_c_k * eke_len * sqrteke)
    else:
        """
        use fixed GM diffusivity
        """
        K_gm[...] = K_gm_0

    if enable_eke and enable_eke_isopycnal_diffusion:
        K_iso[...] = K_gm
    else:
        K_iso[...] = K_iso_0  # always constant

    return L_rossby, L_rhines, eke_len, sqrteke, K_gm, K_iso


@veros_routine(
    inputs=(
        'eke', 'deke', 'sqrteke', 'K_diss_h', 'K_diss_gm', 'K_gm', 'kappaM', 'P_diss_skew',
        'P_diss_hmix', 'P_diss_iso', 'hrms_k0', 'Nsqr', 'kbot', 'coriolis_t', 'AB_eps', 'nz',
        'tau', 'taup1', 'taum1', 'dt_tracer', 'u', 'v', 'u_wgrid', 'v_wgrid', 'w_wgrid', 'c_lee',
        'c_lee0', 'cost', 'cosu', 'dxt', 'dxu', 'dyt', 'dyu', 'dzt', 'dzw', 'maskU', 'maskV', 'maskW',
        'eke_Ri0', 'eke_Ri1', 'c_Ri_diss', 'eke_int_diss0', 'eke_diss_iw', 'eke_r_bot',
        'eke_c_eps', 'eke_len', 'alpha_eke', 'eke_diss_tke', 'eke_lee_flux', 'eke_bot_flux',
    ),
    outputs=(
        'eke', 'deke', 'c_lee', 'c_Ri_diss', 'eke_diss_iw', 'eke_diss_tke', 'eke_lee_flux', 'eke_bot_flux'
    ),
    settings=(
        'enable_store_cabbeling_heat', 'enable_eke_leewave_dissipation',
        'enable_eke_superbee_advection', 'enable_eke_upwind_advection',
        'pyom_compatibility_mode'
    )
)
def integrate_eke(vs):
    (eke, deke, c_lee, c_Ri_diss, eke_diss_iw, eke_diss_tke, eke_lee_flux, eke_bot_flux) = run_kernel(integrate_eke_kernel, vs, hrms_k0=None, c_lee=None, c_Ri_diss=None, eke_lee_flux=None)
    return dict(
        eke=eke, deke=deke, c_lee=c_lee, c_Ri_diss=c_Ri_diss, eke_diss_iw=eke_diss_iw,
        eke_diss_tke=eke_diss_tke, eke_lee_flux=eke_lee_flux, eke_bot_flux=eke_bot_flux
    )


@veros_kernel(static_args=('enable_store_cabbeling_heat',
                           'enable_eke_leewave_dissipation',
                           'pyom_compatibility_mode',
                           'enable_eke_superbee_advection',
                           'enable_eke_upwind_advection'))
def integrate_eke_kernel(eke, deke, sqrteke, K_diss_h, K_diss_gm, K_gm, kappaM, P_diss_skew,
                  P_diss_hmix, P_diss_iso, hrms_k0, Nsqr, kbot, coriolis_t, AB_eps, nz,
                  tau, taup1, taum1, dt_tracer, u, v, u_wgrid, v_wgrid, w_wgrid, c_lee,
                  c_lee0, cost, cosu, dxt, dxu, dyt, dyu, dzt, dzw, maskU, maskV, maskW,
                  eke_Ri0, eke_Ri1, c_Ri_diss, eke_int_diss0, eke_diss_iw, eke_r_bot,
                  eke_c_eps, eke_len, alpha_eke, eke_diss_tke, eke_lee_flux, eke_bot_flux,
                  enable_store_cabbeling_heat, enable_eke_leewave_dissipation,
                  enable_eke_superbee_advection, enable_eke_upwind_advection,
                  pyom_compatibility_mode):
    """
    integrate EKE equation on W grid
    """
    c_int = np.zeros_like(maskW)
    flux_east = np.zeros_like(maskU)
    flux_north = np.zeros_like(maskV)
    flux_top = np.zeros_like(maskW)
    """
    forcing by dissipation by lateral friction and GM using TRM formalism or skew diffusion
    """
    forc = K_diss_h + K_diss_gm - P_diss_skew

    """
    store transfer due to isopycnal and horizontal mixing from dyn. enthalpy
    by non-linear eq.of state either to EKE or to heat
    """
    if not enable_store_cabbeling_heat:
        forc[...] += -P_diss_hmix - P_diss_iso

    """
    coefficient for dissipation of EKE:
    by lee wave generation, Ri-dependent interior loss of balance and bottom friction
    """
    if enable_eke_leewave_dissipation:
        """
        by lee wave generation
        """
        c_lee[...] = 0.
        ks = kbot[2:-2, 2:-2] - 1
        ki = np.arange(nz)[np.newaxis, np.newaxis, :]
        boundary_mask = (ks >= 0) & (ks < nz - 1)
        full_mask = boundary_mask[:, :, np.newaxis] & (ki == ks[:, :, np.newaxis])
        fxa = np.maximum(0, Nsqr[2:-2, 2:-2, :, tau])**0.25
        fxa *= 1.5 * fxa / np.sqrt(np.maximum(1e-6, np.abs(coriolis_t[2:-2, 2:-2, np.newaxis]))) - 2
        c_lee[2:-2, 2:-2] = boundary_mask * c_lee0 * hrms_k0[2:-2, 2:-2] \
                            * np.sum(np.sqrt(sqrteke[2:-2, 2:-2, :]) * np.maximum(0, fxa)
                            / dzw[np.newaxis, np.newaxis, :] * full_mask, axis=-1)

        """
        Ri-dependent dissipation by interior loss of balance
        """
        c_Ri_diss[...] = 0
        uz = (((u[1:, 1:, 1:, tau] - u[1:, 1:, :-1, tau]) / dzt[np.newaxis, np.newaxis, :-1] * maskU[1:, 1:, :-1])**2
              + ((u[:-1, 1:, 1:, tau] - u[:-1, 1:, :-1, tau]) / dzt[np.newaxis, np.newaxis, :-1] * maskU[:-1, 1:, :-1])**2) \
            / (maskU[1:, 1:, :-1] + maskU[:-1, 1:, :-1] + 1e-18)
        vz = (((v[1:, 1:, 1:, tau] - v[1:, 1:, :-1, tau]) / dzt[np.newaxis, np.newaxis, :-1] * maskV[1:, 1:, :-1])**2
              + ((v[1:, :-1, 1:, tau] - v[1:, :-1, :-1, tau]) / dzt[np.newaxis, np.newaxis, :-1] * maskV[1:, :-1, :-1])**2) \
            / (maskV[1:, 1:, :-1] + maskV[1:, :-1, :-1] + 1e-18)
        Ri = np.maximum(1e-8, Nsqr[1:, 1:, :-1, tau]) / (uz + vz + 1e-18)
        fxa = 1 - 0.5 * (1. + np.tanh((Ri - eke_Ri0) / eke_Ri1))
        c_Ri_diss[1:, 1:, :-1] = maskW[1:, 1:, :-1] * fxa * eke_int_diss0
        c_Ri_diss[:, :, -1] = c_Ri_diss[:, :, -2] * maskW[:, :, -1]

        """
        vertically integrate Ri-dependent dissipation and EKE
        """
        a_loc = np.sum(c_Ri_diss[:, :, :-1] * eke[:, :, :-1, tau] * maskW[:, :, :-1] * dzw[:-1], axis=2)
        b_loc = np.sum(eke[:, :, :-1, tau] *
                       maskW[:, :, :-1] * dzw[:-1], axis=2)
        a_loc += c_Ri_diss[:, :, -1] * eke[:, :, -1, tau] * maskW[:, :, -1] * dzw[-1] * 0.5
        b_loc += eke[:, :, -1, tau] * maskW[:, :, -1] * dzw[-1] * 0.5

        """
        add bottom fluxes by lee waves and bottom friction to a_loc
        """
        a_loc[2:-2, 2:-2] += np.sum((c_lee[2:-2, 2:-2, np.newaxis] * eke[2:-2, 2:-2, :, tau]
                                     * maskW[2:-2, 2:-2, :] * dzw[np.newaxis, np.newaxis, :]
                                     + 2 * eke_r_bot * eke[2:-2, 2:-2, :, tau]
                                     * math.sqrt(2.0) * sqrteke[2:-2, 2:-2, :]
                                     * maskW[2:-2, 2:-2, :]) * full_mask, axis=-1) * boundary_mask

        """
        dissipation constant is vertically integrated forcing divided by
        vertically integrated EKE to account for vertical EKE radiation
        """
        mask = b_loc > 0
        a_loc[...] = utilities.where(mask, a_loc / (b_loc + 1e-20), 0.)
        c_int[...] = a_loc[:, :, np.newaxis]
    else:
        """
        dissipation by local interior loss of balance with constant coefficient
        """
        c_int[...] = eke_c_eps * sqrteke / eke_len * maskW

    """
    vertical diffusion of EKE,forcing and dissipation
    """
    ks = kbot[2:-2, 2:-2] - 1
    delta, a_tri, b_tri, c_tri, d_tri = (np.zeros_like(kappaM[2:-2, 2:-2, :]) for _ in range(5))
    delta[:, :, :-1] = dt_tracer / dzt[np.newaxis, np.newaxis, 1:] * 0.5 \
        * (kappaM[2:-2, 2:-2, :-1] + kappaM[2:-2, 2:-2, 1:]) * alpha_eke
    a_tri[:, :, 1:-1] = -delta[:, :, :-2] / dzw[1:-1]
    a_tri[:, :, -1] = -delta[:, :, -2] / (0.5 * dzw[-1])
    b_tri[:, :, 1:-1] = 1 + (delta[:, :, 1:-1] + delta[:, :, :-2]) / \
        dzw[1:-1] + dt_tracer * c_int[2:-2, 2:-2, 1:-1]
    b_tri[:, :, -1] = 1 + delta[:, :, -2] / \
        (0.5 * dzw[-1]) + dt_tracer * c_int[2:-2, 2:-2, -1]
    b_tri_edge = 1 + delta / dzw[np.newaxis, np.newaxis, :] \
        + dt_tracer * c_int[2:-2, 2:-2, :]
    c_tri[:, :, :-1] = -delta[:, :, :-1] / dzw[np.newaxis, np.newaxis, :-1]
    d_tri[:, :, :] = eke[2:-2, 2:-2, :, tau] + dt_tracer * forc[2:-2, 2:-2, :]
    sol, water_mask = utilities.solve_implicit(ks, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)
    eke[2:-2, 2:-2, :, taup1] = utilities.where(water_mask, sol, eke[2:-2, 2:-2, :, taup1])

    """
    store eke dissipation
    """
    if enable_eke_leewave_dissipation:
        eke_diss_iw[...] = 0.
        eke_diss_tke[...] = c_Ri_diss * eke[:, :, :, taup1]

        """
        flux by lee wave generation and bottom friction
        """
        eke_diss_iw[2:-2, 2:-2, :] += (c_lee[2:-2, 2:-2, np.newaxis] * eke[2:-2, 2:-2, :, taup1]
                                       * maskW[2:-2, 2:-2, :]) * full_mask
        if pyom_compatibility_mode:
            eke_diss_tke[2:-2, 2:-2, :] += (2 * eke_r_bot * eke[2:-2, 2:-2, :, taup1] * np.sqrt(np.float32(2.0))
                                            * sqrteke[2:-2, 2:-2, :] * maskW[2:-2, 2:-2, :] / dzw[np.newaxis, np.newaxis, :]) * full_mask
        else:
            eke_diss_tke[2:-2, 2:-2, :] += (2 * eke_r_bot * eke[2:-2, 2:-2, :, taup1] * math.sqrt(2.0)
                                            * sqrteke[2:-2, 2:-2, :] * maskW[2:-2, 2:-2, :] / dzw[np.newaxis, np.newaxis, :]) * full_mask
        """
        account for sligthly incorrect integral of dissipation due to time stepping
        """
        a_loc = np.sum((eke_diss_iw[:, :, :-1] + eke_diss_tke[:, :, :-1])
                       * dzw[np.newaxis, np.newaxis, :-1], axis=2)
        b_loc = np.sum(c_int[:, :, :-1] * eke[:, :, :-1, taup1]
                       * dzw[np.newaxis, np.newaxis, :-1], axis=2)
        a_loc += (eke_diss_iw[:, :, -1] + eke_diss_tke[:, :, -1]) * dzw[-1] * 0.5
        b_loc += c_int[:, :, -1] * eke[:, :, -1, taup1] * dzw[-1] * 0.5
        mask = a_loc != 0.
        b_loc[...] = utilities.where(mask, b_loc / (a_loc + 1e-20), 0.)
        eke_diss_iw[...] *= b_loc[:, :, np.newaxis]
        eke_diss_tke[...] *= b_loc[:, :, np.newaxis]

        """
        store diagnosed flux by lee waves and bottom friction
        """
        eke_lee_flux[2:-2, 2:-2] = utilities.where(boundary_mask, np.sum(c_lee[2:-2, 2:-2, np.newaxis] * eke[2:-2, 2:-2, :, taup1]
                                                   * dzw[np.newaxis, np.newaxis, :] * full_mask, axis=-1), eke_lee_flux[2:-2, 2:-2])
        eke_bot_flux[2:-2, 2:-2] = utilities.where(boundary_mask, np.sum(2 * eke_r_bot * eke[2:-2, 2:-2, :, taup1]
                                                   * math.sqrt(2.0) * sqrteke[2:-2, 2:-2, :] * full_mask, axis=-1), eke_bot_flux[2:-2, 2:-2])
    else:
        eke_diss_iw = c_int * eke[:, :, :, taup1]
        eke_diss_tke[...] = 0.

    """
    add tendency due to lateral diffusion
    """
    flux_east[:-1, :, :] = 0.5 * np.maximum(500., K_gm[:-1, :, :] + K_gm[1:, :, :]) \
        * (eke[1:, :, :, tau] - eke[:-1, :, :, tau]) \
        / (cost[np.newaxis, :, np.newaxis] * dxu[:-1, np.newaxis, np.newaxis]) * maskU[:-1, :, :]
    flux_east[-1, :, :] = 0.
    flux_north[:, :-1, :] = 0.5 * np.maximum(500., K_gm[:, :-1, :] + K_gm[:, 1:, :]) \
        * (eke[:, 1:, :, tau] - eke[:, :-1, :, tau]) \
        / dyu[np.newaxis, :-1, np.newaxis] * maskV[:, :-1, :] * cosu[np.newaxis, :-1, np.newaxis]
    flux_north[:, -1, :] = 0.
    eke[2:-2, 2:-2, :, taup1] += dt_tracer * maskW[2:-2, 2:-2, :] \
        * ((flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
           / (cost[np.newaxis, 2:-2, np.newaxis] * dxt[2:-2, np.newaxis, np.newaxis])
           + (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
           / (cost[np.newaxis, 2:-2, np.newaxis] * dyt[np.newaxis, 2:-2, np.newaxis]))

    """
    add tendency due to advection
    """
    if enable_eke_superbee_advection:
        flux_east, flux_north, flux_top = advection.adv_flux_superbee_wgrid(
            flux_east, flux_north, flux_top, u_wgrid, v_wgrid, w_wgrid,
            eke[:, :, :, tau], maskW, dxt, dyt, dzw, cost, cosu, dt_tracer
            )
    if enable_eke_upwind_advection:
        flux_east, flux_north, flux_top = advection.adv_flux_upwind_wgrid(
            flux_east, flux_north, flux_top, u_wgrid, v_wgrid, w_wgrid,
            eke[:, :, :, tau], maskW, cosu
            )
    if enable_eke_superbee_advection or enable_eke_upwind_advection:
        deke[2:-2, 2:-2, :, tau] = maskW[2:-2, 2:-2, :] * (-(flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
                                                           / (cost[np.newaxis, 2:-2, np.newaxis] * dxt[2:-2, np.newaxis, np.newaxis])
                                                           - (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
                                                           / (cost[np.newaxis, 2:-2, np.newaxis] * dyt[np.newaxis, 2:-2, np.newaxis]))
        deke[:, :, 0, tau] += -flux_top[:, :, 0] / dzw[0]
        deke[:, :, 1:-1, tau] += -(flux_top[:, :, 1:-1] -
                                   flux_top[:, :, :-2]) / dzw[np.newaxis, np.newaxis, 1:-1]
        deke[:, :, -1, tau] += -(flux_top[:, :, -1] - flux_top[:, :, -2]) / (0.5 * dzw[-1])
        """
        Adam Bashforth time stepping
        """
        eke[:, :, :, taup1] += dt_tracer * ((1.5 + AB_eps) * deke[:, :, :, tau]
                                            - (0.5 + AB_eps) * deke[:, :, :, taum1])

    return eke, deke, c_lee, c_Ri_diss, eke_diss_iw, eke_diss_tke, eke_lee_flux, eke_bot_flux

import math

from .. import veros_method
from . import utilities, advection


@veros_method
def init_eke(veros):
    """
    Initialize EKE
    """
    if veros.enable_eke_leewave_dissipation:
        veros.hrms_k0[...] = np.maximum(veros.eke_hrms_k0_min, 2 / veros.pi * veros.eke_topo_hrms**2
                                        / np.maximum(1e-12, veros.eke_topo_lam)**1.5)


@veros_method
def set_eke_diffusivities(veros):
    """
    set skew diffusivity K_gm and isopycnal diffusivity K_iso
    set also vertical viscosity if TEM formalism is chosen
    """
    C_rossby = np.zeros((veros.nx + 4, veros.ny + 4))

    if veros.enable_eke:
        """
        calculate Rossby radius as minimum of mid-latitude and equatorial R. rad.
        """
        C_rossby[...] = np.sum(np.sqrt(np.maximum(0., veros.Nsqr[:, :, :, veros.tau]))
                               * veros.dzw[np.newaxis, np.newaxis, :] * veros.maskW[:, :, :] / veros.pi, axis=2)
        veros.L_rossby[...] = np.minimum(C_rossby / np.maximum(np.abs(veros.coriolis_t), 1e-16),
                                         np.sqrt(C_rossby / np.maximum(2 * veros.beta, 1e-16)))
        """
        calculate vertical viscosity and skew diffusivity
        """
        veros.sqrteke = np.sqrt(np.maximum(0., veros.eke[:, :, :, veros.tau]))
        veros.L_rhines[...] = np.sqrt(
            veros.sqrteke / np.maximum(veros.beta[..., np.newaxis], 1e-16))
        veros.eke_len[...] = np.maximum(veros.eke_lmin, np.minimum(
            veros.eke_cross * veros.L_rossby[..., np.newaxis], veros.eke_crhin * veros.L_rhines))
        veros.K_gm[...] = np.minimum(veros.eke_k_max, veros.eke_c_k * veros.eke_len * veros.sqrteke)
    else:
        """
        use fixed GM diffusivity
        """
        veros.K_gm[...] = veros.K_gm_0

    if veros.enable_TEM_friction:
        veros.kappa_gm[...] = veros.K_gm * np.minimum(0.01, veros.coriolis_t[..., np.newaxis]**2
                                                      / np.maximum(1e-9, veros.Nsqr[..., veros.tau])) * veros.maskW
    if veros.enable_eke and veros.enable_eke_isopycnal_diffusion:
        veros.K_iso[...] = veros.K_gm
    else:
        veros.K_iso[...] = veros.K_iso_0  # always constant


@veros_method
def integrate_eke(veros):
    """
    integrate EKE equation on W grid
    """
    c_int = np.zeros((veros.nx + 4, veros.ny + 4, veros.nz))

    """
    forcing by dissipation by lateral friction and GM using TRM formalism or skew diffusion
    """
    forc = veros.K_diss_h + veros.K_diss_gm - veros.P_diss_skew

    """
    store transfer due to isopycnal and horizontal mixing from dyn. enthalpy
    by non-linear eq.of state either to EKE or to heat
    """
    if not veros.enable_store_cabbeling_heat:
        forc[...] += -veros.P_diss_hmix - veros.P_diss_iso

    """
    coefficient for dissipation of EKE:
    by lee wave generation, Ri-dependent interior loss of balance and bottom friction
    """
    if veros.enable_eke_leewave_dissipation:
        """
        by lee wave generation
        """
        veros.c_lee[...] = 0.
        ks = veros.kbot[2:-2, 2:-2] - 1
        ki = np.arange(veros.nz)[np.newaxis, np.newaxis, :]
        boundary_mask = (ks >= 0) & (ks < veros.nz - 1)
        full_mask = boundary_mask[:, :, np.newaxis] & (ki == ks[:, :, np.newaxis])
        fxa = np.maximum(0, veros.Nsqr[2:-2, 2:-2, :, veros.tau])**0.25
        fxa *= 1.5 * fxa / \
            np.sqrt(np.maximum(1e-6, np.abs(veros.coriolis_t[2:-2, 2:-2, np.newaxis]))) - 2
        veros.c_lee[2:-2, 2:-2] += boundary_mask * np.sum((veros.c_lee0 * veros.hrms_k0[2:-2, 2:-2, np.newaxis] * np.sqrt(veros.sqrteke[2:-2, 2:-2, :])
                                                           * np.maximum(0, fxa) / veros.dzw[np.newaxis, np.newaxis, :]) * full_mask, axis=-1)

        """
        Ri-dependent dissipation by interior loss of balance
        """
        veros.c_Ri_diss[...] = 0
        uz = (((veros.u[1:, 1:, 1:, veros.tau] - veros.u[1:, 1:, :-1, veros.tau]) / veros.dzt[np.newaxis, np.newaxis, :-1] * veros.maskU[1:, 1:, :-1])**2
              + ((veros.u[:-1, 1:, 1:, veros.tau] - veros.u[:-1, 1:, :-1, veros.tau]) / veros.dzt[np.newaxis, np.newaxis, :-1] * veros.maskU[:-1, 1:, :-1])**2) \
            / (veros.maskU[1:, 1:, :-1] + veros.maskU[:-1, 1:, :-1] + 1e-18)
        vz = (((veros.v[1:, 1:, 1:, veros.tau] - veros.v[1:, 1:, :-1, veros.tau]) / veros.dzt[np.newaxis, np.newaxis, :-1] * veros.maskV[1:, 1:, :-1])**2
              + ((veros.v[1:, :-1, 1:, veros.tau] - veros.v[1:, :-1, :-1, veros.tau]) / veros.dzt[np.newaxis, np.newaxis, :-1] * veros.maskV[1:, :-1, :-1])**2) \
            / (veros.maskV[1:, 1:, :-1] + veros.maskV[1:, :-1, :-1] + 1e-18)
        Ri = np.maximum(1e-8, veros.Nsqr[1:, 1:, :-1, veros.tau]) / (uz + vz + 1e-18)
        fxa = 1 - 0.5 * (1. + np.tanh((Ri - veros.eke_Ri0) / veros.eke_Ri1))
        veros.c_Ri_diss[1:, 1:, :-1] = veros.maskW[1:, 1:, :-1] * fxa * veros.eke_int_diss0
        veros.c_Ri_diss[:, :, -1] = veros.c_Ri_diss[:, :, -2] * veros.maskW[:, :, -1]

        """
        vertically integrate Ri-dependent dissipation and EKE
        """
        a_loc = np.sum(veros.c_Ri_diss[:, :, :-1] * veros.eke[:, :, :-1,
                                                              veros.tau] * veros.maskW[:, :, :-1] * veros.dzw[:-1], axis=2)
        b_loc = np.sum(veros.eke[:, :, :-1, veros.tau] *
                       veros.maskW[:, :, :-1] * veros.dzw[:-1], axis=2)
        a_loc += veros.c_Ri_diss[:, :, -1] * veros.eke[:, :, -1,
                                                       veros.tau] * veros.maskW[:, :, -1] * veros.dzw[-1] * 0.5
        b_loc += veros.eke[:, :, -1, veros.tau] * veros.maskW[:, :, -1] * veros.dzw[-1] * 0.5

        """
        add bottom fluxes by lee waves and bottom friction to a_loc
        """
        a_loc[2:-2, 2:-2] += np.sum((veros.c_lee[2:-2, 2:-2, np.newaxis] * veros.eke[2:-2, 2:-2, :, veros.tau]
                                     * veros.maskW[2:-2, 2:-2, :] * veros.dzw[np.newaxis, np.newaxis, :]
                                     + 2 * veros.eke_r_bot *
                                     veros.eke[2:-2, 2:-2, :, veros.tau] *
                                     math.sqrt(2.0) * veros.sqrteke[2:-2, 2:-2, :]
                                     * veros.maskW[2:-2, 2:-2, :]) * full_mask, axis=-1) * boundary_mask

        """
        dissipation constant is vertically integrated forcing divided by
        vertically integrated EKE to account for vertical EKE radiation
        """
        mask = b_loc > 0
        a_loc[...] = np.where(mask, a_loc / (b_loc + 1e-20), 0.)
        c_int[...] = a_loc[:, :, np.newaxis]
    else:
        """
        dissipation by local interior loss of balance with constant coefficient
        """
        c_int[...] = veros.eke_c_eps * veros.sqrteke / veros.eke_len * veros.maskW

    """
    vertical diffusion of EKE,forcing and dissipation
    """
    ks = veros.kbot[2:-2, 2:-2] - 1
    delta, a_tri, b_tri, c_tri, d_tri = (np.zeros((veros.nx, veros.ny, veros.nz)) for _ in range(5))
    delta[:, :, :-1] = veros.dt_tracer / veros.dzt[np.newaxis, np.newaxis, 1:] * 0.5 \
        * (veros.kappaM[2:-2, 2:-2, :-1] + veros.kappaM[2:-2, 2:-2, 1:]) * veros.alpha_eke
    a_tri[:, :, 1:-1] = -delta[:, :, :-2] / veros.dzw[1:-1]
    a_tri[:, :, -1] = -delta[:, :, -2] / (0.5 * veros.dzw[-1])
    b_tri[:, :, 1:-1] = 1 + (delta[:, :, 1:-1] + delta[:, :, :-2]) / \
        veros.dzw[1:-1] + veros.dt_tracer * c_int[2:-2, 2:-2, 1:-1]
    b_tri[:, :, -1] = 1 + delta[:, :, -2] / \
        (0.5 * veros.dzw[-1]) + veros.dt_tracer * c_int[2:-2, 2:-2, -1]
    b_tri_edge = 1 + delta / veros.dzw[np.newaxis, np.newaxis, :] \
        + veros.dt_tracer * c_int[2:-2, 2:-2, :]
    c_tri[:, :, :-1] = -delta[:, :, :-1] / veros.dzw[np.newaxis, np.newaxis, :-1]
    d_tri[:, :, :] = veros.eke[2:-2, 2:-2, :, veros.tau] + veros.dt_tracer * forc[2:-2, 2:-2, :]
    sol, water_mask = utilities.solve_implicit(veros, ks, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)
    veros.eke[2:-2, 2:-2, :, veros.taup1] = np.where(water_mask, sol, veros.eke[2:-2, 2:-2, :, veros.taup1])

    """
    store eke dissipation
    """
    if veros.enable_eke_leewave_dissipation:
        veros.eke_diss_iw[...] = 0.
        veros.eke_diss_tke[...] = veros.c_Ri_diss * veros.eke[:, :, :, veros.taup1]

        """
        flux by lee wave generation and bottom friction
        """
        veros.eke_diss_iw[2:-2, 2:-2, :] += (veros.c_lee[2:-2, 2:-2, np.newaxis] * veros.eke[2:-2, 2:-2, :, veros.taup1]
                                             * veros.maskW[2:-2, 2:-2, :]) * full_mask
        veros.eke_diss_tke[2:-2, 2:-2, :] += (2 * veros.eke_r_bot * veros.eke[2:-2, 2:-2, :, veros.taup1] * math.sqrt(2.0)
                                              * veros.sqrteke[2:-2, 2:-2, :] * veros.maskW[2:-2, 2:-2, :] / veros.dzw[np.newaxis, np.newaxis, :]) * full_mask

        """
        account for sligthly incorrect integral of dissipation due to time stepping
        """
        a_loc = np.sum((veros.eke_diss_iw[:, :, :-1] + veros.eke_diss_tke[:, :, :-1])
                       * veros.dzw[np.newaxis, np.newaxis, :-1], axis=2)
        b_loc = np.sum(c_int[:, :, :-1] * veros.eke[:, :, :-1, veros.taup1]
                       * veros.dzw[np.newaxis, np.newaxis, :-1], axis=2)
        a_loc += (veros.eke_diss_iw[:, :, -1] + veros.eke_diss_tke[:, :, -1]) * veros.dzw[-1] * 0.5
        b_loc += c_int[:, :, -1] * veros.eke[:, :, -1, veros.taup1] * veros.dzw[-1] * 0.5
        mask = a_loc != 0.
        b_loc[...] = np.where(mask, b_loc / (a_loc + 1e-20), 0.)
        veros.eke_diss_iw[...] *= b_loc[:, :, np.newaxis]
        veros.eke_diss_tke[...] *= b_loc[:, :, np.newaxis]

        """
        store diagnosed flux by lee waves and bottom friction
        """
        veros.eke_lee_flux[2:-2, 2:-2] = np.where(boundary_mask, np.sum(veros.c_lee[2:-2, 2:-2, np.newaxis] * veros.eke[2:-2, 2:-2, :, veros.taup1]
                                                                        * veros.dzw[np.newaxis, np.newaxis, :] * full_mask, axis=-1), veros.eke_lee_flux[2:-2, 2:-2])
        veros.eke_bot_flux[2:-2, 2:-2] = np.where(boundary_mask, np.sum(2 * veros.eke_r_bot * veros.eke[2:-2, 2:-2, :, veros.taup1]
                                                                        * math.sqrt(2.0) * veros.sqrteke[2:-2, 2:-2, :] * full_mask, axis=-1), veros.eke_bot_flux[2:-2, 2:-2])
    else:
        veros.eke_diss_iw = c_int * veros.eke[:, :, :, veros.taup1]
        veros.eke_diss_tke[...] = 0.

    """
    add tendency due to lateral diffusion
    """
    veros.flux_east[:-1, :, :] = 0.5 * np.maximum(500., veros.K_gm[:-1, :, :] + veros.K_gm[1:, :, :]) \
        * (veros.eke[1:, :, :, veros.tau] - veros.eke[:-1, :, :, veros.tau]) \
        / (veros.cost[np.newaxis, :, np.newaxis] * veros.dxu[:-1, np.newaxis, np.newaxis]) * veros.maskU[:-1, :, :]
    veros.flux_east[-1, :, :] = 0.
    veros.flux_north[:, :-1, :] = 0.5 * np.maximum(500., veros.K_gm[:, :-1, :] + veros.K_gm[:, 1:, :]) \
        * (veros.eke[:, 1:, :, veros.tau] - veros.eke[:, :-1, :, veros.tau]) \
        / veros.dyu[np.newaxis, :-1, np.newaxis] * veros.maskV[:, :-1, :] * veros.cosu[np.newaxis, :-1, np.newaxis]
    veros.flux_north[:, -1, :] = 0.
    veros.eke[2:-2, 2:-2, :, veros.taup1] += veros.dt_tracer * veros.maskW[2:-2, 2:-2, :] \
        * ((veros.flux_east[2:-2, 2:-2, :] - veros.flux_east[1:-3, 2:-2, :])
           / (veros.cost[np.newaxis, 2:-2, np.newaxis] * veros.dxt[2:-2, np.newaxis, np.newaxis])
           + (veros.flux_north[2:-2, 2:-2, :] - veros.flux_north[2:-2, 1:-3, :])
           / (veros.cost[np.newaxis, 2:-2, np.newaxis] * veros.dyt[np.newaxis, 2:-2, np.newaxis]))

    """
    add tendency due to advection
    """
    if veros.enable_eke_superbee_advection:
        advection.adv_flux_superbee_wgrid(
            veros, veros.flux_east, veros.flux_north, veros.flux_top, veros.eke[:, :, :, veros.tau])
    if veros.enable_eke_upwind_advection:
        advection.adv_flux_upwind_wgrid(
            veros, veros.flux_east, veros.flux_north, veros.flux_top, veros.eke[:, :, :, veros.tau])
    if veros.enable_eke_superbee_advection or veros.enable_eke_upwind_advection:
        veros.deke[2:-2, 2:-2, :, veros.tau] = veros.maskW[2:-2, 2:-2, :] * (-(veros.flux_east[2:-2, 2:-2, :] - veros.flux_east[1:-3, 2:-2, :])
                                                                             / (veros.cost[np.newaxis, 2:-2, np.newaxis] * veros.dxt[2:-2, np.newaxis, np.newaxis])
                                                                             - (veros.flux_north[2:-2, 2:-2, :] - veros.flux_north[2:-2, 1:-3, :])
                                                                             / (veros.cost[np.newaxis, 2:-2, np.newaxis] * veros.dyt[np.newaxis, 2:-2, np.newaxis]))
        veros.deke[:, :, 0, veros.tau] += -veros.flux_top[:, :, 0] / veros.dzw[0]
        veros.deke[:, :, 1:-1, veros.tau] += -(veros.flux_top[:, :, 1:-1] -
                                               veros.flux_top[:, :, :-2]) / veros.dzw[np.newaxis, np.newaxis, 1:-1]
        veros.deke[:, :, -1, veros.tau] += - \
            (veros.flux_top[:, :, -1] - veros.flux_top[:, :, -2]) / (0.5 * veros.dzw[-1])
        """
        Adam Bashforth time stepping
        """
        veros.eke[:, :, :, veros.taup1] += veros.dt_tracer * ((1.5 + veros.AB_eps) * veros.deke[:, :, :, veros.tau]
                                                              - (0.5 + veros.AB_eps) * veros.deke[:, :, :, veros.taum1])

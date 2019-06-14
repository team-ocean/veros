import math

from .. import veros_method, runtime_settings as rs
from ..variables import allocate
from . import utilities, advection


@veros_method
def init_eke(vs):
    """
    Initialize EKE
    """
    if vs.enable_eke_leewave_dissipation:
        vs.hrms_k0[...] = np.maximum(vs.eke_hrms_k0_min, 2 / vs.pi * vs.eke_topo_hrms**2
                                        / np.maximum(1e-12, vs.eke_topo_lam)**1.5)


@veros_method
def set_eke_diffusivities(vs):
    """
    set skew diffusivity K_gm and isopycnal diffusivity K_iso
    set also vertical viscosity if TEM formalism is chosen
    """
    C_rossby = allocate(vs, ('xt', 'yt'))

    if vs.enable_eke:
        """
        calculate Rossby radius as minimum of mid-latitude and equatorial R. rad.
        """
        C_rossby[...] = np.sum(np.sqrt(np.maximum(0., vs.Nsqr[:, :, :, vs.tau]))
                               * vs.dzw[np.newaxis, np.newaxis, :] * vs.maskW[:, :, :] / vs.pi, axis=2)
        vs.L_rossby[...] = np.minimum(C_rossby / np.maximum(np.abs(vs.coriolis_t), 1e-16),
                                         np.sqrt(C_rossby / np.maximum(2 * vs.beta, 1e-16)))
        """
        calculate vertical viscosity and skew diffusivity
        """
        vs.sqrteke = np.sqrt(np.maximum(0., vs.eke[:, :, :, vs.tau]))
        vs.L_rhines[...] = np.sqrt(
            vs.sqrteke / np.maximum(vs.beta[..., np.newaxis], 1e-16))
        vs.eke_len[...] = np.maximum(vs.eke_lmin, np.minimum(
            vs.eke_cross * vs.L_rossby[..., np.newaxis], vs.eke_crhin * vs.L_rhines))
        vs.K_gm[...] = np.minimum(vs.eke_k_max, vs.eke_c_k * vs.eke_len * vs.sqrteke)
    else:
        """
        use fixed GM diffusivity
        """
        vs.K_gm[...] = vs.K_gm_0

    if vs.enable_TEM_friction:
        vs.kappa_gm[...] = vs.K_gm * np.minimum(0.01, vs.coriolis_t[..., np.newaxis]**2
                                                      / np.maximum(1e-9, vs.Nsqr[..., vs.tau])) * vs.maskW
    if vs.enable_eke and vs.enable_eke_isopycnal_diffusion:
        vs.K_iso[...] = vs.K_gm
    else:
        vs.K_iso[...] = vs.K_iso_0  # always constant


@veros_method
def integrate_eke(vs):
    """
    integrate EKE equation on W grid
    """
    c_int = allocate(vs, ('xt', 'yt', 'zw'))

    """
    forcing by dissipation by lateral friction and GM using TRM formalism or skew diffusion
    """
    forc = vs.K_diss_h + vs.K_diss_gm - vs.P_diss_skew

    """
    store transfer due to isopycnal and horizontal mixing from dyn. enthalpy
    by non-linear eq.of state either to EKE or to heat
    """
    if not vs.enable_store_cabbeling_heat:
        forc[...] += -vs.P_diss_hmix - vs.P_diss_iso

    """
    coefficient for dissipation of EKE:
    by lee wave generation, Ri-dependent interior loss of balance and bottom friction
    """
    if vs.enable_eke_leewave_dissipation:
        """
        by lee wave generation
        """
        vs.c_lee[...] = 0.
        ks = vs.kbot[2:-2, 2:-2] - 1
        ki = np.arange(vs.nz)[np.newaxis, np.newaxis, :]
        boundary_mask = (ks >= 0) & (ks < vs.nz - 1)
        full_mask = boundary_mask[:, :, np.newaxis] & (ki == ks[:, :, np.newaxis])
        fxa = np.maximum(0, vs.Nsqr[2:-2, 2:-2, :, vs.tau])**0.25
        fxa *= 1.5 * fxa / np.sqrt(np.maximum(1e-6, np.abs(vs.coriolis_t[2:-2, 2:-2, np.newaxis]))) - 2
        vs.c_lee[2:-2, 2:-2] = boundary_mask * vs.c_lee0 * vs.hrms_k0[2:-2, 2:-2] \
                               * np.sum(np.sqrt(vs.sqrteke[2:-2, 2:-2, :]) * np.maximum(0, fxa)
                                        / vs.dzw[np.newaxis, np.newaxis, :] * full_mask, axis=-1)

        """
        Ri-dependent dissipation by interior loss of balance
        """
        vs.c_Ri_diss[...] = 0
        uz = (((vs.u[1:, 1:, 1:, vs.tau] - vs.u[1:, 1:, :-1, vs.tau]) / vs.dzt[np.newaxis, np.newaxis, :-1] * vs.maskU[1:, 1:, :-1])**2
              + ((vs.u[:-1, 1:, 1:, vs.tau] - vs.u[:-1, 1:, :-1, vs.tau]) / vs.dzt[np.newaxis, np.newaxis, :-1] * vs.maskU[:-1, 1:, :-1])**2) \
            / (vs.maskU[1:, 1:, :-1] + vs.maskU[:-1, 1:, :-1] + 1e-18)
        vz = (((vs.v[1:, 1:, 1:, vs.tau] - vs.v[1:, 1:, :-1, vs.tau]) / vs.dzt[np.newaxis, np.newaxis, :-1] * vs.maskV[1:, 1:, :-1])**2
              + ((vs.v[1:, :-1, 1:, vs.tau] - vs.v[1:, :-1, :-1, vs.tau]) / vs.dzt[np.newaxis, np.newaxis, :-1] * vs.maskV[1:, :-1, :-1])**2) \
            / (vs.maskV[1:, 1:, :-1] + vs.maskV[1:, :-1, :-1] + 1e-18)
        Ri = np.maximum(1e-8, vs.Nsqr[1:, 1:, :-1, vs.tau]) / (uz + vz + 1e-18)
        fxa = 1 - 0.5 * (1. + np.tanh((Ri - vs.eke_Ri0) / vs.eke_Ri1))
        vs.c_Ri_diss[1:, 1:, :-1] = vs.maskW[1:, 1:, :-1] * fxa * vs.eke_int_diss0
        vs.c_Ri_diss[:, :, -1] = vs.c_Ri_diss[:, :, -2] * vs.maskW[:, :, -1]

        """
        vertically integrate Ri-dependent dissipation and EKE
        """
        a_loc = np.sum(vs.c_Ri_diss[:, :, :-1] * vs.eke[:, :, :-1, vs.tau] * vs.maskW[:, :, :-1] * vs.dzw[:-1], axis=2)
        b_loc = np.sum(vs.eke[:, :, :-1, vs.tau] *
                       vs.maskW[:, :, :-1] * vs.dzw[:-1], axis=2)
        a_loc += vs.c_Ri_diss[:, :, -1] * vs.eke[:, :, -1, vs.tau] * vs.maskW[:, :, -1] * vs.dzw[-1] * 0.5
        b_loc += vs.eke[:, :, -1, vs.tau] * vs.maskW[:, :, -1] * vs.dzw[-1] * 0.5

        """
        add bottom fluxes by lee waves and bottom friction to a_loc
        """
        a_loc[2:-2, 2:-2] += np.sum((vs.c_lee[2:-2, 2:-2, np.newaxis] * vs.eke[2:-2, 2:-2, :, vs.tau] \
                                     * vs.maskW[2:-2, 2:-2, :] * vs.dzw[np.newaxis, np.newaxis, :] \
                                     + 2 * vs.eke_r_bot * vs.eke[2:-2, 2:-2, :, vs.tau] \
                                     * math.sqrt(2.0) * vs.sqrteke[2:-2, 2:-2, :]
                                     * vs.maskW[2:-2, 2:-2, :]) * full_mask, axis=-1) * boundary_mask

        """
        dissipation constant is vertically integrated forcing divided by
        vertically integrated EKE to account for vertical EKE radiation
        """
        mask = b_loc > 0
        a_loc[...] = utilities.where(vs, mask, a_loc / (b_loc + 1e-20), 0.)
        c_int[...] = a_loc[:, :, np.newaxis]
    else:
        """
        dissipation by local interior loss of balance with constant coefficient
        """
        c_int[...] = vs.eke_c_eps * vs.sqrteke / vs.eke_len * vs.maskW

    """
    vertical diffusion of EKE,forcing and dissipation
    """
    ks = vs.kbot[2:-2, 2:-2] - 1
    delta, a_tri, b_tri, c_tri, d_tri = (allocate(vs, ('xt', 'yt', 'zt'), include_ghosts=False) for _ in range(5))
    delta[:, :, :-1] = vs.dt_tracer / vs.dzt[np.newaxis, np.newaxis, 1:] * 0.5 \
        * (vs.kappaM[2:-2, 2:-2, :-1] + vs.kappaM[2:-2, 2:-2, 1:]) * vs.alpha_eke
    a_tri[:, :, 1:-1] = -delta[:, :, :-2] / vs.dzw[1:-1]
    a_tri[:, :, -1] = -delta[:, :, -2] / (0.5 * vs.dzw[-1])
    b_tri[:, :, 1:-1] = 1 + (delta[:, :, 1:-1] + delta[:, :, :-2]) / \
        vs.dzw[1:-1] + vs.dt_tracer * c_int[2:-2, 2:-2, 1:-1]
    b_tri[:, :, -1] = 1 + delta[:, :, -2] / \
        (0.5 * vs.dzw[-1]) + vs.dt_tracer * c_int[2:-2, 2:-2, -1]
    b_tri_edge = 1 + delta / vs.dzw[np.newaxis, np.newaxis, :] \
        + vs.dt_tracer * c_int[2:-2, 2:-2, :]
    c_tri[:, :, :-1] = -delta[:, :, :-1] / vs.dzw[np.newaxis, np.newaxis, :-1]
    d_tri[:, :, :] = vs.eke[2:-2, 2:-2, :, vs.tau] + vs.dt_tracer * forc[2:-2, 2:-2, :]
    sol, water_mask = utilities.solve_implicit(vs, ks, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)
    vs.eke[2:-2, 2:-2, :, vs.taup1] = utilities.where(vs, water_mask, sol, vs.eke[2:-2, 2:-2, :, vs.taup1])

    """
    store eke dissipation
    """
    if vs.enable_eke_leewave_dissipation:
        vs.eke_diss_iw[...] = 0.
        vs.eke_diss_tke[...] = vs.c_Ri_diss * vs.eke[:, :, :, vs.taup1]

        """
        flux by lee wave generation and bottom friction
        """
        vs.eke_diss_iw[2:-2, 2:-2, :] += (vs.c_lee[2:-2, 2:-2, np.newaxis] * vs.eke[2:-2, 2:-2, :, vs.taup1]
                                             * vs.maskW[2:-2, 2:-2, :]) * full_mask
        if vs.pyom_compatibility_mode:
            vs.eke_diss_tke[2:-2, 2:-2, :] += (2 * vs.eke_r_bot * vs.eke[2:-2, 2:-2, :, vs.taup1] * np.sqrt(np.float32(2.0))
                                                  * vs.sqrteke[2:-2, 2:-2, :] * vs.maskW[2:-2, 2:-2, :] / vs.dzw[np.newaxis, np.newaxis, :]) * full_mask
        else:
            vs.eke_diss_tke[2:-2, 2:-2, :] += (2 * vs.eke_r_bot * vs.eke[2:-2, 2:-2, :, vs.taup1] * math.sqrt(2.0)
                                                  * vs.sqrteke[2:-2, 2:-2, :] * vs.maskW[2:-2, 2:-2, :] / vs.dzw[np.newaxis, np.newaxis, :]) * full_mask
        """
        account for sligthly incorrect integral of dissipation due to time stepping
        """
        a_loc = np.sum((vs.eke_diss_iw[:, :, :-1] + vs.eke_diss_tke[:, :, :-1])
                       * vs.dzw[np.newaxis, np.newaxis, :-1], axis=2)
        b_loc = np.sum(c_int[:, :, :-1] * vs.eke[:, :, :-1, vs.taup1]
                       * vs.dzw[np.newaxis, np.newaxis, :-1], axis=2)
        a_loc += (vs.eke_diss_iw[:, :, -1] + vs.eke_diss_tke[:, :, -1]) * vs.dzw[-1] * 0.5
        b_loc += c_int[:, :, -1] * vs.eke[:, :, -1, vs.taup1] * vs.dzw[-1] * 0.5
        mask = a_loc != 0.
        b_loc[...] = utilities.where(vs, mask, b_loc / (a_loc + 1e-20), 0.)
        vs.eke_diss_iw[...] *= b_loc[:, :, np.newaxis]
        vs.eke_diss_tke[...] *= b_loc[:, :, np.newaxis]

        """
        store diagnosed flux by lee waves and bottom friction
        """
        vs.eke_lee_flux[2:-2, 2:-2] = utilities.where(vs, boundary_mask, np.sum(vs.c_lee[2:-2, 2:-2, np.newaxis] * vs.eke[2:-2, 2:-2, :, vs.taup1]
                                                                        * vs.dzw[np.newaxis, np.newaxis, :] * full_mask, axis=-1), vs.eke_lee_flux[2:-2, 2:-2])
        vs.eke_bot_flux[2:-2, 2:-2] = utilities.where(vs, boundary_mask, np.sum(2 * vs.eke_r_bot * vs.eke[2:-2, 2:-2, :, vs.taup1]
                                                                        * math.sqrt(2.0) * vs.sqrteke[2:-2, 2:-2, :] * full_mask, axis=-1), vs.eke_bot_flux[2:-2, 2:-2])
    else:
        vs.eke_diss_iw = c_int * vs.eke[:, :, :, vs.taup1]
        vs.eke_diss_tke[...] = 0.

    """
    add tendency due to lateral diffusion
    """
    vs.flux_east[:-1, :, :] = 0.5 * np.maximum(500., vs.K_gm[:-1, :, :] + vs.K_gm[1:, :, :]) \
        * (vs.eke[1:, :, :, vs.tau] - vs.eke[:-1, :, :, vs.tau]) \
        / (vs.cost[np.newaxis, :, np.newaxis] * vs.dxu[:-1, np.newaxis, np.newaxis]) * vs.maskU[:-1, :, :]
    vs.flux_east[-1, :, :] = 0.
    vs.flux_north[:, :-1, :] = 0.5 * np.maximum(500., vs.K_gm[:, :-1, :] + vs.K_gm[:, 1:, :]) \
        * (vs.eke[:, 1:, :, vs.tau] - vs.eke[:, :-1, :, vs.tau]) \
        / vs.dyu[np.newaxis, :-1, np.newaxis] * vs.maskV[:, :-1, :] * vs.cosu[np.newaxis, :-1, np.newaxis]
    vs.flux_north[:, -1, :] = 0.
    vs.eke[2:-2, 2:-2, :, vs.taup1] += vs.dt_tracer * vs.maskW[2:-2, 2:-2, :] \
        * ((vs.flux_east[2:-2, 2:-2, :] - vs.flux_east[1:-3, 2:-2, :])
           / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxt[2:-2, np.newaxis, np.newaxis])
           + (vs.flux_north[2:-2, 2:-2, :] - vs.flux_north[2:-2, 1:-3, :])
           / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dyt[np.newaxis, 2:-2, np.newaxis]))

    """
    add tendency due to advection
    """
    if vs.enable_eke_superbee_advection:
        advection.adv_flux_superbee_wgrid(
            vs, vs.flux_east, vs.flux_north, vs.flux_top, vs.eke[:, :, :, vs.tau]
            )
    if vs.enable_eke_upwind_advection:
        advection.adv_flux_upwind_wgrid(
            vs, vs.flux_east, vs.flux_north, vs.flux_top, vs.eke[:, :, :, vs.tau]
            )
    if vs.enable_eke_superbee_advection or vs.enable_eke_upwind_advection:
        vs.deke[2:-2, 2:-2, :, vs.tau] = vs.maskW[2:-2, 2:-2, :] * (-(vs.flux_east[2:-2, 2:-2, :] - vs.flux_east[1:-3, 2:-2, :])
                                                                    / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxt[2:-2, np.newaxis, np.newaxis])
                                                                    - (vs.flux_north[2:-2, 2:-2, :] - vs.flux_north[2:-2, 1:-3, :])
                                                                    / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dyt[np.newaxis, 2:-2, np.newaxis]))
        vs.deke[:, :, 0, vs.tau] += -vs.flux_top[:, :, 0] / vs.dzw[0]
        vs.deke[:, :, 1:-1, vs.tau] += -(vs.flux_top[:, :, 1:-1] -
                                               vs.flux_top[:, :, :-2]) / vs.dzw[np.newaxis, np.newaxis, 1:-1]
        vs.deke[:, :, -1, vs.tau] += -(vs.flux_top[:, :, -1] - vs.flux_top[:, :, -2]) / (0.5 * vs.dzw[-1])
        """
        Adam Bashforth time stepping
        """
        vs.eke[:, :, :, vs.taup1] += vs.dt_tracer * ((1.5 + vs.AB_eps) * vs.deke[:, :, :, vs.tau]
                                                   - (0.5 + vs.AB_eps) * vs.deke[:, :, :, vs.taum1])

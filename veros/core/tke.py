import math

from .. import veros_method
from ..variables import allocate
from . import advection, utilities


@veros_method
def set_tke_diffusivities(vs):
    """
    set vertical diffusivities based on TKE model
    """
    Rinumber = allocate(vs, ('xt', 'yt', 'zw'))

    if vs.enable_tke:
        vs.sqrttke[...] = np.sqrt(np.maximum(0., vs.tke[:, :, :, vs.tau]))
        """
        calculate buoyancy length scale
        """
        vs.mxl[...] = math.sqrt(2) * vs.sqrttke \
            / np.sqrt(np.maximum(1e-12, vs.Nsqr[:, :, :, vs.tau])) * vs.maskW

        """
        apply limits for mixing length
        """
        if vs.tke_mxl_choice == 1:
            """
            bounded by the distance to surface/bottom
            """
            vs.mxl[...] = np.minimum(
                np.minimum(vs.mxl, -vs.zw[np.newaxis, np.newaxis, :]
                           + vs.dzw[np.newaxis, np.newaxis, :] * 0.5
                           ), vs.ht[:, :, np.newaxis] + vs.zw[np.newaxis, np.newaxis, :]
            )
            vs.mxl[...] = np.maximum(vs.mxl, vs.mxl_min)
        elif vs.tke_mxl_choice == 2:
            """
            bound length scale as in mitgcm/OPA code

            Note that the following code doesn't vectorize. If critical for performance,
            consider re-implementing it in Cython.
            """
            for k in range(vs.nz - 2, -1, -1):
                vs.mxl[:, :, k] = np.minimum(vs.mxl[:, :, k], vs.mxl[:, :, k + 1] + vs.dzt[k + 1])
            vs.mxl[:, :, -1] = np.minimum(vs.mxl[:, :, -1], vs.mxl_min + vs.dzt[-1])
            for k in range(1, vs.nz):
                vs.mxl[:, :, k] = np.minimum(vs.mxl[:, :, k], vs.mxl[:, :, k - 1] + vs.dzt[k])
            vs.mxl[...] = np.maximum(vs.mxl, vs.mxl_min)
        else:
            raise ValueError('unknown mixing length choice in tke_mxl_choice')

        """
        calculate viscosity and diffusivity based on Prandtl number
        """
        utilities.enforce_boundaries(vs, vs.K_diss_v)
        vs.kappaM[...] = np.minimum(vs.kappaM_max, vs.c_k * vs.mxl * vs.sqrttke)
        Rinumber[...] = vs.Nsqr[:, :, :, vs.tau] / \
            np.maximum(vs.K_diss_v / np.maximum(1e-12, vs.kappaM), 1e-12)
        if vs.enable_idemix:
            Rinumber[...] = np.minimum(Rinumber, vs.kappaM * vs.Nsqr[:, :, :, vs.tau]
                                  / np.maximum(1e-12, vs.alpha_c * vs.E_iw[:, :, :, vs.tau]**2))
        if vs.enable_Prandtl_tke:
            vs.Prandtlnumber[...] = np.maximum(1., np.minimum(10, 6.6 * Rinumber))
        else:
            vs.Prandtlnumber[...] = vs.Prandtl_tke0
        vs.kappaH[...] = np.maximum(vs.kappaH_min, vs.kappaM / vs.Prandtlnumber)
        vs.kappaM[...] = np.maximum(vs.kappaM_min, vs.kappaM)
    else:
        vs.kappaM[...] = vs.kappaM_0
        vs.kappaH[...] = vs.kappaH_0


@veros_method
def integrate_tke(vs):
    """
    integrate Tke equation on W grid with surface flux boundary condition
    """
    dt_tke = vs.dt_mom  # use momentum time step to prevent spurious oscillations

    """
    Sources and sinks by vertical friction, vertical mixing, and non-conservative advection
    """
    forc = vs.K_diss_v - vs.P_diss_v - vs.P_diss_adv

    """
    store transfer due to vertical mixing from dyn. enthalpy by non-linear eq.of
    state either to TKE or to heat
    """
    if not vs.enable_store_cabbeling_heat:
        forc[...] += -vs.P_diss_nonlin

    """
    transfer part of dissipation of EKE to TKE
    """
    if vs.enable_eke:
        forc[...] += vs.eke_diss_tke

    if vs.enable_idemix:
        """
        transfer dissipation of internal waves to TKE
        """
        forc[...] += vs.iw_diss
        """
        store bottom friction either in TKE or internal waves
        """
        if vs.enable_store_bottom_friction_tke:
            forc[...] += vs.K_diss_bot
    else:  # short-cut without idemix
        if vs.enable_eke:
            forc[...] += vs.eke_diss_iw
        else:  # and without EKE model
            if vs.enable_store_cabbeling_heat:
                forc[...] += vs.K_diss_gm + vs.K_diss_h - vs.P_diss_skew \
                    - vs.P_diss_hmix - vs.P_diss_iso
            else:
                forc[...] += vs.K_diss_gm + vs.K_diss_h - vs.P_diss_skew
        forc[...] += vs.K_diss_bot

    """
    vertical mixing and dissipation of TKE
    """
    ks = vs.kbot[2:-2, 2:-2] - 1

    a_tri = allocate(vs, ('xt', 'yt', 'zt'), include_ghosts=False)
    b_tri = allocate(vs, ('xt', 'yt', 'zt'), include_ghosts=False)
    c_tri = allocate(vs, ('xt', 'yt', 'zt'), include_ghosts=False)
    d_tri = allocate(vs, ('xt', 'yt', 'zt'), include_ghosts=False)
    delta = allocate(vs, ('xt', 'yt', 'zt'), include_ghosts=False)

    delta[:, :, :-1] = dt_tke / vs.dzt[np.newaxis, np.newaxis, 1:] * vs.alpha_tke * 0.5 \
        * (vs.kappaM[2:-2, 2:-2, :-1] + vs.kappaM[2:-2, 2:-2, 1:])

    a_tri[:, :, 1:-1] = -delta[:, :, :-2] / vs.dzw[np.newaxis, np.newaxis, 1:-1]
    a_tri[:, :, -1] = -delta[:, :, -2] / (0.5 * vs.dzw[-1])

    b_tri[:, :, 1:-1] = 1 + (delta[:, :, 1:-1] + delta[:, :, :-2]) / vs.dzw[np.newaxis, np.newaxis, 1:-1] \
        + dt_tke * vs.c_eps \
        * vs.sqrttke[2:-2, 2:-2, 1:-1] / vs.mxl[2:-2, 2:-2, 1:-1]
    b_tri[:, :, -1] = 1 + delta[:, :, -2] / (0.5 * vs.dzw[-1]) \
        + dt_tke * vs.c_eps / vs.mxl[2:-2, 2:-2, -1] * vs.sqrttke[2:-2, 2:-2, -1]
    b_tri_edge = 1 + delta / vs.dzw[np.newaxis, np.newaxis, :] \
        + dt_tke * vs.c_eps / vs.mxl[2:-2, 2:-2, :] * vs.sqrttke[2:-2, 2:-2, :]

    c_tri[:, :, :-1] = -delta[:, :, :-1] / vs.dzw[np.newaxis, np.newaxis, :-1]

    d_tri[...] = vs.tke[2:-2, 2:-2, :, vs.tau] + dt_tke * forc[2:-2, 2:-2, :]
    d_tri[:, :, -1] += dt_tke * vs.forc_tke_surface[2:-2, 2:-2] / (0.5 * vs.dzw[-1])

    sol, water_mask = utilities.solve_implicit(vs, ks, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)
    vs.tke[2:-2, 2:-2, :, vs.taup1] = utilities.where(vs, water_mask, sol, vs.tke[2:-2, 2:-2, :, vs.taup1])

    """
    store tke dissipation for diagnostics
    """
    vs.tke_diss[...] = vs.c_eps / vs.mxl * vs.sqrttke * vs.tke[:, :, :, vs.taup1]

    """
    Add TKE if surface density flux drains TKE in uppermost box
    """
    mask = vs.tke[2:-2, 2:-2, -1, vs.taup1] < 0.0
    vs.tke_surf_corr[...] = 0.
    vs.tke_surf_corr[2:-2, 2:-2] = utilities.where(vs, mask,
                                            -vs.tke[2:-2, 2:-2, -1, vs.taup1] * 0.5
                                               * vs.dzw[-1] / dt_tke,
                                            0.)
    vs.tke[2:-2, 2:-2, -1, vs.taup1] = np.maximum(0., vs.tke[2:-2, 2:-2, -1, vs.taup1])

    if vs.enable_tke_hor_diffusion:
        """
        add tendency due to lateral diffusion
        """
        vs.flux_east[:-1, :, :] = vs.K_h_tke * (vs.tke[1:, :, :, vs.tau] - vs.tke[:-1, :, :, vs.tau]) \
            / (vs.cost[np.newaxis, :, np.newaxis] * vs.dxu[:-1, np.newaxis, np.newaxis]) * vs.maskU[:-1, :, :]
        if vs.pyom_compatibility_mode:
            vs.flux_east[-5, :, :] = 0.
        else:
            vs.flux_east[-1, :, :] = 0.
        vs.flux_north[:, :-1, :] = vs.K_h_tke * (vs.tke[:, 1:, :, vs.tau] - vs.tke[:, :-1, :, vs.tau]) \
            / vs.dyu[np.newaxis, :-1, np.newaxis] * vs.maskV[:, :-1, :] * vs.cosu[np.newaxis, :-1, np.newaxis]
        vs.flux_north[:, -1, :] = 0.
        vs.tke[2:-2, 2:-2, :, vs.taup1] += dt_tke * vs.maskW[2:-2, 2:-2, :] * \
            ((vs.flux_east[2:-2, 2:-2, :] - vs.flux_east[1:-3, 2:-2, :])
             / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxt[2:-2, np.newaxis, np.newaxis])
             + (vs.flux_north[2:-2, 2:-2, :] - vs.flux_north[2:-2, 1:-3, :])
             / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dyt[np.newaxis, 2:-2, np.newaxis]))

    """
    add tendency due to advection
    """
    if vs.enable_tke_superbee_advection:
        advection.adv_flux_superbee_wgrid(
            vs, vs.flux_east, vs.flux_north, vs.flux_top, vs.tke[:, :, :, vs.tau]
        )
    if vs.enable_tke_upwind_advection:
        advection.adv_flux_upwind_wgrid(
            vs, vs.flux_east, vs.flux_north, vs.flux_top, vs.tke[:, :, :, vs.tau]
        )
    if vs.enable_tke_superbee_advection or vs.enable_tke_upwind_advection:
        vs.dtke[2:-2, 2:-2, :, vs.tau] = vs.maskW[2:-2, 2:-2, :] * (-(vs.flux_east[2:-2, 2:-2, :] - vs.flux_east[1:-3, 2:-2, :])
                                                                     / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxt[2:-2, np.newaxis, np.newaxis])
                                                                    - (vs.flux_north[2:-2, 2:-2, :] - vs.flux_north[2:-2, 1:-3, :])
                                                                     / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dyt[np.newaxis, 2:-2, np.newaxis]))
        vs.dtke[:, :, 0, vs.tau] += -vs.flux_top[:, :, 0] / vs.dzw[0]
        vs.dtke[:, :, 1:-1, vs.tau] += -(vs.flux_top[:, :, 1:-1] - vs.flux_top[:, :, :-2]) / vs.dzw[1:-1]
        vs.dtke[:, :, -1, vs.tau] += -(vs.flux_top[:, :, -1] - vs.flux_top[:, :, -2]) / (0.5 * vs.dzw[-1])
        """
        Adam Bashforth time stepping
        """
        vs.tke[:, :, :, vs.taup1] += vs.dt_tracer * ((1.5 + vs.AB_eps) * vs.dtke[:, :, :, vs.tau]
                                                   - (0.5 + vs.AB_eps) * vs.dtke[:, :, :, vs.taum1])

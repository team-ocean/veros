import numpy as np

from veros import veros_kernel
from veros.core import advection, utilities

"""
IDEMIX as in Olbers and Eden, 2013
"""


@veros_kernel
def set_idemix_parameter(c0, v0, alpha_c, Nsqr, coriolis_t, pi, jstar, gamma, mu0,
                         dzw, maskW, tau, pyom_compatibility_mode):
    """
    set main IDEMIX parameter
    """
    bN0 = np.sum(np.sqrt(np.maximum(0., Nsqr[:, :, :-1, tau]))
                 * dzw[np.newaxis, np.newaxis, :-1] * maskW[:, :, :-1], axis=2) \
        + np.sqrt(np.maximum(0., Nsqr[:, :, -1, tau])) \
        * 0.5 * dzw[-1:] * maskW[:, :, -1]
    fxa = np.sqrt(np.maximum(0., Nsqr[..., tau])) \
        / (1e-22 + np.abs(coriolis_t[..., np.newaxis]))
    cstar = np.maximum(1e-2, bN0[:, :, np.newaxis] / (pi * jstar))
    c0[...] = np.maximum(0., gamma * cstar * gofx2(fxa, pi, pyom_compatibility_mode) * maskW)
    v0[...] = np.maximum(0., gamma * cstar * hofx1(fxa, pi) * maskW)
    alpha_c[...] = np.maximum(1e-4, mu0 * np.arccosh(np.maximum(1., fxa))
                              * np.abs(coriolis_t[..., np.newaxis]) / cstar**2) * maskW

    return c0, v0, alpha_c


@veros_kernel(static_args=('enable_eke', 'enable_store_cabbeling_heat',
                           'enable_eke_diss_bottom', 'enable_store_bottom_friction_tke',
                           'enable_eke_diss_surfbot', 'enable_idemix_hor_diffusion',
                           'pyom_compatibility_mode', 'enable_idemix_superbee_advection',
                           'enable_idemix_upwind_advection'))
def integrate_idemix(K_diss_gm, K_diss_h, K_diss_bot, P_diss_skew, P_diss_hmix, P_diss_iso,
                     eke_diss_iw, eke_diss_surfbot_frac, E_iw, dE_iw, dxt, dxu, dyt, dyu,
                     dzt, dzw, maskU, maskV, maskW, kbot, cost, cosu, nz, tau, taup1, taum1,
                     dt_tracer, tau_v, tau_h, c0, v0, alpha_c, AB_eps, forc_iw_bottom, iw_diss,
                     u_wgrid, v_wgrid, w_wgrid, forc_iw_surface, enable_eke,
                     enable_store_cabbeling_heat, enable_eke_diss_bottom,
                     enable_store_bottom_friction_tke, enable_eke_diss_surfbot,
                     enable_idemix_hor_diffusion, pyom_compatibility_mode,
                     enable_idemix_superbee_advection, enable_idemix_upwind_advection):
    """
    integrate idemix on W grid
    """
    a_tri, b_tri, c_tri, d_tri, delta = (np.zeros_like(maskW[2:-2, 2:-2, :]) for _ in range(5))
    forc = np.zeros_like(maskW)
    maxE_iw = np.zeros_like(maskW)

    """
    forcing by EKE dissipation
    """
    if enable_eke:
        forc[...] = eke_diss_iw
    else:  # shortcut without EKE model
        if enable_store_cabbeling_heat:
            forc[...] = K_diss_gm + K_diss_h - \
                P_diss_skew - P_diss_hmix - P_diss_iso
        else:
            forc[...] = K_diss_gm + K_diss_h - P_diss_skew

    if enable_eke and (enable_eke_diss_bottom or enable_eke_diss_surfbot):
        """
        vertically integrate EKE dissipation and inject at bottom and/or surface
        """
        a_loc = np.sum(dzw[np.newaxis, np.newaxis, :-1] *
                       forc[:, :, :-1] * maskW[:, :, :-1], axis=2)
        a_loc += 0.5 * forc[:, :, -1] * maskW[:, :, -1] * dzw[-1]
        forc[...] = 0.

        ks = np.maximum(0, kbot[2:-2, 2:-2] - 1)
        mask = ks[:, :, np.newaxis] == np.arange(nz)[np.newaxis, np.newaxis, :]
        if enable_eke_diss_bottom:
            forc[2:-2, 2:-2, :] = utilities.where(mask, a_loc[2:-2, 2:-2, np.newaxis] /
                                                  dzw[np.newaxis, np.newaxis, :], forc[2:-2, 2:-2, :])
        else:
            forc[2:-2, 2:-2, :] = utilities.where(mask, eke_diss_surfbot_frac * a_loc[2:-2, 2:-2, np.newaxis]
                                           / dzw[np.newaxis, np.newaxis, :], forc[2:-2, 2:-2, :])
            forc[2:-2, 2:-2, -1] = (1. - eke_diss_surfbot_frac) \
                                    * a_loc[2:-2, 2:-2] / (0.5 * dzw[-1])

    """
    forcing by bottom friction
    """
    if not enable_store_bottom_friction_tke:
        forc += K_diss_bot

    """
    prevent negative dissipation of IW energy
    """
    maxE_iw[...] = np.maximum(0., E_iw[:, :, :, tau])

    """
    vertical diffusion and dissipation is solved implicitly
    """
    ks = kbot[2:-2, 2:-2] - 1
    delta[:, :, :-1] = dt_tracer * tau_v / dzt[np.newaxis, np.newaxis, 1:] * 0.5 \
        * (c0[2:-2, 2:-2, :-1] + c0[2:-2, 2:-2, 1:])
    delta[:, :, -1] = 0.
    a_tri[:, :, 1:-1] = -delta[:, :, :-2] * c0[2:-2, 2:-2, :-2] \
        / dzw[np.newaxis, np.newaxis, 1:-1]
    a_tri[:, :, -1] = -delta[:, :, -2] / (0.5 * dzw[-1:]) * c0[2:-2, 2:-2, -2]
    b_tri[:, :, 1:-1] = 1 + delta[:, :, 1:-1] * c0[2:-2, 2:-2, 1:-1] / dzw[np.newaxis, np.newaxis, 1:-1] \
        + delta[:, :, :-2] * c0[2:-2, 2:-2, 1:-1] / dzw[np.newaxis, np.newaxis, 1:-1] \
        + dt_tracer * alpha_c[2:-2, 2:-2, 1:-1] * maxE_iw[2:-2, 2:-2, 1:-1]
    b_tri[:, :, -1] = 1 + delta[:, :, -2] / (0.5 * dzw[-1:]) * c0[2:-2, 2:-2, -1] \
        + dt_tracer * alpha_c[2:-2, 2:-2, -1] * maxE_iw[2:-2, 2:-2, -1]
    b_tri_edge = 1 + delta / dzw * c0[2:-2, 2:-2, :] \
        + dt_tracer * alpha_c[2:-2, 2:-2, :] * maxE_iw[2:-2, 2:-2, :]
    c_tri[:, :, :-1] = -delta[:, :, :-1] / \
        dzw[np.newaxis, np.newaxis, :-1] * c0[2:-2, 2:-2, 1:]
    d_tri[...] = E_iw[2:-2, 2:-2, :, tau] + dt_tracer * forc[2:-2, 2:-2, :]
    d_tri_edge = d_tri + dt_tracer * \
        forc_iw_bottom[2:-2, 2:-2, np.newaxis] / dzw[np.newaxis, np.newaxis, :]
    d_tri[:, :, -1] += dt_tracer * forc_iw_surface[2:-2, 2:-2] / (0.5 * dzw[-1:])
    sol, water_mask = utilities.solve_implicit(ks, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge, d_edge=d_tri_edge)
    E_iw[2:-2, 2:-2, :, taup1] = utilities.where(water_mask, sol, E_iw[2:-2, 2:-2, :, taup1])

    """
    store IW dissipation
    """
    iw_diss[...] = alpha_c * maxE_iw * E_iw[..., taup1]

    """
    add tendency due to lateral diffusion
    """
    flux_east = np.zeros_like(maskU)
    flux_north = np.zeros_like(maskV)
    flux_top = np.zeros_like(maskW)
    if enable_idemix_hor_diffusion:
        flux_east[:-1, :, :] = tau_h * 0.5 * (v0[1:, :, :] + v0[:-1, :, :]) \
            * (v0[1:, :, :] * E_iw[1:, :, :, tau] - v0[:-1, :, :] * E_iw[:-1, :, :, tau]) \
            / (cost[np.newaxis, :, np.newaxis] * dxu[:-1, np.newaxis, np.newaxis]) * maskU[:-1, :, :]
        if pyom_compatibility_mode:
            flux_east[-5, :, :] = 0.
        else:
            flux_east[-1, :, :] = 0.
        flux_north[:, :-1, :] = tau_h * 0.5 * (v0[:, 1:, :] + v0[:, :-1, :]) \
            * (v0[:, 1:, :] * E_iw[:, 1:, :, tau] - v0[:, :-1, :] * E_iw[:, :-1, :, tau]) \
            / dyu[np.newaxis, :-1, np.newaxis] * maskV[:, :-1, :] * cosu[np.newaxis, :-1, np.newaxis]
        flux_north[:, -1, :] = 0.
        E_iw[2:-2, 2:-2, :, taup1] += dt_tracer * maskW[2:-2, 2:-2, :] \
            * ((flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
               / (cost[np.newaxis, 2:-2, np.newaxis] * dxt[2:-2, np.newaxis, np.newaxis])
               + (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
               / (cost[np.newaxis, 2:-2, np.newaxis] * dyt[np.newaxis, 2:-2, np.newaxis]))

    """
    add tendency due to advection
    """
    if enable_idemix_superbee_advection:
        flux_east, flux_north, flux_top = advection.adv_flux_superbee_wgrid(
            flux_east, flux_north, flux_top, u_wgrid, v_wgrid, w_wgrid,
            E_iw[:, :, :, tau], maskW, dxt, dyt, dzw, cost, cosu, dt_tracer
            )
    if enable_idemix_upwind_advection:
        flux_east, flux_north, flux_top = advection.adv_flux_upwind_wgrid(
            flux_east, flux_north, flux_top, u_wgrid, v_wgrid, w_wgrid,
            E_iw[:, :, :, tau], maskW, cosu
            )
    if enable_idemix_superbee_advection or enable_idemix_upwind_advection:
        dE_iw[2:-2, 2:-2, :, tau] = maskW[2:-2, 2:-2, :] * (-(flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
                                                            / (cost[np.newaxis, 2:-2, np.newaxis] * dxt[2:-2, np.newaxis, np.newaxis])
                                                            - (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
                                                            / (cost[np.newaxis, 2:-2, np.newaxis] * dyt[np.newaxis, 2:-2, np.newaxis]))
        dE_iw[:, :, 0, tau] += -flux_top[:, :, 0] / dzw[0:1]
        dE_iw[:, :, 1:-1, tau] += -(flux_top[:, :, 1:-1] - flux_top[:, :, :-2]) \
            / dzw[np.newaxis, np.newaxis, 1:-1]
        dE_iw[:, :, -1, tau] += - \
            (flux_top[:, :, -1] - flux_top[:, :, -2]) / (0.5 * dzw[-1:])

        """
        Adam Bashforth time stepping
        """
        E_iw[:, :, :, taup1] += dt_tracer * ((1.5 + AB_eps) * dE_iw[:, :, :, tau]
                                             - (0.5 + AB_eps) * dE_iw[:, :, :, taum1])

    return E_iw, dE_iw, iw_diss


@veros_kernel(static_args=('pyom_compatibility_mode',))
def gofx2(x, pi, pyom_compatibility_mode):
    if pyom_compatibility_mode:
        x[x < 3.] = 3.
    else:
        x = np.maximum(3., x)
    c = 1. - (2. / pi) * np.arcsin(1. / x)
    return 2. / pi / c * 0.9 * x**(-2. / 3.) * (1 - np.exp(-x / 4.3))


@veros_kernel
def hofx1(x, pi):
    eps = np.finfo(x.dtype).eps  # prevent division by zero
    x = np.maximum(1. + eps, x)
    return (2. / pi) / (1. - (2. / pi) * np.arcsin(1. / x)) * (x - 1.) / (x + 1.)

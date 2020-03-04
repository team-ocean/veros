import math

import numpy as np

from veros.core import veros_routine, veros_kernel, numerics, utilities, isoneutral


@veros_kernel
def explicit_vert_friction(du_mix, dv_mix, K_diss_v, kappaM, u, v, tau, maskU,
                           maskV, area_v, area_t, dxt, dxu, dzt, dzw, nz):
    """
    explicit vertical friction
    dissipation is calculated and added to K_diss_v
    """
    diss = np.zeros_like(maskU)
    flux_top = np.zeros_like(maskU)

    """
    vertical friction of zonal momentum
    """
    fxa = 0.5 * (kappaM[1:-2, 1:-2, :-1] + kappaM[2:-1, 1:-2, :-1])
    flux_top[1:-2, 1:-2, :-1] = fxa * (u[1:-2, 1:-2, 1:, tau] - u[1:-2, 1:-2, :-1, tau]) \
        / dzw[np.newaxis, np.newaxis, :-1] * maskU[1:-2, 1:-2, 1:] * maskU[1:-2, 1:-2, :-1]
    flux_top[:, :, -1] = 0.0
    du_mix[:, :, 0] = flux_top[:, :, 0] / dzt[0] * maskU[:, :, 0]
    du_mix[:, :, 1:] = (flux_top[:, :, 1:] - flux_top[:, :, :-1]) / dzt[1:] * maskU[:, :, 1:]

    """
    diagnose dissipation by vertical friction of zonal momentum
    """
    diss[1:-2, 1:-2, :-1] = (u[1:-2, 1:-2, 1:, tau] - u[1:-2, 1:-2, :-1, tau]) \
        * flux_top[1:-2, 1:-2, :-1] / dzw[np.newaxis, np.newaxis, :-1]
    diss[:, :, nz - 1] = 0.0
    diss[...] = numerics.ugrid_to_tgrid(diss, dxt, dxu)
    K_diss_v += diss

    """
    vertical friction of meridional momentum
    """
    fxa = 0.5 * (kappaM[1:-2, 1:-2, :-1] + kappaM[1:-2, 2:-1, :-1])
    flux_top[1:-2, 1:-2, :-1] = fxa * (v[1:-2, 1:-2, 1:, tau] - v[1:-2, 1:-2, :-1, tau]) \
        / dzw[np.newaxis, np.newaxis, :-1] * maskV[1:-2, 1:-2, 1:] \
        * maskV[1:-2, 1:-2, :-1]
    flux_top[:, :, -1] = 0.0
    dv_mix[:, :, 1:] = (flux_top[:, :, 1:] - flux_top[:, :, :-1]) \
        / dzt[np.newaxis, np.newaxis, 1:] * maskV[:, :, 1:]
    dv_mix[:, :, 0] = flux_top[:, :, 0] / dzt[0] * maskV[:, :, 0]

    """
    diagnose dissipation by vertical friction of meridional momentum
    """
    diss[1:-2, 1:-2, :-1] = (v[1:-2, 1:-2, 1:, tau] - v[1:-2, 1:-2, :-1, tau]) \
        * flux_top[1:-2, 1:-2, :-1] / dzw[np.newaxis, np.newaxis, :-1]
    diss[:, :, -1] = 0.0
    diss[...] = numerics.vgrid_to_tgrid(diss, area_v, area_t)
    K_diss_v += diss

    return du_mix, dv_mix, K_diss_v


@veros_kernel
def implicit_vert_friction(du_mix, dv_mix, K_diss_v, u, v, kbot, kappaM, tau, taup1,
                           dxt, dxu, area_v, area_t, dt_mom, dzt, dzw, maskU, maskV):
    """
    vertical friction
    dissipation is calculated and added to K_diss_v
    """
    diss = np.zeros_like(maskU)
    a_tri = np.zeros_like(kbot[1:-2, 1:-2])
    b_tri = np.zeros_like(kbot[1:-2, 1:-2])
    c_tri = np.zeros_like(kbot[1:-2, 1:-2])
    d_tri = np.zeros_like(kbot[1:-2, 1:-2])
    delta = np.zeros_like(kbot[1:-2, 1:-2])
    flux_top = np.zeros_like(maskU)

    """
    implicit vertical friction of zonal momentum
    """
    kss = np.maximum(kbot[1:-2, 1:-2], kbot[2:-1, 1:-2]) - 1
    fxa = 0.5 * (kappaM[1:-2, 1:-2, :-1] + kappaM[2:-1, 1:-2, :-1])
    delta[:, :, :-1] = dt_mom / dzw[:-1] * fxa * \
        maskU[1:-2, 1:-2, 1:] * maskU[1:-2, 1:-2, :-1]
    a_tri[:, :, 1:] = -delta[:, :, :-1] / dzt[np.newaxis, np.newaxis, 1:]
    b_tri[:, :, 1:] = 1 + delta[:, :, :-1] / dzt[np.newaxis, np.newaxis, 1:]
    b_tri[:, :, 1:-1] += delta[:, :, 1:-1] / dzt[np.newaxis, np.newaxis, 1:-1]
    b_tri_edge = 1 + delta / dzt[np.newaxis, np.newaxis, :]
    c_tri[...] = -delta / dzt[np.newaxis, np.newaxis, :]
    d_tri[...] = u[1:-2, 1:-2, :, tau]
    res, mask = utilities.solve_implicit(kss, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)
    u[1:-2, 1:-2, :, taup1] = utilities.where(mask, res, u[1:-2, 1:-2, :, taup1])
    du_mix[1:-2, 1:-2] = (u[1:-2, 1:-2, :, taup1] - u[1:-2, 1:-2, :, tau]) / dt_mom

    """
    diagnose dissipation by vertical friction of zonal momentum
    """
    fxa = 0.5 * (kappaM[1:-2, 1:-2, :-1] + kappaM[2:-1, 1:-2, :-1])
    flux_top[1:-2, 1:-2, :-1] = fxa * (u[1:-2, 1:-2, 1:, taup1] - u[1:-2, 1:-2, :-1, taup1]) \
        / dzw[:-1] * maskU[1:-2, 1:-2, 1:] * maskU[1:-2, 1:-2, :-1]
    diss[1:-2, 1:-2, :-1] = (u[1:-2, 1:-2, 1:, tau] - u[1:-2, 1:-2, :-1, tau]) \
        * flux_top[1:-2, 1:-2, :-1] / dzw[:-1]
    diss[:, :, -1] = 0.0
    diss[...] = numerics.ugrid_to_tgrid(diss, dxt, dxu)
    K_diss_v += diss

    """
    implicit vertical friction of meridional momentum
    """
    kss = np.maximum(kbot[1:-2, 1:-2], kbot[1:-2, 2:-1]) - 1
    fxa = 0.5 * (kappaM[1:-2, 1:-2, :-1] + kappaM[1:-2, 2:-1, :-1])
    delta[:, :, :-1] = dt_mom / dzw[np.newaxis, np.newaxis, :-1] * \
        fxa * maskV[1:-2, 1:-2, 1:] * maskV[1:-2, 1:-2, :-1]
    a_tri[:, :, 1:] = -delta[:, :, :-1] / dzt[np.newaxis, np.newaxis, 1:]
    b_tri[:, :, 1:] = 1 + delta[:, :, :-1] / dzt[np.newaxis, np.newaxis, 1:]
    b_tri[:, :, 1:-1] += delta[:, :, 1:-1] / dzt[np.newaxis, np.newaxis, 1:-1]
    b_tri_edge = 1 + delta / dzt[np.newaxis, np.newaxis, :]
    c_tri[:, :, :-1] = -delta[:, :, :-1] / dzt[np.newaxis, np.newaxis, :-1]
    c_tri[:, :, -1] = 0.
    d_tri[...] = v[1:-2, 1:-2, :, tau]
    res, mask = utilities.solve_implicit(kss, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)
    v[1:-2, 1:-2, :, taup1] = utilities.where(mask, res, v[1:-2, 1:-2, :, taup1])
    dv_mix[1:-2, 1:-2] = (v[1:-2, 1:-2, :, taup1] - v[1:-2, 1:-2, :, tau]) / dt_mom

    """
    diagnose dissipation by vertical friction of meridional momentum
    """
    fxa = 0.5 * (kappaM[1:-2, 1:-2, :-1] + kappaM[1:-2, 2:-1, :-1])
    flux_top[1:-2, 1:-2, :-1] = fxa * (v[1:-2, 1:-2, 1:, taup1] - v[1:-2, 1:-2, :-1, taup1]) \
        / dzw[:-1] * maskV[1:-2, 1:-2, 1:] * maskV[1:-2, 1:-2, :-1]
    diss[1:-2, 1:-2, :-1] = (v[1:-2, 1:-2, 1:, tau] - v[1:-2, 1:-2, :-1, tau]) \
        * flux_top[1:-2, 1:-2, :-1] / dzw[:-1]
    diss[:, :, -1] = 0.0
    diss = numerics.vgrid_to_tgrid(diss, area_v, area_t)
    K_diss_v += diss

    return u, v, du_mix, dv_mix, K_diss_v


@veros_kernel(static_args=('enable_conserve_energy'))
def rayleigh_friction(du_mix, dv_mix, K_diss_bot, maskU, maskV, u, v, tau,
                      kbot, nz, dzw, dxt, dxu, r_ray, area_v, area_t,
                      enable_conserve_energy):
    """
    interior Rayleigh friction
    dissipation is calculated and added to K_diss_bot
    """
    du_mix[...] += -maskU * r_ray * u[..., tau]
    if enable_conserve_energy:
        diss = maskU * r_ray * u[..., tau]**2
        K_diss_bot[...] += numerics.calc_diss_u(diss, kbot, nz, dzw, dxt, dxu)
    dv_mix[...] += -maskV * r_ray * v[..., tau]

    if enable_conserve_energy:
        diss = maskV * r_ray * v[..., tau]**2
        K_diss_bot[...] += numerics.calc_diss_v(diss, kbot, nz, dzw, area_v, area_t)

    return du_mix, dv_mix, K_diss_bot


@veros_kernel(static_args=('enable_bottom_friction_var', 'enable_conserve_energy'))
def linear_bottom_friction(u, v, du_mix, dv_mix, K_diss_bot, kbot, nz, maskU, maskV,
                           r_bot, r_bot_var_u, r_bot_var_v, tau, dzw, dxt, dxu,
                           area_v, area_t, enable_bottom_friction_var,
                           enable_conserve_energy):
    """
    linear bottom friction
    dissipation is calculated and added to K_diss_bot
    """
    if enable_bottom_friction_var:
        """
        with spatially varying coefficient
        """
        k = np.maximum(kbot[1:-2, 2:-2], kbot[2:-1, 2:-2]) - 1
        mask = np.arange(nz) == k[:, :, np.newaxis]
        du_mix[1:-2, 2:-2] += -(maskU[1:-2, 2:-2] * r_bot_var_u[1:-2, 2:-2, np.newaxis]) \
            * u[1:-2, 2:-2, :, tau] * mask
        if enable_conserve_energy:
            diss = np.zeros_like(maskU)
            diss[1:-2, 2:-2] = maskU[1:-2, 2:-2] * r_bot_var_u[1:-2, 2:-2, np.newaxis] \
                * u[1:-2, 2:-2, :, tau]**2 * mask
            K_diss_bot[...] += numerics.calc_diss_u(diss, kbot, nz, dzw, dxt, dxu)

        k = np.maximum(kbot[2:-2, 2:-1], kbot[2:-2, 1:-2]) - 1
        mask = np.arange(nz) == k[:, :, np.newaxis]
        dv_mix[2:-2, 1:-2] += -(maskV[2:-2, 1:-2] * r_bot_var_v[2:-2, 1:-2, np.newaxis]) \
            * v[2:-2, 1:-2, :, tau] * mask
        if enable_conserve_energy:
            diss = np.zeros_like(maskV)
            diss[2:-2, 1:-2] = maskV[2:-2, 1:-2] * r_bot_var_v[2:-2, 1:-2, np.newaxis] \
                * v[2:-2, 1:-2, :, tau]**2 * mask
            K_diss_bot[...] += numerics.calc_diss_v(diss, kbot, nz, dzw, area_v, area_t)
    else:
        """
        with constant coefficient
        """
        k = np.maximum(kbot[1:-2, 2:-2], kbot[2:-1, 2:-2]) - 1
        mask = np.arange(nz) == k[:, :, np.newaxis]
        du_mix[1:-2, 2:-2] += -maskU[1:-2, 2:-2] * r_bot * u[1:-2, 2:-2, :, tau] * mask
        if enable_conserve_energy:
            diss = np.zeros_like(maskU)
            diss[1:-2, 2:-2] = maskU[1:-2, 2:-2] * r_bot * u[1:-2, 2:-2, :, tau]**2 * mask
            K_diss_bot[...] += numerics.calc_diss_u(diss, kbot, nz, dzw, dxt, dxu)

        k = np.maximum(kbot[2:-2, 2:-1], kbot[2:-2, 1:-2]) - 1
        mask = np.arange(nz) == k[:, :, np.newaxis]
        dv_mix[2:-2, 1:-2] += -maskV[2:-2, 1:-2] * r_bot * v[2:-2, 1:-2, :, tau] * mask
        if enable_conserve_energy:
            diss = np.zeros_like(maskV)
            diss[2:-2, 1:-2] = maskV[2:-2, 1:-2] * r_bot * v[2:-2, 1:-2, :, tau]**2 * mask
            K_diss_bot[...] += numerics.calc_diss_v(diss, kbot, nz, dzw, area_v, area_t)

    return du_mix, dv_mix, K_diss_bot


@veros_kernel(static_args=('enable_conserve_energy'))
def quadratic_bottom_friction(du_mix, dv_mix, K_diss_bot, u, v, r_quad_bot, dzt,
                              dzw, dxt, dxu, kbot, maskU, maskV, nz, tau,
                              area_v, area_t, enable_conserve_energy):
    """
    quadratic bottom friction
    dissipation is calculated and added to K_diss_bot
    """
    # we might want to account for EKE in the drag, also a tidal residual
    k = np.maximum(kbot[1:-2, 2:-2], kbot[2:-1, 2:-2]) - 1
    mask = k[..., np.newaxis] == np.arange(nz)[np.newaxis, np.newaxis, :]
    fxa = maskV[1:-2, 2:-2, :] * v[1:-2, 2:-2, :, tau]**2 \
        + maskV[1:-2, 1:-3, :] * v[1:-2, 1:-3, :, tau]**2 \
        + maskV[2:-1, 2:-2, :] * v[2:-1, 2:-2, :, tau]**2 \
        + maskV[2:-1, 1:-3, :] * v[2:-1, 1:-3, :, tau]**2
    fxa = np.sqrt(u[1:-2, 2:-2, :, tau]**2 + 0.25 * fxa)
    aloc = maskU[1:-2, 2:-2, :] * r_quad_bot * u[1:-2, 2:-2, :, tau] \
        * fxa / dzt[np.newaxis, np.newaxis, :] * mask
    du_mix[1:-2, 2:-2, :] += -aloc

    if enable_conserve_energy:
        diss = np.zeros_like(maskU)
        diss[1:-2, 2:-2, :] = aloc * u[1:-2, 2:-2, :, tau]
        K_diss_bot[...] += numerics.calc_diss_u(diss, kbot, nz, dzw, dxt, dxu)

    k = np.maximum(kbot[2:-2, 1:-2], kbot[2:-2, 2:-1]) - 1
    mask = k[..., np.newaxis] == np.arange(nz)[np.newaxis, np.newaxis, :]
    fxa = maskU[2:-2, 1:-2, :] * u[2:-2, 1:-2, :, tau]**2 \
        + maskU[1:-3, 1:-2, :] * u[1:-3, 1:-2, :, tau]**2 \
        + maskU[2:-2, 2:-1, :] * u[2:-2, 2:-1, :, tau]**2 \
        + maskU[1:-3, 2:-1, :] * u[1:-3, 2:-1, :, tau]**2
    fxa = np.sqrt(v[2:-2, 1:-2, :, tau]**2 + 0.25 * fxa)
    aloc = maskV[2:-2, 1:-2, :] * r_quad_bot * v[2:-2, 1:-2, :, tau] \
        * fxa / dzt[np.newaxis, np.newaxis, :] * mask
    dv_mix[2:-2, 1:-2, :] += -aloc

    if enable_conserve_energy:
        diss = np.zeros_like(maskV)
        diss[2:-2, 1:-2, :] = aloc * v[2:-2, 1:-2, :, tau]
        K_diss_bot[...] += numerics.calc_diss_v(diss, kbot, nz, dzw, area_v, area_t)

    return du_mix, dv_mix, K_diss_bot


@veros_kernel(static_args=('enable_hor_friction_cos_scaling', 'enable_conserve_energy',
                           'hor_friction_cosPower', 'enable_noslip_lateral')
)
def harmonic_friction(du_mix, dv_mix, K_diss_h, cost, cosu, A_h, u, v, tau,
                      dxt, dxu, dyt, dyu, dzw, maskU, maskV, kbot, nz, area_v, area_t,
                      enable_hor_friction_cos_scaling, enable_noslip_lateral,
                      hor_friction_cosPower, enable_conserve_energy):
    """
    horizontal harmonic friction
    dissipation is calculated and added to K_diss_h
    """
    diss = np.zeros_like(maskU)
    flux_east = np.zeros_like(maskU)
    flux_north = np.zeros_like(maskV)

    """
    Zonal velocity
    """
    if enable_hor_friction_cos_scaling:
        fxa = cost**hor_friction_cosPower
        flux_east[:-1] = A_h * fxa[np.newaxis, :, np.newaxis] * (u[1:, :, :, tau] - u[:-1, :, :, tau]) \
            / (cost * dxt[1:, np.newaxis])[:, :, np.newaxis] * maskU[1:] * maskU[:-1]
        fxa = cosu**hor_friction_cosPower
        flux_north[:, :-1] = A_h * fxa[np.newaxis, :-1, np.newaxis] * (u[:, 1:, :, tau] - u[:, :-1, :, tau]) \
            / dyu[np.newaxis, :-1, np.newaxis] * maskU[:, 1:] * maskU[:, :-1] * cosu[np.newaxis, :-1, np.newaxis]
        if enable_noslip_lateral:
            flux_north[:, :-1] += 2 * A_h * fxa[np.newaxis, :-1, np.newaxis] * (u[:, 1:, :, tau]) \
                / dyu[np.newaxis, :-1, np.newaxis] * maskU[:, 1:] * (1 - maskU[:, :-1]) * cosu[np.newaxis, :-1, np.newaxis]\
                - 2 * A_h * fxa[np.newaxis, :-1, np.newaxis] * (u[:, :-1, :, tau]) \
                / dyu[np.newaxis, :-1, np.newaxis] * (1 - maskU[:, 1:]) * maskU[:, :-1] * cosu[np.newaxis, :-1, np.newaxis]
    else:
        flux_east[:-1, :, :] = A_h * (u[1:, :, :, tau] - u[:-1, :, :, tau]) \
            / (cost * dxt[1:, np.newaxis])[:, :, np.newaxis] * maskU[1:] * maskU[:-1]
        flux_north[:, :-1, :] = A_h * (u[:, 1:, :, tau] - u[:, :-1, :, tau]) \
            / dyu[np.newaxis, :-1, np.newaxis] * maskU[:, 1:] * maskU[:, :-1] * cosu[np.newaxis, :-1, np.newaxis]
        if enable_noslip_lateral:
            flux_north[:, :-1] += 2 * A_h * u[:, 1:, :, tau] / dyu[np.newaxis, :-1, np.newaxis] \
                * maskU[:, 1:] * (1 - maskU[:, :-1]) * cosu[np.newaxis, :-1, np.newaxis]\
                - 2 * A_h * u[:, :-1, :, tau] / dyu[np.newaxis, :-1, np.newaxis] \
                * (1 - maskU[:, 1:]) * maskU[:, :-1] * cosu[np.newaxis, :-1, np.newaxis]

    flux_east[-1, :, :] = 0.
    flux_north[:, -1, :] = 0.

    """
    update tendency
    """
    du_mix[2:-2, 2:-2, :] += maskU[2:-2, 2:-2] * ((flux_east[2:-2, 2:-2] - flux_east[1:-3, 2:-2])
                                                  / (cost[2:-2] * dxu[2:-2, np.newaxis])[:, :, np.newaxis]
                                                  + (flux_north[2:-2, 2:-2] - flux_north[2:-2, 1:-3])
                                                  / (cost[2:-2] * dyt[2:-2])[np.newaxis, :, np.newaxis])

    if enable_conserve_energy:
        """
        diagnose dissipation by lateral friction
        """
        diss[1:-2, 2:-2] = 0.5 * ((u[2:-1, 2:-2, :, tau] - u[1:-2, 2:-2, :, tau]) * flux_east[1:-2, 2:-2]
                                  + (u[1:-2, 2:-2, :, tau] - u[:-3, 2:-2, :, tau]) * flux_east[:-3, 2:-2]) \
            / (cost[2:-2] * dxu[1:-2, np.newaxis])[:, :, np.newaxis]\
            + 0.5 * ((u[1:-2, 3:-1, :, tau] - u[1:-2, 2:-2, :, tau]) * flux_north[1:-2, 2:-2]
                     + (u[1:-2, 2:-2, :, tau] - u[1:-2, 1:-3, :, tau]) * flux_north[1:-2, 1:-3]) \
            / (cost[2:-2] * dyt[2:-2])[np.newaxis, :, np.newaxis]
        K_diss_h[...] = 0.
        K_diss_h[...] += numerics.calc_diss_u(diss, kbot, nz, dzw, dxt, dxu)

    """
    Meridional velocity
    """
    if enable_hor_friction_cos_scaling:
        flux_east[:-1] = A_h * cosu[np.newaxis, :, np.newaxis] ** hor_friction_cosPower \
            * (v[1:, :, :, tau] - v[:-1, :, :, tau]) \
            / (cosu * dxu[:-1, np.newaxis])[:, :, np.newaxis] * maskV[1:] * maskV[:-1]
        if enable_noslip_lateral:
            flux_east[:-1] += 2 * A_h * fxa[np.newaxis, :, np.newaxis] * v[1:, :, :, tau] \
                / (cosu * dxu[:-1, np.newaxis])[:, :, np.newaxis] * maskV[1:] * (1 - maskV[:-1]) \
                - 2 * A_h * fxa[np.newaxis, :, np.newaxis] * v[:-1, :, :, tau] \
                / (cosu * dxu[:-1, np.newaxis])[:, :, np.newaxis] * (1 - maskV[1:]) * maskV[:-1]

        flux_north[:, :-1] = A_h * cost[np.newaxis, 1:, np.newaxis] ** hor_friction_cosPower \
            * (v[:, 1:, :, tau] - v[:, :-1, :, tau]) \
            / dyt[np.newaxis, 1:, np.newaxis] * cost[np.newaxis, 1:, np.newaxis] * maskV[:, :-1] * maskV[:, 1:]
    else:
        flux_east[:-1] = A_h * (v[1:, :, :, tau] - v[:-1, :, :, tau]) \
            / (cosu * dxu[:-1, np.newaxis])[:, :, np.newaxis] * maskV[1:] * maskV[:-1]
        if enable_noslip_lateral:
            flux_east[:-1] += 2 * A_h * v[1:, :, :, tau] / (cosu * dxu[:-1, np.newaxis])[:, :, np.newaxis] \
                * maskV[1:] * (1 - maskV[:-1]) \
                - 2 * A_h * v[:-1, :, :, tau] / (cosu * dxu[:-1, np.newaxis])[:, :, np.newaxis] \
                * (1 - maskV[1:]) * maskV[:-1]
        flux_north[:, :-1] = A_h * (v[:, 1:, :, tau] - v[:, :-1, :, tau]) \
            / dyt[np.newaxis, 1:, np.newaxis] * cost[np.newaxis, 1:, np.newaxis] * maskV[:, :-1] * maskV[:, 1:]
    flux_east[-1, :, :] = 0.
    flux_north[:, -1, :] = 0.

    """
    update tendency
    """
    dv_mix[2:-2, 2:-2] += maskV[2:-2, 2:-2] * ((flux_east[2:-2, 2:-2] - flux_east[1:-3, 2:-2])
                                               / (cosu[2:-2] * dxt[2:-2, np.newaxis])[:, :, np.newaxis]
                                               + (flux_north[2:-2, 2:-2] - flux_north[2:-2, 1:-3])
                                               / (dyu[2:-2] * cosu[2:-2])[np.newaxis, :, np.newaxis])

    if enable_conserve_energy:
        """
        diagnose dissipation by lateral friction
        """
        diss[2:-2, 1:-2] = 0.5 * ((v[3:-1, 1:-2, :, tau] - v[2:-2, 1:-2, :, tau]) * flux_east[2:-2, 1:-2]
                                  + (v[2:-2, 1:-2, :, tau] - v[1:-3, 1:-2, :, tau]) * flux_east[1:-3, 1:-2]) \
            / (cosu[1:-2] * dxt[2:-2, np.newaxis])[:, :, np.newaxis] \
            + 0.5 * ((v[2:-2, 2:-1, :, tau] - v[2:-2, 1:-2, :, tau]) * flux_north[2:-2, 1:-2]
                     + (v[2:-2, 1:-2, :, tau] - v[2:-2, :-3, :, tau]) * flux_north[2:-2, :-3]) \
            / (cosu[1:-2] * dyu[1:-2])[np.newaxis, :, np.newaxis]
        K_diss_h[...] += numerics.calc_diss_v(diss, kbot, nz, dzw, area_v, area_t)

    return du_mix, dv_mix, K_diss_h


@veros_kernel(static_args=('enable_noslip_lateral', 'enable_conserve_energy'))
def biharmonic_friction(du_mix, dv_mix, K_diss_h, A_hbi, u, v, tau, area_v, area_t,
                        cost, cosu, dxt, dxu, dyt, dyu, dzw, maskU, maskV, enable_cyclic_x,
                        kbot, nz, enable_noslip_lateral, enable_conserve_energy):
    """
    horizontal biharmonic friction
    dissipation is calculated and added to K_diss_h
    """
    flux_east = np.zeros_like(maskU)
    flux_north = np.zeros_like(maskV)
    fxa = math.sqrt(abs(A_hbi))

    """
    Zonal velocity
    """
    flux_east[:-1, :, :] = fxa * (u[1:, :, :, tau] - u[:-1, :, :, tau]) \
        / (cost[np.newaxis, :, np.newaxis] * dxt[1:, np.newaxis, np.newaxis]) \
        * maskU[1:, :, :] * maskU[:-1, :, :]
    flux_north[:, :-1, :] = fxa * (u[:, 1:, :, tau] - u[:, :-1, :, tau]) \
        / dyu[np.newaxis, :-1, np.newaxis] * maskU[:, 1:, :] \
        * maskU[:, :-1, :] * cosu[np.newaxis, :-1, np.newaxis]
    if enable_noslip_lateral:
        flux_north[:, :-1] += 2 * fxa * u[:, 1:, :, tau] / dyu[np.newaxis, :-1, np.newaxis] \
            * maskU[:, 1:] * (1 - maskU[:, :-1]) * cosu[np.newaxis, :-1, np.newaxis]\
            - 2 * fxa * u[:, :-1, :, tau] / dyu[np.newaxis, :-1, np.newaxis] \
            * (1 - maskU[:, 1:]) * maskU[:, :-1] * cosu[np.newaxis, :-1, np.newaxis]
    flux_east[-1, :, :] = 0.
    flux_north[:, -1, :] = 0.

    del2 = np.zeros_like(maskU)
    del2[1:, 1:, :] = (flux_east[1:, 1:, :] - flux_east[:-1, 1:, :]) \
        / (cost[np.newaxis, 1:, np.newaxis] * dxu[1:, np.newaxis, np.newaxis]) \
        + (flux_north[1:, 1:, :] - flux_north[1:, :-1, :]) \
        / (cost[np.newaxis, 1:, np.newaxis] * dyt[np.newaxis, 1:, np.newaxis])

    flux_east[:-1, :, :] = fxa * (del2[1:, :, :] - del2[:-1, :, :]) \
        / (cost[np.newaxis, :, np.newaxis] * dxt[1:, np.newaxis, np.newaxis]) \
        * maskU[1:, :, :] * maskU[:-1, :, :]
    flux_north[:, :-1, :] = fxa * (del2[:, 1:, :] - del2[:, :-1, :]) \
        / dyu[np.newaxis, :-1, np.newaxis] * maskU[:, 1:, :] \
        * maskU[:, :-1, :] * cosu[np.newaxis, :-1, np.newaxis]
    if enable_noslip_lateral:
        flux_north[:, :-1, :] += 2 * fxa * del2[:, 1:, :] / dyu[np.newaxis, :-1, np.newaxis] \
            * maskU[:, 1:, :] * (1 - maskU[:, :-1, :]) * cosu[np.newaxis, :-1, np.newaxis] \
            - 2 * fxa * del2[:, :-1, :] / dyu[np.newaxis, :-1, np.newaxis] \
            * (1 - maskU[:, 1:, :]) * maskU[:, :-1, :] * cosu[np.newaxis, :-1, np.newaxis]
    flux_east[-1, :, :] = 0.
    flux_north[:, -1, :] = 0.

    """
    update tendency
    """
    du_mix[2:-2, 2:-2, :] += -maskU[2:-2, 2:-2, :] * ((flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
                                                      / (cost[np.newaxis, 2:-2, np.newaxis] * dxu[2:-2, np.newaxis, np.newaxis])
                                                      + (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
                                                      / (cost[np.newaxis, 2:-2, np.newaxis] * dyt[np.newaxis, 2:-2, np.newaxis]))
    if enable_conserve_energy:
        """
        diagnose dissipation by lateral friction
        """
        utilities.enforce_boundaries(flux_east, enable_cyclic_x)
        utilities.enforce_boundaries(flux_north, enable_cyclic_x)
        diss = np.zeros_like(maskU)
        diss[1:-2, 2:-2, :] = -0.5 * ((u[2:-1, 2:-2, :, tau] - u[1:-2, 2:-2, :, tau]) * flux_east[1:-2, 2:-2, :]
                                      + (u[1:-2, 2:-2, :, tau] - u[:-3, 2:-2, :, tau]) * flux_east[:-3, 2:-2, :]) \
            / (cost[np.newaxis, 2:-2, np.newaxis] * dxu[1:-2, np.newaxis, np.newaxis])  \
            - 0.5 * ((u[1:-2, 3:-1, :, tau] - u[1:-2, 2:-2, :, tau]) * flux_north[1:-2, 2:-2, :]
                     + (u[1:-2, 2:-2, :, tau] - u[1:-2, 1:-3, :, tau]) * flux_north[1:-2, 1:-3, :]) \
            / (cost[np.newaxis, 2:-2, np.newaxis] * dyt[np.newaxis, 2:-2, np.newaxis])
        K_diss_h[...] = 0.
        K_diss_h[...] += numerics.calc_diss_u(diss, kbot, nz, dzw, dxt, dxu)

    """
    Meridional velocity
    """
    flux_east[:-1, :, :] = fxa * (v[1:, :, :, tau] - v[:-1, :, :, tau]) \
        / (cosu[np.newaxis, :, np.newaxis] * dxu[:-1, np.newaxis, np.newaxis]) \
        * maskV[1:, :, :] * maskV[:-1, :, :]
    if enable_noslip_lateral:
        flux_east[:-1, :, :] += 2 * fxa * v[1:, :, :, tau] / (cosu[np.newaxis, :, np.newaxis] * dxu[:-1, np.newaxis, np.newaxis]) \
            * maskV[1:, :, :] * (1 - maskV[:-1, :, :]) \
            - 2 * fxa * v[:-1, :, :, tau] / (cosu[np.newaxis, :, np.newaxis] * dxu[:-1, np.newaxis, np.newaxis]) \
            * (1 - maskV[1:, :, :]) * maskV[:-1, :, :]
    flux_north[:, :-1, :] = fxa * (v[:, 1:, :, tau] - v[:, :-1, :, tau]) \
        / dyt[np.newaxis, 1:, np.newaxis] * cost[np.newaxis, 1:, np.newaxis] \
        * maskV[:, :-1, :] * maskV[:, 1:, :]
    flux_east[-1, :, :] = 0.
    flux_north[:, -1, :] = 0.

    del2[1:, 1:, :] = (flux_east[1:, 1:, :] - flux_east[:-1, 1:, :]) \
        / (cosu[np.newaxis, 1:, np.newaxis] * dxt[1:, np.newaxis, np.newaxis])  \
        + (flux_north[1:, 1:, :] - flux_north[1:, :-1, :]) \
        / (dyu[np.newaxis, 1:, np.newaxis] * cosu[np.newaxis, 1:, np.newaxis])

    flux_east[:-1, :, :] = fxa * (del2[1:, :, :] - del2[:-1, :, :]) \
        / (cosu[np.newaxis, :, np.newaxis] * dxu[:-1, np.newaxis, np.newaxis]) \
        * maskV[1:, :, :] * maskV[:-1, :, :]
    if enable_noslip_lateral:
        flux_east[:-1, :, :] += 2 * fxa * del2[1:, :, :] / (cosu[np.newaxis, :, np.newaxis] * dxu[:-1, np.newaxis, np.newaxis]) \
            * maskV[1:, :, :] * (1 - maskV[:-1, :, :]) \
            - 2 * fxa * del2[:-1, :, :] / (cosu[np.newaxis, :, np.newaxis] * dxu[:-1, np.newaxis, np.newaxis]) \
            * (1 - maskV[1:, :, :]) * maskV[:-1, :, :]
    flux_north[:, :-1, :] = fxa * (del2[:, 1:, :] - del2[:, :-1, :]) \
        / dyt[np.newaxis, 1:, np.newaxis] * cost[np.newaxis, 1:, np.newaxis] \
        * maskV[:, :-1, :] * maskV[:, 1:, :]
    flux_east[-1, :, :] = 0.
    flux_north[:, -1, :] = 0.

    """
    update tendency
    """
    dv_mix[2:-2, 2:-2, :] += -maskV[2:-2, 2:-2, :] * ((flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
                                                      / (cosu[np.newaxis, 2:-2, np.newaxis] * dxt[2:-2, np.newaxis, np.newaxis])
                                                      + (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
                                                      / (dyu[np.newaxis, 2:-2, np.newaxis] * cosu[np.newaxis, 2:-2, np.newaxis]))

    if enable_conserve_energy:
        """
        diagnose dissipation by lateral friction
        """
        utilities.enforce_boundaries(flux_east, enable_cyclic_x)
        utilities.enforce_boundaries(flux_north, enable_cyclic_x)
        diss[2:-2, 1:-2, :] = -0.5 * ((v[3:-1, 1:-2, :, tau] - v[2:-2, 1:-2, :, tau]) * flux_east[2:-2, 1:-2, :]
                                      + (v[2:-2, 1:-2, :, tau] - v[1:-3, 1:-2, :, tau]) * flux_east[1:-3, 1:-2, :]) \
            / (cosu[np.newaxis, 1:-2, np.newaxis] * dxt[2:-2, np.newaxis, np.newaxis]) \
            - 0.5 * ((v[2:-2, 2:-1, :, tau] - v[2:-2, 1:-2, :, tau]) * flux_north[2:-2, 1:-2, :]
                     + (v[2:-2, 1:-2, :, tau] - v[2:-2, :-3, :, tau]) * flux_north[2:-2, :-3, :]) \
            / (cosu[np.newaxis, 1:-2, np.newaxis] * dyu[np.newaxis, 1:-2, np.newaxis])
        K_diss_h[...] += numerics.calc_diss_v(diss, kbot, nz, dzw, area_v, area_t)

    return du_mix, dv_mix, K_diss_h


@veros_kernel(static_args=('enable_conserve_energy'))
def momentum_sources(du_mix, dv_mix, K_diss_bot, u, v, u_source, v_source, area_v, area_t,
                     tau, maskU, maskV, kbot, nz, dzw, dxt, dxu, enable_conserve_energy):
    """
    other momentum sources
    dissipation is calculated and added to K_diss_bot
    """
    du_mix[...] += maskU * u_source
    if enable_conserve_energy:
        diss = -maskU * u[..., tau] * u_source
        K_diss_bot[...] += numerics.calc_diss_u(diss, kbot, nz, dzw, dxt, dxu)
    dv_mix[...] += maskV * v_source
    if enable_conserve_energy:
        diss = -maskV * v[..., tau] * v_source
        K_diss_bot[...] += numerics.calc_diss_v(diss, kbot, nz, dzw, area_v, area_t)

    return du_mix, dv_mix, K_diss_bot


@veros_routine(
    inputs=(
        'u', 'v',
        'du_mix', 'dv_mix',
        'K_diss_v', 'K_diss_h', 'K_diss_bot',
        'u_source', 'v_source',
        'area_v', 'area_t',
        'maskU', 'maskV',
        'dxt', 'dxu', 'dyt', 'dyu', 'dzt', 'dzw',
        'cost', 'cosu', 'kbot',
        'kappaM', 'kappa_gm', 'tau', 'taup1',
        'r_bot_var_u', 'r_bot_var_v',
    ),
    outputs=(
        'u', 'v',
        'du_mix', 'dv_mix', 'K_diss_gm',
        'K_diss_h', 'K_diss_v', 'K_diss_bot',
    ),
    settings=(
        'dt_mom', 'nz', 'A_h', 'A_hbi',
        'r_ray', 'hor_friction_cosPower',
        'dt_mom', 'r_bot', 'r_quad_bot',
        'enable_cyclic_x',
        'enable_bottom_friction_var',
        'enable_implicit_vert_friction',
        'enable_explicit_vert_friction',
        'enable_TEM_friction',
        'enable_hor_friction_cos_scaling',
        'enable_hor_friction',
        'enable_biharmonic_friction',
        'enable_noslip_lateral',
        'enable_ray_friction',
        'enable_bottom_friction',
        'enable_quadratic_bottom_friction',
        'enable_momentum_sources',
        'enable_conserve_energy',
    )
)
def friction(vs):
    """
    vertical friction
    """
    vs.K_diss_v[...] = 0.0
    if enable_implicit_vert_friction:
        u, v, du_mix, dv_mix, K_diss_v = run_kernel(implicit_vert_friction, vs)
    if enable_explicit_vert_friction:
        du_mix, dv_mix, K_diss_v = run_kernel(explicit_vert_friction, vs)

    """
    TEM formalism for eddy-driven velocity
    """
    if enable_TEM_friction:
        du_mix, dv_mix, K_diss_gm, u, v = run_kernel(isoneutral.isoneutral_friction, vs,
                                                     du_mix=du_mix, dv_mix=dv_mix)

    """
    horizontal friction
    """
    if enable_hor_friction:
        du_mix, dv_mix, K_diss_h = run_kernel(harmonic_friction, vs,
                                              du_mix=du_mix, dv_mix=dv_mix)
    if enable_biharmonic_friction:
        du_mix, dv_mix, K_diss_h = run_kernel(biharmonic_friction, vs,
                                              du_mix=du_mix, dv_mix=dv_mix)

    """
    Rayleigh and bottom friction
    """
    vs.K_diss_bot[...] = 0.0
    if enable_ray_friction:
        du_mix, dv_mix, K_diss_bot = run_kernel(rayleigh_friction, vs,
                                                du_mix=du_mix, dv_mix=dv_mix)
    if enable_bottom_friction:
        du_mix, dv_mix, K_diss_bot = run_kernel(linear_bottom_friction, vs,
                                                du_mix=du_mix, dv_mix=dv_mix)
    if enable_quadratic_bottom_friction:
        du_mix, dv_mix, K_diss_bot = run_kernel(quadratic_bottom_friction, vs,
                                                du_mix=du_mix, dv_mix=dv_mix)

    """
    add user defined forcing
    """
    if enable_momentum_sources:
        du_mix, dv_mix, K_diss_bot = run_kernel(momentum_sources, vs,
                                                du_mix=du_mix, dv_mix=dv_mix,
                                                K_diss_bot=K_diss_bot)

    return dict(u=u, v=v,
                du_mix=du_mix,
                dv_mix=dv_mix,
                K_diss_h=K_diss_h,
                K_diss_v=K_diss_v,
                K_diss_gm=K_diss_gm,
                K_diss_bot=K_diss_bot
    )

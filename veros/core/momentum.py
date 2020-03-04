import numpy as np

from veros.core import veros_routine, veros_kernel, run_kernel, friction, streamfunction


@veros_kernel(static_args=('coord_degree'))
def tend_coriolisf(du, dv, coriolis_t, maskU, maskV, dxt, dxu, dyt, dyu,
                   cost, cosu, tau, tantr, coord_degree):
    """
    time tendency due to Coriolis force
    """

    du_cor = np.zeros_like(maskU)
    dv_cor = np.zeros_like(maskV)

    du_cor[2:-2, 2:-2] = maskU[2:-2, 2:-2] \
        * (coriolis_t[2:-2, 2:-2, np.newaxis] * (v[2:-2, 2:-2, :, tau] + v[2:-2, 1:-3, :, tau])
           * dxt[2:-2, np.newaxis, np.newaxis] / dxu[2:-2, np.newaxis, np.newaxis]
            + coriolis_t[3:-1, 2:-2, np.newaxis] *
           (v[3:-1, 2:-2, :, tau] + v[3:-1, 1:-3, :, tau])
           * dxt[3:-1, np.newaxis, np.newaxis] / dxu[2:-2, np.newaxis, np.newaxis]) * 0.25
    dv_cor[2:-2, 2:-2] = -maskV[2:-2, 2:-2] \
        * (coriolis_t[2:-2, 2:-2, np.newaxis] * (u[1:-3, 2:-2, :, tau] + u[2:-2, 2:-2, :, tau])
           * dyt[np.newaxis, 2:-2, np.newaxis] * cost[np.newaxis, 2:-2, np.newaxis]
           / (dyu[np.newaxis, 2:-2, np.newaxis] * cosu[np.newaxis, 2:-2, np.newaxis])
           + coriolis_t[2:-2, 3:-1, np.newaxis]
           * (u[1:-3, 3:-1, :, tau] + u[2:-2, 3:-1, :, tau])
           * dyt[np.newaxis, 3:-1, np.newaxis] * cost[np.newaxis, 3:-1, np.newaxis]
           / (dyu[np.newaxis, 2:-2, np.newaxis] * cosu[np.newaxis, 2:-2, np.newaxis])) * 0.25

    """
    time tendency due to metric terms
    """
    if coord_degree:
        du_cor[2:-2, 2:-2] += maskU[2:-2, 2:-2] * 0.125 * tantr[np.newaxis, 2:-2, np.newaxis] \
            * ((u[2:-2, 2:-2, :, tau] + u[1:-3, 2:-2, :, tau])
               * (v[2:-2, 2:-2, :, tau] + v[2:-2, 1:-3, :, tau])
               * dxt[2:-2, np.newaxis, np.newaxis] / dxu[2:-2, np.newaxis, np.newaxis]
               + (u[3:-1, 2:-2, :, tau] + u[2:-2, 2:-2, :, tau])
               * (v[3:-1, 2:-2, :, tau] + v[3:-1, 1:-3, :, tau])
               * dxt[3:-1, np.newaxis, np.newaxis] / dxu[2:-2, np.newaxis, np.newaxis])
        dv_cor[2:-2, 2:-2] += -maskV[2:-2, 2:-2] * 0.125 \
            * (tantr[np.newaxis, 2:-2, np.newaxis] * (u[2:-2, 2:-2, :, tau] + u[1:-3, 2:-2, :, tau])**2
               * dyt[np.newaxis, 2:-2, np.newaxis] * cost[np.newaxis, 2:-2, np.newaxis]
               / (dyu[np.newaxis, 2:-2, np.newaxis] * cosu[np.newaxis, 2:-2, np.newaxis])
               + tantr[np.newaxis, 3:-1, np.newaxis]
               * (u[2:-2, 3:-1, :, tau] + u[1:-3, 3:-1, :, tau])**2
               * dyt[np.newaxis, 3:-1, np.newaxis] * cost[np.newaxis, 3:-1, np.newaxis]
               / (dyu[np.newaxis, 2:-2, np.newaxis] * cosu[np.newaxis, 2:-2, np.newaxis]))

    """
    transfer to time tendencies
    """
    du[2:-2, 2:-2, :, tau] = du_cor[2:-2, 2:-2]
    dv[2:-2, 2:-2, :, tau] = dv_cor[2:-2, 2:-2]

    return du, dv


@veros_kernel(static_args=('pyom_compatibility_mode'))
def tend_tauxyf(tend_u, tend_v, maskU, maskV, dzt, tau,
                surface_taux, surface_tauy, rho_0, pyom_compatibility_mode):
    """
    wind stress forcing
    """
    if pyom_compatibility_mode:
        tend_u[2:-2, 2:-2, -1, tau] += maskU[2:-2, 2:-2, -1] * surface_taux[2:-2, 2:-2] / dzt[-1]
        tend_v[2:-2, 2:-2, -1, tau] += maskV[2:-2, 2:-2, -1] * surface_tauy[2:-2, 2:-2] / dzt[-1]
    else:
        tend_u[2:-2, 2:-2, -1, tau] += maskU[2:-2, 2:-2, -1] * surface_taux[2:-2, 2:-2] / dzt[-1] / rho_0
        tend_v[2:-2, 2:-2, -1, tau] += maskV[2:-2, 2:-2, -1] * surface_tauy[2:-2, 2:-2] / dzt[-1] / rho_0

    return tend_u, tend_v


@veros_kernel
def momentum_advection(tend_u, tend_v, u, v, w, maskU, maskV, maskW,
                       area_t, area_u, area_v, dxt, dyt, dzt, cosu, tau):
    """
    Advection of momentum with second order which is energy conserving
    """

    """
    Code from MITgcm
    """

    utr = u[..., tau] * maskU * dyt[np.newaxis, :, np.newaxis] \
        * dzt[np.newaxis, np.newaxis, :]
    vtr = dzt[np.newaxis, np.newaxis, :] * cosu[np.newaxis, :, np.newaxis] \
        * dxt[:, np.newaxis, np.newaxis] * v[..., tau] * maskV
    wtr = w[..., tau] * maskW * area_t[:, :, np.newaxis]

    """
    for zonal momentum
    """
    du_adv = np.zeros_like(maskU)
    dv_adv = np.zeros_like(maskV)
    flux_east = np.zeros_like(maskU)
    flux_north = np.zeros_like(maskV)
    flux_top = np.zeros_like(maskW)

    flux_top[...] = 0.
    flux_east[1:-2, 2:-2] = 0.25 * (u[1:-2, 2:-2, :, tau]
                                    + u[2:-1, 2:-2, :, tau]) \
                                 * (utr[2:-1, 2:-2] + utr[1:-2, 2:-2])
    flux_north[2:-2, 1:-2] = 0.25 * (u[2:-2, 1:-2, :, tau]
                                     + u[2:-2, 2:-1, :, tau]) \
                                  * (vtr[3:-1, 1:-2] + vtr[2:-2, 1:-2])
    flux_top[2:-2, 2:-2, :-1] = 0.25 * (u[2:-2, 2:-2, 1:, tau]
                                        + u[2:-2, 2:-2, :-1, tau]) \
                                     * (wtr[2:-2, 2:-2, :-1] + wtr[3:-1, 2:-2, :-1])
    du_adv[2:-2, 2:-2] = -maskU[2:-2, 2:-2] * (flux_east[2:-2, 2:-2] - flux_east[1:-3, 2:-2]
                                               + flux_north[2:-2, 2:-2] - flux_north[2:-2, 1:-3]) \
                         / (dzt[np.newaxis, np.newaxis, :] * area_u[2:-2, 2:-2, np.newaxis])

    tmp = np.zeros_like(maskU)
    tmp = -maskU / (dzt * area_u[:, :, np.newaxis])
    du_adv += tmp * flux_top
    du_adv[:, :, 1:] += tmp[:, :, 1:] * -flux_top[:, :, :-1]

    """
    for meridional momentum
    """
    flux_top[...] = 0.
    flux_east[1:-2, 2:-2] = 0.25 * (v[1:-2, 2:-2, :, tau]
                                    + v[2:-1, 2:-2, :, tau]) * (utr[1:-2, 3:-1] + utr[1:-2, 2:-2])
    flux_north[2:-2, 1:-2] = 0.25 * (v[2:-2, 1:-2, :, tau]
                                     + v[2:-2, 2:-1, :, tau]) * (vtr[2:-2, 2:-1] + vtr[2:-2, 1:-2])
    flux_top[2:-2, 2:-2, :-1] = 0.25 * (v[2:-2, 2:-2, 1:, tau]
                                        + v[2:-2, 2:-2, :-1, tau]) * (wtr[2:-2, 2:-2, :-1] + wtr[2:-2, 3:-1, :-1])
    dv_adv[2:-2, 2:-2] = -maskV[2:-2, 2:-2] * (flux_east[2:-2, 2:-2] - flux_east[1:-3, 2:-2]
                                               + flux_north[2:-2, 2:-2] - flux_north[2:-2, 1:-3]) \
                         / (dzt * area_v[2:-2, 2:-2, np.newaxis])

    tmp = np.zeros_like(maskV)
    tmp = dzt * area_v[:, :, np.newaxis]
    dv_adv[:, :, 0] += -maskV[:, :, 0] * flux_top[:, :, 0] / tmp[:, :, 0]
    dv_adv[:, :, 1:] += -maskV[:, :, 1:] \
        * (flux_top[:, :, 1:] - flux_top[:, :, :-1]) / tmp[:, :, 1:]

    tend_u[:, :, :, tau] += du_adv
    tend_v[:, :, :, tau] += dv_adv

    return tend_u, tend_v


@veros_kernel
def vertical_velocity(u, v, w, maskW, dxt, dyt, dzt, cost, cosu, taup1):
    """
    vertical velocity from continuity :
    \\int_0^z w_z dz = w(z)-w(0) = - \\int dz (u_x + v_y)
    w(z) = -int dz u_x + v_y
    """
    fxa = np.zeros_like(maskW[1:, 1:])
    # integrate from bottom to surface to see error in w
    fxa[:, :, 0] = -maskW[1:, 1:, 0] * dzt[0] * \
        ((u[1:, 1:, 0, taup1] - u[:-1, 1:, 0, taup1])
         / (cost[np.newaxis, 1:] * dxt[1:, np.newaxis])
         + (cosu[np.newaxis, 1:] * v[1:, 1:, 0, taup1]
            - cosu[np.newaxis, :-1] * v[1:, :-1, 0, taup1])
         / (cost[np.newaxis, 1:] * dyt[np.newaxis, 1:]))
    fxa[:, :, 1:] = -maskW[1:, 1:, 1:] * dzt[np.newaxis, np.newaxis, 1:] \
        * ((u[1:, 1:, 1:, taup1] - u[:-1, 1:, 1:, taup1])
           / (cost[np.newaxis, 1:, np.newaxis] * dxt[1:, np.newaxis, np.newaxis])
           + (cosu[np.newaxis, 1:, np.newaxis] * v[1:, 1:, 1:, taup1]
              - cosu[np.newaxis, :-1, np.newaxis] * v[1:, :-1, 1:, taup1])
           / (cost[np.newaxis, 1:, np.newaxis] * dyt[np.newaxis, 1:, np.newaxis]))
    w[1:, 1:, :, taup1] = np.cumsum(fxa, axis=2)

    return w


@veros_routine(
    inputs=(
        'u', 'v', 'w',
        'du_mix', 'dv_mix',
        'maskU', 'maskV', 'maskW',
        'dxt', 'dxu', 'dyt', 'dyu', 'dzt', 'dzw',
        'cost', 'cosu', 'area_t', 'area_u', 'area_v',
        'coriolis_t', 'tau', 'tantr',
        'surface_taux', 'surface_tauy', 'rho_0',
    ),
    outputs=(
        'u', 'v',
        'p_hydro',
        'dpsi',
        'dpsin',
        'psi',
    ),
    settings=(
        'coord_degree',
        'pyom_compatibility_mode',
    ),
    subroutines=(friction.friction),
)
def momentum(vs):
    """
    solve for momentum for taup1
    """

    """
    time tendency due to Coriolis force
    """
    du, dv = run_kernel(tend_coriolisf, vs)

    """
    wind stress forcing
    """
    du, dv = run_kernel(tend_tauxyf, vs, tend_u=du, tend_v=dv)

    """
    advection
    """
    du, dv = run_kernel(momentum_advection, vs, tend_u=du, tend_v=dv)

    with vs.timers['friction']:
        friction.friction(vs)

    """
    external mode
    """
    with vs.timers['pressure']:
        u, v, du, dv, p_hydro, psi, dpsi, dpsin = run_kernel(streamfunction.solve_streamfunction,
                                                             vs, tend_u=du, tend_v=dv)

    return dict(
        u=u,
        v=v,
        du=du,
        dv=dv,
        p_hydro=p_hydro,
        psi=psi,
        dpsi=dpsi,
        dpsin=dpsin,
    )

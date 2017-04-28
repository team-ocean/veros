from .. import veros_method
from . import friction, isoneutral, external


@veros_method
def momentum(veros):
    """
    solve for momentum for taup1
    """

    """
    time tendency due to Coriolis force
    """
    veros.du_cor[2:-2, 2:-2] = veros.maskU[2:-2, 2:-2] \
        * (veros.coriolis_t[2:-2, 2:-2, np.newaxis] * (veros.v[2:-2, 2:-2, :, veros.tau] + veros.v[2:-2, 1:-3, :, veros.tau])
           * veros.dxt[2:-2, np.newaxis, np.newaxis] / veros.dxu[2:-2, np.newaxis, np.newaxis]
            + veros.coriolis_t[3:-1, 2:-2, np.newaxis] *
           (veros.v[3:-1, 2:-2, :, veros.tau] + veros.v[3:-1, 1:-3, :, veros.tau])
           * veros.dxt[3:-1, np.newaxis, np.newaxis] / veros.dxu[2:-2, np.newaxis, np.newaxis]) * 0.25
    veros.dv_cor[2:-2, 2:-2] = -veros.maskV[2:-2, 2:-2] \
        * (veros.coriolis_t[2:-2, 2:-2, np.newaxis] * (veros.u[1:-3, 2:-2, :, veros.tau] + veros.u[2:-2, 2:-2, :, veros.tau])
           * veros.dyt[np.newaxis, 2:-2, np.newaxis] * veros.cost[np.newaxis, 2:-2, np.newaxis]
           / (veros.dyu[np.newaxis, 2:-2, np.newaxis] * veros.cosu[np.newaxis, 2:-2, np.newaxis])
           + veros.coriolis_t[2:-2, 3:-1, np.newaxis]
           * (veros.u[1:-3, 3:-1, :, veros.tau] + veros.u[2:-2, 3:-1, :, veros.tau])
           * veros.dyt[np.newaxis, 3:-1, np.newaxis] * veros.cost[np.newaxis, 3:-1, np.newaxis]
           / (veros.dyu[np.newaxis, 2:-2, np.newaxis] * veros.cosu[np.newaxis, 2:-2, np.newaxis])) * 0.25

    """
    time tendency due to metric terms
    """
    if veros.coord_degree:
        veros.du_cor[2:-2, 2:-2] += veros.maskU[2:-2, 2:-2] * 0.125 * veros.tantr[np.newaxis, 2:-2, np.newaxis] \
            * ((veros.u[2:-2, 2:-2, :, veros.tau] + veros.u[1:-3, 2:-2, :, veros.tau])
               * (veros.v[2:-2, 2:-2, :, veros.tau] + veros.v[2:-2, 1:-3, :, veros.tau])
               * veros.dxt[2:-2, np.newaxis, np.newaxis] / veros.dxu[2:-2, np.newaxis, np.newaxis]
               + (veros.u[3:-1, 2:-2, :, veros.tau] + veros.u[2:-2, 2:-2, :, veros.tau])
               * (veros.v[3:-1, 2:-2, :, veros.tau] + veros.v[3:-1, 1:-3, :, veros.tau])
               * veros.dxt[3:-1, np.newaxis, np.newaxis] / veros.dxu[2:-2, np.newaxis, np.newaxis])
        veros.dv_cor[2:-2, 2:-2] += -veros.maskV[2:-2, 2:-2] * 0.125 \
            * (veros.tantr[np.newaxis, 2:-2, np.newaxis] * (veros.u[2:-2, 2:-2, :, veros.tau] + veros.u[1:-3, 2:-2, :, veros.tau])**2
               * veros.dyt[np.newaxis, 2:-2, np.newaxis] * veros.cost[np.newaxis, 2:-2, np.newaxis]
               / (veros.dyu[np.newaxis, 2:-2, np.newaxis] * veros.cosu[np.newaxis, 2:-2, np.newaxis])
               + veros.tantr[np.newaxis, 3:-1, np.newaxis]
               * (veros.u[2:-2, 3:-1, :, veros.tau] + veros.u[1:-3, 3:-1, :, veros.tau])**2
               * veros.dyt[np.newaxis, 3:-1, np.newaxis] * veros.cost[np.newaxis, 3:-1, np.newaxis]
               / (veros.dyu[np.newaxis, 2:-2, np.newaxis] * veros.cosu[np.newaxis, 2:-2, np.newaxis]))

    """
    transfer to time tendencies
    """
    veros.du[2:-2, 2:-2, :, veros.tau] = veros.du_cor[2:-2, 2:-2]
    veros.dv[2:-2, 2:-2, :, veros.tau] = veros.dv_cor[2:-2, 2:-2]

    """
    wind stress forcing
    """
    veros.du[2:-2, 2:-2, -1, veros.tau] += veros.maskU[2:-2, 2:-2, -1] * \
        veros.surface_taux[2:-2, 2:-2] / veros.dzt[-1]
    veros.dv[2:-2, 2:-2, -1, veros.tau] += veros.maskV[2:-2, 2:-2, -1] * \
        veros.surface_tauy[2:-2, 2:-2] / veros.dzt[-1]

    """
    advection
    """
    momentum_advection(veros)
    veros.du[:, :, :, veros.tau] += veros.du_adv
    veros.dv[:, :, :, veros.tau] += veros.dv_adv

    with veros.timers["friction"]:
        """
        vertical friction
        """
        veros.K_diss_v[...] = 0.0
        if veros.enable_implicit_vert_friction:
            friction.implicit_vert_friction(veros)
        if veros.enable_explicit_vert_friction:
            friction.explicit_vert_friction(veros)

        """
        TEM formalism for eddy-driven velocity
        """
        if veros.enable_TEM_friction:
            isoneutral.isoneutral_friction(veros)

        """
        horizontal friction
        """
        if veros.enable_hor_friction:
            friction.harmonic_friction(veros)
        if veros.enable_biharmonic_friction:
            friction.biharmonic_friction(veros)

        """
        Rayleigh and bottom friction
        """
        veros.K_diss_bot[...] = 0.0
        if veros.enable_ray_friction:
            friction.rayleigh_friction(veros)
        if veros.enable_bottom_friction:
            friction.linear_bottom_friction(veros)
        if veros.enable_quadratic_bottom_friction:
            friction.quadratic_bottom_friction(veros)

        """
        add user defined forcing
        """
        if veros.enable_momentum_sources:
            friction.momentum_sources(veros)

    """
    external mode
    """
    with veros.timers["pressure"]:
        external.solve_streamfunction(veros)


@veros_method
def vertical_velocity(veros):
    """
           vertical velocity from continuity :
           \int_0^z w_z dz = w(z)-w(0) = - \int dz (u_x + v_y)
           w(z) = -int dz u_x + v_y
    """
    fxa = np.empty((veros.nx + 3, veros.ny + 3, veros.nz))
    # integrate from bottom to surface to see error in w
    fxa[:, :, 0] = -veros.maskW[1:, 1:, 0] * veros.dzt[0] * \
        ((veros.u[1:, 1:, 0, veros.taup1] - veros.u[:-1, 1:, 0, veros.taup1])
         / (veros.cost[np.newaxis, 1:] * veros.dxt[1:, np.newaxis])
         + (veros.cosu[np.newaxis, 1:] * veros.v[1:, 1:, 0, veros.taup1]
            - veros.cosu[np.newaxis, :-1] * veros.v[1:, :-1, 0, veros.taup1])
         / (veros.cost[np.newaxis, 1:] * veros.dyt[np.newaxis, 1:]))
    fxa[:, :, 1:] = -veros.maskW[1:, 1:, 1:] * veros.dzt[np.newaxis, np.newaxis, 1:] \
        * ((veros.u[1:, 1:, 1:, veros.taup1] - veros.u[:-1, 1:, 1:, veros.taup1])
           / (veros.cost[np.newaxis, 1:, np.newaxis] * veros.dxt[1:, np.newaxis, np.newaxis])
           + (veros.cosu[np.newaxis, 1:, np.newaxis] * veros.v[1:, 1:, 1:, veros.taup1]
              - veros.cosu[np.newaxis, :-1, np.newaxis] * veros.v[1:, :-1, 1:, veros.taup1])
           / (veros.cost[np.newaxis, 1:, np.newaxis] * veros.dyt[np.newaxis, 1:, np.newaxis]))
    veros.w[1:, 1:, :, veros.taup1] = np.cumsum(fxa, axis=2)


@veros_method
def momentum_advection(veros):
    """
    Advection of momentum with second order which is energy conserving
    """

    """
    Code from MITgcm
    """
    utr = veros.u[..., veros.tau] * veros.maskU * veros.dyt[np.newaxis, :, np.newaxis] \
        * veros.dzt[np.newaxis, np.newaxis, :]
    vtr = veros.dzt[np.newaxis, np.newaxis, :] * veros.cosu[np.newaxis, :, np.newaxis] \
        * veros.dxt[:, np.newaxis, np.newaxis] * veros.v[..., veros.tau] * veros.maskV
    wtr = veros.w[..., veros.tau] * veros.maskW * veros.area_t[:, :, np.newaxis]

    """
    for zonal momentum
    """
    veros.flux_east[1:-2, 2:-2] = 0.25 * (veros.u[1:-2, 2:-2, :, veros.tau] +
                                          veros.u[2:-1, 2:-2, :, veros.tau]) * (utr[2:-1, 2:-2] + utr[1:-2, 2:-2])
    veros.flux_north[2:-2, 1:-2] = 0.25 * (veros.u[2:-2, 1:-2, :, veros.tau] +
                                           veros.u[2:-2, 2:-1, :, veros.tau]) * (vtr[3:-1, 1:-2] + vtr[2:-2, 1:-2])
    veros.flux_top[2:-2, 2:-2, :-1] = 0.25 * (veros.u[2:-2, 2:-2, 1:, veros.tau] +
                                              veros.u[2:-2, 2:-2, :-1, veros.tau]) * (wtr[2:-2, 2:-2, :-1] + wtr[3:-1, 2:-2, :-1])
    veros.flux_top[:, :, -1] = 0.0
    veros.du_adv[2:-2, 2:-2] = -veros.maskU[2:-2, 2:-2] * (veros.flux_east[2:-2, 2:-2] - veros.flux_east[1:-3, 2:-2]
                                                           + veros.flux_north[2:-2, 2:-2] - veros.flux_north[2:-2, 1:-3]) \
        / (veros.dzt[np.newaxis, np.newaxis, :] * veros.area_u[2:-2, 2:-2, np.newaxis])

    tmp = -veros.maskU / (veros.dzt * veros.area_u[:, :, np.newaxis])
    veros.du_adv += tmp * veros.flux_top
    veros.du_adv[:, :, 1:] += tmp[:, :, 1:] * -veros.flux_top[:, :, :-1]

    """
    for meridional momentum
    """
    veros.flux_east[1:-2, 2:-2] = 0.25 * (veros.v[1:-2, 2:-2, :, veros.tau]
                                          + veros.v[2:-1, 2:-2, :, veros.tau]) * (utr[1:-2, 3:-1] + utr[1:-2, 2:-2])
    veros.flux_north[2:-2, 1:-2] = 0.25 * (veros.v[2:-2, 1:-2, :, veros.tau]
                                           + veros.v[2:-2, 2:-1, :, veros.tau]) * (vtr[2:-2, 2:-1] + vtr[2:-2, 1:-2])
    veros.flux_top[2:-2, 2:-2, :-1] = 0.25 * (veros.v[2:-2, 2:-2, 1:, veros.tau]
                                              + veros.v[2:-2, 2:-2, :-1, veros.tau]) * (wtr[2:-2, 2:-2, :-1] + wtr[2:-2, 3:-1, :-1])
    veros.flux_top[:, :, -1] = 0.0
    veros.dv_adv[2:-2, 2:-2] = -veros.maskV[2:-2, 2:-2] * (veros.flux_east[2:-2, 2:-2] - veros.flux_east[1:-3, 2:-2]
                                                           + veros.flux_north[2:-2, 2:-2] - veros.flux_north[2:-2, 1:-3]) \
        / (veros.dzt * veros.area_v[2:-2, 2:-2, np.newaxis])
    tmp = veros.dzt * veros.area_v[:, :, np.newaxis]
    veros.dv_adv[:, :, 0] += -veros.maskV[:, :, 0] * veros.flux_top[:, :, 0] / tmp[:, :, 0]
    veros.dv_adv[:, :, 1:] += -veros.maskV[:, :, 1:] \
        * (veros.flux_top[:, :, 1:] - veros.flux_top[:, :, :-1]) / tmp[:, :, 1:]

from .. import veros_method
from ..variables import allocate
from . import friction, isoneutral, streamfunction


@veros_method
def momentum(vs):
    """
    solve for momentum for taup1
    """

    """
    time tendency due to Coriolis force
    """
    vs.du_cor[2:-2, 2:-2] = vs.maskU[2:-2, 2:-2] \
        * (vs.coriolis_t[2:-2, 2:-2, np.newaxis] * (vs.v[2:-2, 2:-2, :, vs.tau] + vs.v[2:-2, 1:-3, :, vs.tau])
           * vs.dxt[2:-2, np.newaxis, np.newaxis] / vs.dxu[2:-2, np.newaxis, np.newaxis]
            + vs.coriolis_t[3:-1, 2:-2, np.newaxis] *
           (vs.v[3:-1, 2:-2, :, vs.tau] + vs.v[3:-1, 1:-3, :, vs.tau])
           * vs.dxt[3:-1, np.newaxis, np.newaxis] / vs.dxu[2:-2, np.newaxis, np.newaxis]) * 0.25
    vs.dv_cor[2:-2, 2:-2] = -vs.maskV[2:-2, 2:-2] \
        * (vs.coriolis_t[2:-2, 2:-2, np.newaxis] * (vs.u[1:-3, 2:-2, :, vs.tau] + vs.u[2:-2, 2:-2, :, vs.tau])
           * vs.dyt[np.newaxis, 2:-2, np.newaxis] * vs.cost[np.newaxis, 2:-2, np.newaxis]
           / (vs.dyu[np.newaxis, 2:-2, np.newaxis] * vs.cosu[np.newaxis, 2:-2, np.newaxis])
           + vs.coriolis_t[2:-2, 3:-1, np.newaxis]
           * (vs.u[1:-3, 3:-1, :, vs.tau] + vs.u[2:-2, 3:-1, :, vs.tau])
           * vs.dyt[np.newaxis, 3:-1, np.newaxis] * vs.cost[np.newaxis, 3:-1, np.newaxis]
           / (vs.dyu[np.newaxis, 2:-2, np.newaxis] * vs.cosu[np.newaxis, 2:-2, np.newaxis])) * 0.25

    """
    time tendency due to metric terms
    """
    if vs.coord_degree:
        vs.du_cor[2:-2, 2:-2] += vs.maskU[2:-2, 2:-2] * 0.125 * vs.tantr[np.newaxis, 2:-2, np.newaxis] \
            * ((vs.u[2:-2, 2:-2, :, vs.tau] + vs.u[1:-3, 2:-2, :, vs.tau])
               * (vs.v[2:-2, 2:-2, :, vs.tau] + vs.v[2:-2, 1:-3, :, vs.tau])
               * vs.dxt[2:-2, np.newaxis, np.newaxis] / vs.dxu[2:-2, np.newaxis, np.newaxis]
               + (vs.u[3:-1, 2:-2, :, vs.tau] + vs.u[2:-2, 2:-2, :, vs.tau])
               * (vs.v[3:-1, 2:-2, :, vs.tau] + vs.v[3:-1, 1:-3, :, vs.tau])
               * vs.dxt[3:-1, np.newaxis, np.newaxis] / vs.dxu[2:-2, np.newaxis, np.newaxis])
        vs.dv_cor[2:-2, 2:-2] += -vs.maskV[2:-2, 2:-2] * 0.125 \
            * (vs.tantr[np.newaxis, 2:-2, np.newaxis] * (vs.u[2:-2, 2:-2, :, vs.tau] + vs.u[1:-3, 2:-2, :, vs.tau])**2
               * vs.dyt[np.newaxis, 2:-2, np.newaxis] * vs.cost[np.newaxis, 2:-2, np.newaxis]
               / (vs.dyu[np.newaxis, 2:-2, np.newaxis] * vs.cosu[np.newaxis, 2:-2, np.newaxis])
               + vs.tantr[np.newaxis, 3:-1, np.newaxis]
               * (vs.u[2:-2, 3:-1, :, vs.tau] + vs.u[1:-3, 3:-1, :, vs.tau])**2
               * vs.dyt[np.newaxis, 3:-1, np.newaxis] * vs.cost[np.newaxis, 3:-1, np.newaxis]
               / (vs.dyu[np.newaxis, 2:-2, np.newaxis] * vs.cosu[np.newaxis, 2:-2, np.newaxis]))

    """
    transfer to time tendencies
    """
    vs.du[2:-2, 2:-2, :, vs.tau] = vs.du_cor[2:-2, 2:-2]
    vs.dv[2:-2, 2:-2, :, vs.tau] = vs.dv_cor[2:-2, 2:-2]

    """
    wind stress forcing
    """
    if vs.pyom_compatibility_mode:
        vs.du[2:-2, 2:-2, -1, vs.tau] += vs.maskU[2:-2, 2:-2, -1] * vs.surface_taux[2:-2, 2:-2] / vs.dzt[-1]
        vs.dv[2:-2, 2:-2, -1, vs.tau] += vs.maskV[2:-2, 2:-2, -1] * vs.surface_tauy[2:-2, 2:-2] / vs.dzt[-1]
    else:
        vs.du[2:-2, 2:-2, -1, vs.tau] += vs.maskU[2:-2, 2:-2, -1] * vs.surface_taux[2:-2, 2:-2] / vs.dzt[-1] / vs.rho_0
        vs.dv[2:-2, 2:-2, -1, vs.tau] += vs.maskV[2:-2, 2:-2, -1] * vs.surface_tauy[2:-2, 2:-2] / vs.dzt[-1] / vs.rho_0

    """
    advection
    """
    momentum_advection(vs)
    vs.du[:, :, :, vs.tau] += vs.du_adv
    vs.dv[:, :, :, vs.tau] += vs.dv_adv

    with vs.timers['friction']:
        """
        vertical friction
        """
        vs.K_diss_v[...] = 0.0
        if vs.enable_implicit_vert_friction:
            friction.implicit_vert_friction(vs)
        if vs.enable_explicit_vert_friction:
            friction.explicit_vert_friction(vs)

        """
        TEM formalism for eddy-driven velocity
        """
        if vs.enable_TEM_friction:
            isoneutral.isoneutral_friction(vs)

        """
        horizontal friction
        """
        if vs.enable_hor_friction:
            friction.harmonic_friction(vs)
        if vs.enable_biharmonic_friction:
            friction.biharmonic_friction(vs)

        """
        Rayleigh and bottom friction
        """
        vs.K_diss_bot[...] = 0.0
        if vs.enable_ray_friction:
            friction.rayleigh_friction(vs)
        if vs.enable_bottom_friction:
            friction.linear_bottom_friction(vs)
        if vs.enable_quadratic_bottom_friction:
            friction.quadratic_bottom_friction(vs)

        """
        add user defined forcing
        """
        if vs.enable_momentum_sources:
            friction.momentum_sources(vs)

    """
    external mode
    """
    with vs.timers['pressure']:
        streamfunction.solve_streamfunction(vs)


@veros_method
def vertical_velocity(vs):
    """
    vertical velocity from continuity :
    \\int_0^z w_z dz = w(z)-w(0) = - \\int dz (u_x + v_y)
    w(z) = -int dz u_x + v_y
    """
    fxa = allocate(vs, ('xt', 'yt', 'zw'))[1:, 1:]
    # integrate from bottom to surface to see error in w
    fxa[:, :, 0] = -vs.maskW[1:, 1:, 0] * vs.dzt[0] * \
        ((vs.u[1:, 1:, 0, vs.taup1] - vs.u[:-1, 1:, 0, vs.taup1])
        / (vs.cost[np.newaxis, 1:] * vs.dxt[1:, np.newaxis])
        + (vs.cosu[np.newaxis, 1:] * vs.v[1:, 1:, 0, vs.taup1]
            - vs.cosu[np.newaxis, :-1] * vs.v[1:, :-1, 0, vs.taup1])
        / (vs.cost[np.newaxis, 1:] * vs.dyt[np.newaxis, 1:]))
    fxa[:, :, 1:] = -vs.maskW[1:, 1:, 1:] * vs.dzt[np.newaxis, np.newaxis, 1:] \
        * ((vs.u[1:, 1:, 1:, vs.taup1] - vs.u[:-1, 1:, 1:, vs.taup1])
        / (vs.cost[np.newaxis, 1:, np.newaxis] * vs.dxt[1:, np.newaxis, np.newaxis])
        + (vs.cosu[np.newaxis, 1:, np.newaxis] * vs.v[1:, 1:, 1:, vs.taup1]
            - vs.cosu[np.newaxis, :-1, np.newaxis] * vs.v[1:, :-1, 1:, vs.taup1])
        / (vs.cost[np.newaxis, 1:, np.newaxis] * vs.dyt[np.newaxis, 1:, np.newaxis]))
    vs.w[1:, 1:, :, vs.taup1] = np.cumsum(fxa, axis=2)


@veros_method
def momentum_advection(vs):
    """
    Advection of momentum with second order which is energy conserving
    """

    """
    Code from MITgcm
    """
    utr = vs.u[..., vs.tau] * vs.maskU * vs.dyt[np.newaxis, :, np.newaxis] \
        * vs.dzt[np.newaxis, np.newaxis, :]
    vtr = vs.dzt[np.newaxis, np.newaxis, :] * vs.cosu[np.newaxis, :, np.newaxis] \
        * vs.dxt[:, np.newaxis, np.newaxis] * vs.v[..., vs.tau] * vs.maskV
    wtr = vs.w[..., vs.tau] * vs.maskW * vs.area_t[:, :, np.newaxis]

    """
    for zonal momentum
    """
    vs.flux_top[...] = 0.
    vs.flux_east[1:-2, 2:-2] = 0.25 * (vs.u[1:-2, 2:-2, :, vs.tau] \
                                     + vs.u[2:-1, 2:-2, :, vs.tau]) \
                                    * (utr[2:-1, 2:-2] + utr[1:-2, 2:-2])
    vs.flux_north[2:-2, 1:-2] = 0.25 * (vs.u[2:-2, 1:-2, :, vs.tau] \
                                      + vs.u[2:-2, 2:-1, :, vs.tau]) \
                                     * (vtr[3:-1, 1:-2] + vtr[2:-2, 1:-2])
    vs.flux_top[2:-2, 2:-2, :-1] = 0.25 * (vs.u[2:-2, 2:-2, 1:, vs.tau] \
                                         + vs.u[2:-2, 2:-2, :-1, vs.tau]) \
                                        * (wtr[2:-2, 2:-2, :-1] + wtr[3:-1, 2:-2, :-1])
    vs.du_adv[2:-2, 2:-2] = -vs.maskU[2:-2, 2:-2] * (vs.flux_east[2:-2, 2:-2] - vs.flux_east[1:-3, 2:-2]
                                                   + vs.flux_north[2:-2, 2:-2] - vs.flux_north[2:-2, 1:-3]) \
                            / (vs.dzt[np.newaxis, np.newaxis, :] * vs.area_u[2:-2, 2:-2, np.newaxis])

    tmp = -vs.maskU / (vs.dzt * vs.area_u[:, :, np.newaxis])
    vs.du_adv += tmp * vs.flux_top
    vs.du_adv[:, :, 1:] += tmp[:, :, 1:] * -vs.flux_top[:, :, :-1]

    """
    for meridional momentum
    """
    vs.flux_top[...] = 0.
    vs.flux_east[1:-2, 2:-2] = 0.25 * (vs.v[1:-2, 2:-2, :, vs.tau]
                                     + vs.v[2:-1, 2:-2, :, vs.tau]) * (utr[1:-2, 3:-1] + utr[1:-2, 2:-2])
    vs.flux_north[2:-2, 1:-2] = 0.25 * (vs.v[2:-2, 1:-2, :, vs.tau]
                                      + vs.v[2:-2, 2:-1, :, vs.tau]) * (vtr[2:-2, 2:-1] + vtr[2:-2, 1:-2])
    vs.flux_top[2:-2, 2:-2, :-1] = 0.25 * (vs.v[2:-2, 2:-2, 1:, vs.tau]
                                         + vs.v[2:-2, 2:-2, :-1, vs.tau]) * (wtr[2:-2, 2:-2, :-1] + wtr[2:-2, 3:-1, :-1])
    vs.dv_adv[2:-2, 2:-2] = -vs.maskV[2:-2, 2:-2] * (vs.flux_east[2:-2, 2:-2] - vs.flux_east[1:-3, 2:-2]
                                                   + vs.flux_north[2:-2, 2:-2] - vs.flux_north[2:-2, 1:-3]) \
                            / (vs.dzt * vs.area_v[2:-2, 2:-2, np.newaxis])
    tmp = vs.dzt * vs.area_v[:, :, np.newaxis]
    vs.dv_adv[:, :, 0] += -vs.maskV[:, :, 0] * vs.flux_top[:, :, 0] / tmp[:, :, 0]
    vs.dv_adv[:, :, 1:] += -vs.maskV[:, :, 1:] \
        * (vs.flux_top[:, :, 1:] - vs.flux_top[:, :, :-1]) / tmp[:, :, 1:]

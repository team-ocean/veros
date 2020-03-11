from veros.core.operators import numpy as np

from veros import veros_kernel, veros_routine, run_kernel
from veros.core.utilities import pad_z_edges, where
from veros.core.operators import update, update_add, update_multiply, at


@veros_kernel
def _calc_cr(rjp, rj, rjm, vel):
    """
    Calculates cr value used in superbee advection scheme
    """
    eps = 1e-20  # prevent division by 0
    return where(vel > 0., rjm, rjp) / where(np.abs(rj) < eps, eps, rj)


@veros_kernel
def limiter(cr):
    return np.maximum(0., np.maximum(np.minimum(1., 2 * cr), np.minimum(2., cr)))


@veros_kernel(static_args=('axis',))
def _adv_superbee(vel, var, mask, dx, axis, cost, cosu, dt_tracer):
    velfac = 1
    if axis == 0:
        sm1, s, sp1, sp2 = ((slice(1 + n, -2 + n or None), slice(2, -2), slice(None))
                            for n in range(-1, 3))
        dx = cost[np.newaxis, 2:-2, np.newaxis] * \
            dx[1:-2, np.newaxis, np.newaxis]
    elif axis == 1:
        sm1, s, sp1, sp2 = ((slice(2, -2), slice(1 + n, -2 + n or None), slice(None))
                            for n in range(-1, 3))
        dx = (cost * dx)[np.newaxis, 1:-2, np.newaxis]
        velfac = cosu[np.newaxis, 1:-2, np.newaxis]
    elif axis == 2:
        vel, var, mask = (pad_z_edges(a) for a in (vel, var, mask))
        sm1, s, sp1, sp2 = ((slice(2, -2), slice(2, -2), slice(1 + n, -2 + n or None))
                            for n in range(-1, 3))
        dx = dx[np.newaxis, np.newaxis, :-1]
    else:
        raise ValueError('axis must be 0, 1, or 2')
    uCFL = np.abs(velfac * vel[s] * dt_tracer / dx)
    rjp = (var[sp2] - var[sp1]) * mask[sp1]
    rj = (var[sp1] - var[s]) * mask[s]
    rjm = (var[s] - var[sm1]) * mask[sm1]
    cr = limiter(_calc_cr(rjp, rj, rjm, vel[s]))

    return velfac * vel[s] * (var[sp1] + var[s]) * 0.5 - np.abs(velfac * vel[s]) * ((1. - cr) + uCFL * cr) * rj * 0.5


@veros_kernel
def adv_flux_2nd(adv_fe, adv_fn, adv_ft, var, u, v, w,
                 cosu, maskU, maskV, maskW, tau):
    """
    2th order advective tracer flux
    """
    adv_fe = update(adv_fe, at[1:-2, 2:-2, :], 0.5 * (var[1:-2, 2:-2, :] + var[2:-1, 2:-2, :]) \
        * u[1:-2, 2:-2, :, tau] * maskU[1:-2, 2:-2, :])
    adv_fn = update(adv_fn, at[2:-2, 1:-2, :], cosu[np.newaxis, 1:-2, np.newaxis] * 0.5 * (var[2:-2, 1:-2, :] + var[2:-2, 2:-1, :]) \
        * v[2:-2, 1:-2, :, tau] * maskV[2:-2, 1:-2, :])
    adv_ft = update(adv_ft, at[2:-2, 2:-2, :-1], 0.5 * (var[2:-2, 2:-2, :-1] + var[2:-2, 2:-2, 1:]) \
        * w[2:-2, 2:-2, :-1, tau] * maskW[2:-2, 2:-2, :-1])
    adv_ft = update(adv_ft, at[:, :, -1], 0.)

    return adv_fe, adv_fn, adv_ft


@veros_kernel
def adv_flux_superbee(adv_fe, adv_fn, adv_ft, var, u, v, w, dxt, dyt, dzt,
                      maskU, maskV, maskW, cost, cosu, dt_tracer, tau):
    r"""
    from MITgcm
    Calculates advection of a tracer
    using second-order interpolation with a flux limiter:

    \begin{equation*}
    F^x_{adv} = U \overline{ \theta }^i
    - \frac{1}{2} \left([ 1 - \psi(C_r) ] |U|
       + U \frac{u \Delta t}{\Delta x_c} \psi(C_r)
                 \right) \delta_i \theta
    \end{equation*}

    where the $\psi(C_r)$ is the limiter function and $C_r$ is
    the slope ratio.
    """
    adv_fe = update(adv_fe, at[1:-2, 2:-2, :], _adv_superbee(u[..., tau], var, maskU, dxt, 0,
                                          cost, cosu, dt_tracer))
    adv_fn = update(adv_fn, at[2:-2, 1:-2, :], _adv_superbee(v[..., tau], var, maskV, dyt, 1,
                                          cost, cosu, dt_tracer))
    adv_ft = update(adv_ft, at[2:-2, 2:-2, :-1], _adv_superbee(w[..., tau], var, maskW, dzt, 2,
                                            cost, cosu, dt_tracer))
    adv_ft = update(adv_ft, at[..., -1], 0.)

    return adv_fe, adv_fn, adv_ft


@veros_routine(
    inputs=('u', 'v', 'w', 'maskU', 'maskV', 'maskW', 'dxt', 'dyt', 'dzt', 'dzw',
            'cosu', 'cost', 'tau'),
    outputs=('u_wgrid', 'v_wgrid', 'w_wgrid')
)
def calculate_velocity_on_wgrid(vs):
    u_wgrid, v_wgrid, w_wgrid = run_kernel(calculate_velocity_on_wgrid_kernel, vs)
    return dict(
        u_wgrid=u_wgrid, v_wgrid=v_wgrid, w_wgrid=w_wgrid
    )

@veros_kernel
def calculate_velocity_on_wgrid_kernel(u, v, w, maskU, maskV, maskW, dxt, dyt, dzt, dzw,
                                       cosu, cost, tau):
    """
    calculates advection velocity for tracer on W grid

    Note: this implementation is not strictly equal to the Fortran version. They only match
    if maskW has exactly one true value across each depth slice.
    """
    u_wgrid = np.zeros_like(maskU)
    v_wgrid = np.zeros_like(maskV)
    w_wgrid = np.zeros_like(maskU)

    # lateral advection velocities on W grid
    u_wgrid = update(u_wgrid, at[:, :, :-1], u[:, :, 1:, tau] * maskU[:, :, 1:] * 0.5 \
        * dzt[np.newaxis, np.newaxis, 1:] / dzw[np.newaxis, np.newaxis, :-1] \
        + u[:, :, :-1, tau] * maskU[:, :, :-1] * 0.5 \
        * dzt[np.newaxis, np.newaxis, :-1] / dzw[np.newaxis, np.newaxis, :-1])
    v_wgrid = update(v_wgrid, at[:, :, :-1], v[:, :, 1:, tau] * maskV[:, :, 1:] * 0.5 \
        * dzt[np.newaxis, np.newaxis, 1:] / dzw[np.newaxis, np.newaxis, :-1] \
        + v[:, :, :-1, tau] * maskV[:, :, :-1] * 0.5 \
        * dzt[np.newaxis, np.newaxis, :-1] / dzw[np.newaxis, np.newaxis, :-1])
    u_wgrid = update(u_wgrid, at[:, :, -1], u[:, :, -1, tau] * \
        maskU[:, :, -1] * 0.5 * dzt[-1:] / dzw[-1:])
    v_wgrid = update(v_wgrid, at[:, :, -1], v[:, :, -1, tau] * \
        maskV[:, :, -1] * 0.5 * dzt[-1:] / dzw[-1:])

    # redirect velocity at bottom and at topography
    u_wgrid = update(u_wgrid, at[:, :, 0], u_wgrid[:, :, 0] + u[:, :, 0, tau] \
        * maskU[:, :, 0] * 0.5 * dzt[0:1] / dzw[0:1])
    v_wgrid = update(v_wgrid, at[:, :, 0], v_wgrid[:, :, 0] + v[:, :, 0, tau] \
        * maskV[:, :, 0] * 0.5 * dzt[0:1] / dzw[0:1])
    mask = maskW[:-1, :, :-1] * maskW[1:, :, :-1]
    u_wgrid = update_add(u_wgrid, at[:-1, :, 1:], (u_wgrid[:-1, :, :-1] * dzw[np.newaxis, np.newaxis, :-1]
                            / dzw[np.newaxis, np.newaxis, 1:]) * (1. - mask))
    u_wgrid = update_multiply(u_wgrid, at[:-1, :, :-1], mask)
    mask = maskW[:, :-1, :-1] * maskW[:, 1:, :-1]
    v_wgrid = update_add(v_wgrid, at[:, :-1, 1:], (v_wgrid[:, :-1, :-1] * dzw[np.newaxis, np.newaxis, :-1]
                            / dzw[np.newaxis, np.newaxis, 1:]) * (1. - mask))
    v_wgrid = update_multiply(v_wgrid, at[:, :-1, :-1], mask)

    # vertical advection velocity on W grid from continuity
    w_wgrid = update(w_wgrid, at[:, :, 0], 0.)
    w_wgrid = update(w_wgrid, at[1:, 1:, :], np.cumsum(-dzw[np.newaxis, np.newaxis, :] *
                                   ((u_wgrid[1:, 1:, :] - u_wgrid[:-1, 1:, :]) / (cost[np.newaxis, 1:, np.newaxis] * dxt[1:, np.newaxis, np.newaxis])
                                    + (cosu[np.newaxis, 1:, np.newaxis] * v_wgrid[1:, 1:, :] -
                                       cosu[np.newaxis, :-1, np.newaxis] * v_wgrid[1:, :-1, :])
                                    / (cost[np.newaxis, 1:, np.newaxis] * dyt[np.newaxis, 1:, np.newaxis])), axis=2))

    return u_wgrid, v_wgrid, w_wgrid


@veros_kernel
def adv_flux_superbee_wgrid(adv_fe, adv_fn, adv_ft, u_wgrid, v_wgrid, w_wgrid,
                            var, maskW, dxt, dyt, dzw, cost, cosu, dt_tracer):
    """
    Calculates advection of a tracer defined on Wgrid
    """
    maskUtr = np.zeros_like(maskW)
    maskUtr = update(maskUtr, at[:-1, :, :], maskW[1:, :, :] * maskW[:-1, :, :])
    adv_fe = update(adv_fe, at[1:-2, 2:-2, :], _adv_superbee(u_wgrid, var, maskUtr, dxt, 0,
                                          cost, cosu, dt_tracer))

    maskVtr = np.zeros_like(maskW)
    maskVtr = update(maskVtr, at[:, :-1, :], maskW[:, 1:, :] * maskW[:, :-1, :])
    adv_fn = update(adv_fn, at[2:-2, 1:-2, :], _adv_superbee(v_wgrid, var, maskVtr, dyt, 1,
                                          cost, cosu, dt_tracer))

    maskWtr = np.zeros_like(maskW)
    maskWtr = update(maskWtr, at[:, :, :-1], maskW[:, :, 1:] * maskW[:, :, :-1])
    adv_ft = update(adv_ft, at[2:-2, 2:-2, :-1], _adv_superbee(w_wgrid, var, maskWtr, dzw, 2,
                                            cost, cosu, dt_tracer))
    adv_ft = update(adv_ft, at[..., -1], 0.0)

    return adv_fe, adv_fn, adv_ft


@veros_kernel
def adv_flux_upwind_wgrid(adv_fe, adv_fn, adv_ft, u_wgrid, v_wgrid, w_wgrid,
                          var, maskW, cosu):
    """
    Calculates advection of a tracer defined on Wgrid
    """
    maskUtr = maskW[2:-1, 2:-2, :] * maskW[1:-2, 2:-2, :]
    rj = (var[2:-1, 2:-2, :] - var[1:-2, 2:-2, :]) * maskUtr
    adv_fe = update(adv_fe, at[1:-2, 2:-2, :], u_wgrid[1:-2, 2:-2, :] * (var[2:-1, 2:-2, :] + var[1:-2, 2:-2, :]) * 0.5 \
        - np.abs(u_wgrid[1:-2, 2:-2, :]) * rj * 0.5)

    maskVtr = maskW[2:-2, 2:-1, :] * maskW[2:-2, 1:-2, :]
    rj = (var[2:-2, 2:-1, :] - var[2:-2, 1:-2, :]) * maskVtr
    adv_fn = update(adv_fn, at[2:-2, 1:-2, :], cosu[np.newaxis, 1:-2, np.newaxis] * v_wgrid[2:-2, 1:-2, :] * \
        (var[2:-2, 2:-1, :] + var[2:-2, 1:-2, :]) * 0.5 \
        - np.abs(cosu[np.newaxis, 1:-2, np.newaxis] * v_wgrid[2:-2, 1:-2, :]) * rj * 0.5)

    maskWtr = maskW[2:-2, 2:-2, 1:] * maskW[2:-2, 2:-2, :-1]
    rj = (var[2:-2, 2:-2, 1:] - var[2:-2, 2:-2, :-1]) * maskWtr
    adv_ft = update(adv_ft, at[2:-2, 2:-2, :-1], w_wgrid[2:-2, 2:-2, :-1] * (var[2:-2, 2:-2, 1:] + var[2:-2, 2:-2, :-1]) * 0.5 \
        - np.abs(w_wgrid[2:-2, 2:-2, :-1]) * rj * 0.5)
    adv_ft = update(adv_ft, at[:, :, -1], 0.)

    return adv_fe, adv_fn, adv_ft

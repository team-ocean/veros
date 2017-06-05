import warnings

from .. import veros_method
from .utilities import pad_z_edges


@veros_method
def _calc_cr(veros, rjp, rj, rjm, vel):
    """
    Calculates cr value used in superbee advection scheme
    """
    eps = 1e-20  # prevent division by 0
    return np.where(vel > 0., rjm, rjp) / np.where(np.abs(rj) < eps, eps, rj)


@veros_method
def _adv_superbee(veros, vel, var, mask, dx, axis):
    def limiter(cr):
        return np.maximum(0., np.maximum(np.minimum(1., 2 * cr), np.minimum(2., cr)))
    velfac = 1
    if axis == 0:
        sm1, s, sp1, sp2 = ((slice(1 + n, -2 + n or None), slice(2, -2), slice(None))
                            for n in range(-1, 3))
        dx = veros.cost[np.newaxis, 2:-2, np.newaxis] * dx[1:-2, np.newaxis, np.newaxis]
    elif axis == 1:
        sm1, s, sp1, sp2 = ((slice(2, -2), slice(1 + n, -2 + n or None), slice(None))
                            for n in range(-1, 3))
        dx = (veros.cost * dx)[np.newaxis, 1:-2, np.newaxis]
        velfac = veros.cosu[np.newaxis, 1:-2, np.newaxis]
    elif axis == 2:
        vel, var, mask = (pad_z_edges(veros, a) for a in (vel, var, mask))
        sm1, s, sp1, sp2 = ((slice(2, -2), slice(2, -2), slice(1 + n, -2 + n or None))
                            for n in range(-1, 3))
        dx = dx[np.newaxis, np.newaxis, :-1]
    else:
        raise ValueError("axis must be 0, 1, or 2")
    uCFL = np.abs(velfac * vel[s] * veros.dt_tracer / dx)
    rjp = (var[sp2] - var[sp1]) * mask[sp1]
    rj = (var[sp1] - var[s]) * mask[s]
    rjm = (var[s] - var[sm1]) * mask[sm1]
    cr = limiter(_calc_cr(veros, rjp, rj, rjm, vel[s]))
    return velfac * vel[s] * (var[sp1] + var[s]) * 0.5 - np.abs(velfac * vel[s]) * ((1. - cr) + uCFL * cr) * rj * 0.5


@veros_method
def adv_flux_2nd(veros, adv_fe, adv_fn, adv_ft, var):
    """
    2th order advective tracer flux
    """
    adv_fe[1:-2, 2:-2, :] = 0.5 * (var[1:-2, 2:-2, :] + var[2:-1, 2:-2, :]) \
        * veros.u[1:-2, 2:-2, :, veros.tau] * veros.maskU[1:-2, 2:-2, :]
    adv_fn[2:-2, 1:-2, :] = veros.cosu[np.newaxis, 1:-2, np.newaxis] * 0.5 * (var[2:-2, 1:-2, :] + var[2:-2, 2:-1, :]) \
        * veros.v[2:-2, 1:-2, :, veros.tau] * veros.maskV[2:-2, 1:-2, :]
    adv_ft[2:-2, 2:-2, :-1] = 0.5 * (var[2:-2, 2:-2, :-1] + var[2:-2, 2:-2, 1:]) \
        * veros.w[2:-2, 2:-2, :-1, veros.tau] * veros.maskW[2:-2, 2:-2, :-1]
    adv_ft[:, :, -1] = 0.


@veros_method
def adv_flux_superbee(veros, adv_fe, adv_fn, adv_ft, var):
    """
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
    adv_fe[1:-2, 2:-2, :] = _adv_superbee(veros, veros.u[..., veros.tau],
                                          var, veros.maskU, veros.dxt, 0)
    adv_fn[2:-2, 1:-2, :] = _adv_superbee(veros, veros.v[..., veros.tau],
                                          var, veros.maskV, veros.dyt, 1)
    adv_ft[2:-2, 2:-2, :-1] = _adv_superbee(veros, veros.w[..., veros.tau],
                                            var, veros.maskW, veros.dzt, 2)
    adv_ft[..., -1] = 0.


@veros_method
def calculate_velocity_on_wgrid(veros):
    """
    calculates advection velocity for tracer on W grid

    Note: this implementation is not strictly equal to the Fortran version. They only match
    if maskW has exactly one true value across each depth slice.
    """
    # lateral advection velocities on W grid
    veros.u_wgrid[:, :, :-1] = veros.u[:, :, 1:, veros.tau] * veros.maskU[:, :, 1:] * 0.5 \
        * veros.dzt[np.newaxis, np.newaxis, 1:] / veros.dzw[np.newaxis, np.newaxis, :-1] \
        + veros.u[:, :, :-1, veros.tau] * veros.maskU[:, :, :-1] * 0.5 \
        * veros.dzt[np.newaxis, np.newaxis, :-1] / veros.dzw[np.newaxis, np.newaxis, :-1]
    veros.v_wgrid[:, :, :-1] = veros.v[:, :, 1:, veros.tau] * veros.maskV[:, :, 1:] * 0.5 \
        * veros.dzt[np.newaxis, np.newaxis, 1:] / veros.dzw[np.newaxis, np.newaxis, :-1] \
        + veros.v[:, :, :-1, veros.tau] * veros.maskV[:, :, :-1] * 0.5 \
        * veros.dzt[np.newaxis, np.newaxis, :-1] / veros.dzw[np.newaxis, np.newaxis, :-1]
    veros.u_wgrid[:, :, -1] = veros.u[:, :, -1, veros.tau] * \
        veros.maskU[:, :, -1] * 0.5 * veros.dzt[-1:] / veros.dzw[-1:]
    veros.v_wgrid[:, :, -1] = veros.v[:, :, -1, veros.tau] * \
        veros.maskV[:, :, -1] * 0.5 * veros.dzt[-1:] / veros.dzw[-1:]

    # redirect velocity at bottom and at topography
    veros.u_wgrid[:, :, 0] = veros.u_wgrid[:, :, 0] + veros.u[:, :, 0, veros.tau] \
        * veros.maskU[:, :, 0] * 0.5 * veros.dzt[0:1] / veros.dzw[0:1]
    veros.v_wgrid[:, :, 0] = veros.v_wgrid[:, :, 0] + veros.v[:, :, 0, veros.tau] \
        * veros.maskV[:, :, 0] * 0.5 * veros.dzt[0:1] / veros.dzw[0:1]
    mask = veros.maskW[:-1, :, :-1] * veros.maskW[1:, :, :-1]
    veros.u_wgrid[:-1, :, 1:] += (veros.u_wgrid[:-1, :, :-1] * veros.dzw[np.newaxis, np.newaxis, :-1]
                                  / veros.dzw[np.newaxis, np.newaxis, 1:]) * (1. - mask)
    veros.u_wgrid[:-1, :, :-1] *= mask
    mask = veros.maskW[:, :-1, :-1] * veros.maskW[:, 1:, :-1]
    veros.v_wgrid[:, :-1, 1:] += (veros.v_wgrid[:, :-1, :-1] * veros.dzw[np.newaxis, np.newaxis, :-1]
                                  / veros.dzw[np.newaxis, np.newaxis, 1:]) * (1. - mask)
    veros.v_wgrid[:, :-1, :-1] *= mask

    # vertical advection velocity on W grid from continuity
    veros.w_wgrid[:, :, 0] = 0.
    veros.w_wgrid[1:, 1:, :] = np.cumsum(-veros.dzw[np.newaxis, np.newaxis, :] *
                                         ((veros.u_wgrid[1:, 1:, :] - veros.u_wgrid[:-1, 1:, :]) / (veros.cost[np.newaxis, 1:, np.newaxis] * veros.dxt[1:, np.newaxis, np.newaxis])
                                          + (veros.cosu[np.newaxis, 1:, np.newaxis] * veros.v_wgrid[1:, 1:, :] -
                                             veros.cosu[np.newaxis, :-1, np.newaxis] * veros.v_wgrid[1:, :-1, :])
                                          / (veros.cost[np.newaxis, 1:, np.newaxis] * veros.dyt[np.newaxis, 1:, np.newaxis])), axis=2)


@veros_method
def adv_flux_superbee_wgrid(veros, adv_fe, adv_fn, adv_ft, var):
    """
    Calculates advection of a tracer defined on Wgrid
    """
    maskUtr = np.zeros_like(adv_fe)
    maskUtr[:-1, :, :] = veros.maskW[1:, :, :] * veros.maskW[:-1, :, :]
    adv_fe[1:-2, 2:-2, :] = _adv_superbee(veros, veros.u_wgrid, var, maskUtr, veros.dxt, 0)

    maskVtr = np.zeros_like(adv_fn)
    maskVtr[:, :-1, :] = veros.maskW[:, 1:, :] * veros.maskW[:, :-1, :]
    adv_fn[2:-2, 1:-2, :] = _adv_superbee(veros, veros.v_wgrid, var, maskVtr, veros.dyt, 1)

    maskWtr = np.zeros_like(adv_ft)
    maskWtr[:, :, :-1] = veros.maskW[:, :, 1:] * veros.maskW[:, :, :-1]
    adv_ft[2:-2, 2:-2, :-1] = _adv_superbee(veros, veros.w_wgrid, var, maskWtr, veros.dzw, 2)
    adv_ft[..., -1] = 0.0


@veros_method
def adv_flux_upwind_wgrid(veros, adv_fe, adv_fn, adv_ft, var):
    """
    Calculates advection of a tracer defined on Wgrid
    """
    maskUtr = veros.maskW[2:-1, 2:-2, :] * veros.maskW[1:-2, 2:-2, :]
    rj = (var[2:-1, 2:-2, :] - var[1:-2, 2:-2, :]) * maskUtr
    adv_fe[1:-2, 2:-2, :] = veros.u_wgrid[1:-2, 2:-2, :] * (var[2:-1, 2:-2, :] + var[1:-2, 2:-2, :]) * 0.5 \
        - np.abs(veros.u_wgrid[1:-2, 2:-2, :]) * rj * 0.5

    maskVtr = veros.maskW[2:-2, 2:-1, :] * veros.maskW[2:-2, 1:-2, :]
    rj = (var[2:-2, 2:-1, :] - var[2:-2, 1:-2, :]) * maskVtr
    adv_fn[2:-2, 1:-2, :] = veros.cosu[np.newaxis, 1:-2, np.newaxis] * veros.v_wgrid[2:-2, 1:-2, :] * \
        (var[2:-2, 2:-1, :] + var[2:-2, 1:-2, :]) * 0.5 \
        - np.abs(veros.cosu[np.newaxis, 1:-2, np.newaxis] * veros.v_wgrid[2:-2, 1:-2, :]) * rj * 0.5

    maskWtr = veros.maskW[2:-2, 2:-2, 1:] * veros.maskW[2:-2, 2:-2, :-1]
    rj = (var[2:-2, 2:-2, 1:] - var[2:-2, 2:-2, :-1]) * maskWtr
    adv_ft[2:-2, 2:-2, :-1] = veros.w_wgrid[2:-2, 2:-2, :-1] * (var[2:-2, 2:-2, 1:] + var[2:-2, 2:-2, :-1]) * 0.5 \
        - np.abs(veros.w_wgrid[2:-2, 2:-2, :-1]) * rj * 0.5
    adv_ft[:, :, -1] = 0.

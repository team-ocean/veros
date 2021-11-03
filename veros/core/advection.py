from veros import veros_kernel, veros_routine, KernelOutput
from veros.variables import allocate
from veros.core.utilities import pad_z_edges
from veros.core.operators import numpy as npx, update, update_add, update_multiply, at


@veros_kernel
def _calc_cr(rjp, rj, rjm, vel):
    """
    Calculates cr value used in superbee advection scheme
    """
    eps = 1e-20  # prevent division by 0
    return npx.where(vel > 0.0, rjm, rjp) / npx.where(npx.abs(rj) < eps, eps, rj)


@veros_kernel
def limiter(cr):
    return npx.maximum(npx.clip(2 * cr, 0, 1), npx.clip(cr, 0, 2))


@veros_kernel(static_args=("axis"))
def _adv_superbee(state, vel, var, mask, dx, axis):
    vs = state.variables
    settings = state.settings

    if axis == 0:
        sm1, s, sp1, sp2 = ((slice(1 + n, -2 + n or None), slice(2, -2), slice(None)) for n in range(-1, 3))
        dx = vs.cost[npx.newaxis, 2:-2, npx.newaxis] * dx[1:-2, npx.newaxis, npx.newaxis]
    elif axis == 1:
        sm1, s, sp1, sp2 = ((slice(2, -2), slice(1 + n, -2 + n or None), slice(None)) for n in range(-1, 3))
        dx = (vs.cost * dx)[npx.newaxis, 1:-2, npx.newaxis]
    elif axis == 2:
        sm1, s, sp1, sp2 = ((slice(2, -2), slice(2, -2), slice(1 + n, -2 + n or None)) for n in range(-1, 3))
        dx = dx[npx.newaxis, npx.newaxis, :-1]
        vel, var, mask = (pad_z_edges(a) for a in (vel, var, mask))
    else:
        raise ValueError("axis must be 0, 1, or 2")

    rjp = (var[sp2] - var[sp1]) * mask[sp1]
    rj = (var[sp1] - var[s]) * mask[s]
    rjm = (var[s] - var[sm1]) * mask[sm1]
    cr = limiter(_calc_cr(rjp, rj, rjm, vel[s]))

    if axis == 1:
        vel = vel * vs.cosu[npx.newaxis, :, npx.newaxis]

    uCFL = npx.abs(vel[s] * settings.dt_tracer / dx)
    return vel[s] * (var[sp1] + var[s]) * 0.5 - npx.abs(vel[s]) * ((1.0 - cr) + uCFL * cr) * rj * 0.5


@veros_kernel
def adv_flux_2nd(state, var):
    """
    2nd order advective tracer flux
    """
    vs = state.variables

    adv_fe = allocate(state.dimensions, ("xt", "yt", "zt"))
    adv_fn = allocate(state.dimensions, ("xt", "yt", "zt"))
    adv_ft = allocate(state.dimensions, ("xt", "yt", "zt"))

    adv_fe = update(
        adv_fe,
        at[1:-2, 2:-2, :],
        0.5 * (var[1:-2, 2:-2, :] + var[2:-1, 2:-2, :]) * vs.u[1:-2, 2:-2, :, vs.tau] * vs.maskU[1:-2, 2:-2, :],
    )
    adv_fn = update(
        adv_fn,
        at[2:-2, 1:-2, :],
        vs.cosu[npx.newaxis, 1:-2, npx.newaxis]
        * 0.5
        * (var[2:-2, 1:-2, :] + var[2:-2, 2:-1, :])
        * vs.v[2:-2, 1:-2, :, vs.tau]
        * vs.maskV[2:-2, 1:-2, :],
    )
    adv_ft = update(
        adv_ft,
        at[2:-2, 2:-2, :-1],
        0.5 * (var[2:-2, 2:-2, :-1] + var[2:-2, 2:-2, 1:]) * vs.w[2:-2, 2:-2, :-1, vs.tau] * vs.maskW[2:-2, 2:-2, :-1],
    )
    adv_ft = update(adv_ft, at[:, :, -1], 0.0)

    return adv_fe, adv_fn, adv_ft


@veros_kernel
def adv_flux_superbee(state, var):
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
    vs = state.variables

    adv_fe = allocate(state.dimensions, ("xt", "yt", "zt"))
    adv_fn = allocate(state.dimensions, ("xt", "yt", "zt"))
    adv_ft = allocate(state.dimensions, ("xt", "yt", "zt"))

    adv_fe = update(adv_fe, at[1:-2, 2:-2, :], _adv_superbee(state, vs.u[..., vs.tau], var, vs.maskU, vs.dxt, 0))
    adv_fn = update(adv_fn, at[2:-2, 1:-2, :], _adv_superbee(state, vs.v[..., vs.tau], var, vs.maskV, vs.dyt, 1))
    adv_ft = update(adv_ft, at[2:-2, 2:-2, :-1], _adv_superbee(state, vs.w[..., vs.tau], var, vs.maskW, vs.dzt, 2))
    adv_ft = update(adv_ft, at[..., -1], 0.0)

    return adv_fe, adv_fn, adv_ft


@veros_routine
def calculate_velocity_on_wgrid(state):
    vs = state.variables
    vs.update(calculate_velocity_on_wgrid_kernel(state))


@veros_kernel
def calculate_velocity_on_wgrid_kernel(state):
    """
    calculates advection velocity for tracer on vs.W grid

    Note: this implementation is not strictly equal to the Fortran version. They only match
    if vs.maskW has exactly one true value across each depth slice.
    """
    vs = state.variables

    # lateral advection velocities on W grid
    vs.u_wgrid = update(
        vs.u_wgrid,
        at[:, :, :-1],
        vs.u[:, :, 1:, vs.tau]
        * vs.maskU[:, :, 1:]
        * 0.5
        * vs.dzt[npx.newaxis, npx.newaxis, 1:]
        / vs.dzw[npx.newaxis, npx.newaxis, :-1]
        + vs.u[:, :, :-1, vs.tau]
        * vs.maskU[:, :, :-1]
        * 0.5
        * vs.dzt[npx.newaxis, npx.newaxis, :-1]
        / vs.dzw[npx.newaxis, npx.newaxis, :-1],
    )
    vs.v_wgrid = update(
        vs.v_wgrid,
        at[:, :, :-1],
        vs.v[:, :, 1:, vs.tau]
        * vs.maskV[:, :, 1:]
        * 0.5
        * vs.dzt[npx.newaxis, npx.newaxis, 1:]
        / vs.dzw[npx.newaxis, npx.newaxis, :-1]
        + vs.v[:, :, :-1, vs.tau]
        * vs.maskV[:, :, :-1]
        * 0.5
        * vs.dzt[npx.newaxis, npx.newaxis, :-1]
        / vs.dzw[npx.newaxis, npx.newaxis, :-1],
    )
    vs.u_wgrid = update(
        vs.u_wgrid, at[:, :, -1], vs.u[:, :, -1, vs.tau] * vs.maskU[:, :, -1] * 0.5 * vs.dzt[-1:] / vs.dzw[-1:]
    )
    vs.v_wgrid = update(
        vs.v_wgrid, at[:, :, -1], vs.v[:, :, -1, vs.tau] * vs.maskV[:, :, -1] * 0.5 * vs.dzt[-1:] / vs.dzw[-1:]
    )

    # redirect velocity at bottom and at topography
    vs.u_wgrid = update(
        vs.u_wgrid,
        at[:, :, 0],
        vs.u_wgrid[:, :, 0] + vs.u[:, :, 0, vs.tau] * vs.maskU[:, :, 0] * 0.5 * vs.dzt[0] / vs.dzw[0],
    )
    vs.v_wgrid = update(
        vs.v_wgrid,
        at[:, :, 0],
        vs.v_wgrid[:, :, 0] + vs.v[:, :, 0, vs.tau] * vs.maskV[:, :, 0] * 0.5 * vs.dzt[0] / vs.dzw[0],
    )
    mask = vs.maskW[:-1, :, :-1] * vs.maskW[1:, :, :-1]
    vs.u_wgrid = update_add(
        vs.u_wgrid,
        at[:-1, :, 1:],
        (vs.u_wgrid[:-1, :, :-1] * vs.dzw[npx.newaxis, npx.newaxis, :-1] / vs.dzw[npx.newaxis, npx.newaxis, 1:])
        * (1.0 - mask),
    )
    vs.u_wgrid = update_multiply(vs.u_wgrid, at[:-1, :, :-1], mask)
    mask = vs.maskW[:, :-1, :-1] * vs.maskW[:, 1:, :-1]
    vs.v_wgrid = update_add(
        vs.v_wgrid,
        at[:, :-1, 1:],
        (vs.v_wgrid[:, :-1, :-1] * vs.dzw[npx.newaxis, npx.newaxis, :-1] / vs.dzw[npx.newaxis, npx.newaxis, 1:])
        * (1.0 - mask),
    )
    vs.v_wgrid = update_multiply(vs.v_wgrid, at[:, :-1, :-1], mask)

    # vertical advection velocity on W grid from continuity
    vs.w_wgrid = update(vs.w_wgrid, at[:, :, 0], 0.0)
    vs.w_wgrid = update(
        vs.w_wgrid,
        at[1:, 1:, :],
        npx.cumsum(
            -vs.dzw[npx.newaxis, npx.newaxis, :]
            * (
                (vs.u_wgrid[1:, 1:, :] - vs.u_wgrid[:-1, 1:, :])
                / (vs.cost[npx.newaxis, 1:, npx.newaxis] * vs.dxt[1:, npx.newaxis, npx.newaxis])
                + (
                    vs.cosu[npx.newaxis, 1:, npx.newaxis] * vs.v_wgrid[1:, 1:, :]
                    - vs.cosu[npx.newaxis, :-1, npx.newaxis] * vs.v_wgrid[1:, :-1, :]
                )
                / (vs.cost[npx.newaxis, 1:, npx.newaxis] * vs.dyt[npx.newaxis, 1:, npx.newaxis])
            ),
            axis=2,
        ),
    )

    return KernelOutput(u_wgrid=vs.u_wgrid, v_wgrid=vs.v_wgrid, w_wgrid=vs.w_wgrid)


@veros_kernel
def adv_flux_superbee_wgrid(state, var):
    """
    Calculates advection of a tracer defined on Wgrid
    """
    vs = state.variables

    adv_fe = allocate(state.dimensions, ("xt", "yt", "zt"))
    adv_fn = allocate(state.dimensions, ("xt", "yt", "zt"))
    adv_ft = allocate(state.dimensions, ("xt", "yt", "zt"))

    maskUtr = allocate(state.dimensions, ("xt", "yt", "zw"))
    maskUtr = update(maskUtr, at[:-1, :, :], vs.maskW[1:, :, :] * vs.maskW[:-1, :, :])
    adv_fe = update(adv_fe, at[1:-2, 2:-2, :], _adv_superbee(state, vs.u_wgrid, var, maskUtr, vs.dxt, axis=0))

    maskVtr = allocate(state.dimensions, ("xt", "yt", "zw"))
    maskVtr = update(maskVtr, at[:, :-1, :], vs.maskW[:, 1:, :] * vs.maskW[:, :-1, :])
    adv_fn = update(adv_fn, at[2:-2, 1:-2, :], _adv_superbee(state, vs.v_wgrid, var, maskVtr, vs.dyt, axis=1))

    maskWtr = allocate(state.dimensions, ("xt", "yt", "zw"))
    maskWtr = update(maskWtr, at[:, :, :-1], vs.maskW[:, :, 1:] * vs.maskW[:, :, :-1])
    adv_ft = update(adv_ft, at[2:-2, 2:-2, :-1], _adv_superbee(state, vs.w_wgrid, var, maskWtr, vs.dzw, axis=2))
    adv_ft = update(adv_ft, at[..., -1], 0.0)

    return adv_fe, adv_fn, adv_ft


@veros_kernel
def adv_flux_upwind_wgrid(state, var):
    """
    Calculates advection of a tracer defined on Wgrid
    """
    vs = state.variables

    adv_fe = allocate(state.dimensions, ("xt", "yt", "zt"))
    adv_fn = allocate(state.dimensions, ("xt", "yt", "zt"))
    adv_ft = allocate(state.dimensions, ("xt", "yt", "zt"))

    maskUtr = vs.maskW[2:-1, 2:-2, :] * vs.maskW[1:-2, 2:-2, :]
    rj = (var[2:-1, 2:-2, :] - var[1:-2, 2:-2, :]) * maskUtr
    adv_fe = update(
        adv_fe,
        at[1:-2, 2:-2, :],
        vs.u_wgrid[1:-2, 2:-2, :] * (var[2:-1, 2:-2, :] + var[1:-2, 2:-2, :]) * 0.5
        - npx.abs(vs.u_wgrid[1:-2, 2:-2, :]) * rj * 0.5,
    )

    maskVtr = vs.maskW[2:-2, 2:-1, :] * vs.maskW[2:-2, 1:-2, :]
    rj = (var[2:-2, 2:-1, :] - var[2:-2, 1:-2, :]) * maskVtr
    adv_fn = update(
        adv_fn,
        at[2:-2, 1:-2, :],
        vs.cosu[npx.newaxis, 1:-2, npx.newaxis]
        * vs.v_wgrid[2:-2, 1:-2, :]
        * (var[2:-2, 2:-1, :] + var[2:-2, 1:-2, :])
        * 0.5
        - npx.abs(vs.cosu[npx.newaxis, 1:-2, npx.newaxis] * vs.v_wgrid[2:-2, 1:-2, :]) * rj * 0.5,
    )

    maskWtr = vs.maskW[2:-2, 2:-2, 1:] * vs.maskW[2:-2, 2:-2, :-1]
    rj = (var[2:-2, 2:-2, 1:] - var[2:-2, 2:-2, :-1]) * maskWtr
    adv_ft = update(
        adv_ft,
        at[2:-2, 2:-2, :-1],
        vs.w_wgrid[2:-2, 2:-2, :-1] * (var[2:-2, 2:-2, 1:] + var[2:-2, 2:-2, :-1]) * 0.5
        - npx.abs(vs.w_wgrid[2:-2, 2:-2, :-1]) * rj * 0.5,
    )
    adv_ft = update(adv_ft, at[:, :, -1], 0.0)

    return adv_fe, adv_fn, adv_ft

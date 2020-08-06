from veros.core.operators import numpy as np

from veros import veros_kernel, veros_routine, run_kernel
from veros.core import density, diffusion, utilities
from veros.core.operators import update, at, solve_tridiagonal


@veros_kernel
def u_centered_grid(dyt, dyu, yt, yu):
    yu = update(yu, at[0], 0)
    yu = update(yu, at[1:], np.cumsum(dyt[1:]))

    yt = update(yt, at[0], yu[0] - dyt[0] * 0.5)
    yt = update(yt, at[1:], 2 * yu[:-1])

    alternating_pattern = np.ones_like(yt)
    alternating_pattern = update(alternating_pattern, at[::2], -1)
    yt = update(yt, at[...], alternating_pattern * np.cumsum(alternating_pattern * yt))

    dyu = update(dyu, at[:-1], yt[1:] - yt[:-1])
    dyu = update(dyu, at[-1], 2 * dyt[-1] - dyu[-2])
    return dyu, yt, yu


@veros_kernel(
    static_args=('enable_cyclic_x', 'coord_degree')
)
def calc_grid_kernel(dxt, dxu, dyt, dyu,
                     xt, xu, yt, yu,
                     dzt, dzw, zt, zw,
                     x_origin, y_origin,
                     cost, cosu, tantr,
                     pi, radius, degtom,
                     area_t, area_u, area_v,
                     enable_cyclic_x, coord_degree):
    if enable_cyclic_x:
        dxt = update(dxt, at[-2:], dxt[2:4])
        dxt = update(dxt, at[:2], dxt[-4:-2])
    else:
        dxt = update(dxt, at[-2:], dxt[-3])
        dxt = update(dxt, at[:2], dxt[2])

    dyt = update(dyt, at[-2:], dyt[-3])
    dyt = update(dyt, at[:2], dyt[2])

    """
    grid in east/west direction
    """
    dxu, xt, xu = u_centered_grid(dxt, dxu, xt, xu)
    xt += x_origin - xu[2]
    xu += x_origin - xu[2]

    if enable_cyclic_x:
        xt = update(xt, at[-2:], xt[2:4])
        xt = update(xt, at[:2], xt[-4:-2])
        xu = update(xu, at[-2:], xt[2:4])
        xu = update(xu, at[:2], xu[-4:-2])
        dxu = update(dxu, at[-2:], dxu[2:4])
        dxu = update(dxu, at[:2], dxu[-4:-2])

    """
    grid in north/south direction
    """
    dyu, yt, yu = u_centered_grid(dyt, dyu, yt, yu)
    yt += y_origin - yu[2]
    yu += y_origin - yu[2]

    if coord_degree:
        """
        convert from degrees to pseudo cartesian grid
        """
        dxt *= degtom
        dxu *= degtom
        dyt *= degtom
        dyu *= degtom

    """
    grid in vertical direction
    """
    dzw, zt, zw = u_centered_grid(dzt, dzw, zt, zw)
    zt -= zw[-1]
    zw -= zw[-1]  # enforce 0 boundary height

    """
    metric factors
    """
    if coord_degree:
        cost = update(cost, at[...], np.cos(yt * pi / 180.))
        cosu = update(cosu, at[...], np.cos(yu * pi / 180.))
        tantr = update(tantr, at[...], np.tan(yt * pi / 180.) / radius)
    else:
        cost = update(cost, at[...], 1.0)
        cosu = update(cosu, at[...], 1.0)
        tantr = update(tantr, at[...], 0.0)

    """
    precalculate area of boxes
    """
    area_t = update(area_t, at[...], cost * dyt * dxt[:, np.newaxis])
    area_u = update(area_u, at[...], cost * dyt * dxu[:, np.newaxis])
    area_v = update(area_v, at[...], cosu * dyu * dxt[:, np.newaxis])

    return (
        dxt, dxu, dyt, dyu,
        xt, xu, yt, yu,
        dzt, dzw, zt, zw,
        cost, cosu, tantr,
        area_t, area_u, area_v,
    )


@veros_routine(
    inputs=(
        'dxt', 'dxu', 'dyt', 'dyu',
        'xt', 'xu', 'yt', 'yu',
        'dzt', 'dzw', 'zt', 'zw',
        'x_origin', 'y_origin',
        'cost', 'cosu', 'tantr',
        'pi', 'radius', 'degtom',
        'area_t', 'area_u', 'area_v',
    ),
    outputs=(
        'dxt', 'dxu', 'dyt', 'dyu',
        'xt', 'xu', 'yt', 'yu',
        'dzt', 'dzw', 'zt', 'zw',
        'cost', 'cosu', 'tantr',
        'area_t', 'area_u', 'area_v',
    ),
    settings=('enable_cyclic_x', 'coord_degree'),
    dist_safe=False,
)
def calc_grid(vs):
    """
    setup grid based on dxt,dyt,dzt and x_origin, y_origin
    """

    (
        dxt, dxu, dyt, dyu,
        xt, xu, yt, yu,
        dzt, dzw, zt, zw,
        cost, cosu, tantr,
        area_t, area_u, area_v,
    ) = run_kernel(calc_grid_kernel, vs)

    return dict(
        dxt=dxt,
        dxu=dxu,
        dyt=dyt,
        dyu=dyu,
        xt=xt,
        xu=xu,
        yt=yt,
        yu=yu,
        dzt=dzt,
        dzw=dzw,
        zt=zt,
        zw=zw,
        cost=cost,
        cosu=cosu,
        tantr=tantr,
        area_t=area_t,
        area_u=area_u,
        area_v=area_v
    )


@veros_routine(
    inputs=(
        'dyu', 'beta', 'coriolis_t'
    ),
    outputs=('beta'),
    settings=('enable_cyclic_x')
)
def calc_beta(vs):
    """
    calculate beta = df/dy
    """
    beta = update(vs.beta, at[:, 2:-2], 0.5 * ((vs.coriolis_t[:, 3:-1] - vs.coriolis_t[:, 2:-2]) / vs.dyu[2:-2]
                           + (vs.coriolis_t[:, 2:-2] - vs.coriolis_t[:, 1:-3]) / vs.dyu[1:-3]))

    beta = utilities.enforce_boundaries(beta, vs.enable_cyclic_x)

    return dict(beta=beta)


@veros_routine(
    inputs=(
        'kbot', 'maskT', 'maskU', 'maskV', 'maskW', 'maskZ', 'ht', 'hu', 'hv', 'hur', 'hvr', 'dzt'
    ),
    outputs=(
        'maskT', 'maskU', 'maskV', 'maskW', 'maskZ', 'ht', 'hu', 'hv', 'hur', 'hvr', 'kbot'
    ),
    settings=('enable_cyclic_x')
)
def calc_topo(vs):
    """
    calulate masks, total depth etc
    """

    """
    close domain
    """
    kbot, maskT, maskU, maskV, maskW, maskZ, ht, hu, hv, hur, hvr, dzt, enable_cyclic_x = (
        getattr(vs, k) for k in ('kbot', 'maskT', 'maskU', 'maskV', 'maskW', 'maskZ', 'ht', 'hu', 'hv', 'hur', 'hvr', 'dzt', 'enable_cyclic_x')
    )

    kbot = update(kbot, at[:, :2], 0)
    kbot = update(kbot, at[:, -2:], 0)

    kbot = utilities.enforce_boundaries(kbot, enable_cyclic_x)

    if not enable_cyclic_x:
        kbot = update(kbot, at[:2, :], 0)
        kbot = update(kbot, at[-2:, :], 0)

    """
    Land masks
    """
    maskT = update(maskT, at[...], 0.0)
    land_mask = kbot > 0
    ks = np.arange(maskT.shape[2])[np.newaxis, np.newaxis, :]

    maskT = update(maskT, at[...], land_mask[..., np.newaxis] & (kbot[..., np.newaxis] - 1 <= ks))
    maskT = utilities.enforce_boundaries(maskT, enable_cyclic_x)

    maskU = update(maskU, at[...], maskT)
    maskU = update(maskU, at[:-1, :, :], np.minimum(maskT[:-1, :, :], maskT[1:, :, :]))
    maskU = utilities.enforce_boundaries(maskU, enable_cyclic_x)

    maskV = update(maskV, at[...], maskT)
    maskV = update(maskV, at[:, :-1], np.minimum(maskT[:, :-1], maskT[:, 1:]))
    maskV = utilities.enforce_boundaries(maskV, enable_cyclic_x)

    maskZ = update(maskZ, at[...], maskT)
    maskZ = update(maskZ, at[:-1, :-1], np.minimum(np.minimum(maskT[:-1, :-1], maskT[:-1, 1:]), maskT[1:, :-1]))
    maskZ = utilities.enforce_boundaries(maskZ, enable_cyclic_x)

    maskW = update(maskW, at[...], maskT)
    maskW = update(maskW, at[:, :, :-1], np.minimum(maskT[:, :, :-1], maskT[:, :, 1:]))

    """
    total depth
    """
    ht = np.sum(maskT * dzt[np.newaxis, np.newaxis, :], axis=2)
    hu = np.sum(maskU * dzt[np.newaxis, np.newaxis, :], axis=2)
    hv = np.sum(maskV * dzt[np.newaxis, np.newaxis, :], axis=2)

    mask = (hu == 0).astype('float')
    hur = 1. / (hu + mask) * (1 - mask)
    mask = (hv == 0).astype('float')
    hvr = 1. / (hv + mask) * (1 - mask)

    return dict(
        maskT=maskT,
        maskU=maskU,
        maskV=maskV,
        maskW=maskW,
        maskZ=maskZ,
        ht=ht,
        hu=hu,
        hv=hv,
        hur=hur,
        hvr=hvr,
        kbot=kbot
    )


@veros_routine(
    inputs=(
        'salt', 'temp', 'rho', 'Hd', 'int_drhodT', 'int_drhodS', 'Nsqr',
        'dzw', 'zt', 'maskT', 'maskW', 'grav', 'rho_0',
    ),
    outputs=('salt', 'temp', 'rho', 'Hd', 'int_drhodT', 'int_drhodS', 'Nsqr'),
    settings=('enable_cyclic_x', 'eq_of_state_type')
)
def calc_initial_conditions(vs):
    """
    calculate dyn. enthalp, etc
    """
    (salt, temp, rho, Hd, int_drhodT, int_drhodS, Nsqr,
     dzw, zt, maskT, maskW, grav, rho_0, enable_cyclic_x, eq_of_state_type) = (
        getattr(vs, k) for k in (
            'salt', 'temp', 'rho', 'Hd', 'int_drhodT', 'int_drhodS', 'Nsqr',
            'dzw', 'zt', 'maskT', 'maskW', 'grav', 'rho_0', 'enable_cyclic_x', 'eq_of_state_type'
        )
    )
    if np.any(salt < 0.0):
        raise RuntimeError('encountered negative salinity')

    temp = utilities.enforce_boundaries(temp, enable_cyclic_x)
    salt = utilities.enforce_boundaries(salt, enable_cyclic_x)

    rho = density.get_rho(eq_of_state_type, salt, temp, np.abs(zt)[:, np.newaxis]) * maskT[..., np.newaxis]
    Hd = density.get_dyn_enthalpy(eq_of_state_type, salt, temp, np.abs(zt)[:, np.newaxis]) * maskT[..., np.newaxis]
    int_drhodT = update(
        int_drhodT, at[...],
        density.get_int_drhodT(eq_of_state_type, salt, temp, np.abs(zt)[:, np.newaxis])
    )
    int_drhodS = update(
        int_drhodS, at[...],
        density.get_int_drhodS(eq_of_state_type, salt, temp, np.abs(zt)[:, np.newaxis])
    )

    fxa = -grav / rho_0 / dzw[np.newaxis, np.newaxis, :] * maskW
    Nsqr = update(Nsqr, at[:, :, :-1, :], fxa[:, :, :-1, np.newaxis] \
        * (density.get_rho(eq_of_state_type, salt[:, :, 1:, :], temp[:, :, 1:, :], np.abs(zt)[:-1, np.newaxis])
           - rho[:, :, :-1, :]))
    Nsqr = update(Nsqr, at[:, :, -1, :], Nsqr[:, :, -2, :])

    return dict(
        salt=salt,
        temp=temp,
        rho=rho,
        Hd=Hd,
        int_drhodT=int_drhodT,
        int_drhodS=int_drhodS,
        Nsqr=Nsqr
    )


@veros_kernel
def ugrid_to_tgrid(a, dxt, dxu):
    b = np.zeros_like(a)
    b = update(b, at[2:-2, :, :], (dxu[2:-2, np.newaxis, np.newaxis] * a[2:-2, :, :]
                     + dxu[1:-3, np.newaxis, np.newaxis] * a[1:-3, :, :]) \
        / (2 * dxt[2:-2, np.newaxis, np.newaxis]))
    return b


@veros_kernel
def vgrid_to_tgrid(a, area_v, area_t):
    b = np.zeros_like(a)
    b = update(b, at[:, 2:-2, :], (area_v[:, 2:-2, np.newaxis] * a[:, 2:-2, :]
                     + area_v[:, 1:-3, np.newaxis] * a[:, 1:-3, :]) \
        / (2 * area_t[:, 2:-2, np.newaxis]))
    return b


@veros_kernel
def solve_tridiag(a, b, c, d):
    """
    Solves a tridiagonal matrix system with diagonals a, b, c and RHS vector d.
    Uses LAPACK when running with NumPy, and otherwise the Thomas algorithm iterating over the
    last axis of the input arrays.
    """
    assert a.shape == b.shape and a.shape == c.shape and a.shape == d.shape
    return solve_tridiagonal(a, b, c, d)


@veros_kernel(static_args=('nz',))
def calc_diss_u(diss, kbot, nz, dzw, dxt, dxu):
    ks = np.zeros_like(kbot)
    ks = update(ks, at[1:-2, 2:-2], np.maximum(kbot[1:-2, 2:-2], kbot[2:-1, 2:-2]))
    diss_u = diffusion.dissipation_on_wgrid(diss, nz, dzw, ks)
    return ugrid_to_tgrid(diss_u, dxt, dxu)


@veros_kernel(static_args=('nz',))
def calc_diss_v(diss, kbot, nz, dzw, area_v, area_t):
    ks = np.zeros_like(kbot)
    ks = update(ks, at[2:-2, 1:-2], np.maximum(kbot[2:-2, 1:-2], kbot[2:-2, 2:-1]))
    diss_v = diffusion.dissipation_on_wgrid(diss, nz, dzw, ks)
    return vgrid_to_tgrid(diss_v, area_v, area_t)

from . import density, diffusion, utilities
from veros.core import veros_kernel


@veros_kernel(
    dist_safe=False,
    local_variables=(
    'dxt', 'dxu', 'xt', 'xu',
    'dyt', 'dyu', 'yt', 'yu',
    'dzt', 'dzw', 'zt', 'zw',
    'cost', 'cosu', 'tantr',
    'area_t', 'area_u', 'area_v',
    ),
    static_args=('enable_cyclic_x', 'coord_degree',)
)
def calc_grid(dxt, dxu, dyt, dyu, xt, xu, yt, yu, dzt, dzw, zt, zw, x_origin, y_origin,
              cost, cosu, tantr, pi, radius, area_t, area_u, area_v, degtom,
              enable_cyclic_x, coord_degree):
    """
    setup grid based on dxt,dyt,dzt and x_origin, y_origin
    """

    def u_centered_grid(dyt, dyu, yt, yu):
        yu[0] = 0
        yu[1:] = np.cumsum(dyt[1:])

        yt[0] = yu[0] - dyt[0] * 0.5
        yt[1:] = 2 * yu[:-1]

        alternating_pattern = np.ones_like(yt)
        alternating_pattern[::2] = -1
        yt[...] = alternating_pattern * np.cumsum(alternating_pattern * yt)

        dyu[:-1] = yt[1:] - yt[:-1]
        dyu[-1] = 2 * dyt[-1] - dyu[-2]

    if enable_cyclic_x:
        dxt[-2:] = dxt[2:4]
        dxt[:2] = dxt[-4:-2]
    else:
        dxt[-2:] = dxt[-3]
        dxt[:2] = dxt[2]

    dyt[-2:] = dyt[-3]
    dyt[:2] = dyt[2]

    """
    grid in east/west direction
    """
    u_centered_grid(dxt, dxu, xt, xu)
    xt += x_origin - xu[2]
    xu += x_origin - xu[2]

    if enable_cyclic_x:
        xt[-2:] = xt[2:4]
        xt[:2] = xt[-4:-2]
        xu[-2:] = xt[2:4]
        xu[:2] = xu[-4:-2]
        dxu[-2:] = dxu[2:4]
        dxu[:2] = dxu[-4:-2]

    """
    grid in north/south direction
    """
    u_centered_grid(dyt, dyu, yt, yu)
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
    u_centered_grid(dzt, dzw, zt, zw)
    zt -= zw[-1]
    zw -= zw[-1]  # enforce 0 boundary height

    """
    metric factors
    """
    if coord_degree:
        cost[...] = np.cos(yt * pi / 180.)
        cosu[...] = np.cos(yu * pi / 180.)
        tantr[...] = np.tan(yt * pi / 180.) / radius
    else:
        cost[...] = 1.0
        cosu[...] = 1.0
        tantr[...] = 0.0

    """
    precalculate area of boxes
    """
    area_t[...] = cost * dyt * dxt[:, np.newaxis]
    area_u[...] = cost * dyt * dxu[:, np.newaxis]
    area_v[...] = cosu * dyu * dxt[:, np.newaxis]

    return dxt, dxu, dyt, dyu, xt, xu, yt, yu, dzt, dzw, zt, zw,\
        cost, cosu, tantr, area_t, area_u, area_v


@veros_kernel
def calc_beta(dyu, beta, coriolis_t, enable_cyclic_x):
    """
    calculate beta = df/dy
    """
    beta[:, 2:-2] = 0.5 * ((coriolis_t[:, 3:-1] - coriolis_t[:, 2:-2]) / dyu[2:-2]
                           + (coriolis_t[:, 2:-2] - coriolis_t[:, 1:-3]) / dyu[1:-3])

    utilities.enforce_boundaries(beta, enable_cyclic_x)

    return beta


@veros_kernel(static_args=('enable_cyclic_x',))
def calc_topo(kbot, maskT, maskU, maskV, maskW, maskZ, ht, hu, hv, hur, hvr, dzt, enable_cyclic_x):
    """
    calulate masks, total depth etc
    """

    """
    close domain
    """

    kbot[:, :2] = 0
    kbot[:, -2:] = 0

    utilities.enforce_boundaries(kbot, enable_cyclic_x)

    if not enable_cyclic_x:
        kbot[:2, :] = 0
        kbot[-2:, :] = 0

    """
    Land masks
    """
    maskT[...] = 0.0
    land_mask = kbot > 0
    ks = np.arange(maskT.shape[2])[np.newaxis, np.newaxis, :]
    maskT[...] = land_mask[..., np.newaxis] & (kbot[..., np.newaxis] - 1 <= ks)
    utilities.enforce_boundaries(maskT, enable_cyclic_x)
    maskU[...] = maskT
    maskU[:-1, :, :] = np.minimum(maskT[:-1, :, :], maskT[1:, :, :])
    utilities.enforce_boundaries(maskU, enable_cyclic_x)
    maskV[...] = maskT
    maskV[:, :-1] = np.minimum(maskT[:, :-1], maskT[:, 1:])
    utilities.enforce_boundaries(maskV, enable_cyclic_x)
    maskZ[...] = maskT
    maskZ[:-1, :-1] = np.minimum(np.minimum(maskT[:-1, :-1],
                                            maskT[:-1, 1:]),
                                 maskT[1:, :-1])
    utilities.enforce_boundaries(maskZ, enable_cyclic_x)
    maskW[...] = maskT
    maskW[:, :, :-1] = np.minimum(maskT[:, :, :-1], maskT[:, :, 1:])

    """
    total depth
    """
    ht[...] = np.sum(maskT * dzt[np.newaxis, np.newaxis, :], axis=2)
    hu[...] = np.sum(maskU * dzt[np.newaxis, np.newaxis, :], axis=2)
    hv[...] = np.sum(maskV * dzt[np.newaxis, np.newaxis, :], axis=2)

    mask = (hu == 0).astype(np.float)
    hur[...] = 1. / (hu + mask) * (1 - mask)
    mask = (hv == 0).astype(np.float)
    hvr[...] = 1. / (hv + mask) * (1 - mask)

    return maskT, maskU, maskV, maskW, maskZ, ht, hu, hv, hur, hvr


@veros_kernel
def calc_initial_conditions(salt, temp, rho, Hd, int_drhodT, int_drhodS, Nsqr, dzw, zt, maskT,
                            maskW, grav, rho_0, enable_cyclic_x):
    """
    calculate dyn. enthalp, etc
    """
    if np.any(salt < 0.0):
        raise RuntimeError('encountered negative salinity')

    utilities.enforce_boundaries(temp, enable_cyclic_x)
    utilities.enforce_boundaries(salt, enable_cyclic_x)

    rho[...] = density.get_rho(salt, temp, np.abs(zt)[:, np.newaxis]) \
        * maskT[..., np.newaxis]
    Hd[...] = density.get_dyn_enthalpy(salt, temp, np.abs(zt)[:, np.newaxis]) \
        * maskT[..., np.newaxis]
    int_drhodT[...] = density.get_int_drhodT(salt, temp, np.abs(zt)[:, np.newaxis])
    int_drhodS[...] = density.get_int_drhodS(salt, temp, np.abs(zt)[:, np.newaxis])

    fxa = -grav / rho_0 / dzw[np.newaxis, np.newaxis, :] * maskW
    Nsqr[:, :, :-1, :] = fxa[:, :, :-1, np.newaxis] \
        * (density.get_rho(salt[:, :, 1:, :], temp[:, :, 1:, :], np.abs(zt)[:-1, np.newaxis])
           - rho[:, :, :-1, :])
    Nsqr[:, :, -1, :] = Nsqr[:, :, -2, :]

    return salt, temp, rho, Hd, int_drhodT, int_drhodS, Nsqr


@veros_kernel
def ugrid_to_tgrid(a, dxt, dxu):
    b = np.zeros_like(a)
    b[2:-2, :, :] = (dxu[2:-2, np.newaxis, np.newaxis] * a[2:-2, :, :]
                     + dxu[1:-3, np.newaxis, np.newaxis] * a[1:-3, :, :]) \
        / (2 * dxt[2:-2, np.newaxis, np.newaxis])
    return b


@veros_kernel
def vgrid_to_tgrid(a, area_v, area_t):
    b = np.zeros_like(a)
    b[:, 2:-2, :] = (area_v[:, 2:-2, np.newaxis] * a[:, 2:-2, :]
                     + area_v[:, 1:-3, np.newaxis] * a[:, 1:-3, :]) \
        / (2 * area_t[:, 2:-2, np.newaxis])
    return b


@veros_kernel
def solve_tridiag(a, b, c, d):
    """
    Solves a tridiagonal matrix system with diagonals a, b, c and RHS vector d.
    Uses LAPACK when running with NumPy, and otherwise the Thomas algorithm iterating over the
    last axis of the input arrays.
    """
    assert a.shape == b.shape and a.shape == c.shape and a.shape == d.shape
    from scipy.linalg import lapack
    a[..., 0] = c[..., -1] = 0  # remove couplings between slices
    return lapack.dgtsv(a.flatten()[1:], b.flatten(), c.flatten()[:-1], d.flatten())[3].reshape(a.shape)


@veros_kernel
def calc_diss_u(diss, kbot, nz, dzw, dxt, dxu):
    diss_u = np.zeros_like(diss)
    ks = np.zeros_like(kbot)
    ks[1:-2, 2:-2] = np.maximum(kbot[1:-2, 2:-2], kbot[2:-1, 2:-2]) - 1
    diss_u = diffusion.dissipation_on_wgrid(diss_u, nz, dzw, aloc=diss, ks=ks)
    return ugrid_to_tgrid(diss_u, dxt, dxu)


@veros_kernel
def calc_diss_v(diss, kbot, nz, dzw, area_v, area_t):
    diss_v = np.zeros_like(diss)
    ks = np.zeros_like(kbot)
    ks[2:-2, 1:-2] = np.maximum(kbot[2:-2, 1:-2], kbot[2:-2, 2:-1]) - 1
    diss_v = diffusion.dissipation_on_wgrid(diss_v, nz, dzw, aloc=diss, ks=ks)
    return vgrid_to_tgrid(diss_v, area_v, area_t)

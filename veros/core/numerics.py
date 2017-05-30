from .. import veros_method
from . import cyclic, density, utilities, diffusion
from scipy.linalg import lapack


@veros_method
def u_centered_grid(veros, dyt, dyu, yt, yu):
    yu[0] = 0
    yu[1:] = np.cumsum(dyt[1:])

    yt[0] = yu[0] - dyt[0] * 0.5
    yt[1:] = 2 * yu[:-1]

    alternating_pattern = np.ones_like(yt)
    alternating_pattern[::2] = -1
    yt[...] = alternating_pattern * np.cumsum(alternating_pattern * yt)

    dyu[:-1] = yt[1:] - yt[:-1]
    dyu[-1] = 2 * dyt[-1] - dyu[-2]


@veros_method
def calc_grid(veros):
    """
    setup grid based on dxt,dyt,dzt and x_origin, y_origin
    """
    aloc = np.zeros((veros.nx, veros.ny), dtype=veros.default_float_type)
    dxt_gl = np.zeros(veros.nx + 4, dtype=veros.default_float_type)
    dxu_gl = np.zeros(veros.nx + 4, dtype=veros.default_float_type)
    xt_gl = np.zeros(veros.nx + 4, dtype=veros.default_float_type)
    xu_gl = np.zeros(veros.nx + 4, dtype=veros.default_float_type)
    dyt_gl = np.zeros(veros.ny + 4, dtype=veros.default_float_type)
    dyu_gl = np.zeros(veros.ny + 4, dtype=veros.default_float_type)
    yt_gl = np.zeros(veros.ny + 4, dtype=veros.default_float_type)
    yu_gl = np.zeros(veros.ny + 4, dtype=veros.default_float_type)

    """
    transfer from locally defined variables to global ones
    """
    aloc[:, 0] = veros.dxt[2:-2]

    dxt_gl[2:-2] = aloc[:, 0]

    if veros.enable_cyclic_x:
        dxt_gl[veros.nx + 2:veros.nx + 4] = dxt_gl[2:4]
        dxt_gl[:2] = dxt_gl[veros.nx:-2]
    else:
        dxt_gl[veros.nx + 2:veros.nx + 4] = dxt_gl[veros.nx + 1]
        dxt_gl[:2] = dxt_gl[2]

    aloc[0, :] = veros.dyt[2:-2]
    dyt_gl[2:-2] = aloc[0, :]

    dyt_gl[veros.ny + 2:veros.ny + 4] = dyt_gl[veros.ny + 1]
    dyt_gl[:2] = dyt_gl[2]

    """
    grid in east/west direction
    """
    u_centered_grid(veros, dxt_gl, dxu_gl, xt_gl, xu_gl)
    xt_gl += veros.x_origin - xu_gl[2]
    xu_gl += veros.x_origin - xu_gl[2]

    if veros.enable_cyclic_x:
        xt_gl[veros.nx + 2:veros.nx + 4] = xt_gl[2:4]
        xt_gl[:2] = xt_gl[veros.nx:-2]
        xu_gl[veros.nx + 2:veros.nx + 4] = xt_gl[2:4]
        xu_gl[:2] = xu_gl[veros.nx:-2]
        dxu_gl[veros.nx + 2:veros.nx + 4] = dxu_gl[2:4]
        dxu_gl[:2] = dxu_gl[veros.nx:-2]

    """
    grid in north/south direction
    """
    u_centered_grid(veros, dyt_gl, dyu_gl, yt_gl, yu_gl)
    yt_gl += veros.y_origin - yu_gl[2]
    yu_gl += veros.y_origin - yu_gl[2]

    if veros.coord_degree:
        """
        convert from degrees to pseudo cartesian grid
        """
        dxt_gl *= veros.degtom
        dxu_gl *= veros.degtom
        dyt_gl *= veros.degtom
        dyu_gl *= veros.degtom

    """
    transfer to locally defined variables
    """
    veros.xt[:] = xt_gl[:]
    veros.xu[:] = xu_gl[:]
    veros.dxu[:] = dxu_gl[:]
    veros.dxt[:] = dxt_gl[:]

    veros.yt[:] = yt_gl[:]
    veros.yu[:] = yu_gl[:]
    veros.dyu[:] = dyu_gl[:]
    veros.dyt[:] = dyt_gl[:]

    """
    grid in vertical direction
    """
    u_centered_grid(veros, veros.dzt, veros.dzw, veros.zt, veros.zw)
    veros.zt -= veros.zw[-1]
    veros.zw -= veros.zw[-1]  # zero at zw(nz)

    """
    metric factors
    """
    if veros.coord_degree:
        veros.cost[...] = np.cos(veros.yt * veros.pi / 180.)
        veros.cosu[...] = np.cos(veros.yu * veros.pi / 180.)
        veros.tantr[...] = np.tan(veros.yt * veros.pi / 180.) / veros.radius
    else:
        veros.cost[...] = 1.0
        veros.cosu[...] = 1.0
        veros.tantr[...] = 0.0

    """
    precalculate area of boxes
    """
    veros.area_t[...] = veros.cost * veros.dyt * veros.dxt[:, np.newaxis]
    veros.area_u[...] = veros.cost * veros.dyt * veros.dxu[:, np.newaxis]
    veros.area_v[...] = veros.cosu * veros.dyu * veros.dxt[:, np.newaxis]


@veros_method
def calc_beta(veros):
    """
    calculate beta = df/dy
    """
    veros.beta[:, 2:-2] = 0.5 * ((veros.coriolis_t[:, 3:-1] - veros.coriolis_t[:, 2:-2])
                                 / veros.dyu[2:-2]
                                 + (veros.coriolis_t[:, 2:-2] - veros.coriolis_t[:, 1:-3])
                                 / veros.dyu[1:-3])


@veros_method
def calc_topo(veros):
    """
    calulate masks, total depth etc
    """

    """
    close domain
    """
    veros.kbot[:, :2] = 0
    veros.kbot[:, -2:] = 0
    if veros.enable_cyclic_x:
        cyclic.setcyclic_x(veros.kbot)
    else:
        veros.kbot[:2, :] = 0
        veros.kbot[-2:, :] = 0

    """
    Land masks
    """
    veros.maskT[...] = 0.0
    land_mask = veros.kbot > 0
    ks = np.arange(veros.maskT.shape[2])[np.newaxis, np.newaxis, :]
    veros.maskT[...] = land_mask[..., np.newaxis] & (veros.kbot[..., np.newaxis] - 1 <= ks)

    if veros.enable_cyclic_x:
        cyclic.setcyclic_x(veros.maskT)
    veros.maskU[...] = veros.maskT
    veros.maskU[:veros.nx + 3, :,
                :] = np.minimum(veros.maskT[:veros.nx + 3, :, :], veros.maskT[1:veros.nx + 4, :, :])
    if veros.enable_cyclic_x:
        cyclic.setcyclic_x(veros.maskU)
    veros.maskV[...] = veros.maskT
    veros.maskV[:, :-1] = np.minimum(veros.maskT[:, :-1], veros.maskT[:, 1:veros.ny + 4])
    if veros.enable_cyclic_x:
        cyclic.setcyclic_x(veros.maskV)
    veros.maskZ[...] = veros.maskT
    veros.maskZ[:veros.nx + 3, :-1] = np.minimum(np.minimum(veros.maskT[:veros.nx + 3, :-1],
                                                            veros.maskT[:veros.nx + 3, 1:veros.ny + 4]),
                                                 veros.maskT[1:veros.nx + 4, :-1])
    if veros.enable_cyclic_x:
        cyclic.setcyclic_x(veros.maskZ)
    veros.maskW[...] = veros.maskT
    veros.maskW[:, :, :veros.nz -
                1] = np.minimum(veros.maskT[:, :, :veros.nz - 1], veros.maskT[:, :, 1:veros.nz])

    """
    total depth
    """
    veros.ht[...] = np.sum(veros.maskT * veros.dzt[np.newaxis, np.newaxis, :], axis=2)
    veros.hu[...] = np.sum(veros.maskU * veros.dzt[np.newaxis, np.newaxis, :], axis=2)
    veros.hv[...] = np.sum(veros.maskV * veros.dzt[np.newaxis, np.newaxis, :], axis=2)

    mask = (veros.hu == 0).astype(np.float)
    veros.hur[...] = 1. / (veros.hu + mask) * (1 - mask)
    mask = (veros.hv == 0).astype(np.float)
    veros.hvr[...] = 1. / (veros.hv + mask) * (1 - mask)


@veros_method
def calc_initial_conditions(veros):
    """
    calculate dyn. enthalp, etc
    """
    if np.sum(veros.salt < 0.0):
        raise RuntimeError("encountered negative salinity")

    if veros.enable_cyclic_x:
        cyclic.setcyclic_x(veros.temp)
        cyclic.setcyclic_x(veros.salt)

    veros.rho[...] = density.get_rho(veros, veros.salt, veros.temp, np.abs(veros.zt)[
                                     :, np.newaxis]) * veros.maskT[..., np.newaxis]
    veros.Hd[...] = density.get_dyn_enthalpy(veros, veros.salt, veros.temp, np.abs(veros.zt)[
                                             :, np.newaxis]) * veros.maskT[..., np.newaxis]
    veros.int_drhodT[...] = density.get_int_drhodT(
        veros, veros.salt, veros.temp, np.abs(veros.zt)[:, np.newaxis])
    veros.int_drhodS[...] = density.get_int_drhodS(
        veros, veros.salt, veros.temp, np.abs(veros.zt)[:, np.newaxis])

    fxa = -veros.grav / veros.rho_0 / veros.dzw[np.newaxis, np.newaxis, :] * veros.maskW
    veros.Nsqr[:, :, :-1, :] = fxa[:, :, :-1, np.newaxis] * \
        (density.get_rho(veros, veros.salt[:, :, 1:, :], veros.temp[:, :, 1:, :], np.abs(veros.zt)[:-1, np.newaxis])
         - veros.rho[:, :, :-1, :])
    veros.Nsqr[:, :, -1, :] = veros.Nsqr[:, :, -2, :]


@veros_method
def ugrid_to_tgrid(veros, a):
    b = np.zeros_like(a)
    b[2:-2, :, :] = (veros.dxu[2:-2, np.newaxis, np.newaxis] * a[2:-2, :, :] + veros.dxu[1:-3, np.newaxis, np.newaxis] * a[1:-3, :, :]) \
        / (2 * veros.dxt[2:-2, np.newaxis, np.newaxis])
    return b


@veros_method
def vgrid_to_tgrid(veros, a):
    b = np.zeros_like(a)
    b[:, 2:-2, :] = (veros.area_v[:, 2:-2, np.newaxis] * a[:, 2:-2, :] + veros.area_v[:, 1:-3, np.newaxis] * a[:, 1:-3, :]) \
        / (2 * veros.area_t[:, 2:-2, np.newaxis])
    return b


@veros_method
def solve_tridiag(veros, a, b, c, d):
    """
    Solves a tridiagonal matrix system with diagonals a, b, c and RHS vector d.
    Uses LAPACK when running with NumPy, and otherwise the Thomas algorithm iterating over the
    last axis of the input arrays.
    """
    assert a.shape == b.shape and a.shape == c.shape and a.shape == d.shape
    try:
        return np.linalg.solve_tridiagonal(a, b, c, d)
    except AttributeError:
        return lapack.dgtsv(a.flatten()[1:], b.flatten(), c.flatten()[:-1], d.flatten())[3].reshape(a.shape)


@veros_method
def calc_diss(veros, diss, K_diss, tag):
    diss_u = np.zeros_like(diss)
    ks = np.zeros_like(veros.kbot)
    if tag == 'U':
        ks[1:-2, 2:-2] = np.maximum(veros.kbot[1:-2, 2:-2], veros.kbot[2:-1, 2:-2]) - 1
        interpolator = ugrid_to_tgrid
    elif tag == 'V':
        ks[2:-2, 1:-2] = np.maximum(veros.kbot[2:-2, 1:-2], veros.kbot[2:-2, 2:-1]) - 1
        interpolator = vgrid_to_tgrid
    else:
        raise ValueError("unknown tag {} (must be 'U' or 'V')".format(tag))
    diffusion.dissipation_on_wgrid(veros, diss_u, aloc=diss, ks=ks)
    return K_diss + interpolator(veros, diss_u)

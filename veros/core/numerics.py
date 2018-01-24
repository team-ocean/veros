from scipy.linalg import lapack

from .. import veros_method, veros_inline_method
from . import cyclic, density, diffusion

try:
    from .special import tdma_opencl
except ImportError:
    import warnings
    warnings.warn("Special OpenCL implementations could not be imported")


@veros_method
def u_centered_grid(vs, dyt, dyu, yt, yu):
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
def calc_grid(vs):
    """
    setup grid based on dxt,dyt,dzt and x_origin, y_origin
    """
    aloc = np.zeros((vs.nx, vs.ny), dtype=vs.default_float_type)
    dxt_gl = np.zeros(vs.nx + 4, dtype=vs.default_float_type)
    dxu_gl = np.zeros(vs.nx + 4, dtype=vs.default_float_type)
    xt_gl = np.zeros(vs.nx + 4, dtype=vs.default_float_type)
    xu_gl = np.zeros(vs.nx + 4, dtype=vs.default_float_type)
    dyt_gl = np.zeros(vs.ny + 4, dtype=vs.default_float_type)
    dyu_gl = np.zeros(vs.ny + 4, dtype=vs.default_float_type)
    yt_gl = np.zeros(vs.ny + 4, dtype=vs.default_float_type)
    yu_gl = np.zeros(vs.ny + 4, dtype=vs.default_float_type)

    """
    transfer from locally defined variables to global ones
    """
    aloc[:, 0] = vs.dxt[2:-2]

    dxt_gl[2:-2] = aloc[:, 0]

    if vs.enable_cyclic_x:
        dxt_gl[vs.nx + 2:vs.nx + 4] = dxt_gl[2:4]
        dxt_gl[:2] = dxt_gl[vs.nx:-2]
    else:
        dxt_gl[vs.nx + 2:vs.nx + 4] = dxt_gl[vs.nx + 1]
        dxt_gl[:2] = dxt_gl[2]

    aloc[0, :] = vs.dyt[2:-2]
    dyt_gl[2:-2] = aloc[0, :]

    dyt_gl[vs.ny + 2:vs.ny + 4] = dyt_gl[vs.ny + 1]
    dyt_gl[:2] = dyt_gl[2]

    """
    grid in east/west direction
    """
    u_centered_grid(vs, dxt_gl, dxu_gl, xt_gl, xu_gl)
    xt_gl += vs.x_origin - xu_gl[2]
    xu_gl += vs.x_origin - xu_gl[2]

    if vs.enable_cyclic_x:
        xt_gl[vs.nx + 2:vs.nx + 4] = xt_gl[2:4]
        xt_gl[:2] = xt_gl[vs.nx:-2]
        xu_gl[vs.nx + 2:vs.nx + 4] = xt_gl[2:4]
        xu_gl[:2] = xu_gl[vs.nx:-2]
        dxu_gl[vs.nx + 2:vs.nx + 4] = dxu_gl[2:4]
        dxu_gl[:2] = dxu_gl[vs.nx:-2]

    """
    grid in north/south direction
    """
    u_centered_grid(vs, dyt_gl, dyu_gl, yt_gl, yu_gl)
    yt_gl += vs.y_origin - yu_gl[2]
    yu_gl += vs.y_origin - yu_gl[2]

    if vs.coord_degree:
        """
        convert from degrees to pseudo cartesian grid
        """
        dxt_gl *= vs.degtom
        dxu_gl *= vs.degtom
        dyt_gl *= vs.degtom
        dyu_gl *= vs.degtom

    """
    transfer to locally defined variables
    """
    vs.xt[:] = xt_gl[:]
    vs.xu[:] = xu_gl[:]
    vs.dxu[:] = dxu_gl[:]
    vs.dxt[:] = dxt_gl[:]

    vs.yt[:] = yt_gl[:]
    vs.yu[:] = yu_gl[:]
    vs.dyu[:] = dyu_gl[:]
    vs.dyt[:] = dyt_gl[:]

    """
    grid in vertical direction
    """
    u_centered_grid(vs, vs.dzt, vs.dzw, vs.zt, vs.zw)
    vs.zt -= vs.zw[-1]
    vs.zw -= vs.zw[-1]  # zero at zw(nz)

    """
    metric factors
    """
    if vs.coord_degree:
        vs.cost[...] = np.cos(vs.yt * vs.pi / 180.)
        vs.cosu[...] = np.cos(vs.yu * vs.pi / 180.)
        vs.tantr[...] = np.tan(vs.yt * vs.pi / 180.) / vs.radius
    else:
        vs.cost[...] = 1.0
        vs.cosu[...] = 1.0
        vs.tantr[...] = 0.0

    """
    precalculate area of boxes
    """
    vs.area_t[...] = vs.cost * vs.dyt * vs.dxt[:, np.newaxis]
    vs.area_u[...] = vs.cost * vs.dyt * vs.dxu[:, np.newaxis]
    vs.area_v[...] = vs.cosu * vs.dyu * vs.dxt[:, np.newaxis]


@veros_method
def calc_beta(vs):
    """
    calculate beta = df/dy
    """
    vs.beta[:, 2:-2] = 0.5 * ((vs.coriolis_t[:, 3:-1] - vs.coriolis_t[:, 2:-2]) / vs.dyu[2:-2] \
                            + (vs.coriolis_t[:, 2:-2] - vs.coriolis_t[:, 1:-3]) / vs.dyu[1:-3])


@veros_method
def calc_topo(vs):
    """
    calulate masks, total depth etc
    """

    """
    close domain
    """
    vs.kbot[:, :2] = 0
    vs.kbot[:, -2:] = 0
    if vs.enable_cyclic_x:
        cyclic.setcyclic_x(vs.kbot)
    else:
        vs.kbot[:2, :] = 0
        vs.kbot[-2:, :] = 0

    """
    Land masks
    """
    vs.maskT[...] = 0.0
    land_mask = vs.kbot > 0
    ks = np.arange(vs.maskT.shape[2])[np.newaxis, np.newaxis, :]
    vs.maskT[...] = land_mask[..., np.newaxis] & (vs.kbot[..., np.newaxis] - 1 <= ks)

    if vs.enable_cyclic_x:
        cyclic.setcyclic_x(vs.maskT)
    vs.maskU[...] = vs.maskT
    vs.maskU[:vs.nx + 3, :, :] = np.minimum(vs.maskT[:vs.nx + 3, :, :], vs.maskT[1:vs.nx + 4, :, :])
    if vs.enable_cyclic_x:
        cyclic.setcyclic_x(vs.maskU)
    vs.maskV[...] = vs.maskT
    vs.maskV[:, :-1] = np.minimum(vs.maskT[:, :-1], vs.maskT[:, 1:vs.ny + 4])
    if vs.enable_cyclic_x:
        cyclic.setcyclic_x(vs.maskV)
    vs.maskZ[...] = vs.maskT
    vs.maskZ[:vs.nx + 3, :-1] = np.minimum(np.minimum(vs.maskT[:vs.nx + 3, :-1],
                                                      vs.maskT[:vs.nx + 3, 1:vs.ny + 4]),
                                                 vs.maskT[1:vs.nx + 4, :-1])
    if vs.enable_cyclic_x:
        cyclic.setcyclic_x(vs.maskZ)
    vs.maskW[...] = vs.maskT
    vs.maskW[:, :, :vs.nz - 1] = np.minimum(vs.maskT[:, :, :vs.nz - 1], vs.maskT[:, :, 1:vs.nz])

    """
    total depth
    """
    vs.ht[...] = np.sum(vs.maskT * vs.dzt[np.newaxis, np.newaxis, :], axis=2)
    vs.hu[...] = np.sum(vs.maskU * vs.dzt[np.newaxis, np.newaxis, :], axis=2)
    vs.hv[...] = np.sum(vs.maskV * vs.dzt[np.newaxis, np.newaxis, :], axis=2)

    mask = (vs.hu == 0).astype(np.float)
    vs.hur[...] = 1. / (vs.hu + mask) * (1 - mask)
    mask = (vs.hv == 0).astype(np.float)
    vs.hvr[...] = 1. / (vs.hv + mask) * (1 - mask)


@veros_method
def calc_initial_conditions(vs):
    """
    calculate dyn. enthalp, etc
    """
    if np.sum(vs.salt < 0.0):
        raise RuntimeError("encountered negative salinity")

    if vs.enable_cyclic_x:
        cyclic.setcyclic_x(vs.temp)
        cyclic.setcyclic_x(vs.salt)

    vs.rho[...] = density.get_rho(vs, vs.salt, vs.temp, np.abs(vs.zt)[:, np.newaxis]) \
                  * vs.maskT[..., np.newaxis]
    vs.Hd[...] = density.get_dyn_enthalpy(vs, vs.salt, vs.temp, np.abs(vs.zt)[:, np.newaxis]) \
                 * vs.maskT[..., np.newaxis]
    vs.int_drhodT[...] = density.get_int_drhodT(vs, vs.salt, vs.temp, np.abs(vs.zt)[:, np.newaxis])
    vs.int_drhodS[...] = density.get_int_drhodS(vs, vs.salt, vs.temp, np.abs(vs.zt)[:, np.newaxis])

    fxa = -vs.grav / vs.rho_0 / vs.dzw[np.newaxis, np.newaxis, :] * vs.maskW
    vs.Nsqr[:, :, :-1, :] = fxa[:, :, :-1, np.newaxis] \
        * (density.get_rho(vs, vs.salt[:, :, 1:, :], vs.temp[:, :, 1:, :], np.abs(vs.zt)[:-1, np.newaxis]) \
         - vs.rho[:, :, :-1, :])
    vs.Nsqr[:, :, -1, :] = vs.Nsqr[:, :, -2, :]


@veros_inline_method
def ugrid_to_tgrid(vs, a):
    b = np.zeros_like(a)
    b[2:-2, :, :] = (vs.dxu[2:-2, np.newaxis, np.newaxis] * a[2:-2, :, :] + vs.dxu[1:-3, np.newaxis, np.newaxis] * a[1:-3, :, :]) \
        / (2 * vs.dxt[2:-2, np.newaxis, np.newaxis])
    return b


@veros_inline_method
def vgrid_to_tgrid(vs, a):
    b = np.zeros_like(a)
    b[:, 2:-2, :] = (vs.area_v[:, 2:-2, np.newaxis] * a[:, 2:-2, :] + vs.area_v[:, 1:-3, np.newaxis] * a[:, 1:-3, :]) \
        / (2 * vs.area_t[:, 2:-2, np.newaxis])
    return b


@veros_method
def solve_tridiag(vs, a, b, c, d):
    """
    Solves a tridiagonal matrix system with diagonals a, b, c and RHS vector d.
    Uses LAPACK when running with NumPy, and otherwise the Thomas algorithm iterating over the
    last axis of the input arrays.
    """
    assert a.shape == b.shape and a.shape == c.shape and a.shape == d.shape

    if vs.backend_name == "numpy":
        a[..., 0] = c[..., -1] = 0 # remove couplings between slices
        return lapack.dgtsv(a.flatten()[1:], b.flatten(), c.flatten()[:-1], d.flatten())[3].reshape(a.shape)

    if vs.vector_engine == "opencl":
        return tdma_opencl.tdma(a, b, c, d)

    return np.linalg.solve_tridiagonal(a, b, c, d)


@veros_inline_method
def calc_diss(vs, diss, tag):
    diss_u = np.zeros_like(diss)
    ks = np.zeros_like(vs.kbot)
    if tag == "U":
        ks[1:-2, 2:-2] = np.maximum(vs.kbot[1:-2, 2:-2], vs.kbot[2:-1, 2:-2]) - 1
        interpolator = ugrid_to_tgrid
    elif tag == "V":
        ks[2:-2, 1:-2] = np.maximum(vs.kbot[2:-2, 1:-2], vs.kbot[2:-2, 2:-1]) - 1
        interpolator = vgrid_to_tgrid
    else:
        raise ValueError("unknown tag {} (must be \"U\" or \"V\")".format(tag))
    diffusion.dissipation_on_wgrid(vs, diss_u, aloc=diss, ks=ks)
    return interpolator(vs, diss_u)

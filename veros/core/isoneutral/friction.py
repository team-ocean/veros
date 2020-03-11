from veros.core.operators import numpy as np

from veros import veros_kernel
from veros.core import numerics, utilities
from veros.core.operators import update, update_add, at


@veros_kernel(static_args=('enable_implicit_vert_friction', 'enable_conserve_energy'))
def isoneutral_friction(du_mix, dv_mix, K_diss_gm, u, v, tau, taup1, kbot, kappa_gm,
                        dt_mom, dxt, dxu, dzt, dzw, maskU, maskV, area_v, area_t,
                        enable_implicit_vert_friction, enable_conserve_energy):
    """
    vertical friction using TEM formalism for eddy driven velocity
    """
    flux_top = np.zeros_like(maskU)

    if enable_implicit_vert_friction:
        aloc = u[:, :, :, taup1]
    else:
        aloc = u[:, :, :, tau]

    # implicit vertical friction of zonal momentum by GM
    ks = np.maximum(kbot[1:-2, 1:-2], kbot[2:-1, 1:-2]) - 1
    fxa = 0.5 * (kappa_gm[1:-2, 1:-2, :] + kappa_gm[2:-1, 1:-2, :])
    delta, a_tri, b_tri, c_tri = (
        np.zeros_like(maskU[1:-2, 1:-2, :])
        for _ in range(4)
    )
    delta = update(delta, at[:, :, :-1], dt_mom / dzw[np.newaxis, np.newaxis, :-1] * \
        fxa[:, :, :-1] * maskU[1:-2, 1:-2, 1:] * maskU[1:-2, 1:-2, :-1])
    delta = update(delta, at[-1], 0.)
    a_tri = update(a_tri, at[:, :, 1:], -delta[:, :, :-1] / dzt[np.newaxis, np.newaxis, 1:])
    b_tri_edge = 1 + delta / dzt[np.newaxis, np.newaxis, :]
    b_tri = update(b_tri, at[:, :, 1:-1], 1 + delta[:, :, 1:-1] / dzt[np.newaxis, np.newaxis, 1:-1] + \
        delta[:, :, :-2] / dzt[np.newaxis, np.newaxis, 1:-1])
    b_tri = update(b_tri, at[:, :, -1], 1 + delta[:, :, -2] / dzt[-1])
    c_tri = update(c_tri, at[...], - delta / dzt[np.newaxis, np.newaxis, :])
    sol, water_mask = utilities.solve_implicit(
        ks, a_tri, b_tri, c_tri, aloc[1:-2, 1:-2, :], b_edge=b_tri_edge
    )
    u = update(u, at[1:-2, 1:-2, :, taup1], utilities.where(water_mask, sol, u[1:-2, 1:-2, :, taup1]))
    du_mix = update_add(du_mix, at[1:-2, 1:-2, :], (u[1:-2, 1:-2, :, taup1] - aloc[1:-2, 1:-2, :]) / dt_mom * water_mask)

    if enable_conserve_energy:
        # diagnose dissipation
        diss = np.zeros_like(maskU)
        fxa = 0.5 * (kappa_gm[1:-2, 1:-2, :-1] + kappa_gm[2:-1, 1:-2, :-1])
        flux_top = update(flux_top, at[1:-2, 1:-2, :-1], fxa * (u[1:-2, 1:-2, 1:, taup1] - u[1:-2, 1:-2, :-1, taup1]) \
            / dzw[np.newaxis, np.newaxis, :-1] * maskU[1:-2, 1:-2, 1:] * maskU[1:-2, 1:-2, :-1])
        diss = update(diss, at[1:-2, 1:-2, :-1], (u[1:-2, 1:-2, 1:, tau] - u[1:-2, 1:-2, :-1, tau]) \
            * flux_top[1:-2, 1:-2, :-1] / dzw[np.newaxis, np.newaxis, :-1])
        diss = update(diss, at[:, :, -1], 0.0)
        diss = numerics.ugrid_to_tgrid(diss, dxt, dxu)
        K_diss_gm = update(K_diss_gm, at[...], diss)

    if enable_implicit_vert_friction:
        aloc = v[:, :, :, taup1]
    else:
        aloc = v[:, :, :, tau]

    # implicit vertical friction of zonal momentum by GM
    ks = np.maximum(kbot[1:-2, 1:-2], kbot[1:-2, 2:-1]) - 1
    fxa = 0.5 * (kappa_gm[1:-2, 1:-2, :] + kappa_gm[1:-2, 2:-1, :])
    delta, a_tri, b_tri, c_tri = (np.zeros_like(maskV[1:-2, 1:-2, :]) for _ in range(4))
    delta = update(delta, at[:, :, :-1], dt_mom / dzw[np.newaxis, np.newaxis, :-1] * \
        fxa[:, :, :-1] * maskV[1:-2, 1:-2, 1:] * maskV[1:-2, 1:-2, :-1])
    delta = update(delta, at[-1], 0.)
    a_tri = update(a_tri, at[:, :, 1:], -delta[:, :, :-1] / dzt[np.newaxis, np.newaxis, 1:])
    b_tri_edge = 1 + delta / dzt[np.newaxis, np.newaxis, :]
    b_tri = update(b_tri, at[:, :, 1:-1], 1 + delta[:, :, 1:-1] / dzt[np.newaxis, np.newaxis, 1:-1] + \
        delta[:, :, :-2] / dzt[np.newaxis, np.newaxis, 1:-1])
    b_tri = update(b_tri, at[:, :, -1], 1 + delta[:, :, -2] / dzt[-1])
    c_tri = update(c_tri, at[...], - delta / dzt[np.newaxis, np.newaxis, :])
    sol, water_mask = utilities.solve_implicit(
        ks, a_tri, b_tri, c_tri, aloc[1:-2, 1:-2, :], b_edge=b_tri_edge
    )
    v = update(v, at[1:-2, 1:-2, :, taup1], utilities.where(water_mask, sol, v[1:-2, 1:-2, :, taup1]))
    dv_mix = update_add(dv_mix, at[1:-2, 1:-2, :], (v[1:-2, 1:-2, :, taup1] - aloc[1:-2, 1:-2, :]) / dt_mom * water_mask)

    if enable_conserve_energy:
        # diagnose dissipation
        diss = np.zeros_like(maskV)
        fxa = 0.5 * (kappa_gm[1:-2, 1:-2, :-1] + kappa_gm[1:-2, 2:-1, :-1])
        flux_top = update(flux_top, at[1:-2, 1:-2, :-1], fxa * (v[1:-2, 1:-2, 1:, taup1] - v[1:-2, 1:-2, :-1, taup1]) \
            / dzw[np.newaxis, np.newaxis, :-1] * maskV[1:-2, 1:-2, 1:] * maskV[1:-2, 1:-2, :-1])
        diss = update(diss, at[1:-2, 1:-2, :-1], (v[1:-2, 1:-2, 1:, tau] - v[1:-2, 1:-2, :-1, tau]) \
            * flux_top[1:-2, 1:-2, :-1] / dzw[np.newaxis, np.newaxis, :-1])
        diss = update(diss, at[:, :, -1], 0.0)
        diss = numerics.vgrid_to_tgrid(diss, area_v, area_t)
        K_diss_gm += diss

    return u, v, du_mix, dv_mix, K_diss_gm

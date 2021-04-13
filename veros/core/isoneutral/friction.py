from veros.core.operators import numpy as np

from veros import veros_kernel, KernelOutput
from veros.variables import allocate
from veros.core import numerics, utilities
from veros.core.operators import update, update_add, at


@veros_kernel
def isoneutral_friction(state):
    """
    vertical friction using TEM formalism for eddy driven velocity
    """
    vs = state.variables
    settings = state.settings

    flux_top = allocate(state.dimensions, ("xt", "yt", "zt"))

    u = vs.u

    if settings.enable_implicit_vert_friction:
        aloc = u[:, :, :, vs.taup1]
    else:
        aloc = u[:, :, :, vs.tau]

    # implicit vertical friction of zonal momentum by GM
    ks = np.maximum(vs.kbot[1:-2, 1:-2], vs.kbot[2:-1, 1:-2])
    _, water_mask, edge_mask = utilities.create_water_masks(ks, settings.nz)

    fxa = 0.5 * (vs.kappa_gm[1:-2, 1:-2, :] + vs.kappa_gm[2:-1, 1:-2, :])

    delta, a_tri, b_tri, c_tri = (
        allocate(state.dimensions, ("xt", "yt", "zt"))[1:-2, 1:-2, :]
        for _ in range(4)
    )

    delta = update(delta, at[:, :, :-1], settings.dt_mom / vs.dzw[np.newaxis, np.newaxis, :-1] * \
        fxa[:, :, :-1] * vs.maskU[1:-2, 1:-2, 1:] * vs.maskU[1:-2, 1:-2, :-1])
    delta = update(delta, at[-1], 0.)
    a_tri = update(a_tri, at[:, :, 1:], -delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, 1:])
    b_tri_edge = 1 + delta / vs.dzt[np.newaxis, np.newaxis, :]
    b_tri = update(b_tri, at[:, :, 1:-1], 1 + delta[:, :, 1:-1] / vs.dzt[np.newaxis, np.newaxis, 1:-1] + \
        delta[:, :, :-2] / vs.dzt[np.newaxis, np.newaxis, 1:-1])
    b_tri = update(b_tri, at[:, :, -1], 1 + delta[:, :, -2] / vs.dzt[-1])
    c_tri = update(c_tri, at[...], - delta / vs.dzt[np.newaxis, np.newaxis, :])

    sol = utilities.solve_implicit(
        a_tri, b_tri, c_tri, aloc[1:-2, 1:-2, :], water_mask, b_edge=b_tri_edge, edge_mask=edge_mask
    )
    u = update(u, at[1:-2, 1:-2, :, vs.taup1], np.where(water_mask, sol, u[1:-2, 1:-2, :, vs.taup1]))
    du_mix = vs.du_mix
    du_mix = update_add(du_mix, at[1:-2, 1:-2, :], (u[1:-2, 1:-2, :, vs.taup1] - aloc[1:-2, 1:-2, :]) / settings.dt_mom * water_mask)

    if settings.enable_conserve_energy:
        # diagnose dissipation
        diss = np.zeros_like(vs.maskU)
        fxa = 0.5 * (vs.kappa_gm[1:-2, 1:-2, :-1] + vs.kappa_gm[2:-1, 1:-2, :-1])
        flux_top = update(flux_top, at[1:-2, 1:-2, :-1], fxa * (u[1:-2, 1:-2, 1:, vs.taup1] - u[1:-2, 1:-2, :-1, vs.taup1]) \
            / vs.dzw[np.newaxis, np.newaxis, :-1] * vs.maskU[1:-2, 1:-2, 1:] * vs.maskU[1:-2, 1:-2, :-1])
        diss = update(diss, at[1:-2, 1:-2, :-1], (u[1:-2, 1:-2, 1:, vs.tau] - u[1:-2, 1:-2, :-1, vs.tau]) \
            * flux_top[1:-2, 1:-2, :-1] / vs.dzw[np.newaxis, np.newaxis, :-1])
        diss = update(diss, at[:, :, -1], 0.0)
        diss = numerics.ugrid_to_tgrid(state, diss)
        K_diss_gm = diss

    v = vs.v

    if settings.enable_implicit_vert_friction:
        aloc = v[:, :, :, vs.taup1]
    else:
        aloc = v[:, :, :, vs.tau]

    # implicit vertical friction of zonal momentum by GM
    ks = np.maximum(vs.kbot[1:-2, 1:-2], vs.kbot[1:-2, 2:-1])
    _, water_mask, edge_mask = utilities.create_water_masks(ks, settings.nz)

    fxa = 0.5 * (vs.kappa_gm[1:-2, 1:-2, :] + vs.kappa_gm[1:-2, 2:-1, :])
    delta, a_tri, b_tri, c_tri = (np.zeros_like(vs.maskV[1:-2, 1:-2, :]) for _ in range(4))
    delta = update(delta, at[:, :, :-1], settings.dt_mom / vs.dzw[np.newaxis, np.newaxis, :-1] * \
        fxa[:, :, :-1] * vs.maskV[1:-2, 1:-2, 1:] * vs.maskV[1:-2, 1:-2, :-1])
    delta = update(delta, at[-1], 0.)
    a_tri = update(a_tri, at[:, :, 1:], -delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, 1:])
    b_tri_edge = 1 + delta / vs.dzt[np.newaxis, np.newaxis, :]
    b_tri = update(b_tri, at[:, :, 1:-1], 1 + delta[:, :, 1:-1] / vs.dzt[np.newaxis, np.newaxis, 1:-1] + \
        delta[:, :, :-2] / vs.dzt[np.newaxis, np.newaxis, 1:-1])
    b_tri = update(b_tri, at[:, :, -1], 1 + delta[:, :, -2] / vs.dzt[-1])
    c_tri = update(c_tri, at[...], - delta / vs.dzt[np.newaxis, np.newaxis, :])

    sol = utilities.solve_implicit(
        a_tri, b_tri, c_tri, aloc[1:-2, 1:-2, :], water_mask, b_edge=b_tri_edge, edge_mask=edge_mask
    )
    v = update(v, at[1:-2, 1:-2, :, vs.taup1], np.where(water_mask, sol, v[1:-2, 1:-2, :, vs.taup1]))
    dv_mix = vs.dv_mix
    dv_mix = update_add(dv_mix, at[1:-2, 1:-2, :], (v[1:-2, 1:-2, :, vs.taup1] - aloc[1:-2, 1:-2, :]) / settings.dt_mom * water_mask)

    if settings.enable_conserve_energy:
        # diagnose dissipation
        diss = np.zeros_like(vs.maskV)
        fxa = 0.5 * (vs.kappa_gm[1:-2, 1:-2, :-1] + vs.kappa_gm[1:-2, 2:-1, :-1])
        flux_top = update(flux_top, at[1:-2, 1:-2, :-1], fxa * (v[1:-2, 1:-2, 1:, vs.taup1] - v[1:-2, 1:-2, :-1, vs.taup1]) \
            / vs.dzw[np.newaxis, np.newaxis, :-1] * vs.maskV[1:-2, 1:-2, 1:] * vs.maskV[1:-2, 1:-2, :-1])
        diss = update(diss, at[1:-2, 1:-2, :-1], (v[1:-2, 1:-2, 1:, vs.tau] - v[1:-2, 1:-2, :-1, vs.tau]) \
            * flux_top[1:-2, 1:-2, :-1] / vs.dzw[np.newaxis, np.newaxis, :-1])
        diss = update(diss, at[:, :, -1], 0.0)
        diss = numerics.vgrid_to_tgrid(state, diss)
        K_diss_gm = K_diss_gm + diss

    return KernelOutput(u=u, v=v, du_mix=du_mix, dv_mix=dv_mix, K_diss_gm=K_diss_gm)

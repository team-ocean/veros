from ... import veros_method
from .. import numerics, utilities


@veros_method
def isoneutral_friction(veros):
    """
    vertical friction using TEM formalism for eddy driven velocity
    """
    diss = np.zeros((veros.nx + 4, veros.ny + 4, veros.nz))
    aloc = np.zeros((veros.nx + 4, veros.ny + 4, veros.nz))

    if veros.enable_implicit_vert_friction:
        aloc[...] = veros.u[:, :, :, veros.taup1]
    else:
        aloc[...] = veros.u[:, :, :, veros.tau]

    # implicit vertical friction of zonal momentum by GM
    ks = np.maximum(veros.kbot[1:-2, 1:-2], veros.kbot[2:-1, 1:-2]) - 1
    fxa = 0.5 * (veros.kappa_gm[1:-2, 1:-2, :] + veros.kappa_gm[2:-1, 1:-2, :])
    delta, a_tri, b_tri, c_tri = (
        np.zeros((veros.nx + 1, veros.ny + 1, veros.nz)) for _ in range(4))
    delta[:, :, :-1] = veros.dt_mom / veros.dzw[np.newaxis, np.newaxis, :-1] * \
        fxa[:, :, :-1] * veros.maskU[1:-2, 1:-2, 1:] * veros.maskU[1:-2, 1:-2, :-1]
    delta[-1] = 0.
    a_tri[:, :, 1:] = -delta[:, :, :-1] / veros.dzt[np.newaxis, np.newaxis, 1:]
    b_tri_edge = 1 + delta / veros.dzt[np.newaxis, np.newaxis, :]
    b_tri[:, :, 1:-1] = 1 + delta[:, :, 1:-1] / veros.dzt[np.newaxis, np.newaxis, 1:-1] + \
        delta[:, :, :-2] / veros.dzt[np.newaxis, np.newaxis, 1:-1]
    b_tri[:, :, -1] = 1 + delta[:, :, -2] / veros.dzt[-1]
    c_tri[...] = - delta / veros.dzt[np.newaxis, np.newaxis, :]
    sol, water_mask = utilities.solve_implicit(
        veros, ks, a_tri, b_tri, c_tri, aloc[1:-2, 1:-2, :], b_edge=b_tri_edge)
    veros.u[1:-2, 1:-2, :,
            veros.taup1] = np.where(water_mask, sol, veros.u[1:-2, 1:-2, :, veros.taup1])
    veros.du_mix[1:-2, 1:-2, :] += (veros.u[1:-2, 1:-2, :, veros.taup1] -
                                    aloc[1:-2, 1:-2, :]) / veros.dt_mom * water_mask

    if veros.enable_conserve_energy:
        # diagnose dissipation
        fxa = 0.5 * (veros.kappa_gm[1:-2, 1:-2, :-1] + veros.kappa_gm[2:-1, 1:-2, :-1])
        veros.flux_top[1:-2, 1:-2, :-1] = fxa * (veros.u[1:-2, 1:-2, 1:, veros.taup1] - veros.u[1:-2, 1:-2, :-1, veros.taup1]) \
            / veros.dzw[np.newaxis, np.newaxis, :-1] * veros.maskU[1:-2, 1:-2, 1:] * veros.maskU[1:-2, 1:-2, :-1]
        diss[1:-2, 1:-2, :-1] = (veros.u[1:-2, 1:-2, 1:, veros.tau] - veros.u[1:-2, 1:-2, :-1, veros.tau]) \
            * veros.flux_top[1:-2, 1:-2, :-1] / veros.dzw[np.newaxis, np.newaxis, :-1]
        diss[:, :, -1] = 0.0
        diss = numerics.ugrid_to_tgrid(veros, diss)
        veros.K_diss_gm[...] = diss

    if veros.enable_implicit_vert_friction:
        aloc[...] = veros.v[:, :, :, veros.taup1]
    else:
        aloc[...] = veros.v[:, :, :, veros.tau]

    # implicit vertical friction of zonal momentum by GM
    ks = np.maximum(veros.kbot[1:-2, 1:-2], veros.kbot[1:-2, 2:-1]) - 1
    fxa = 0.5 * (veros.kappa_gm[1:-2, 1:-2, :] + veros.kappa_gm[1:-2, 2:-1, :])
    delta, a_tri, b_tri, c_tri = (
        np.zeros((veros.nx + 1, veros.ny + 1, veros.nz)) for _ in range(4))
    delta[:, :, :-1] = veros.dt_mom / veros.dzw[np.newaxis, np.newaxis, :-1] * \
        fxa[:, :, :-1] * veros.maskV[1:-2, 1:-2, 1:] * veros.maskV[1:-2, 1:-2, :-1]
    delta[-1] = 0.
    a_tri[:, :, 1:] = -delta[:, :, :-1] / veros.dzt[np.newaxis, np.newaxis, 1:]
    b_tri_edge = 1 + delta / veros.dzt[np.newaxis, np.newaxis, :]
    b_tri[:, :, 1:-1] = 1 + delta[:, :, 1:-1] / veros.dzt[np.newaxis, np.newaxis, 1:-1] + \
        delta[:, :, :-2] / veros.dzt[np.newaxis, np.newaxis, 1:-1]
    b_tri[:, :, -1] = 1 + delta[:, :, -2] / veros.dzt[-1]
    c_tri[...] = - delta / veros.dzt[np.newaxis, np.newaxis, :]
    sol, water_mask = utilities.solve_implicit(
        veros, ks, a_tri, b_tri, c_tri, aloc[1:-2, 1:-2, :], b_edge=b_tri_edge)
    veros.v[1:-2, 1:-2, :,
            veros.taup1] = np.where(water_mask, sol, veros.v[1:-2, 1:-2, :, veros.taup1])
    veros.dv_mix[1:-2, 1:-2, :] += (veros.v[1:-2, 1:-2, :, veros.taup1] -
                                    aloc[1:-2, 1:-2, :]) / veros.dt_mom * water_mask

    if veros.enable_conserve_energy:
        # diagnose dissipation
        fxa = 0.5 * (veros.kappa_gm[1:-2, 1:-2, :-1] + veros.kappa_gm[1:-2, 2:-1, :-1])
        veros.flux_top[1:-2, 1:-2, :-1] = fxa * (veros.v[1:-2, 1:-2, 1:, veros.taup1] - veros.v[1:-2, 1:-2, :-1, veros.taup1]) \
            / veros.dzw[np.newaxis, np.newaxis, :-1] * veros.maskV[1:-2, 1:-2, 1:] * veros.maskV[1:-2, 1:-2, :-1]
        diss[1:-2, 1:-2, :-1] = (veros.v[1:-2, 1:-2, 1:, veros.tau] - veros.v[1:-2, 1:-2, :-1, veros.tau]) \
            * veros.flux_top[1:-2, 1:-2, :-1] / veros.dzw[np.newaxis, np.newaxis, :-1]
        diss[:, :, -1] = 0.0
        diss = numerics.vgrid_to_tgrid(veros, diss)
        veros.K_diss_gm += diss

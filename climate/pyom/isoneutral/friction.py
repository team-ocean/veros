import numpy as np

from climate.pyom import numerics, utilities

def isoneutral_friction(pyom):
    """
    vertical friction using TEM formalism for eddy driven velocity
    """
    diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    aloc = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    if pyom.enable_implicit_vert_friction:
        aloc[...] = pyom.u[:,:,:,pyom.taup1]
    else:
        aloc[...] = pyom.u[:,:,:,pyom.tau]

    # implicit vertical friction of zonal momentum by GM
    ks = np.maximum(pyom.kbot[1:-2, 1:-2], pyom.kbot[2:-1, 1:-2]) - 1
    fxa = 0.5 * (pyom.kappa_gm[1:-2, 1:-2, :] + pyom.kappa_gm[2:-1, 1:-2, :])
    delta, a_tri, b_tri, c_tri = (np.zeros((pyom.nx+1,pyom.ny+1,pyom.nz)) for _ in range(4))
    delta[:, :, :-1] = pyom.dt_mom / pyom.dzw[None, None, :-1] * fxa[:, :, :-1] * pyom.maskU[1:-2, 1:-2, 1:] * pyom.maskU[1:-2, 1:-2, :-1]
    delta[-1] = 0.
    a_tri[:, :, 1:] = -delta[:, :, :-1] / pyom.dzt[None, None, 1:]
    b_tri_edge = 1 + delta / pyom.dzt[None, None, :]
    b_tri[:, :, 1:-1] = 1 + delta[:, :, 1:-1] / pyom.dzt[None, None, 1:-1] + delta[:, :, :-2] / pyom.dzt[None, None, 1:-1]
    b_tri[:, :, -1] = 1 + delta[:, :, -2] / pyom.dzt[-1]
    c_tri[...] = - delta / pyom.dzt[None, None, :]
    sol, water_mask = utilities.solve_implicit(ks, a_tri, b_tri, c_tri, aloc[1:-2, 1:-2, :], b_edge=b_tri_edge)
    pyom.u[1:-2, 1:-2, :, pyom.taup1] = np.where(water_mask, sol, pyom.u[1:-2, 1:-2, :, pyom.taup1])
    pyom.du_mix[1:-2, 1:-2, :] += (pyom.u[1:-2, 1:-2, :, pyom.taup1] - aloc[1:-2, 1:-2, :]) / pyom.dt_mom * water_mask

    if pyom.enable_conserve_energy:
        # diagnose dissipation
        fxa = 0.5 * (pyom.kappa_gm[1:-2, 1:-2, :-1] + pyom.kappa_gm[2:-1, 1:-2, :-1])
        pyom.flux_top[1:-2, 1:-2, :-1] = fxa * (pyom.u[1:-2, 1:-2, 1:, pyom.taup1] - pyom.u[1:-2, 1:-2, :-1, pyom.taup1]) \
                                             / pyom.dzw[None, None, :-1] * pyom.maskU[1:-2, 1:-2, 1:] * pyom.maskU[1:-2, 1:-2, :-1]
        diss[1:-2, 1:-2, :-1] = (pyom.u[1:-2, 1:-2, 1:, pyom.tau] - pyom.u[1:-2, 1:-2, :-1, pyom.tau]) \
                                                                    * pyom.flux_top[1:-2, 1:-2, :-1] / pyom.dzw[None, None, :-1]
        diss[:,:,-1] = 0.0
        diss = numerics.ugrid_to_tgrid(diss,pyom)
        pyom.K_diss_gm[...] = diss

    if pyom.enable_implicit_vert_friction:
        aloc[...] = pyom.v[:,:,:,pyom.taup1]
    else:
        aloc[...] = pyom.v[:,:,:,pyom.tau]

    # implicit vertical friction of zonal momentum by GM
    ks = np.maximum(pyom.kbot[1:-2, 1:-2], pyom.kbot[1:-2, 2:-1]) - 1
    fxa = 0.5 * (pyom.kappa_gm[1:-2, 1:-2, :] + pyom.kappa_gm[1:-2, 2:-1, :])
    delta, a_tri, b_tri, c_tri = (np.zeros((pyom.nx+1,pyom.ny+1,pyom.nz)) for _ in range(4))
    delta[:, :, :-1] = pyom.dt_mom / pyom.dzw[None, None, :-1] * fxa[:, :, :-1] * pyom.maskV[1:-2, 1:-2, 1:] * pyom.maskV[1:-2, 1:-2, :-1]
    delta[-1] = 0.
    a_tri[:, :, 1:] = -delta[:, :, :-1] / pyom.dzt[None, None, 1:]
    b_tri_edge = 1 + delta / pyom.dzt[None, None, :]
    b_tri[:, :, 1:-1] = 1 + delta[:, :, 1:-1] / pyom.dzt[None, None, 1:-1] + delta[:, :, :-2] / pyom.dzt[None, None, 1:-1]
    b_tri[:, :, -1] = 1 + delta[:, :, -2] / pyom.dzt[-1]
    c_tri[...] = - delta / pyom.dzt[None, None, :]
    sol, water_mask = utilities.solve_implicit(ks, a_tri, b_tri, c_tri, aloc[1:-2, 1:-2, :], b_edge=b_tri_edge)
    pyom.v[1:-2, 1:-2, :, pyom.taup1] = np.where(water_mask, sol, pyom.v[1:-2, 1:-2, :, pyom.taup1])
    pyom.dv_mix[1:-2, 1:-2, :] += (pyom.v[1:-2, 1:-2, :, pyom.taup1] - aloc[1:-2, 1:-2, :]) / pyom.dt_mom * water_mask

    if pyom.enable_conserve_energy:
        # diagnose dissipation
        fxa = 0.5 * (pyom.kappa_gm[1:-2, 1:-2, :-1] + pyom.kappa_gm[1:-2, 2:-1, :-1])
        pyom.flux_top[1:-2, 1:-2, :-1] = fxa * (pyom.v[1:-2, 1:-2, 1:, pyom.taup1] - pyom.v[1:-2, 1:-2, :-1, pyom.taup1]) \
                                             / pyom.dzw[None, None, :-1] * pyom.maskV[1:-2, 1:-2, 1:] * pyom.maskV[1:-2, 1:-2, :-1]
        diss[1:-2, 1:-2, :-1] = (pyom.v[1:-2, 1:-2, 1:, pyom.tau] - pyom.v[1:-2, 1:-2, :-1, pyom.tau]) \
                                                                    * pyom.flux_top[1:-2, 1:-2, :-1] / pyom.dzw[None, None, :-1]
        diss[:,:,-1] = 0.0
        diss = numerics.vgrid_to_tgrid(diss,pyom)
        pyom.K_diss_gm += diss

import math

from .. import veros_method
from ..variables import allocate
from . import utilities


@veros_method(inline=True)
def dissipation_on_wgrid(vs, out, int_drhodX=None, aloc=None, ks=None):
    if aloc is None:
        aloc = allocate(vs, ('xt', 'yt', 'zw'))
        aloc[1:-1, 1:-1, :] = 0.5 * vs.grav / vs.rho_0 \
            * ((int_drhodX[2:, 1:-1, :] - int_drhodX[1:-1, 1:-1, :]) * vs.flux_east[1:-1, 1:-1, :]
             + (int_drhodX[1:-1, 1:-1, :] - int_drhodX[:-2, 1:-1, :]) * vs.flux_east[:-2, 1:-1, :]) \
            / (vs.dxt[1:-1, np.newaxis, np.newaxis] * vs.cost[np.newaxis, 1:-1, np.newaxis]) \
            + 0.5 * vs.grav / vs.rho_0 * ((int_drhodX[1:-1, 2:, :] - int_drhodX[1:-1, 1:-1, :]) * vs.flux_north[1:-1, 1:-1, :]
                                        + (int_drhodX[1:-1, 1:-1, :] - int_drhodX[1:-1, :-2, :]) * vs.flux_north[1:-1, :-2, :]) \
            / (vs.dyt[np.newaxis, 1:-1, np.newaxis] * vs.cost[np.newaxis, 1:-1, np.newaxis])

    if ks is None:
        ks = vs.kbot[:, :] - 1

    land_mask = ks >= 0
    edge_mask = land_mask[:, :, np.newaxis] & (
        np.arange(vs.nz - 1)[np.newaxis, np.newaxis, :] == ks[:, :, np.newaxis])
    water_mask = land_mask[:, :, np.newaxis] & (
        np.arange(vs.nz - 1)[np.newaxis, np.newaxis, :] > ks[:, :, np.newaxis])

    dzw_pad = utilities.pad_z_edges(vs, vs.dzw)
    out[:, :, :-1] += (0.5 * (aloc[:, :, :-1] + aloc[:, :, 1:])
                       + 0.5 * (aloc[:, :, :-1] * dzw_pad[np.newaxis, np.newaxis, :-3]
                               / vs.dzw[np.newaxis, np.newaxis, :-1])) * edge_mask
    out[:, :, :-1] += 0.5 * (aloc[:, :, :-1] + aloc[:, :, 1:]) * water_mask
    out[:, :, -1] += aloc[:, :, -1] * land_mask


@veros_method
def tempsalt_biharmonic(vs):
    """
    biharmonic mixing of temp and salinity,
    dissipation of dyn. Enthalpy is stored
    """
    fxa = math.sqrt(abs(vs.K_hbi))

    # update temp
    vs.dtemp_hmix[1:, 1:, :] = biharmonic_diffusion(vs, vs.temp[:, :, :, vs.tau], fxa)[1:, 1:, :]
    vs.temp[:, :, :, vs.taup1] += vs.dt_tracer * vs.dtemp_hmix * vs.maskT

    if vs.enable_conserve_energy:
        if vs.pyom_compatibility_mode:
            fxa = vs.int_drhodT[-3, -3, -1, vs.tau]
        vs.P_diss_hmix[...] = 0.
        dissipation_on_wgrid(vs, vs.P_diss_hmix, int_drhodX=vs.int_drhodT[..., vs.tau])

    # update salt
    vs.dsalt_hmix[1:, 1:, :] = biharmonic_diffusion(vs, vs.salt[:, :, :, vs.tau], fxa)[1:, 1:, :]
    vs.salt[:, :, :, vs.taup1] += vs.dt_tracer * vs.dsalt_hmix * vs.maskT

    if vs.enable_conserve_energy:
        dissipation_on_wgrid(vs, vs.P_diss_hmix, int_drhodX=vs.int_drhodS[..., vs.tau])


@veros_method
def tempsalt_diffusion(vs):
    """
    Diffusion of temp and salinity,
    dissipation of dyn. Enthalpy is stored
    """
    # horizontal diffusion of temperature
    vs.dtemp_hmix[1:, 1:, :] = horizontal_diffusion(vs, vs.temp[:, :, :, vs.tau], vs.K_h)[1:, 1:, :]
    vs.temp[:, :, :, vs.taup1] += vs.dt_tracer * vs.dtemp_hmix * vs.maskT

    if vs.enable_conserve_energy:
        vs.P_diss_hmix[...] = 0.
        dissipation_on_wgrid(vs, vs.P_diss_hmix, int_drhodX=vs.int_drhodT[..., vs.tau])

    # horizontal diffusion of salinity
    vs.dsalt_hmix[1:, 1:, :] = horizontal_diffusion(vs, vs.salt[:, :, :, vs.tau], vs.K_h)[1:, 1:, :]
    vs.salt[:, :, :, vs.taup1] += vs.dt_tracer * vs.dsalt_hmix * vs.maskT

    if vs.enable_conserve_energy:
        dissipation_on_wgrid(vs, vs.P_diss_hmix, int_drhodX=vs.int_drhodS[..., vs.tau])


@veros_method
def tempsalt_sources(vs):
    """
    Sources of temp and salinity,
    effect on dyn. Enthalpy is stored
    """
    vs.temp[:, :, :, vs.taup1] += vs.dt_tracer * vs.temp_source * vs.maskT
    vs.salt[:, :, :, vs.taup1] += vs.dt_tracer * vs.salt_source * vs.maskT

    if vs.enable_conserve_energy:
        aloc = -vs.grav / vs.rho_0 * vs.maskT * \
            (vs.int_drhodT[..., vs.tau] * vs.temp_source +
             vs.int_drhodS[..., vs.tau] * vs.salt_source)
        vs.P_diss_sources[...] = 0.
        dissipation_on_wgrid(vs, vs.P_diss_sources, aloc=aloc)


@veros_method
def biharmonic_diffusion(vs, tr, diffusivity):
    """
    Biharmonic mixing of tracer tr
    """
    del2 = allocate(vs, ('xt', 'yt', 'zt'))
    dtr = allocate(vs, ('xt', 'yt', 'zt'))

    vs.flux_east[:-1, :, :] = -diffusivity * (tr[1:, :, :] - tr[:-1, :, :]) \
            / (vs.cost[np.newaxis, :, np.newaxis] * vs.dxu[:-1, np.newaxis, np.newaxis]) \
            * vs.maskU[:-1, :, :]

    vs.flux_north[:, :-1, :] = -diffusivity * (tr[:, 1:, :] - tr[:, :-1, :]) \
            / vs.dyu[np.newaxis, :-1, np.newaxis] * vs.maskV[:, :-1, :] \
            * vs.cosu[np.newaxis, :-1, np.newaxis]

    del2[1:, 1:, :] = vs.maskT[1:, 1:, :] * (vs.flux_east[1:, 1:, :] - vs.flux_east[:-1, 1:, :]) \
            / (vs.cost[np.newaxis, 1:, np.newaxis] * vs.dxt[1:, np.newaxis, np.newaxis]) \
            + (vs.flux_north[1:, 1:, :] - vs.flux_north[1:, :-1, :]) \
            / (vs.cost[np.newaxis, 1:, np.newaxis] * vs.dyt[np.newaxis, 1:, np.newaxis])

    utilities.enforce_boundaries(vs, del2)

    vs.flux_east[:-1, :, :] = diffusivity * (del2[1:, :, :] - del2[:-1, :, :]) \
            / (vs.cost[np.newaxis, :, np.newaxis] * vs.dxu[:-1, np.newaxis, np.newaxis]) \
            * vs.maskU[:-1, :, :]
    vs.flux_north[:, :-1, :] = diffusivity * (del2[:, 1:, :] - del2[:, :-1, :]) \
            / vs.dyu[np.newaxis, :-1, np.newaxis] * vs.maskV[:, :-1, :] \
            * vs.cosu[np.newaxis, :-1, np.newaxis]

    vs.flux_east[-1, :, :] = 0.
    vs.flux_north[:, -1, :] = 0.

    dtr[1:, 1:, :] = (vs.flux_east[1:, 1:, :] - vs.flux_east[:-1, 1:, :]) \
            / (vs.cost[np.newaxis, 1:, np.newaxis] * vs.dxt[1:, np.newaxis, np.newaxis]) \
            + (vs.flux_north[1:, 1:, :] - vs.flux_north[1:, :-1, :]) \
            / (vs.cost[np.newaxis, 1:, np.newaxis] * vs.dyt[np.newaxis, 1:, np.newaxis])

    dtr[...] *= vs.maskT

    return dtr


@veros_method
def horizontal_diffusion(vs, tr, diffusivity):
    """
    Diffusion of tracer tr
    """
    dtr_hmix = allocate(vs, ('xt', 'yt', 'zt'))

    # horizontal diffusion of tracer
    vs.flux_east[:-1, :, :] = diffusivity * (tr[1:, :, :] - tr[:-1, :, :]) \
        / (vs.cost[np.newaxis, :, np.newaxis] * vs.dxu[:-1, np.newaxis, np.newaxis])\
        * vs.maskU[:-1, :, :]
    vs.flux_east[-1, :, :] = 0.

    vs.flux_north[:, :-1, :] = diffusivity * (tr[:, 1:, :] - tr[:, :-1, :]) \
        / vs.dyu[np.newaxis, :-1, np.newaxis] * vs.maskV[:, :-1, :]\
        * vs.cosu[np.newaxis, :-1, np.newaxis]
    vs.flux_north[:, -1, :] = 0.

    if vs.enable_hor_friction_cos_scaling:
        vs.flux_east[...] *= vs.cost[np.newaxis, :, np.newaxis] ** vs.hor_friction_cosPower
        vs.flux_north[...] *= vs.cosu[np.newaxis, :, np.newaxis] ** vs.hor_friction_cosPower

    dtr_hmix[1:, 1:, :] = ((vs.flux_east[1:, 1:, :] - vs.flux_east[:-1, 1:, :])
                           / (vs.cost[np.newaxis, 1:, np.newaxis] * vs.dxt[1:, np.newaxis, np.newaxis])
                           + (vs.flux_north[1:, 1:, :] - vs.flux_north[1:, :-1, :])
                           / (vs.cost[np.newaxis, 1:, np.newaxis] * vs.dyt[np.newaxis, 1:, np.newaxis]))\
                                * vs.maskT[1:, 1:, :]

    return dtr_hmix

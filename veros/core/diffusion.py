import math

from .. import veros_method, veros_inline_method
from . import cyclic, utilities

@veros_inline_method
def dissipation_on_wgrid(veros, p_arr, int_drhodX=None, aloc=None, ks=None):
    if aloc is None:
        aloc = np.zeros_like(p_arr)
        aloc[1:-1,1:-1,:] = 0.5 * veros.grav / veros.rho_0 * ((int_drhodX[2:,1:-1,:] - int_drhodX[1:-1,1:-1,:]) * veros.flux_east[1:-1,1:-1,:] \
                                                           +(int_drhodX[1:-1,1:-1,:] - int_drhodX[:-2,1:-1,:]) * veros.flux_east[:-2,1:-1,:]) \
                                                         / (veros.dxt[1:-1,np.newaxis,np.newaxis] * veros.cost[np.newaxis,1:-1,np.newaxis]) \
                          + 0.5 * veros.grav / veros.rho_0 * ((int_drhodX[1:-1,2:,:] - int_drhodX[1:-1,1:-1,:]) * veros.flux_north[1:-1,1:-1,:] \
                                                           +(int_drhodX[1:-1,1:-1,:] - int_drhodX[1:-1,:-2,:]) * veros.flux_north[1:-1,:-2,:]) \
                                                         / (veros.dyt[np.newaxis,1:-1,np.newaxis] * veros.cost[np.newaxis,1:-1,np.newaxis])
    if ks is None:
        ks = veros.kbot[:,:] - 1

    land_mask = ks >= 0
    edge_mask = land_mask[:, :, np.newaxis] & (np.arange(veros.nz-1)[np.newaxis, np.newaxis, :] == ks[:,:,np.newaxis])
    water_mask = land_mask[:, :, np.newaxis] & (np.arange(veros.nz-1)[np.newaxis, np.newaxis, :] > ks[:,:,np.newaxis])

    dzw_pad = utilities.pad_z_edges(veros, veros.dzw)
    p_arr[:, :, :-1] += (0.5 * (aloc[:,:,:-1] + aloc[:,:,1:]) + 0.5 * (aloc[:, :, :-1] * dzw_pad[np.newaxis, np.newaxis, :-3] / veros.dzw[np.newaxis, np.newaxis, :-1])) * edge_mask
    p_arr[:, :, :-1] += 0.5 * (aloc[:,:,:-1] + aloc[:,:,1:]) * water_mask
    p_arr[:, :, -1] += aloc[:,:,-1] * land_mask

@veros_method
def tempsalt_biharmonic(veros):
    """
    biharmonic mixing of veros.temp and salinity,
    dissipation of dyn. Enthalpy is stored
    """
    aloc = np.zeros((veros.nx+4, veros.ny+4, veros.nz))
    del2 = np.zeros((veros.nx+4, veros.ny+4, veros.nz))

    fxa = math.sqrt(abs(veros.K_hbi))

    veros.flux_east[:-1, :, :] = -fxa * (veros.temp[1:, :, :, veros.tau] - veros.temp[:-1, :, :, veros.tau]) \
                                / (veros.cost[np.newaxis, :, np.newaxis] * veros.dxu[:-1, np.newaxis, np.newaxis]) * veros.maskU[:-1, :, :]
    veros.flux_east[:, -1, :] = 0.
    veros.flux_north[:, :-1, :] = -fxa * (veros.temp[:, 1:, :, veros.tau] - veros.temp[:, :-1, :, veros.tau]) \
                                 / veros.dyu[np.newaxis, :-1, np.newaxis] * veros.maskV[:, :-1, :] * veros.cosu[np.newaxis, :-1, np.newaxis]
    veros.flux_north[:, -1, :] = 0.

    del2[1:, 1:, :] = veros.maskT[1:, 1:, :] * (veros.flux_east[1:, 1:, :] - veros.flux_east[:-1, 1:, :]) \
                      / (veros.cost[np.newaxis, 1:, np.newaxis] * veros.dxt[1:, np.newaxis, np.newaxis]) \
                      + (veros.flux_north[1:, 1:, :] - veros.flux_north[1:, :-1, :]) \
                      / (veros.cost[np.newaxis, 1:, np.newaxis] * veros.dyt[np.newaxis, 1:, np.newaxis])

    if veros.enable_cyclic_x:
        cyclic.setcyclic_x(del2)

    veros.flux_east[:-1, :, :] = fxa * (del2[1:, :, :] - del2[:-1, :, :]) / (veros.cost[np.newaxis, :, np.newaxis] * veros.dxu[:-1, np.newaxis, np.newaxis]) * veros.maskU[:-1, :, :]
    veros.flux_north[:, :-1, :] = fxa * (del2[:, 1:, :] - del2[:, :-1, :]) / veros.dyu[np.newaxis, :-1, np.newaxis] * veros.maskV[:, :-1, :] * veros.cosu[np.newaxis, :-1, np.newaxis]
    veros.flux_east[-1,:,:] = 0.
    veros.flux_north[:,-1,:] = 0.

    # update tendency
    veros.dtemp_hmix[1:, 1:, :] = veros.maskT[1:, 1:, :] * (veros.flux_east[1:, 1:, :] - veros.flux_east[:-1, 1:, :]) \
                                 / (veros.cost[np.newaxis, 1:, np.newaxis] * veros.dxt[1:, np.newaxis, np.newaxis]) \
                                 + (veros.flux_north[1:, 1:, :] - veros.flux_north[1:, :-1, :]) \
                                 / (veros.cost[np.newaxis, 1:, np.newaxis] * veros.dyt[np.newaxis, 1:, np.newaxis])
    veros.temp[:,:,:,veros.taup1] += veros.dt_tracer * veros.dtemp_hmix * veros.maskT

    if veros.enable_conserve_energy:
        if veros.pyom_compatibility_mode:
            fxa = veros.int_drhodT[-3, -3, -1, veros.tau]
        veros.P_diss_hmix[...] = 0.
        dissipation_on_wgrid(veros, veros.P_diss_hmix, int_drhodX=veros.int_drhodT[..., veros.tau])

    veros.flux_east[:-1, :, :] = -fxa * (veros.salt[1:, :, :, veros.tau] - veros.salt[:-1, :, :, veros.tau]) \
                                    / (veros.cost[np.newaxis, :, np.newaxis] * veros.dxu[:-1, np.newaxis, np.newaxis]) * veros.maskU[:-1, :, :]
    veros.flux_north[:, :-1, :] = -fxa * (veros.salt[:, 1:, :, veros.tau] - veros.salt[:, :-1, :, veros.tau]) \
                                  / veros.dyu[np.newaxis, :-1, np.newaxis] * veros.maskV[:, :-1, :] * veros.cosu[np.newaxis, :-1, np.newaxis]
    veros.flux_east[-1,:,:] = 0.

    veros.flux_north[:,-1,:] = 0.

    del2[1:, 1:, :] = veros.maskT[1:, 1:, :] * (veros.flux_east[1:, 1:, :] - veros.flux_east[:-1, 1:, :]) \
                        / (veros.cost[np.newaxis, 1:, np.newaxis] * veros.dxt[1:, np.newaxis, np.newaxis]) \
                                            + (veros.flux_north[1:, 1:, :] - veros.flux_north[1:, :-1, :]) \
                        / (veros.cost[np.newaxis, 1:, np.newaxis] * veros.dyt[np.newaxis, 1:, np.newaxis])
    if veros.enable_cyclic_x:
        cyclic.setcyclic_x(del2)

    veros.flux_east[:-1, :, :] = fxa * (del2[1:, :, :] - del2[:-1, :, :]) / (veros.cost[np.newaxis, :, np.newaxis] * veros.dxu[:-1, np.newaxis, np.newaxis]) \
                                * veros.maskU[:-1, :, :]
    veros.flux_north[:, :-1, :] = fxa * (del2[:, 1:, :] - del2[:, :-1, :]) / veros.dyu[np.newaxis, :-1, np.newaxis] \
                                * veros.maskV[:, :-1, :] * veros.cosu[np.newaxis, :-1, np.newaxis]
    veros.flux_east[-1,:,:] = 0.
    veros.flux_north[:,-1,:] = 0.

    # update tendency
    veros.dsalt_hmix[1:, 1:, :] = veros.maskT[1:, 1:, :] * (veros.flux_east[1:, 1:, :] - veros.flux_east[:-1, 1:, :]) \
                                 / (veros.cost[np.newaxis, 1:, np.newaxis] * veros.dxt[1:, np.newaxis, np.newaxis]) \
                                                       + (veros.flux_north[1:, 1:, :] - veros.flux_north[1:, :-1, :]) \
                                 / (veros.cost[np.newaxis, 1:, np.newaxis] * veros.dyt[np.newaxis, 1:, np.newaxis])
    veros.salt[:,:,:,veros.taup1] += veros.dt_tracer * veros.dsalt_hmix * veros.maskT

    if veros.enable_conserve_energy:
        dissipation_on_wgrid(veros, veros.P_diss_hmix, int_drhodX=veros.int_drhodS[..., veros.tau])

@veros_method
def tempsalt_diffusion(veros):
    """
    Diffusion of veros.temp and salinity,
    dissipation of dyn. Enthalpy is stored
    """
    aloc = np.zeros((veros.nx+4, veros.ny+4, veros.nz))

    # horizontal diffusion of temperature
    veros.flux_east[:-1, :, :] = veros.K_h * (veros.temp[1:, :, :, veros.tau] - veros.temp[:-1, :, :, veros.tau]) \
                                / (veros.cost[np.newaxis, :, np.newaxis] * veros.dxu[:-1, np.newaxis, np.newaxis]) * veros.maskU[:-1, :, :]
    veros.flux_east[-1,:,:] = 0.

    veros.flux_north[:, :-1, :] = veros.K_h * (veros.temp[:, 1:, :, veros.tau] - veros.temp[:, :-1, :, veros.tau]) \
                                 / veros.dyu[np.newaxis, :-1, np.newaxis] * veros.maskV[:, :-1, :] * veros.cosu[np.newaxis, :-1, np.newaxis]
    veros.flux_north[:,-1,:] = 0.

    if veros.enable_hor_friction_cos_scaling:
        veros.flux_east[...] *= veros.cost[np.newaxis, :, np.newaxis] ** veros.hor_friction_cosPower
        veros.flux_north[...] *= veros.cosu[np.newaxis, :, np.newaxis] ** veros.hor_friction_cosPower

    veros.dtemp_hmix[1:, 1:, :] = veros.maskT[1:, 1:, :] * ((veros.flux_east[1:, 1:, :] - veros.flux_east[:-1, 1:, :]) \
                                                          / (veros.cost[np.newaxis, 1:, np.newaxis] * veros.dxt[1:, np.newaxis, np.newaxis]) \
                                                        + (veros.flux_north[1:, 1:, :] - veros.flux_north[1:, :-1, :]) \
                                                          / (veros.cost[np.newaxis, 1:, np.newaxis] * veros.dyt[np.newaxis, 1:, np.newaxis]))
    veros.temp[:,:,:,veros.taup1] += veros.dt_tracer * veros.dtemp_hmix * veros.maskT

    if veros.enable_conserve_energy:
        if veros.pyom_compatibility_mode:
            fxa = veros.int_drhodT[-3, -3, -1, veros.tau]
        veros.P_diss_hmix[...] = 0.
        dissipation_on_wgrid(veros, veros.P_diss_hmix, int_drhodX=veros.int_drhodT[..., veros.tau])

    # horizontal diffusion of salinity
    veros.flux_east[:-1, :, :] = veros.K_h * (veros.salt[1:, :, :, veros.tau] - veros.salt[:-1, :, :, veros.tau]) \
                                / (veros.cost[np.newaxis, :, np.newaxis] * veros.dxu[:-1, np.newaxis, np.newaxis]) * veros.maskU[:-1, :, :]
    veros.flux_east[-1,:,:] = 0.

    veros.flux_north[:, :-1, :] = veros.K_h * (veros.salt[:, 1:, :, veros.tau] - veros.salt[:, :-1, :, veros.tau]) \
                                 / veros.dyu[np.newaxis, :-1, np.newaxis] * veros.maskV[:, :-1, :] * veros.cosu[np.newaxis, :-1, np.newaxis]
    veros.flux_north[:,-1,:] = 0.

    if veros.enable_hor_friction_cos_scaling:
        veros.flux_east[...] *= veros.cost[np.newaxis, :, np.newaxis] ** veros.hor_friction_cosPower
        veros.flux_north[...] *= veros.cosu[np.newaxis, :, np.newaxis] ** veros.hor_friction_cosPower

    veros.dsalt_hmix[1:, 1:, :] = veros.maskT[1:, 1:, :] * ((veros.flux_east[1:, 1:, :] - veros.flux_east[:-1, 1:, :]) \
                                                            / (veros.cost[np.newaxis, 1:, np.newaxis] * veros.dxt[1:, np.newaxis, np.newaxis]) \
                                                       + (veros.flux_north[1:, 1:, :] - veros.flux_north[1:, :-1, :]) \
                                                            / (veros.cost[np.newaxis, 1:, np.newaxis] * veros.dyt[np.newaxis, 1:, np.newaxis]))
    veros.salt[:,:,:,veros.taup1] += veros.dt_tracer * veros.dsalt_hmix * veros.maskT

    if veros.enable_conserve_energy:
        dissipation_on_wgrid(veros, veros.P_diss_hmix, int_drhodX=veros.int_drhodS[..., veros.tau])

@veros_method
def tempsalt_sources(veros):
    """
    Sources of veros.temp and salinity,
    effect on dyn. Enthalpy is stored
    """
    veros.temp[:,:,:,veros.taup1] += veros.dt_tracer * veros.temp_source * veros.maskT
    veros.salt[:,:,:,veros.taup1] += veros.dt_tracer * veros.salt_source * veros.maskT

    if veros.enable_conserve_energy:
        aloc = -veros.grav / veros.rho_0 * veros.maskT * (veros.int_drhodT[...,veros.tau] * veros.temp_source + veros.int_drhodS[...,veros.tau] * veros.salt_source)
        veros.P_diss_sources[...] = 0.
        dissipation_on_wgrid(veros, veros.P_diss_sources, aloc=aloc)

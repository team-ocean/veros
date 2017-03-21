import math

from .. import pyom_method, pyom_inline_method
from . import cyclic, utilities

@pyom_inline_method
def dissipation_on_wgrid(pyom, p_arr, int_drhodX=None, aloc=None, ks=None):
    if aloc is None:
        aloc = np.zeros_like(p_arr)
        aloc[1:-1,1:-1,:] = 0.5 * pyom.grav / pyom.rho_0 * ((int_drhodX[2:,1:-1,:] - int_drhodX[1:-1,1:-1,:]) * pyom.flux_east[1:-1,1:-1,:] \
                                                           +(int_drhodX[1:-1,1:-1,:] - int_drhodX[:-2,1:-1,:]) * pyom.flux_east[:-2,1:-1,:]) \
                                                         / (pyom.dxt[1:-1,np.newaxis,np.newaxis] * pyom.cost[np.newaxis,1:-1,np.newaxis]) \
                          + 0.5 * pyom.grav / pyom.rho_0 * ((int_drhodX[1:-1,2:,:] - int_drhodX[1:-1,1:-1,:]) * pyom.flux_north[1:-1,1:-1,:] \
                                                           +(int_drhodX[1:-1,1:-1,:] - int_drhodX[1:-1,:-2,:]) * pyom.flux_north[1:-1,:-2,:]) \
                                                         / (pyom.dyt[np.newaxis,1:-1,np.newaxis] * pyom.cost[np.newaxis,1:-1,np.newaxis])
    if ks is None:
        ks = pyom.kbot[:,:] - 1

    land_mask = ks >= 0
    edge_mask = land_mask[:, :, np.newaxis] & (np.arange(pyom.nz-1)[np.newaxis, np.newaxis, :] == ks[:,:,np.newaxis])
    water_mask = land_mask[:, :, np.newaxis] & (np.arange(pyom.nz-1)[np.newaxis, np.newaxis, :] > ks[:,:,np.newaxis])

    dzw_pad = utilities.pad_z_edges(pyom, pyom.dzw)
    p_arr[:, :, :-1] += (0.5 * (aloc[:,:,:-1] + aloc[:,:,1:]) + 0.5 * (aloc[:, :, :-1] * dzw_pad[np.newaxis, np.newaxis, :-3] / pyom.dzw[np.newaxis, np.newaxis, :-1])) * edge_mask
    p_arr[:, :, :-1] += 0.5 * (aloc[:,:,:-1] + aloc[:,:,1:]) * water_mask
    p_arr[:, :, -1] += aloc[:,:,-1] * land_mask

@pyom_method
def tempsalt_biharmonic(pyom):
    """
    biharmonic mixing of pyom.temp and salinity,
    dissipation of dyn. Enthalpy is stored
    """
    aloc = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    del2 = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    fxa = math.sqrt(abs(pyom.K_hbi))

    pyom.flux_east[:-1, :, :] = -fxa * (pyom.temp[1:, :, :, pyom.tau] - pyom.temp[:-1, :, :, pyom.tau]) \
                                / (pyom.cost[np.newaxis, :, np.newaxis] * pyom.dxu[:-1, np.newaxis, np.newaxis]) * pyom.maskU[:-1, :, :]
    pyom.flux_east[:, -1, :] = 0.
    pyom.flux_north[:, :-1, :] = -fxa * (pyom.temp[:, 1:, :, pyom.tau] - pyom.temp[:, :-1, :, pyom.tau]) \
                                 / pyom.dyu[np.newaxis, :-1, np.newaxis] * pyom.maskV[:, :-1, :] * pyom.cosu[np.newaxis, :-1, np.newaxis]
    pyom.flux_north[:, -1, :] = 0.

    del2[1:, 1:, :] = pyom.maskT[1:, 1:, :] * (pyom.flux_east[1:, 1:, :] - pyom.flux_east[:-1, 1:, :]) \
                      / (pyom.cost[np.newaxis, 1:, np.newaxis] * pyom.dxt[1:, np.newaxis, np.newaxis]) \
                      + (pyom.flux_north[1:, 1:, :] - pyom.flux_north[1:, :-1, :]) \
                      / (pyom.cost[np.newaxis, 1:, np.newaxis] * pyom.dyt[np.newaxis, 1:, np.newaxis])

    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(del2)

    pyom.flux_east[:-1, :, :] = fxa * (del2[1:, :, :] - del2[:-1, :, :]) / (pyom.cost[np.newaxis, :, np.newaxis] * pyom.dxu[:-1, np.newaxis, np.newaxis]) * pyom.maskU[:-1, :, :]
    pyom.flux_north[:, :-1, :] = fxa * (del2[:, 1:, :] - del2[:, :-1, :]) / pyom.dyu[np.newaxis, :-1, np.newaxis] * pyom.maskV[:, :-1, :] * pyom.cosu[np.newaxis, :-1, np.newaxis]
    pyom.flux_east[-1,:,:] = 0.
    pyom.flux_north[:,-1,:] = 0.

    # update tendency
    pyom.dtemp_hmix[1:, 1:, :] = pyom.maskT[1:, 1:, :] * (pyom.flux_east[1:, 1:, :] - pyom.flux_east[:-1, 1:, :]) \
                                 / (pyom.cost[np.newaxis, 1:, np.newaxis] * pyom.dxt[1:, np.newaxis, np.newaxis]) \
                                 + (pyom.flux_north[1:, 1:, :] - pyom.flux_north[1:, :-1, :]) \
                                 / (pyom.cost[np.newaxis, 1:, np.newaxis] * pyom.dyt[np.newaxis, 1:, np.newaxis])
    pyom.temp[:,:,:,pyom.taup1] += pyom.dt_tracer * pyom.dtemp_hmix * pyom.maskT

    if pyom.enable_conserve_energy:
        fxa = pyom.int_drhodT[-3, -3, -1, pyom.tau] # NOTE: probably a mistake in the fortran code
        pyom.P_diss_hmix[...] = 0.
        dissipation_on_wgrid(pyom, pyom.P_diss_hmix, int_drhodX=pyom.int_drhodT[..., pyom.tau])

    pyom.flux_east[:-1, :, :] = -fxa * (pyom.salt[1:, :, :, pyom.tau] - pyom.salt[:-1, :, :, pyom.tau]) \
                                    / (pyom.cost[np.newaxis, :, np.newaxis] * pyom.dxu[:-1, np.newaxis, np.newaxis]) * pyom.maskU[:-1, :, :]
    pyom.flux_north[:, :-1, :] = -fxa * (pyom.salt[:, 1:, :, pyom.tau] - pyom.salt[:, :-1, :, pyom.tau]) \
                                  / pyom.dyu[np.newaxis, :-1, np.newaxis] * pyom.maskV[:, :-1, :] * pyom.cosu[np.newaxis, :-1, np.newaxis]
    pyom.flux_east[-1,:,:] = 0.

    pyom.flux_north[:,-1,:] = 0.

    del2[1:, 1:, :] = pyom.maskT[1:, 1:, :] * (pyom.flux_east[1:, 1:, :] - pyom.flux_east[:-1, 1:, :]) \
                        / (pyom.cost[np.newaxis, 1:, np.newaxis] * pyom.dxt[1:, np.newaxis, np.newaxis]) \
                                            + (pyom.flux_north[1:, 1:, :] - pyom.flux_north[1:, :-1, :]) \
                        / (pyom.cost[np.newaxis, 1:, np.newaxis] * pyom.dyt[np.newaxis, 1:, np.newaxis])
    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(del2)

    pyom.flux_east[:-1, :, :] = fxa * (del2[1:, :, :] - del2[:-1, :, :]) / (pyom.cost[np.newaxis, :, np.newaxis] * pyom.dxu[:-1, np.newaxis, np.newaxis]) \
                                * pyom.maskU[:-1, :, :]
    pyom.flux_north[:, :-1, :] = fxa * (del2[:, 1:, :] - del2[:, :-1, :]) / pyom.dyu[np.newaxis, :-1, np.newaxis] \
                                * pyom.maskV[:, :-1, :] * pyom.cosu[np.newaxis, :-1, np.newaxis]
    pyom.flux_east[-1,:,:] = 0.
    pyom.flux_north[:,-1,:] = 0.

    # update tendency
    pyom.dsalt_hmix[1:, 1:, :] = pyom.maskT[1:, 1:, :] * (pyom.flux_east[1:, 1:, :] - pyom.flux_east[:-1, 1:, :]) \
                                 / (pyom.cost[np.newaxis, 1:, np.newaxis] * pyom.dxt[1:, np.newaxis, np.newaxis]) \
                                                       + (pyom.flux_north[1:, 1:, :] - pyom.flux_north[1:, :-1, :]) \
                                 / (pyom.cost[np.newaxis, 1:, np.newaxis] * pyom.dyt[np.newaxis, 1:, np.newaxis])
    pyom.salt[:,:,:,pyom.taup1] += pyom.dt_tracer * pyom.dsalt_hmix * pyom.maskT

    if pyom.enable_conserve_energy:
        dissipation_on_wgrid(pyom, pyom.P_diss_hmix, int_drhodX=pyom.int_drhodS[..., pyom.tau])

@pyom_method
def tempsalt_diffusion(pyom):
    """
    Diffusion of pyom.temp and salinity,
    dissipation of dyn. Enthalpy is stored
    """
    aloc = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    # horizontal diffusion of temperature
    pyom.flux_east[:-1, :, :] = pyom.K_h * (pyom.temp[1:, :, :, pyom.tau] - pyom.temp[:-1, :, :, pyom.tau]) \
                                / (pyom.cost[np.newaxis, :, np.newaxis] * pyom.dxu[:-1, np.newaxis, np.newaxis]) * pyom.maskU[:-1, :, :]
    pyom.flux_east[-1,:,:] = 0.

    pyom.flux_north[:, :-1, :] = pyom.K_h * (pyom.temp[:, 1:, :, pyom.tau] - pyom.temp[:, :-1, :, pyom.tau]) \
                                 / pyom.dyu[np.newaxis, :-1, np.newaxis] * pyom.maskV[:, :-1, :] * pyom.cosu[np.newaxis, :-1, np.newaxis]
    pyom.flux_north[:,-1,:] = 0.

    if pyom.enable_hor_friction_cos_scaling:
        pyom.flux_east[...] *= pyom.cost[np.newaxis, :, np.newaxis] ** pyom.hor_friction_cosPower
        pyom.flux_north[...] *= pyom.cosu[np.newaxis, :, np.newaxis] ** pyom.hor_friction_cosPower

    pyom.dtemp_hmix[1:, 1:, :] = pyom.maskT[1:, 1:, :] * ((pyom.flux_east[1:, 1:, :] - pyom.flux_east[:-1, 1:, :]) \
                                                          / (pyom.cost[np.newaxis, 1:, np.newaxis] * pyom.dxt[1:, np.newaxis, np.newaxis]) \
                                                        + (pyom.flux_north[1:, 1:, :] - pyom.flux_north[1:, :-1, :]) \
                                                          / (pyom.cost[np.newaxis, 1:, np.newaxis] * pyom.dyt[np.newaxis, 1:, np.newaxis]))
    pyom.temp[:,:,:,pyom.taup1] += pyom.dt_tracer * pyom.dtemp_hmix * pyom.maskT

    if pyom.enable_conserve_energy:
        fxa = pyom.int_drhodT[-3, -3, -1, pyom.tau] # NOTE: probably a mistake in the fortran code
        pyom.P_diss_hmix[...] = 0.
        dissipation_on_wgrid(pyom, pyom.P_diss_hmix, int_drhodX=pyom.int_drhodT[..., pyom.tau])

    # horizontal diffusion of salinity
    pyom.flux_east[:-1, :, :] = pyom.K_h * (pyom.salt[1:, :, :, pyom.tau] - pyom.salt[:-1, :, :, pyom.tau]) \
                                / (pyom.cost[np.newaxis, :, np.newaxis] * pyom.dxu[:-1, np.newaxis, np.newaxis]) * pyom.maskU[:-1, :, :]
    pyom.flux_east[-1,:,:] = 0.

    pyom.flux_north[:, :-1, :] = pyom.K_h * (pyom.salt[:, 1:, :, pyom.tau] - pyom.salt[:, :-1, :, pyom.tau]) \
                                 / pyom.dyu[np.newaxis, :-1, np.newaxis] * pyom.maskV[:, :-1, :] * pyom.cosu[np.newaxis, :-1, np.newaxis]
    pyom.flux_north[:,-1,:] = 0.

    if pyom.enable_hor_friction_cos_scaling:
        pyom.flux_east[...] *= pyom.cost[np.newaxis, :, np.newaxis] ** pyom.hor_friction_cosPower
        pyom.flux_north[...] *= pyom.cosu[np.newaxis, :, np.newaxis] ** pyom.hor_friction_cosPower

    pyom.dsalt_hmix[1:, 1:, :] = pyom.maskT[1:, 1:, :] * ((pyom.flux_east[1:, 1:, :] - pyom.flux_east[:-1, 1:, :]) \
                                                            / (pyom.cost[np.newaxis, 1:, np.newaxis] * pyom.dxt[1:, np.newaxis, np.newaxis]) \
                                                       + (pyom.flux_north[1:, 1:, :] - pyom.flux_north[1:, :-1, :]) \
                                                            / (pyom.cost[np.newaxis, 1:, np.newaxis] * pyom.dyt[np.newaxis, 1:, np.newaxis]))
    pyom.salt[:,:,:,pyom.taup1] += pyom.dt_tracer * pyom.dsalt_hmix * pyom.maskT

    if pyom.enable_conserve_energy:
        dissipation_on_wgrid(pyom, pyom.P_diss_hmix, int_drhodX=pyom.int_drhodS[..., pyom.tau])

@pyom_method
def tempsalt_sources(pyom):
    """
    Sources of pyom.temp and salinity,
    effect on dyn. Enthalpy is stored
    """
    pyom.temp[:,:,:,pyom.taup1] += pyom.dt_tracer * pyom.temp_source * pyom.maskT
    pyom.salt[:,:,:,pyom.taup1] += pyom.dt_tracer * pyom.salt_source * pyom.maskT

    if pyom.enable_conserve_energy:
        aloc = -pyom.grav / pyom.rho_0 * pyom.maskT * (pyom.int_drhodT[...,pyom.tau] * pyom.temp_source + pyom.int_drhodS[...,pyom.tau] * pyom.salt_source)
        pyom.P_diss_sources[...] = 0.
        dissipation_on_wgrid(pyom, pyom.P_diss_sources, aloc=aloc)

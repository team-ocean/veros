import numpy as np
import math

from climate.pyom import cyclic, utilities


def dissipation_on_wgrid(P, pyom, int_drhodX=None, aloc=None):
    if aloc is None:
        aloc = np.zeros_like(P)
        aloc[1:-1,1:-1,:] = 0.5 * pyom.grav / pyom.rho_0 * ((int_drhodX[2:,1:-1,:]-int_drhodX[1:-1,1:-1,:]) * pyom.flux_east[1:-1,1:-1,:] \
                                                           +(int_drhodX[1:-1,1:-1,:]-int_drhodX[:-2,1:-1,:]) * pyom.flux_east[:-2,1:-1,:]) \
                                                         / (pyom.dxt[1:-1,None,None] * pyom.cost[None,1:-1,None]) \
                          + 0.5 * pyom.grav / pyom.rho_0 * ((int_drhodX[1:-1,2:,:]-int_drhodX[1:-1,1:-1,:]) * pyom.flux_north[1:-1,1:-1,:] \
                                                           +(int_drhodX[1:-1,1:-1,:]-int_drhodX[1:-1,:-2,:]) * pyom.flux_north[1:-1,:-2,:]) \
                                                         / (pyom.dyt[None,1:-1,None] * pyom.cost[None,1:-1,None])
    ks = pyom.kbot[:,:] - 1
    land_mask = (ks >= 0)
    edge_mask = land_mask[:, :, None] & (np.indices((pyom.nx+4, pyom.ny+4, pyom.nz-1))[2] == ks[:,:,None])
    water_mask = land_mask[:, :, None] & (np.indices((pyom.nx+4, pyom.ny+4, pyom.nz-1))[2] > ks[:,:,None])
    if np.count_nonzero(land_mask):
        dzw_pad = utilities.pad_z_edges(pyom.dzw)
        P[:, :, :-1][edge_mask] += (0.5 * (aloc[:,:,:-1] + aloc[:,:,1:]) + 0.5 * (aloc[:, :, :-1] * dzw_pad[None, None, :-3] / pyom.dzw[None, None, :-1]))[edge_mask]
        P[:, :, :-1][water_mask] += 0.5 * (aloc[:,:,:-1] + aloc[:,:,1:])[water_mask]
        P[:, :, -1][land_mask] += aloc[:,:,-1][land_mask]


def tempsalt_biharmonic(pyom):
    """
    biharmonic mixing of pyom.temp and salinity,
    dissipation of dyn. Enthalpy is stored
    """
    aloc = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    del2 = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    fxa = math.sqrt(abs(pyom.K_hbi))

    pyom.flux_east[:-1, :, :] = -fxa * (pyom.temp[1:, :, :, pyom.tau] - pyom.temp[:-1, :, :, pyom.tau]) \
                                / (pyom.cost[None, :, None] * pyom.dxu[:-1, None, None]) * pyom.maskU[:-1, :, :]
    pyom.flux_east[:, -1, :] = 0.
    pyom.flux_north[:, :-1, :] = -fxa * (pyom.temp[:, 1:, :, pyom.tau] - pyom.temp[:, :-1, :, pyom.tau]) \
                                 / pyom.dyu[None, :-1, None] * pyom.maskV[:, :-1, :] * pyom.cosu[None, :-1, None]
    pyom.flux_north[:, -1, :] = 0.

    del2[1:, 1:, :] = pyom.maskT[1:, 1:, :] * (pyom.flux_east[1:, 1:, :] - pyom.flux_east[:-1, 1:, :]) \
                      / (pyom.cost[None, 1:, None] * pyom.dxt[1:, None, None]) \
                      + (pyom.flux_north[1:, 1:, :] - pyom.flux_north[1:, :-1, :]) \
                      / (pyom.cost[None, 1:, None] * pyom.dyt[None, 1:, None])

    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(del2)

    pyom.flux_east[:-1, :, :] = fxa * (del2[1:, :, :] - del2[:-1, :, :]) / (pyom.cost[None, :, None] * pyom.dxu[:-1, None, None]) * pyom.maskU[:-1, :, :]
    pyom.flux_north[:, :-1, :] = fxa * (del2[:, 1:, :] - del2[:, :-1, :]) / pyom.dyu[None, :-1, None] * pyom.maskV[:, :-1, :] * pyom.cosu[None, :-1, None]
    pyom.flux_east[-1,:,:] = 0.
    pyom.flux_north[:,-1,:] = 0.

    # update tendency
    pyom.dtemp_hmix[1:, 1:, :] = pyom.maskT[1:, 1:, :] * (pyom.flux_east[1:, 1:, :] - pyom.flux_east[:-1, 1:, :]) \
                                 / (pyom.cost[None, 1:, None] * pyom.dxt[1:, None, None]) \
                                 + (pyom.flux_north[1:, 1:, :] - pyom.flux_north[1:, :-1, :]) \
                                 / (pyom.cost[None, 1:, None] * pyom.dyt[None, 1:, None])
    pyom.temp[:,:,:,pyom.taup1] = pyom.temp[:,:,:,pyom.taup1] + pyom.dt_tracer * pyom.dtemp_hmix * pyom.maskT

    if pyom.enable_conserve_energy:
        fxa = pyom.int_drhodT[-3, -3, -1, pyom.tau] # NOTE: probably a mistake in the fortran code
        pyom.P_diss_hmix[...] = 0.
        dissipation_on_wgrid(pyom.P_diss_hmix, pyom, int_drhodX=pyom.int_drhodT[..., pyom.tau])

    pyom.flux_east[:-1, :, :] = -fxa * (pyom.salt[1:, :, :, pyom.tau] - pyom.salt[:-1, :, :, pyom.tau]) \
                                    / (pyom.cost[None, :, None] * pyom.dxu[:-1, None, None]) * pyom.maskU[:-1, :, :]
    pyom.flux_north[:, :-1, :] = -fxa * (pyom.salt[:, 1:, :, pyom.tau] - pyom.salt[:, :-1, :, pyom.tau]) \
                                  / pyom.dyu[None, :-1, None] * pyom.maskV[:, :-1, :] * pyom.cosu[None, :-1, None]
    pyom.flux_east[-1,:,:] = 0.
    pyom.flux_north[:,-1,:] = 0.

    del2[1:, 1:, :] = pyom.maskT[1:, 1:, :] * (pyom.flux_east[1:, 1:, :] - pyom.flux_east[:-1, 1:, :]) \
                        / (pyom.cost[None, 1:, None] * pyom.dxt[1:, None, None]) \
                                            + (pyom.flux_north[1:, 1:, :] - pyom.flux_north[1:, :-1, :]) \
                        / (pyom.cost[None, 1:, None] * pyom.dyt[None, 1:, None])
    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(del2)

    pyom.flux_east[:-1, :, :] = fxa * (del2[1:, :, :] - del2[:-1, :, :]) / (pyom.cost[None, :, None] * pyom.dxu[:-1, None, None]) \
                                * pyom.maskU[:-1, :, :]
    pyom.flux_north[:, :-1, :] = fxa * (del2[:, 1:, :] - del2[:, :-1, :]) / pyom.dyu[None, :-1, None] \
                                * pyom.maskV[:, :-1, :] * pyom.cosu[None, :-1, None]
    pyom.flux_east[-1,:,:] = 0.
    pyom.flux_north[:,-1,:] = 0.

    # update tendency
    pyom.dsalt_hmix[1:, 1:, :] = pyom.maskT[1:, 1:, :] * (pyom.flux_east[1:, 1:, :] - pyom.flux_east[:-1, 1:, :]) \
                                 / (pyom.cost[None, 1:, None] * pyom.dxt[1:, None, None]) \
                                                       + (pyom.flux_north[1:, 1:, :] - pyom.flux_north[1:, :-1, :]) \
                                 / (pyom.cost[None, 1:, None] * pyom.dyt[None, 1:, None])
    pyom.salt[:,:,:,pyom.taup1] += pyom.dt_tracer * pyom.dsalt_hmix * pyom.maskT

    if pyom.enable_conserve_energy:
        dissipation_on_wgrid(pyom.P_diss_hmix, pyom, int_drhodX=pyom.int_drhodS[..., pyom.tau])


def tempsalt_diffusion(pyom):
    """
    Diffusion of pyom.temp and salinity,
    dissipation of dyn. Enthalpy is stored
    """
    aloc = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    # horizontal diffusion of temperature
    pyom.flux_east[:-1, :, :] = pyom.K_h * (pyom.temp[1:, :, :, pyom.tau] - pyom.temp[:-1, :, :, pyom.tau]) \
                                / (pyom.cost[None, :, None] * pyom.dxu[:-1, None, None]) * pyom.maskU[:-1, :, :]
    pyom.flux_east[-1,:,:] = 0.

    pyom.flux_north[:, :-1, :] = pyom.K_h * (pyom.temp[:, 1:, :, pyom.tau] - pyom.temp[:, :-1, :, pyom.tau]) \
                                 / pyom.dyu[None, :-1, None] * pyom.maskV[:, :-1, :] * pyom.cosu[None, :-1, None]
    pyom.flux_north[:,-1,:] = 0.

    if pyom.enable_hor_friction_cos_scaling:
        pyom.flux_east[...] *= pyom.cost[None, :, None] ** pyom.hor_friction_cosPower
        pyom.flux_north[...] *= pyom.cosu[None, :, None] ** pyom.hor_friction_cosPower

    pyom.dtemp_hmix[1:, 1:, :] = pyom.maskT[1:, 1:, :] * ((pyom.flux_east[1:, 1:, :] - pyom.flux_east[:-1, 1:, :]) \
                                                          / (pyom.cost[None, 1:, None] * pyom.dxt[1:, None, None]) \
                                                        + (pyom.flux_north[1:, 1:, :] - pyom.flux_north[1:, :-1, :]) \
                                                          / (pyom.cost[None, 1:, None] * pyom.dyt[None, 1:, None]))
    pyom.temp[:,:,:,pyom.taup1] += pyom.dt_tracer * pyom.dtemp_hmix * pyom.maskT

    if pyom.enable_conserve_energy:
        fxa = pyom.int_drhodT[-3, -3, -1, pyom.tau] # NOTE: probably a mistake in the fortran code
        pyom.P_diss_hmix[...] = 0.
        dissipation_on_wgrid(pyom.P_diss_hmix, pyom, int_drhodX=pyom.int_drhodT[..., pyom.tau])

    # horizontal diffusion of salinity
    pyom.flux_east[:-1, :, :] = pyom.K_h * (pyom.salt[1:, :, :, pyom.tau] - pyom.salt[:-1, :, :, pyom.tau]) \
                                / (pyom.cost[None, :, None] * pyom.dxu[:-1, None, None]) * pyom.maskU[:-1, :, :]
    pyom.flux_east[-1,:,:] = 0.

    pyom.flux_north[:, :-1, :] = pyom.K_h * (pyom.salt[:, 1:, :, pyom.tau] - pyom.salt[:, :-1, :, pyom.tau]) \
                                 / pyom.dyu[None, :-1, None] * pyom.maskV[:, :-1, :] * pyom.cosu[None, :-1, None]
    pyom.flux_north[:,-1,:] = 0.

    if pyom.enable_hor_friction_cos_scaling:
        pyom.flux_east[...] *= pyom.cost[None, :, None] ** pyom.hor_friction_cosPower
        pyom.flux_north[...] *= pyom.cosu[None, :, None] ** pyom.hor_friction_cosPower

    pyom.dsalt_hmix[1:, 1:, :] = pyom.maskT[1:, 1:, :] * ((pyom.flux_east[1:, 1:, :] - pyom.flux_east[:-1, 1:, :]) \
                                                            / (pyom.cost[None, 1:, None] * pyom.dxt[1:, None, None]) \
                                                       + (pyom.flux_north[1:, 1:, :] - pyom.flux_north[1:, :-1, :]) \
                                                            / (pyom.cost[None, 1:, None] * pyom.dyt[None, 1:, None]))
    pyom.salt[:,:,:,pyom.taup1] = pyom.salt[:,:,:,pyom.taup1] + pyom.dt_tracer * pyom.dsalt_hmix * pyom.maskT

    if pyom.enable_conserve_energy:
        dissipation_on_wgrid(pyom.P_diss_hmix, pyom, int_drhodX=pyom.int_drhodS[..., pyom.tau])


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
        dissipation_on_wgrid(pyom.P_diss_sources, pyom, aloc=aloc)

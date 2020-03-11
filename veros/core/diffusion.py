import math

from veros.core.operators import numpy as np

from veros import veros_kernel
from veros.core import utilities
from veros.core.operators import update, update_add, update_multiply, at


@veros_kernel(static_args=('nz',))
def dissipation_on_wgrid(out, nz, dzw,
                         grav, rho_0, flux_east,
                         flux_north, dxt, dyt, cost, kbot,
                         aloc=None, int_drhodX=None, ks=None):
    if aloc is None:
        aloc = np.zeros_like(out)
        aloc = update(aloc, at[1:-1, 1:-1, :], 0.5 * grav / rho_0 \
            * ((int_drhodX[2:, 1:-1, :] - int_drhodX[1:-1, 1:-1, :]) * flux_east[1:-1, 1:-1, :]
               + (int_drhodX[1:-1, 1:-1, :] - int_drhodX[:-2, 1:-1, :]) * flux_east[:-2, 1:-1, :]) \
            / (dxt[1:-1, np.newaxis, np.newaxis] * cost[np.newaxis, 1:-1, np.newaxis]) \
            + 0.5 * grav / rho_0 * ((int_drhodX[1:-1, 2:, :] - int_drhodX[1:-1, 1:-1, :]) * flux_north[1:-1, 1:-1, :]
                                    + (int_drhodX[1:-1, 1:-1, :] - int_drhodX[1:-1, :-2, :]) * flux_north[1:-1, :-2, :]) \
            / (dyt[np.newaxis, 1:-1, np.newaxis] * cost[np.newaxis, 1:-1, np.newaxis]))

    if ks is None:
        ks = kbot[:, :] - 1

    land_mask = ks >= 0
    edge_mask = land_mask[:, :, np.newaxis] & (
        np.arange(nz - 1)[np.newaxis, np.newaxis, :] == ks[:, :, np.newaxis])
    water_mask = land_mask[:, :, np.newaxis] & (
        np.arange(nz - 1)[np.newaxis, np.newaxis, :] > ks[:, :, np.newaxis])

    dzw_pad = utilities.pad_z_edges(dzw)
    out = update_add(out, at[:, :, :-1], (0.5 * (aloc[:, :, :-1] + aloc[:, :, 1:])
                       + 0.5 * (aloc[:, :, :-1] * dzw_pad[np.newaxis, np.newaxis, :-3]
                                / dzw[np.newaxis, np.newaxis, :-1])) * edge_mask)
    out = update_add(out, at[:, :, :-1], 0.5 * (aloc[:, :, :-1] + aloc[:, :, 1:]) * water_mask)
    out = update_add(out, at[:, :, -1], aloc[:, :, -1] * land_mask)

    return out


@veros_kernel(static_args=('enable_conserve_energy', 'pyom_compatibility_mode',))
def tempsalt_biharmonic(K_hbi, temp, salt, int_drhodT, int_drhodS, maskT, maskU, maskV, maskW,
                        dxt, dxu, dyt, dyu, dzw, nz, tau, cost, cosu, taup1, dt_tracer,
                        enable_cyclic_x, enable_conserve_energy, pyom_compatibility_mode):
    """
    biharmonic mixing of temp and salinity,
    dissipation of dyn. Enthalpy is stored
    """
    fxa = math.sqrt(abs(K_hbi))

    # update temp
    dtemp_hmix = np.zeros_like(maskT)
    dtemp_hmix = update(dtemp_hmix, at[1:, 1:, :], biharmonic_diffusion(temp[:, :, :, tau], fxa,
                                                 dxt, dxu, dyt, dyu, maskT,
                                                 maskU, maskV, cost, cosu,
                                                 enable_cyclic_x)[1:, 1:, :])
    temp = update_add(temp, at[:, :, :, taup1], dt_tracer * dtemp_hmix * maskT)

    P_diss_hmix = np.zeros_like(maskW)
    if enable_conserve_energy:
        if pyom_compatibility_mode:
            fxa = int_drhodT[-3, -3, -1, tau]
        P_diss_hmix = dissipation_on_wgrid(P_diss_hmix, nz, dzw, int_drhodX=int_drhodT[..., tau])

    # update salt
    dsalt_hmix = np.zeros_like(maskT)
    dsalt_hmix = update(dsalt_hmix, at[1:, 1:, :], biharmonic_diffusion(salt[:, :, :, tau], fxa,
                                                 dxt, dxu, dyt, dyu, maskT,
                                                 maskU, maskV, cost, cosu,
                                                 enable_cyclic_x)[1:, 1:, :])
    salt = update_add(salt, at[:, :, :, taup1], dt_tracer * dsalt_hmix * maskT)

    if enable_conserve_energy:
        P_diss_hmix = dissipation_on_wgrid(P_diss_hmix, nz, dzw, int_drhodX=int_drhodS[..., tau])

    return temp, salt, dtemp_hmix, dsalt_hmix, P_diss_hmix


@veros_kernel(static_args=('enable_conserve_energy',))
def tempsalt_diffusion(int_drhodT, int_drhodS, temp, salt, maskT, maskU, maskV, maskW,
                       dxt, dxu, dyt, dyu, nz, dzw, tau, taup1, dt_tracer, K_h, cost, cosu,
                       hor_friction_cosPower, enable_hor_friction_cos_scaling,
                       enable_conserve_energy):
    """
    Diffusion of temp and salinity,
    dissipation of dyn. Enthalpy is stored
    """
    # horizontal diffusion of temperature
    dtemp_hmix = np.zeros_like(maskT)
    dtemp_hmix = update(dtemp_hmix, at[1:, 1:, :], horizontal_diffusion(temp[:, :, :, tau], K_h,
                                                 dxt, dxu, dyt, dyu, maskT,
                                                 maskU, maskV, cost, cosu,
                                                 hor_friction_cosPower,
                                                 enable_hor_friction_cos_scaling)[1:, 1:, :])
    temp = update_add(temp, at[:, :, :, taup1], dt_tracer * dtemp_hmix * maskT)

    P_diss_hmix = np.zeros_like(maskW)
    if enable_conserve_energy:
        P_diss_hmix = dissipation_on_wgrid(P_diss_hmix, nz, dzw, int_drhodX=int_drhodT[..., tau])

    # horizontal diffusion of salinity
    dsalt_hmix = np.zeros_like(maskT)
    dsalt_hmix = update(dsalt_hmix, at[1:, 1:, :], horizontal_diffusion(salt[:, :, :, tau], K_h,
                                                 dxt, dxu, dyt, dyu, maskT,
                                                 maskU, maskV, cost, cosu,
                                                 hor_friction_cosPower,
                                                 enable_hor_friction_cos_scaling)[1:, 1:, :])
    salt = update_add(salt, at[:, :, :, taup1], dt_tracer * dsalt_hmix * maskT)

    if enable_conserve_energy:
        P_diss_hmix = dissipation_on_wgrid(P_diss_hmix, nz, dzw, int_drhodX=int_drhodS[..., tau])

    return temp, salt, dtemp_hmix, dsalt_hmix, P_diss_hmix


@veros_kernel(static_args=('enable_conserve_energy',))
def tempsalt_sources(temp, salt, temp_source, salt_source, maskT, maskW,
                     tau, taup1, dt_tracer, grav, rho_0, nz, dzw,
                     int_drhodT, int_drhodS, enable_conserve_energy):
    """
    Sources of temp and salinity,
    effect on dyn. Enthalpy is stored
    """
    temp = update_add(temp, at[:, :, :, taup1], dt_tracer * temp_source * maskT)
    salt = update_add(salt, at[:, :, :, taup1], dt_tracer * salt_source * maskT)

    P_diss_sources = np.zeros_like(maskW)
    if enable_conserve_energy:
        aloc = -grav / rho_0 * maskT * \
            (int_drhodT[..., tau] * temp_source +
             int_drhodS[..., tau] * salt_source)
        P_diss_sources = dissipation_on_wgrid(P_diss_sources, nz, dzw, aloc=aloc)

    return temp, salt, P_diss_sources


@veros_kernel
def biharmonic_diffusion(tr, diffusivity, dxt, dxu, dyt, dyu, maskT, maskU, maskV,
                         cost, cosu, enable_cyclic_x):
    """
    Biharmonic mixing of tracer tr
    """
    del2 = np.zeros_like(maskT)
    dtr = np.zeros_like(maskT)
    flux_east = np.zeros_like(maskU)
    flux_north = np.zeros_like(maskV)

    flux_east = update(flux_east, at[:-1, :, :], -diffusivity * (tr[1:, :, :] - tr[:-1, :, :]) \
        / (cost[np.newaxis, :, np.newaxis] * dxu[:-1, np.newaxis, np.newaxis]) \
        * maskU[:-1, :, :])

    flux_north = update(flux_north, at[:, :-1, :], -diffusivity * (tr[:, 1:, :] - tr[:, :-1, :]) \
        / dyu[np.newaxis, :-1, np.newaxis] * maskV[:, :-1, :] \
        * cosu[np.newaxis, :-1, np.newaxis])

    del2 = update(del2, at[1:, 1:, :], maskT[1:, 1:, :] * (flux_east[1:, 1:, :] - flux_east[:-1, 1:, :]) \
        / (cost[np.newaxis, 1:, np.newaxis] * dxt[1:, np.newaxis, np.newaxis]) \
        + (flux_north[1:, 1:, :] - flux_north[1:, :-1, :]) \
        / (cost[np.newaxis, 1:, np.newaxis] * dyt[np.newaxis, 1:, np.newaxis]))

    del2 = utilities.enforce_boundaries(del2, enable_cyclic_x)

    flux_east = update(flux_east, at[:-1, :, :], diffusivity * (del2[1:, :, :] - del2[:-1, :, :]) \
        / (cost[np.newaxis, :, np.newaxis] * dxu[:-1, np.newaxis, np.newaxis]) \
        * maskU[:-1, :, :])
    flux_north = update(flux_north, at[:, :-1, :], diffusivity * (del2[:, 1:, :] - del2[:, :-1, :]) \
        / dyu[np.newaxis, :-1, np.newaxis] * maskV[:, :-1, :] \
        * cosu[np.newaxis, :-1, np.newaxis])

    flux_east = update(flux_east, at[-1, :, :], 0.)
    flux_north = update(flux_north, at[:, -1, :], 0.)

    dtr = update(dtr, at[1:, 1:, :], (flux_east[1:, 1:, :] - flux_east[:-1, 1:, :]) \
        / (cost[np.newaxis, 1:, np.newaxis] * dxt[1:, np.newaxis, np.newaxis]) \
        + (flux_north[1:, 1:, :] - flux_north[1:, :-1, :]) \
        / (cost[np.newaxis, 1:, np.newaxis] * dyt[np.newaxis, 1:, np.newaxis]))

    dtr = update_multiply(dtr, at[...], maskT)

    return dtr


@veros_kernel(static_args=('enable_hor_friction_cos_scaling'))
def horizontal_diffusion(tr, diffusivity, dxt, dxu, dyt, dyu, maskT, maskU, maskV, cost,
                         cosu, hor_friction_cosPower, enable_hor_friction_cos_scaling):
    """
    Diffusion of tracer tr
    """
    dtr_hmix = np.zeros_like(maskT)
    flux_east = np.zeros_like(maskU)
    flux_north = np.zeros_like(maskV)

    # horizontal diffusion of tracer
    flux_east = update(flux_east, at[:-1, :, :], diffusivity * (tr[1:, :, :] - tr[:-1, :, :]) \
        / (cost[np.newaxis, :, np.newaxis] * dxu[:-1, np.newaxis, np.newaxis])\
        * maskU[:-1, :, :])
    flux_east = update(flux_east, at[-1, :, :], 0.)

    flux_north = update(flux_north, at[:, :-1, :], diffusivity * (tr[:, 1:, :] - tr[:, :-1, :]) \
        / dyu[np.newaxis, :-1, np.newaxis] * maskV[:, :-1, :]\
        * cosu[np.newaxis, :-1, np.newaxis])
    flux_north = update(flux_north, at[:, -1, :], 0.)

    if enable_hor_friction_cos_scaling:
        flux_east = update_multiply(flux_east, at[...], cost[np.newaxis, :, np.newaxis] ** hor_friction_cosPower)
        flux_north = update_multiply(flux_north, at[...], cosu[np.newaxis, :, np.newaxis] ** hor_friction_cosPower)

    dtr_hmix = update(dtr_hmix, at[1:, 1:, :], ((flux_east[1:, 1:, :] - flux_east[:-1, 1:, :])
                           / (cost[np.newaxis, 1:, np.newaxis] * dxt[1:, np.newaxis, np.newaxis])
                           + (flux_north[1:, 1:, :] - flux_north[1:, :-1, :])
                           / (cost[np.newaxis, 1:, np.newaxis] * dyt[np.newaxis, 1:, np.newaxis]))\
                        * maskT[1:, 1:, :])

    return dtr_hmix

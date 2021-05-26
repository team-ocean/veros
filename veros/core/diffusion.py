from veros.core.operators import numpy as npx

from veros import veros_kernel, KernelOutput, runtime_settings
from veros.variables import allocate
from veros.core import utilities
from veros.core.operators import update, update_add, update_multiply, at


@veros_kernel
def compute_dissipation(state, int_drhodX, flux_east, flux_north):
    vs = state.variables
    settings = state.settings

    diss = allocate(state.dimensions, ("xt", "yt", "zt"))
    diss = update(
        diss,
        at[1:-1, 1:-1, :],
        0.5
        * settings.grav
        / settings.rho_0
        * (
            (int_drhodX[2:, 1:-1, :] - int_drhodX[1:-1, 1:-1, :]) * flux_east[1:-1, 1:-1, :]
            + (int_drhodX[1:-1, 1:-1, :] - int_drhodX[:-2, 1:-1, :]) * flux_east[:-2, 1:-1, :]
        )
        / (vs.dxt[1:-1, npx.newaxis, npx.newaxis] * vs.cost[npx.newaxis, 1:-1, npx.newaxis])
        + 0.5
        * settings.grav
        / settings.rho_0
        * (
            (int_drhodX[1:-1, 2:, :] - int_drhodX[1:-1, 1:-1, :]) * flux_north[1:-1, 1:-1, :]
            + (int_drhodX[1:-1, 1:-1, :] - int_drhodX[1:-1, :-2, :]) * flux_north[1:-1, :-2, :]
        )
        / (vs.dyt[npx.newaxis, 1:-1, npx.newaxis] * vs.cost[npx.newaxis, 1:-1, npx.newaxis]),
    )

    return diss


@veros_kernel
def dissipation_on_wgrid(state, diss, ks):
    vs = state.variables
    settings = state.settings

    land_mask, water_mask, edge_mask = utilities.create_water_masks(ks, settings.nz)
    water_mask = npx.logical_and(water_mask, npx.logical_not(edge_mask))

    dzw_pad = utilities.pad_z_edges(vs.dzw)

    diss_w = allocate(state.dimensions, ("xt", "yt", "zt"))
    diss_w = update(
        diss_w,
        at[:, :, :-1],
        (
            0.5 * (diss[:, :, :-1] + diss[:, :, 1:])
            + 0.5 * (diss[:, :, :-1] * dzw_pad[npx.newaxis, npx.newaxis, :-3] / vs.dzw[npx.newaxis, npx.newaxis, :-1])
        )
        * edge_mask[:, :, :-1]
        + 0.5 * (diss[:, :, :-1] + diss[:, :, 1:]) * water_mask[:, :, :-1],
    )
    diss_w = update(diss_w, at[:, :, -1], diss[:, :, -1] * land_mask)

    return diss_w


@veros_kernel
def tempsalt_biharmonic(state):
    """
    biharmonic mixing of temp and salinity,
    dissipation of dyn. Enthalpy is stored
    """
    vs = state.variables
    settings = state.settings

    fxa = npx.sqrt(abs(settings.K_hbi))

    # update temp
    dtemp, flux_east, flux_north = biharmonic_diffusion(state, vs.temp[:, :, :, vs.tau], fxa)
    vs.dtemp_hmix = update(vs.dtemp_hmix, at[1:, 1:, :], dtemp[1:, 1:, :])
    vs.temp = update_add(vs.temp, at[:, :, :, vs.taup1], settings.dt_tracer * vs.dtemp_hmix * vs.maskT)

    vs.P_diss_hmix = allocate(state.dimensions, ("xt", "yt", "zt"))
    if settings.enable_conserve_energy:
        if runtime_settings.pyom_compatibility_mode:
            fxa = vs.int_drhodT[-3, -3, -1, vs.tau]

        diss = compute_dissipation(state, vs.int_drhodT[..., vs.tau], flux_east, flux_north)
        vs.P_diss_hmix = vs.P_diss_hmix + dissipation_on_wgrid(state, diss, vs.kbot)

    # update salt
    dsalt, flux_east, flux_north = biharmonic_diffusion(state, vs.salt[:, :, :, vs.tau], fxa)
    vs.dsalt_hmix = update(vs.dsalt_hmix, at[1:, 1:, :], dsalt[1:, 1:, :])
    vs.salt = update_add(vs.salt, at[:, :, :, vs.taup1], settings.dt_tracer * vs.dsalt_hmix * vs.maskT)

    if settings.enable_conserve_energy:
        diss = compute_dissipation(state, vs.int_drhodS[..., vs.tau], flux_east, flux_north)
        vs.P_diss_hmix = vs.P_diss_hmix + dissipation_on_wgrid(state, diss, vs.kbot)

    return KernelOutput(
        temp=vs.temp, salt=vs.salt, dtemp_hmix=vs.dtemp_hmix, dsalt_hmix=vs.dsalt_hmix, P_diss_hmix=vs.P_diss_hmix
    )


@veros_kernel
def tempsalt_diffusion(state):
    """
    Diffusion of temp and salinity,
    dissipation of dyn. Enthalpy is stored
    """
    vs = state.variables
    settings = state.settings

    # horizontal diffusion of temperature
    dtemp, flux_east, flux_north = horizontal_diffusion(state, vs.temp[:, :, :, vs.tau], settings.K_h)
    vs.dtemp_hmix = update(vs.dtemp_hmix, at[1:, 1:, :], dtemp[1:, 1:, :])
    vs.temp = update_add(vs.temp, at[:, :, :, vs.taup1], settings.dt_tracer * vs.dtemp_hmix * vs.maskT)

    if settings.enable_conserve_energy:
        diss = compute_dissipation(state, vs.int_drhodT[..., vs.tau], flux_east, flux_north)
        vs.P_diss_hmix = dissipation_on_wgrid(state, diss, vs.kbot)

    # horizontal diffusion of salinity
    dsalt, flux_east, flux_north = horizontal_diffusion(state, vs.salt[:, :, :, vs.tau], settings.K_h)
    vs.dsalt_hmix = update(vs.dsalt_hmix, at[1:, 1:, :], dsalt[1:, 1:, :])
    vs.salt = update_add(vs.salt, at[:, :, :, vs.taup1], settings.dt_tracer * vs.dsalt_hmix * vs.maskT)

    if settings.enable_conserve_energy:
        diss = compute_dissipation(state, vs.int_drhodS[..., vs.tau], flux_east, flux_north)
        vs.P_diss_hmix = vs.P_diss_hmix + dissipation_on_wgrid(state, diss, vs.kbot)

    return KernelOutput(
        temp=vs.temp, salt=vs.salt, dtemp_hmix=vs.dtemp_hmix, dsalt_hmix=vs.dsalt_hmix, P_diss_hmix=vs.P_diss_hmix
    )


@veros_kernel
def tempsalt_sources(state):
    """
    Sources of temp and salinity,
    effect on dyn. Enthalpy is stored
    """
    vs = state.variables
    settings = state.settings

    vs.temp = update_add(vs.temp, at[:, :, :, vs.taup1], settings.dt_tracer * vs.temp_source * vs.maskT)
    vs.salt = update_add(vs.salt, at[:, :, :, vs.taup1], settings.dt_tracer * vs.salt_source * vs.maskT)

    if settings.enable_conserve_energy:
        diss = (
            -settings.grav
            / settings.rho_0
            * vs.maskT
            * (vs.int_drhodT[..., vs.tau] * vs.temp_source + vs.int_drhodS[..., vs.tau] * vs.salt_source)
        )

        vs.P_diss_sources = dissipation_on_wgrid(state, diss, vs.kbot)

    return KernelOutput(temp=vs.temp, salt=vs.salt, P_diss_sources=vs.P_diss_sources)


@veros_kernel
def biharmonic_diffusion(state, tr, diffusivity):
    """
    Biharmonic mixing of tracer tr
    """
    vs = state.variables
    settings = state.settings

    del2 = allocate(state.dimensions, ("xt", "yt", "zt"))
    dtr = allocate(state.dimensions, ("xt", "yt", "zt"))

    flux_east = allocate(state.dimensions, ("xt", "yt", "zt"))
    flux_north = allocate(state.dimensions, ("xt", "yt", "zt"))

    flux_east = update(
        flux_east,
        at[:-1, :, :],
        -diffusivity
        * (tr[1:, :, :] - tr[:-1, :, :])
        / (vs.cost[npx.newaxis, :, npx.newaxis] * vs.dxu[:-1, npx.newaxis, npx.newaxis])
        * vs.maskU[:-1, :, :],
    )

    flux_north = update(
        flux_north,
        at[:, :-1, :],
        -diffusivity
        * (tr[:, 1:, :] - tr[:, :-1, :])
        / vs.dyu[npx.newaxis, :-1, npx.newaxis]
        * vs.maskV[:, :-1, :]
        * vs.cosu[npx.newaxis, :-1, npx.newaxis],
    )

    del2 = update(
        del2,
        at[1:, 1:, :],
        vs.maskT[1:, 1:, :]
        * (flux_east[1:, 1:, :] - flux_east[:-1, 1:, :])
        / (vs.cost[npx.newaxis, 1:, npx.newaxis] * vs.dxt[1:, npx.newaxis, npx.newaxis])
        + (flux_north[1:, 1:, :] - flux_north[1:, :-1, :])
        / (vs.cost[npx.newaxis, 1:, npx.newaxis] * vs.dyt[npx.newaxis, 1:, npx.newaxis]),
    )

    del2 = utilities.enforce_boundaries(del2, settings.enable_cyclic_x)

    flux_east = update(
        flux_east,
        at[:-1, :, :],
        diffusivity
        * (del2[1:, :, :] - del2[:-1, :, :])
        / (vs.cost[npx.newaxis, :, npx.newaxis] * vs.dxu[:-1, npx.newaxis, npx.newaxis])
        * vs.maskU[:-1, :, :],
    )
    flux_north = update(
        flux_north,
        at[:, :-1, :],
        diffusivity
        * (del2[:, 1:, :] - del2[:, :-1, :])
        / vs.dyu[npx.newaxis, :-1, npx.newaxis]
        * vs.maskV[:, :-1, :]
        * vs.cosu[npx.newaxis, :-1, npx.newaxis],
    )

    flux_east = update(flux_east, at[-1, :, :], 0.0)
    flux_north = update(flux_north, at[:, -1, :], 0.0)

    dtr = update(
        dtr,
        at[1:, 1:, :],
        (flux_east[1:, 1:, :] - flux_east[:-1, 1:, :])
        / (vs.cost[npx.newaxis, 1:, npx.newaxis] * vs.dxt[1:, npx.newaxis, npx.newaxis])
        + (flux_north[1:, 1:, :] - flux_north[1:, :-1, :])
        / (vs.cost[npx.newaxis, 1:, npx.newaxis] * vs.dyt[npx.newaxis, 1:, npx.newaxis]),
    )

    dtr = dtr * vs.maskT

    return dtr, flux_east, flux_north


@veros_kernel
def horizontal_diffusion(state, tr, diffusivity):
    """
    Diffusion of tracer tr
    """
    vs = state.variables
    settings = state.settings

    dtr_hmix = allocate(state.dimensions, ("xt", "yt", "zt"))

    flux_east = allocate(state.dimensions, ("xt", "yt", "zt"))
    flux_north = allocate(state.dimensions, ("xt", "yt", "zt"))

    # horizontal diffusion of tracer
    flux_east = update(
        flux_east,
        at[:-1, :, :],
        diffusivity
        * (tr[1:, :, :] - tr[:-1, :, :])
        / (vs.cost[npx.newaxis, :, npx.newaxis] * vs.dxu[:-1, npx.newaxis, npx.newaxis])
        * vs.maskU[:-1, :, :],
    )
    flux_east = update(flux_east, at[-1, :, :], 0.0)

    flux_north = update(
        flux_north,
        at[:, :-1, :],
        diffusivity
        * (tr[:, 1:, :] - tr[:, :-1, :])
        / vs.dyu[npx.newaxis, :-1, npx.newaxis]
        * vs.maskV[:, :-1, :]
        * vs.cosu[npx.newaxis, :-1, npx.newaxis],
    )
    flux_north = update(flux_north, at[:, -1, :], 0.0)

    if settings.enable_hor_friction_cos_scaling:
        flux_east = update_multiply(
            flux_east, at[...], vs.cost[npx.newaxis, :, npx.newaxis] ** settings.hor_friction_cosPower
        )
        flux_north = update_multiply(
            flux_north, at[...], vs.cosu[npx.newaxis, :, npx.newaxis] ** settings.hor_friction_cosPower
        )

    dtr_hmix = update(
        dtr_hmix,
        at[1:, 1:, :],
        (
            (flux_east[1:, 1:, :] - flux_east[:-1, 1:, :])
            / (vs.cost[npx.newaxis, 1:, npx.newaxis] * vs.dxt[1:, npx.newaxis, npx.newaxis])
            + (flux_north[1:, 1:, :] - flux_north[1:, :-1, :])
            / (vs.cost[npx.newaxis, 1:, npx.newaxis] * vs.dyt[npx.newaxis, 1:, npx.newaxis])
        )
        * vs.maskT[1:, 1:, :],
    )

    return dtr_hmix, flux_east, flux_north

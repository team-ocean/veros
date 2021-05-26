from veros.core.operators import numpy as npx

from veros import veros_kernel, veros_routine, KernelOutput
from veros.variables import allocate
from veros.core import utilities, advection
from veros.core.operators import update, update_add, at


@veros_routine
def set_eke_diffusivities(state):
    vs = state.variables
    settings = state.settings

    eke_diff_out = set_eke_diffusivities_kernel(state)
    vs.update(eke_diff_out)

    if settings.enable_TEM_friction:
        kappa_gm_out = update_kappa_gm(state)
        vs.update(kappa_gm_out)


@veros_kernel
def update_kappa_gm(state):
    vs = state.variables

    kappa_gm = (
        vs.K_gm
        * npx.minimum(0.01, vs.coriolis_t[..., npx.newaxis] ** 2 / npx.maximum(1e-9, vs.Nsqr[..., vs.tau]))
        * vs.maskW
    )
    return KernelOutput(kappa_gm=kappa_gm)


@veros_kernel
def set_eke_diffusivities_kernel(state):
    """
    set skew diffusivity K_gm and isopycnal diffusivity K_iso
    set also vertical viscosity if TEM formalism is chosen
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_eke:
        """
        calculate Rossby radius as minimum of mid-latitude and equatorial R. rad.
        """
        C_rossby = npx.sum(
            npx.sqrt(npx.maximum(0.0, vs.Nsqr[:, :, :, vs.tau]))
            * vs.dzw[npx.newaxis, npx.newaxis, :]
            * vs.maskW[:, :, :]
            / settings.pi,
            axis=2,
        )
        vs.L_rossby = npx.minimum(
            C_rossby / npx.maximum(npx.abs(vs.coriolis_t), 1e-16), npx.sqrt(C_rossby / npx.maximum(2 * vs.beta, 1e-16))
        )

        """
        calculate vertical viscosity and skew diffusivity
        """
        vs.sqrteke = npx.sqrt(npx.maximum(0.0, vs.eke[:, :, :, vs.tau]))
        vs.L_rhines = npx.sqrt(vs.sqrteke / npx.maximum(vs.beta[..., npx.newaxis], 1e-16))
        vs.eke_len = npx.maximum(
            settings.eke_lmin,
            npx.minimum(settings.eke_cross * vs.L_rossby[..., npx.newaxis], settings.eke_crhin * vs.L_rhines),
        )
        vs.K_gm = npx.minimum(settings.eke_k_max, settings.eke_c_k * vs.eke_len * vs.sqrteke)
    else:
        """
        use fixed GM diffusivity
        """
        vs.K_gm = update(vs.K_gm, at[...], settings.K_gm_0)

    if settings.enable_eke and settings.enable_eke_isopycnal_diffusion:
        vs.K_iso = update(vs.K_iso, at[...], vs.K_gm)
    else:
        vs.K_iso = update(vs.K_iso, at[...], settings.K_iso_0)  # always constant

    if not settings.enable_eke:
        return KernelOutput(K_gm=vs.K_gm, K_iso=vs.K_iso)

    return KernelOutput(
        L_rossby=vs.L_rossby, L_rhines=vs.L_rhines, eke_len=vs.eke_len, sqrteke=vs.sqrteke, K_gm=vs.K_gm, K_iso=vs.K_iso
    )


@veros_routine
def integrate_eke(state):
    vs = state.variables
    vs.update(integrate_eke_kernel(state))


@veros_kernel
def integrate_eke_kernel(state):
    """
    integrate EKE equation on W grid
    """
    vs = state.variables
    settings = state.settings

    c_int = allocate(state.dimensions, ("xt", "yt", "zt"))

    flux_east = allocate(state.dimensions, ("xt", "yt", "zt"))
    flux_north = allocate(state.dimensions, ("xt", "yt", "zt"))
    flux_top = allocate(state.dimensions, ("xt", "yt", "zt"))

    """
    forcing by dissipation by lateral friction and GM using TRM formalism or skew diffusion
    """
    forc = vs.K_diss_h - vs.P_diss_skew

    if settings.enable_TEM_friction:
        forc = forc + vs.K_diss_gm

    """
    store transfer due to isopycnal and horizontal mixing from dyn. enthalpy
    by non-linear eq.of state either to EKE or to heat
    """
    if not settings.enable_store_cabbeling_heat:
        forc = forc - vs.P_diss_hmix - vs.P_diss_iso

    conditional_outputs = {}

    """
    dissipation by local interior loss of balance with constant coefficient
    """
    c_int = settings.eke_c_eps * vs.sqrteke / vs.eke_len * vs.maskW

    """
    vertical diffusion of EKE,forcing and dissipation
    """
    _, water_mask, edge_mask = utilities.create_water_masks(vs.kbot[2:-2, 2:-2], settings.nz)

    delta, a_tri, b_tri, c_tri, d_tri = (
        allocate(state.dimensions, ("xt", "yt", "zt"))[2:-2, 2:-2, :] for _ in range(5)
    )
    delta = update(
        delta,
        at[:, :, :-1],
        settings.dt_tracer
        / vs.dzt[npx.newaxis, npx.newaxis, 1:]
        * 0.5
        * (vs.kappaM[2:-2, 2:-2, :-1] + vs.kappaM[2:-2, 2:-2, 1:])
        * settings.alpha_eke,
    )
    a_tri = update(a_tri, at[:, :, 1:-1], -delta[:, :, :-2] / vs.dzw[1:-1])
    a_tri = update(a_tri, at[:, :, -1], -delta[:, :, -2] / (0.5 * vs.dzw[-1]))
    b_tri = update(
        b_tri,
        at[:, :, 1:-1],
        1 + (delta[:, :, 1:-1] + delta[:, :, :-2]) / vs.dzw[1:-1] + settings.dt_tracer * c_int[2:-2, 2:-2, 1:-1],
    )
    b_tri = update(
        b_tri, at[:, :, -1], 1 + delta[:, :, -2] / (0.5 * vs.dzw[-1]) + settings.dt_tracer * c_int[2:-2, 2:-2, -1]
    )
    b_tri_edge = 1 + delta / vs.dzw[npx.newaxis, npx.newaxis, :] + settings.dt_tracer * c_int[2:-2, 2:-2, :]
    c_tri = update(c_tri, at[:, :, :-1], -delta[:, :, :-1] / vs.dzw[npx.newaxis, npx.newaxis, :-1])
    d_tri = update(d_tri, at[:, :, :], vs.eke[2:-2, 2:-2, :, vs.tau] + settings.dt_tracer * forc[2:-2, 2:-2, :])

    sol = utilities.solve_implicit(a_tri, b_tri, c_tri, d_tri, water_mask, b_edge=b_tri_edge, edge_mask=edge_mask)
    vs.eke = update(vs.eke, at[2:-2, 2:-2, :, vs.taup1], npx.where(water_mask, sol, vs.eke[2:-2, 2:-2, :, vs.taup1]))

    """
    store eke dissipation
    """
    vs.eke_diss_iw = c_int * vs.eke[:, :, :, vs.taup1]
    vs.eke_diss_tke = update(vs.eke_diss_tke, at[...], 0.0)

    """
    add tendency due to lateral diffusion
    """
    flux_east = update(
        flux_east,
        at[:-1, :, :],
        0.5
        * npx.maximum(500.0, vs.K_gm[:-1, :, :] + vs.K_gm[1:, :, :])
        * (vs.eke[1:, :, :, vs.tau] - vs.eke[:-1, :, :, vs.tau])
        / (vs.cost[npx.newaxis, :, npx.newaxis] * vs.dxu[:-1, npx.newaxis, npx.newaxis])
        * vs.maskU[:-1, :, :],
    )
    flux_east = update(flux_east, at[-1, :, :], 0.0)
    flux_north = update(
        flux_north,
        at[:, :-1, :],
        0.5
        * npx.maximum(500.0, vs.K_gm[:, :-1, :] + vs.K_gm[:, 1:, :])
        * (vs.eke[:, 1:, :, vs.tau] - vs.eke[:, :-1, :, vs.tau])
        / vs.dyu[npx.newaxis, :-1, npx.newaxis]
        * vs.maskV[:, :-1, :]
        * vs.cosu[npx.newaxis, :-1, npx.newaxis],
    )
    flux_north = update(flux_north, at[:, -1, :], 0.0)
    vs.eke = update_add(
        vs.eke,
        at[2:-2, 2:-2, :, vs.taup1],
        settings.dt_tracer
        * vs.maskW[2:-2, 2:-2, :]
        * (
            (flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
            / (vs.cost[npx.newaxis, 2:-2, npx.newaxis] * vs.dxt[2:-2, npx.newaxis, npx.newaxis])
            + (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
            / (vs.cost[npx.newaxis, 2:-2, npx.newaxis] * vs.dyt[npx.newaxis, 2:-2, npx.newaxis])
        ),
    )

    """
    add tendency due to advection
    """
    if settings.enable_eke_superbee_advection:
        flux_east, flux_north, flux_top = advection.adv_flux_superbee_wgrid(state, vs.eke[:, :, :, vs.tau])

    if settings.enable_eke_upwind_advection:
        flux_east, flux_north, flux_top = advection.adv_flux_upwind_wgrid(state, vs.eke[:, :, :, vs.tau])

    if settings.enable_eke_superbee_advection or settings.enable_eke_upwind_advection:
        vs.deke = update(
            vs.deke,
            at[2:-2, 2:-2, :, vs.tau],
            vs.maskW[2:-2, 2:-2, :]
            * (
                -(flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
                / (vs.cost[npx.newaxis, 2:-2, npx.newaxis] * vs.dxt[2:-2, npx.newaxis, npx.newaxis])
                - (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
                / (vs.cost[npx.newaxis, 2:-2, npx.newaxis] * vs.dyt[npx.newaxis, 2:-2, npx.newaxis])
            ),
        )
        vs.deke = update_add(vs.deke, at[:, :, 0, vs.tau], -flux_top[:, :, 0] / vs.dzw[0])
        vs.deke = update_add(
            vs.deke,
            at[:, :, 1:-1, vs.tau],
            -(flux_top[:, :, 1:-1] - flux_top[:, :, :-2]) / vs.dzw[npx.newaxis, npx.newaxis, 1:-1],
        )
        vs.deke = update_add(
            vs.deke, at[:, :, -1, vs.tau], -(flux_top[:, :, -1] - flux_top[:, :, -2]) / (0.5 * vs.dzw[-1])
        )
        """
        Adam Bashforth time stepping
        """
        vs.eke = update_add(
            vs.eke,
            at[:, :, :, vs.taup1],
            settings.dt_tracer
            * (
                (1.5 + settings.AB_eps) * vs.deke[:, :, :, vs.tau]
                - (0.5 + settings.AB_eps) * vs.deke[:, :, :, vs.taum1]
            ),
        )

        conditional_outputs.update(deke=vs.deke)

    return KernelOutput(eke=vs.eke, eke_diss_iw=vs.eke_diss_iw, eke_diss_tke=vs.eke_diss_tke, **conditional_outputs)

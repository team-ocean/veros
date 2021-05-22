"""
solve two dimensional Possion equation
     A * dpsi = forc,  where A = nabla_h^2
with Dirichlet boundary conditions
used for streamfunction
"""


from veros import veros_kernel, veros_routine, KernelOutput, runtime_settings
from veros.variables import allocate
from veros.core import utilities as mainutils
from veros.core.operators import update, update_add, at, for_loop
from veros.core.operators import numpy as npx
from veros.core.streamfunction import line_integrals
from veros.core.streamfunction.solvers import get_linear_solver


@veros_routine
def solve_streamfunction(state):
    vs = state.variables

    state_update, (forc, uloc, vloc) = prepare_forcing(state)
    vs.update(state_update)

    linear_solver = get_linear_solver(state)
    linear_sol = linear_solver.solve(state, forc, vs.dpsi[..., vs.taup1])
    vs.dpsi = update(vs.dpsi, at[..., vs.taup1], linear_sol)

    vs.update(barotropic_velocity_update(state, uloc=uloc, vloc=vloc))


@veros_kernel
def prepare_forcing(state):
    vs = state.variables
    settings = state.settings

    # hydrostatic pressure

    if runtime_settings.pyom_compatibility_mode:
        fac = npx.float32(settings.grav) / npx.float32(settings.rho_0)
    else:
        fac = settings.grav / settings.rho_0

    vs.p_hydro = update(
        vs.p_hydro, at[:, :, -1], 0.5 * vs.rho[:, :, -1, vs.tau] * fac * vs.dzw[-1] * vs.maskT[:, :, -1]
    )

    def compute_p_hydro(k_inv, p_hydro):
        k = settings.nz - k_inv - 1
        p_hydro = update(
            p_hydro,
            at[..., k],
            vs.maskT[:, :, k]
            * (p_hydro[:, :, k + 1] + 0.5 * vs.dzw[k] * fac * (vs.rho[:, :, k + 1, vs.tau] + vs.rho[:, :, k, vs.tau])),
        )
        return p_hydro

    vs.p_hydro = for_loop(1, settings.nz, compute_p_hydro, vs.p_hydro)

    # add hydrostatic pressure gradient
    vs.du = update_add(
        vs.du,
        at[2:-2, 2:-2, :, vs.tau],
        -(vs.p_hydro[3:-1, 2:-2, :] - vs.p_hydro[2:-2, 2:-2, :])
        / (vs.cost[npx.newaxis, 2:-2, npx.newaxis] * vs.dxu[2:-2, npx.newaxis, npx.newaxis])
        * vs.maskU[2:-2, 2:-2, :],
    )
    vs.dv = update_add(
        vs.dv,
        at[2:-2, 2:-2, :, vs.tau],
        -(vs.p_hydro[2:-2, 3:-1, :] - vs.p_hydro[2:-2, 2:-2, :])
        / vs.dyu[npx.newaxis, 2:-2, npx.newaxis]
        * vs.maskV[2:-2, 2:-2, :],
    )

    # forcing for barotropic streamfunction
    uloc = npx.sum((vs.du[:, :, :, vs.tau] + vs.du_mix) * vs.maskU * vs.dzt, axis=(2,)) * vs.hur
    vloc = npx.sum((vs.dv[:, :, :, vs.tau] + vs.dv_mix) * vs.maskV * vs.dzt, axis=(2,)) * vs.hvr

    uloc = mainutils.enforce_boundaries(uloc, settings.enable_cyclic_x)
    vloc = mainutils.enforce_boundaries(vloc, settings.enable_cyclic_x)

    forc = allocate(state.dimensions, ("xt", "yt"))
    forc = update(
        forc,
        at[2:-2, 2:-2],
        (vloc[3:-1, 2:-2] - vloc[2:-2, 2:-2]) / (vs.cosu[2:-2] * vs.dxu[2:-2, npx.newaxis])
        - (vs.cost[3:-1] * uloc[2:-2, 3:-1] - vs.cost[2:-2] * uloc[2:-2, 2:-2]) / (vs.cosu[2:-2] * vs.dyu[2:-2]),
    )

    # solve for interior streamfunction
    vs.dpsi = update(vs.dpsi, at[:, :, vs.taup1], 2 * vs.dpsi[:, :, vs.tau] - vs.dpsi[:, :, vs.taum1])

    return KernelOutput(du=vs.du, dv=vs.dv, dpsi=vs.dpsi, p_hydro=vs.p_hydro), (forc, uloc, vloc)


@veros_kernel
def barotropic_velocity_update(state, uloc, vloc):
    """
    solve for barotropic streamfunction
    """
    vs = state.variables
    settings = state.settings

    vs.dpsi = update(
        vs.dpsi, at[:, :, vs.taup1], mainutils.enforce_boundaries(vs.dpsi[:, :, vs.taup1], settings.enable_cyclic_x)
    )

    line_forc = allocate(state.dimensions, ("isle",))

    if state.dimensions["isle"] > 1:
        # calculate island integrals of forcing, keep psi constant on island 1
        line_forc = update(
            line_forc,
            at[1:],
            line_integrals.line_integrals(state, uloc=uloc[..., npx.newaxis], vloc=vloc[..., npx.newaxis], kind="same")[
                1:
            ],
        )

        # calculate island integrals of interior streamfunction
        uloc = update(uloc, at[...], 0.0)
        vloc = update(vloc, at[...], 0.0)
        uloc = update(
            uloc,
            at[1:, 1:],
            -1
            * vs.maskU[1:, 1:, -1]
            * (vs.dpsi[1:, 1:, vs.taup1] - vs.dpsi[1:, :-1, vs.taup1])
            / vs.dyt[npx.newaxis, 1:]
            * vs.hur[1:, 1:],
        )
        vloc = update(
            vloc,
            at[1:, 1:],
            vs.maskV[1:, 1:, -1]
            * (vs.dpsi[1:, 1:, vs.taup1] - vs.dpsi[:-1, 1:, vs.taup1])
            / (vs.cosu[npx.newaxis, 1:] * vs.dxt[1:, npx.newaxis])
            * vs.hvr[1:, 1:],
        )
        line_forc = update_add(
            line_forc,
            at[1:],
            -line_integrals.line_integrals(
                state, uloc=uloc[..., npx.newaxis], vloc=vloc[..., npx.newaxis], kind="same"
            )[1:],
        )

        # solve for time dependent boundary values
        vs.dpsin = update(vs.dpsin, at[1:, vs.tau], npx.linalg.solve(vs.line_psin[1:, 1:], line_forc[1:]))

    # integrate barotropic and baroclinic velocity forward in time
    vs.psi = update(
        vs.psi,
        at[:, :, vs.taup1],
        vs.psi[:, :, vs.tau]
        + settings.dt_mom
        * ((1.5 + settings.AB_eps) * vs.dpsi[:, :, vs.taup1] - (0.5 + settings.AB_eps) * vs.dpsi[:, :, vs.tau]),
    )
    vs.psi = update_add(
        vs.psi,
        at[:, :, vs.taup1],
        settings.dt_mom
        * npx.sum(
            ((1.5 + settings.AB_eps) * vs.dpsin[1:, vs.tau] - (0.5 + settings.AB_eps) * vs.dpsin[1:, vs.taum1])
            * vs.psin[:, :, 1:],
            axis=2,
        ),
    )
    vs.u = update(
        vs.u,
        at[:, :, :, vs.taup1],
        vs.u[:, :, :, vs.tau]
        + settings.dt_mom
        * (
            vs.du_mix
            + (1.5 + settings.AB_eps) * vs.du[:, :, :, vs.tau]
            - (0.5 + settings.AB_eps) * vs.du[:, :, :, vs.taum1]
        )
        * vs.maskU,
    )
    vs.v = update(
        vs.v,
        at[:, :, :, vs.taup1],
        vs.v[:, :, :, vs.tau]
        + settings.dt_mom
        * (
            vs.dv_mix
            + (1.5 + settings.AB_eps) * vs.dv[:, :, :, vs.tau]
            - (0.5 + settings.AB_eps) * vs.dv[:, :, :, vs.taum1]
        )
        * vs.maskV,
    )

    # subtract incorrect vertical mean from baroclinic velocity
    uloc = npx.sum(vs.u[:, :, :, vs.taup1] * vs.maskU * vs.dzt, axis=2)
    vloc = npx.sum(vs.v[:, :, :, vs.taup1] * vs.maskV * vs.dzt, axis=2)
    vs.u = update_add(vs.u, at[:, :, :, vs.taup1], -uloc[:, :, npx.newaxis] * vs.maskU * vs.hur[:, :, npx.newaxis])
    vs.v = update_add(vs.v, at[:, :, :, vs.taup1], -vloc[:, :, npx.newaxis] * vs.maskV * vs.hvr[:, :, npx.newaxis])

    # add barotropic mode to baroclinic velocity
    vs.u = update_add(
        vs.u,
        at[2:-2, 2:-2, :, vs.taup1],
        -1
        * vs.maskU[2:-2, 2:-2, :]
        * (vs.psi[2:-2, 2:-2, vs.taup1, npx.newaxis] - vs.psi[2:-2, 1:-3, vs.taup1, npx.newaxis])
        / vs.dyt[npx.newaxis, 2:-2, npx.newaxis]
        * vs.hur[2:-2, 2:-2, npx.newaxis],
    )
    vs.v = update_add(
        vs.v,
        at[2:-2, 2:-2, :, vs.taup1],
        vs.maskV[2:-2, 2:-2, :]
        * (vs.psi[2:-2, 2:-2, vs.taup1, npx.newaxis] - vs.psi[1:-3, 2:-2, vs.taup1, npx.newaxis])
        / (vs.cosu[2:-2, npx.newaxis] * vs.dxt[2:-2, npx.newaxis, npx.newaxis])
        * vs.hvr[2:-2, 2:-2][:, :, npx.newaxis],
    )

    return KernelOutput(u=vs.u, v=vs.v, psi=vs.psi, dpsi=vs.dpsi, dpsin=vs.dpsin)

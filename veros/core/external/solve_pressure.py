"""
solve two dimensional Possion equation
    A * dpsi = forc,  where A = nabla_h^2
with Neumann boundary conditions
used for surface pressure or free surface
method same as pressure method in MITgcm
"""


from veros import veros_routine
from veros.routines import veros_kernel
from veros.state import KernelOutput
from veros.variables import allocate
from veros.core import utilities as mainutils
from veros.core.operators import update, update_add, at, for_loop
from veros.core.operators import numpy as npx
from veros.core.external.solvers import get_linear_solver


@veros_routine
def solve_pressure(state):
    vs = state.variables
    state_update, forc = prepare_forcing(state)
    vs.update(state_update)

    linear_solver = get_linear_solver(state)
    linear_sol = linear_solver.solve(state, forc, vs.psi[..., vs.taup1])
    linear_sol = mainutils.enforce_boundaries(linear_sol, state.settings.enable_cyclic_x)

    if vs.itt == 0:
        vs.psi = update(vs.psi, at[...], linear_sol[..., npx.newaxis])
    else:
        vs.psi = update(vs.psi, at[..., vs.taup1], linear_sol)

    vs.update(barotropic_velocity_update(state))


@veros_kernel
def prepare_forcing(state):
    vs = state.variables
    settings = state.settings

    # hydrostatic pressure
    vs.p_hydro = update(
        vs.p_hydro,
        at[:, :, -1],
        0.5 * vs.rho[:, :, -1, vs.tau] * settings.grav / settings.rho_0 * vs.dzw[-1] * vs.maskT[:, :, -1],
    )

    def compute_p_hydro(k_inv, p_hydro):
        k = settings.nz - k_inv - 1
        p_hydro = update(
            p_hydro,
            at[..., k],
            vs.maskT[:, :, k]
            * (
                p_hydro[:, :, k + 1]
                + 0.5
                * vs.dzw[k]
                * settings.grav
                / settings.rho_0
                * (vs.rho[:, :, k + 1, vs.tau] + vs.rho[:, :, k, vs.tau])
            ),
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

    # Integrate forward in time
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

    # forcing for surface pressure
    uloc = allocate(state.dimensions, ("xt", "yt"))
    vloc = allocate(state.dimensions, ("xt", "yt"))

    uloc = update(
        uloc,
        at[2:-2, 2:-2],
        npx.sum((vs.u[2:-2, 2:-2, :, vs.taup1]) * vs.maskU[2:-2, 2:-2, :] * vs.dzt, axis=(2,)) / settings.dt_mom,
    )
    vloc = update(
        vloc,
        at[2:-2, 2:-2],
        npx.sum((vs.v[2:-2, 2:-2, :, vs.taup1]) * vs.maskV[2:-2, 2:-2, :] * vs.dzt, axis=(2,)) / settings.dt_mom,
    )

    uloc = mainutils.enforce_boundaries(uloc, settings.enable_cyclic_x)
    vloc = mainutils.enforce_boundaries(vloc, settings.enable_cyclic_x)

    forc = allocate(state.dimensions, ("xt", "yt"))

    forc = update(
        forc,
        at[2:-2, 2:-2],
        (uloc[2:-2, 2:-2] - uloc[1:-3, 2:-2]) / (vs.cost[2:-2] * vs.dxt[2:-2, npx.newaxis])
        + (vs.cosu[2:-2] * vloc[2:-2, 2:-2] - vs.cosu[1:-3] * vloc[2:-2, 1:-3]) / (vs.cost[2:-2] * vs.dyt[2:-2])
        # free surface
        - vs.psi[2:-2, 2:-2, vs.tau]
        / (settings.grav * settings.dt_mom * settings.dt_tracer)
        * vs.maskT[2:-2, 2:-2, -1],
    )

    # first guess
    vs.psi = update(vs.psi, at[:, :, vs.taup1], 2 * vs.psi[:, :, vs.tau] - vs.psi[:, :, vs.taum1])

    return KernelOutput(du=vs.du, dv=vs.dv, u=vs.u, v=vs.v, psi=vs.psi, p_hydro=vs.p_hydro), forc


@veros_kernel
def barotropic_velocity_update(state):
    """
    solve for surface pressure
    """
    vs = state.variables
    settings = state.settings

    vs.u = update_add(
        vs.u,
        at[2:-2, 2:-2, :, vs.taup1],
        -settings.dt_mom
        * (vs.psi[3:-1, 2:-2, vs.taup1, npx.newaxis] - vs.psi[2:-2, 2:-2, vs.taup1, npx.newaxis])
        / (vs.dxu[2:-2, npx.newaxis, npx.newaxis] * vs.cost[2:-2, npx.newaxis])
        * vs.maskU[2:-2, 2:-2, :],
    )

    vs.v = update_add(
        vs.v,
        at[2:-2, 2:-2, :, vs.taup1],
        -settings.dt_mom
        * (vs.psi[2:-2, 3:-1, vs.taup1, npx.newaxis] - vs.psi[2:-2, 2:-2, vs.taup1, npx.newaxis])
        / vs.dyu[npx.newaxis, 2:-2, npx.newaxis]
        * vs.maskV[2:-2, 2:-2, :],
    )

    vs.ssh = vs.psi[..., vs.tau] / settings.grav

    return KernelOutput(u=vs.u, v=vs.v, ssh=vs.ssh)

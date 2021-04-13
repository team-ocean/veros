"""
solve two dimensional Possion equation
     A * dpsi = forc,  where A = nabla_h^2
with Dirichlet boundary conditions
used for streamfunction
"""

from veros.core.operators import numpy as np

from veros import veros_kernel, veros_routine, KernelOutput
from veros.variables import allocate
from veros.core import utilities as mainutils
from veros.core.operators import update, update_add, at
from veros.core.streamfunction import line_integrals
from veros.core.streamfunction.solvers import get_linear_solver


@veros_routine
def solve_streamfunction(state):
    vs = state.variables

    state_update, (forc, fpx, fpy) = prepare_forcing(state)
    vs.update(state_update)

    linear_solver = get_linear_solver(state)
    linear_sol = linear_solver.solve(state, forc, vs.dpsi[..., vs.taup1])
    vs.dpsi = update(vs.dpsi, at[..., vs.taup1], linear_sol)

    vs.update(barotropic_velocity_update(state, fpx=fpx, fpy=fpy))


@veros_kernel
def prepare_forcing(state):
    vs = state.variables
    settings = state.settings

    # hydrostatic pressure
    # TODO: rename these variables
    fxa = settings.grav / settings.rho_0
    tmp = 0.5 * (vs.rho[:, :, :, vs.tau]) * fxa * vs.dzw * vs.maskT
    p_hydro = vs.p_hydro
    p_hydro = update(p_hydro, at[:, :, -1], tmp[:, :, -1])
    tmp = update_add(tmp, at[:, :, :-1], 0.5 * vs.rho[:, :, 1:, vs.tau] * \
        fxa * vs.dzw[:-1] * vs.maskT[:, :, :-1])
    # TODO: replace cumsum
    p_hydro = update(p_hydro, at[:, :, -2::-1], vs.maskT[:, :, -2::-1] * \
        (p_hydro[:, :, -1, np.newaxis] + np.cumsum(tmp[:, :, -2::-1], axis=2)))

    # add hydrostatic pressure gradient
    du, dv = vs.du, vs.dv
    du = update_add(du, at[2:-2, 2:-2, :, vs.tau], \
        -(p_hydro[3:-1, 2:-2, :] - p_hydro[2:-2, 2:-2, :]) \
        / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxu[2:-2, np.newaxis, np.newaxis]) \
        * vs.maskU[2:-2, 2:-2, :])
    dv = update_add(dv, at[2:-2, 2:-2, :, vs.tau], \
        -(p_hydro[2:-2, 3:-1, :] - p_hydro[2:-2, 2:-2, :]) \
        / vs.dyu[np.newaxis, 2:-2, np.newaxis] \
        * vs.maskV[2:-2, 2:-2, :])

    # forcing for barotropic streamfunction
    fpx = np.sum((du[:, :, :, vs.tau] + vs.du_mix)
                 * vs.maskU * vs.dzt, axis=(2,)) * vs.hur
    fpy = np.sum((dv[:, :, :, vs.tau] + vs.dv_mix)
                 * vs.maskV * vs.dzt, axis=(2,)) * vs.hvr

    fpx = mainutils.enforce_boundaries(fpx, settings.enable_cyclic_x)
    fpy = mainutils.enforce_boundaries(fpy, settings.enable_cyclic_x)

    forc = np.zeros_like(fpy)
    forc = update(forc, at[2:-2, 2:-2], (fpy[3:-1, 2:-2] - fpy[2:-2, 2:-2]) \
        / (vs.cosu[2:-2] * vs.dxu[2:-2, np.newaxis]) \
        - (vs.cost[3:-1] * fpx[2:-2, 3:-1] - vs.cost[2:-2] * fpx[2:-2, 2:-2]) \
        / (vs.cosu[2:-2] * vs.dyu[2:-2]))

    # solve for interior streamfunction
    dpsi = vs.dpsi
    dpsi = update(dpsi, at[:, :, vs.taup1], 2 * dpsi[:, :, vs.tau] - dpsi[:, :, vs.taum1])

    return KernelOutput(du=du, dv=dv, dpsi=dpsi), (forc, fpx, fpy)


@veros_kernel
def barotropic_velocity_update(state, fpx, fpy):
    """
    solve for barotropic streamfunction
    """
    vs = state.variables
    settings = state.settings

    dpsi = vs.dpsi
    dpsi = update(dpsi, at[:, :, vs.taup1], mainutils.enforce_boundaries(dpsi[:, :, vs.taup1], settings.enable_cyclic_x))

    line_forc = allocate(state.dimensions, ("isle",))
    dpsin = vs.dpsin

    if state.dimensions["isle"] > 1:
        # calculate island integrals of forcing, keep psi constant on island 1
        line_forc = update(line_forc, at[1:], line_integrals.line_integrals(
            state,
            uloc=fpx[..., np.newaxis],
            vloc=fpy[..., np.newaxis], kind='same')[1:])

        # calculate island integrals of interior streamfunction
        fpx = update(fpx, at[...], 0.)
        fpy = update(fpy, at[...], 0.)
        fpx = update(fpx, at[1:, 1:], -1 * vs.maskU[1:, 1:, -1] \
            * (dpsi[1:, 1:, vs.taup1] - dpsi[1:, :-1, vs.taup1]) \
            / vs.dyt[np.newaxis, 1:] * vs.hur[1:, 1:])
        fpy = update(fpy, at[1:, 1:], vs.maskV[1:, 1:, -1] \
            * (dpsi[1:, 1:, vs.taup1] - dpsi[:-1, 1:, vs.taup1]) \
            / (vs.cosu[np.newaxis, 1:] * vs.dxt[1:, np.newaxis]) * vs.hvr[1:, 1:])
        line_forc = update_add(line_forc, at[1:], -line_integrals.line_integrals(state,
                                                   uloc=fpx[..., np.newaxis],
            vloc=fpy[..., np.newaxis], kind='same')[1:])

        # solve for time dependent boundary values
        dpsin = update(dpsin, at[1:, vs.tau], np.linalg.solve(vs.line_psin[1:, 1:], line_forc[1:]))

    # integrate barotropic and baroclinic velocity forward in time
    psi = vs.psi
    psi = update(psi, at[:, :, vs.taup1], psi[:, :, vs.tau] + settings.dt_mom * ((1.5 + settings.AB_eps) * dpsi[:, :, vs.taup1]
                                                  - (0.5 + settings.AB_eps) * dpsi[:, :, vs.tau]))
    psi = update_add(psi, at[:, :, vs.taup1], settings.dt_mom * np.sum(((1.5 + settings.AB_eps) * dpsin[1:, vs.tau]
                                         - (0.5 + settings.AB_eps) * dpsin[1:, vs.taum1]) * vs.psin[:, :, 1:], axis=2))
    u, v = vs.u, vs.v
    u = update(u, at[:, :, :, vs.taup1], u[:, :, :, vs.tau] + settings.dt_mom * (vs.du_mix + (1.5 + settings.AB_eps) * vs.du[:, :, :, vs.tau]
                                                    - (0.5 + settings.AB_eps) * vs.du[:, :, :, vs.taum1]) * vs.maskU)
    v = update(v, at[:, :, :, vs.taup1], v[:, :, :, vs.tau] + settings.dt_mom * (vs.dv_mix + (1.5 + settings.AB_eps) * vs.dv[:, :, :, vs.tau]
                                                    - (0.5 + settings.AB_eps) * vs.dv[:, :, :, vs.taum1]) * vs.maskV)

    # subtract incorrect vertical mean from baroclinic velocity
    fpx = np.sum(u[:, :, :, vs.taup1] * vs.maskU * vs.dzt, axis=2)
    fpy = np.sum(v[:, :, :, vs.taup1] * vs.maskV * vs.dzt, axis=2)
    u = update_add(u, at[:, :, :, vs.taup1], -fpx[:, :, np.newaxis] * \
        vs.maskU * vs.hur[:, :, np.newaxis])
    v = update_add(v, at[:, :, :, vs.taup1], -fpy[:, :, np.newaxis] * \
        vs.maskV * vs.hvr[:, :, np.newaxis])

    # add barotropic mode to baroclinic velocity
    u = update_add(u, at[2:-2, 2:-2, :, vs.taup1], \
        -1 * vs.maskU[2:-2, 2:-2, :]\
        * (psi[2:-2, 2:-2, vs.taup1, np.newaxis] - psi[2:-2, 1:-3, vs.taup1, np.newaxis]) \
        / vs.dyt[np.newaxis, 2:-2, np.newaxis]\
        * vs.hur[2:-2, 2:-2, np.newaxis])
    v = update_add(v, at[2:-2, 2:-2, :, vs.taup1], \
        vs.maskV[2:-2, 2:-2, :]\
        * (psi[2:-2, 2:-2, vs.taup1, np.newaxis] - psi[1:-3, 2:-2, vs.taup1, np.newaxis]) \
        / (vs.cosu[2:-2, np.newaxis] * vs.dxt[2:-2, np.newaxis, np.newaxis])\
        * vs.hvr[2:-2, 2:-2][:, :, np.newaxis])

    return KernelOutput(u=u, v=v, psi=psi, dpsi=dpsi, dpsin=dpsin)

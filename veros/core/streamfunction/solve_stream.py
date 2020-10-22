"""
solve two dimensional Possion equation
     A * dpsi = forc,  where A = nabla_h^2
with Dirichlet boundary conditions
used for streamfunction
"""

from veros.core.operators import numpy as np

from veros import veros_kernel, veros_routine, run_kernel
from veros.core import utilities as mainutils
from veros.core.operators import update, update_add, at
from veros.core.streamfunction import line_integrals


@veros_routine(
    inputs=(
        'rho', 'dxu', 'dxt', 'dyu', 'dyt', 'dzw', 'dzt',
        'maskT', 'maskU', 'maskV', 'du', 'dv', 'du_mix', 'dv_mix', 'cost', 'cosu', 'psi', 'psin',
        'dpsi', 'dpsin', 'p_hydro', 'hur', 'hvr', 'u', 'v', 'line_psin',
        'line_dir_east_mask', 'line_dir_west_mask',
        'line_dir_north_mask', 'line_dir_south_mask', 'boundary_mask',
    ),
    outputs=('u', 'v', 'du', 'dv', 'p_hydro', 'psi', 'dpsi', 'dpsin'),
    settings=(
        'tau', 'taup1', 'taum1', 'grav', 'rho_0', 'linear_solver', 'enable_cyclic_x',
        'nisle', 'dt_mom', 'AB_eps'
    )
)
def solve_streamfunction(vs):
    du, dv, forc, dpsi, fpx, fpy = run_kernel(prepare_forcing, vs)

    linear_sol = vs.linear_solver.solve(
        vs,
        forc,
        dpsi[..., vs.taup1]
    )

    dpsi = update(dpsi, at[..., vs.taup1], linear_sol)

    u, v, du, dv, p_hydro, psi, dpsi, dpsin = run_kernel(
        barotropic_velocity_update, vs, dpsi=dpsi, fpx=fpx, fpy=fpy, du=du, dv=dv
    )

    return dict(
        u=u,
        v=v,
        du=du,
        dv=dv,
        p_hydro=p_hydro,
        psi=psi,
        dpsi=dpsi,
        dpsin=dpsin
    )


@veros_kernel(static_args=('enable_cyclic_x',))
def prepare_forcing(grav, rho_0, tau, taup1, taum1, dzt, maskT, p_hydro, rho, dzw, du, dv, dxu, dyu,
                    maskU, maskV, cost, du_mix, dv_mix, hur, hvr, enable_cyclic_x, cosu, dpsi):
    # hydrostatic pressure
    # TODO: rename these variables
    fxa = grav / rho_0
    tmp = 0.5 * (rho[:, :, :, tau]) * fxa * dzw * maskT
    p_hydro = update(p_hydro, at[:, :, -1], tmp[:, :, -1])
    tmp = update_add(tmp, at[:, :, :-1], 0.5 * rho[:, :, 1:, tau] * \
        fxa * dzw[:-1] * maskT[:, :, :-1])
    # TODO: replace cumsum
    p_hydro = update(p_hydro, at[:, :, -2::-1], maskT[:, :, -2::-1] * \
        (p_hydro[:, :, -1, np.newaxis] + np.cumsum(tmp[:, :, -2::-1], axis=2)))

    # add hydrostatic pressure gradient
    du = update_add(du, at[2:-2, 2:-2, :, tau], \
        -(p_hydro[3:-1, 2:-2, :] - p_hydro[2:-2, 2:-2, :]) \
        / (cost[np.newaxis, 2:-2, np.newaxis] * dxu[2:-2, np.newaxis, np.newaxis]) \
        * maskU[2:-2, 2:-2, :])
    dv = update_add(dv, at[2:-2, 2:-2, :, tau], \
        -(p_hydro[2:-2, 3:-1, :] - p_hydro[2:-2, 2:-2, :]) \
        / dyu[np.newaxis, 2:-2, np.newaxis] \
        * maskV[2:-2, 2:-2, :])

    # forcing for barotropic streamfunction
    fpx = np.sum((du[:, :, :, tau] + du_mix)
                 * maskU * dzt, axis=(2,)) * hur
    fpy = np.sum((dv[:, :, :, tau] + dv_mix)
                 * maskV * dzt, axis=(2,)) * hvr

    fpx = mainutils.enforce_boundaries(fpx, enable_cyclic_x)
    fpy = mainutils.enforce_boundaries(fpy, enable_cyclic_x)

    forc = np.zeros_like(fpy)
    forc = update(forc, at[2:-2, 2:-2], (fpy[3:-1, 2:-2] - fpy[2:-2, 2:-2]) \
        / (cosu[2:-2] * dxu[2:-2, np.newaxis]) \
        - (cost[3:-1] * fpx[2:-2, 3:-1] - cost[2:-2] * fpx[2:-2, 2:-2]) \
        / (cosu[2:-2] * dyu[2:-2]))

    # solve for interior streamfunction
    dpsi = update(dpsi, at[:, :, taup1], 2 * dpsi[:, :, tau] - dpsi[:, :, taum1])

    return du, dv, forc, dpsi, fpx, fpy


@veros_kernel(static_args=('nisle', 'enable_cyclic_x'))
def barotropic_velocity_update(dxu, dxt, dyu, dyt, dzw, dzt, tau, taup1, taum1,
                               maskT, maskU, maskV, du, dv, du_mix, dv_mix, cost, cosu, psi, psin,
                               dpsi, dpsin, p_hydro, hur, hvr, nisle, dt_mom, AB_eps, u, v, line_psin,
                               enable_cyclic_x, line_dir_east_mask, line_dir_west_mask,
                               line_dir_north_mask, line_dir_south_mask, boundary_mask, fpx, fpy):
    """
    solve for barotropic streamfunction
    """
    dpsi = update(dpsi, at[:, :, taup1], mainutils.enforce_boundaries(dpsi[:, :, taup1], enable_cyclic_x))

    line_forc = np.zeros_like(dpsin[..., -1])

    if nisle > 1:
        # calculate island integrals of forcing, keep psi constant on island 1
        line_forc = update(line_forc, at[1:], line_integrals.line_integrals(dxu=dxu, dyu=dyu, cost=cost,
                                                 line_dir_east_mask=line_dir_east_mask,
                                                 line_dir_west_mask=line_dir_west_mask,
                                                 line_dir_north_mask=line_dir_north_mask,
                                                 line_dir_south_mask=line_dir_south_mask,
                                                 boundary_mask=boundary_mask, nisle=nisle,
                                                 uloc=fpx[..., np.newaxis],
                                                 vloc=fpy[..., np.newaxis])[1:], kind='same')

        # calculate island integrals of interior streamfunction
        fpx = update(fpx, at[...], 0.)
        fpy = update(fpy, at[...], 0.)
        fpx = update(fpx, at[1:, 1:], -maskU[1:, 1:, -1] \
            * (dpsi[1:, 1:, taup1] - dpsi[1:, :-1, taup1]) \
            / dyt[np.newaxis, 1:] * hur[1:, 1:])
        fpy = update(fpy, at[1:, 1:], maskV[1:, 1:, -1] \
            * (dpsi[1:, 1:, taup1] - dpsi[:-1, 1:, taup1]) \
            / (cosu[np.newaxis, 1:] * dxt[1:, np.newaxis]) * hvr[1:, 1:])
        line_forc = update_add(line_forc, at[1:], -line_integrals.line_integrals(dxu=dxu, dyu=dyu, cost=cost,
                                                   line_dir_east_mask=line_dir_east_mask,
                                                   line_dir_west_mask=line_dir_west_mask,
                                                   line_dir_north_mask=line_dir_north_mask,
                                                   line_dir_south_mask=line_dir_south_mask,
                                                   boundary_mask=boundary_mask, nisle=nisle,
                                                   uloc=fpx[..., np.newaxis],
                                                   vloc=fpy[..., np.newaxis])[1:], kind='same')

        # solve for time dependent boundary values
        dpsin = update(dpsin, at[1:, tau], np.linalg.solve(line_psin[1:, 1:], line_forc[1:]))

    # integrate barotropic and baroclinic velocity forward in time
    psi = update(psi, at[:, :, taup1], psi[:, :, tau] + dt_mom * ((1.5 + AB_eps) * dpsi[:, :, taup1]
                                                  - (0.5 + AB_eps) * dpsi[:, :, tau]))
    psi = update_add(psi, at[:, :, taup1], dt_mom * np.sum(((1.5 + AB_eps) * dpsin[1:, tau]
                                         - (0.5 + AB_eps) * dpsin[1:, taum1]) * psin[:, :, 1:], axis=2))
    u = update(u, at[:, :, :, taup1], u[:, :, :, tau] + dt_mom * (du_mix + (1.5 + AB_eps) * du[:, :, :, tau]
                                                    - (0.5 + AB_eps) * du[:, :, :, taum1]) * maskU)
    v = update(v, at[:, :, :, taup1], v[:, :, :, tau] + dt_mom * (dv_mix + (1.5 + AB_eps) * dv[:, :, :, tau]
                                                    - (0.5 + AB_eps) * dv[:, :, :, taum1]) * maskV)

    # subtract incorrect vertical mean from baroclinic velocity
    fpx = np.sum(u[:, :, :, taup1] * maskU * dzt, axis=2)
    fpy = np.sum(v[:, :, :, taup1] * maskV * dzt, axis=2)
    u = update_add(u, at[:, :, :, taup1], -fpx[:, :, np.newaxis] * \
        maskU * hur[:, :, np.newaxis])
    v = update_add(v, at[:, :, :, taup1], -fpy[:, :, np.newaxis] * \
        maskV * hvr[:, :, np.newaxis])

    # add barotropic mode to baroclinic velocity
    u = update_add(u, at[2:-2, 2:-2, :, taup1], \
        -maskU[2:-2, 2:-2, :]\
        * (psi[2:-2, 2:-2, taup1, np.newaxis] - psi[2:-2, 1:-3, taup1, np.newaxis]) \
        / dyt[np.newaxis, 2:-2, np.newaxis]\
        * hur[2:-2, 2:-2, np.newaxis])
    v = update_add(v, at[2:-2, 2:-2, :, taup1], \
        maskV[2:-2, 2:-2, :]\
        * (psi[2:-2, 2:-2, taup1, np.newaxis] - psi[1:-3, 2:-2, taup1, np.newaxis]) \
        / (cosu[2:-2, np.newaxis] * dxt[2:-2, np.newaxis, np.newaxis])\
        * hvr[2:-2, 2:-2][:, :, np.newaxis])

    return u, v, du, dv, p_hydro, psi, dpsi, dpsin

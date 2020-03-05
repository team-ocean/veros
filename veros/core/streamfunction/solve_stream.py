"""
solve two dimensional Possion equation
     A * dpsi = forc,  where A = nabla_h^2
with Dirichlet boundary conditions
used for streamfunction
"""

import numpy as np

from veros import veros_kernel
from veros.core import utilities as mainutils
from veros.core.streamfunction import utilities


@veros_kernel(static_args=('nisle', 'enable_cyclic_x'))
def solve_streamfunction(grav, rho_0, rho, dxu, dxt, dyu, dyt, dzw, dzt, tau, taup1, taum1,
                         maskT, maskU, maskV, du, dv, du_mix, dv_mix, cost, cosu, psi, psin,
                         dpsi, dpsin, p_hydro, hur, hvr, nisle, dt_mom, AB_eps, u, v, line_psin,
                         linear_solver, enable_cyclic_x, line_dir_east_mask, line_dir_west_mask,
                         line_dir_north_mask, line_dir_south_mask, boundary_mask):
    """
    solve for barotropic streamfunction
    """
    # hydrostatic pressure
    fxa = grav / rho_0
    tmp = 0.5 * (rho[:, :, :, tau]) * fxa * dzw * maskT
    p_hydro[:, :, -1] = tmp[:, :, -1]
    tmp[:, :, :-1] += 0.5 * rho[:, :, 1:, tau] * \
        fxa * dzw[:-1] * maskT[:, :, :-1]
    p_hydro[:, :, -2::-1] = maskT[:, :, -2::-1] * \
        (p_hydro[:, :, -1, np.newaxis] + np.cumsum(tmp[:, :, -2::-1], axis=2))

    # add hydrostatic pressure gradient
    du[2:-2, 2:-2, :, tau] += \
        -(p_hydro[3:-1, 2:-2, :] - p_hydro[2:-2, 2:-2, :]) \
        / (cost[np.newaxis, 2:-2, np.newaxis] * dxu[2:-2, np.newaxis, np.newaxis]) \
        * maskU[2:-2, 2:-2, :]
    dv[2:-2, 2:-2, :, tau] += \
        -(p_hydro[2:-2, 3:-1, :] - p_hydro[2:-2, 2:-2, :]) \
        / dyu[np.newaxis, 2:-2, np.newaxis] \
        * maskV[2:-2, 2:-2, :]

    # forcing for barotropic streamfunction
    fpx = np.sum((du[:, :, :, tau] + du_mix)
                 * maskU * dzt, axis=(2,)) * hur
    fpy = np.sum((dv[:, :, :, tau] + dv_mix)
                 * maskV * dzt, axis=(2,)) * hvr

    mainutils.enforce_boundaries(fpx, enable_cyclic_x)
    mainutils.enforce_boundaries(fpy, enable_cyclic_x)

    forc = np.zeros_like(fpy)
    forc[2:-2, 2:-2] = (fpy[3:-1, 2:-2] - fpy[2:-2, 2:-2]) \
        / (cosu[2:-2] * dxu[2:-2, np.newaxis]) \
        - (cost[3:-1] * fpx[2:-2, 3:-1] - cost[2:-2] * fpx[2:-2, 2:-2]) \
        / (cosu[2:-2] * dyu[2:-2])

    # solve for interior streamfunction
    dpsi[:, :, taup1] = 2 * dpsi[:, :, tau] - dpsi[:, :, taum1]

    linear_solver.solve(
        forc,
        dpsi[..., taup1]
    )

    mainutils.enforce_boundaries(dpsi[:, :, taup1], enable_cyclic_x)

    line_forc = np.zeros_like(dpsin[..., -1])

    if nisle > 1:
        # calculate island integrals of forcing, keep psi constant on island 1
        line_forc[1:] = utilities.line_integrals(dxu=dxu, dyu=dyu, cost=cost,
                                                 line_dir_east_mask=line_dir_east_mask,
                                                 line_dir_west_mask=line_dir_west_mask,
                                                 line_dir_north_mask=line_dir_north_mask,
                                                 line_dir_south_mask=line_dir_south_mask,
                                                 boundary_mask=boundary_mask, nisle=nisle,
                                                 uloc=fpx[..., np.newaxis],
                                                 vloc=fpy[..., np.newaxis], kind='same')[1:]

        # calculate island integrals of interior streamfunction
        fpx[...] = 0.
        fpy[...] = 0.
        fpx[1:, 1:] = -maskU[1:, 1:, -1] \
            * (dpsi[1:, 1:, taup1] - dpsi[1:, :-1, taup1]) \
            / dyt[np.newaxis, 1:] * hur[1:, 1:]
        fpy[1:, 1:] = maskV[1:, 1:, -1] \
            * (dpsi[1:, 1:, taup1] - dpsi[:-1, 1:, taup1]) \
            / (cosu[np.newaxis, 1:] * dxt[1:, np.newaxis]) * hvr[1:, 1:]
        line_forc[1:] += -utilities.line_integrals(dxu=dxu, dyu=dyu, cost=cost,
                                                   line_dir_east_mask=line_dir_east_mask,
                                                   line_dir_west_mask=line_dir_west_mask,
                                                   line_dir_north_mask=line_dir_north_mask,
                                                   line_dir_south_mask=line_dir_south_mask,
                                                   boundary_mask=boundary_mask, nisle=nisle,
                                                   uloc=fpx[..., np.newaxis],
                                                   vloc=fpy[..., np.newaxis], kind='same')[1:]

        # solve for time dependent boundary values
        dpsin[1:, tau] = np.linalg.solve(line_psin[1:, 1:], line_forc[1:])

    # integrate barotropic and baroclinic velocity forward in time
    psi[:, :, taup1] = psi[:, :, tau] + dt_mom * ((1.5 + AB_eps) * dpsi[:, :, taup1]
                                                  - (0.5 + AB_eps) * dpsi[:, :, tau])
    psi[:, :, taup1] += dt_mom * np.sum(((1.5 + AB_eps) * dpsin[1:, tau]
                                         - (0.5 + AB_eps) * dpsin[1:, taum1]) * psin[:, :, 1:], axis=2)
    u[:, :, :, taup1] = u[:, :, :, tau] + dt_mom * (du_mix + (1.5 + AB_eps) * du[:, :, :, tau]
                                                    - (0.5 + AB_eps) * du[:, :, :, taum1]) * maskU
    v[:, :, :, taup1] = v[:, :, :, tau] + dt_mom * (dv_mix + (1.5 + AB_eps) * dv[:, :, :, tau]
                                                    - (0.5 + AB_eps) * dv[:, :, :, taum1]) * maskV

    # subtract incorrect vertical mean from baroclinic velocity
    fpx = np.sum(u[:, :, :, taup1] * maskU * dzt, axis=2)
    fpy = np.sum(v[:, :, :, taup1] * maskV * dzt, axis=2)
    u[:, :, :, taup1] += -fpx[:, :, np.newaxis] * \
        maskU * hur[:, :, np.newaxis]
    v[:, :, :, taup1] += -fpy[:, :, np.newaxis] * \
        maskV * hvr[:, :, np.newaxis]

    # add barotropic mode to baroclinic velocity
    u[2:-2, 2:-2, :, taup1] += \
        -maskU[2:-2, 2:-2, :]\
        * (psi[2:-2, 2:-2, taup1, np.newaxis] - psi[2:-2, 1:-3, taup1, np.newaxis]) \
        / dyt[np.newaxis, 2:-2, np.newaxis]\
        * hur[2:-2, 2:-2, np.newaxis]
    v[2:-2, 2:-2, :, taup1] += \
        maskV[2:-2, 2:-2, :]\
        * (psi[2:-2, 2:-2, taup1, np.newaxis] - psi[1:-3, 2:-2, taup1, np.newaxis]) \
        / (cosu[2:-2, np.newaxis] * dxt[2:-2, np.newaxis, np.newaxis])\
        * hvr[2:-2, 2:-2][:, :, np.newaxis]

    return u, v, du, dv, p_hydro, psi, dpsi, dpsin

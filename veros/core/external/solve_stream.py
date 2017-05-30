"""
solve two dimensional Possion equation
     A * dpsi = forc,  where A = nabla_h^2
with Dirichlet boundary conditions
used for streamfunction
"""

from . import island, utilities, solve_poisson
from .. import cyclic
from ... import veros_method

import scipy.linalg.lapack


@veros_method
def solve_streamfunction(veros):
    """
    solve for barotropic streamfunction
    """
    line_forc = np.zeros(veros.nisle, dtype=veros.default_float_type)
    aloc = np.zeros((veros.nisle, veros.nisle), dtype=veros.default_float_type)

    # hydrostatic pressure
    fxa = veros.grav / veros.rho_0
    tmp = 0.5 * (veros.rho[:, :, :, veros.tau]) * fxa * veros.dzw * veros.maskT
    veros.p_hydro[:, :, -1] = tmp[:, :, -1]
    tmp[:, :, :-1] += 0.5 * veros.rho[:, :, 1:, veros.tau] * \
        fxa * veros.dzw[:-1] * veros.maskT[:, :, :-1]
    veros.p_hydro[:, :, -2::-1] = veros.maskT[:, :, -2::-1] * \
        (veros.p_hydro[:, :, -1, np.newaxis] + np.cumsum(tmp[:, :, -2::-1], axis=2))

    # add hydrostatic pressure gradient
    veros.du[2:-2, 2:-2, :, veros.tau] += \
        -(veros.p_hydro[3:-1, 2:-2, :] - veros.p_hydro[2:-2, 2:-2, :]) \
        / (veros.cost[np.newaxis, 2:-2, np.newaxis] * veros.dxu[2:-2, np.newaxis, np.newaxis]) \
        * veros.maskU[2:-2, 2:-2, :]
    veros.dv[2:-2, 2:-2, :, veros.tau] += \
        -(veros.p_hydro[2:-2, 3:-1, :] - veros.p_hydro[2:-2, 2:-2, :]) \
        / veros.dyu[np.newaxis, 2:-2, np.newaxis] \
        * veros.maskV[2:-2, 2:-2, :]

    # forcing for barotropic streamfunction
    fpx = np.sum((veros.du[:, :, :, veros.tau] + veros.du_mix) *
                 veros.maskU * veros.dzt, axis=(2,)) * veros.hur
    fpy = np.sum((veros.dv[:, :, :, veros.tau] + veros.dv_mix) *
                 veros.maskV * veros.dzt, axis=(2,)) * veros.hvr

    if veros.enable_cyclic_x:
        cyclic.setcyclic_x(fpx)
        cyclic.setcyclic_x(fpy)

    forc = np.zeros((veros.nx + 4, veros.ny + 4), dtype=veros.default_float_type)
    forc[2:-2, 2:-2] = (fpy[3:-1, 2:-2] - fpy[2:-2, 2:-2]) \
        / (veros.cosu[2:-2] * veros.dxu[2:-2, np.newaxis]) \
        - (veros.cost[3:-1] * fpx[2:-2, 3:-1] - veros.cost[2:-2] * fpx[2:-2, 2:-2]) \
        / (veros.cosu[2:-2] * veros.dyu[2:-2])

    # solve for interior streamfunction
    veros.dpsi[:, :, veros.taup1] = 2 * veros.dpsi[:, :, veros.tau] - \
        veros.dpsi[:, :, veros.taum1]  # first guess, we need three time levels here
    solve_poisson.solve(veros, forc, veros.dpsi[:, :, veros.taup1])

    if veros.enable_cyclic_x:
        cyclic.setcyclic_x(veros.dpsi[:, :, veros.taup1])

    if veros.nisle > 1:
        # calculate island integrals of forcing, keep psi constant on island 1
        line_forc[1:] = utilities.line_integrals(
            veros, fpx[..., np.newaxis], fpy[..., np.newaxis], kind="same")[1:]

        # calculate island integrals of interior streamfunction
        fpx[...] = 0.
        fpy[...] = 0.
        fpx[1:, 1:] = -veros.maskU[1:, 1:, -1] \
            * (veros.dpsi[1:, 1:, veros.taup1] - veros.dpsi[1:, :-1, veros.taup1]) \
            / veros.dyt[np.newaxis, 1:] * veros.hur[1:, 1:]
        fpy[1:, 1:] = veros.maskV[1:, 1:, -1] \
            * (veros.dpsi[1:, 1:, veros.taup1] - veros.dpsi[:-1, 1:, veros.taup1]) \
            / (veros.cosu[np.newaxis, 1:] * veros.dxt[1:, np.newaxis]) * veros.hvr[1:, 1:]
        line_forc[1:] += -utilities.line_integrals(veros, fpx[..., np.newaxis],
                                                   fpy[..., np.newaxis], kind="same")[1:]

        # solve for time dependent boundary values
        if veros.backend_name == "bohrium":
            line_forc[1:] = np.lapack.gesv(veros.line_psin[1:, 1:], line_forc[1:])
        else:
            line_forc[1:] = scipy.linalg.lapack.dgesv(veros.line_psin[1:, 1:], line_forc[1:])[2]
        veros.dpsin[1:, veros.tau] = line_forc[1:]

    # integrate barotropic and baroclinic velocity forward in time
    veros.psi[:, :, veros.taup1] = veros.psi[:, :, veros.tau] + veros.dt_mom * ((1.5 + veros.AB_eps) * veros.dpsi[:, :, veros.taup1]
                                                                                - (0.5 + veros.AB_eps) * veros.dpsi[:, :, veros.tau])
    veros.psi[:, :, veros.taup1] += veros.dt_mom * np.sum(((1.5 + veros.AB_eps) * veros.dpsin[1:, veros.tau]
                                                           - (0.5 + veros.AB_eps) * veros.dpsin[1:, veros.taum1]) * veros.psin[:, :, 1:], axis=2)
    veros.u[:, :, :, veros.taup1] = veros.u[:, :, :, veros.tau] + veros.dt_mom * (veros.du_mix + (1.5 + veros.AB_eps) * veros.du[:, :, :, veros.tau]
                                                                                  - (0.5 + veros.AB_eps) * veros.du[:, :, :, veros.taum1]) * veros.maskU
    veros.v[:, :, :, veros.taup1] = veros.v[:, :, :, veros.tau] + veros.dt_mom * (veros.dv_mix + (1.5 + veros.AB_eps) * veros.dv[:, :, :, veros.tau]
                                                                                  - (0.5 + veros.AB_eps) * veros.dv[:, :, :, veros.taum1]) * veros.maskV

    # subtract incorrect vertical mean from baroclinic velocity
    fpx = np.sum(veros.u[:, :, :, veros.taup1] * veros.maskU * veros.dzt, axis=(2,))
    fpy = np.sum(veros.v[:, :, :, veros.taup1] * veros.maskV * veros.dzt, axis=(2,))
    veros.u[:, :, :, veros.taup1] += -fpx[:, :, np.newaxis] * \
        veros.maskU * veros.hur[:, :, np.newaxis]
    veros.v[:, :, :, veros.taup1] += -fpy[:, :, np.newaxis] * \
        veros.maskV * veros.hvr[:, :, np.newaxis]

    # add barotropic mode to baroclinic velocity
    veros.u[2:-2, 2:-2, :, veros.taup1] += \
        -veros.maskU[2:-2, 2:-2, :]\
        * (veros.psi[2:-2, 2:-2, veros.taup1, np.newaxis] - veros.psi[2:-2, 1:-3, veros.taup1, np.newaxis]) \
        / veros.dyt[np.newaxis, 2:-2, np.newaxis]\
        * veros.hur[2:-2, 2:-2, np.newaxis]
    veros.v[2:-2, 2:-2, :, veros.taup1] += \
        veros.maskV[2:-2, 2:-2, :]\
        * (veros.psi[2:-2, 2:-2, veros.taup1, np.newaxis] - veros.psi[1:-3, 2:-2, veros.taup1, np.newaxis]) \
        / (veros.cosu[2:-2, np.newaxis] * veros.dxt[2:-2, np.newaxis, np.newaxis])\
        * veros.hvr[2:-2, 2:-2][:, :, np.newaxis]

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
def solve_streamfunction(vs):
    """
    solve for barotropic streamfunction
    """
    line_forc = np.zeros(vs.nisle, dtype=vs.default_float_type)
    aloc = np.zeros((vs.nisle, vs.nisle), dtype=vs.default_float_type)

    # hydrostatic pressure
    fxa = vs.grav / vs.rho_0
    tmp = 0.5 * (vs.rho[:, :, :, vs.tau]) * fxa * vs.dzw * vs.maskT
    vs.p_hydro[:, :, -1] = tmp[:, :, -1]
    tmp[:, :, :-1] += 0.5 * vs.rho[:, :, 1:, vs.tau] * \
        fxa * vs.dzw[:-1] * vs.maskT[:, :, :-1]
    vs.p_hydro[:, :, -2::-1] = vs.maskT[:, :, -2::-1] * \
        (vs.p_hydro[:, :, -1, np.newaxis] + np.cumsum(tmp[:, :, -2::-1], axis=2))

    # add hydrostatic pressure gradient
    vs.du[2:-2, 2:-2, :, vs.tau] += \
        -(vs.p_hydro[3:-1, 2:-2, :] - vs.p_hydro[2:-2, 2:-2, :]) \
        / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxu[2:-2, np.newaxis, np.newaxis]) \
        * vs.maskU[2:-2, 2:-2, :]
    vs.dv[2:-2, 2:-2, :, vs.tau] += \
        -(vs.p_hydro[2:-2, 3:-1, :] - vs.p_hydro[2:-2, 2:-2, :]) \
        / vs.dyu[np.newaxis, 2:-2, np.newaxis] \
        * vs.maskV[2:-2, 2:-2, :]

    # forcing for barotropic streamfunction
    fpx = np.sum((vs.du[:, :, :, vs.tau] + vs.du_mix) *
                 vs.maskU * vs.dzt, axis=(2,)) * vs.hur
    fpy = np.sum((vs.dv[:, :, :, vs.tau] + vs.dv_mix) *
                 vs.maskV * vs.dzt, axis=(2,)) * vs.hvr

    if vs.enable_cyclic_x:
        cyclic.setcyclic_x(fpx)
        cyclic.setcyclic_x(fpy)

    forc = np.zeros((vs.nx + 4, vs.ny + 4), dtype=vs.default_float_type)
    forc[2:-2, 2:-2] = (fpy[3:-1, 2:-2] - fpy[2:-2, 2:-2]) \
        / (vs.cosu[2:-2] * vs.dxu[2:-2, np.newaxis]) \
        - (vs.cost[3:-1] * fpx[2:-2, 3:-1] - vs.cost[2:-2] * fpx[2:-2, 2:-2]) \
        / (vs.cosu[2:-2] * vs.dyu[2:-2])

    # solve for interior streamfunction
    vs.dpsi[:, :, vs.taup1] = 2 * vs.dpsi[:, :, vs.tau] - vs.dpsi[:, :, vs.taum1]
    solve_poisson.solve(vs, forc, vs.dpsi[:, :, vs.taup1])

    if vs.enable_cyclic_x:
        cyclic.setcyclic_x(vs.dpsi[:, :, vs.taup1])

    if vs.nisle > 1:
        # calculate island integrals of forcing, keep psi constant on island 1
        line_forc[1:] = utilities.line_integrals(vs, fpx[..., np.newaxis],
                                                 fpy[..., np.newaxis], kind="same")[1:]

        # calculate island integrals of interior streamfunction
        fpx[...] = 0.
        fpy[...] = 0.
        fpx[1:, 1:] = -vs.maskU[1:, 1:, -1] \
            * (vs.dpsi[1:, 1:, vs.taup1] - vs.dpsi[1:, :-1, vs.taup1]) \
            / vs.dyt[np.newaxis, 1:] * vs.hur[1:, 1:]
        fpy[1:, 1:] = vs.maskV[1:, 1:, -1] \
            * (vs.dpsi[1:, 1:, vs.taup1] - vs.dpsi[:-1, 1:, vs.taup1]) \
            / (vs.cosu[np.newaxis, 1:] * vs.dxt[1:, np.newaxis]) * vs.hvr[1:, 1:]
        line_forc[1:] += -utilities.line_integrals(vs, fpx[..., np.newaxis],
                                                   fpy[..., np.newaxis], kind="same")[1:]

        # solve for time dependent boundary values
        if vs.backend_name == "bohrium":
            line_forc[1:] = np.lapack.gesv(vs.line_psin[1:, 1:], line_forc[1:])
        else:
            line_forc[1:] = scipy.linalg.lapack.dgesv(vs.line_psin[1:, 1:], line_forc[1:])[2]
        vs.dpsin[1:, vs.tau] = line_forc[1:]

    # integrate barotropic and baroclinic velocity forward in time
    vs.psi[:, :, vs.taup1] = vs.psi[:, :, vs.tau] + vs.dt_mom * ((1.5 + vs.AB_eps) * vs.dpsi[:, :, vs.taup1]
                                                               - (0.5 + vs.AB_eps) * vs.dpsi[:, :, vs.tau])
    vs.psi[:, :, vs.taup1] += vs.dt_mom * np.sum(((1.5 + vs.AB_eps) * vs.dpsin[1:, vs.tau]
                                                           - (0.5 + vs.AB_eps) * vs.dpsin[1:, vs.taum1]) * vs.psin[:, :, 1:], axis=2)
    vs.u[:, :, :, vs.taup1] = vs.u[:, :, :, vs.tau] + vs.dt_mom * (vs.du_mix + (1.5 + vs.AB_eps) * vs.du[:, :, :, vs.tau]
                                                                             - (0.5 + vs.AB_eps) * vs.du[:, :, :, vs.taum1]) * vs.maskU
    vs.v[:, :, :, vs.taup1] = vs.v[:, :, :, vs.tau] + vs.dt_mom * (vs.dv_mix + (1.5 + vs.AB_eps) * vs.dv[:, :, :, vs.tau]
                                                                             - (0.5 + vs.AB_eps) * vs.dv[:, :, :, vs.taum1]) * vs.maskV

    # subtract incorrect vertical mean from baroclinic velocity
    fpx = np.sum(vs.u[:, :, :, vs.taup1] * vs.maskU * vs.dzt, axis=(2,))
    fpy = np.sum(vs.v[:, :, :, vs.taup1] * vs.maskV * vs.dzt, axis=(2,))
    vs.u[:, :, :, vs.taup1] += -fpx[:, :, np.newaxis] * \
        vs.maskU * vs.hur[:, :, np.newaxis]
    vs.v[:, :, :, vs.taup1] += -fpy[:, :, np.newaxis] * \
        vs.maskV * vs.hvr[:, :, np.newaxis]

    # add barotropic mode to baroclinic velocity
    vs.u[2:-2, 2:-2, :, vs.taup1] += \
        -vs.maskU[2:-2, 2:-2, :]\
        * (vs.psi[2:-2, 2:-2, vs.taup1, np.newaxis] - vs.psi[2:-2, 1:-3, vs.taup1, np.newaxis]) \
        / vs.dyt[np.newaxis, 2:-2, np.newaxis]\
        * vs.hur[2:-2, 2:-2, np.newaxis]
    vs.v[2:-2, 2:-2, :, vs.taup1] += \
        vs.maskV[2:-2, 2:-2, :]\
        * (vs.psi[2:-2, 2:-2, vs.taup1, np.newaxis] - vs.psi[1:-3, 2:-2, vs.taup1, np.newaxis]) \
        / (vs.cosu[2:-2, np.newaxis] * vs.dxt[2:-2, np.newaxis, np.newaxis])\
        * vs.hvr[2:-2, 2:-2][:, :, np.newaxis]

"""
solve two dimensional Possion equation
     A * dpsi = forc,  where A = nabla_h^2
with Dirichlet boundary conditions
used for streamfunction
"""

import numpy as np

import climate
from climate.pyom.external import solve_pressure, island, utilities, solve_poisson
from climate.pyom import cyclic


def solve_streamfunction(pyom):
    """
    solve for barotropic streamfunction
    """
    line_forc = np.empty(pyom.nisle)
    aloc = np.empty((pyom.nisle, pyom.nisle))

    #hydrostatic pressure
    fxa = pyom.grav / pyom.rho_0
    tmp = 0.5 * (pyom.rho[:,:,:,pyom.tau]) * fxa * pyom.dzw * pyom.maskT
    pyom.p_hydro[:,:,pyom.nz-1] = tmp[:,:,-1]
    tmp[:,:,:-1] += 0.5 * pyom.rho[:,:,1:,pyom.tau] * fxa*pyom.dzw[:-1] * pyom.maskT[:,:,:-1]
    pyom.p_hydro[:,:,-2::-1] = np.cumsum(pyom.maskT[:,:,-2::-1] * tmp[:,:,:0:-1], axis=2)

    # add hydrostatic pressure gradient
    pyom.du[2:-2,2:-2,:,pyom.tau] += \
            -(pyom.p_hydro[3:-1,2:-2,:] - pyom.p_hydro[2:-2,2:-2,:]) \
            / (pyom.cost[np.newaxis,2:-2,np.newaxis] * pyom.dxu[2:-2,np.newaxis,np.newaxis]) \
            * pyom.maskU[2:-2,2:-2,:]
    pyom.dv[2:-2,2:-2,:,pyom.tau] += \
            -(pyom.p_hydro[2:-2,3:-1,:] - pyom.p_hydro[2:-2,2:-2,:]) \
            / pyom.dyu[np.newaxis, 2:-2, np.newaxis] \
            * pyom.maskV[2:-2,2:-2,:]

    # forcing for barotropic streamfunction
    fpx = np.sum((pyom.du[:,:,:,pyom.tau] + pyom.du_mix) * pyom.maskU * pyom.dzt, axis=(2,)) * pyom.hur
    fpy = np.sum((pyom.dv[:,:,:,pyom.tau] + pyom.dv_mix) * pyom.maskV * pyom.dzt, axis=(2,)) * pyom.hvr

    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(fpx)
        cyclic.setcyclic_x(fpy)

    forc = np.zeros((pyom.nx+4, pyom.ny+4))
    forc[2:-2, 2:-2] = (fpy[3:-1, 2:-2] - fpy[2:-2, 2:-2]) \
            / (pyom.cosu[2:-2] * pyom.dxu[2:-2, np.newaxis]) \
            - (pyom.cost[3:-1] * fpx[2:-2, 3:-1] - pyom.cost[2:-2] * fpx[2:-2, 2:-2]) \
            / (pyom.cosu[2:-2] * pyom.dyu[2:-2])

    # solve for interior streamfunction
    pyom.dpsi[:,:,pyom.taup1] = 2 * pyom.dpsi[:,:,pyom.tau] - pyom.dpsi[:,:,pyom.taum1] # first guess, we need three time levels here
    solve_poisson.solve(forc, pyom.dpsi[:,:,pyom.taup1], pyom)

    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(pyom.dpsi[:,:,pyom.taup1])

    if pyom.nisle > 1:
        # calculate island integrals of forcing, keep psi constant on island 1
        line_forc[1:] = utilities.line_integrals(fpx[...,np.newaxis], fpy[...,np.newaxis], pyom, kind="same")[1:]

        # calculate island integrals of interior streamfunction
        fpx[...] = 0.
        fpy[...] = 0.
        fpx[1:, 1:] = -pyom.maskU[1:, 1:, -1] \
                        * (pyom.dpsi[1:, 1:, pyom.taup1] - pyom.dpsi[1:, :-1, pyom.taup1]) \
                        / pyom.dyt[np.newaxis, 1:] * pyom.hur[1:, 1:]
        fpy[1:, 1:] = pyom.maskV[1:, 1:, -1] \
                        * (pyom.dpsi[1:, 1:, pyom.taup1] - pyom.dpsi[:-1, 1:, pyom.taup1]) \
                        / (pyom.cosu[np.newaxis, 1:] * pyom.dxt[1:, np.newaxis]) * pyom.hvr[1:, 1:]
        line_forc[1:] += -utilities.line_integrals(fpx[..., np.newaxis], fpy[..., np.newaxis], pyom, kind="same")[1:]

        # solve for time dependent boundary values
        if climate.is_bohrium:
            line_forc[1:] = np.linalg.jacobi(pyom.line_psin[1:, 1:], line_forc[1:])
        else:
            line_forc[1:] = np.linalg.solve(pyom.line_psin[1:, 1:], line_forc[1:])
        pyom.dpsin[1:,pyom.tau] = line_forc[1:]

    # integrate barotropic and baroclinic velocity forward in time
    pyom.psi[:,:,pyom.taup1] = pyom.psi[:,:,pyom.tau] + pyom.dt_mom * ((1.5 + pyom.AB_eps) * pyom.dpsi[:,:,pyom.taup1] \
                                - (0.5 + pyom.AB_eps) * pyom.dpsi[:,:,pyom.tau])
    pyom.psi[:, :, pyom.taup1] += pyom.dt_mom * np.sum(((1.5 + pyom.AB_eps) * pyom.dpsin[1:,pyom.tau] \
                                - (0.5 + pyom.AB_eps) * pyom.dpsin[1:,pyom.taum1]) * pyom.psin[:,:,1:], axis=2)
    pyom.u[:,:,:,pyom.taup1] = pyom.u[:,:,:,pyom.tau] + pyom.dt_mom*(pyom.du_mix + (1.5 + pyom.AB_eps) * pyom.du[:,:,:,pyom.tau] \
                                - (0.5 + pyom.AB_eps) * pyom.du[:,:,:,pyom.taum1]) * pyom.maskU
    pyom.v[:,:,:,pyom.taup1] = pyom.v[:,:,:,pyom.tau] + pyom.dt_mom*(pyom.dv_mix + (1.5 + pyom.AB_eps) * pyom.dv[:,:,:,pyom.tau] \
                                - (0.5 + pyom.AB_eps) * pyom.dv[:,:,:,pyom.taum1]) * pyom.maskV

    # subtract incorrect vertical mean from baroclinic velocity
    fpx = np.sum(pyom.u[:,:,:,pyom.taup1] * pyom.maskU * pyom.dzt, axis=(2,))
    fpy = np.sum(pyom.v[:,:,:,pyom.taup1] * pyom.maskV * pyom.dzt, axis=(2,))
    pyom.u[:,:,:,pyom.taup1] += -fpx[:, :, np.newaxis] * pyom.maskU * pyom.hur[:, :, np.newaxis]
    pyom.v[:,:,:,pyom.taup1] += -fpy[:, :, np.newaxis] * pyom.maskV * pyom.hvr[:, :, np.newaxis]

    # add barotropic mode to baroclinic velocity
    pyom.u[2:-2, 2:-2, :, pyom.taup1] += \
            -pyom.maskU[2:-2,2:-2,:]\
            * (pyom.psi[2:-2,2:-2,pyom.taup1,np.newaxis] - pyom.psi[2:-2,1:-3,pyom.taup1,np.newaxis]) \
            / pyom.dyt[np.newaxis, 2:-2, np.newaxis]\
            * pyom.hur[2:-2, 2:-2, np.newaxis]
    pyom.v[2:-2, 2:-2, :, pyom.taup1] += \
            pyom.maskV[2:-2,2:-2,:]\
            * (pyom.psi[2:-2,2:-2,pyom.taup1,np.newaxis] - pyom.psi[1:-3,2:-2,pyom.taup1,np.newaxis]) \
            / (pyom.cosu[2:-2,np.newaxis] * pyom.dxt[2:-2, np.newaxis, np.newaxis])\
            * pyom.hvr[2:-2,2:-2][:, :, np.newaxis]

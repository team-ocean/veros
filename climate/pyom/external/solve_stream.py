"""
solve two dimensional Possion equation
     A * dpsi = forc,  where A = nabla_h^2
with Dirichlet boundary conditions
used for streamfunction
"""

import numpy as np

import climate
from climate.pyom.external import solve_pressure, island, utilities, congrad_streamfunction
from climate.pyom import cyclic, diagnostics


def solve_streamfunction(pyom,benchtest=False):
    """
    solve for barotropic streamfunction
    """
    line_forc = np.empty(pyom.nisle)
    line_psi0 = np.empty(pyom.nisle)
    aloc = np.empty((pyom.nisle, pyom.nisle))

    #hydrostatic pressure
    fxa = pyom.grav/pyom.rho_0
    tmp = 0.5*(pyom.rho[:,:,:,pyom.tau])*fxa*pyom.dzw*pyom.maskT
    pyom.p_hydro[:,:,pyom.nz-1] = tmp[:,:,-1] #0.5*pyom.rho[:,:,pyom.nz-1,pyom.tau]*fxa*pyom.dzw[pyom.nz-1]*pyom.maskT[:,:,pyom.nz-1]
    tmp[:,:,:-1] += 0.5*pyom.rho[:,:,1:,pyom.tau]*fxa*pyom.dzw[:-1]*pyom.maskT[:,:,:-1]
    for k in xrange(pyom.nz-2, -1, -1): #k=nz-1,1,-1
        pyom.p_hydro[:,:,k] = pyom.maskT[:,:,k]*pyom.p_hydro[:,:,k+1]+ tmp[:,:,k] #0.5*(pyom.rho[:,:,k+1,pyom.tau]+pyom.rho[:,:,k,pyom.tau])*fxa*pyom.dzw[k])

    # add hydrostatic pressure gradient
    pyom.du[2:-2,2:-2,:,pyom.tau] += \
            -(pyom.p_hydro[3:-1,2:-2,:]-pyom.p_hydro[2:-2,2:-2,:]) \
            / (pyom.cost[np.newaxis,2:-2,np.newaxis] * pyom.dxu[2:-2,np.newaxis,np.newaxis]) \
            *pyom.maskU[2:-2,2:-2,:]
    pyom.dv[2:-2,2:-2,:,pyom.tau] += \
            -(pyom.p_hydro[2:-2,3:-1,:]-pyom.p_hydro[2:-2,2:-2,:]) \
            / pyom.dyu[np.newaxis, 2:-2, np.newaxis] \
            * pyom.maskV[2:-2,2:-2,:]

    # forcing for barotropic streamfunction
    fpx = np.zeros((pyom.nx+4, pyom.ny+4))
    fpy = np.zeros((pyom.nx+4, pyom.ny+4))
    fpx += np.sum((pyom.du[:,:,:,pyom.tau] + pyom.du_mix) * pyom.maskU * pyom.dzt, axis=(2,))
    fpy += np.sum((pyom.dv[:,:,:,pyom.tau] + pyom.dv_mix) * pyom.maskV * pyom.dzt, axis=(2,))

    fpx *= pyom.hur
    fpy *= pyom.hvr

    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(fpx)
        cyclic.setcyclic_x(fpy)

    forc = np.empty((pyom.nx+4, pyom.ny+4))
    forc[2:-2, 2:-2] = (fpy[3:-1, 2:-2] - fpy[2:-2, 2:-2]) \
            / (pyom.cosu[2:-2] * pyom.dxu[2:-2, np.newaxis]) \
            - (pyom.cost[3:-1] * fpx[2:-2, 3:-1] - pyom.cost[2:-2] * fpx[2:-2, 2:-2]) \
            / (pyom.cosu[2:-2] * pyom.dyu[2:-2])

    # solve for interior streamfunction
    pyom.dpsi[:,:,pyom.taup1] = 2 * pyom.dpsi[:,:,pyom.tau] - pyom.dpsi[:,:,pyom.taum1] # first guess, we need three time levels here
    congrad_streamfunction.congrad_streamfunction(forc,pyom.dpsi[:,:,pyom.taup1], pyom)

    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(pyom.dpsi[:,:,pyom.taup1])

    if pyom.nisle > 1:
        # calculate island integrals of forcing, keep psi constant on island 1
        for k in xrange(1,pyom.nisle): #k=2,nisle
            line_forc[k] = utilities.line_integral(k,fpx,fpy,pyom)

        # calculate island integrals of interior streamfunction
        for k in xrange(1, pyom.nisle): #k=2,nisle
            fpx[...] = 0.0
            fpy[...] = 0.0
            fpx[1:, 1:] = \
                    -pyom.maskU[1:, 1:, pyom.nz-1] \
                    * (pyom.dpsi[1:, 1:,pyom.taup1]-pyom.dpsi[1:, :-1,pyom.taup1]) \
                    /pyom.dyt[1:] *pyom.hur[1:, 1:]
            fpy[1:, 1:] = \
                    pyom.maskV[1:, 1:, pyom.nz-1] \
                    * (pyom.dpsi[1:, 1:,pyom.taup1]-pyom.dpsi[:-1, 1:,pyom.taup1]) \
                    /(pyom.cosu[1:]*pyom.dxt[1:, np.newaxis])*pyom.hvr[1:,1:]
            line_psi0[k] = utilities.line_integral(k,fpx,fpy,pyom)

        line_forc -= line_psi0

        # solve for time dependent boundary values
        if climate.is_bohrium:
            line_forc[1:pyom.nisle] = np.linalg.jacobi(pyom.line_psin[1:pyom.nisle, 1:pyom.nisle], line_forc[1:pyom.nisle])
        else:
            line_forc[1:pyom.nisle] = np.linalg.solve(pyom.line_psin[1:pyom.nisle, 1:pyom.nisle], line_forc[1:pyom.nisle])
        pyom.dpsin[1:pyom.nisle,pyom.tau] = line_forc[1:pyom.nisle]

    # integrate barotropic and baroclinic velocity forward in time
    pyom.psi[:,:,pyom.taup1] = pyom.psi[:,:,pyom.tau] + pyom.dt_mom*((1.5+pyom.AB_eps)*pyom.dpsi[:,:,pyom.taup1] - (0.5+pyom.AB_eps)*pyom.dpsi[:,:,pyom.tau] )
    pyom.psi[:, :, pyom.taup1] += pyom.dt_mom * np.sum(((1.5+pyom.AB_eps)*pyom.dpsin[1:pyom.nisle,pyom.tau] - (0.5+pyom.AB_eps)*pyom.dpsin[1:pyom.nisle,pyom.taum1]) * pyom.psin[:,:,1:pyom.nisle], axis=2)
    pyom.u[:,:,:,pyom.taup1] = pyom.u[:,:,:,pyom.tau] + pyom.dt_mom*(pyom.du_mix + (1.5+pyom.AB_eps)*pyom.du[:,:,:,pyom.tau] - (0.5+pyom.AB_eps)*pyom.du[:,:,:,pyom.taum1] )*pyom.maskU
    pyom.v[:,:,:,pyom.taup1] = pyom.v[:,:,:,pyom.tau] + pyom.dt_mom*(pyom.dv_mix + (1.5+pyom.AB_eps)*pyom.dv[:,:,:,pyom.tau] - (0.5+pyom.AB_eps)*pyom.dv[:,:,:,pyom.taum1] )*pyom.maskV

    # subtract incorrect vertical mean from baroclinic velocity
    fpx = np.sum(pyom.u[:,:,:,pyom.taup1]*pyom.maskU*pyom.dzt, axis=(2,))
    fpy = np.sum(pyom.v[:,:,:,pyom.taup1]*pyom.maskV*pyom.dzt, axis=(2,))
    pyom.u[:,:,:,pyom.taup1] += (np.ones(pyom.nz)*-fpx[:, :, np.newaxis])*pyom.maskU*(np.ones(pyom.nz)*pyom.hur[:, :, np.newaxis])
    pyom.v[:,:,:,pyom.taup1] += (np.ones(pyom.nz)*-fpy[:, :, np.newaxis])*pyom.maskV*(np.ones(pyom.nz)*pyom.hvr[:, :, np.newaxis])

    # add barotropic mode to baroclinic velocity
    pyom.u[2:-2, 2:-2, :, pyom.taup1] += \
            -pyom.maskU[2:-2,2:-2,:]\
            *(np.ones(pyom.nz) * (pyom.psi[2:-2,2:-2,pyom.taup1]-pyom.psi[2:-2,1:-3,pyom.taup1])[:, :,np.newaxis])\
            /(np.ones(pyom.nz) * pyom.dyt[2:-2,np.newaxis]*np.ones(pyom.nx)[:, np.newaxis, np.newaxis])\
            *(np.ones(pyom.nz) * pyom.hur[2:-2,2:-2][:, :, np.newaxis])
    pyom.v[2:-2, 2:-2, :, pyom.taup1] += \
            pyom.maskV[2:-2,2:-2,:]\
            *(np.ones(pyom.nz) * (pyom.psi[2:-2,2:-2,pyom.taup1]-pyom.psi[1:-3,2:-2,pyom.taup1])[:, :,np.newaxis])\
            /(np.ones(pyom.nz) * pyom.cosu[2:-2,np.newaxis]*pyom.dxt[2:-2, np.newaxis, np.newaxis])\
            *(np.ones(pyom.nz) * pyom.hvr[2:-2,2:-2][:, :, np.newaxis])

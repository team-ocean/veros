import numpy as np
import math

from climate.pyom import cyclic

def tempsalt_biharmonic(pyom):
    """
    biharmonic mixing of pyom.temp and salinity,
    dissipation of dyn. Enthalpy is stored
    """
    # integer :: i,j,k,ks,is_,ie,js,je
    # real*8 :: aloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    # real*8 :: del2(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz),fxa

    aloc = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    del2 = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    is_ = pyom.is_pe-pyom.onx
    ie = pyom.ie_pe+pyom.onx
    js = pyom.js_pe-pyom.onx
    je = pyom.je_pe+pyom.onx

    fxa = math.sqrt(abs(pyom.K_hbi))

    for j in xrange(js,je): # j = js,je
        for i in xrange(is_,ie-1): # i = is_,ie-1
            pyom.flux_east[i,j,:] = -fxa*(pyom.temp[i+1,j,:,pyom.tau]-pyom.temp[i,j,:,pyom.tau])/(pyom.cost[j]*pyom.dxu[i])*pyom.maskU[i,j,:]

    for j in xrange(js,je-1): # j = js,je-1
        pyom.flux_north[:,j,:] = -fxa*(pyom.temp[:,j+1,:,pyom.tau]-pyom.temp[:,j,:,pyom.tau])/pyom.dyu[j]*pyom.maskV[:,j,:]*pyom.cosu[j]
    pyom.flux_east[ie-1,:,:] = 0.
    pyom.flux_north[:,je-1,:] = 0.

    for j in xrange(js+1,je): # j = js+1,je
        for i in xrange(is_+1,ie): # i = is_+1,ie
            del2[i,j,:] = pyom.maskT[i,j,:] * (pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:])/(pyom.cost[j]*pyom.dxt[i]) \
                                        +(pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:])/(pyom.cost[j]*pyom.dyt[j])

    # border_exchg_xyz(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx,pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx,nz,del2)
    cyclic.setcyclic_xyz(del2,pyom.enable_cyclic_x,pyom.nx,pyom.nz)

    for j in xrange(js,je): # j = js,je
        for i in xrange(is_,ie-1): # i = is_,ie-1
            pyom.flux_east[i,j,:] = fxa*(del2[i+1,j,:]-del2[i,j,:])/(pyom.cost[j]*pyom.dxu[i])*pyom.maskU[i,j,:]

    for j in xrange(js,je-1): # j = js,je-1
        pyom.flux_north[:,j,:] = fxa*(del2[:,j+1,:]-del2[:,j,:])/pyom.dyu[j]*pyom.maskV[:,j,:]*pyom.cosu[j]

    pyom.flux_east[ie-1,:,:] = 0.
    pyom.flux_north[:,je-1,:] = 0.

    # update tendency
    for j in xrange(pyom.js_pe-pyom.onx+1,pyom.je_pe+pyom.onx): # j = js_pe-onx+1,je_pe+onx
        for i in xrange(pyom.is_pe-pyom.onx+1,pyom.ie_pe+pyom.onx): # i = is_pe-onx+1,ie_pe+onx
            pyom.dtemp_hmix[i,j,:] = pyom.maskT[i,j,:] * (pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:])/(pyom.cost[j]*pyom.dxt[i]) \
                                                 + (pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:])/(pyom.cost[j]*pyom.dyt[j])

    pyom.temp[:,:,:,pyom.taup1] = pyom.temp[:,:,:,pyom.taup1] + pyom.dt_tracer * pyom.dtemp_hmix * pyom.maskT

    if pyom.enable_conserve_energy:
        # diagnose dissipation of dynamic enthalpy by hor. mixing of temperature
        for k in xrange(1,pyom.nz): # k = 1,nz
            for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
                for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                    fxa = pyom.int_drhodT[i,j,k,pyom.tau]
                    aloc[i,j,k] = 0.5*pyom.grav/pyom.rho_0*((pyom.int_drhodT[i+1,j,k,pyom.tau]-fxa)*pyom.flux_east[i,j,k] \
                                     +(fxa-pyom.int_drhodT[i-1,j,k,pyom.tau])*pyom.flux_east[i-1,j,k]) / (pyom.dxt[i]*pyom.cost[j])  \
                                 +0.5*pyom.grav/pyom.rho_0*((pyom.int_drhodT[i,j+1,k,pyom.tau]-fxa)*pyom.flux_north[i,j,k] \
                                     +(fxa-pyom.int_drhodT[i,j-1,k,pyom.tau])*pyom.flux_north[i,j-1,k]) / (pyom.dyt[j]*pyom.cost[j])

        # dissipation interpolated on W-grid
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                ks = pyom.kbot[i,j] - 1
                if ks >= 0:
                    k = ks
                    pyom.P_diss_hmix[i,j,k] = 0.5*(aloc[i,j,k]+aloc[i,j,k+1]) + 0.5*aloc[i,j,k]*pyom.dzw[max(1,k-1)]/pyom.dzw[k]
                    for k in xrange(ks+1,pyom.nz-2): # k = ks+1,nz-1
                        pyom.P_diss_hmix[i,j,k] = 0.5*(aloc[i,j,k] + aloc[i,j,k+1])
                    k = pyom.nz - 1
                    pyom.P_diss_hmix[i,j,k] = aloc[i,j,k]

    for j in xrange(js,je): # j = js,je
        for i in xrange(is_,ie-1): # i = is_,ie-1
            pyom.flux_east[i,j,:] = -fxa*(pyom.salt[i+1,j,:,pyom.tau]-pyom.salt[i,j,:,pyom.tau])/(pyom.cost[j]*pyom.dxu[i])*pyom.maskU[i,j,:]

    for j in xrange(js,je-1): # j = js,je-1
        pyom.flux_north[:,j,:] = -fxa*(pyom.salt[:,j+1,:,pyom.tau]-pyom.salt[:,j,:,pyom.tau])/pyom.dyu[j]*pyom.maskV[:,j,:]*pyom.cosu[j]

    pyom.flux_east[ie-1,:,:] = 0.
    pyom.flux_north[:,je-1,:] = 0.

    for j in xrange(js+1,je): # j = js+1,je
        for i in xrange(is_+1,ie): # i = is_+1,ie
            del2[i,j,:] = pyom.maskT[i,j,:]* (pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:])/(pyom.cost[j]*pyom.dxt[i]) \
                                        +(pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:])/(pyom.cost[j]*pyom.dyt[j])

    # border_exchg_xyz(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx,pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx,nz,del2)
    cyclic.setcyclic_xyz(del2,pyom.enable_cyclic_x,pyom.nx,pyom.nz)

    for j in xrange(js,je): # j = js,je
        for i in xrange(is_,ie-1): # i = is_,ie-1
            pyom.flux_east[i,j,:] = fxa*(del2[i+1,j,:]-del2[i,j,:])/(pyom.cost[j]*pyom.dxu[i])*pyom.maskU[i,j,:]

    for j in xrange(js,je-1): # j = js,je-1
        pyom.flux_north[:,j,:] = fxa*(del2[:,j+1,:]-del2[:,j,:])/pyom.dyu[j]*pyom.maskV[:,j,:]*pyom.cosu[j]

    pyom.flux_east[ie-1,:,:] = 0.
    pyom.flux_north[:,je-1,:] = 0.

    # update tendency
    for j in xrange(pyom.js_pe-pyom.onx+1,pyom.je_pe+pyom.onx): # j = js_pe-onx+1,je_pe+onx
        for i in xrange(pyom.is_pe-pyom.onx+1,pyom.ie_pe+pyom.onx): # i = is_pe-onx+1,ie_pe+onx
            pyom.dsalt_hmix[i,j,:] = pyom.maskT[i,j,:] * (pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:])/(pyom.cost[j]*pyom.dxt[i]) \
                                                +(pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:])/(pyom.cost[j]*pyom.dyt[j])

    pyom.salt[:,:,:,pyom.taup1] = pyom.salt[:,:,:,pyom.taup1] + pyom.dt_tracer * pyom.dsalt_hmix * pyom.maskT

    if pyom.enable_conserve_energy:
        # diagnose dissipation of dynamic enthalpy by hor. mixing of salinity
        for k in xrange(1,pyom.nz): # k = 1,nz
            for j in xrange(pyom.js_pe-pyom.onx+1,pyom.je_pe+pyom.onx-1): # j = js_pe-onx+1,je_pe+onx-1
                for i in xrange(pyom.is_pe-pyom.onx+1,pyom.ie_pe+pyom.onx-1): # i = is_pe-onx+1,ie_pe+onx-1
                    fxa = pyom.int_drhodS[i,j,k,pyom.tau]
                    aloc[i,j,k] = 0.5*pyom.grav/pyom.rho_0*((pyom.int_drhodS[i+1,j,k,pyom.tau]-fxa)*pyom.flux_east[i,j,k] \
                                                  +(fxa-pyom.int_drhodS[i-1,j,k,pyom.tau])*pyom.flux_east[i-1,j,k]) /(pyom.dxt[i]*pyom.cost[j])  \
                                 +0.5*pyom.grav/pyom.rho_0*((pyom.int_drhodS[i,j+1,k,pyom.tau]-fxa)*pyom.flux_north[i,j,k] \
                                                  +(fxa-pyom.int_drhodS[i,j-1,k,pyom.tau])*pyom.flux_north[i,j-1,k]) /(pyom.dyt[j]*pyom.cost[j])

        # dissipation interpolated on W-grid
        for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): # j = js_pe-onx,je_pe+onx
            for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx): # i = is_pe-onx,ie_pe+onx
                ks = pyom.kbot[i,j] - 1
                if ks >= 0:
                    k = ks
                    pyom.P_diss_hmix[i,j,k] = pyom.P_diss_hmix[i,j,k]+ \
                                         0.5*(aloc[i,j,k] + aloc[i,j,k+1]) + 0.5*aloc[i,j,k]*pyom.dzw[max(1,k-1)]/pyom.dzw[k]
                    for k in xrange(ks+1,pyom.nz-2): # k = ks+1,nz-1
                        pyom.P_diss_hmix[i,j,k] = pyom.P_diss_hmix[i,j,k] + 0.5*(aloc[i,j,k] + aloc[i,j,k+1])
                    k = pyom.nz - 1
                    pyom.P_diss_hmix[i,j,k] = pyom.P_diss_hmix[i,j,k] + aloc[i,j,k]


def tempsalt_diffusion(pyom):
    """
    Diffusion of pyom.temp and salinity,
    dissipation of dyn. Enthalpy is stored
    """
    # integer :: i,j,k,ks
    # real*8 :: fxa
    # real*8 :: aloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)

    # horizontal diffusion of temperature
    for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): # j = js_pe-onx,je_pe+onx
        for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx-1): # i = is_pe-onx,ie_pe+onx-1
            pyom.flux_east[i,j,:] = K_h * (pyom.temp[i+1,j,:,pyom.tau] - pyom.temp[i,j,:,pyom.tau]) / (pyom.cost[j]*pyom.dxu[i])*pyom.maskU[i,j,:]

    pyom.flux_east[pyom.ie_pe+pyom.onx,:,:] = 0.
    for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx-1): # j = js_pe-onx,je_pe+onx-1
        pyom.flux_north[:,j,:] = K_h * (pyom.temp[:,j+1,:,pyom.tau] - pyom.temp[:,j,:,pyom.tau]) / pyom.dyu[j]*pyom.maskV[:,j,:]*pyom.cosu[j]
    pyom.flux_north[:,pyom.je_pe+pyom.onx,:] = 0.

    if pyom.enable_hor_friction_cos_scaling:
        for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): # j = js_pe-onx,je_pe+onx
            pyom.flux_east[:,j,:] = pyom.flux_east[:,j,:]*pyom.cost[j]**pyom.hor_friction_cosPower
            pyom.flux_north[:,j,:] = pyom.flux_north[:,j,:]*pyom.cosu[j]**pyom.hor_friction_cosPower

    for j in xrange(pyom.js_pe-pyom.onx+1,pyom.je_pe+pyom.onx): # j = js_pe-onx+1,je_pe+onx
        for i in xrange(pyom.is_pe-pyom.onx+1,pyom.ie_pe+pyom.onx): # i = is_pe-onx+1,ie_pe+onx
            pyom.dtemp_hmix[i,j,:] = pyom.maskT[i,j,:]*((pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:])/(pyom.cost[j]*pyom.dxt[i]) \
                                            +(pyom.flux_north[i,j,:] -pyom.flux_north[i,j-1,:])/(pyom.cost[j]*pyom.dyt[j]))

    pyom.temp[:,:,:,pyom.taup1] = pyom.temp[:,:,:,pyom.taup1] + pyom.dt_tracer * pyom.dtemp_hmix * pyom.maskT

    if pyom.enable_conserve_energy:
        # diagnose dissipation of dynamic enthalpy by hor. mixing of temperature
        for k in xrange(1,pyom.nz): # k = 1,nz
            for j in xrange(pyom.js_pe-pyom.onx+1,pyom.je_pe+pyom.onx-1): # j = js_pe-onx+1,je_pe+onx-1
                for i in xrange(pyom.is_pe-pyom.onx+1,pyom.ie_pe+pyom.onx-1): # i = is_pe-onx+1,ie_pe+onx-1
                    fxa = pyom.int_drhodT[i,j,k,pyom.tau]
                    aloc[i,j,k] = 0.5*pyom.grav/pyom.rho_0*((pyom.int_drhodT[i+1,j,k,pyom.tau]-fxa)*pyom.flux_east[i,j,k] \
                                                 +(fxa-pyom.int_drhodT[i-1,j,k,pyom.tau])*pyom.flux_east[i-1,j,k]) / (pyom.dxt[i]*pyom.cost[j])  \
                                + 0.5*pyom.grav/pyom.rho_0*((pyom.int_drhodT(i,j+1,k,pyom.tau)-fxa)*pyom.flux_north[i,j,k] \
                                                 +(fxa-pyom.int_drhodT[i,j-1,k,pyom.tau])*pyom.flux_north[i,j-1,k]) /(pyom.dyt[j]*pyom.cost[j])

        # dissipation interpolated on W-grid
        for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): # j = js_pe-onx,je_pe+onx
            for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx): # i = is_pe-onx,ie_pe+onx
                ks = pyom.kbot[i,j]
                if ks>0:
                    k = ks
                    pyom.P_diss_hmix[i,j,k] = 0.5*(aloc[i,j,k]+aloc[i,j,k+1]) + 0.5*aloc[i,j,k]*pyom.dzw[max(1,k-1)]/pyom.dzw[k]
                    for k in xrange(ks+1,pyom.nz-1): # k = ks+1,nz-1
                        pyom.P_diss_hmix[i,j,k] = 0.5*(aloc[i,j,k] + aloc[i,j,k+1])
                    k = pyom.nz
                    pyom.P_diss_hmix[i,j,k] = aloc[i,j,k]

    # horizontal diffusion of salinity
    for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): # j = js_pe-onx,je_pe+onx
        for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx-1): # i = is_pe-onx,ie_pe+onx-1
            pyom.flux_east[i,j,:] = K_h * (pyom.salt[i+1,j,:,pyom.tau] - pyom.salt[i,j,:,pyom.tau]) / (pyom.cost[j]*pyom.dxu[i])*pyom.maskU[i,j,:]

    pyom.flux_east[pyom.ie_pe+pyom.onx,:,:] = 0.
    for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx-1): # j = js_pe-onx,je_pe+onx-1
        pyom.flux_north[:,j,:] = K_h * (pyom.salt[:,j+1,:,pyom.tau] - pyom.salt[:,j,:,pyom.tau]) / pyom.dyu[j]*pyom.maskV[:,j,:]*pyom.cosu[j]
    pyom.flux_north[:,pyom.je_pe+pyom.onx,:] = 0.

    if pyom.enable_hor_friction_cos_scaling:
        for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): # j = js_pe-onx,je_pe+onx
            pyom.flux_east[:,j,:] = pyom.flux_east[:,j,:]*pyom.cost[j]**pyom.hor_friction_cosPower
            pyom.flux_north[:,j,:] = pyom.flux_north[:,j,:]*pyom.cosu[j]**pyom.hor_friction_cosPower

    for j in xrange(pyom.js_pe-pyom.onx+1,pyom.je_pe+pyom.onx): # j = js_pe-onx+1,je_pe+onx
        for i in xrange(pyom.is_pe-pyom.onx+1,pyom.ie_pe+pyom.onx): # i = is_pe-onx+1,ie_pe+onx
            pyom.dsalt_hmix[i,j,:] = pyom.maskT[i,j,:] * ((pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:]) / (pyom.cost[j]*pyom.dxt[i]) \
                                               + (pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:]) / (pyom.cost[j]*pyom.dyt[j]))

    pyom.salt[:,:,:,pyom.taup1] = pyom.salt[:,:,:,pyom.taup1] + pyom.dt_tracer * pyom.dsalt_hmix * pyom.maskT

    if pyom.enable_conserve_energy:
        # diagnose dissipation of dynamic enthalpy by hor. mixing of salinity
        for k in xrange(1,pyom.nz): # k = 1,nz
            for j in xrange(pyom.js_pe-pyom.onx+1,pyom.je_pe+pyom.onx-1): # j = js_pe-onx+1,je_pe+onx-1
                for i in xrange(pyom.is_pe-pyom.onx+1,pyom.ie_pe+pyom.onx-1): # i = is_pe-onx+1,ie_pe+onx-1
                    fxa = pyom.int_drhodS[i,j,k,pyom.tau]
                    aloc[i,j,k] = 0.5*pyom.grav/pyom.rho_0*((pyom.int_drhodS[i+1,j,k,pyom.tau]-fxa)*pyom.flux_east[i,j,k] \
                                                   +(fxa-pyom.int_drhodS[i-1,j,k,pyom.tau])*pyom.flux_east[i-1,j,k]) / (pyom.dxt[i]*pyom.cost[j])  \
                                  + 0.5*pyom.grav/pyom.rho_0*((pyom.int_drhodS(i,j+1,k,pyom.tau)-fxa)*pyom.flux_north(i,j  ,k) \
                                                   +(fxa-pyom.int_drhodS[i,j-1,k,pyom.tau])*pyom.flux_north[i,j-1,k]) / (pyom.dyt[j]*pyom.cost[j])

        # dissipation interpolated on W-grid
        for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): # j = js_pe-onx,je_pe+onx
            for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx): # i = is_pe-onx,ie_pe+onx
                ks = pyom.kbot[i,j] - 1
                if ks >= 0:
                    k = ks
                    pyom.P_diss_hmix[i,j,k] = pyom.P_diss_hmix[i,j,k]+ \
                                        0.5*(aloc[i,j,k]+aloc[i,j,k+1]) + 0.5*aloc[i,j,k]*pyom.dzw[max(1,k-1)]/pyom.dzw[k]
                    for k in xrange(ks+1,pyom.nz-2): # k = ks+1,nz-1
                        pyom.P_diss_hmix[i,j,k] = pyom.P_diss_hmix[i,j,k]+ 0.5*(aloc[i,j,k] + aloc[i,j,k+1])
                    k = pyom.nz - 1
                    pyom.P_diss_hmix[i,j,k] = pyom.P_diss_hmix[i,j,k] + aloc[i,j,k]


def tempsalt_sources(pyom):
    """
    Sources of pyom.temp and salinity,
    effect on dyn. Enthalpy is stored
    """
    # integer :: i,j,k,ks
    # real*8 :: aloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)

    pyom.temp[:,:,:,pyom.taup1] = pyom.temp[:,:,:,pyom.taup1] + pyom.dt_tracer * temp_source * pyom.maskT
    pyom.salt[:,:,:,pyom.taup1] = pyom.salt[:,:,:,pyom.taup1] + pyom.dt_tracer * salt_source * pyom.maskT

    if pyom.enable_conserve_energy:
        # diagnose effect on dynamic enthalpy
        for k in xrange(1,pyom.nz): # k = 1,nz
            for j in xrange(pyom.js_pe-pyom.onx+1,pyom.je_pe+pyom.onx-1): # j = js_pe-onx+1,je_pe+onx-1
                for i in xrange(pyom.is_pe-pyom.onx+1,pyom.ie_pe+pyom.onx-1): # i = is_pe-onx+1,ie_pe+onx-1
                    aloc[i,j,k] = -pyom.grav/pyom.rho_0*pyom.maskT[i,j,k] \
                                    * (pyom.int_drhodT[i,j,k,pyom.tau]*temp_source[i,j,k]+pyom.int_drhodS[i,j,k,pyom.tau]*salt_source[i,j,k])

        # dissipation interpolated on W-grid
        for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): # j = js_pe-onx,je_pe+onx
            for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx): # i = is_pe-onx,ie_pe+onx
                ks = pyom.kbot[i,j]
                if ks>0:
                    k = ks
                    P_diss_sources[i,j,k] = 0.5*(aloc[i,j,k]+aloc[i,j,k+1]) + 0.5*aloc[i,j,k]*pyom.dzw[max(1,k-1)]/pyom.dzw[k]
                    for k in xrange(ks+1,pyom.nz-1): # k = ks+1,nz-1
                        P_diss_sources[i,j,k] = 0.5*(aloc[i,j,k] + aloc[i,j,k+1])
                    k = pyom.nz
                    P_diss_sources[i,j,k] = aloc[i,j,k]

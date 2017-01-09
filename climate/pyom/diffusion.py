import numpy as np
import math

import climate.pyom.cyclic as cyclic

def tempsalt_biharmonic(pyom):
    """
    biharmonic mixing of temp and salinity,
    dissipation of dyn. Enthalpy is stored
    """
    # integer :: i,j,k,ks,is_,ie,js,je
    # real*8 :: aloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    # real*8 :: del2(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz),fxa

    is_ = is_pe-onx
    ie = ie_pe+onx
    js = js_pe-onx
    je = je_pe+onx

    fxa = math.sqrt(abs(K_hbi))

    for j in xrange(js,je): # j = js,je
        for i in xrange(is_,ie-1): # i = is_,ie-1
            flux_east[i,j,:] = -fxa*(temp[i+1,j,:,tau]-temp[i,j,:,tau])/(cost[j]*dxu[i])*maskU[i,j,:]

    for j in xrange(js,je-1): # j = js,je-1
        flux_north[:,j,:] = -fxa*(temp[:,j+1,:,tau]-temp[:,j,:,tau])/dyu[j]*maskV[:,j,:]*cosu[j]
    flux_east[ie,:,:] = 0.; flux_north[:,je,:] = 0.

    for j in xrange(js+1,je): # j = js+1,je
        for i in xrange(is_+1,ie): # i = is_+1,ie
            del2[i,j,:] = maskT[i,j,:] * (flux_east[i,j,:] - flux_east[i-1,j,:])/(cost[j]*dxt[i]) \
                                        +(flux_north[i,j,:] - flux_north[i,j-1,:])/(cost[j]*dyt[j])

    border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,del2)
    cyclic.setcyclic_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,del2)

    for j in xrange(js,je): # j = js,je
        for i in xrange(is_,ie-1): # i = is_,ie-1
            flux_east[i,j,:] = fxa*(del2[i+1,j,:]-del2[i,j,:])/(cost[j]*dxu[i])*maskU[i,j,:]

    for j in xrange(js,je-1): # j = js,je-1
        flux_north[:,j,:] = fxa*(del2(:,j+1,:)-del2[:,j,:])/dyu[j]*maskV[:,j,:]*cosu[j]

    flux_east[ie,:,:] = 0.; flux_north[:,je,:] = 0.

    # update tendency
    for j in xrange(js_pe-onx+1,je_pe+onx): # j = js_pe-onx+1,je_pe+onx
        for i in xrange(is_pe-onx+1,ie_pe+onx): # i = is_pe-onx+1,ie_pe+onx
            dtemp_hmix[i,j,:] = maskT[i,j,:] * (flux_east[i,j,:] - flux_east[i-1,j,:])/(cost[j]*dxt[i]) \
                                                 + (flux_north[i,j,:] - flux_north[i,j-1,:])/(cost[j]*dyt[j])

    temp[:,:,:,taup1] = temp[:,:,:,taup1] + dt_tracer * dtemp_hmix * maskT

    if pyom.enable_conserve_energy:
        # diagnose dissipation of dynamic enthalpy by hor. mixing of temperature
        for k in xrange(1,nz): # k = 1,nz
            for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
                for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
                    fxa = int_drhodT[i,j,k,tau]
                    aloc[i,j,k] = 0.5*grav/rho_0*((int_drhodT[i+1,j,k,tau]-fxa)*flux_east[i,j,k] \
                                     +(fxa-int_drhodT[i-1,j,k,tau])*flux_east[i-1,j,k]) / (dxt[i]*cost[j])  \
                                 +0.5*grav/rho_0*((int_drhodT(i,j+1,k,tau)-fxa)*flux_north[i,j,k] \
                                     +(fxa-int_drhodT[i,j-1,k,tau])*flux_north[i,j-1,k]) / (dyt[j]*cost[j])

        # dissipation interpolated on W-grid
        for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
            for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
                ks = kbot[i,j]
                if ks>0:
                    k = ks
                    P_diss_hmix[i,j,k] = 0.5*(aloc[i,j,k]+aloc[i,j,k+1]) + 0.5*aloc[i,j,k]*dzw[max(1,k-1)]/dzw[k]
                    for k in xrange(ks+1,nz-1): # k = ks+1,nz-1
                        P_diss_hmix[i,j,k] = 0.5*(aloc[i,j,k] + aloc[i,j,k+1])
                    k = nz
                    P_diss_hmix[i,j,k] = aloc[i,j,k]

    for j in xrange(js,je): # j = js,je
        for i in xrange(is_,ie-1): # i = is_,ie-1
            flux_east[i,j,:] = -fxa*(salt[i+1,j,:,tau]-salt[i,j,:,tau])/(cost[j]*dxu[i])*maskU[i,j,:]

    for j in xrange(js,je-1): # j = js,je-1
        flux_north[:,j,:] = -fxa*(salt[:,j+1,:,tau]-salt[:,j,:,tau])/dyu[j]*maskV[:,j,:]*cosu[j]

    flux_east[ie,:,:] = 0.; flux_north[:,je,:] = 0.

    for j in xrange(js+1,je): # j = js+1,je
        for i in xrange(is_+1,ie): # i = is_+1,ie
            del2[i,j,:] = maskT[i,j,:]* (flux_east[i,j,:] - flux_east[i-1,j,:])/(cost[j]*dxt[i]) \
                                        +(flux_north[i,j,:] - flux_north[i,j-1,:])/(cost[j]*dyt[j])

    border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,del2)
    cyclic.setcyclic_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,del2)

    for j in xrange(js,je): # j = js,je
        for i in xrange(is_,ie-1): # i = is_,ie-1
            flux_east[i,j,:] = fxa*(del2[i+1,j,:]-del2[i,j,:])/(cost[j]*dxu[i])*maskU[i,j,:]

    for j in xrange(js,je-1): # j = js,je-1
        flux_north[:,j,:] = fxa*(del2(:,j+1,:)-del2[:,j,:])/dyu[j]*maskV[:,j,:]*cosu[j]

    flux_east[ie,:,:] = 0.
    flux_north[:,je,:] = 0.

    # update tendency
    for j in xrange(js_pe-onx+1,je_pe+onx): # j = js_pe-onx+1,je_pe+onx
        for i in xrange(is_pe-onx+1,ie_pe+onx): # i = is_pe-onx+1,ie_pe+onx
            dsalt_hmix[i,j,:] = maskT[i,j,:] * (flux_east[i,j,:] - flux_east[i-1,j,:])/(cost[j]*dxt[i]) \
                                                +(flux_north[i,j,:] - flux_north[i,j-1,:])/(cost[j]*dyt[j])

    salt[:,:,:,taup1] = salt[:,:,:,taup1] + dt_tracer * dsalt_hmix * maskT

    if pyom.enable_conserve_energy:
        # diagnose dissipation of dynamic enthalpy by hor. mixing of salinity
        for k in xrange(1,nz): # k = 1,nz
            for j in xrange(js_pe-onx+1,je_pe+onx-1): # j = js_pe-onx+1,je_pe+onx-1
                for i in xrange(is_pe-onx+1,ie_pe+onx-1): # i = is_pe-onx+1,ie_pe+onx-1
                    fxa = int_drhodS[i,j,k,tau]
                    aloc[i,j,k] = 0.5*grav/rho_0*((int_drhodS[i+1,j,k,tau]-fxa)*flux_east[i,j,k] \
                                                  +(fxa-int_drhodS[i-1,j,k,tau])*flux_east[i-1,j,k]) /(dxt[i]*cost[j])  \
                                 +0.5*grav/rho_0*((int_drhodS(i,j+1,k,tau)-fxa)*flux_north[i,j,k] \
                                                  +(fxa-int_drhodS[i,j-1,k,tau])*flux_north[i,j-1,k]) /(dyt[j]*cost[j])

        # dissipation interpolated on W-grid
        for j in xrange(js_pe-onx,je_pe+onx): # j = js_pe-onx,je_pe+onx
            for i in xrange(is_pe-onx,ie_pe+onx): # i = is_pe-onx,ie_pe+onx
                ks = kbot[i,j]
                if ks>0:
                    k = ks
                    P_diss_hmix[i,j,k] = P_diss_hmix[i,j,k]+ \
                                         0.5*(aloc[i,j,k] + aloc[i,j,k+1]) + 0.5*aloc[i,j,k]*dzw[max(1,k-1)]/dzw[k]
                    for k in xrange(ks+1,nz-1): # k = ks+1,nz-1
                        P_diss_hmix[i,j,k] = P_diss_hmix[i,j,k] + 0.5*(aloc[i,j,k] + aloc[i,j,k+1])
                    k = nz
                    P_diss_hmix[i,j,k] = P_diss_hmix[i,j,k] + aloc[i,j,k]


def tempsalt_diffusion(pyom):
    """
    Diffusion of temp and salinity,
    dissipation of dyn. Enthalpy is stored
    """
    # integer :: i,j,k,ks
    # real*8 :: fxa
    # real*8 :: aloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)

    # horizontal diffusion of temperature
    for j in xrange(js_pe-onx,je_pe+onx): # j = js_pe-onx,je_pe+onx
        for i in xrange(is_pe-onx,ie_pe+onx-1): # i = is_pe-onx,ie_pe+onx-1
            flux_east[i,j,:] = K_h * (temp[i+1,j,:,tau] - temp[i,j,:,tau]) / (cost[j]*dxu[i])*maskU[i,j,:]

    flux_east(ie_pe+onx,:,:) = 0.
    for j in xrange(js_pe-onx,je_pe+onx-1): # j = js_pe-onx,je_pe+onx-1
        flux_north[:,j,:] = K_h * (temp[:,j+1,:,tau] - temp[:,j,:,tau]) / dyu[j]*maskV[:,j,:]*cosu[j]
    flux_north(:,je_pe+onx,:) = 0.

    if pyom.enable_hor_friction_cos_scaling:
        for j in xrange(js_pe-onx,je_pe+onx): # j = js_pe-onx,je_pe+onx
            flux_east[:,j,:] = flux_east[:,j,:]*cost[j]**pyom.hor_friction_cosPower
            flux_north[:,j,:] = flux_north[:,j,:]*cosu[j]**pyom.hor_friction_cosPower

    for j in xrange(js_pe-onx+1,je_pe+onx): # j = js_pe-onx+1,je_pe+onx
        for i in xrange(is_pe-onx+1,ie_pe+onx): # i = is_pe-onx+1,ie_pe+onx
            dtemp_hmix[i,j,:] = maskT[i,j,:]*((flux_east[i,j,:] - flux_east[i-1,j,:])/(cost[j]*dxt[i]) \
                                            +(flux_north[i,j,:] -flux_north[i,j-1,:])/(cost[j]*dyt[j]))

    temp[:,:,:,taup1] = temp[:,:,:,taup1] + dt_tracer * dtemp_hmix * maskT

    if pyom.enable_conserve_energy:
        # diagnose dissipation of dynamic enthalpy by hor. mixing of temperature
        for k in xrange(1,nz): # k = 1,nz
            for j in xrange(js_pe-onx+1,je_pe+onx-1): # j = js_pe-onx+1,je_pe+onx-1
                for i in xrange(is_pe-onx+1,ie_pe+onx-1): # i = is_pe-onx+1,ie_pe+onx-1
                    fxa = int_drhodT[i,j,k,tau]
                    aloc[i,j,k] = 0.5*grav/rho_0*((int_drhodT[i+1,j,k,tau]-fxa)*flux_east[i,j,k] \
                                                 +(fxa-int_drhodT[i-1,j,k,tau])*flux_east[i-1,j,k]) / (dxt[i]*cost[j])  \
                                + 0.5*grav/rho_0*((int_drhodT(i,j+1,k,tau)-fxa)*flux_north[i,j,k] \
                                                 +(fxa-int_drhodT[i,j-1,k,tau])*flux_north[i,j-1,k]) /(dyt[j]*cost[j])

        # dissipation interpolated on W-grid
        for j in xrange(js_pe-onx,je_pe+onx): # j = js_pe-onx,je_pe+onx
            for i in xrange(is_pe-onx,ie_pe+onx): # i = is_pe-onx,ie_pe+onx
                ks = kbot[i,j]
                if ks>0:
                    k = ks
                    P_diss_hmix[i,j,k] = 0.5*(aloc[i,j,k]+aloc[i,j,k+1]) + 0.5*aloc[i,j,k]*dzw[max(1,k-1)]/dzw[k]
                    for k in xrange(ks+1,nz-1): # k = ks+1,nz-1
                        P_diss_hmix[i,j,k] = 0.5*(aloc[i,j,k] + aloc[i,j,k+1])
                    k = nz
                    P_diss_hmix[i,j,k] = aloc[i,j,k]

    # horizontal diffusion of salinity
    for j in xrange(js_pe-onx,je_pe+onx): # j = js_pe-onx,je_pe+onx
        for i in xrange(is_pe-onx,ie_pe+onx-1): # i = is_pe-onx,ie_pe+onx-1
            flux_east[i,j,:] = K_h * (salt[i+1,j,:,tau] - salt[i,j,:,tau]) / (cost[j]*dxu[i])*maskU[i,j,:]

    flux_east(ie_pe+onx,:,:) = 0.
    for j in xrange(js_pe-onx,je_pe+onx-1): # j = js_pe-onx,je_pe+onx-1
        flux_north[:,j,:] = K_h * (salt[:,j+1,:,tau] - salt[:,j,:,tau]) / dyu[j]*maskV[:,j,:]*cosu[j]
    flux_north(:,je_pe+onx,:) = 0.

    if pyom.enable_hor_friction_cos_scaling:
        for j in xrange(js_pe-onx,je_pe+onx): # j = js_pe-onx,je_pe+onx
            flux_east[:,j,:] = flux_east[:,j,:]*cost[j]**pyom.hor_friction_cosPower
            flux_north[:,j,:] = flux_north[:,j,:]*cosu[j]**pyom.hor_friction_cosPower

    for j in xrange(js_pe-onx+1,je_pe+onx): # j = js_pe-onx+1,je_pe+onx
        for i in xrange(is_pe-onx+1,ie_pe+onx): # i = is_pe-onx+1,ie_pe+onx
            dsalt_hmix[i,j,:] = maskT[i,j,:] * ((flux_east[i,j,:] - flux_east[i-1,j,:]) / (cost[j]*dxt[i]) \
                                               + (flux_north[i,j,:] - flux_north[i,j-1,:]) / (cost[j]*dyt[j]))

    salt[:,:,:,taup1] = salt[:,:,:,taup1] + dt_tracer * dsalt_hmix * maskT

    if pyom.enable_conserve_energy:
        # diagnose dissipation of dynamic enthalpy by hor. mixing of salinity
        for k in xrange(1,nz): # k = 1,nz
            for j in xrange(js_pe-onx+1,je_pe+onx-1): # j = js_pe-onx+1,je_pe+onx-1
                for i in xrange(is_pe-onx+1,ie_pe+onx-1): # i = is_pe-onx+1,ie_pe+onx-1
                    fxa = int_drhodS[i,j,k,tau]
                        aloc[i,j,k] = 0.5*grav/rho_0*((int_drhodS[i+1,j,k,tau]-fxa)*flux_east[i,j,k] \
                                                     +(fxa-int_drhodS[i-1,j,k,tau])*flux_east[i-1,j,k]) / (dxt[i]*cost[j])  \
                                    + 0.5*grav/rho_0*((int_drhodS(i,j+1,k,tau)-fxa)*flux_north(i,j  ,k) \
                                                     +(fxa-int_drhodS[i,j-1,k,tau])*flux_north[i,j-1,k]) / (dyt[j]*cost[j])

        # dissipation interpolated on W-grid
        for j in xrange(js_pe-onx,je_pe+onx): # j = js_pe-onx,je_pe+onx
            for i in xrange(is_pe-onx,ie_pe+onx): # i = is_pe-onx,ie_pe+onx
                ks = kbot[i,j]
                if ks>0:
                    k = ks
                    P_diss_hmix[i,j,k] = P_diss_hmix[i,j,k]+ \
                                        0.5*(aloc[i,j,k]+aloc[i,j,k+1]) + 0.5*aloc[i,j,k]*dzw[max(1,k-1)]/dzw[k]
                    for k in xrange(ks+1,nz-1): # k = ks+1,nz-1
                        P_diss_hmix[i,j,k] = P_diss_hmix[i,j,k]+ 0.5*(aloc[i,j,k] + aloc[i,j,k+1])
                    k = nz
                    P_diss_hmix[i,j,k] = P_diss_hmix[i,j,k] + aloc[i,j,k]


def tempsalt_sources(pyom):
    """
    Sources of temp and salinity,
    effect on dyn. Enthalpy is stored
    """
    # integer :: i,j,k,ks
    # real*8 :: aloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)

    temp[:,:,:,taup1] = temp[:,:,:,taup1] + dt_tracer * temp_source * maskT
    salt[:,:,:,taup1] = salt[:,:,:,taup1] + dt_tracer * salt_source * maskT

    if pyom.enable_conserve_energy:
        # diagnose effect on dynamic enthalpy
        for k in xrange(1,nz): # k = 1,nz
            for j in xrange(js_pe-onx+1,je_pe+onx-1): # j = js_pe-onx+1,je_pe+onx-1
                for i in xrange(is_pe-onx+1,ie_pe+onx-1): # i = is_pe-onx+1,ie_pe+onx-1
                    aloc[i,j,k] = -grav/rho_0*maskT[i,j,k] \
                                    * (int_drhodT[i,j,k,tau]*temp_source[i,j,k]+int_drhodS[i,j,k,tau]*salt_source[i,j,k])

        # dissipation interpolated on W-grid
        for j in xrange(js_pe-onx,je_pe+onx): # j = js_pe-onx,je_pe+onx
            for i in xrange(is_pe-onx,ie_pe+onx): # i = is_pe-onx,ie_pe+onx
                ks = kbot[i,j]
                if ks>0:
                    k = ks
                    P_diss_sources[i,j,k] = 0.5*(aloc[i,j,k]+aloc[i,j,k+1]) + 0.5*aloc[i,j,k]*dzw[max(1,k-1)]/dzw[k]
                    for k in xrange(ks+1,nz-1): # k = ks+1,nz-1
                        P_diss_sources[i,j,k] = 0.5*(aloc[i,j,k] + aloc[i,j,k+1])
                    k = nz
                    P_diss_sources[i,j,k] = aloc[i,j,k]

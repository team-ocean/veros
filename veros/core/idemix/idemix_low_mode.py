from .. import advection
from ... import veros_method

@veros_method
def integrate_idemix_M2(veros):
    """
    integrate M2 wave compartment in time
    """
    # real*8 :: advp_fe(veros.is_pe-veros.onx:veros.ie_pe+veros.onx,veros.js_pe-veros.onx:veros.je_pe+veros.onx,veros.np)
    # real*8 :: advp_fn(veros.is_pe-veros.onx:veros.ie_pe+veros.onx,veros.js_pe-veros.onx:veros.je_pe+veros.onx,veros.np)
    # real*8 :: advp_ft(veros.is_pe-veros.onx:veros.ie_pe+veros.onx,veros.js_pe-veros.onx:veros.je_pe+veros.onx,veros.np)
    advp_fe = np.zeros((veros.nx+4, veros.ny+4, veros.np))
    advp_fn = np.zeros((veros.nx+4, veros.ny+4, veros.np))
    advp_ft = np.zeros((veros.nx+4, veros.ny+4, veros.np))

    advection.adv_flux_superbee_spectral(advp_fe,advp_fn,advp_ft,veros.E_M2,u_M2,v_M2,w_M2)
    advection.reflect_flux(advp_fe,advp_fn)
    for k in xrange(1,veros.np-1): # k = 2,np-1
        for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
            for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
                veros.dE_M2p[i,j,k,veros.tau] = veros.maskTp[i,j,k] * \
                        (-(advp_fe[i,j,k] - advp_fe[i-1,j,k]) / (veros.cost[j] * veros.dxt[i]) \
                         -(advp_fn[i,j,k] - advp_fn[i,j-1,k]) / (veros.cost[j] * veros.dyt[j]) \
                         -(advp_ft[i,j,k] - advp_ft[i,j,k-1]) / veros.dphit[k])
    for k in xrange(1,veros.np-1): # k = 2,np-1
        veros.E_M2[:,:,k,veros.taup1] = veros.E_M2[:,:,k,veros.tau] + veros.dt_tracer*(forc_M2[:,:,k] - tau_M2*veros.E_M2[:,:,k,veros.tau] \
                                        + (1.5 + veros.AB_eps) * veros.dE_M2p[:,:,k,veros.tau] - (0.5 + veros.AB_eps) * veros.dE_M2p[:,:,k,veros.taum1])

@veros_method
def integrate_idemix_niw(veros):
    """
    integrate NIW wave compartment in time
    """
    # real*8 :: advp_fe(veros.is_pe-veros.onx:veros.ie_pe+veros.onx,veros.js_pe-veros.onx:veros.je_pe+veros.onx,veros.np)
    # real*8 :: advp_fn(veros.is_pe-veros.onx:veros.ie_pe+veros.onx,veros.js_pe-veros.onx:veros.je_pe+veros.onx,veros.np)
    # real*8 :: advp_ft(veros.is_pe-veros.onx:veros.ie_pe+veros.onx,veros.js_pe-veros.onx:veros.je_pe+veros.onx,veros.np)
    advp_fe = np.zeros((veros.nx+4, veros.ny+4, veros.np))
    advp_fn = np.zeros((veros.nx+4, veros.ny+4, veros.np))
    advp_ft = np.zeros((veros.nx+4, veros.ny+4, veros.np))

    advection.adv_flux_superbee_spectral(advp_fe,advp_fn,advp_ft,veros.E_niw,u_niw,v_niw,w_niw)
    advection.reflect_flux(advp_fe,advp_fn)
    for k in xrange(1,veros.np-1): # k = 2,np-1
        for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
            for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
                dE_niwp[i,j,k,veros.tau] = veros.maskTp[i,j,k] * \
                         (-(advp_fe[i,j,k]- advp_fe[i-1,j,k]) / (veros.cost[j] * veros.dxt[i]) \
                         - (advp_fn[i,j,k]- advp_fn[i,j-1,k]) / (veros.cost[j] * veros.dyt[j]) \
                         - (advp_ft[i,j,k]- advp_ft[i,j,k-1]) / veros.dphit[k])

    for k in xrange(1,veros.np-1): # k = 2,np-1
        veros.E_niw[:,:,k,veros.taup1] = veros.E_niw[:,:,k,veros.tau] + veros.dt_tracer * (forc_niw[:,:,k] - tau_niw[:,:] * veros.E_niw[:,:,k,veros.tau] \
                                                                                     + (1.5 + veros.AB_eps) * dE_niwp[:,:,k,veros.tau] \
                                                                                     - (0.5 + veros.AB_eps) * dE_niwp[:,:,k,veros.taum1])

@veros_method
def wave_interaction(veros):
    """
    interaction of wave components
    """
    # real*8 :: fmin,cont(veros.is_pe-veros.onx:veros.ie_pe+veros.onx,veros.js_pe-veros.onx:veros.je_pe+veros.onx)
    cont = np.zeros((np.nx+4, np.ny+4))

    if veros.enable_idemix:
        cont[...] = 0.0
        for k in xrange(veros.nz): # k = 1,nz
            for j in xrange(veros.js_pe-veros.onx,veros.je_pe+veros.onx): # j = js_pe-onx,je_pe+onx
                for i in xrange(veros.is_pe-veros.onx,veros.ie_pe+veros.onx): # i = is_pe-onx,ie_pe+onx
                    cont[i,j] += veros.E_iw[i,j,k,veros.tau] * veros.dzt[k] * veros.maskT[i,j,k]

    if veros.enable_idemix_M2:
        # integrate M2 energy over angle
        veros.E_M2_int[...] = 0.0
        for k in xrange(1,veros.np-1): # k = 2,np-1
            for j in xrange(veros.js_pe-veros.onx,veros.je_pe+veros.onx): # j = js_pe-onx,je_pe+onx
                for i in xrange(veros.is_pe-veros.onx,veros.ie_pe+veros.onx): # i = is_pe-onx,ie_pe+onx
                    veros.E_M2_int[i,j] += veros.E_M2[i,j,k,veros.tau] * veros.dphit[k] * veros.maskTp[i,j,k]

    if enable_idemix_niw:
        # integrate niw energy over angle
        veros.E_niw_int[...] = 0
        for k in xrange(1,veros.np-1): # k = 2,np-1
            for j in xrange(veros.js_pe-veros.onx,veros.je_pe+veros.onx): # j = js_pe-onx,je_pe+onx
                for i in xrange(veros.is_pe-veros.onx,veros.ie_pe+veros.onx): # i = is_pe-onx,ie_pe+onx
                    E_niw_int[i,j] += veros.E_niw[i,j,k,veros.tau] * veros.dphit[k] * veros.maskTp[i,j,k]

    if veros.enable_idemix_M2 and veros.enable_idemix:
        # update M2 energy: interaction of M2 and continuum
        for k in xrange(1,veros.np-1): # k = 2,np-1
            for j in xrange(veros.js_pe-veros.onx,veros.je_pe+veros.onx): # j = js_pe-onx,je_pe+onx
                for i in xrange(veros.is_pe-veros.onx,veros.ie_pe+veros.onx): # i = is_pe-onx,ie_pe+onx
                    fmin = min(0.5 / veros.dt_tracer, veros.alpha_M2_cont[i,j] * cont[i,j]) # flux limiter
                    veros.M2_psi_diss[i,j,k] = fmin * veros.E_M2[i,j,k,veros.tau] * veros.maskTp[i,j,k]
                    veros.E_M2[i,j,k,veros.taup1] = veros.E_M2[i,j,k,veros.taup1] - veros.dt_tracer * veros.M2_psi_diss[i,j,k]

    if veros.enable_idemix and veros.enable_idemix_M2:
        for k in xrange(veros.nz): # k = 1,nz
            for j in xrange(veros.js_pe-veros.onx,veros.je_pe+veros.onx): # j = js_pe-onx,je_pe+onx
                for i in xrange(veros.is_pe-veros.onx,veros.ie_pe+veros.onx): # i = is_pe-onx,ie_pe+onx
                    fmin = min(0.5 / veros.dt_tracer, veros.alpha_M2_cont[i,j] * veros.cont[i,j]) # flux limiter
                    veros.E_iw[i,j,k,veros.taup1] += veros.dt_tracer * veros.tau_M2[i,j] * E_M2_int[i,j] * E_struct_M2[i,j,k] * veros.maskT[i,j,k] \
                                                 + veros.dt_tracer * fmin * veros.E_M2_int[i,j] * E_struct_M2[i,j,k] * veros.maskT[i,j,k]

    if veros.enable_idemix and enable_idemix_niw:
        for k in xrange(veros.nz): # k = 1,nz
            for j in xrange(veros.js_pe-veros.onx,veros.je_pe+veros.onx): # j = js_pe-onx,je_pe+onx
                for i in xrange(veros.is_pe-veros.onx,veros.ie_pe+veros.onx): # i = is_pe-onx,ie_pe+onx
                    veros.E_iw[i,j,k,veros.taup1] += veros.dt_tracer * veros.tau_niw[i,j] * veros.E_niw_int[i,j] * veros.E_struct_niw[i,j,k] * veros.maskT[i,j,k]

import numpy as np

from climate.pyom import advection


def integrate_idemix_M2(pyom):
    """
    integrate M2 wave compartment in time
    """
    # real*8 :: advp_fe(pyom.is_pe-pyom.onx:pyom.ie_pe+pyom.onx,pyom.js_pe-pyom.onx:pyom.je_pe+pyom.onx,pyom.np)
    # real*8 :: advp_fn(pyom.is_pe-pyom.onx:pyom.ie_pe+pyom.onx,pyom.js_pe-pyom.onx:pyom.je_pe+pyom.onx,pyom.np)
    # real*8 :: advp_ft(pyom.is_pe-pyom.onx:pyom.ie_pe+pyom.onx,pyom.js_pe-pyom.onx:pyom.je_pe+pyom.onx,pyom.np)
    advp_fe = np.zeros((pyom.nx+4, pyom.ny+4, pyom.np))
    advp_fn = np.zeros((pyom.nx+4, pyom.ny+4, pyom.np))
    advp_ft = np.zeros((pyom.nx+4, pyom.ny+4, pyom.np))

    advection.adv_flux_superbee_spectral(advp_fe,advp_fn,advp_ft,pyom.E_M2,u_M2,v_M2,w_M2)
    advection.reflect_flux(advp_fe,advp_fn)
    for k in xrange(1,pyom.np-1): # k = 2,np-1
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                pyom.dE_M2p[i,j,k,pyom.tau] = pyom.maskTp[i,j,k] * \
                        (-(advp_fe[i,j,k] - advp_fe[i-1,j,k]) / (pyom.cost[j] * pyom.dxt[i]) \
                         -(advp_fn[i,j,k] - advp_fn[i,j-1,k]) / (pyom.cost[j] * pyom.dyt[j]) \
                         -(advp_ft[i,j,k] - advp_ft[i,j,k-1]) / pyom.dphit[k])
    for k in xrange(1,pyom.np-1): # k = 2,np-1
        pyom.E_M2[:,:,k,pyom.taup1] = pyom.E_M2[:,:,k,pyom.tau] + pyom.dt_tracer*(forc_M2[:,:,k] - tau_M2*pyom.E_M2[:,:,k,pyom.tau] \
                                        + (1.5 + pyom.AB_eps) * pyom.dE_M2p[:,:,k,pyom.tau] - (0.5 + pyom.AB_eps) * pyom.dE_M2p[:,:,k,pyom.taum1])


def integrate_idemix_niw(pyom):
    """
    integrate NIW wave compartment in time
    """
    # real*8 :: advp_fe(pyom.is_pe-pyom.onx:pyom.ie_pe+pyom.onx,pyom.js_pe-pyom.onx:pyom.je_pe+pyom.onx,pyom.np)
    # real*8 :: advp_fn(pyom.is_pe-pyom.onx:pyom.ie_pe+pyom.onx,pyom.js_pe-pyom.onx:pyom.je_pe+pyom.onx,pyom.np)
    # real*8 :: advp_ft(pyom.is_pe-pyom.onx:pyom.ie_pe+pyom.onx,pyom.js_pe-pyom.onx:pyom.je_pe+pyom.onx,pyom.np)
    advp_fe = np.zeros((pyom.nx+4, pyom.ny+4, pyom.np))
    advp_fn = np.zeros((pyom.nx+4, pyom.ny+4, pyom.np))
    advp_ft = np.zeros((pyom.nx+4, pyom.ny+4, pyom.np))

    advection.adv_flux_superbee_spectral(advp_fe,advp_fn,advp_ft,pyom.E_niw,u_niw,v_niw,w_niw)
    advection.reflect_flux(advp_fe,advp_fn)
    for k in xrange(1,pyom.np-1): # k = 2,np-1
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                dE_niwp[i,j,k,pyom.tau] = pyom.maskTp[i,j,k] * \
                         (-(advp_fe[i,j,k]- advp_fe[i-1,j,k]) / (pyom.cost[j] * pyom.dxt[i]) \
                         - (advp_fn[i,j,k]- advp_fn[i,j-1,k]) / (pyom.cost[j] * pyom.dyt[j]) \
                         - (advp_ft[i,j,k]- advp_ft[i,j,k-1]) / pyom.dphit[k])

    for k in xrange(1,pyom.np-1): # k = 2,np-1
        pyom.E_niw[:,:,k,pyom.taup1] = pyom.E_niw[:,:,k,pyom.tau] + pyom.dt_tracer * (forc_niw[:,:,k] - tau_niw[:,:] * pyom.E_niw[:,:,k,pyom.tau] \
                                                                                     + (1.5 + pyom.AB_eps) * dE_niwp[:,:,k,pyom.tau] \
                                                                                     - (0.5 + pyom.AB_eps) * dE_niwp[:,:,k,pyom.taum1])


def wave_interaction(pyom):
    """
    interaction of wave components
    """
    # real*8 :: fmin,cont(pyom.is_pe-pyom.onx:pyom.ie_pe+pyom.onx,pyom.js_pe-pyom.onx:pyom.je_pe+pyom.onx)
    cont = np.zeros((np.nx+4, np.ny+4))

    if pyom.enable_idemix:
        cont[...] = 0.0
        for k in xrange(pyom.nz): # k = 1,nz
            for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): # j = js_pe-onx,je_pe+onx
                for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx): # i = is_pe-onx,ie_pe+onx
                    cont[i,j] += pyom.E_iw[i,j,k,pyom.tau] * pyom.dzt[k] * pyom.maskT[i,j,k]

    if pyom.enable_idemix_M2:
        # integrate M2 energy over angle
        pyom.E_M2_int[...] = 0.0
        for k in xrange(1,pyom.np-1): # k = 2,np-1
            for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): # j = js_pe-onx,je_pe+onx
                for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx): # i = is_pe-onx,ie_pe+onx
                    pyom.E_M2_int[i,j] += pyom.E_M2[i,j,k,pyom.tau] * pyom.dphit[k] * pyom.maskTp[i,j,k]

    if enable_idemix_niw:
        # integrate niw energy over angle
        pyom.E_niw_int[...] = 0
        for k in xrange(1,pyom.np-1): # k = 2,np-1
            for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): # j = js_pe-onx,je_pe+onx
                for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx): # i = is_pe-onx,ie_pe+onx
                    E_niw_int[i,j] += pyom.E_niw[i,j,k,pyom.tau] * pyom.dphit[k] * pyom.maskTp[i,j,k]

    if pyom.enable_idemix_M2 and pyom.enable_idemix:
        # update M2 energy: interaction of M2 and continuum
        for k in xrange(1,pyom.np-1): # k = 2,np-1
            for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): # j = js_pe-onx,je_pe+onx
                for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx): # i = is_pe-onx,ie_pe+onx
                    fmin = min(0.5 / pyom.dt_tracer, pyom.alpha_M2_cont[i,j] * cont[i,j]) # flux limiter
                    pyom.M2_psi_diss[i,j,k] = fmin * pyom.E_M2[i,j,k,pyom.tau] * pyom.maskTp[i,j,k]
                    pyom.E_M2[i,j,k,pyom.taup1] = pyom.E_M2[i,j,k,pyom.taup1] - pyom.dt_tracer * pyom.M2_psi_diss[i,j,k]

    if pyom.enable_idemix and pyom.enable_idemix_M2:
        for k in xrange(pyom.nz): # k = 1,nz
            for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): # j = js_pe-onx,je_pe+onx
                for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx): # i = is_pe-onx,ie_pe+onx
                    fmin = min(0.5 / pyom.dt_tracer, pyom.alpha_M2_cont[i,j] * pyom.cont[i,j]) # flux limiter
                    pyom.E_iw[i,j,k,pyom.taup1] += pyom.dt_tracer * pyom.tau_M2[i,j] * E_M2_int[i,j] * E_struct_M2[i,j,k] * pyom.maskT[i,j,k] \
                                                 + pyom.dt_tracer * fmin * pyom.E_M2_int[i,j] * E_struct_M2[i,j,k] * pyom.maskT[i,j,k]

    if pyom.enable_idemix and enable_idemix_niw:
        for k in xrange(pyom.nz): # k = 1,nz
            for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): # j = js_pe-onx,je_pe+onx
                for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx): # i = is_pe-onx,ie_pe+onx
                    pyom.E_iw[i,j,k,pyom.taup1] += pyom.dt_tracer * pyom.tau_niw[i,j] * pyom.E_niw_int[i,j] * pyom.E_struct_niw[i,j,k] * pyom.maskT[i,j,k]

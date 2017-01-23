import numpy as np

from climate.pyom import advection, numerics

def set_idemix_parameter(pyom):
    """
    set main IDEMIX parameter
    """
    for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): # j = js_pe-onx,je_pe+onx
        for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx): # i = is_pe-onx,ie_pe+onx
            bN0 = 0.0
            for k in xrange(pyom.nz-1): # k = 1,nz-1
                bN0 = bN0 + max(0., pyom.Nsqr[i,j,k,pyom.tau])**0.5 * pyom.dzw[k] * pyom.maskW[i,j,k]
            bN0 = bN0 + max(0., pyom.Nsqr(i,j,pyom.nz,pyom.tau))**0.5 * 0.5 * pyom.dzw[pyom.nz] * pyom.maskW[i,j,pyom.nz]
            for k in xrange(pyom.nz): # k = 1,nz
                fxa = max(0., pyom.Nsqr[i,j,k,pyom.tau])**0.5 / (1e-22 + abs(pyom.coriolis_t(i,j)))
                cstar = max(1e-2, bN0 / (pyom.pi*pyom.jstar))
                c0[i,j,k] = max(0., gamma * cstar * gofx2(fxa) * pyom.maskW[i,j,k])
                v0[i,j,k] = max(0., gamma * cstar * hofx1(fxa) * pyom.maskW[i,j,k])
                pyom.alpha_c[i,j,k] = max(1e-4, mu0*np.arccosh(max(1.,fxa))*abs(pyom.coriolis_t(i,j))/cstar**2) * pyom.maskW[i,j,k]


def integrate_idemix(pyom):
    """
    integrate idemix on W grid
    """
    # real*8 :: a_tri(pyom.nz),b_tri(pyom.nz),c_tri(pyom.nz),d_tri(pyom.nz),delta(pyom.nz)
    # real*8 :: forc(pyom.is_pe-pyom.onx:pyom.ie_pe+pyom.onx,pyom.js_pe-pyom.onx:pyom.je_pe+pyom.onx,pyom.nz)
    # real*8 :: maxE_iw(pyom.is_pe-pyom.onx:pyom.ie_pe+pyom.onx,pyom.js_pe-pyom.onx:pyom.je_pe+pyom.onx,pyom.nz)
    # real*8 :: a_loc(pyom.is_pe-pyom.onx:pyom.ie_pe+pyom.onx,pyom.js_pe-pyom.onx:pyom.je_pe+pyom.onx)
    a_tri = np.zeros(pyom.nz)
    b_tri = np.zeros(pyom.nz)
    c_tri = np.zeros(pyom.nz)
    d_tri = np.zeros(pyom.nz)
    forc = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    maxE_iw = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    a_loc = np.zeros((pyom.nx+4, pyom.ny+4))

    ke = pyom.nz-1

    """
    forcing by EKE dissipation
    """
    if pyom.enable_eke:
        forc = pyom.eke_diss_iw
    else: # shortcut without EKE model
        if pyom.enable_store_cabbeling_heat:
            forc = pyom.K_diss_gm + pyom.K_diss_h - pyom.P_diss_skew - pyom.P_diss_hmix  - pyom.P_diss_iso
        else:
            forc = pyom.K_diss_gm + pyom.K_diss_h - pyom.P_diss_skew

    if pyom.enable_eke and (pyom.enable_eke_diss_bottom or pyom.enable_eke_diss_surfbot):
        """
        vertically integrate EKE dissipation and inject at bottom and/or surface
        """
        a_loc[...] = 0.
        for k in xrange(pyom.nz-1): # k = 1,nz-1
            a_loc += pyom.dzw[k] * forc[:,:,k] * pyom.maskW[:,:,k]
        k = pyom.nz-1
        a_loc = a_loc + 0.5*pyom.dzw(k)*forc[:,:,k]*pyom.maskW[:,:,k]
        forc[...] = 0.
        if pyom.enable_eke_diss_bottom:
            for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
                for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                    ks = max(0,pyom.kbot[i,j]-1)
                    forc[i,j,ks] = a_loc[i,j] / pyom.dzw[ks]
        else:
            for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
                for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                    ks = max(0,pyom.kbot[i,j]-1)
                    forc[i,j,ks] = pyom.eke_diss_surfbot_frac * a_loc[i,j] / pyom.dzw[ks]
                    forc[i,j,ke] = (1.-pyom.eke_diss_surfbot_frac) * a_loc[i,j] / (0.5*pyom.dzw[ke])

    """
    forcing by bottom friction
    """
    if not pyom.enable_store_bottom_friction_tke:
        forc += pyom.K_diss_bot

    """
    prevent negative dissipation of IW energy
    """
    maxE_iw[...] = np.maximum(0., pyom.E_iw[:,:,:,pyom.tau])

    """
    vertical diffusion and dissipation is solved implicitely
    """
    for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
        for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
            ks = pyom.kbot[i,j] - 1
            if ks >= 0:
                for k in xrange(ks,ke-1): # k = ks,ke-1
                    delta[k] = pyom.dt_tracer * tau_v / pyom.dzt[k+1] * 0.5 * (c0[i,j,k]+c0[i,j,k+1])
                delta[ke] = 0.0
                for k in xrange(ks+1,ke-1): # k = ks+1,ke-1
                    a_tri[k] = -delta[k-1]*c0[i,j,k-1]/pyom.dzw[k]
                a_tri[ks] = 0.0
                a_tri[ke] = -delta[ke-1] / (0.5*pyom.dzw[ke]) * c0[i,j,ke-1]
                for k in xrange(ks+1,ke-1): # k = ks+1,ke-1
                    b_tri[k] = 1 + delta[k] * c0[i,j,k] / pyom.dzw[k] + delta[k-1] * c0[i,j,k] / pyom.dzw[k] \
                                 + pyom.dt_tracer * pyom.alpha_c[i,j,k] * maxE_iw[i,j,k]
                b_tri[ke] = 1 + delta[ke-1] / (0.5*pyom.dzw[ke]) * c0[i,j,ke] + pyom.dt_tracer * pyom.alpha_c[i,j,ke] * maxE_iw[i,j,ke]
                b_tri[ks] = 1 + delta[ks] / pyom.dzw[ks] * c0[i,j,ks] + pyom.dt_tracer * pyom.alpha_c[i,j,ks] * maxE_iw[i,j,ks]
                for k in xrange(ks,ke-1): # k = ks,ke-1
                    c_tri[k] = -delta[k] / pyom.dzw[k] * c0[i,j,k+1]
                c_tri[ke] = 0.0
                d_tri[ks:] = pyom.E_iw[i,j,ks:,pyom.tau] + pyom.dt_tracer * forc[i,j,ks:]
                d_tri[ks] += pyom.dt_tracer * pyom.forc_iw_bottom[i,j] / pyom.dzw[ks]
                d_tri[ke] += pyom.dt_tracer * pyom.forc_iw_surface[i,j] / (0.5*pyom.dzw[ke])
                pyom.E_iw[i,j,ks:,taup1] = numerics.solve_tridiag(a_tri[ks:],b_tri[ks:],c_tri[ks:],d_tri[ks:])

    """
    store IW dissipation
    """
    pyom.iw_diss = pyom.alpha_c * maxE_iw[:,:,:] * pyom.E_iw[:,:,:,taup1]

    """
    add tendency due to lateral diffusion
    """
    if pyom.enable_idemix_hor_diffusion:
        for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): # j = js_pe-onx,je_pe+onx
            for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx-1): # i = is_pe-onx,ie_pe+onx-1
                pyom.flux_east[i,j,:] = tau_h * 0.5 * (v0[i+1,j,:] + v0[i,j,:]) \
                                        * (v0[i+1,j,:] * pyom.E_iw[i+1,j,:,pyom.tau] - v0[i,j,:] * pyom.E_iw[i,j,:,pyom.tau]) \
                                        / (pyom.cost[j] * dxu[i]) * pyom.maskU[i,j,:]
        pyom.flux_east[pyom.ie_pe-pyom.onx,:,:] = 0. # NOTE: should this really be ie_pe-onx instead of ie_pe+onx?
        for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx-1): # j = js_pe-onx,je_pe+onx-1
            pyom.flux_north[:,j,:] = tau_h * 0.5 * (v0[:,j+1,:] + v0[:,j,:]) \
                                     * (v0[:,j+1,:] * pyom.E_iw[:,j+1,:,pyom.tau] - v0[:,j,:] * pyom.E_iw[:,j,:,pyom.tau]) \
                                     / dyu[j] * pyom.maskV[:,j,:] * pyom.cosu[j]
        pyom.flux_north[:,pyom.je_pe+pyom.onx-1,:] = 0.
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                pyom.E_iw(i,j,:,taup1) += pyom.dt_tracer*pyom.maskW[i,j,:] \
                                        * ((pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:])/(pyom.cost[j]*pyom.dxt[i]) \
                                        + (pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:])/(pyom.cost[j]*pyom.dyt[j]))

    """
    add tendency due to advection
    """
    if pyom.enable_idemix_superbee_advection:
        advection.adv_flux_superbee_wgrid(pyom.flux_east,pyom.flux_north,pyom.flux_top,pyom.E_iw[:,:,:,pyom.tau],pyom)

    if pyom.enable_idemix_upwind_advection:
        advection.adv_flux_upwind_wgrid(pyom.flux_east,pyom.flux_north,pyom.flux_top,pyom.E_iw[:,:,:,pyom.tau],pyom)

    if pyom.enable_idemix_superbee_advection or pyom.enable_idemix_upwind_advection:
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                dE_iw[i,j,:,pyom.tau] = pyom.maskW[i,j,:] * (-(pyom.flux_east[i,j,:] -  pyom.flux_east[i-1,j,:]) / (pyom.cost[j] * pyom.dxt[i]) \
                                                             -(pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:]) / (pyom.cost[j] * pyom.dyt[j]))
        k = 0
        dE_iw[:,:,k,pyom.tau] = dE_iw[:,:,k,pyom.tau] - pyom.flux_top[:,:,k] / pyom.dzw[k]
        for k in xrange(1,pyom.nz-1): # k = 2,nz-1
            dE_iw[:,:,k,pyom.tau] = dE_iw[:,:,k,pyom.tau] - (pyom.flux_top[:,:,k] - pyom.flux_top[:,:,k-1]) / pyom.dzw[k]
        k = pyom.nz - 1
        dE_iw[:,:,k,pyom.tau] = dE_iw[:,:,k,pyom.tau] - (pyom.flux_top[:,:,k] - pyom.flux_top[:,:,k-1]) / (0.5*pyom.dzw[k])
        """
        Adam Bashforth time stepping
        """
        pyom.E_iw[:,:,:,taup1] += pyom.dt_tracer * ((1.5 + pyom.AB_eps) * dE_iw[:,:,:,pyom.tau] \
                                                  - (0.5 + pyom.AB_eps) * dE_iw[:,:,:,pyom.taum1])


def gofx2(x,pyom):
    """
    a function g(x)
    """
    x = max(3.,x)
    c = 1.-(2./pyom.pi) * np.arcsin(1./x)
    return 2. / pyom.pi / c * 0.9 * x**(-2./3.) * (1 - np.exp(-x/4.3))


def hofx1(x,pyom):
    """
    a function h(x)
    """
    return (2. / pyom.pi) / (1. - (2. / pyom.pi) * np.arcsin(1./x)) * (x-1.) / (x+1.)

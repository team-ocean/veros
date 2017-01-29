import numpy as np
import math

from climate.pyom import utilities, advection

def init_eke(pyom):
    """
    Initialize EKE
    """
    if pyom.enable_eke_leewave_dissipation:
        pyom.hrms_k0[...] = np.maximum(pyom.eke_hrms_k0_min, 2 / pyom.pi * pyom.eke_topo_hrms**2 / np.maximum(1e-12, pyom.eke_topo_lam)**1.5)


def set_eke_diffusivities(pyom):
    """
    set skew diffusivity K_gm and isopycnal diffusivity K_iso
    set also vertical viscosity if TEM formalism is chosen
    """
    C_rossby = np.zeros((pyom.nx+4, pyom.ny+4))

    if pyom.enable_eke:
        """
        calculate Rossby radius as minimum of mid-latitude and equatorial R. rad.
        """
        for k in xrange(pyom.nz): # k = 1,nz
            C_Rossby[:,:] += np.sqrt(np.maximum(0.,pyom.Nsqr[:,:,k,pyom.tau])) * pyom.dzw[k] * pyom.maskW[:,:,k] / pyom.pi
        for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): # j = js_pe-onx,je_pe+onx
            for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx): # i = is_pe-onx,ie_pe+onx
                pyom.L_Rossby[i,j] = np.minimum(C_Rossby[i,j] / np.maximum(np.abs(pyom.coriolis_t[i,j]), 1e-16), \
                                                  np.sqrt(C_Rossby[i,j] / np.maximum(2 * pyom.beta[i,j], 1e-16)))
        """
        calculate vertical viscosity and skew diffusivity
        """
        pyom.sqrteke = np.sqrt(np.maximum(0.,pyom.eke[:,:,:,pyom.tau]))
        for k in xrange(pyom.nz): # k = 1,nz
            pyom.L_Rhines[:,:,k] = np.sqrt(pyom.sqrteke[:,:,k] / np.maximum(pyom.beta, 1e-16))
            pyom.eke_len[:,:,k] = np.maximum(pyom.eke_lmin, np.minimum(pyom.eke_cross * pyom.L_Rossby, pyom.eke_crhin * pyom.L_Rhines[:,:,k]))
        pyom.K_gm = np.minimum(pyom.eke_k_max, pyom.eke_c_k * pyom.eke_len * pyom.sqrteke)
    else: # enable_eke
        """
        use fixed GM diffusivity
        """
        pyom.K_gm = pyom.K_gm_0

    if pyom.enable_TEM_friction:
        for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): # j = js_pe-onx,je_pe+onx
            for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx): # i = is_pe-onx,ie_pe+onx
                pyom.kappa_gm[i,j,:] = pyom.K_gm[i,j,:] * np.minimum(0.01, pyom.coriolis_t[i,j]**2 \
                                       / np.maximum(1e-9, pyom.Nsqr[i,j,:,pyom.tau])) * pyom.maskW[i,j,:]

    if pyom.enable_eke and pyom.enable_eke_isopycnal_diffusion:
        pyom.K_iso = pyom.K_gm
    else:
        pyom.K_iso = pyom.K_iso_0 # always constant


def integrate_eke(pyom):
    """
    integrate EKE equation on W grid
    """
    # real*8 :: a_tri(nz),b_tri(nz),c_tri(nz),d_tri(nz),delta(nz)
    # real*8 :: forc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz),fxa,uz,vz,Ri
    # real*8 :: c_int(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    # real*8 :: a_loc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    # real*8 :: b_loc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    a_loc = np.zeros((pyom.nx+4, pyom.ny+4))
    b_loc = np.zeros((pyom.nx+4, pyom.ny+4))
    c_int = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    forc = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    """
    forcing by dissipation by lateral friction and GM using TRM formalism or skew diffusion
    """
    forc = pyom.K_diss_h + pyom.K_diss_gm  - pyom.P_diss_skew

    """
    store transfer due to isopycnal and horizontal mixing from dyn. enthalpy
    by non-linear eq.of state either to EKE or to heat
    """
    if not pyom.enable_store_cabbeling_heat:
        forc += - pyom.P_diss_hmix - pyom.P_diss_iso

    """
    coefficient for dissipation of EKE:
    by lee wave generation, Ri-dependent interior loss of balance and bottom friction
    """
    if pyom.enable_eke_leewave_dissipation:
        """
        by lee wave generation
        """
        c_lee = 0
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                k = pyom.kbot[i,j] - 1
                if k >= 0 and k < pyom.nz-1: # could be surface: factor 0.5
                    fxa = max(0,pyom.Nsqr[i,j,k,pyom.tau])**0.25
                    fxa = fxa * (1.5 * fxa / np.sqrt(np.maximum(1e-6, np.abs(coriolis_t[i,j])))-2)
                    pyom.c_lee[i,j] = pyom.c_lee0 * pyom.hrms_k0[i,j] * np.sqrt(pyom.sqrteke[i,j,k])  * max(0, fxa) / pyom.dzw[k]
        """
        Ri-dependent dissipation by interior loss of balance
        """
        pyom.c_Ri_diss = 0
        for k in xrange(pyom.nz-1): # k = 1,nz-1
            for j in xrange(pyom.js_pe-pyom.onx+1,pyom.je_pe+pyom.onx): # j = js_pe-onx+1,je_pe+onx
                for i in xrange(pyom.is_pe-pyom.onx+1,pyom.ie_pe+pyom.onx): # i = is_pe-onx+1,ie_pe+onx
                    pyom.uz = (((pyom.u[i,j,k+1,pyom.tau] - pyom.u[i,j,k,pyom.tau]) / pyom.dzt[k] * pyom.maskU[i,j,k])**2 \
                                + ((pyom.u[i-1,j,k+1,pyom.tau] - pyom.u[i-1,j,k,pyom.tau]) / pyom.dzt[k] * pyom.maskU[i-1,j,k])**2) \
                                / (pyom.maskU[i,j,k] + pyom.maskU[i-1,j,k] + 1e-18)
                    pyom.vz = (((pyom.v[i,j,k+1,pyom.tau] - pyom.v[i,j,k,pyom.tau]) / pyom.dzt[k] * pyom.maskV[i,j,k])**2 \
                                + ((pyom.v[i,j-1,k+1,pyom.tau] - pyom.v[i,j-1,k,pyom.tau]) / pyom.dzt[k] * pyom.maskV[i,j-1,k])**2) \
                                / (pyom.maskV[i,j,k] + pyom.maskV[i,j-1,k] + 1e-18)
                    pyom.Ri = max(1e-8, pyom.Nsqr[i,j,k,pyom.tau]) / (pyom.uz + pyom.vz + 1e-18)
                    fxa = 1 - 0.5 * (1. + np.tanh((pyom.Ri - pyom.eke_Ri0) / pyom.eke_Ri1))
                    pyom.c_Ri_diss[i,j,k] = pyom.maskW[i,j,k] * fxa * pyom.eke_int_diss0
        pyom.c_Ri_diss[:,:,-1] = pyom.c_Ri_diss[:,:,-2] * pyom.maskW[:,:,-1]

        """
        vertically integrate Ri-dependent dissipation and EKE
        """
        for k in xrange(pyom.nz-1): # k = 1,nz-1
            a_loc[...] += pyom.c_Ri_diss[:,:,k] * pyom.eke[:,:,k,pyom.tau] * pyom.maskW[:,:,k] * pyom.dzw[k]
            b_loc[...] += pyom.eke[:,:,k,pyom.tau] * pyom.maskW[:,:,k] * pyom.dzw[k]
        k = -1
        a_loc += pyom.c_Ri_diss[:,:,k] * pyom.eke[:,:,k,pyom.tau] * pyom.maskW[:,:,k] * pyom.dzw[k] * 0.5
        b_loc += pyom.eke[:,:,k,pyom.tau] * pyom.maskW[:,:,k] * pyom.dzw[k] * 0.5

        """
        add bottom fluxes by lee waves and bottom friction to a_loc
        """
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                k = pyom.kbot[i,j] - 1
                if k >= 0 and k < pyom.nz-1:
                    a_loc(i,j) += pyom.c_lee[i,j] * pyom.eke[i,j,k,pyom.tau] * pyom.maskW[i,j,k] * pyom.dzw[k] \
                                  + 2 * pyom.eke_r_bot * pyom.eke[i,j,k,pyom.tau] * math.sqrt(2.0) * pyom.sqrteke[i,j,k] \
                                    * pyom.maskW[i,j,k] # could be surface: factor 0.5

        """
        dissipation constant is vertically integrated forcing divided by
        vertically integrated EKE to account for vertical EKE radiation
        """
        a_loc = np.where(b_loc > 0., a_loc / b_loc, 0.)
        for k in xrange(pyom.nz): # k = 1,nz
            c_int[:,:,k] = a_loc
    else:
        """
        dissipation by local interior loss of balance with constant coefficient
        """
        for k in xrange(pyom.nz): # k = 1,nz
            for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): # j = js_pe-onx,je_pe+onx
                for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx): # i = is_pe-onx,ie_pe+onx
                    c_int[i,j,k] = pyom.eke_c_eps * pyom.sqrteke[i,j,k] / pyom.eke_len[i,j,k] * pyom.maskW[i,j,k]
    """
    vertical diffusion of EKE,forcing and dissipation
    """
    ks = pyom.kbot - 1
    delta, a_tri, b_tri, c_tri, d_tri = (np.zeros((pyom.nx, pyom.ny, pyom.nz)) for _ in range(5))
    delta[:,:,:-1] = pyom.dt_tracer / pyom.dzt[None, None, 1:] * 0.5 \
                 * (pyom.kappaM[2:-2, 2:-2, :-1] + pyom.kappaM[2:-2, 2:-2, 1:]) * pyom.alpha_eke
    a_tri[:, :, 1:-1] = -delta[:-2] / pyom.dzw[1:-1]
    a_tri[:, :, -1] = -delta[:,:,-2] / (0.5 * pyom.dzw[-1])
    b_tri[:, :, 1:-1] = 1 + (delta[:,:,1:-1] + delta[:,:,:-2]) / pyom.dzw[1:-1] + pyom.dt_tracer * c_int[2:-2, 2:-2, 1:-1]
    b_tri[:, :, -1] = 1 + delta[-2] / (0.5 * pyom.dzw[-1]) + pyom.dt_tracer * c_int[2:-2, 2:-2, -1]
    b_tri_edge = 1 + delta / dzw[None, None, :] + pyom.dt_tracer * c_int[2:-2, 2:-2, :]
    c_tri[:, :, :-1] = -delta[:, :, :-1] / pyom.dzw[None, None, :-1]
    d_tri[:, :, :] = pyom.eke[2:-2, 2:-2, :, pyom.tau] + pyom.dt_tracer * forc[2:-2, 2:-2, :]
    sol, water_mask = utilities.solve_implicit(ks, a_tri, b_tri, c_tri, d_tri, pyom, b_edge=b_tri_edge)
    pyom.eke[2:-2, 2:-2, :, pyom.taup1] = sol[water_mask]

 #    for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
 #        for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
 #    ks = kbot(i,j)
 #    if ks>0:
 #     for k in xrange(ks,ke-1): # k = ks,ke-1
 #      delta[k] = dt_tracer/dzt(k+1)*0.5*(kappaM[i,j,k]+KappaM[i,j,k+1])*alpha_eke
 #     enddo
 #     delta(ke) = 0.0
 #     for k in xrange(ks+1,ke-1): # k = ks+1,ke-1
 #       a_tri[k] = - delta(k-1)/dzw[k]
 #     enddo
 #     a_tri(ks) = 0.0
 #     a_tri(ke) = - delta(ke-1)/(0.5*dzw(ke))
 #     for k in xrange(ks+1,ke-1): # k = ks+1,ke-1
 #      b_tri[k] = 1+ delta[k]/dzw[k] + delta(k-1)/dzw[k] + dt_tracer*c_int[i,j,k]
 #     enddo
 #     b_tri(ke) = 1+ delta(ke-1)/(0.5*dzw(ke)) + dt_tracer*c_int(i,j,ke)
 #     b_tri(ks) = 1+ delta(ks)/dzw(ks)         + dt_tracer*c_int(i,j,ks)
 #     for k in xrange(ks,ke-1): # k = ks,ke-1
 #      c_tri[k] = - delta[k]/dzw[k]
 #     enddo
 #     c_tri(ke) = 0.0
 #     d_tri(ks:ke) = eke(i,j,ks:ke,tau)  + dt_tracer*forc(i,j,ks:ke)
 #     d_tri(ks) = d_tri(ks)
 #     d_tri(ke) = d_tri(ke) !+ dt_tracer*forc_eke_surfac(i,j)/(0.5*dzw(ke))
 #     solve_tridiag(a_tri(ks:ke),b_tri(ks:ke),c_tri(ks:ke),d_tri(ks:ke),eke(i,j,ks:ke,taup1),ke-ks+1)
 #    endif
 #  enddo
 # enddo

    """
    store eke dissipation
    """
    if pyom.enable_eke_leewave_dissipation:
        pyom.eke_diss_iw[...] = 0.
        pyom.eke_diss_tke[...] = pyom.c_Ri_diss * pyom.eke[:,:,:,pyom.taup1]

        """
        flux by lee wave generation and bottom friction
        """
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                k = pyom.kbot[i,j] - 1
                if k >= 0 and k < pyom.nz-1:
                    pyom.eke_diss_iw[i,j,k] += pyom.c_lee[i,j] * pyom.eke[i,j,k,pyom.taup1] * pyom.maskW[i,j,k]
                    pyom.eke_diss_tke[i,j,k] += 2 * pyom.eke_r_bot * pyom.eke[i,j,k,pyom.taup1] * math.sqrt(2.0) \
                                                * pyom.sqrteke[i,j,k] * pyom.maskW[i,j,k] / pyom.dzw[k]
        """
        account for sligthly incorrect integral of dissipation due to time stepping
        """
        a_loc[...] = 0.
        b_loc[...] = 0.
        for k in xrange(pyom.nz-1): # k = 1,nz-1
            a_loc += (pyom.eke_diss_iw[:,:,k] + pyom.eke_diss_tke[:,:,k]) * pyom.dzw[k]
            b_loc += pyom.c_int[:,:,k] * pyom.eke[:,:,k,pyom.taup1] * pyom.dzw[k]
        k = -1
        a_loc += (pyom.eke_diss_iw[:,:,k] + pyom.eke_diss_tke[:,:,k]) * pyom.dzw[k] * 0.5
        b_loc += pyom.c_int[:,:,k] * pyom.eke[:,:,k,pyom.taup1] * pyom.dzw[k] * 0.5
        b_loc = np.where(a_loc != 0., b_loc / a_loc, 0.0)
        # F = eke_diss,  a = sum F,  b = sum c_int e
        # G = F*b/a -> sum G = (sum c_int e) /(sum F)  sum F
        for k in xrange(pyom.nz): # k = 1,nz
            pyom.eke_diss_iw[:,:,k] = pyom.eke_diss_iw[:,:,k] * b_loc
            pyom.eke_diss_tke[:,:,k] = pyom.eke_diss_tke[:,:,k] * b_loc
        """
        store diagnosed flux by lee waves and bottom friction
        """
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                k = pyom.kbot[i,j] - 1
                if k >= 0 and k < pyom.nz-1:
                    pyom.eke_lee_flux[i,j] = pyom.c_lee[i,j] * pyom.eke[i,j,k,pyom.taup1] * pyom.dzw[k]
                    pyom.eke_bot_flux[i,j] = 2 * pyom.eke_r_bot * pyom.eke[i,j,k,pyom.taup1] * math.sqrt(2.0) * pyom.sqrteke[i,j,k]
    else:
        pyom.eke_diss_iw = pyom.c_int * pyom.eke[:,:,:,pyom.taup1]
        pyom.eke_diss_tke[...] = 0.

    """
    add tendency due to lateral diffusion
    """
    for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): # j = js_pe-onx,je_pe+onx
        for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx-1): # i = is_pe-onx,ie_pe+onx-1
            pyom.flux_east[i,j,:] = 0.5 * np.maximum(500, pyom.K_gm[i,j,:] + pyom.K_gm[i+1,j,:]) \
                                    * (pyom.eke[i+1,j,:,pyom.tau] - pyom.eke[i,j,:,pyom.tau]) \
                                    / (pyom.cost[j] * pyom.dxu[i]) * pyom.maskU[i,j,:]
    pyom.flux_east[-1,:,:] = 0.
    for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx-1): # j = js_pe-onx,je_pe+onx-1
        pyom.flux_north[:,j,:] = 0.5 * np.maximum(500, pyom.K_gm[:,j,:] + pyom.K_gm[:,j+1,:]) \
                                 * (pyom.eke[:,j+1,:,pyom.tau] - pyom.eke[:,j,:,pyom.tau]) \
                                 / pyom.dyu[j] * pyom.maskV[:,j,:] * pyom.cosu[j]
    pyom.flux_north[:,-1,:] = 0.
    for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
        for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
            pyom.eke[i,j,:,pyom.taup1] += pyom.dt_tracer * pyom.maskW[i,j,:] \
                                         * ((pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:]) \
                                         / (pyom.cost[j] * pyom.dxt[i]) \
                                         + (pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:]) \
                                         / (pyom.cost[j] * pyom.dyt[j]))
    """
    add tendency due to advection
    """
    if pyom.enable_eke_superbee_advection:
        advection.adv_flux_superbee_wgrid(pyom.flux_east, pyom.flux_north, pyom.flux_top, pyom.eke[:,:,:,pyom.tau])
    if pyom.enable_eke_upwind_advection:
        advection.adv_flux_upwind_wgrid(pyom.flux_east, pyom.flux_north, pyom.flux_top, pyom.eke[:,:,:,pyom.tau])
    if pyom.enable_eke_superbee_advection or pyom.enable_eke_upwind_advection:
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                pyom.deke[i,j,:,pyom.tau] = pyom.maskW[i,j,:] * (-(pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:]) \
                                               / (pyom.cost[j] * pyom.dxt[i]) \
                                            - (pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:]) \
                                               / (pyom.cost[j] * pyom.dyt[j]))
        k = 0
        pyom.deke[:,:,k,pyom.tau] += -pyom.flux_top[:,:,k] / pyom.dzw[k]
        for k in xrange(1,pyom.nz-1): # k = 2,nz-1
            pyom.deke[:,:,k,pyom.tau] += -(pyom.flux_top[:,:,k] - pyom.flux_top[:,:,k-1]) / pyom.dzw[k]
        k = -1
        pyom.deke[:,:,k,pyom.tau] += -(pyom.flux_top[:,:,k] - pyom.flux_top[:,:,k-1]) / (0.5 * pyom.dzw[k])
        """
        Adam Bashforth time stepping
        """
        pyom.eke[:,:,:,pyom.taup1] += pyom.dt_tracer * ((1.5 + pyom.AB_eps) * pyom.deke[:,:,:,pyom.tau] \
                                                      - (0.5 + pyom.AB_eps) * pyom.deke[:,:,:,pyom.taum1])

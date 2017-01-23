import numpy as np
import warnings
import math

from climate.pyom import numerics

def set_spectral_parameter(pyom):
    """
    calculate spectral parameter for low mode wave propagation
    """

    calc_wave_speed(pyom)
    cfl = group_velocity(pyom)
    set_time_scales(pyom)
    calc_vertical_struct_fct(pyom)

    if cfl > 0.6:
        warnings.warn("low mode CFL number = {}".format(fxa))


def calc_wave_speed(pyom):
    """
    calculate barolinic wave speed
    """

    pyom.cn = 0.0 # calculate cn = int_(-h)^0 N/pi dz
    for k in xrange(pyom.nz): # k = 1,nz
        for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): # j = js_pe-onx,je_pe+onx
            for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx): # i = is_pe-onx,ie_pe+onx
                pyom.cn[i,j] += math.sqrt(max(0., pyom.Nsqr[i,j,k,tau])) * pyom.dzt[k] * pyom.maskT[i,j,k] / pyom.pi


def get_shelf(pyom):
    # real*8 :: map2(pyom.is_pe-pyom.onx:pyom.ie_pe+pyom.onx,pyom.js_pe-pyom.onx:pyom.je_pe+pyom.onx),fxa
    map2 = np.zeros((pyom.nx+4, pyom.ny+4))
    pyom.topo_shelf = np.where(pyom.ht == 0.0, 1, 0)

    fxa = 0
    if pyom.js_pe >= pyom.ny/2 and pyom.je_pe <= pyom.ny/2: # I guess this is always False
        fxa = pyom.dyt(pyom.ny/2)

    for k in xrange(max(1,int(300e3/fxa))): # k = 1,max(1,int(300e3/fxa))
        map2[...] = pyom.topo_shelf
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                if map2[i,j] == 1:
                    pyom.topo_shelf[i-1:i+2,j-1:j+2] = 1 # (i-1:i+1,j-1:j+1)
        numerics.setcyclic_xy(pyom.topo_shelf,pyom.enable_cyclic_x,pyom.nx,pyom.nz)


def set_time_scales(pyom):
    """
    set decay and interaction time scales
    """
    # real*8 :: mstar = 0.01,M2_f = 2*pi/(12.42*60*60)
    mstar = 0.01
    pyom.M2_f = 2 * pyom.pi / (12.42 * 60 * 60)

    if pyom.enable_idemix_niw:
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                if pyom.ht[i,j] > 0:
                    N0 = pyom.cn[i,j] * pyom.pi / pyom.ht[i,j]
                    if N0 > abs(pyom.coriolis_t[i,j]) and pyom.omega_niw[i,j] > abs(pyom.coriolis_t[i,j]):
                        fxc = pyom.topo_hrms[i,j]**2 * 2*pyom.pi / (1e-12 + pyom.topo_lam[i,j]) # Goff
                        fxb = 0.5 * N0 * ((pyom.omega_niw[i,j]**2 + pyom.coriolis_t[i,j]**2) / pyom.omega_niw[i,j]**2) ** 2  \
                                        * (pyom.omega_niw[i,j]**2 - pyom.coriolis_t[i,j]**2) ** 0.5 / pyom.omega_niw[i,j]
                        pyom.tau_niw[i,j] = min(0.5 / pyom.dt_tracer, fxb * fxc / pyom.ht[i,j])
        pyom.tau_niw[pyom.topo_shelf == 1.0] = 1. / (3. * 86400)
        pyom.tau_niw = np.maximum(1/(50.*86400), pyom.tau_niw) * pyom.maskT[:,:,pyom.nz-1]

    if pyom.enable_idemix_M2:
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                if pyom.ht[i,j] > 0:
                    N0 = pyom.cn[i,j] * pyom.pi / pyom.ht[i,j]
                    if N0 > abs(pyom.coriolis_t[i,j]) and pyom.omega_M2 > abs(pyom.coriolis_t[i,j]):
                        fxc = pyom.topo_hrms[i,j]**2 * 2 * pyom.pi / (1e-12 + pyom.topo_lam[i,j]) # Goff
                        fxb = 0.5 * N0 * ((pyom.omega_M2**2 + pyom.coriolis_t[i,j]**2) / pyom.omega_M2**2) ** 2 \
                                  * (pyom.omega_M2**2 - pyom.coriolis_t[i,j]**2) ** 0.5 / pyom.omega_M2
                        pyom.tau_m2[i,j] = min(0.5 / pyom.dt_tracer, fxc * fxb / pyom.ht[i,j])
        pyom.tau_m2[pyom.topo_shelf == 1.0] = 1. / (3. * 86400)
        pyom.tau_m2 = np.maximum(1 / (50. * 86400), pyom.tau_m2) * pyom.maskT[:,:,pyom.nz-1]

        pyom.alpha_M2_cont[...] = 0.0
        for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
            for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
                if pyom.ht[i,j] > 0.:
                    N0 = pyom.cn[i,j] * pyom.pi / pyom.ht[i,j] + 1e-20
                    if abs(pyom.yt[j]) < 28.5:
                        #! lambda+/M2 = 15*E*mstar/N * (sin(phi-28.5)/sin(28.5))^1/2
                        pyom.alpha_M2_cont[i,j] += pyom.M2_f * 15 * mstar / N0 * (np.sin(abs(abs(pyom.yt[j]) -28.5) / 180. * pyom.pi) \
                                                                                  / np.sin(28.5 / 180. * pyom.pi)) ** 0.5
                    if abs(pyom.yt(j)) < 74.5 :
                        #! lambda-/M2 = 0.7*E*mstar/N *sin^2(phi)
                        pyom.alpha_M2_cont[i,j] += pyom.M2_f * 0.7 * mstar / N0 * np.sin(abs(pyom.yt[j]) / 180. * pyom.pi) ** 2
                    pyom.alpha_M2_cont[i,j] *= 1./pyom.ht[i,j]
        pyom.alpha_M2_cont = np.clip(pyom.alpha_M2_cont,0., 1e-5) * pyom.maskT[:,:,pyom.nz-1]


def group_velocity(pyom):
    """
    calculate (modulus of) group velocity of long gravity waves and change of wavenumber angle phi
    """
    # real*8 :: gradx(pyom.is_pe-pyom.onx:pyom.ie_pe+pyom.onx,pyom.js_pe-pyom.onx:pyom.je_pe+pyom.onx),fxa
    # real*8 :: grady(pyom.is_pe-pyom.onx:pyom.ie_pe+pyom.onx,pyom.js_pe-pyom.onx:pyom.je_pe+pyom.onx)
    gradx = np.zeros((pyom.nx+4, pyom.ny+4))
    grady = np.zeros((pyom.nx+4, pyom.ny+4))

    if pyom.enable_idemix_M2:
        pyom.omega_M2 = 2 * pyom.pi / (12.*60*60 + 25.2*60) # M2 frequency in 1/s

    if pyom.enable_idemix_niw:
        pyom.omega_niw = np.maximum(1e-8, np.abs(1.05 * pyom.coriolis_t))

    if pyom.enable_idemix_M2:
        pyom.cg_M2 = np.sqrt(np.maximum(0., pyom.omega_M2**2 - pyom.coriolis_t**2)) * pyom.cn / pyom.omega_M2

    if pyom.enable_idemix_niw:
        pyom.cg_niw = np.sqrt(np.maximum(0.,pyom.omega_niw**2 - pyom.coriolis_t**2)) * pyom.cn / pyom.omega_niw

    grady[...] = 0.0
    for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
        grady[:,j] = (pyom.coriolis_t[:,j+1] - pyom.coriolis_t[:,j-1]) / (pyom.dyu[j] + pyom.dyu[j-1])

    if pyom.enable_idemix_M2:
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                fxa = max(1e-10, pyom.omega_M2**2 - pyom.coriolis_t[i,j]**2)
                pyom.kdot_y_M2[i,j] = -pyom.cn[i,j] / math.sqrt(fxa) * pyom.coriolis_t[i,j] / pyom.omega_M2 * grady[i,j]

    if pyom.enable_idemix_niw:
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                fxa = max(1e-10, pyom.omega_niw[i,j]**2 - pyom.coriolis_t[i,j]**2)
                pyom.kdot_y_niw[i,j] = -pyom.cn[i,j] / math.sqrt(fxa) * pyom.coriolis_t[i,j] / pyom.omega_niw[i,j] * grady[i,j]

    grady = 0.0
    for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
        grady[:,j] = 0.5*(pyom.cn[:,j+1] - pyom.cn[:,j]) / pyom.dyu[j] * pyom.maskTp[:,j,0] * pyom.maskTp[:,j+1,0] \
                   + 0.5*(pyom.cn[:,j] - pyom.cn[:,j-1]) / pyom.dyu[j-1] * pyom.maskTp[:,j-1,0] * pyom.maskTp[:,j,0]

    if pyom.enable_idemix_M2:
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                fxa = max(0., pyom.omega_M2**2 - pyom.coriolis_t[i,j]**2)
                pyom.kdot_y_M2[i,j] += -math.sqrt(fxa) / pyom.omega_M2 * grady[i,j]

    if pyom.enable_idemix_niw:
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                fxa = max(0., pyom.omega_niw[i,j]**2 - pyom.coriolis_t[i,j]**2)
                pyom.kdot_y_niw[i,j] = pyom.kdot_y_niw[i,j] - math.sqrt[fxa] / pyom.omega_niw[i,j] * grady[i,j]

    gradx = 0.0
    for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
        for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
            gradx[i,j] = 0.5 * (pyom.cn[i+1,j] - pyom.cn[i,j]) / (pyom.dxu[i] * pyom.cost(j)) * pyom.maskTp[i,j,1] * pyom.maskTp[i+1,j,1] \
                       + 0.5 * (pyom.cn[i,j] - pyom.cn[i-1,j]) / (pyom.dxu[i-1] * pyom.cost(j)) * pyom.maskTp[i-1,j,1] * pyom.maskTp[i,j,1]

    if pyom.enable_idemix_M2:
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                fxa = max(0.,pyom.omega_M2**2 - pyom.coriolis_t[i,j]**2)
                pyom.kdot_x_M2[i,j] = math.sqrt(fxa) / pyom.omega_M2 * gradx[i,j]

    if pyom.enable_idemix_niw:
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                fxa = max(0.,pyom.omega_niw[i,j]**2 - pyom.coriolis_t[i,j]**2)
                pyom.kdot_x_niw[i,j] = math.sqrt(fxa) / pyom.omega_niw[i,j] * gradx[i,j]

    if pyom.enable_idemix_M2:
        for k in xrange(1,pyom.np-1): # k = 2,np-1
            for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
                for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                    pyom.u_M2[i,j,k] = 0.5 * (pyom.cg_M2[i+1,j] + pyom.cg_M2[i,j]) * np.cos(pyom.phit[k]) * pyom.maskUp[i,j,k]
        for k in xrange(1,pyom.np-1): # k = 2,np-1
            for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
                for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                    pyom.v_M2[i,j,k] = 0.5 * (pyom.cg_M2[i,j] + pyom.cg_M2[i,j+1]) * np.sin(pyom.phit[k]) * pyom.cosu[j] * pyom.maskVp[i,j,k]
        for k in xrange(0,pyom.np-1): # k = 1,np-1
            for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
                for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                    pyom.w_M2[i,j,k] = (pyom.kdot_y_M2[i,j] * np.cos(pyom.phiu[k]) + pyom.kdot_x_M2[i,j] * np.sin(pyom.phiu[k])) * pyom.maskWp[i,j,k]

    if pyom.enable_idemix_niw:
        for k in xrange(1,pyom.np-1): # k = 2,np-1
            for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
                for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                    pyom.u_niw[i,j,k] = 0.5 * (pyom.cg_niw[i+1,j] + pyom.cg_niw[i,j]) * np.cos(pyom.phit[k]) * pyom.maskUp[i,j,k]
        for k in xrange(1,pyom.np-1): # k = 2,np-1
            for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
                for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                    pyom.v_niw[i,j,k] = 0.5 * (pyom.cg_niw[i,j] + pyom.cg_niw[i,j+1]) * np.sin(pyom.phit[k]) * pyom.cosu[j] * pyom.maskVp[i,j,k]
        for k in xrange(pyom.np-1): # k = 1,np-1
            for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
                for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                    pyom.w_niw[i,j,k] = (pyom.kdot_y_niw[i,j] * np.cos(pyom.phiu[k]) + pyom.kdot_x_niw[i,j] * np.sin(pyom.phiu[k])) * pyom.maskWp[i,j,k]

    cfl = 0.0
    if pyom.enable_idemix_M2:
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                cfl = max(cfl, 0.5*(pyom.cg_M2[i,j] + pyom.cg_M2[i+1,j]) * pyom.dt_tracer / (pyom.cost[j] * pyom.dxt[i]))
                cfl = max(cfl, 0.5*(pyom.cg_M2[i,j] + pyom.cg_M2[i,j+1]) * pyom.dt_tracer / pyom.dyt[j])
                cfl = max(cfl, pyom.kdot_y_M2[i,j] * pyom.dt_tracer / dphit[1])
                cfl = max(cfl, pyom.kdot_x_M2[i,j] * pyom.dt_tracer / dphit[1])
                # !if (cfl>0.5) print*,' WARNING: CFL =',cfl,' at i=',i,' j=',j
    if pyom.enable_idemix_niw:
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                cfl = max(cfl, 0.5*(pyom.cg_niw[i,j] + pyom.cg_niw[i+1,j]) * pyom.dt_tracer / (pyom.cost[j] * pyom.dxt[i]))
                cfl = max(cfl, 0.5*(pyom.cg_niw[i,j] + pyom.cg_niw[i,j+1]) * pyom.dt_tracer / pyom.dyt[j])
                cfl = max(cfl, pyom.kdot_y_niw[i,j] * pyom.dt_tracer / dphit[0])
                cfl = max(cfl, pyom.kdot_x_niw[i,j] * pyom.dt_tracer / dphit[0])
    return cfl


def calc_vertical_struct_fct(pyom):
    """
    calculate vertical structure function for low modes
    """
    # real*8 :: norm(pyom.is_pe-pyom.onx:pyom.ie_pe+pyom.onx,pyom.js_pe-pyom.onx:pyom.je_pe+pyom.onx)
    # real*8 :: Nsqr_lim(pyom.is_pe-pyom.onx:pyom.ie_pe+pyom.onx,pyom.js_pe-pyom.onx:pyom.je_pe+pyom.onx,pyom.nz)
    # real*8, parameter :: small = 1d-12
    norm = np.zeros((pyom.nx+4, pyom.ny+4))
    Nsqr_lim = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    small = 1e-12

    Nsqr_lim = np.maximum(small,pyom.Nsqr[:,:,:,tau])

    # calculate int_(-h)^z N dz
    pyom.phin[...] = 0.
    for k in xrange(pyom.nz): # k = 1,nz
        km1 = max(0,k-1)
        for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): # j = js_pe-onx,je_pe+onx
            for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx): # i = is_pe-onx,ie_pe+onx
                #!fxa = (Nsqr_lim[i,j,k]*pyom.maskT[i,j,k]+Nsqr_lim(i,j,km1)*pyom.maskT(i,j,km1))/(pyom.maskT[i,j,k]+pyom.maskT(i,j,km1)+1d-22)
                fxa = Nsqr_lim[i,j,k] * pyom.maskW[i,j,k]
                pyom.phin[i,j,k] = pyom.phin[i,j,km1] * pyom.maskT[i,j,km1] + math.sqrt(fxa) * pyom.dzw[km1] #!*pyom.maskT[i,j,k]

    # calculate phi_n = cos(int_(-h)^z N/c_n dz)*N^0.5
    # and   dphi_n/dz = sin(int_(-h)^z N/c_n dz)/N^0.5
    for k in xrange(pyom.nz): # k = 1,nz
        for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): # j = js_pe-onx,je_pe+onx
            for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx): # i = is_pe-onx,ie_pe+onx
                fxa = pyom.phin[i,j,k] / (small + pyom.cn[i,j])
                pyom.phinz[i,j,k] = np.sin(fxa) / Nsqr_lim[i,j,k]**0.25
                pyom.phin[i,j,k] = np.cos(fxa) * Nsqr_lim[i,j,k]**0.25

    # normalization with int_(-h)^0 dz (dphi_n/dz)^2 /N^2 = 1
    norm[...] = 0.
    #!for k in xrange(1,pyom.nz): # k = 1,nz
    #!norm = norm+ pyom.phinz[:,:,k]**2/Nsqr_lim[:,:,k]*pyom.dzt(k)*pyom.maskT[:,:,k]
    for k in xrange(pyom.nz-1): # k = 1,nz-1
        norm += pyom.phinz[:,:,k]**2 / Nsqr_lim[:,:,k] * pyom.dzw[k] * pyom.maskW[:,:,k]
    k = pyom.nz-1
    norm += pyom.phinz[:,:,k]**2 / Nsqr_lim[:,:,k] * 0.5 * pyom.dzw[k] * pyom.maskW[:,:,k]
    for k in xrange(pyom.nz): # k = 1,nz
        pyom.phinz[:,:,k] = np.where(norm > 0, pyom.phinz[:,:,k] / norm**0.5, pyom.phinz[:,:,k])

    # normalization with int_(-h)^0 dz phi_n^2 /c_n^2 = 1
    norm[...] = 0.
    #!for k in xrange(1,pyom.nz): # k = 1,nz
    #!norm = norm+ pyom.phin[:,:,k]**2/max(1d-22,pyom.cn)**2*pyom.dzt(k)*pyom.maskT[:,:,k]
    for k in xrange(pyom.nz-1): # k = 1,nz-1
        norm += pyom.phin[:,:,k]**2 / (small + pyom.cn**2) * pyom.dzw[k] * pyom.maskW[:,:,k]
    k = pyom.nz-1
    norm += pyom.phin[:,:,k]**2 / (small + pyom.cn**2) * 0.5 * pyom.dzw[k] * pyom.maskW[:,:,k]
    for k in xrange(pyom.nz): # k = 1,nz
        pyom.phin[:,:,k] = np.where(norm > 0, pyom.phin[:,:,k] / norm**0.5, pyom.phin[:,:,k])

    if pyom.enable_idemix_M2:
        # calculate structure function for energy:
        # E(z) = E_0 0.5((1+f^2/om^2) phi_n^2/c_n^2 + (1-f^2/om^2) (dphi_n/dz)^2/N^2)
        for k in xrange(pyom.nz): # k = 1,nz
            for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): # j = js_pe-onx,je_pe+onx
                for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx): # i = is_pe-onx,ie_pe+onx
                    pyom.E_struct_M2[i,j,k] = 0.5*((1+pyom.coriolis_t[i,j]**2 / pyom.omega_M2**2) * pyom.phin[i,j,k]**2 / (small + pyom.cn[i,j]**2) \
                                                  +(1-pyom.coriolis_t[i,j]**2 / pyom.omega_M2**2) * pyom.phinz[i,j,k]**2 / Nsqr_lim[i,j,k]) #\
                                                  #!)*pyom.maskT[i,j,k])*pyom.maskW[i,j,k]

    if pyom.enable_idemix_niw:
        for k in xrange(pyom.nz): # k = 1,nz
            for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): # j = js_pe-onx,je_pe+onx
                for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx): # i = is_pe-onx,ie_pe+onx
                    pyom.E_struct_niw[i,j,k] = 0.5*((1+pyom.coriolis_t[i,j]**2 / pyom.omega_niw[i,j]**2) * pyom.phin[i,j,k]**2 / (small+pyom.cn[i,j]**2) \
                                              #!+(1-pyom.coriolis_t[i,j]**2/pyom.omega_niw[i,j]**2)*pyom.phinz[i,j,k]**2/Nsqr_lim[i,j,k])*pyom.maskT[i,j,k]
                                              + (1 - pyom.coriolis_t[i,j]**2 / pyom.omega_niw[i,j]**2) * pyom.phinz[i,j,k]**2 / Nsqr_lim[i,j,k]) * pyom.maskW[i,j,k]

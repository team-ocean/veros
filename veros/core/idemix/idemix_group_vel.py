import warnings
import math

from ... import veros_method
from .. import numerics, cyclic

@veros_method
def set_spectral_parameter(veros):
    """
    calculate spectral parameter for low mode wave propagation
    """

    calc_wave_speed(veros)
    cfl = group_velocity(veros)
    set_time_scales(veros)
    calc_vertical_struct_fct(veros)

    if cfl > 0.6:
        warnings.warn("low mode CFL number = {}".format(fxa))

@veros_method
def calc_wave_speed(veros):
    """
    calculate barolinic wave speed
    """

    veros.cn = 0.0 # calculate cn = int_(-h)^0 N/pi dz
    for k in xrange(veros.nz): # k = 1,nz
        for j in xrange(veros.js_pe-veros.onx,veros.je_pe+veros.onx): # j = js_pe-onx,je_pe+onx
            for i in xrange(veros.is_pe-veros.onx,veros.ie_pe+veros.onx): # i = is_pe-onx,ie_pe+onx
                veros.cn[i,j] += math.sqrt(max(0., veros.Nsqr[i,j,k,tau])) * veros.dzt[k] * veros.maskT[i,j,k] / veros.pi

@veros_method
def get_shelf(veros):
    # real*8 :: map2(veros.is_pe-veros.onx:veros.ie_pe+veros.onx,veros.js_pe-veros.onx:veros.je_pe+veros.onx),fxa
    map2 = np.zeros((veros.nx+4, veros.ny+4))
    veros.topo_shelf = np.where(veros.ht == 0.0, 1, 0)

    fxa = 0
    if veros.js_pe >= veros.ny/2 and veros.je_pe <= veros.ny/2: # I guess this is always False
        fxa = veros.dyt(veros.ny/2)

    for k in xrange(max(1,int(300e3/fxa))): # k = 1,max(1,int(300e3/fxa))
        map2[...] = veros.topo_shelf
        for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
            for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
                if map2[i,j] == 1:
                    veros.topo_shelf[i-1:i+2,j-1:j+2] = 1 # (i-1:i+1,j-1:j+1)
    if veros.enable_cyclic_x:
        cyclic.setcyclic_x(veros.topo_shelf)

@veros_method
def set_time_scales(veros):
    """
    set decay and interaction time scales
    """
    # real*8 :: mstar = 0.01,M2_f = 2*pi/(12.42*60*60)
    mstar = 0.01
    veros.M2_f = 2 * veros.pi / (12.42 * 60 * 60)

    if veros.enable_idemix_niw:
        for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
            for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
                if veros.ht[i,j] > 0:
                    N0 = veros.cn[i,j] * veros.pi / veros.ht[i,j]
                    if N0 > abs(veros.coriolis_t[i,j]) and veros.omega_niw[i,j] > abs(veros.coriolis_t[i,j]):
                        fxc = veros.topo_hrms[i,j]**2 * 2*veros.pi / (1e-12 + veros.topo_lam[i,j]) # Goff
                        fxb = 0.5 * N0 * ((veros.omega_niw[i,j]**2 + veros.coriolis_t[i,j]**2) / veros.omega_niw[i,j]**2) ** 2  \
                                        * (veros.omega_niw[i,j]**2 - veros.coriolis_t[i,j]**2) ** 0.5 / veros.omega_niw[i,j]
                        veros.tau_niw[i,j] = min(0.5 / veros.dt_tracer, fxb * fxc / veros.ht[i,j])
        veros.tau_niw[veros.topo_shelf == 1.0] = 1. / (3. * 86400)
        veros.tau_niw = np.maximum(1/(50.*86400), veros.tau_niw) * veros.maskT[:,:,veros.nz-1]

    if veros.enable_idemix_M2:
        for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
            for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
                if veros.ht[i,j] > 0:
                    N0 = veros.cn[i,j] * veros.pi / veros.ht[i,j]
                    if N0 > abs(veros.coriolis_t[i,j]) and veros.omega_M2 > abs(veros.coriolis_t[i,j]):
                        fxc = veros.topo_hrms[i,j]**2 * 2 * veros.pi / (1e-12 + veros.topo_lam[i,j]) # Goff
                        fxb = 0.5 * N0 * ((veros.omega_M2**2 + veros.coriolis_t[i,j]**2) / veros.omega_M2**2) ** 2 \
                                  * (veros.omega_M2**2 - veros.coriolis_t[i,j]**2) ** 0.5 / veros.omega_M2
                        veros.tau_m2[i,j] = min(0.5 / veros.dt_tracer, fxc * fxb / veros.ht[i,j])
        veros.tau_m2[veros.topo_shelf == 1.0] = 1. / (3. * 86400)
        veros.tau_m2 = np.maximum(1 / (50. * 86400), veros.tau_m2) * veros.maskT[:,:,veros.nz-1]

        veros.alpha_M2_cont[...] = 0.0
        for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
            for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
                if veros.ht[i,j] > 0.:
                    N0 = veros.cn[i,j] * veros.pi / veros.ht[i,j] + 1e-20
                    if abs(veros.yt[j]) < 28.5:
                        #! lambda+/M2 = 15*E*mstar/N * (sin(phi-28.5)/sin(28.5))^1/2
                        veros.alpha_M2_cont[i,j] += veros.M2_f * 15 * mstar / N0 * (np.sin(abs(abs(veros.yt[j]) -28.5) / 180. * veros.pi) \
                                                                                  / np.sin(28.5 / 180. * veros.pi)) ** 0.5
                    if abs(veros.yt(j)) < 74.5 :
                        #! lambda-/M2 = 0.7*E*mstar/N *sin^2(phi)
                        veros.alpha_M2_cont[i,j] += veros.M2_f * 0.7 * mstar / N0 * np.sin(abs(veros.yt[j]) / 180. * veros.pi) ** 2
                    veros.alpha_M2_cont[i,j] *= 1./veros.ht[i,j]
        veros.alpha_M2_cont = np.clip(veros.alpha_M2_cont,0., 1e-5) * veros.maskT[:,:,veros.nz-1]

@veros_method
def group_velocity(veros):
    """
    calculate (modulus of) group velocity of long gravity waves and change of wavenumber angle phi
    """
    # real*8 :: gradx(veros.is_pe-veros.onx:veros.ie_pe+veros.onx,veros.js_pe-veros.onx:veros.je_pe+veros.onx),fxa
    # real*8 :: grady(veros.is_pe-veros.onx:veros.ie_pe+veros.onx,veros.js_pe-veros.onx:veros.je_pe+veros.onx)
    gradx = np.zeros((veros.nx+4, veros.ny+4))
    grady = np.zeros((veros.nx+4, veros.ny+4))

    if veros.enable_idemix_M2:
        veros.omega_M2 = 2 * veros.pi / (12.*60*60 + 25.2*60) # M2 frequency in 1/s

    if veros.enable_idemix_niw:
        veros.omega_niw = np.maximum(1e-8, np.abs(1.05 * veros.coriolis_t))

    if veros.enable_idemix_M2:
        veros.cg_M2 = np.sqrt(np.maximum(0., veros.omega_M2**2 - veros.coriolis_t**2)) * veros.cn / veros.omega_M2

    if veros.enable_idemix_niw:
        veros.cg_niw = np.sqrt(np.maximum(0.,veros.omega_niw**2 - veros.coriolis_t**2)) * veros.cn / veros.omega_niw

    grady[...] = 0.0
    for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
        grady[:,j] = (veros.coriolis_t[:,j+1] - veros.coriolis_t[:,j-1]) / (veros.dyu[j] + veros.dyu[j-1])

    if veros.enable_idemix_M2:
        for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
            for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
                fxa = max(1e-10, veros.omega_M2**2 - veros.coriolis_t[i,j]**2)
                veros.kdot_y_M2[i,j] = -veros.cn[i,j] / math.sqrt(fxa) * veros.coriolis_t[i,j] / veros.omega_M2 * grady[i,j]

    if veros.enable_idemix_niw:
        for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
            for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
                fxa = max(1e-10, veros.omega_niw[i,j]**2 - veros.coriolis_t[i,j]**2)
                veros.kdot_y_niw[i,j] = -veros.cn[i,j] / math.sqrt(fxa) * veros.coriolis_t[i,j] / veros.omega_niw[i,j] * grady[i,j]

    grady = 0.0
    for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
        grady[:,j] = 0.5*(veros.cn[:,j+1] - veros.cn[:,j]) / veros.dyu[j] * veros.maskTp[:,j,0] * veros.maskTp[:,j+1,0] \
                   + 0.5*(veros.cn[:,j] - veros.cn[:,j-1]) / veros.dyu[j-1] * veros.maskTp[:,j-1,0] * veros.maskTp[:,j,0]

    if veros.enable_idemix_M2:
        for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
            for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
                fxa = max(0., veros.omega_M2**2 - veros.coriolis_t[i,j]**2)
                veros.kdot_y_M2[i,j] += -math.sqrt(fxa) / veros.omega_M2 * grady[i,j]

    if veros.enable_idemix_niw:
        for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
            for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
                fxa = max(0., veros.omega_niw[i,j]**2 - veros.coriolis_t[i,j]**2)
                veros.kdot_y_niw[i,j] = veros.kdot_y_niw[i,j] - math.sqrt[fxa] / veros.omega_niw[i,j] * grady[i,j]

    gradx = 0.0
    for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
        for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
            gradx[i,j] = 0.5 * (veros.cn[i+1,j] - veros.cn[i,j]) / (veros.dxu[i] * veros.cost(j)) * veros.maskTp[i,j,1] * veros.maskTp[i+1,j,1] \
                       + 0.5 * (veros.cn[i,j] - veros.cn[i-1,j]) / (veros.dxu[i-1] * veros.cost(j)) * veros.maskTp[i-1,j,1] * veros.maskTp[i,j,1]

    if veros.enable_idemix_M2:
        for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
            for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
                fxa = max(0.,veros.omega_M2**2 - veros.coriolis_t[i,j]**2)
                veros.kdot_x_M2[i,j] = math.sqrt(fxa) / veros.omega_M2 * gradx[i,j]

    if veros.enable_idemix_niw:
        for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
            for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
                fxa = max(0.,veros.omega_niw[i,j]**2 - veros.coriolis_t[i,j]**2)
                veros.kdot_x_niw[i,j] = math.sqrt(fxa) / veros.omega_niw[i,j] * gradx[i,j]

    if veros.enable_idemix_M2:
        for k in xrange(1,veros.np-1): # k = 2,np-1
            for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
                for i in xrange(veros.is_pe-1,veros.ie_pe): # i = is_pe-1,ie_pe
                    veros.u_M2[i,j,k] = 0.5 * (veros.cg_M2[i+1,j] + veros.cg_M2[i,j]) * np.cos(veros.phit[k]) * veros.maskUp[i,j,k]
        for k in xrange(1,veros.np-1): # k = 2,np-1
            for j in xrange(veros.js_pe-1,veros.je_pe): # j = js_pe-1,je_pe
                for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
                    veros.v_M2[i,j,k] = 0.5 * (veros.cg_M2[i,j] + veros.cg_M2[i,j+1]) * np.sin(veros.phit[k]) * veros.cosu[j] * veros.maskVp[i,j,k]
        for k in xrange(0,veros.np-1): # k = 1,np-1
            for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
                for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
                    veros.w_M2[i,j,k] = (veros.kdot_y_M2[i,j] * np.cos(veros.phiu[k]) + veros.kdot_x_M2[i,j] * np.sin(veros.phiu[k])) * veros.maskWp[i,j,k]

    if veros.enable_idemix_niw:
        for k in xrange(1,veros.np-1): # k = 2,np-1
            for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
                for i in xrange(veros.is_pe-1,veros.ie_pe): # i = is_pe-1,ie_pe
                    veros.u_niw[i,j,k] = 0.5 * (veros.cg_niw[i+1,j] + veros.cg_niw[i,j]) * np.cos(veros.phit[k]) * veros.maskUp[i,j,k]
        for k in xrange(1,veros.np-1): # k = 2,np-1
            for j in xrange(veros.js_pe-1,veros.je_pe): # j = js_pe-1,je_pe
                for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
                    veros.v_niw[i,j,k] = 0.5 * (veros.cg_niw[i,j] + veros.cg_niw[i,j+1]) * np.sin(veros.phit[k]) * veros.cosu[j] * veros.maskVp[i,j,k]
        for k in xrange(veros.np-1): # k = 1,np-1
            for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
                for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
                    veros.w_niw[i,j,k] = (veros.kdot_y_niw[i,j] * np.cos(veros.phiu[k]) + veros.kdot_x_niw[i,j] * np.sin(veros.phiu[k])) * veros.maskWp[i,j,k]

    cfl = 0.0
    if veros.enable_idemix_M2:
        for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
            for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
                cfl = max(cfl, 0.5*(veros.cg_M2[i,j] + veros.cg_M2[i+1,j]) * veros.dt_tracer / (veros.cost[j] * veros.dxt[i]))
                cfl = max(cfl, 0.5*(veros.cg_M2[i,j] + veros.cg_M2[i,j+1]) * veros.dt_tracer / veros.dyt[j])
                cfl = max(cfl, veros.kdot_y_M2[i,j] * veros.dt_tracer / dphit[1])
                cfl = max(cfl, veros.kdot_x_M2[i,j] * veros.dt_tracer / dphit[1])
                # !if (cfl>0.5) print*,' WARNING: CFL =',cfl,' at i=',i,' j=',j
    if veros.enable_idemix_niw:
        for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
            for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
                cfl = max(cfl, 0.5*(veros.cg_niw[i,j] + veros.cg_niw[i+1,j]) * veros.dt_tracer / (veros.cost[j] * veros.dxt[i]))
                cfl = max(cfl, 0.5*(veros.cg_niw[i,j] + veros.cg_niw[i,j+1]) * veros.dt_tracer / veros.dyt[j])
                cfl = max(cfl, veros.kdot_y_niw[i,j] * veros.dt_tracer / dphit[0])
                cfl = max(cfl, veros.kdot_x_niw[i,j] * veros.dt_tracer / dphit[0])
    return cfl

@veros_method
def calc_vertical_struct_fct(veros):
    """
    calculate vertical structure function for low modes
    """
    # real*8 :: norm(veros.is_pe-veros.onx:veros.ie_pe+veros.onx,veros.js_pe-veros.onx:veros.je_pe+veros.onx)
    # real*8 :: Nsqr_lim(veros.is_pe-veros.onx:veros.ie_pe+veros.onx,veros.js_pe-veros.onx:veros.je_pe+veros.onx,veros.nz)
    # real*8, parameter :: small = 1d-12
    norm = np.zeros((veros.nx+4, veros.ny+4))
    Nsqr_lim = np.zeros((veros.nx+4, veros.ny+4, veros.nz))
    small = 1e-12

    Nsqr_lim = np.maximum(small,veros.Nsqr[:,:,:,tau])

    # calculate int_(-h)^z N dz
    veros.phin[...] = 0.
    for k in xrange(veros.nz): # k = 1,nz
        km1 = max(0,k-1)
        for j in xrange(veros.js_pe-veros.onx,veros.je_pe+veros.onx): # j = js_pe-onx,je_pe+onx
            for i in xrange(veros.is_pe-veros.onx,veros.ie_pe+veros.onx): # i = is_pe-onx,ie_pe+onx
                #!fxa = (Nsqr_lim[i,j,k]*veros.maskT[i,j,k]+Nsqr_lim(i,j,km1)*veros.maskT(i,j,km1))/(veros.maskT[i,j,k]+veros.maskT(i,j,km1)+1d-22)
                fxa = Nsqr_lim[i,j,k] * veros.maskW[i,j,k]
                veros.phin[i,j,k] = veros.phin[i,j,km1] * veros.maskT[i,j,km1] + math.sqrt(fxa) * veros.dzw[km1] #!*veros.maskT[i,j,k]

    # calculate phi_n = cos(int_(-h)^z N/c_n dz)*N^0.5
    # and   dphi_n/dz = sin(int_(-h)^z N/c_n dz)/N^0.5
    for k in xrange(veros.nz): # k = 1,nz
        for j in xrange(veros.js_pe-veros.onx,veros.je_pe+veros.onx): # j = js_pe-onx,je_pe+onx
            for i in xrange(veros.is_pe-veros.onx,veros.ie_pe+veros.onx): # i = is_pe-onx,ie_pe+onx
                fxa = veros.phin[i,j,k] / (small + veros.cn[i,j])
                veros.phinz[i,j,k] = np.sin(fxa) / Nsqr_lim[i,j,k]**0.25
                veros.phin[i,j,k] = np.cos(fxa) * Nsqr_lim[i,j,k]**0.25

    # normalization with int_(-h)^0 dz (dphi_n/dz)^2 /N^2 = 1
    norm[...] = 0.
    #!for k in xrange(1,veros.nz): # k = 1,nz
    #!norm = norm+ veros.phinz[:,:,k]**2/Nsqr_lim[:,:,k]*veros.dzt(k)*veros.maskT[:,:,k]
    for k in xrange(veros.nz-1): # k = 1,nz-1
        norm += veros.phinz[:,:,k]**2 / Nsqr_lim[:,:,k] * veros.dzw[k] * veros.maskW[:,:,k]
    k = veros.nz-1
    norm += veros.phinz[:,:,k]**2 / Nsqr_lim[:,:,k] * 0.5 * veros.dzw[k] * veros.maskW[:,:,k]
    for k in xrange(veros.nz): # k = 1,nz
        veros.phinz[:,:,k] = np.where(norm > 0, veros.phinz[:,:,k] / norm**0.5, veros.phinz[:,:,k])

    # normalization with int_(-h)^0 dz phi_n^2 /c_n^2 = 1
    norm[...] = 0.
    #!for k in xrange(1,veros.nz): # k = 1,nz
    #!norm = norm+ veros.phin[:,:,k]**2/max(1d-22,veros.cn)**2*veros.dzt(k)*veros.maskT[:,:,k]
    for k in xrange(veros.nz-1): # k = 1,nz-1
        norm += veros.phin[:,:,k]**2 / (small + veros.cn**2) * veros.dzw[k] * veros.maskW[:,:,k]
    k = veros.nz-1
    norm += veros.phin[:,:,k]**2 / (small + veros.cn**2) * 0.5 * veros.dzw[k] * veros.maskW[:,:,k]
    for k in xrange(veros.nz): # k = 1,nz
        veros.phin[:,:,k] = np.where(norm > 0, veros.phin[:,:,k] / norm**0.5, veros.phin[:,:,k])

    if veros.enable_idemix_M2:
        # calculate structure function for energy:
        # E(z) = E_0 0.5((1+f^2/om^2) phi_n^2/c_n^2 + (1-f^2/om^2) (dphi_n/dz)^2/N^2)
        for k in xrange(veros.nz): # k = 1,nz
            for j in xrange(veros.js_pe-veros.onx,veros.je_pe+veros.onx): # j = js_pe-onx,je_pe+onx
                for i in xrange(veros.is_pe-veros.onx,veros.ie_pe+veros.onx): # i = is_pe-onx,ie_pe+onx
                    veros.E_struct_M2[i,j,k] = 0.5*((1+veros.coriolis_t[i,j]**2 / veros.omega_M2**2) * veros.phin[i,j,k]**2 / (small + veros.cn[i,j]**2) \
                                                  +(1-veros.coriolis_t[i,j]**2 / veros.omega_M2**2) * veros.phinz[i,j,k]**2 / Nsqr_lim[i,j,k]) #\
                                                  #!)*veros.maskT[i,j,k])*veros.maskW[i,j,k]

    if veros.enable_idemix_niw:
        for k in xrange(veros.nz): # k = 1,nz
            for j in xrange(veros.js_pe-veros.onx,veros.je_pe+veros.onx): # j = js_pe-onx,je_pe+onx
                for i in xrange(veros.is_pe-veros.onx,veros.ie_pe+veros.onx): # i = is_pe-onx,ie_pe+onx
                    veros.E_struct_niw[i,j,k] = 0.5*((1+veros.coriolis_t[i,j]**2 / veros.omega_niw[i,j]**2) * veros.phin[i,j,k]**2 / (small+veros.cn[i,j]**2) \
                                              #!+(1-veros.coriolis_t[i,j]**2/veros.omega_niw[i,j]**2)*veros.phinz[i,j,k]**2/Nsqr_lim[i,j,k])*veros.maskT[i,j,k]
                                              + (1 - veros.coriolis_t[i,j]**2 / veros.omega_niw[i,j]**2) * veros.phinz[i,j,k]**2 / Nsqr_lim[i,j,k]) * veros.maskW[i,j,k]

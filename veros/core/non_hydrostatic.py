import numpy as np
import warnings

from . import cyclic

def solve_non_hydrostatic(veros):
    """
    solve for non hydrostatic pressure
    """
    forc = np.zeros((veros.nx+4, veros.ny+4, veros.nz))

    """
    integrate forward in time
    """
    for k in xrange(veros.nz-1): # k = 1,nz-1
        veros.w[:,:,k,veros.taup1] = veros.w[:,:,k,veros.tau] + veros.dt_mom * (veros.dw_mix[:,:,k] \
                                                                + (1.5 + veros.AB_eps) * veros.dw[:,:,k,veros.tau] \
                                                                - (0.5 + veros.AB_eps) * veros.dw[:,:,k,veros.taum1]) \
                                                                * veros.maskW[:,:,k]
    """
    forcing for non-hydrostatic pressure
    """
    if veros.enable_cyclic_x:
        cyclic.setcyclic_x(veros.u[:,:,:,veros.taup1])
        cyclic.setcyclic_x(veros.v[:,:,:,veros.taup1])
        cyclic.setcyclic_x(veros.w[:,:,:,veros.taup1])
    for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
        for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
            forc[i,j,:] = (veros.u[i,j,:,veros.taup1] - veros.u[i-1,j,:,taup1]) / (veros.cost[j] * veros.dxt[i]) \
                         + (veros.cosu[j] * veros.v[i,j,:,veros.taup1] - veros.cosu[j-1] * veros.v[i,j-1,:,veros.taup1]) \
                         / (veros.cost[j] * veros.dyt[j])
    k = 0
    forc[:,:,k] += veros.w[:,:,k,veros.taup1] / veros.dzt[k]
    for k in xrange(1,veros.nz): # k = 2,nz
        forc[:,:,k] += (veros.w[:,:,k,veros.taup1] - veros.w[:,:,k-1,veros.taup1]) / veros.dzt[k]
    forc *= 1. / veros.dt_mom

    """
    solve for non-hydrostatic pressure
    """
    veros.p_non_hydro[:,:,:,veros.taup1] = 2 * veros.p_non_hydro[:,:,:,veros.tau] - veros.p_non_hydro[:,:,:,veros.taum1] # first guess
    forc = congrad_non_hydro(veros.congr_itts_non_hydro)
    if veros.enable_cyclic_x:
        cyclic.setcyclic_x(veros.p_non_hydro[:,:,:,veros.taup1])
    if veros.itt == 0:
       veros.p_non_hydro[:,:,:,veros.tau] = veros.p_non_hydro[:,:,:,veros.taup1]
       veros.p_non_hydro[:,:,:,veros.taum1] = veros.p_non_hydro[:,:,:,veros.taup1]

    """
    add non-hydrostatic pressure gradient to tendencies
    """
    for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
        for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
            veros.u[i,j,:,veros.taup1] += -veros.dt_mom * (veros.p_non_hydro[i+1,j,:,veros.taup1] \
                                                        - veros.p_non_hydro[i,j,:,veros.taup1]) \
                                                     / (veros.dxu[i] * veros.cost[j]) * veros.maskU[i,j,:]
            veros.v[i,j,:,veros.taup1] += -veros.dt_mom * (veros.p_non_hydro[i,j+1,:,veros.taup1] \
                                                        - veros.p_non_hydro[i,j,:,veros.taup1]) \
                                                     / veros.dyu[j] * veros.maskV[i,j,:]
    for k in xrange(1,nz-1): # k = 1,nz-1
        veros.w[:,:,k,veros.taup1] += -veros.dt_mom * (veros.p_non_hydro[:,:,k+1,veros.taup1] - veros.p_non_hydro[:,:,k,veros.taup1]) \
                                                 / veros.dzw[k] * veros.maskW[:,:,k]


def make_coeff_non_hydro(veros):
    """
                 A * dpsi = forc
                      res = A * p
         res = res +  cf(...,ii,jj,kk) * p(i+ii,j+jj,k+kk)

         forc = p_xx + p_yy + p_zz
         forc = (p(i+1) - 2p(i) + p(i-1))  /dx^2 ...
              = [ (p(i+1) - p(i))/dx - (p(i)-p(i-1))/dx ] /dx
    """
    cf = np.zeros((veros.nx+4, veros.ny+4, veros.nz, 3, 3, 3))
    for k in xrange(veros.nz): # k = 1,nz
        for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
            for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
                mp = veros.maskU[i,j,k]
                mm = veros.maskU[i-1,j,k]
                cf[i,j,k, 1, 1, 1] += -mp / veros.dxu[i] / veros.dxt[i] / veros.cost[j]**2
                cf[i,j,k, 2, 1, 1] += mp / veros.dxu[i] / veros.dxt[i] / veros.cost[j]**2
                cf[i,j,k, 1, 1, 1] += -mm / veros.dxu[i-1] / veros.dxt[i] / veros.cost[j]**2
                cf[i,j,k, 0, 1, 1] += mm / veros.dxu[i-1] / veros.dxt[i] / veros.cost[j]**2

                mp = veros.maskV[i,j,k]
                mm = veros.maskV[i,j-1,k]
                cf[i,j,k, 1, 1, 1] += -mp / veros.dyu[j] / veros.dyt[j] * veros.cosu[j] / veros.cost[j]
                cf[i,j,k, 1, 2, 1] += mp / veros.dyu[j] / veros.dyt[j] * veros.cosu[j] / veros.cost[j]
                cf[i,j,k, 1, 1, 1] += -mm / veros.dyu[j-1] / veros.dyt[j] * veros.cosu[j-1] / veros.cost[j]
                cf[i,j,k, 1, 0, 1] += mm / veros.dyu[j-1] / veros.dyt[j] * veros.cosu[j-1] / veros.cost[j]

    k = 0
    for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
        for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
            mp = veros.maskW[i,j,k]
            cf[i,j,k, 1, 1, 1] += -mp / veros.dzw[k] / veros.dzt[k]
            cf[i,j,k, 1, 1, 2] += mp / veros.dzw[k] / veros.dzt[k]
    for k in xrange(1,veros.nz-1): # k = 2,nz-1
        for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
            for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
                mp = veros.maskW[i,j,k]
                mm = veros.maskW[i,j,k-1]
                cf[i,j,k, 1, 1, 1] += -mp / veros.dzw[k] / veros.dzt[k]
                cf[i,j,k, 1, 1, 2] += mp / veros.dzw[k] / veros.dzt[k]
                cf[i,j,k, 1, 1, 1] += -mm / veros.dzw[k-1] / veros.dzt[k]
                cf[i,j,k, 1, 1, 0] += mm / veros.dzw[k-1] / veros.dzt[k]
    k = veros.nz - 1
    for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
        for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
            mm = veros.maskW[i,j,k-1]
            cf[i,j,k, 1, 1, 1] += -mm / veros.dzw[k-1] / veros.dzt[k]
            cf[i,j,k, 1, 1, 0] += mm / veros.dzw[k-1] / veros.dzt[k]
    return cf


def congrad_non_hydro(forc,iterations,veros):
    """
    simple conjugate gradient solver
    """
    #real*8  :: forc(is_:ie_,js_:je_,nz_)
    #logical, save :: first = .true.
    #real*8, allocatable,save :: cf(:,:,:,:,:,:)
    #real*8  :: res(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    #real*8  :: p(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    #real*8  :: Ap(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    #real*8  :: rsold,alpha,rsnew,dot_3D,absmax_3D
    #real*8  :: step,step1 = 0,convergence_rate,estimated_error,smax,rs_min = 0
    res = np.zeros((veros.nx+4, veros.ny+4, veros.nz))
    p = np.zeros((veros.nx+4, veros.ny+4, veros.nz))
    Ap = np.zeros((veros.nx+4, veros.ny+4, veros.nz))
    step1 = 0
    rs_min = 0

    if congrad_non_hydro.first:
        congrad_non_hydro.cf = make_coeff_non_hydro(veros)
        congrad_non_hydro.first = False

    apply_op_3D(congrad_non_hydro.cf, veros.p_non_hydro[:,:,:,veros.taup1], res) # res = A * psi
    for k in xrange(veros.nz): # k = 1,nz
        for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
            for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
                res[i,j,k] = forc[i,j,k] - res[i,j,k]

    p[...] = res
    if veros.enable_cyclic_x:
        setcyclic_x(veros.p)
    rsold = dot_3d(res,res)

    def print_residual(n, estimated_error, congr_epsilon_non_hydro):
        print(" estimated error={}/{}".format(estimated_error, congr_epsilon_non_hydro))
        print(" iterations={}".format(n))

    for n in xrange(veros.congr_max_itts_non_hydro): # n = 1,congr_max_itts_non_hydro
        """
        key algorithm
        """
        apply_op_3D(cf, p, Ap) # Ap = A *p
        alpha = rsold / dot_3D(p,Ap)
        veros.p_non_hydro[:,:,:,veros.taup1] += alpha * p
        res += -alpha * Ap
        rsnew = dot_3D(res,res)
        p[...] = res + rsnew / rsold * p
        if veros.enable_cyclic_x:
            setcyclic_x(p)
        rsold = rsnew
        """
        test for divergence
        """
        if n == 1:
            rs_min = abs(rsnew)
        elif n > 2:
            rs_min = min(rs_min, abs(rsnew))
            if abs(rsnew) > 100.0 * rs_min:
                warnings.warn("non hydrostatic solver diverging after {} iterations".format(n))
                print_residual(n, estimated_error, veros.congr_epsilon_non_hydro)
                return
        """
        test for convergence
        """
        smax = absmax_3D(p)
        step = abs(alpha) * smax
        if n == 1:
            step1 = step
            estimated_error = step
            if step > veros.congr_epsilon_non_hydro:
                print_residual(n, estimated_error, veros.congr_epsilon_non_hydro)
            elif step > veros.congr_epsilon_non_hydro:
                convergence_rate = np.exp(np.log(step / step1) / (n-1))
                estimated_error = step * convergence_rate / (1.0 - convergence_rate)
                if estimated_error > veros.congr_epsilon_non_hydro:
                    print_residual(n, estimated_error, veros.congr_epsilon_non_hydro)
    warnings.warn("max iterations exceeded at itt={}".format(itt))
    print_residual(n, estimated_error, veros.congr_epsilon_non_hydro)
    return

congrad_non_hydro.first = False


def apply_op_3D(cf, p1, res, veros):
    """
    apply operator A,  res = A *p1
    """
    res[...] = 0.
    for kk in xrange(-1,2):
        for jj in xrange(-1,2):
            for ii in xrange(-1,2):
                for k in xrange(veros.nz): # k = 1,nz
                    kpkk = min(veros.nz-1,max(0,k+kk))
                    for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
                        for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
                            res[i,j,k] += cf[i,j,k,ii+1,jj+1,kk+1] * p1[i+ii,j+jj,kpkk]


def absmax_3D(p1,veros):
    s2 = 0.
    for k in xrange(veros.nz): # k = 1,nz
        for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
            for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
                s2 = max(abs(p1[i,j,k] * veros.maskT[i,j,k]), s2)
    return s2


def dot_3D(p1,p2,veros):
    s2 = 0.
    for k in xrange(veros.nz): # k = 1,nz
        for j in xrange(veros.js_pe,veros.je_pe): # j = js_pe,je_pe
            for i in xrange(veros.is_pe,veros.ie_pe): # i = is_pe,ie_pe
                s2 += p1[i,j,k] * p2[i,j,k] * veros.maskT[i,j,k]
    return s2

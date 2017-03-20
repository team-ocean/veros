import numpy as np
import warnings

from . import cyclic

def solve_non_hydrostatic(pyom):
    """
    solve for non hydrostatic pressure
    """
    forc = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    """
    integrate forward in time
    """
    for k in xrange(pyom.nz-1): # k = 1,nz-1
        pyom.w[:,:,k,pyom.taup1] = pyom.w[:,:,k,pyom.tau] + pyom.dt_mom * (pyom.dw_mix[:,:,k] \
                                                                + (1.5 + pyom.AB_eps) * pyom.dw[:,:,k,pyom.tau] \
                                                                - (0.5 + pyom.AB_eps) * pyom.dw[:,:,k,pyom.taum1]) \
                                                                * pyom.maskW[:,:,k]
    """
    forcing for non-hydrostatic pressure
    """
    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(pyom.u[:,:,:,pyom.taup1])
        cyclic.setcyclic_x(pyom.v[:,:,:,pyom.taup1])
        cyclic.setcyclic_x(pyom.w[:,:,:,pyom.taup1])
    for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
        for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
            forc[i,j,:] = (pyom.u[i,j,:,pyom.taup1] - pyom.u[i-1,j,:,taup1]) / (pyom.cost[j] * pyom.dxt[i]) \
                         + (pyom.cosu[j] * pyom.v[i,j,:,pyom.taup1] - pyom.cosu[j-1] * pyom.v[i,j-1,:,pyom.taup1]) \
                         / (pyom.cost[j] * pyom.dyt[j])
    k = 0
    forc[:,:,k] += pyom.w[:,:,k,pyom.taup1] / pyom.dzt[k]
    for k in xrange(1,pyom.nz): # k = 2,nz
        forc[:,:,k] += (pyom.w[:,:,k,pyom.taup1] - pyom.w[:,:,k-1,pyom.taup1]) / pyom.dzt[k]
    forc *= 1. / pyom.dt_mom

    """
    solve for non-hydrostatic pressure
    """
    pyom.p_non_hydro[:,:,:,pyom.taup1] = 2 * pyom.p_non_hydro[:,:,:,pyom.tau] - pyom.p_non_hydro[:,:,:,pyom.taum1] # first guess
    forc = congrad_non_hydro(pyom.congr_itts_non_hydro)
    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(pyom.p_non_hydro[:,:,:,pyom.taup1])
    if pyom.itt == 0:
       pyom.p_non_hydro[:,:,:,pyom.tau] = pyom.p_non_hydro[:,:,:,pyom.taup1]
       pyom.p_non_hydro[:,:,:,pyom.taum1] = pyom.p_non_hydro[:,:,:,pyom.taup1]

    """
    add non-hydrostatic pressure gradient to tendencies
    """
    for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
        for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
            pyom.u[i,j,:,pyom.taup1] += -pyom.dt_mom * (pyom.p_non_hydro[i+1,j,:,pyom.taup1] \
                                                        - pyom.p_non_hydro[i,j,:,pyom.taup1]) \
                                                     / (pyom.dxu[i] * pyom.cost[j]) * pyom.maskU[i,j,:]
            pyom.v[i,j,:,pyom.taup1] += -pyom.dt_mom * (pyom.p_non_hydro[i,j+1,:,pyom.taup1] \
                                                        - pyom.p_non_hydro[i,j,:,pyom.taup1]) \
                                                     / pyom.dyu[j] * pyom.maskV[i,j,:]
    for k in xrange(1,nz-1): # k = 1,nz-1
        pyom.w[:,:,k,pyom.taup1] += -pyom.dt_mom * (pyom.p_non_hydro[:,:,k+1,pyom.taup1] - pyom.p_non_hydro[:,:,k,pyom.taup1]) \
                                                 / pyom.dzw[k] * pyom.maskW[:,:,k]


def make_coeff_non_hydro(pyom):
    """
                 A * dpsi = forc
                      res = A * p
         res = res +  cf(...,ii,jj,kk) * p(i+ii,j+jj,k+kk)

         forc = p_xx + p_yy + p_zz
         forc = (p(i+1) - 2p(i) + p(i-1))  /dx^2 ...
              = [ (p(i+1) - p(i))/dx - (p(i)-p(i-1))/dx ] /dx
    """
    cf = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz, 3, 3, 3))
    for k in xrange(pyom.nz): # k = 1,nz
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                mp = pyom.maskU[i,j,k]
                mm = pyom.maskU[i-1,j,k]
                cf[i,j,k, 1, 1, 1] += -mp / pyom.dxu[i] / pyom.dxt[i] / pyom.cost[j]**2
                cf[i,j,k, 2, 1, 1] += mp / pyom.dxu[i] / pyom.dxt[i] / pyom.cost[j]**2
                cf[i,j,k, 1, 1, 1] += -mm / pyom.dxu[i-1] / pyom.dxt[i] / pyom.cost[j]**2
                cf[i,j,k, 0, 1, 1] += mm / pyom.dxu[i-1] / pyom.dxt[i] / pyom.cost[j]**2

                mp = pyom.maskV[i,j,k]
                mm = pyom.maskV[i,j-1,k]
                cf[i,j,k, 1, 1, 1] += -mp / pyom.dyu[j] / pyom.dyt[j] * pyom.cosu[j] / pyom.cost[j]
                cf[i,j,k, 1, 2, 1] += mp / pyom.dyu[j] / pyom.dyt[j] * pyom.cosu[j] / pyom.cost[j]
                cf[i,j,k, 1, 1, 1] += -mm / pyom.dyu[j-1] / pyom.dyt[j] * pyom.cosu[j-1] / pyom.cost[j]
                cf[i,j,k, 1, 0, 1] += mm / pyom.dyu[j-1] / pyom.dyt[j] * pyom.cosu[j-1] / pyom.cost[j]

    k = 0
    for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
        for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
            mp = pyom.maskW[i,j,k]
            cf[i,j,k, 1, 1, 1] += -mp / pyom.dzw[k] / pyom.dzt[k]
            cf[i,j,k, 1, 1, 2] += mp / pyom.dzw[k] / pyom.dzt[k]
    for k in xrange(1,pyom.nz-1): # k = 2,nz-1
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                mp = pyom.maskW[i,j,k]
                mm = pyom.maskW[i,j,k-1]
                cf[i,j,k, 1, 1, 1] += -mp / pyom.dzw[k] / pyom.dzt[k]
                cf[i,j,k, 1, 1, 2] += mp / pyom.dzw[k] / pyom.dzt[k]
                cf[i,j,k, 1, 1, 1] += -mm / pyom.dzw[k-1] / pyom.dzt[k]
                cf[i,j,k, 1, 1, 0] += mm / pyom.dzw[k-1] / pyom.dzt[k]
    k = pyom.nz - 1
    for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
        for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
            mm = pyom.maskW[i,j,k-1]
            cf[i,j,k, 1, 1, 1] += -mm / pyom.dzw[k-1] / pyom.dzt[k]
            cf[i,j,k, 1, 1, 0] += mm / pyom.dzw[k-1] / pyom.dzt[k]
    return cf


def congrad_non_hydro(forc,iterations,pyom):
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
    res = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    p = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    Ap = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    step1 = 0
    rs_min = 0

    if congrad_non_hydro.first:
        congrad_non_hydro.cf = make_coeff_non_hydro(pyom)
        congrad_non_hydro.first = False

    apply_op_3D(congrad_non_hydro.cf, pyom.p_non_hydro[:,:,:,pyom.taup1], res) # res = A * psi
    for k in xrange(pyom.nz): # k = 1,nz
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                res[i,j,k] = forc[i,j,k] - res[i,j,k]

    p[...] = res
    if pyom.enable_cyclic_x:
        setcyclic_x(pyom.p)
    rsold = dot_3d(res,res)

    def print_residual(n, estimated_error, congr_epsilon_non_hydro):
        print(" estimated error={}/{}".format(estimated_error, congr_epsilon_non_hydro))
        print(" iterations={}".format(n))

    for n in xrange(pyom.congr_max_itts_non_hydro): # n = 1,congr_max_itts_non_hydro
        """
        key algorithm
        """
        apply_op_3D(cf, p, Ap) # Ap = A *p
        alpha = rsold / dot_3D(p,Ap)
        pyom.p_non_hydro[:,:,:,pyom.taup1] += alpha * p
        res += -alpha * Ap
        rsnew = dot_3D(res,res)
        p[...] = res + rsnew / rsold * p
        if pyom.enable_cyclic_x:
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
                print_residual(n, estimated_error, pyom.congr_epsilon_non_hydro)
                return
        """
        test for convergence
        """
        smax = absmax_3D(p)
        step = abs(alpha) * smax
        if n == 1:
            step1 = step
            estimated_error = step
            if step > pyom.congr_epsilon_non_hydro:
                print_residual(n, estimated_error, pyom.congr_epsilon_non_hydro)
            elif step > pyom.congr_epsilon_non_hydro:
                convergence_rate = np.exp(np.log(step / step1) / (n-1))
                estimated_error = step * convergence_rate / (1.0 - convergence_rate)
                if estimated_error > pyom.congr_epsilon_non_hydro:
                    print_residual(n, estimated_error, pyom.congr_epsilon_non_hydro)
    warnings.warn("max iterations exceeded at itt={}".format(itt))
    print_residual(n, estimated_error, pyom.congr_epsilon_non_hydro)
    return

congrad_non_hydro.first = False


def apply_op_3D(cf, p1, res, pyom):
    """
    apply operator A,  res = A *p1
    """
    res[...] = 0.
    for kk in xrange(-1,2):
        for jj in xrange(-1,2):
            for ii in xrange(-1,2):
                for k in xrange(pyom.nz): # k = 1,nz
                    kpkk = min(pyom.nz-1,max(0,k+kk))
                    for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
                        for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                            res[i,j,k] += cf[i,j,k,ii+1,jj+1,kk+1] * p1[i+ii,j+jj,kpkk]


def absmax_3D(p1,pyom):
    s2 = 0.
    for k in xrange(pyom.nz): # k = 1,nz
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                s2 = max(abs(p1[i,j,k] * pyom.maskT[i,j,k]), s2)
    return s2


def dot_3D(p1,p2,pyom):
    s2 = 0.
    for k in xrange(pyom.nz): # k = 1,nz
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                s2 += p1[i,j,k] * p2[i,j,k] * pyom.maskT[i,j,k]
    return s2

"""
      solve two dimensional Possion equation
           A * dpsi = forc,  where A = nabla_h^2
      with Neumann boundary conditions
      used for surface pressure or free surface
      method same as pressure method in MITgcm
"""
import warnings

from .. import cyclic
from ... import veros_method

@veros_method
def solve_pressure(veros):
    fpx = np.zeros((veros.nx+4, veros.ny+4))
    fpy = np.zeros((veros.nx+4, veros.ny+4))
    forc = np.zeros((veros.nx+4, veros.ny+4))

    # hydrostatic pressure
    fxa = veros.grav / veros.rho_0
    veros.p_hydro[:,:,veros.nz-1] = 0.5*veros.rho[:,:,veros.nz-1,veros.tau]*fxa*veros.dzw[veros.nz-1]*veros.maskT[:,:,veros.nz-1]
    for k in xrange(veros.nz-2, 0, -1): #k=veros.nz-1,1,-1
        veros.p_hydro[:,:,k] = veros.maskT[:,:,k]*(veros.p_hydro[:,:,k+1] + 0.5*(veros.rho[:,:,k+1,veros.tau]+veros.rho[:,:,k,veros.tau])*fxa*veros.dzw[k])

    # add hydrostatic pressure gradient to tendencies
    for j in xrange(veros.js_pe, veros.je_pe+1): #j=veros.js_pe,veros.je_pe
        for i in xrange(veros.is_pe, veros.ie_pe): #i=veros.is_pe,veros.ie_pe
            veros.du[i,j,:,veros.tau] -= (veros.p_hydro[i+1,j,:]-veros.p_hydro[i,j,:]) / (veros.dxu[i] * veros.cost[j]) * veros.maskU[i,j,:]
            veros.dv[i,j,:,veros.tau] -= (veros.p_hydro[i,j+1,:]-veros.p_hydro[i,j,:]) / veros.dyu[j] * veros.maskV[i,j,:]

    # integrate forward in time
    veros.u[:,:,:,veros.taup1] = veros.u[:,:,:,veros.tau] + veros.dt_mom * \
                                                        (veros.du_mix + (1.5+veros.AB_eps) \
                                                                     * veros.du[:,:,:,veros.tau] \
                                                        - (0.5+veros.AB_eps) * veros.du[:,:,:,veros.taum1]) * veros.maskU
    veros.v[:,:,:,veros.taup1] = veros.v[:,:,:,veros.tau] + veros.dt_mom * \
                                                        (veros.dv_mix + (1.5+veros.AB_eps) \
                                                                     * veros.dv[:,:,:,veros.tau] \
                                                        - (0.5+veros.AB_eps) * veros.dv[:,:,:,veros.taum1]) * veros.maskV

    # forcing for surface pressure
    for k in xrange(veros.nz): #k=1,veros.nz
        for j in xrange(veros.js_pe, veros.je_pe): #j=veros.js_pe,veros.je_pe
            for i in xrange(veros.is_pe, veros.ie_pe): #i=veros.is_pe,veros.ie_pe
                fpx[i,j] += veros.u[i,j,k,veros.taup1] * veros.maskU[i,j,k] * veros.dzt[k] / veros.dt_mom
                fpy[i,j] += veros.v[i,j,k,veros.taup1] * veros.maskV[i,j,k] * veros.dzt[k] / veros.dt_mom
    if veros.enable_cyclic_x:
        cyclic.setcyclic_x(fpx)
        cyclic.setcyclic_x(fpy)

    # forc = 1/cos (u_x + (cos veros.v)_y )
    for j in xrange(veros.js_pe, veros.je_pe+1): #j=veros.js_pe,veros.je_pe
        for i in xrange(veros.is_pe, veros.ie_pe+1): #i=veros.is_pe,veros.ie_pe
            forc[i,j] = (fpx[i,j]-fpx[i-1,j])/(veros.cost[j]*veros.dxt[i])+(veros.cosu[j]*fpy[i,j]-veros.cosu[j-1]*fpy[i,j-1])/(veros.cost[j]*veros.dyt[j])
    if veros.enable_free_surface:
        for j in xrange(veros.js_pe, veros.je_pe+1): #j=veros.js_pe,veros.je_pe
            for i in xrange(veros.is_pe, veros.ie_pe+1): #i=veros.is_pe,veros.ie_pe
                forc[i,j] -= veros.psi[i,j,veros.tau]/(veros.grav*veros.dt_mom**2)*veros.maskT[i,j,veros.nz]

    veros.psi[:,:,veros.taup1] = 2*veros.psi[:,:,veros.tau]-veros.psi[:,:,veros.taum1] # first guess
    #solve for surface pressure
    congrad_surf_press(forc,veros.congr_itts,veros)
    if veros.enable_cyclic_x:
        cyclic.setcyclic_x(veros.psi[:,:,veros.taup1])

    # remove surface pressure gradient
    for j in xrange(veros.js_pe, veros.je_pe+1): #j=veros.js_pe,veros.je_pe
        for i in xrange(veros.is_pe, veros.ie_pe+1): #i=veros.is_pe,veros.ie_pe
            veros.u[i,j,:,veros.taup1] -= veros.dt_mom*( veros.psi[i+1,j,veros.taup1]-veros.psi[i,j,veros.taup1])/(veros.dxu[i]*veros.cost[j]) *veros.maskU[i,j,:]
            veros.v[i,j,:,veros.taup1] -= veros.dt_mom*( veros.psi[i,j+1,veros.taup1]-veros.psi[i,j,veros.taup1]) /veros.dyu[j]*veros.maskV[i,j,:]

@veros_method
def make_coeff_surf_press(veros):
    """
    -----------------------------------------------------------------------
             A * p = forc
             res = A * p
             res = res +  cf(...,ii,jj,kk) * p(i+ii,j+jj,k+kk)

             forc = (h p_x)_x +(h p_y)_y
             forc = [ hu(i)(p(i+1) - p(i))/veros.dxu(i) - hu(i-1)(p(i)-p(i-1))/veros.dxu(i-1) ] /veros.dxt(i)
             forc = hu(i) p(i+1)/veros.dxu(i)/veros.dxt(i)  - p(i) hu(i)/veros.dxu(i)/veros.dxt(i) -p(i)*hu(i-1)/veros.dxu(i-1)/veros.dxt(i)  + hu(i-1) p(i-1)/veros.dxu(i-1)/veros.dxt(i)

             in spherical coord.:
             forc = 1/cos^2 ( (h p_x)_x + cos (cos h p_y )_y )
    -----------------------------------------------------------------------
    """
    maskM = veros.maskT[:,:,-1]
    cf = np.zeros((veros.nx+4, veros.ny+4, 3, 3))
    for j in xrange(veros.js_pe, veros.je_pe+1): #j=veros.js_pe,veros.je_pe
        for i in xrange(veros.is_pe, veros.ie_pe+1): #i=veros.is_pe,veros.ie_pe
            mp = maskM[i,j] * maskM[i+1,j]
            mm = maskM[i,j] * maskM[i-1,j]
            cf[i,j, 1, 1] -= mp*veros.hu[i  ,j]/veros.dxu[i  ]/veros.dxt[i] /veros.cost[j]**2
            cf[i,j, 2, 1] += mp*veros.hu[i  ,j]/veros.dxu[i  ]/veros.dxt[i] /veros.cost[j]**2
            cf[i,j, 1, 1] -= mm*veros.hu[i-1,j]/veros.dxu[i-1]/veros.dxt[i] /veros.cost[j]**2
            cf[i,j, 0, 1] += mm*veros.hu[i-1,j]/veros.dxu[i-1]/veros.dxt[i] /veros.cost[j]**2

            mp = maskM[i,j] * maskM[i,j+1]
            mm = maskM[i,j] * maskM[i,j-1]
            cf[i,j, 1, 1] -= mp*veros.hv[i,j  ]/veros.dyu[j  ]/veros.dyt[j] *veros.cosu[j  ]/veros.cost[j]
            cf[i,j, 1, 2] += mp*veros.hv[i,j  ]/veros.dyu[j  ]/veros.dyt[j] *veros.cosu[j  ]/veros.cost[j]
            cf[i,j, 1, 1] -= mm*veros.hv[i,j-1]/veros.dyu[j-1]/veros.dyt[j] *veros.cosu[j-1]/veros.cost[j]
            cf[i,j, 1, 0] += mm*veros.hv[i,j-1]/veros.dyu[j-1]/veros.dyt[j] *veros.cosu[j-1]/veros.cost[j]

    if veros.enable_free_surface:
        for j in xrange(veros.js_pe, veros.je_pe+1): #j=veros.js_pe,veros.je_pe
            for i in xrange(veros.is_pe, veros.ie_pe+1): #i=veros.is_pe,veros.ie_pe
                cf[i,j,1,1] += -1./(veros.grav*veros.dt_mom**2) *maskM[i,j]
    return cf

@veros_method
def congrad_surf_press(veros, forc, iterations):
    """
    simple conjugate gradient solver
    """
    #real*8  :: forc(is_:ie_,js_:je_)
    #logical, save :: first = .true.
    #real*8 , allocatable,save :: cf(:,:,:,:)
    #real*8  :: res(veros.is_pe-onx:veros.ie_pe+onx,veros.js_pe-onx:veros.je_pe+onx)
    #real*8  :: p(veros.is_pe-onx:veros.ie_pe+onx,veros.js_pe-onx:veros.je_pe+onx)
    #real*8  :: Ap(veros.is_pe-onx:veros.ie_pe+onx,veros.js_pe-onx:veros.je_pe+onx)
    #real*8  :: rsold,alpha,rsnew,dot_sfp,absmax_sfp
    #real*8  :: step,step1=0,convergence_rate,estimated_error,smax,rs_min=0
    res = np.zeros((veros.nx+4, veros.ny+4))
    p = np.zeros((veros.nx+4, veros.ny+4))
    Ap = np.zeros((veros.nx+4, veros.ny+4))

    # congrad_surf_press.first is basically like a static variable
    if congrad_surf_press.first:
        cf = make_coeff_surf_press(veros)
        congrad_surf_press.first = False

    apply_op(cf, veros.psi[:,:,veros.taup1], res, veros) #  res = A *veros.psi
    for j in xrange(veros.js_pe, veros.je_pe): #j=veros.js_pe,veros.je_pe
        for i in xrange(veros.is_pe, veros.ie_pe): #i=veros.is_pe,veros.ie_pe
            res[i,j] = forc[i,j]-res[i,j]

    p[...] = res

    if veros.enable_cyclic_x:
        cyclic.setcyclic_x(p)
    rsold = dot_sfp(res,res,veros)

    for n in xrange(1, veros.congr_max_iterations + 1): #n=1,congr_max_iterations
        """
        key algorithm
        """
        apply_op(cf, p, Ap, veros) #  Ap = A *p
        alpha = rsold/dot_sfp(p,Ap,veros)
        veros.psi[:,:,veros.taup1] += alpha*p
        res -= alpha*Ap
        rsnew = dot_sfp(res,res,veros)
        p = res+rsnew/rsold*p
        if veros.enable_cyclic_x:
            cyclic.setcyclic_x(p)
        rsold = rsnew
        """
        test for divergence
        """
        if n == 1:
            rs_min = abs(rsnew)
        elif n > 2:
            rs_min = min(rs_min, abs(rsnew))
            if abs(rsnew) > 100.0 * rs_min:
                warnings.warn("solver diverging after {} iterations".format(n))
                fail(n, veros.enable_congrad_verbose, estimated_error, veros.congr_epsilon)
        """
        test for convergence
        """
        smax = absmax_sfp(p, veros)
        step = abs(alpha) * smax
        if n == 1:
            step1 = step
            estimated_error = step
            if step < veros.congr_epsilon:
                info(n, veros.enable_congrad_verbose, estimated_error, veros.congr_epsilon)
                return
        elif step < veros.congr_epsilon:
            convergence_rate = np.exp(np.log(step/step1)/(n-1))
            estimated_error = step*convergence_rate/(1.0-convergence_rate)
            if estimated_error < veros.congr_epsilon:
                info(n, veros.enable_congrad_verbose, estimated_error, veros.congr_epsilon)
                return
        """
        check for NaN
        """
        if np.isnan(estimated_error):
            warnings.warn("estimated error is NaN at iteration step {}".format(n))
            fail(n, veros.enable_congrad_verbose, estimated_error, veros.congr_epsilon)

    warnings.warn("max iterations exceeded at itt={}".format(veros.itt))
    fail(n, veros.enable_congrad_verbose, estimated_error, veros.congr_epsilon)

congrad_surf_press.first = True

def info(n, enable_congrad_verbose, estimated_error, congr_epsilon):
    if enable_congrad_verbose:
        print ' estimated error=',estimated_error,'/',congr_epsilon
        print ' iterations=',n

def fail(n, enable_congrad_verbose, estimated_error, congr_epsilon):
    print ' estimated error=',estimated_error,'/',congr_epsilon
    print ' iterations=',n
    # check for NaN
    if np.isnan(estimated_error):
        raise RuntimeError("error is NaN, stopping integration")

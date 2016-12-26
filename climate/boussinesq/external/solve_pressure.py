"""
=======================================================================
      solve two dimensional Possion equation
           A * dpsi = forc,  where A = nabla_h^2
      with Neumann boundary conditions
      used for surface pressure or free surface
      method same as pressure method in MITgcm
=======================================================================
"""

import numpy as np
import sys

def solve_pressure():
    #use main_module
    #implicit none
    #integer :: i,j,k
    #real*8 :: fxa
    #real*8 :: fpx(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    #real*8 :: fpy(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    #real*8 :: forc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)

    #hydrostatic pressure
    fxa = grav/rho_0
    p_hydro[:,:,nz] = 0.5*rho[:,:,nz,tau]*fxa*dzw[nz]*maskT[:,:,nz]
    for k in xrange(nz-1, 0, -1): #k=nz-1,1,-1
        p_hydro[:,:,k] = maskT[:,:,k]*(p_hydro[:,:,k+1]+ 0.5*(rho[:,:,k+1,tau]+rho[:,:,k,tau])*fxa*dzw[k])

    # add hydrostatic pressure gradient to tendencies
    for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
        for i in xrange(is_pe, ie_pe): #i=is_pe,ie_pe
            du[i,j,:,tau] -= ( p_hydro[i+1,j,:]-p_hydro[i,j,:]  )/(dxu[i]*cost[j]) *maskU[i,j,:]
            dv[i,j,:,tau] -= ( p_hydro[i,j+1,:]-p_hydro[i,j,:]  ) /dyu[j]*maskV[i,j,:]

    # integrate forward in time
    u[:,:,:,taup1] = u[:,:,:,tau]+dt_mom*( du_mix+ (1.5+AB_eps)*du[:,:,:,tau] - (0.5+AB_eps)*du[:,:,:,taum1] )*maskU
    v[:,:,:,taup1] = v[:,:,:,tau]+dt_mom*( dv_mix+ (1.5+AB_eps)*dv[:,:,:,tau] - (0.5+AB_eps)*dv[:,:,:,taum1] )*maskV

    # forcing for surface pressure
    fpx[...] = 0.
    fpy[...] = 0.
    for k in xrange(1, nz+1): #k=1,nz
        for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
            for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
                fpx[i,j] += u[i,j,k,taup1]*maskU[i,j,k]*dzt[k]/dt_mom
                fpy[i,j] += v[i,j,k,taup1]*maskV[i,j,k]*dzt[k]/dt_mom
    #mpi stuff
    #call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,fpx)
    #call setcyclic_xy   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,fpx)
    #call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,fpy)
    #call setcyclic_xy   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,fpy)

    # forc = 1/cos (u_x + (cos v)_y )
    for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
        for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
            forc[i,j] = (fpx[i,j]-fpx[i-1,j])/(cost[j]*dxt[i])+(cosu[j]*fpy[i,j]-cosu[j-1]*fpy[i,j-1])/(cost[j]*dyt[j])
    if enable_free_surface:
        for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
            for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
                forc[i,j] -= psi[i,j,tau]/(grav*dt_mom**2)*maskT[i,j,nz]

    psi[:,:,taup1] = 2*psi[:,:,tau]-psi[:,:,taum1] # first guess
    #solve for surface pressure
    congrad_surf_press(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,forc,congr_itts)
    #MPI stuff
    #call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,psi(:,:,taup1));
    #call setcyclic_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,psi(:,:,taup1))

    # remove surface pressure gradient
    for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
        for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
            u[i,j,:,taup1] -= dt_mom*( psi[i+1,j,taup1]-psi[i,j,taup1])/(dxu[i]*cost[j]) *maskU[i,j,:]
            v[i,j,:,taup1] -= dt_mom*( psi[i,j+1,taup1]-psi[i,j,taup1]) /dyu[j]*maskV[i,j,:]


def make_coeff_surf_press(is_, ie_, js_, je_, cf):
    """
    -----------------------------------------------------------------------
             A * p = forc
             res = A * p
             res = res +  cf(...,ii,jj,kk) * p(i+ii,j+jj,k+kk)

             forc = (h p_x)_x +(h p_y)_y
             forc = [ hu(i)(p(i+1) - p(i))/dxu(i) - hu(i-1)(p(i)-p(i-1))/dxu(i-1) ] /dxt(i)
             forc = hu(i) p(i+1)/dxu(i)/dxt(i)  - p(i) hu(i)/dxu(i)/dxt(i) -p(i)*hu(i-1)/dxu(i-1)/dxt(i)  + hu(i-1) p(i-1)/dxu(i-1)/dxt(i)

             in spherical coord.:
             forc = 1/cos^2 ( (h p_x)_x + cos (cos h p_y )_y )
    -----------------------------------------------------------------------
    """
    #integer :: is_,ie_,js_,je_
    #!real*8 :: cf(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,3,3)
    #!real*8 :: maskM(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx),mp,mm
    #real*8 :: cf(is_:ie_,js_:je_,3,3),maskM(is_:ie_,js_:je_),mp,mm
    #integer :: i,j
    maskM = maskT[:,:,nz]
    cf[...] = 0.
    for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
        for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
            mp=maskM[i,j]*maskM[i+1,j]
            mm=maskM[i,j]*maskM[i-1,j]
            cf[i,j, 0+2, 0+2] -= mp*hu[i  ,j]/dxu[i  ]/dxt[i] /cost[j]**2
            cf[i,j, 1+2, 0+2] += mp*hu[i  ,j]/dxu[i  ]/dxt[i] /cost[j]**2
            cf[i,j, 0+2, 0+2] -= mm*hu[i-1,j]/dxu[i-1]/dxt[i] /cost[j]**2
            cf[i,j,-1+2, 0+2] += mm*hu[i-1,j]/dxu[i-1]/dxt[i] /cost[j]**2

            mp=maskM[i,j]*maskM[i,j+1]
            mm=maskM[i,j]*maskM[i,j-1]
            cf[i,j, 0+2, 0+2] -= mp*hv[i,j  ]/dyu[j  ]/dyt[j] *cosu[j  ]/cost[j]
            cf[i,j, 0+2, 1+2] += mp*hv[i,j  ]/dyu[j  ]/dyt[j] *cosu[j  ]/cost[j]
            cf[i,j, 0+2, 0+2] -= mm*hv[i,j-1]/dyu[j-1]/dyt[j] *cosu[j-1]/cost[j]
            cf[i,j, 0+2,-1+2] += mm*hv[i,j-1]/dyu[j-1]/dyt[j] *cosu[j-1]/cost[j]

    if enable_free_surface:
        for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
            for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
                cf[i,j,0+2,0+2] = cf[i,j,0+2,0+2] - 1./(grav*dt_mom**2) *maskM[i,j]


def congrad_surf_press(is_, ie_, js_, je_, forc, iterations):
    """
    =======================================================================
      simple conjugate gradient solver
    =======================================================================
    """
    #integer :: is_,ie_,js_,je_
    #integer :: iterations,n,i,j
    #!real*8  :: forc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    #real*8  :: forc(is_:ie_,js_:je_)
    #logical, save :: first = .true.
    #real*8 , allocatable,save :: cf(:,:,:,:)
    #real*8  :: res(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    #real*8  :: p(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    #real*8  :: Ap(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    #real*8  :: rsold,alpha,rsnew,dot_sfp,absmax_sfp
    #real*8  :: step,step1=0,convergence_rate,estimated_error,smax,rs_min=0

    #TODO: fix first, it is locally saved
    if first:
        cf = np.zeros((ie_pe+onx+1-(is_pe-onx), je_pe+onx+1-(js_pe-onx), 3,3))
        make_coeff_surf_press(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,cf)
        first = False

    res[...] = 0
    apply_op(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx, cf, psi[:,:,taup1], res) #  res = A *psi
    for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
        for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
            res[i,j] = forc[i,j]-res[i,j]

    p = res
    #mpi stuff
    #call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,p)
    #call setcyclic_xy   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,p)
    rsold =  dot_sfp(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,res,res)

    for n in xrange(1, congr_max_iterations+1): #n=1,congr_max_iterations
        """
        ----------------------------------------------------------------------
               key algorithm
        ----------------------------------------------------------------------
        """
        apply_op(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,cf, p, Ap) #  Ap = A *p
        alpha = rsold/dot_sfp(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,p,Ap)
        psi[:,:,taup1] += alpha*p
        res -= alpha*Ap
        rsnew = dot_sfp(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,res,res)
        p = res+rsnew/rsold*p
        #MPI stuff
        #call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,p)
        #call setcyclic_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,p)
        rsold = rsnew
        """
        -----------------------------------------------------------------------
               test for divergence
        -----------------------------------------------------------------------
        """
        if n == 1:
            rs_min = abs(rsnew)
        elif n > 2:
            rs_min = min(rs_min, abs(rsnew))
            if abs(rsnew) > 100.0*rs_min:
                if my_pe == 0:
                    print 'WARNING: solver diverging after ',n,' iterations'
                fail(n, my_pe, enable_congrad_verbose, estimated_error, congr_epsilon)
        """
        -----------------------------------------------------------------------
               test for convergence
        -----------------------------------------------------------------------
        """
        smax = absmax_sfp(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,p)
        step = abs(alpha) * smax
        if n == 1:
            step1 = step
            estimated_error = step
            if step < congr_epsilon:
                info(n, my_pe, enable_congrad_verbose, estimated_error, congr_epsilon)
                return
        elif step < congr_epsilon:
            convergence_rate = exp(log(step/step1)/(n-1))
            estimated_error = step*convergence_rate/(1.0-convergence_rate)
            if estimated_error < congr_epsilon:
                info(n, my_pe, enable_congrad_verbose, estimated_error, congr_epsilon)
                return
        """
        -----------------------------------------------------------------------
               check for NaN
        -----------------------------------------------------------------------
        """
        if np.isnan(estimated_error):
            if my_pe == 0:
                print'Warning: estimated error is NaN at iteration step ',n
            fail(n, my_pe, enable_congrad_verbose, estimated_error, congr_epsilon)

    if my_pe == 0:
        print ' WARNING: max iterations exceeded at itt=',itt
    fail(n, my_pe, enable_congrad_verbose, estimated_error, congr_epsilon)

def info(n, my_pe, enable_congrad_verbose, estimated_error, congr_epsilon):
    if my_pe == 0 and enable_congrad_verbose:
        print ' estimated error=',estimated_error,'/',congr_epsilon
        print ' iterations=',n

def fail(n, my_pe, enable_congrad_verbose, estimated_error, congr_epsilon):
    if my_pe == 0:
        print ' estimated error=',estimated_error,'/',congr_epsilon
        print ' iterations=',n
    # check for NaN
    if np.isnan(estimated_error):
        if my_pe == 0:
            print ' error is NaN, stopping integration '
        #TODO: snapshot
        #call panic_snap
        sys.exit(' in solve_pressure')


def apply_op(is_,ie_,js_,je_,cf, p1, res):
    """
    -----------------------------------------------------------------------
         apply operator A,  res = A *p1
    -----------------------------------------------------------------------
    """
    #integer :: is_,ie_,js_,je_
    #!real*8 :: cf(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,3,3)
    #!real*8 :: p1(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx), res(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    #real*8 :: cf(is_:ie_,js_:je_,3,3), p1(is_:ie_,js_:je_), res(is_:ie_,js_:je_)
    #integer :: i,j,ii,jj

    res=0.
    for jj in xrange(-1, 2): #jj=-1,1
        for ii in xrange(-1, 2): #ii=-1,1
            for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
                for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
                    res[i,j] += cf[i,j,ii+2,jj+2]*p1[i+ii,j+jj]

def absmax_sfp(is_,ie_,js_,je_,p1):
    #integer :: is_,ie_,js_,je_
    #real*8 :: absmax_sfp,s2
    #!real*8 :: p1(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    #real*8 :: p1(is_:ie_,js_:je_)
    #integer :: i,j
    s2 = 0
    for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
        for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
            s2 = max( abs(p1[i,j]*maskT[i,j,nz]), s2 )
            #s2 = max( abs(p1(i,j)), s2 )
    return s2


def dot_sfp(is_,ie_,js_,je_,p1,p2):
    #real*8 :: dot_sfp,s2
    #!real*8 :: p1(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx),p2(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    #real*8 :: p1(is_:ie_,js_:je_),p2(is_:ie_,js_:je_)
    #integer :: is_,ie_,js_,je_,i,j
    s2 = 0
    for j in xrange(js_pe, je_pe+1): #j=js_pe,je_pe
        for i in xrange(is_pe, ie_pe+1): #i=is_pe,ie_pe
            s2 = s2+p1[i,j]*p2[i,j]*maskT[i,j,nz]
            #s2 = s2+p1(i,j)*p2(i,j)
    return s2

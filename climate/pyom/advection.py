import numpy as np


def adv_flux_2nd(is_,ie_,js_,je_,nz_,adv_fe,adv_fn,adv_ft,var,pyom):
    """
    2th order advective tracer flux
    """
    # integer, intent(in) :: is_,ie_,js_,je_,nz_
    # real*8, intent(inout) :: adv_fe(is_:ie_,js_:je_,nz_), adv_fn(is_:ie_,js_:je_,nz_)
    # real*8, intent(inout) :: adv_ft(is_:ie_,js_:je_,nz_),    var(is_:ie_,js_:je_,nz_)
    # integer :: i,j,k

    for k in xrange(0,nz): # k = 1,nz
        for j in xrange(js_pe,je_pe):
            for i in xrange(is_pe-1,ie_pe):
                adv_fe[i,j,k] = 0.5*(var[i,j,k] + var[i+1,j,k])*u[i,j,k,tau]*maskU[i,j,k]

    for k in xrange(0,nz): # k = 1,nz
        for j in xrange(js_pe-1,je_pe):
            for i in xrange(is_pe,ie_pe):
                adv_fn[i,j,k] = cosu[j]*0.5*(var[i,j,k] + var[i,j+1,k])*v[i,j,k,tau]*maskV[i,j,k]

    for k in xrange(0,nz-1): # k = 1,nz-1
        for j in xrange(js_pe,je_pe):
            for i in xrange(is_pe,ie_pe):
                adv_ft[i,j,k] = 0.5*(var[i,j,k] + var[i,j,k+1])*w[i,j,k,tau]*maskW[i,j,k]
                
    adv_ft[:,:,nz] = 0.0

def adv_flux_superbee(is_,ie_,js_,je_,nz_,adv_fe,adv_fn,adv_ft,var,pyom):
    """
    from MITgcm
    Calculates advection of a tracer
    using second-order interpolation with a flux limiter:
    \begin{equation*}
    F^x_{adv} = U \overline{ \theta }^i
    - \frac{1}{2} \left([ 1 - \psi(C_r) ] |U|
       + U \frac{u \Delta t}{\Delta x_c} \psi(C_r)
                 \right) \delta_i \theta
    \end{equation*}
    where the $\psi(C_r)$ is the limiter function and $C_r$ is
    the slope ratio.
    """

    # integer, intent(in) :: is_,ie_,js_,je_,nz_
    # real*8, intent(inout) :: adv_fe(is_:ie_,js_:je_,nz_), adv_fn(is_:ie_,js_:je_,nz_)
    # real*8, intent(inout) :: adv_ft(is_:ie_,js_:je_,nz_),    var(is_:ie_,js_:je_,nz_)
    # integer :: i,j,k,km1,kp2
    # real*8 :: Rjp,Rj,Rjm,uCFL = 0.5,Cr

    # Statement function to describe flux limiter
    # Upwind        Limiter = lambda Cr: 0.
    # Lax-Wendroff  Limiter = lambda Cr: 1.
    # Suberbee      Limiter = lambda Cr: max(0.,max(min(1.,2*Cr),min(2.,Cr)))
    # Sweby         Limiter = lambda Cr: max(0.,max(min(1.,1.5*Cr),min(1.5.,Cr)))
    # real*8 :: Limiter
    Limiter = lambda Cr: max(0.,max(min(1.,2.*Cr), min(2.,Cr)))
    # ! Limiter = lambda Cr: max(0.,max(min(1.,1.5*Cr), min(1.5,Cr)))

    for k in xrange(1,nz): # k = 1,nz
        for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
            for i in xrange(is_pe-1,ie_pe): # i = is_pe-1,ie_pe
                uCFL = abs(u[i,j,k,tau]*dt_tracer/(cost[j]*dxt[i]))
                Rjp = (var[i+2,j,k]-var[i+1,j,k])*maskU[i+1,j,k]
                Rj = (var[i+1,j,k]-var[i,j,k])*maskU[i,j,k]
                Rjm = (var[i,j,k]-var[i-1,j,k])*maskU[i-1,j,k]
                if Rj != 0.:
                    if u[i,j,k,tau] > 0:
                        Cr = Rjm/Rj
                    else:
                        Cr = Rjp/Rj
                else:
                    if u[i,j,k,tau] > 0:
                        Cr = Rjm*1e20
                    else:
                        Cr = Rjp*1e20
                Cr = Limiter(Cr)
                adv_fe[i,j,k] = u[i,j,k,tau]*(var[i+1,j,k]+var[i,j,k])*0.5 \
                                - abs(u[i,j,k,tau])*((1.-Cr)+uCFL*Cr)*Rj*0.5

    for k in xrange(1,nz): # k = 1,nz
        for j in xrange(js_pe-1,je_pe): # j = js_pe-1,je_pe
            for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
                Rjp = (var[i,j+2,k]-var[i,j+1,k])*maskV[i,j+1,k]
                Rj = (var[i,j+1,k]-var[i,j,k])*maskV[i,j,k]
                Rjm = (var[i,j,k]-var(i,j-1,k))*maskV(i,j-1,k)
                uCFL = abs(cosu[j]*v[i,j,k,tau]*dt_tracer/(cost[j]*dyt[j]))
                if Rj != 0.:
                    if v[i,j,k,tau] > 0:
                        Cr = Rjm/Rj
                    else:
                        Cr = Rjp/Rj
                else:
                    if v[i,j,k,tau] > 0:
                        Cr = Rjm*1e20
                    else:
                        Cr = Rjp*1e20
                Cr = Limiter(Cr)
                adv_fn[i,j,k] = cosu[j]*v[i,j,k,tau]*(var[i,j+1,k]+var[i,j,k])*0.5   \
                                -abs(cosu[j]*v[i,j,k,tau])*((1.-Cr)+uCFL*Cr)*Rj*0.5

    for k in xrange(1,nz): # k = 1,nz-1
        kp2 = min(nz,k+2); #if (kp2>np) kp2 = 3
        km1 = max(1,k-1) #if (km1<1) km1 = np-2
        for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
            for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
                Rjp=(var[i,j,kp2]-var[i,j,k+1])*maskW[i,j,k+1]
                Rj =(var[i,j,k+1]-var[i,j,k])*maskW[i,j,k]
                Rjm=(var[i,j,k]-var[i,j,km1])*maskW[i,j,km1]
                uCFL = ABS(w[i,j,k,tau]*dt_tracer/dzt[k])
                if Rj != 0.:
                    if w[i,j,k,tau] > 0:
                        Cr = Rjm/Rj
                    else:
                        Cr = Rjp/Rj
                else:
                    if w[i,j,k,tau] > 0:
                        Cr = Rjm*1e20
                    else:
                        Cr = Rjp*1e20
                Cr = Limiter(Cr)
                adv_ft[i,j,k] = w[i,j,k,tau]*(var[i,j,k+1]+var[i,j,k])*0.5 \
                                -abs(w[i,j,k,tau])*((1.-Cr)+uCFL*Cr)*Rj*0.5
    adv_ft[:,:,nz] = 0.0


def calculate_velocity_on_wgrid(pyom):
    """
    calculates advection velocity for tracer on W grid
    """
    #integer :: i,j,k
    #real*8 :: fxa,fxb

    # lateral advection velocities on W grid
    for k in xrange(1,nz): # k = 1,nz-1
        u_wgrid[:,:,k] = u[:,:,k+1,tau]*maskU[:,:,k+1]*0.5*dzt[k+1]/dzw[k] + u[:,:,k,tau]*maskU[:,:,k]*0.5*dzt[k]/dzw[k]
        v_wgrid[:,:,k] = v[:,:,k+1,tau]*maskV[:,:,k+1]*0.5*dzt[k+1]/dzw[k] + v[:,:,k,tau]*maskV[:,:,k]*0.5*dzt[k]/dzw[k]
    k = nz
    u_wgrid[:,:,k] = u[:,:,k,tau]*maskU[:,:,k]*0.5*dzt[k]/dzw[k]
    v_wgrid[:,:,k] = v[:,:,k,tau]*maskV[:,:,k]*0.5*dzt[k]/dzw[k]

    # redirect velocity at bottom and at topography
    k = 0 # k = 1
    u_wgrid[:,:,k] = u_wgrid[:,:,k] + u[:,:,k,tau]*maskU[:,:,k]*0.5*dzt[k]/dzw[k]
    v_wgrid[:,:,k] = v_wgrid[:,:,k] + v[:,:,k,tau]*maskV[:,:,k]*0.5*dzt[k]/dzw[k]
    for k in xrange(1,nz): # k = 1,nz-1
        for j in xrange(js_pe-onx,je_pe+onx): # j = js_pe-onx,je_pe+onx
            for i in xrange(is_pe-onx,ie_pe+onx-1): # i = is_pe-onx,ie_pe+onx-1
                if maskW[i,j,k]*maskW[i+1,j,k] == 0:
                    u_wgrid[i,j,k+1] = u_wgrid[i,j,k+1]+u_wgrid[i,j,k]*dzw[k]/dzw[k+1]
                    u_wgrid[i,j,k] = 0

    for j in xrange(js_pe-onx,je_pe+onx-1): # j = js_pe-onx,je_pe+onx-1
        for i in xrange(is_pe-onx,ie_pe+onx): # i = is_pe-onx,ie_pe+onx
            if maskW[i,j,k]*maskW[i,j+1,k] == 0 :
                v_wgrid[i,j,k+1] = v_wgrid[i,j,k+1]+v_wgrid[i,j,k]*dzw[k]/dzw[k+1]
                v_wgrid[i,j,k] = 0

    # vertical advection velocity on W grid from continuity
    k = 0 # k = 1
    w_wgrid[:,:,k] = 0
    for k in xrange(1,nz): # k = 1,nz
        for j in xrange(js_pe-onx+1,je_pe+onx): # j = js_pe-onx+1,je_pe+onx
            for i in xrange(is_pe-onx+1,ie_pe+onx): # i = is_pe-onx+1,ie_pe+onx
                w_wgrid[i,j,k] = w_wgrid[i,j,max(1,k-1)]-dzw[k]* \
                    ((u_wgrid[i,j,k]-u_wgrid[i-1,j,k])/(cost[j]*dxt[i]) \
                    +(cosu[j]*v_wgrid[i,j,k]-cosu[j-1]*v_wgrid(i,j-1,k))/(cost[j]*dyt[j]))

 # test continuity
 #if  modulo(itt*dt_tracer,ts_monint) < dt_tracer .and. .false.:
 # fxa = 0;fxb = 0;
 # for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
 #  for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
 #    fxa = fxa + w_wgrid(i,j,nz) *area_t(i,j)
 #    fxb = fxb +   w(i,j,nz,tau) *area_t(i,j)
 #  enddo
 # enddo
 # call global_sum(fxa); call global_sum(fxb);
 # if (my_pe==0) print'(a,e12.6,a)',' transport at sea surface on t grid = ',fxb,' m^3/s'
 # if (my_pe==0) print'(a,e12.6,a)',' transport at sea surface on w grid = ',fxa,' m^3/s'
#
#
#  fxa = 0;fxb = 0;
#  for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
#   for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
#     fxa = fxa + w_wgrid(i,j,nz)**2 *area_t(i,j)
#     fxb = fxb +   w(i,j,nz,tau)**2 *area_t(i,j)
#   enddo
#  enddo
#  call global_sum(fxa); call global_sum(fxb);
#  if (my_pe==0) print'(a,e12.6,a)',' w variance on t grid = ',fxb,' (m^3/s)^2'
#  if (my_pe==0) print'(a,e12.6,a)',' w variance on w grid = ',fxa,' (m^3/s)^2'

# endif


def adv_flux_superbee_wgrid(is_,ie_,js_,je_,nz_,adv_fe,adv_fn,adv_ft,var,pyom):
    """
    Calculates advection of a tracer defined on Wgrid
    """
    # integer, intent(in) :: is_,ie_,js_,je_,nz_
    # real*8, intent(inout) :: adv_fe(is_:ie_,js_:je_,nz_), adv_fn(is_:ie_,js_:je_,nz_)
    # real*8, intent(inout) :: adv_ft(is_:ie_,js_:je_,nz_),    var(is_:ie_,js_:je_,nz_)
    # integer :: i,j,k,km1,kp2,kp1
    # real*8 :: Rjp,Rj,Rjm,uCFL = 0.5,Cr

    # Statement function to describe flux limiter
    # Upwind        Limiter = lambda Cr: 0.
    # Lax-Wendroff  Limiter = lambda Cr: 1.
    # Suberbee      Limiter = lambda Cr: max(0.,max(min(1.,2*Cr),min(2.,Cr)))
    # Sweby         Limiter = lambda Cr: max(0.,max(min(1.,1.5*Cr),min(1.5.,Cr)))
    # real*8 :: Limiter

    Limiter = lambda Cr: max(0.,max(min(1.,2.*Cr), min(2.,Cr)))
    # real*8 :: maskUtr,maskVtr,maskWtr
    maskUtr[i,j,k] = maskW[i+1,j,k]*maskW[i,j,k]
    maskVtr[i,j,k] = maskW[i,j+1,k]*maskW[i,j,k]
    maskWtr[i,j,k] = maskW[i,j,k+1]*maskW[i,j,k]

    for k in xrange(1,nz): # k = 1,nz
        for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
            for i in xrange(is_pe-1,ie_pe): # i = is_pe-1,ie_pe
                uCFL = ABS(u_wgrid[i,j,k]*dt_tracer/(cost[j]*dxt[i]))
                Rjp = (var[i+2,j,k]-var[i+1,j,k])*maskUtr[i+1,j,k]
                Rj = (var[i+1,j,k]-var[i,j,k])*maskUtr[i,j,k]
                Rjm = (var[i,j,k]-var[i-1,j,k])*maskUtr[i-1,j,k]
                if Rj != 0.:
                    if u_wgrid[i,j,k] > 0:
                        Cr = Rjm/Rj
                    else:
                        Cr = Rjp/Rj
                else:
                    if u_wgrid[i,j,k] > 0:
                        Cr = Rjm*1e20
                    else:
                        Cr = Rjp*1e20
                Cr = Limiter(Cr)
                adv_fe[i,j,k] = u_wgrid[i,j,k]*(var[i+1,j,k]+var[i,j,k])*0.5   \
                                -abs(u_wgrid[i,j,k])*((1.-Cr)+uCFL*Cr)*Rj*0.5

    for k in xrange(1,nz): # k = 1,nz
        for j in xrange(js_pe-1,je_pe): # j = js_pe-1,je_pe
            for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
                Rjp = (var[i,j+2,k]-var[i,j+1,k])*maskVtr[i,j+1,k]
                Rj = (var[i,j+1,k]-var[i,j,k])*maskVtr[i,j,k]
                Rjm = (var[i,j,k]-var(i,j-1,k))*maskVtr(i,j-1,k)
                uCFL = abs(cosu[j]*v_wgrid[i,j,k]*dt_tracer/(cost[j]*dyt[j]))
                if Rj != 0.:
                    if v_wgrid[i,j,k] > 0:
                        Cr = Rjm/Rj
                    else:
                        Cr = Rjp/Rj
                else:
                    if v_wgrid[i,j,k] > 0:
                        Cr = Rjm*1e20
                    else:
                        Cr = Rjp*1e20
                Cr = Limiter(Cr)
                adv_fn[i,j,k] = cosu[j]*v_wgrid[i,j,k]*(var[i,j+1,k]+var[i,j,k])*0.5   \
                                -abs(cosu[j]*v_wgrid[i,j,k])*((1.-Cr)+uCFL*Cr)*Rj*0.5

    for k in xrange(1,nz): # k = 1,nz-1
        kp1 = min(nz-1,k+1)
        kp2 = min(nz,k+2);
        km1 = max(1,k-1)
        for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
            for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
                Rjp = (var[i,j,kp2]-var[i,j,k+1])*maskWtr(i,j,kp1)
                Rj = (var[i,j,k+1]-var[i,j,k])*maskWtr[i,j,k]
                Rjm = (var[i,j,k]-var[i,j,km1])*maskWtr[i,j,km1]
                uCFL = abs(w_wgrid[i,j,k]*dt_tracer/dzw[k])
                if Rj != 0.:
                    if w_wgrid[i,j,k] > 0:
                        Cr = Rjm/Rj
                    else:
                        Cr = Rjp/Rj
                else:
                    if w_wgrid[i,j,k] > 0:
                        Cr = Rjm*1e20
                    else:
                        Cr = Rjp*1e20
                Cr = Limiter(Cr)
                adv_ft[i,j,k] = w_wgrid[i,j,k]*(var[i,j,k+1]+var[i,j,k])*0.5   \
                                -abs(w_wgrid[i,j,k])*((1.-Cr)+uCFL*Cr)*Rj*0.5
    adv_ft[:,:,nz] = 0.0


def adv_flux_upwind_wgrid(is_,ie_,js_,je_,nz_,adv_fe,adv_fn,adv_ft,var,pyom):
    """
    Calculates advection of a tracer defined on Wgrid
    """
    #  integer, intent(in) :: is_,ie_,js_,je_,nz_
    #  real*8, intent(inout) :: adv_fe(is_:ie_,js_:je_,nz_), adv_fn(is_:ie_,js_:je_,nz_)
    #  real*8, intent(inout) :: adv_ft(is_:ie_,js_:je_,nz_),    var(is_:ie_,js_:je_,nz_)
    #  integer :: i,j,k
    #  real*8 :: Rj
    #  real*8 :: maskUtr,maskVtr,maskWtr
    maskUtr[i,j,k] = maskW[i+1,j,k]*maskW[i,j,k]
    maskVtr[i,j,k] = maskW[i,j+1,k]*maskW[i,j,k]
    maskWtr[i,j,k] = maskW[i,j,k+1]*maskW[i,j,k]

    for k in xrange(1,nz): # k = 1,nz
        for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
            for i in xrange(is_pe-1,ie_pe): # i = is_pe-1,ie_pe
                Rj = (var[i+1,j,k]-var[i,j,k])*maskUtr[i,j,k]
                adv_fe[i,j,k] = u_wgrid[i,j,k]*(var[i+1,j,k]+var[i,j,k])*0.5 - abs(u_wgrid[i,j,k])*Rj*0.5

    for k in xrange(1,nz): # k = 1,nz
        for j in xrange(js_pe-1,je_pe): # j = js_pe-1,je_pe
            for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
                Rj = (var[i,j+1,k]-var[i,j,k])*maskVtr[i,j,k]
                adv_fn[i,j,k] = cosu[j]*v_wgrid[i,j,k]*(var[i,j+1,k]+var[i,j,k])*0.5 -ABS(cosu[j]*v_wgrid[i,j,k])*Rj*0.5

    for k in xrange(1,nz): # k = 1,nz-1
        for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
            for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
                Rj = (var[i,j,k+1]-var[i,j,k])*maskWtr[i,j,k]
                adv_ft[i,j,k] = w_wgrid[i,j,k]*(var[i,j,k+1]+var[i,j,k])*0.5 - abs(w_wgrid[i,j,k])*Rj*0.5

    adv_ft[:,:,nz] = 0.0

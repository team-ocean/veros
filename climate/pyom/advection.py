import numpy as np

from climate import make_slice

def adv_flux_2nd(adv_fe,adv_fn,adv_ft,var,pyom):
    """
    2th order advective tracer flux
    """
    # integer, intent(in) :: is_,ie_,js_,je_,nz_
    # real*8, intent(inout) :: adv_fe(is_:ie_,js_:je_,nz_), adv_fn(is_:ie_,js_:je_,nz_)
    # real*8, intent(inout) :: adv_ft(is_:ie_,js_:je_,nz_),    var(is_:ie_,js_:je_,nz_)
    # integer :: i,j,k

    i, ii = make_slice(1,-2)
    j, jj = make_slice(2,-2)
    k, kk = make_slice()
    adv_fe[i, j, k] = 0.5*(var[i,j,k] + var[ii+1,j,k]) * pyom.u[i,j,k,pyom.tau] * pyom.maskU[i,j,k]

    i, ii = make_slice(2,-2)
    j, jj = make_slice(1,-2)
    k, kk = make_slice()
    adv_fn[i,j,k] = pyom.cosu[None,j,None] * 0.5 * (var[i,j,k] + var[i,jj+1,k]) * pyom.v[i,j,k,pyom.tau] * pyom.maskV[i,j,k]

    i, ii = make_slice(2,-2)
    j, jj = make_slice(2,-2)
    k, kk = make_slice(None,-1)
    adv_ft[i,j,k] = 0.5 * (var[i,j,k] + var[i,j,kk+1]) * pyom.w[i,j,k,pyom.tau] * pyom.maskW[i,j,k]
    adv_ft[:,:,-1] = 0.

    #for k in xrange(pyom.nz): # k = 1,nz
    #    for j in xrange(pyom.js_pe,pyom.je_pe):
    #        for i in xrange(pyom.is_pe-1,pyom.ie_pe):
    #            adv_fe[i,j,k] = 0.5*(var[i,j,k] + var[i+1,j,k])*pyom.u[i,j,k,pyom.tau]*pyom.maskU[i,j,k]

    #for k in xrange(pyom.nz): # k = 1,nz
    #    for j in xrange(pyom.js_pe-1,pyom.je_pe):
    #        for i in xrange(pyom.is_pe,pyom.ie_pe):
    #            adv_fn[i,j,k] = pyom.cosu[j]*0.5*(var[i,j,k] + var[i,j+1,k])*pyom.v[i,j,k,pyom.tau]*pyom.maskV[i,j,k]

    #for k in xrange(pyom.nz-1): # k = 1,nz-1
    #    for j in xrange(pyom.js_pe,pyom.je_pe):
    #        for i in xrange(pyom.is_pe,pyom.ie_pe):
    #            adv_ft[i,j,k] = 0.5*(var[i,j,k] + var[i,j,k+1])*pyom.w[i,j,k,pyom.tau]*pyom.maskW[i,j,k]

    #adv_ft[:,:,pyom.nz-1] = 0.0

def _calc_cr(rjp,rj,rjm,vel):
    cr = np.zeros_like(rj)
    mask_rj = rj == 0.
    mask_vel = vel > 0
    mask1 = [~mask_rj & mask_vel]
    cr[mask1] = rjm[mask1] / rj[mask1]
    mask2 = [~mask_rj & ~mask_vel]
    cr[mask2] = rjp[mask2] / rj[mask2]
    mask3 = [mask_rj & mask_vel]
    cr[mask3] = rjm[mask3] * 1e20
    mask4 = [mask_rj & ~mask_vel]
    cr[mask4] = rjp[mask4] * 1e20
    return cr

def _pad_z_edges(array):
    """
    Pads the third axis of an array by repeating its edge values
    """
    a = list(array.shape)
    a[2] += 2
    newarray = np.empty(a)
    newarray[:,:,1:-1,...] = array
    newarray[:,:,0,...] = array[:,:,0,...]
    newarray[:,:,-1,...] = array[:,:,-1,...]
    return newarray

def adv_flux_superbee(adv_fe,adv_fn,adv_ft,var,pyom):
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
    # real*8 :: rjp,rj,rjm,uCFL = 0.5,cr

    # Statement function to describe flux limiter
    # Upwind        limiter = lambda cr: 0.
    # Lax-Wendroff  limiter = lambda cr: 1.
    # Suberbee      limiter = lambda cr: max(0.,max(min(1.,2*cr),min(2.,cr)))
    # Sweby         limiter = lambda cr: max(0.,max(min(1.,1.5*cr),min(1.5.,cr)))
    # real*8 :: limiter
    limiter = lambda cr: np.maximum(0.,np.maximum(np.minimum(1.,2*cr), np.minimum(2.,cr)))
    # ! limiter = lambda cr: max(0.,max(min(1.,1.5*cr), min(1.5,cr)))

    uCFL = np.abs(pyom.u[1:-2,2:-2,:,pyom.tau] * pyom.dt_tracer / (pyom.cost[None,2:-2,None] * pyom.dxt[1:-2,None,None]))
    rjp = (var[3:,2:-2,:] - var[2:-1,2:-2,:]) * pyom.maskU[2:-1,2:-2,:]
    rj = (var[2:-1,2:-2,:] - var[1:-2,2:-2,:]) * pyom.maskU[1:-2,2:-2,:]
    rjm = (var[1:-2,2:-2,:] - var[:-3,2:-2,:]) * pyom.maskU[:-3,2:-2,:]

    cr = limiter(_calc_cr(rjp,rj,rjm,pyom.u[1:-2,2:-2,:,pyom.tau]))
    adv_fe[1:-2,2:-2,:] = pyom.u[1:-2,2:-2,:,pyom.tau] * (var[2:-1,2:-2,:] + var[1:-2,2:-2,:]) * 0.5 \
                          - np.abs(pyom.u[1:-2,2:-2,:,pyom.tau]) * ((1.-cr) + uCFL*cr) * rj * 0.5

    #for k in xrange(pyom.nz): # k = 1,nz
    #    for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
    #        for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
    #            uCFL = abs(pyom.u[i,j,k,pyom.tau]*dt_tracer/(pyom.cost[j]*pyom.dxt[i]))
    #            rjp = (var[i+2,j,k]-var[i+1,j,k])*pyom.maskU[i+1,j,k]
    #            rj = (var[i+1,j,k]-var[i,j,k])*pyom.maskU[i,j,k]
    #            rjm = (var[i,j,k]-var[i-1,j,k])*pyom.maskU[i-1,j,k]
    #            if rj != 0.:
    #                if pyom.u[i,j,k,pyom.tau] > 0:
    #                    cr = rjm/rj
    #                else:
    #                    cr = rjp/rj
    #            else:
    #                if pyom.u[i,j,k,pyom.tau] > 0:
    #                    cr = rjm*1e20
    #                else:
    #                    cr = rjp*1e20
    #            cr = limiter(cr)
    #            adv_fe[i,j,k] = pyom.u[i,j,k,pyom.tau]*(var[i+1,j,k]+var[i,j,k])*0.5 \
    #                            - abs(pyom.u[i,j,k,pyom.tau])*((1.-cr)+uCFL*cr)*rj*0.5

    uCFL = np.abs(pyom.cosu[None,1:-2,None] * pyom.v[2:-2,1:-2,:,pyom.tau] * pyom.dt_tracer / (pyom.cost[None,1:-2,None] * pyom.dyt[None,1:-2,None]))
    rjp = (var[2:-2,3:,:] - var[2:-2,2:-1,:]) * pyom.maskV[2:-2,2:-1,:]
    rj = (var[2:-2,2:-1,:] - var[2:-2,1:-2,:]) * pyom.maskV[2:-2,1:-2,:]
    rjm = (var[2:-2,1:-2,:] - var[2:-2,:-3,:]) * pyom.maskV[2:-2,:-3,:]

    cr = limiter(_calc_cr(rjp,rj,rjm,pyom.v[2:-2,1:-2,:,pyom.tau]))
    adv_fn[2:-2,1:-2,:] = pyom.cosu[None,1:-2,None] * pyom.v[2:-2,1:-2,:,pyom.tau] * (var[2:-2,2:-1,:]+var[2:-2,1:-2,:])*0.5   \
                    - np.abs(pyom.cosu[None,1:-2,None] * pyom.v[2:-2,1:-2,:,pyom.tau]) * ((1.-cr)+uCFL*cr)*rj*0.5

    #for k in xrange(pyom.nz-1): # k = 1,nz
    #    for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
    #        for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
    #            rjp = (var[i,j+2,k]-var[i,j+1,k])*pyom.maskV[i,j+1,k]
    #            rj = (var[i,j+1,k]-var[i,j,k])*pyom.maskV[i,j,k]
    #            rjm = (var[i,j,k]-var[i,j-1,k])*pyom.maskV[i,j-1,k]
    #            uCFL = abs(pyom.cosu[j]*pyom.v[i,j,k,pyom.tau]*pyom.dt_tracer/(pyom.cost[j]*pyom.dyt[j]))
    #            if rj != 0.:
    #                if pyom.v[i,j,k,pyom.tau] > 0:
    #                    cr = rjm/rj
    #                else:
    #                    cr = rjp/rj
    #            else:
    #                if pyom.v[i,j,k,pyom.tau] > 0:
    #                    cr = rjm*1e20
    #                else:
    #                    cr = rjp*1e20
    #            cr = limiter(cr)
    #            adv_fn[i,j,k] = pyom.cosu[j]*pyom.v[i,j,k,pyom.tau]*(var[i,j+1,k]+var[i,j,k])*0.5   \
    #                            -abs(pyom.cosu[j]*pyom.v[i,j,k,pyom.tau])*((1.-cr)+uCFL*cr)*rj*0.5

    var_pad = _pad_z_edges(var)
    maskW_pad = _pad_z_edges(pyom.maskW)
    uCFL = np.abs(pyom.w[2:-2,2:-2,:-1,pyom.tau] * pyom.dt_tracer / pyom.dzt[None,None,:-1])
    rjp = (var_pad[2:-2,2:-2,3:] - var_pad[2:-2,2:-2,2:-1]) * maskW_pad[2:-2,2:-2,2:-1]
    rj = (var_pad[2:-2,2:-2,2:-1] - var_pad[2:-2,2:-2,1:-2]) * maskW_pad[2:-2,2:-2,1:-2]
    rjm = (var_pad[2:-2,2:-2,1:-2] - var_pad[2:-2,2:-2,:-3] * maskW_pad[2:-2,2:-2,:-3])

    cr = limiter(_calc_cr(rjp,rj,rjm,pyom.w[2:-2,2:-2,:-1,pyom.tau]))
    adv_ft[2:-2,2:-2,:-1] = pyom.w[2:-2,2:-2,:-1,pyom.tau] * (var[2:-2,2:-2,1:] + var[2:-2,2:-2,:-1]) *0.5 \
                    - np.abs(pyom.w[2:-2,2:-2,:-1,pyom.tau]) * ((1.-cr) + uCFL * cr) * rj * 0.5
    adv_ft[:,:,pyom.nz-1] = 0.0
    #for k in xrange(pyom.nz-1): # k = 1,nz-1
    #    kp2 = min(pyom.nz-1,k+2); #if (kp2>np) kp2 = 3
    #    km1 = max(0,k-1) #if (km1<1) km1 = np-2
    #    for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
    #        for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
    #            rjp=(var[i,j,kp2]-var[i,j,k+1])*pyom.maskW[i,j,k+1]
    #            rj =(var[i,j,k+1]-var[i,j,k])*pyom.maskW[i,j,k]
    #            rjm=(var[i,j,k]-var[i,j,km1])*pyom.maskW[i,j,km1]
    #            uCFL = abs(pyom.w[i,j,k,pyom.tau]*pyom.dt_tracer/pyom.dzt[k])
    #            if rj != 0.:
    #                if pyom.w[i,j,k,pyom.tau] > 0:
    #                    cr = rjm/rj
    #                else:
    #                    cr = rjp/rj
    #            else:
    #                if pyom.w[i,j,k,pyom.tau] > 0:
    #                    cr = rjm*1e20
    #                else:
    #                    cr = rjp*1e20
    #            cr = limiter(cr)
    #            adv_ft[i,j,k] = pyom.w[i,j,k,pyom.tau]*(var[i,j,k+1]+var[i,j,k])*0.5 \
    #                            -abs(pyom.w[i,j,k,pyom.tau])*((1.-cr)+uCFL*cr)*rj*0.5
    #adv_ft[:,:,pyom.nz-1] = 0.0


def calculate_velocity_on_wgrid(pyom):
    """
    calculates advection velocity for tracer on W grid
    """
    #integer :: i,j,k
    #real*8 :: fxa,fxb

    # lateral advection velocities on W grid
    pyom.u_wgrid[:,:,:-1] = pyom.u[:,:,1:,pyom.tau] * pyom.maskU[:,:,1:] * 0.5 * pyom.dzt[None,None,1:] / pyom.dzw[None,None,:-1] \
                          + pyom.u[:,:,:-1,pyom.tau] * pyom.maskU[:,:,:-1] * 0.5 * pyom.dzt[None,None,:-1] / pyom.dzw[None,None,:-1]
    pyom.v_wgrid[:,:,:-1] = pyom.v[:,:,1:,pyom.tau] * pyom.maskV[:,:,1:] * 0.5 * pyom.dzt[None,None,1:] / pyom.dzw[None,None,:-1] \
                          + pyom.v[:,:,:-1,pyom.tau] * pyom.maskV[:,:,:-1] * 0.5 * pyom.dzt[None,None,:-1] / pyom.dzw[None,None,:-1]
    pyom.u_wgrid[:,:,-1] = pyom.u[:,:,-1,pyom.tau] * pyom.maskU[:,:,-1] * 0.5 * pyom.dzt[-1] / pyom.dzw[-1]
    pyom.v_wgrid[:,:,-1] = pyom.v[:,:,-1,pyom.tau] * pyom.maskV[:,:,-1] * 0.5 * pyom.dzt[-1] / pyom.dzw[-1]

    #for k in xrange(pyom.nz-1): # k = 1,nz-1
    #    pyom.u_wgrid[:,:,k] = pyom.u[:,:,k+1,pyom.tau]*pyom.maskU[:,:,k+1]*0.5*pyom.dzt[k+1]/pyom.dzw[k] + pyom.u[:,:,k,pyom.tau]*pyom.maskU[:,:,k]*0.5*pyom.dzt[k]/pyom.dzw[k]
    #    pyom.v_wgrid[:,:,k] = pyom.v[:,:,k+1,pyom.tau]*pyom.maskV[:,:,k+1]*0.5*pyom.dzt[k+1]/pyom.dzw[k] + pyom.v[:,:,k,pyom.tau]*pyom.maskV[:,:,k]*0.5*pyom.dzt[k]/pyom.dzw[k]
    #k = pyom.nz-1
    #pyom.u_wgrid[:,:,k] = pyom.u[:,:,k,pyom.tau]*pyom.maskU[:,:,k]*0.5*pyom.dzt[k]/pyom.dzw[k]
    #pyom.v_wgrid[:,:,k] = pyom.v[:,:,k,pyom.tau]*pyom.maskV[:,:,k]*0.5*pyom.dzt[k]/pyom.dzw[k]

    # redirect velocity at bottom and at topography
    pyom.u_wgrid[:,:,0] = pyom.u_wgrid[:,:,0] + pyom.u[:,:,0,pyom.tau] * pyom.maskU[:,:,0] * 0.5 * pyom.dzt[0] / pyom.dzw[0]
    pyom.v_wgrid[:,:,0] = pyom.v_wgrid[:,:,0] + pyom.v[:,:,0,pyom.tau] * pyom.maskV[:,:,0] * 0.5 * pyom.dzt[0] / pyom.dzw[0]
    for k in xrange(pyom.nz-1): #TODO: vectorize
        mask = pyom.maskW[:-1, :, k] * pyom.maskW[1:, :, k] == 0
        pyom.u_wgrid[:-1, :, k+1][mask] = (pyom.u_wgrid[:-1, :, k+1] + pyom.u_wgrid[:-1, :, k] * pyom.dzw[None, None, k] / pyom.dzw[None, None, k+1])[mask]
        pyom.u_wgrid[:-1, :, k][mask] = 0.

        mask = pyom.maskW[:, :-1, k] * pyom.maskW[:, 1:, k] == 0
        pyom.v_wgrid[:, :-1, k+1][mask] = (pyom.v_wgrid[:, :-1, k+1] + pyom.v_wgrid[:, :-1, k] * pyom.dzw[None, None, k] / pyom.dzw[None, None, k+1])[mask]
        pyom.v_wgrid[:, :-1, k][mask] = 0.

    #k = 0 # k = 1
    #pyom.u_wgrid[:,:,k] = pyom.u_wgrid[:,:,k] + pyom.u[:,:,k,pyom.tau]*pyom.maskU[:,:,k]*0.5*pyom.dzt[k]/pyom.dzw[k]
    #pyom.v_wgrid[:,:,k] = pyom.v_wgrid[:,:,k] + pyom.v[:,:,k,pyom.tau]*pyom.maskV[:,:,k]*0.5*pyom.dzt[k]/pyom.dzw[k]
    #for k in xrange(pyom.nz-1): # k = 1,nz-1
    #    for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx): # j = js_pe-onx,je_pe+onx
    #        for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx-1): # i = is_pe-onx,ie_pe+onx-1
    #            if pyom.maskW[i,j,k]*pyom.maskW[i+1,j,k] == 0:
    #                pyom.u_wgrid[i,j,k+1] = pyom.u_wgrid[i,j,k+1]+pyom.u_wgrid[i,j,k]*pyom.dzw[k]/pyom.dzw[k+1]
    #                pyom.u_wgrid[i,j,k] = 0.
    #
    #for j in xrange(pyom.js_pe-pyom.onx,pyom.je_pe+pyom.onx-1): # j = js_pe-onx,je_pe+onx-1
    #    for i in xrange(pyom.is_pe-pyom.onx,pyom.ie_pe+pyom.onx): # i = is_pe-onx,ie_pe+onx
    #        if pyom.maskW[i,j,k]*pyom.maskW[i,j+1,k] == 0 :
    #            pyom.v_wgrid[i,j,k+1] = pyom.v_wgrid[i,j,k+1]+pyom.v_wgrid[i,j,k]*pyom.dzw[k]/pyom.dzw[k+1]
    #            pyom.v_wgrid[i,j,k] = 0.

    # vertical advection velocity on W grid from continuity
    pyom.w_wgrid[:,:,0] = 0.
    w_wgrid_pad = _pad_z_edges(pyom.w_wgrid)
    pyom.w_wgrid[1:, 1:, :] = w_wgrid_pad[1:, 1:, :-2] - pyom.dzw[None, None, :] * \
                              ((pyom.u_wgrid[1:, 1:, :] - pyom.u_wgrid[:-1, 1:, :]) / (pyom.cost[None, 1:, None] * pyom.dxt[1:, None, None]) \
                               + (pyom.cosu[None, 1:, None] * pyom.v_wgrid[1:, 1:, :] - pyom.cosu[None, :-1, None] * pyom.v_wgrid[1:, :-1, :]) \
                                     / (pyom.cost[None, 1:, None] * pyom.dyt[None, 1:, None]))

    #k = 0 # k = 1
    #pyom.w_wgrid[:,:,k] = 0.
    #for k in xrange(pyom.nz): # k = 1,nz
    #    for j in xrange(pyom.js_pe-pyom.onx+1,pyom.je_pe+pyom.onx): # j = js_pe-onx+1,je_pe+onx
    #        for i in xrange(pyom.is_pe-pyom.onx+1,pyom.ie_pe+pyom.onx): # i = is_pe-onx+1,ie_pe+onx
    #            pyom.w_wgrid[i,j,k] = pyom.w_wgrid[i,j,max(1,k-1)]-pyom.dzw[k]* \
    #                ((pyom.u_wgrid[i,j,k]-pyom.u_wgrid[i-1,j,k])/(pyom.cost[j]*pyom.dxt[i]) \
    #                +(pyom.cosu[j]*pyom.v_wgrid[i,j,k]-pyom.cosu[j-1]*pyom.v_wgrid(i,j-1,k))/(pyom.cost[j]*pyom.dyt[j]))

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


def adv_flux_superbee_wgrid(adv_fe,adv_fn,adv_ft,var,pyom):
    """
    Calculates advection of a tracer defined on Wgrid
    """
    # integer, intent(in) :: is_,ie_,js_,je_,nz_
    # real*8, intent(inout) :: adv_fe(is_:ie_,js_:je_,nz_), adv_fn(is_:ie_,js_:je_,nz_)
    # real*8, intent(inout) :: adv_ft(is_:ie_,js_:je_,nz_),    var(is_:ie_,js_:je_,nz_)
    # integer :: i,j,k,km1,kp2,kp1
    # real*8 :: rjp,rj,rjm,uCFL = 0.5,cr

    # Statement function to describe flux limiter
    # Upwind        limiter = lambda cr: 0.
    # Lax-Wendroff  limiter = lambda cr: 1.
    # Suberbee      limiter = lambda cr: max(0.,max(min(1.,2*cr),min(2.,cr)))
    # Sweby         limiter = lambda cr: max(0.,max(min(1.,1.5*cr),min(1.5.,cr)))
    # real*8 :: limiter
    limiter = lambda cr: np.maximum(0.,np.maximum(np.minimum(1.,2.*cr), np.minimum(2.,cr)))
    # real*8 :: maskUtr,maskVtr,maskWtr
    maskUtr = lambda i,j,k: pyom.maskW[i+1,j,k] * pyom.maskW[i,j,k]
    maskVtr = lambda i,j,k: pyom.maskW[i,j+1,k] * pyom.maskW[i,j,k]
    maskWtr = lambda i,j,k: pyom.maskW[i,j,k+1] * pyom.maskW[i,j,k]

    for k in xrange(pyom.nz): # k = 1,nz
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                uCFL = abs(pyom.u_wgrid[i,j,k]*pyom.dt_tracer/(pyom.cost[j]*pyom.dxt[i]))
                rjp = (var[i+2,j,k] - var[i+1,j,k]) * maskUtr[i+1,j,k]
                rj = (var[i+1,j,k] - var[i,j,k]) * maskUtr[i,j,k]
                rjm = (var[i,j,k] - var[i-1,j,k]) * maskUtr[i-1,j,k]
                if rj != 0.:
                    if pyom.u_wgrid[i,j,k] > 0:
                        cr = rjm/rj
                    else:
                        cr = rjp/rj
                else:
                    if pyom.u_wgrid[i,j,k] > 0:
                        cr = rjm*1e20
                    else:
                        cr = rjp*1e20
                cr = limiter(cr)
                adv_fe[i,j,k] = pyom.u_wgrid[i,j,k]*(var[i+1,j,k]+var[i,j,k])*0.5   \
                                -abs(pyom.u_wgrid[i,j,k])*((1.-cr)+uCFL*cr)*rj*0.5

    for k in xrange(pyom.nz): # k = 1,nz
        for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                rjp = (var[i,j+2,k] - var[i,j+1,k]) * maskVtr[i,j+1,k]
                rj = (var[i,j+1,k] - var[i,j,k]) * maskVtr[i,j,k]
                rjm = (var[i,j,k] - var(i,j-1,k)) * maskVtr(i,j-1,k)
                uCFL = abs(pyom.cosu[j]*pyom.v_wgrid[i,j,k]*pyom.dt_tracer/(pyom.cost[j]*pyom.dyt[j]))
                if rj != 0.:
                    if pyom.v_wgrid[i,j,k] > 0:
                        cr = rjm/rj
                    else:
                        cr = rjp/rj
                else:
                    if pyom.v_wgrid[i,j,k] > 0:
                        cr = rjm*1e20
                    else:
                        cr = rjp*1e20
                cr = limiter(cr)
                adv_fn[i,j,k] = pyom.cosu[j]*pyom.v_wgrid[i,j,k]*(var[i,j+1,k]+var[i,j,k])*0.5   \
                                -abs(pyom.cosu[j]*pyom.v_wgrid[i,j,k])*((1.-cr)+uCFL*cr)*rj*0.5

    for k in xrange(pyom.nz-1): # k = 1,nz-1
        kp1 = min(pyom.nz-2,k+1)
        kp2 = min(pyom.nz-1,k+2);
        km1 = max(1,k-1)
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                rjp = (var[i,j,kp2]-var[i,j,k+1])*maskWtr(i,j,kp1)
                rj = (var[i,j,k+1]-var[i,j,k])*maskWtr[i,j,k]
                rjm = (var[i,j,k]-var[i,j,km1])*maskWtr[i,j,km1]
                uCFL = abs(pyom.w_wgrid[i,j,k]*pyom.dt_tracer/pyom.dzw[k])
                if rj != 0.:
                    if pyom.w_wgrid[i,j,k] > 0:
                        cr = rjm/rj
                    else:
                        cr = rjp/rj
                else:
                    if pyom.w_wgrid[i,j,k] > 0:
                        cr = rjm*1e20
                    else:
                        cr = rjp*1e20
                cr = limiter(cr)
                adv_ft[i,j,k] = pyom.w_wgrid[i,j,k]*(var[i,j,k+1]+var[i,j,k])*0.5   \
                                -abs(pyom.w_wgrid[i,j,k])*((1.-cr)+uCFL*cr)*rj*0.5
    adv_ft[:,:,pyom.nz] = 0.0


def adv_flux_upwind_wgrid(adv_fe,adv_fn,adv_ft,var,pyom):
    """
    Calculates advection of a tracer defined on Wgrid
    """
    #  integer, intent(in) :: is_,ie_,js_,je_,nz_
    #  real*8, intent(inout) :: adv_fe(is_:ie_,js_:je_,nz_), adv_fn(is_:ie_,js_:je_,nz_)
    #  real*8, intent(inout) :: adv_ft(is_:ie_,js_:je_,nz_),    var(is_:ie_,js_:je_,nz_)
    #  integer :: i,j,k
    #  real*8 :: rj
    #  real*8 :: maskUtr,maskVtr,maskWtr
    maskUtr[i,j,k] = pyom.maskW[i+1,j,k]*pyom.maskW[i,j,k]
    maskVtr[i,j,k] = pyom.maskW[i,j+1,k]*pyom.maskW[i,j,k]
    maskWtr[i,j,k] = pyom.maskW[i,j,k+1]*pyom.maskW[i,j,k]

    for k in xrange(pyom.nz): # k = 1,nz
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                rj = (var[i+1,j,k]-var[i,j,k])*maskUtr[i,j,k]
                adv_fe[i,j,k] = pyom.u_wgrid[i,j,k]*(var[i+1,j,k]+var[i,j,k])*0.5 - abs(pyom.u_wgrid[i,j,k])*rj*0.5

    for k in xrange(pyom.nz): # k = 1,nz
        for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                rj = (var[i,j+1,k]-var[i,j,k])*maskVtr[i,j,k]
                adv_fn[i,j,k] = pyom.cosu[j]*pyom.v_wgrid[i,j,k]*(var[i,j+1,k]+var[i,j,k])*0.5 - abs(pyom.cosu[j]*pyom.v_wgrid[i,j,k])*rj*0.5

    for k in xrange(pyom.nz-1): # k = 1,nz-1
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                rj = (var[i,j,k+1]-var[i,j,k])*maskWtr[i,j,k]
                adv_ft[i,j,k] = pyom.w_wgrid[i,j,k]*(var[i,j,k+1]+var[i,j,k])*0.5 - abs(pyom.w_wgrid[i,j,k])*rj*0.5

    adv_ft[:,:,pyom.nz] = 0.0

from climate.pyom import cyclic, density
import numpy as np
import sys

def u_centered_grid(dyt,dyu,yt,yu,n):
    yu[0] = 0
    yu[1:n] = np.add.accumulate(dyt[1:n])

    yt[0] = yu[0]-dyt[0]*0.5
    for i in xrange(1, n):
        yt[i] = 2*yu[i-1] - yt[i-1]

    dyu[:n-1] = yt[1:] - yt[:n-1]
    dyu[n-1] = 2*dyt[n-1] - dyu[n-2]
#  yt(1)=yu(1)-dyt(1)*0.5
#  do i=2,n
#   yt(i) = 2*yu(i-1) - yt(i-1)
#  enddo
#  do i=1,n-1
#   dyu(i)= yt(i+1)-yt(i)
#  enddo
#  dyu(n)=2*dyt(n)- dyu(n-1)

def calc_grid(pyom):
    """
    ---------------------------------------------------------------------------------
        setup grid based on dxt,dyt,dzt and x_origin, y_origin
    ---------------------------------------------------------------------------------
    """""
    #use main_module
    #implicit none
    #integer :: i,j
    #real*8 :: aloc(nx,ny)
    #real*8, dimension(1-onx:nx+onx) :: dxt_gl,dxu_gl,xt_gl,xu_gl
    #real*8, dimension(1-onx:ny+onx) :: dyt_gl,dyu_gl,yt_gl,yu_gl

    aloc = np.zeros((pyom.nx,pyom.ny))
    dxt_gl = np.empty(pyom.nx+4)
    dxu_gl = np.empty(pyom.nx+4)
    xt_gl  = np.empty(pyom.nx+4)
    xu_gl  = np.empty(pyom.nx+4)
    dyt_gl = np.empty(pyom.ny+4)
    dyu_gl = np.empty(pyom.ny+4)
    yt_gl  = np.empty(pyom.ny+4)
    yu_gl  = np.empty(pyom.ny+4)
    """
    --------------------------------------------------------------
     transfer from locally defined variables to global ones
    --------------------------------------------------------------
    """
    aloc[:,0] = pyom.dxt[2:pyom.nx+2]

    dxt_gl[2:pyom.nx+2] = aloc[:,0]

    if pyom.enable_cyclic_x:
        for i in xrange(1,3): #i=1,onx
            dxt_gl[pyom.nx+i+1] = dxt_gl[i+1]
            dxt_gl[2-i] = dxt_gl[pyom.nx-i+2]
    else:
        for i in xrange(1,3): #i=1,onx
            dxt_gl[pyom.nx+i+1] = dxt_gl[pyom.nx+1]
            dxt_gl[2-i] = dxt_gl[2]

    aloc[0,:] = pyom.dyt[2:pyom.ny+2]
    dyt_gl[2:pyom.ny+2] = aloc[0, :]

    for i in xrange(1, 3): #i=1,onx
        dyt_gl[pyom.ny+i+1] = dyt_gl[pyom.ny+1]
        dyt_gl[2-i] = dyt_gl[2]
    """
    -------------------------------------------------------------
    grid in east/west direction
    -------------------------------------------------------------
    """
    u_centered_grid(dxt_gl,dxu_gl,xt_gl,xu_gl,pyom.nx+4)
    xt_gl += -xu_gl[2]+pyom.x_origin
    xu_gl += -xu_gl[2]+pyom.x_origin

    if pyom.enable_cyclic_x:
        for i in xrange(1,3): #i=1,onx
            xt_gl[pyom.nx+i+1] = xt_gl[i+1]
            xt_gl[2-i]=xt_gl[pyom.nx-i+2]
            xu_gl[pyom.nx+i+1] = xt_gl[i+1]
            xu_gl[2-i]=xu_gl[pyom.nx-i+2]
            dxu_gl[pyom.nx+i+1] = dxu_gl[i+1]
            dxu_gl[2-i] = dxu_gl[pyom.nx-i+2]

    """
    --------------------------------------------------------------
     grid in north/south direction
    --------------------------------------------------------------
    """
    u_centered_grid(dyt_gl,dyu_gl,yt_gl,yu_gl,pyom.ny+4)
    yt_gl += -yu_gl[2]+pyom.y_origin
    yu_gl += -yu_gl[2]+pyom.y_origin

    if pyom.coord_degree:
        """
        --------------------------------------------------------------
         convert from degrees to pseudo cartesian grid
        --------------------------------------------------------------
        """
        dxt_gl *= pyom.degtom
        dxu_gl *= pyom.degtom
        dyt_gl *= pyom.degtom
        dyu_gl *= pyom.degtom

    """
    --------------------------------------------------------------
      transfer to locally defined variables
    --------------------------------------------------------------
    """
    pyom.xt[:]  = xt_gl[:]
    pyom.xu[:]  = xu_gl[:]
    pyom.dxu[:] = dxu_gl[:]
    pyom.dxt[:] = dxt_gl[:]

    pyom.yt[:]  = yt_gl[:]
    pyom.yu[:]  = yu_gl[:]
    pyom.dyu[:] = dyu_gl[:]
    pyom.dyt[:] = dyt_gl[:]

    """
    --------------------------------------------------------------
     grid in vertical direction
    --------------------------------------------------------------
    """
    u_centered_grid(pyom.dzt,pyom.dzw,pyom.zt,pyom.zw,pyom.nz)
    #dzw(nz)=dzt(nz) #*0.5 # this is account for in the model directly
    pyom.zt -= pyom.zw[pyom.nz-1]
    pyom.zw -= pyom.zw[pyom.nz-1]  # zero at zw(nz)

    """
    --------------------------------------------------------------
     metric factors
    --------------------------------------------------------------
    """
    if pyom.coord_degree:
        for j in xrange(pyom.ny+4):
            pyom.cost[j] = np.cos( pyom.yt[j]/180.*np.pi )
            pyom.cosu[j] = np.cos( pyom.yu[j]/180.*np.pi )
            pyom.tantr[j] = np.tan( pyom.yt[j]/180.*np.pi ) /pyom.radius
    else:
        pyom.cost[...] = 1.0
        pyom.cosu[...] = 1.0
        pyom.tantr[...] = 0.0

    """
    --------------------------------------------------------------
     precalculate area of boxes
    --------------------------------------------------------------
    """
    for j in xrange(pyom.ny+4): #j=js_pe-onx,je_pe+onx
        for i in xrange(pyom.nx+4): #i=is_pe-onx,ie_pe+onx
            pyom.area_t[i,j] = pyom.dxt[i]*pyom.cost[j]*pyom.dyt[j]
            pyom.area_u[i,j] = pyom.dxu[i]*pyom.cost[j]*pyom.dyt[j]
            pyom.area_v[i,j] = pyom.dxt[i]*pyom.cosu[j]*pyom.dyu[j]

def calc_beta(pyom):
    """
    --------------------------------------------------------------
     calculate beta = df/dy
    --------------------------------------------------------------
    """
    for j in xrange(2,pyom.ny+2): # j=js_pe,je_pe
        pyom.beta[:,j] = 0.5*(  (pyom.coriolis_t[:,j+1]-pyom.coriolis_t[:,j])/pyom.dyu[j] + (pyom.coriolis_t[:,j]-pyom.coriolis_t[:,j-1])/pyom.dyu[j-1] )

def calc_topo(pyom):
    """
    --------------------------------------------------------------
     calulate masks, total depth etc
    --------------------------------------------------------------
    """

    """
    --------------------------------------------------------------
     close domain
    --------------------------------------------------------------
    """
    pyom.kbot[:,:2] = 0
    pyom.kbot[:,-2:] = 0
    if not pyom.enable_cyclic_x:
        pyom.kbot[:2,:] = 0
        pyom.kbot[-2:,:] = 0

    cyclic.setcyclic_xy(pyom.kbot,pyom.enable_cyclic_x,pyom.nx)

    """
    --------------------------------------------------------------
     Land masks
    --------------------------------------------------------------
    """
    pyom.maskT[...] = 0.0
    for k in xrange(pyom.nz): # k=1,nz
        for j in xrange(pyom.ny+4): # j=js_pe-onx,je_pe+onx
            for i in xrange(pyom.nx+4): # i=is_pe-onx,ie_pe+onx
                if pyom.kbot[i,j] > 0 and pyom.kbot[i,j]-1 <= k:
                    pyom.maskT[i,j,k] = pyom.kbot[i,j]
    cyclic.setcyclic_xyz(pyom.maskT, pyom.enable_cyclic_x, pyom.nx, pyom.nz)
    pyom.maskU[...] = pyom.maskT
    for i in xrange(pyom.nx+3): # i=is_pe-onx,ie_pe+onx-1
        pyom.maskU[i,:,:] = np.minimum(pyom.maskT[i,:,:], pyom.maskT[i+1,:,:])
    cyclic.setcyclic_xyz(pyom.maskU, pyom.enable_cyclic_x, pyom.nx, pyom.nz)
    pyom.maskV[...] = pyom.maskT
    for j in xrange(pyom.ny+3): # j=js_pe-onx,je_pe+onx-1
        pyom.maskV[:,j,:] = np.minimum(pyom.maskT[:,j,:], pyom.maskT[:,j+1,:])
    cyclic.setcyclic_xyz(pyom.maskV, pyom.enable_cyclic_x, pyom.nx, pyom.nz)
    pyom.maskZ[...] = pyom.maskT
    for j in xrange(pyom.ny+3): # j=js_pe-onx,je_pe+onx-1
        for i in xrange(pyom.nx+3): # i=is_pe-onx,ie_pe+onx-1
            pyom.maskZ[i,j,:] = np.minimum(np.minimum(pyom.maskT[i,j,:],pyom.maskT[i,j+1,:]),pyom.maskT[i+1,j,:])
    cyclic.setcyclic_xyz(pyom.maskZ, pyom.enable_cyclic_x, pyom.nx, pyom.nz)
    pyom.maskW[...] = pyom.maskT
    for k in xrange(pyom.nz-1): # k=1,nz-1
        pyom.maskW[:,:,k] = np.minimum(pyom.maskT[:,:,k],pyom.maskT[:,:,k+1])
    """
    --------------------------------------------------------------
     total depth
    --------------------------------------------------------------
    """
    pyom.ht[...] = 0.0
    pyom.hu[...] = 0.0
    pyom.hv[...] = 0.0
    for k in xrange(pyom.nz): #k=1,nz
        pyom.ht += pyom.maskT[:,:,k]*pyom.dzt[k]
        pyom.hu += pyom.maskU[:,:,k]*pyom.dzt[k]
        pyom.hv += pyom.maskV[:,:,k]*pyom.dzt[k]
    pyom.hur[pyom.hu != 0.0] = 1./pyom.hu[pyom.hu != 0.0]
    pyom.hvr[pyom.hv != 0.0] = 1./pyom.hv[pyom.hv != 0.0]

#TODO: you are here

def calc_initial_conditions(pyom):
    """
    calculate dyn. enthalp, etc
    """
    for n in xrange(3): # n=1,3
        # boundary exchange
        cyclic.setcyclic_xyz(pyom.temp[:,:,:,n],pyom.enable_cyclic_x,pyom.nx,pyom.nz)
        cyclic.setcyclic_xyz(pyom.salt[:,:,:,n],pyom.enable_cyclic_x,pyom.nx,pyom.nz)
        # calculate density, etc
        for k in xrange(pyom.nz): # k=1,nz
            for j in xrange(pyom.ny+4): # j=js_pe-onx,je_pe+onx
                for i in xrange(pyom.nx+4): # i=is_pe-onx,ie_pe+onx
                    if pyom.salt[i,j,k,n] < 0.0:
                        raise RuntimeError("salinity <0 at i={} j={} k={}".format(i,j,k))
                    pyom.rho[i,j,k,n] = density.get_rho(pyom.salt[i,j,k,n],pyom.temp[i,j,k,n],abs(pyom.zt[k]),pyom) * pyom.maskT[i,j,k]
                    pyom.Hd[i,j,k,n] = density.get_dyn_enthalpy(pyom.salt[i,j,k,n],pyom.temp[i,j,k,n],abs(pyom.zt[k]),pyom) * pyom.maskT[i,j,k]
                    pyom.int_drhodT[i,j,k,n] = density.get_int_drhodT(pyom.salt[i,j,k,n],pyom.temp[i,j,k,n],abs(pyom.zt[k]),pyom)
                    pyom.int_drhodS[i,j,k,n] = density.get_int_drhodS(pyom.salt[i,j,k,n],pyom.temp[i,j,k,n],abs(pyom.zt[k]),pyom)
        # stability frequency
        for k in xrange(pyom.nz-1):
            for j in xrange(pyom.ny+4):
                for i in xrange(pyom.nx+4):
                    fxa = -pyom.grav/pyom.rho_0/pyom.dzw[k]*pyom.maskW[i,j,k]
                    pyom.Nsqr[i,j,k,n] = fxa * density.get_rho(pyom.salt[i,j,k+1,n],pyom.temp[i,j,k+1,n],abs(pyom.zt[k]),pyom) - pyom.rho[i,j,k,n]
        pyom.Nsqr[:,:,pyom.nz-1,n] = pyom.Nsqr[:,:,pyom.nz-2,n]

def ugrid_to_tgrid():
    pass

def vgrid_to_tgrid():
    pass

def solve_tridiag(a, b, c, d, n):
    x = np.zeros(n)
    cp = np.zeros(n)
    dp = np.zeros(n)

    # initialize c-prime and d-prime
    cp[0] = c[0]/b[0]
    dp[0] = d[0]/b[0]

    # solve for vectors c-prime and d-prime
    for i in xrange(1, n):
        m = b[i] - cp[i-1] * a[i]
        fxa = 1.0 / m
        cp[i] = c[i] * fxa
        dp[i] = (d[i]-dp[i-1]*a[i]) * fxa
    x[n-1] = dp[n-1]
    for i in xrange(n-2, -1, -1):
        x[i] = dp[i] - cp[i]*x[i+1]
    return x

def calc_diss():
    pass

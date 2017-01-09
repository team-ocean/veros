from climate.pyom import cyclic
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

def calc_grid(boussine):
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

    aloc = np.zeros((boussine.nx,boussine.ny))
    dxt_gl = np.empty(boussine.nx+4)
    dxu_gl = np.empty(boussine.nx+4)
    xt_gl  = np.empty(boussine.nx+4)
    xu_gl  = np.empty(boussine.nx+4)
    dyt_gl = np.empty(boussine.ny+4)
    dyu_gl = np.empty(boussine.ny+4)
    yt_gl  = np.empty(boussine.ny+4)
    yu_gl  = np.empty(boussine.ny+4)
    """
    --------------------------------------------------------------
     transfer from locally defined variables to global ones
    --------------------------------------------------------------
    """
    aloc[:,0] = boussine.dxt[2:boussine.nx+2]

    dxt_gl[2:boussine.nx+2] = aloc[:,0]

    if boussine.enable_cyclic_x:
        for i in xrange(1,3): #i=1,onx
            dxt_gl[boussine.nx+i+1] = dxt_gl[i+1]
            dxt_gl[2-i] = dxt_gl[boussine.nx-i+2]
    else:
        for i in xrange(1,3): #i=1,onx
            dxt_gl[boussine.nx+i+1] = dxt_gl[boussine.nx+1]
            dxt_gl[2-i] = dxt_gl[2]

    aloc[0,:] = boussine.dyt[2:boussine.ny+2]
    dyt_gl[2:boussine.ny+2] = aloc[0, :]

    for i in xrange(1, 3): #i=1,onx
        dyt_gl[boussine.ny+i+1] = dyt_gl[boussine.ny+1]
        dyt_gl[2-i] = dyt_gl[2]
    """
    -------------------------------------------------------------
    grid in east/west direction
    -------------------------------------------------------------
    """
    u_centered_grid(dxt_gl,dxu_gl,xt_gl,xu_gl,boussine.nx+4)
    xt_gl += -xu_gl[2]+boussine.x_origin
    xu_gl += -xu_gl[2]+boussine.x_origin

    if boussine.enable_cyclic_x:
        for i in xrange(1,3): #i=1,onx
            xt_gl[boussine.nx+i+1] = xt_gl[i+1]
            xt_gl[2-i]=xt_gl[boussine.nx-i+2]
            xu_gl[boussine.nx+i+1] = xt_gl[i+1]
            xu_gl[2-i]=xu_gl[boussine.nx-i+2]
            dxu_gl[boussine.nx+i+1] = dxu_gl[i+1]
            dxu_gl[2-i] = dxu_gl[boussine.nx-i+2]

    """
    --------------------------------------------------------------
     grid in north/south direction
    --------------------------------------------------------------
    """
    u_centered_grid(dyt_gl,dyu_gl,yt_gl,yu_gl,boussine.ny+4)
    yt_gl += -yu_gl[2]+boussine.y_origin
    yu_gl += -yu_gl[2]+boussine.y_origin

    if boussine.coord_degree:
        """
        --------------------------------------------------------------
         convert from degrees to pseudo cartesian grid
        --------------------------------------------------------------
        """
        dxt_gl *= boussine.degtom
        dxu_gl *= boussine.degtom
        dyt_gl *= boussine.degtom
        dyu_gl *= boussine.degtom

    """
    --------------------------------------------------------------
      transfer to locally defined variables
    --------------------------------------------------------------
    """
    boussine.xt[:]  = xt_gl[:]
    boussine.xu[:]  = xu_gl[:]
    boussine.dxu[:] = dxu_gl[:]
    boussine.dxt[:] = dxt_gl[:]

    boussine.yt[:]  = yt_gl[:]
    boussine.yu[:]  = yu_gl[:]
    boussine.dyu[:] = dyu_gl[:]
    boussine.dyt[:] = dyt_gl[:]

    """
    --------------------------------------------------------------
     grid in vertical direction
    --------------------------------------------------------------
    """
    u_centered_grid(boussine.dzt,boussine.dzw,boussine.zt,boussine.zw,boussine.nz)
    #dzw(nz)=dzt(nz) #*0.5 # this is account for in the model directly
    boussine.zt -= boussine.zw[boussine.nz-1]
    boussine.zw -= boussine.zw[boussine.nz-1]  # zero at zw(nz)

    """
    --------------------------------------------------------------
     metric factors
    --------------------------------------------------------------
    """
    if boussine.coord_degree:
        for j in xrange(boussine.ny+4):
            boussine.cost[j] = np.cos( boussine.yt[j]/180.*np.pi )
            boussine.cosu[j] = np.cos( boussine.yu[j]/180.*np.pi )
            boussine.tantr[j] = np.tan( boussine.yt[j]/180.*np.pi ) /boussine.radius
    else:
        boussine.cost[...] = 1.0
        boussine.cosu[...] = 1.0
        boussine.tantr[...] = 0.0

    """
    --------------------------------------------------------------
     precalculate area of boxes
    --------------------------------------------------------------
    """
    for j in xrange(boussine.ny+4): #j=js_pe-onx,je_pe+onx
        for i in xrange(boussine.nx+4): #i=is_pe-onx,ie_pe+onx
            boussine.area_t[i,j] = boussine.dxt[i]*boussine.cost[j]*boussine.dyt[j]
            boussine.area_u[i,j] = boussine.dxu[i]*boussine.cost[j]*boussine.dyt[j]
            boussine.area_v[i,j] = boussine.dxt[i]*boussine.cosu[j]*boussine.dyu[j]

def calc_beta(boussine):
    """
    --------------------------------------------------------------
     calculate beta = df/dy
    --------------------------------------------------------------
    """
    for j in xrange(2,boussine.ny+2): # j=js_pe,je_pe
        boussine.beta[:,j] = 0.5*(  (boussine.coriolis_t[:,j+1]-boussine.coriolis_t[:,j])/boussine.dyu[j] + (boussine.coriolis_t[:,j]-boussine.coriolis_t[:,j-1])/boussine.dyu[j-1] )

def calc_topo(boussine):
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
    boussine.kbot[:,:2] = 0
    boussine.kbot[:,-2:] = 0
    if not boussine.enable_cyclic_x:
        boussine.kbot[:2,:] = 0
        boussine.kbot[-2:,:] = 0

    cyclic.setcyclic_xy(boussine.kbot,boussine.enable_cyclic_x,boussine.nx)

    """
    --------------------------------------------------------------
     Land masks
    --------------------------------------------------------------
    """
    boussine.maskT[...] = 0.0
    for k in xrange(boussine.nz): # k=1,nz
        for j in xrange(boussine.ny+4): # j=js_pe-onx,je_pe+onx
            for i in xrange(boussine.nx+4): # i=is_pe-onx,ie_pe+onx
                if boussine.kbot[i,j] != 0 and boussine.kbot[i,j] <= k:
                    boussine.maskT[i,j,k] = 1.0
    cyclic.setcyclic_xyz(boussine.maskT, boussine.enable_cyclic_x, boussine.nx, boussine.nz)
    boussine.maskU[...] = boussine.maskT
    for i in xrange(boussine.nx+3): # i=is_pe-onx,ie_pe+onx-1
        boussine.maskU[i,:,:] = np.minimum(boussine.maskT[i,:,:], boussine.maskT[i+1,:,:])
    cyclic.setcyclic_xyz(boussine.maskU, boussine.enable_cyclic_x, boussine.nx, boussine.nz)
    boussine.maskV[...] = boussine.maskT
    for j in xrange(boussine.ny+3): # j=js_pe-onx,je_pe+onx-1
        boussine.maskV[:,j,:] = np.minimum(boussine.maskT[:,j,:], boussine.maskT[:,j+1,:])
    cyclic.setcyclic_xyz(boussine.maskV, boussine.enable_cyclic_x, boussine.nx, boussine.nz)
    boussine.maskZ[...] = boussine.maskT
    for j in xrange(boussine.ny+3): # j=js_pe-onx,je_pe+onx-1
        for i in xrange(boussine.nx+3): # i=is_pe-onx,ie_pe+onx-1
            boussine.maskZ[i,j,:] = np.minimum(boussine.maskT[i,j,:],boussine.maskT[i,j+1,:],boussine.maskT[i+1,j,:])
    cyclic.setcyclic_xyz(boussine.maskZ, boussine.enable_cyclic_x, boussine.nx, boussine.nz)
    boussine.maskW[...] = boussine.maskT
    for k in xrange(boussine.nz-1): # k=1,nz-1
        boussine.maskW[:,:,k] = np.minimum(boussine.maskT[:,:,k],boussine.maskT[:,:,k+1])
    """
    --------------------------------------------------------------
     total depth
    --------------------------------------------------------------
    """
    boussine.ht[...] = 0.0
    boussine.hu[...] = 0.0
    boussine.hv[...] = 0.0
    for k in xrange(boussine.nz): #k=1,nz
        boussine.ht += boussine.maskT[:,:,k]*boussine.dzt[k]
        boussine.hu += boussine.maskU[:,:,k]*boussine.dzt[k]
        boussine.hv += boussine.maskV[:,:,k]*boussine.dzt[k]
    boussine.hur[boussine.hu != 0.0] = 1./boussine.hu[boussine.hu != 0.0]
    boussine.hvr[boussine.hv != 0.0] = 1./boussine.hv[boussine.hv != 0.0]

#TODO: you are here

def calc_initial_conditions():
    pass

def ugrid_to_tgrid():
    pass

def vgrid_to_tgrid():
    pass

def solve_tridiag(a, b, c, d, n):
    cp = np.zeros(n)
    dp = np.zeros(n)

    # initialize c-prime and d-prime
    cp[0] = c[0]/b[0]
    dp[0] = d[0]/b[0]

    # solve for vectors c-prime and d-prime
    for i in xrange(1, n):
        m = b[i] - cp[i-1] * a[i]
        fxa = 1.0 / m
        cp[i] = c[i] * fxz
        dp[i] = d[i]-dp[i-1]*a[i]
    x[n-1] = dp[n-1]
    for i in xrange(n-2, -1, -1):
        x[i] = dp[i] - cp[i]*x[i+1]

    return x

def calc_diss():
    pass

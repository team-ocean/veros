from climate.pyom import cyclic, density
import climate

import numpy as np
from scipy import linalg

def u_centered_grid(dyt,dyu,yt,yu,n):
    yu[0] = 0
    yu[1:n] = np.add.accumulate(dyt[1:n])

    #if climate.is_bohrium:
    #    YT = yt.copy2numpy()
    #else:
    #    YT = yt
    yt[0] = yu[0]-dyt[0]*0.5
    for i in xrange(1, n):
        yt[i] = 2*yu[i-1] - yt[i-1]
    #if climate.is_bohrium:
    #    yt[...] = np.array(YT)

    dyu[:n-1] = yt[1:] - yt[:n-1]
    dyu[n-1] = 2*dyt[n-1] - dyu[n-2]
#  yt(1)=yu(1)-dyt(1)*0.5
#  for i in xrange(2,n): # i=2,n
#   yt(i) = 2*yu(i-1) - yt(i-1)
#  enddo
#  for i in xrange(1,n-1): # i=1,n-1
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
        dxt_gl[pyom.nx+2:pyom.nx+4] = dxt_gl[2:4]
        dxt_gl[:2] = dxt_gl[pyom.nx:pyom.nx+2]
        #for i in xrange(1,3): #i=1,onx
        #    dxt_gl[pyom.nx+i+1] = dxt_gl[i+1]
        #    dxt_gl[2-i] = dxt_gl[pyom.nx-i+2]
    else:
        dxt_gl[pyom.nx+2:pyom.nx+4] = dxt_gl[pyom.nx+1]
        dxt_gl[:2] = dxt_gl[2]
        #for i in xrange(1,3): #i=1,onx
        #    dxt_gl[pyom.nx+i+1] = dxt_gl[pyom.nx+1]
        #    dxt_gl[2-i] = dxt_gl[2]

    aloc[0,:] = pyom.dyt[2:pyom.ny+2]
    dyt_gl[2:pyom.ny+2] = aloc[0, :]

    dyt_gl[pyom.ny+2:pyom.ny+4] = dyt_gl[pyom.ny+1]
    dyt_gl[:2] = dxt_gl[2]
    #for i in xrange(1, 3): #i=1,onx
    #    dyt_gl[pyom.ny+i+1] = dyt_gl[pyom.ny+1]
    #    dyt_gl[2-i] = dyt_gl[2]
    """
    -------------------------------------------------------------
    grid in east/west direction
    -------------------------------------------------------------
    """
    u_centered_grid(dxt_gl,dxu_gl,xt_gl,xu_gl,pyom.nx+4)
    xt_gl += -xu_gl[2]+pyom.x_origin
    xu_gl += -xu_gl[2]+pyom.x_origin

    if pyom.enable_cyclic_x:
        xt_gl[pyom.nx+2:pyom.nx+4] = xt_gl[2:4]
        xt_gl[:2] = xt_gl[pyom.nx:pyom.nx+2]
        xu_gl[pyom.nx+2:pyom.nx+4] = xt_gl[2:4]
        xu_gl[:2] = xu_gl[pyom.nx:pyom.nx+2]
        dxu_gl[pyom.nx+2:pyom.nx+4] = dxu_gl[2:4]
        dxu_gl[:2] = dxu_gl[pyom.nx:pyom.nx+2]
        #for i in xrange(1,3): #i=1,onx
        #    xt_gl[pyom.nx+i+1] = xt_gl[i+1]
        #    xt_gl[2-i]=xt_gl[pyom.nx-i+2]
        #    xu_gl[pyom.nx+i+1] = xt_gl[i+1]
        #    xu_gl[2-i]=xu_gl[pyom.nx-i+2]
        #    dxu_gl[pyom.nx+i+1] = dxu_gl[i+1]
        #    dxu_gl[2-i] = dxu_gl[pyom.nx-i+2]

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
        pyom.cost = np.cos(pyom.yt*np.pi/180.)
        pyom.cosu = np.cos( pyom.yu*np.pi/180. )
        pyom.tantr = np.tan( pyom.yt*np.pi/180. ) /pyom.radius
    else:
        pyom.cost[...] = 1.0
        pyom.cosu[...] = 1.0
        pyom.tantr[...] = 0.0

    """
    --------------------------------------------------------------
     precalculate area of boxes
    --------------------------------------------------------------
    """
    pyom.area_t = pyom.cost * pyom.dyt * pyom.dxt[:, np.newaxis]
    pyom.area_u = pyom.cost*pyom.dyt * pyom.dxu[:, np.newaxis]
    pyom.area_v = pyom.cosu*pyom.dyu*pyom.dxt[:, np.newaxis]

def calc_beta(pyom):
    """
    --------------------------------------------------------------
     calculate beta = df/dy
    --------------------------------------------------------------
    """
    pyom.beta[:, 2:pyom.ny+2] = 0.5*(  (pyom.coriolis_t[:,3:pyom.ny+3]-pyom.coriolis_t[:,2:pyom.ny+2])/pyom.dyu[2:pyom.ny+2] + (pyom.coriolis_t[:,2:pyom.ny+2]-pyom.coriolis_t[:,1:pyom.ny+1])/pyom.dyu[1:pyom.ny+1] )
    #for j in xrange(2,pyom.ny+2): # j=js_pe,je_pe
    #    pyom.beta[:,j] = 0.5*(  (pyom.coriolis_t[:,j+1]-pyom.coriolis_t[:,j])/pyom.dyu[j] + (pyom.coriolis_t[:,j]-pyom.coriolis_t[:,j-1])/pyom.dyu[j-1] )

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

    if climate.is_bohrium:
        kbot = pyom.kbot.copy2numpy()
    else:
        kbot = pyom.kbot

    k_kbot = np.ones(pyom.nz, dtype=np.int) * kbot[:, :, np.newaxis]
    ks = np.arange(pyom.nz, dtype=np.int) * np.ones(pyom.ny+4, dtype=np.int)[:, np.newaxis] * np.ones(pyom.nx+4, dtype=np.int)[:,np.newaxis,np.newaxis]
    if climate.is_bohrium:
        k_kbot = k_kbot.copy2numpy()
        ks = ks.copy2numpy()
        maskT = pyom.maskT.copy2numpy()
    else:
        maskT = pyom.maskT
    maskT[(k_kbot > 0) & (k_kbot-1 <= ks)] = k_kbot[(k_kbot > 0) & (k_kbot-1 <= ks)]
    if climate.is_bohrium:
        pyom.maskT = np.array(maskT)
    #for i in xrange(pyom.nx+4): # i=is_pe-onx,ie_pe+onx
    #    for j in xrange(pyom.ny+4): # j=js_pe-onx,je_pe+onx
    #        for k in xrange(pyom.nz): # k=1,nz
    #            if kbot[i,j] > 0 and kbot[i,j]-1 <= k:
    #                pyom.maskT[i,j,k] = kbot[i,j]

    cyclic.setcyclic_xyz(pyom.maskT, pyom.enable_cyclic_x, pyom.nx, pyom.nz)
    pyom.maskU[...] = pyom.maskT
    pyom.maskU[:pyom.nx+3, :, :] = np.minimum(pyom.maskT[:pyom.nx+3, :, :], pyom.maskT[1:pyom.nx+4, :, :])
    #for i in xrange(pyom.nx+3): # i=is_pe-onx,ie_pe+onx-1
    #    pyom.maskU[i,:,:] = np.minimum(pyom.maskT[i,:,:], pyom.maskT[i+1,:,:])
    cyclic.setcyclic_xyz(pyom.maskU, pyom.enable_cyclic_x, pyom.nx, pyom.nz)
    pyom.maskV[...] = pyom.maskT
    pyom.maskV[:, :pyom.ny+3] = np.minimum(pyom.maskT[:,:pyom.ny+3], pyom.maskT[:,1:pyom.ny+4])
    #for j in xrange(pyom.ny+3): # j=js_pe-onx,je_pe+onx-1
    #    pyom.maskV[:,j,:] = np.minimum(pyom.maskT[:,j,:], pyom.maskT[:,j+1,:])
    cyclic.setcyclic_xyz(pyom.maskV, pyom.enable_cyclic_x, pyom.nx, pyom.nz)

    pyom.maskZ[...] = pyom.maskT
    maskZ = np.empty(pyom.maskZ.shape)
    maskZ[...] = pyom.maskZ
    maskZ[:pyom.nx+3, :pyom.ny+3] = np.minimum(np.minimum(pyom.maskT[:pyom.nx+3, :pyom.ny+3],pyom.maskT[:pyom.nx+3, 1:pyom.ny+4]),pyom.maskT[1:pyom.nx+4, :pyom.ny+3])
    #for j in xrange(pyom.ny+3): # j=js_pe-onx,je_pe+onx-1
    #    for i in xrange(pyom.nx+3): # i=is_pe-onx,ie_pe+onx-1
    #        pyom.maskZ[i,j,:] = np.minimum(np.minimum(pyom.maskT[i,j,:],pyom.maskT[i,j+1,:]),pyom.maskT[i+1,j,:])
    cyclic.setcyclic_xyz(pyom.maskZ, pyom.enable_cyclic_x, pyom.nx, pyom.nz)
    pyom.maskW[...] = pyom.maskT
    pyom.maskW[:,:,:pyom.nz-1] = np.minimum(pyom.maskT[:,:,:pyom.nz-1],pyom.maskT[:,:,1:pyom.nz])
    #for k in xrange(pyom.nz-1): # k=1,nz-1
    #    pyom.maskW[:,:,k] = np.minimum(pyom.maskT[:,:,k],pyom.maskT[:,:,k+1])
    """
    --------------------------------------------------------------
     total depth
    --------------------------------------------------------------
    """
    pyom.ht[...] = 0.0
    pyom.hu[...] = 0.0
    pyom.hv[...] = 0.0
    pyom.ht += np.add.reduce(pyom.maskT*pyom.dzt, axis=2)
    pyom.hu += np.add.reduce(pyom.maskU*pyom.dzt, axis=2)
    pyom.hv += np.add.reduce(pyom.maskV*pyom.dzt, axis=2)
    #for k in xrange(pyom.nz): #k=1,nz
    #    pyom.ht += pyom.maskT[:,:,k]*pyom.dzt[k]
    #    pyom.hu += pyom.maskU[:,:,k]*pyom.dzt[k]
    #    pyom.hv += pyom.maskV[:,:,k]*pyom.dzt[k]
    if climate.is_bohrium:
        hur = pyom.hur.copy2numpy()
        hvr = pyom.hvr.copy2numpy()
        hu = pyom.hu.copy2numpy()
        hv = pyom.hv.copy2numpy()
    else:
        hur = pyom.hur
        hvr = pyom.hvr
        hu = pyom.hu
        hv = pyom.hv
    hur[hu != 0.0] = 1./hu[hu != 0.0]
    hvr[hv != 0.0] = 1./hv[hv != 0.0]
    if climate.is_bohrium:
        pyom.hur = np.array(hur)
        pyom.hvr = np.array(hvr)

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
                    pyom.Nsqr[i,j,k,n] = fxa * (density.get_rho(pyom.salt[i,j,k+1,n],pyom.temp[i,j,k+1,n],abs(pyom.zt[k]),pyom) - pyom.rho[i,j,k,n])
        pyom.Nsqr[:,:,pyom.nz-1,n] = pyom.Nsqr[:,:,pyom.nz-2,n]


def ugrid_to_tgrid(A,pyom):
    # real*8, dimension(is_:ie_,js_:je_,nz_) :: A,B
    B = np.zeros_like(A)
    for i in xrange(pyom.is_pe,pyom.ie_pe):
        B[i,:,:] = (pyom.dxu[i] * A[i,:,:] + pyom.dxu[i-1] * A[i-1,:,:]) / (2*pyom.dxt[i])
    return B


def vgrid_to_tgrid(A,pyom):
    # real*8, dimension(is_:ie_,js_:je_,nz_) :: A,B
    B = np.zeros_like(A)
    for k in xrange(pyom.nz):
        for j in xrange(pyom.js_pe,pyom.je_pe):
            B[:,j,k] = (pyom.area_v[:,j] * A[:,j,k] + pyom.area_v[:,j-1] * A[:,j-1,k]) / (2*pyom.area_t[:,j])
    return B

def solve_tridiag(a, b, c, d):
    assert a.shape == b.shape and a.shape == c.shape and a.shape == d.shape
    ab = np.zeros((3,a.shape[0]))
    ab[0,1:] = c[:-1]
    ab[1,:] = b
    ab[2,:-1] = a[1:]
    return linalg.solve_banded((1,1),ab,d)

def calc_diss(diss,K_diss,tag,pyom):
    # !real*8 :: diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    # !real*8 :: K_diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    # !real*8 :: diss_u(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    # real*8, dimension(is_:ie_,js_:je_,nz_) :: diss,K_diss,diss_u
    # character*1 :: tag

    diss_u = np.zeros_like(diss)

    if tag == 'U':
        # dissipation interpolated on W-grid
        for j in xrange(pyom.js_pe,pyom.je_pe): # j=js_pe,je_pe
            for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i=is_pe-1,ie_pe
                ks = max(pyom.kbot[i,j],pyom.kbot[i+1,j]) - 1
                if ks >= 0:
                    k = ks
                    diss_u[i,j,k] = 0.5 * (diss[i,j,k] + diss[i,j,k+1]) + 0.5 * diss[i,j,k] * pyom.dzw[max(0,k-1)] / pyom.dzw[k]
                    for k in xrange(ks+1,pyom.nz-1): # k=ks+1,nz-1
                        diss_u[i,j,k] = 0.5 * (diss[i,j,k] + diss[i,j,k+1])
                    k = pyom.nz-1
                    diss_u[i,j,k] = diss[i,j,k]
        # dissipation interpolated from U-grid to T-grid
        diss_u = ugrid_to_tgrid(diss_u,pyom)
        return K_diss + diss_u
    elif tag == 'V':
        # dissipation interpolated on W-grid
        for j in xrange(pyom.js_pe-1,pyom.je_pe): # j=js_pe-1,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i=is_pe,ie_pe
                ks = max(pyom.kbot[i,j],pyom.kbot[i,j+1]) - 1
                if ks >= 0:
                    k=ks
                    diss_u[i,j,k] = 0.5*(diss[i,j,k] + diss[i,j,k+1]) + 0.5 * diss[i,j,k] * pyom.dzw[max(0,k-1)] / pyom.dzw[k]
                    for k in xrange(ks+1,pyom.nz-1): # k=ks+1,nz-1
                        diss_u[i,j,k] = 0.5*(diss[i,j,k] + diss[i,j,k+1])
                    k = pyom.nz-1
                    diss_u[i,j,k] = diss[i,j,k]
        # dissipation interpolated from V-grid to T-grid
        diss_u = vgrid_to_tgrid(diss_u,pyom)
        return K_diss + diss_u
    else:
        raise ValueError("unknown tag {}".format(tag))

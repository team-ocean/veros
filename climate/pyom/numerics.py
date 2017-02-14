from climate.pyom import cyclic, density, utilities, diffusion
import climate

import numpy as np
from scipy.linalg import lapack

def u_centered_grid(dyt,dyu,yt,yu,n):
    yu[0] = 0
    yu[1:n] = np.add.accumulate(dyt[1:n])

    yt[0] = yu[0]-dyt[0]*0.5
    yt[1:n] = 2*yu[:n-1]
    if climate.is_bohrium:
        YT = yt.copy2numpy()
    else:
        YT = yt
    for i in xrange(1, n):
        YT[i] -= YT[i-1]
    if climate.is_bohrium:
        YT = np.array(YT)
        yt[1:n] = YT[1:n]

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
    setup grid based on dxt,dyt,dzt and x_origin, y_origin
    """

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
    transfer from locally defined variables to global ones
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
    grid in east/west direction
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
    grid in north/south direction
    """
    u_centered_grid(dyt_gl,dyu_gl,yt_gl,yu_gl,pyom.ny+4)
    yt_gl += -yu_gl[2]+pyom.y_origin
    yu_gl += -yu_gl[2]+pyom.y_origin

    if pyom.coord_degree:
        """
        convert from degrees to pseudo cartesian grid
        """
        dxt_gl *= pyom.degtom
        dxu_gl *= pyom.degtom
        dyt_gl *= pyom.degtom
        dyu_gl *= pyom.degtom

    """
    transfer to locally defined variables
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
    grid in vertical direction
    """
    u_centered_grid(pyom.dzt,pyom.dzw,pyom.zt,pyom.zw,pyom.nz)
    #dzw(nz)=dzt(nz) #*0.5 # this is account for in the model directly
    pyom.zt -= pyom.zw[pyom.nz-1]
    pyom.zw -= pyom.zw[pyom.nz-1]  # zero at zw(nz)

    """
    metric factors
    """
    if pyom.coord_degree:
        pyom.cost = np.cos(pyom.yt*pyom.pi/180.)
        pyom.cosu = np.cos( pyom.yu*pyom.pi/180. )
        pyom.tantr = np.tan( pyom.yt*pyom.pi/180. ) /pyom.radius
    else:
        pyom.cost[...] = 1.0
        pyom.cosu[...] = 1.0
        pyom.tantr[...] = 0.0

    """
    precalculate area of boxes
    """
    pyom.area_t = pyom.cost * pyom.dyt * pyom.dxt[:, np.newaxis]
    pyom.area_u = pyom.cost*pyom.dyt * pyom.dxu[:, np.newaxis]
    pyom.area_v = pyom.cosu*pyom.dyu*pyom.dxt[:, np.newaxis]

def calc_beta(pyom):
    """
    calculate beta = df/dy
    """
    pyom.beta[:, 2:pyom.ny+2] = 0.5*(  (pyom.coriolis_t[:,3:pyom.ny+3]-pyom.coriolis_t[:,2:pyom.ny+2])/pyom.dyu[2:pyom.ny+2] + (pyom.coriolis_t[:,2:pyom.ny+2]-pyom.coriolis_t[:,1:pyom.ny+1])/pyom.dyu[1:pyom.ny+1] )
    #for j in xrange(2,pyom.ny+2): # j=js_pe,je_pe
    #    pyom.beta[:,j] = 0.5*(  (pyom.coriolis_t[:,j+1]-pyom.coriolis_t[:,j])/pyom.dyu[j] + (pyom.coriolis_t[:,j]-pyom.coriolis_t[:,j-1])/pyom.dyu[j-1] )

def calc_topo(pyom):
    """
    calulate masks, total depth etc
    """

    """
    close domain
    """
    pyom.kbot[:,:2] = 0
    pyom.kbot[:,-2:] = 0
    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(pyom.kbot)
    else:
        pyom.kbot[:2,:] = 0
        pyom.kbot[-2:,:] = 0

    """
    Land masks
    """
    pyom.maskT[...] = 0.0
    land_mask = pyom.kbot > 0
    ks = np.indices(pyom.maskT.shape)[2]
    pyom.maskT[...] = land_mask[...,np.newaxis] & (pyom.kbot[...,np.newaxis]-1 <= ks)

    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(pyom.maskT)
    pyom.maskU[...] = pyom.maskT
    pyom.maskU[:pyom.nx+3, :, :] = np.minimum(pyom.maskT[:pyom.nx+3, :, :], pyom.maskT[1:pyom.nx+4, :, :])
    #for i in xrange(pyom.nx+3): # i=is_pe-onx,ie_pe+onx-1
    #    pyom.maskU[i,:,:] = np.minimum(pyom.maskT[i,:,:], pyom.maskT[i+1,:,:])
    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(pyom.maskU)
    pyom.maskV[...] = pyom.maskT
    pyom.maskV[:, :pyom.ny+3] = np.minimum(pyom.maskT[:,:pyom.ny+3], pyom.maskT[:,1:pyom.ny+4])
    #for j in xrange(pyom.ny+3): # j=js_pe-onx,je_pe+onx-1
    #    pyom.maskV[:,j,:] = np.minimum(pyom.maskT[:,j,:], pyom.maskT[:,j+1,:])
    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(pyom.maskV)
    pyom.maskZ[...] = pyom.maskT
    pyom.maskZ[:pyom.nx+3, :pyom.ny+3] = np.minimum(np.minimum(pyom.maskT[:pyom.nx+3, :pyom.ny+3],pyom.maskT[:pyom.nx+3, 1:pyom.ny+4]),pyom.maskT[1:pyom.nx+4, :pyom.ny+3])
    #for j in xrange(pyom.ny+3): # j=js_pe-onx,je_pe+onx-1
    #    for i in xrange(pyom.nx+3): # i=is_pe-onx,ie_pe+onx-1
    #        pyom.maskZ[i,j,:] = np.minimum(np.minimum(pyom.maskT[i,j,:],pyom.maskT[i,j+1,:]),pyom.maskT[i+1,j,:])
    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(pyom.maskZ)
    pyom.maskW[...] = pyom.maskT
    pyom.maskW[:,:,:pyom.nz-1] = np.minimum(pyom.maskT[:,:,:pyom.nz-1],pyom.maskT[:,:,1:pyom.nz])
    #for k in xrange(pyom.nz-1): # k=1,nz-1
    #    pyom.maskW[:,:,k] = np.minimum(pyom.maskT[:,:,k],pyom.maskT[:,:,k+1])
    """
    total depth
    """
    #pyom.ht[...] = 0.0
    #pyom.hu[...] = 0.0
    #pyom.hv[...] = 0.0
    pyom.ht = np.add.reduce(pyom.maskT*pyom.dzt, axis=2)
    pyom.hu = np.add.reduce(pyom.maskU*pyom.dzt, axis=2)
    pyom.hv = np.add.reduce(pyom.maskV*pyom.dzt, axis=2)

    pyom.hur[...] = (1. / (pyom.hu+(pyom.hu==0)))*(pyom.hu!=0)
    pyom.hvr[...] = (1. / (pyom.hv+(pyom.hv==0)))*(pyom.hv!=0)


def calc_initial_conditions(pyom):
    """
    calculate dyn. enthalp, etc
    """
    if pyom.enable_cyclic_x:
        # boundary exchange
        cyclic.setcyclic_x(pyom.temp)
        cyclic.setcyclic_x(pyom.salt)
    # calculate density, etc
    if np.any(pyom.salt < 0.0):
        raise RuntimeError("encountered negative salinity")
    pyom.rho[...] = density.get_rho(pyom.salt,pyom.temp,np.abs(pyom.zt)[:,None],pyom) * pyom.maskT[...,None]
    pyom.Hd[...] = density.get_dyn_enthalpy(pyom.salt,pyom.temp,np.abs(pyom.zt)[:,None],pyom) * pyom.maskT[...,None]
    pyom.int_drhodT[...] = density.get_int_drhodT(pyom.salt,pyom.temp,np.abs(pyom.zt)[:,None],pyom)
    pyom.int_drhodS[...] = density.get_int_drhodS(pyom.salt,pyom.temp,np.abs(pyom.zt)[:,None],pyom)
    # stability frequency
    fxa = -pyom.grav / pyom.rho_0 / pyom.dzw[None,None,:] * pyom.maskW
    pyom.Nsqr[:,:,:-1,:] = fxa[:,:,:-1,None] * (density.get_rho(pyom.salt[:,:,1:,:],pyom.temp[:,:,1:,:],np.abs(pyom.zt)[:-1,None],pyom) - pyom.rho[:,:,:-1,:])
    pyom.Nsqr[:,:,-1,:] = pyom.Nsqr[:,:,-2,:]


def ugrid_to_tgrid(A,pyom):
    B = np.zeros_like(A)
    B[2:-2,:,:] = (pyom.dxu[2:-2, None, None] * A[2:-2, :, :] + pyom.dxu[1:-3, None, None] * A[1:-3, :, :]) / (2*pyom.dxt[2:-2, None, None])
    return B


def vgrid_to_tgrid(A,pyom):
    B = np.zeros_like(A)
    B[:,2:-2,:] = (pyom.area_v[:,2:-2,None] * A[:,2:-2,:] + pyom.area_v[:,1:-3,None] * A[:,1:-3,:]) / (2*pyom.area_t[:,2:-2,None])
    return B


def solve_tridiag(a, b, c, d):
    assert a.shape == b.shape and a.shape == c.shape and a.shape == d.shape
    return lapack.dgtsv(a[1:],b,c[:-1],d)[3]


def calc_diss(diss,K_diss,tag,pyom):
    diss_u = np.zeros_like(diss)
    ks = np.zeros_like(pyom.kbot)
    if tag == 'U':
        ks[1:-2,2:-2] = np.maximum(pyom.kbot[1:-2,2:-2],pyom.kbot[2:-1,2:-2]) - 1
        interpolator = ugrid_to_tgrid
    elif tag == 'V':
        ks[2:-2,1:-2] = np.maximum(pyom.kbot[2:-2,1:-2],pyom.kbot[2:-2,2:-1]) - 1
        interpolator = vgrid_to_tgrid
    else:
        raise ValueError("unknown tag {} (must be 'U' or 'V')".format(tag))
    diffusion.dissipation_on_wgrid(diss_u, pyom, aloc=diss, ks=ks)
    return K_diss + interpolator(diss_u, pyom)

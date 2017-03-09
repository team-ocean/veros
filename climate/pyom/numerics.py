from climate.pyom import cyclic, density, utilities, diffusion, pyom_method
from scipy.linalg import lapack

@pyom_method
def u_centered_grid(pyom, dyt, dyu, yt, yu):
    yu[0] = 0
    yu[1:] = np.cumsum(dyt[1:])

    yt[0] = yu[0] - dyt[0] * 0.5
    yt[1:] = 2 * yu[:-1]

    alternating_pattern = np.ones_like(yt)
    alternating_pattern[::2] = -1
    yt[...] = alternating_pattern * np.cumsum(alternating_pattern * yt)

    dyu[:-1] = yt[1:] - yt[:-1]
    dyu[-1] = 2*dyt[-1] - dyu[-2]

@pyom_method
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
    else:
        dxt_gl[pyom.nx+2:pyom.nx+4] = dxt_gl[pyom.nx+1]
        dxt_gl[:2] = dxt_gl[2]

    aloc[0,:] = pyom.dyt[2:pyom.ny+2]
    dyt_gl[2:pyom.ny+2] = aloc[0, :]

    dyt_gl[pyom.ny+2:pyom.ny+4] = dyt_gl[pyom.ny+1]
    dyt_gl[:2] = dyt_gl[2]

    """
    grid in east/west direction
    """
    u_centered_grid(pyom, dxt_gl, dxu_gl, xt_gl, xu_gl)
    xt_gl += pyom.x_origin - xu_gl[2]
    xu_gl += pyom.x_origin - xu_gl[2]

    if pyom.enable_cyclic_x:
        xt_gl[pyom.nx+2:pyom.nx+4] = xt_gl[2:4]
        xt_gl[:2] = xt_gl[pyom.nx:pyom.nx+2]
        xu_gl[pyom.nx+2:pyom.nx+4] = xt_gl[2:4]
        xu_gl[:2] = xu_gl[pyom.nx:pyom.nx+2]
        dxu_gl[pyom.nx+2:pyom.nx+4] = dxu_gl[2:4]
        dxu_gl[:2] = dxu_gl[pyom.nx:pyom.nx+2]

    """
    grid in north/south direction
    """
    u_centered_grid(pyom, dyt_gl, dyu_gl, yt_gl, yu_gl)
    yt_gl += pyom.y_origin - yu_gl[2]
    yu_gl += pyom.y_origin - yu_gl[2]

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
    u_centered_grid(pyom, pyom.dzt, pyom.dzw, pyom.zt, pyom.zw)
    pyom.zt -= pyom.zw[-1]
    pyom.zw -= pyom.zw[-1]  # zero at zw(nz)

    """
    metric factors
    """
    if pyom.coord_degree:
        pyom.cost = np.cos(pyom.yt*pyom.pi/180.)
        pyom.cosu = np.cos(pyom.yu*pyom.pi/180.)
        pyom.tantr = np.tan(pyom.yt*pyom.pi/180.) / pyom.radius
    else:
        pyom.cost[...] = 1.0
        pyom.cosu[...] = 1.0
        pyom.tantr[...] = 0.0

    """
    precalculate area of boxes
    """
    pyom.area_t = pyom.cost * pyom.dyt * pyom.dxt[:, np.newaxis]
    pyom.area_u = pyom.cost * pyom.dyt * pyom.dxu[:, np.newaxis]
    pyom.area_v = pyom.cosu * pyom.dyu * pyom.dxt[:, np.newaxis]

@pyom_method
def calc_beta(pyom):
    """
    calculate beta = df/dy
    """
    pyom.beta[:, 2:pyom.ny+2] = 0.5*((pyom.coriolis_t[:,3:pyom.ny+3] - pyom.coriolis_t[:,2:pyom.ny+2]) \
                                / pyom.dyu[2:pyom.ny+2] \
                                + (pyom.coriolis_t[:,2:pyom.ny+2] - pyom.coriolis_t[:,1:pyom.ny+1]) \
                                /pyom.dyu[1:pyom.ny+1])

@pyom_method
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
    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(pyom.maskU)
    pyom.maskV[...] = pyom.maskT
    pyom.maskV[:, :pyom.ny+3] = np.minimum(pyom.maskT[:,:pyom.ny+3], pyom.maskT[:,1:pyom.ny+4])
    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(pyom.maskV)
    pyom.maskZ[...] = pyom.maskT
    pyom.maskZ[:pyom.nx+3, :pyom.ny+3] = np.minimum(np.minimum(pyom.maskT[:pyom.nx+3, :pyom.ny+3],pyom.maskT[:pyom.nx+3, 1:pyom.ny+4]),pyom.maskT[1:pyom.nx+4, :pyom.ny+3])
    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(pyom.maskZ)
    pyom.maskW[...] = pyom.maskT
    pyom.maskW[:,:,:pyom.nz-1] = np.minimum(pyom.maskT[:,:,:pyom.nz-1],pyom.maskT[:,:,1:pyom.nz])

    """
    total depth
    """
    pyom.ht[...] = np.sum(pyom.maskT * pyom.dzt, axis=2)
    pyom.hu[...] = np.sum(pyom.maskU * pyom.dzt, axis=2)
    pyom.hv[...] = np.sum(pyom.maskV * pyom.dzt, axis=2)

    mask = (pyom.hu == 0).astype(np.float)
    pyom.hur[...] = 1. / (pyom.hu + mask) * (1-mask)
    mask = (pyom.hv == 0).astype(np.float)
    pyom.hvr[...] = 1. / (pyom.hv + mask) * (1-mask)

@pyom_method
def calc_initial_conditions(pyom):
    """
    calculate dyn. enthalp, etc
    """
    if np.sum(pyom.salt < 0.0):
        raise RuntimeError("encountered negative salinity")

    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(pyom.temp)
        cyclic.setcyclic_x(pyom.salt)

    pyom.rho[...] = density.get_rho(pyom,pyom.salt,pyom.temp,np.abs(pyom.zt)[:,None]) * pyom.maskT[...,None]
    pyom.Hd[...] = density.get_dyn_enthalpy(pyom,pyom.salt,pyom.temp,np.abs(pyom.zt)[:,None]) * pyom.maskT[...,None]
    pyom.int_drhodT[...] = density.get_int_drhodT(pyom,pyom.salt,pyom.temp,np.abs(pyom.zt)[:,None])
    pyom.int_drhodS[...] = density.get_int_drhodS(pyom,pyom.salt,pyom.temp,np.abs(pyom.zt)[:,None])

    fxa = -pyom.grav / pyom.rho_0 / pyom.dzw[None,None,:] * pyom.maskW
    pyom.Nsqr[:,:,:-1,:] = fxa[:,:,:-1,None] * (density.get_rho(pyom,pyom.salt[:,:,1:,:],pyom.temp[:,:,1:,:],np.abs(pyom.zt)[:-1,None]) - pyom.rho[:,:,:-1,:])
    pyom.Nsqr[:,:,-1,:] = pyom.Nsqr[:,:,-2,:]

@pyom_method
def ugrid_to_tgrid(pyom, a):
    b = np.zeros_like(a)
    b[2:-2,:,:] = (pyom.dxu[2:-2, None, None] * a[2:-2, :, :] + pyom.dxu[1:-3, None, None] * a[1:-3, :, :]) / (2*pyom.dxt[2:-2, None, None])
    return b

@pyom_method
def vgrid_to_tgrid(pyom, a):
    b = np.zeros_like(a)
    b[:,2:-2,:] = (pyom.area_v[:,2:-2,None] * a[:,2:-2,:] + pyom.area_v[:,1:-3,None] * a[:,1:-3,:]) / (2*pyom.area_t[:,2:-2,None])
    return b

@pyom_method
def solve_tridiag(pyom, a, b, c, d):
    """
    Solves a tridiagonal matrix system with diagonals a, b, c and RHS vector d.
    Uses LAPACK when running with NumPy, and otherwise the Thomas algorithm iterating over the
    last axis of the input arrays.
    """
    assert a.shape == b.shape and a.shape == c.shape and a.shape == d.shape
    try:
        return np.linalg.solve_tridiagonal(a,b,c,d)
    except AttributeError:
        return lapack.dgtsv(a.flatten()[1:],b.flatten(),c.flatten()[:-1],d.flatten())[3].reshape(a.shape)

@pyom_method
def calc_diss(pyom, diss, K_diss, tag):
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
    diffusion.dissipation_on_wgrid(pyom, diss_u, aloc=diss, ks=ks)
    return K_diss + interpolator(pyom, diss_u)

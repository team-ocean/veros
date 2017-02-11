import numpy as np

from climate.pyom import density, utilities

def isoneutral_diffusion_pre(pyom):
    """
    Isopycnal diffusion for tracer
    following functional formulation by Griffies et al
    Code adopted from MOM2.1
    """
    drdTS = np.zeros((pyom.nx+4,pyom.ny+4,pyom.nz,2))
    ddzt = np.zeros((pyom.nx+4,pyom.ny+4,pyom.nz,2))
    ddxt = np.zeros((pyom.nx+4,pyom.ny+4,pyom.nz,2))
    ddyt = np.zeros((pyom.nx+4,pyom.ny+4,pyom.nz,2))
    epsln = 1.e-20  # for double precision

    """
    drho_dt and drho_ds at centers of T cells
    """
    drdTS[:,:,:,0] = density.get_drhodT(pyom.salt[:,:,:,pyom.tau],pyom.temp[:,:,:,pyom.tau],np.abs(pyom.zt),pyom)*pyom.maskT
    drdTS[:,:,:,1] = density.get_drhodS(pyom.salt[:,:,:,pyom.tau],pyom.temp[:,:,:,pyom.tau],np.abs(pyom.zt),pyom)*pyom.maskT

    """
    gradients at top face of T cells
    """
    ddzt[:,:,:-1,0] = pyom.maskW[:,:,:-1] * (pyom.temp[:,:,1:,pyom.tau] - pyom.temp[:,:,:-1,pyom.tau]) / pyom.dzw[None,None,:-1]
    ddzt[:,:,:-1,1] = pyom.maskW[:,:,:-1] * (pyom.salt[:,:,1:,pyom.tau] - pyom.salt[:,:,:-1,pyom.tau]) / pyom.dzw[None,None,:-1]
    ddzt[...,-1,:] = 0.

    """
    gradients at eastern face of T cells
    """
    ddxt[:-1,:,:,0] = pyom.maskU[:-1,:,:] * (pyom.temp[1:,:,:,pyom.tau] - pyom.temp[:-1,:,:,pyom.tau]) / (pyom.dxu[:-1,None,None] * pyom.cost[None,:,None])
    ddxt[:-1,:,:,1] = pyom.maskU[:-1,:,:] * (pyom.salt[1:,:,:,pyom.tau] - pyom.salt[:-1,:,:,pyom.tau]) / (pyom.dxu[:-1,None,None] * pyom.cost[None,:,None])

    """
    gradients at northern face of T cells
    """
    ddyt[:,:-1,:,0] = pyom.maskV[:,:-1,:] * (pyom.temp[:,1:,:,pyom.tau] - pyom.temp[:,:-1,:,pyom.tau]) / pyom.dyu[None,:-1,None]
    ddyt[:,:-1,:,1] = pyom.maskV[:,:-1,:] * (pyom.salt[:,1:,:,pyom.tau] - pyom.salt[:,:-1,:,pyom.tau]) / pyom.dyu[None,:-1,None]

    """
    Compute Ai_ez and K11 on center of east face of T cell.
    """
    diffloc = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    diffloc[1:-2,2:-2,1:] = 0.25 * (pyom.K_iso[1:-2,2:-2,1:] + pyom.K_iso[1:-2,2:-2,:-1] + pyom.K_iso[2:-1,2:-2,1:] + pyom.K_iso[2:-1,2:-2,:-1])
    diffloc[1:-2,2:-2,0] = 0.5 * (pyom.K_iso[1:-2,2:-2,0] + pyom.K_iso[2:-1,2:-2,0])

    sumz = np.zeros((pyom.nx+1, pyom.ny, pyom.nz))
    for kr in xrange(2):
        ki = 0 if kr == 1 else 1
        for ip in xrange(2):
            drodxe = drdTS[1+ip:-2+ip,2:-2,ki:,0]*ddxt[1:-2,2:-2,ki:,0] + drdTS[1+ip:-2+ip,2:-2,ki:,1]*ddxt[1:-2,2:-2,ki:,1]
            drodze = drdTS[1+ip:-2+ip,2:-2,ki:,0]*ddzt[1+ip:-2+ip,2:-2,:-1+kr or None,0] + drdTS[1+ip:-2+ip,2:-2,ki:,1]*ddzt[1+ip:-2+ip,2:-2,:-1+kr or None,1]
            sxe = -drodxe / (np.minimum(0.,drodze)-epsln)
            taper = dm_taper(sxe,pyom.iso_slopec,pyom.iso_dslope)
            sumz[:,:,ki:] += pyom.dzw[None,None,:-1+kr or None] * pyom.maskU[1:-2,2:-2,ki:] * np.maximum(pyom.K_iso_steep, diffloc[1:-2,2:-2,ki:]*taper)
            pyom.Ai_ez[1:-2,2:-2,ki:,ip,kr] = taper * sxe * pyom.maskU[1:-2,2:-2,ki:]
    pyom.K_11[1:-2,2:-2,:] = sumz / (4. * pyom.dzt[None,None,:])

    """
    Compute Ai_nz and K_22 on center of north face of T cell.
    """
    diffloc = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    diffloc[2:-2, 1:-2, 1:] = 0.25 * (pyom.K_iso[2:-2,1:-2,1:] + pyom.K_iso[2:-2,1:-2,:-1] + pyom.K_iso[2:-2,2:-1,1:] + pyom.K_iso[2:-2,2:-1,:-1])
    diffloc[2:-2, 1:-2 ,0] = 0.5 * (pyom.K_iso[2:-2,1:-2,0] + pyom.K_iso[2:-2,2:-1,0])

    sumz = np.zeros((pyom.nx, pyom.ny+1, pyom.nz))
    for kr in xrange(2):
        ki = 0 if kr == 1 else 1
        for jp in xrange(2):
            drodyn = drdTS[2:-2,1+jp:-2+jp,ki:,0] * ddyt[2:-2,1:-2,ki:,0] + drdTS[2:-2,1+jp:-2+jp,ki:,1] * ddyt[2:-2,1:-2,ki:,1]
            drodzn = drdTS[2:-2,1+jp:-2+jp,ki:,0] * ddzt[2:-2,1+jp:-2+jp,:-1+kr or None,0] + drdTS[2:-2,1+jp:-2+jp,ki:,1] * ddzt[2:-2,1+jp:-2+jp,:-1+kr or None,1]
            syn = -drodyn / (np.minimum(0.,drodzn) - epsln)
            taper = dm_taper(syn,pyom.iso_slopec,pyom.iso_dslope)
            sumz[:,:,ki:] += pyom.dzw[None,None,:-1+kr or None] * pyom.maskV[2:-2,1:-2,ki:] * np.maximum(pyom.K_iso_steep, diffloc[2:-2,1:-2,ki:]*taper)
            pyom.Ai_nz[2:-2,1:-2,ki:,jp,kr] = taper * syn * pyom.maskV[2:-2,1:-2,ki:]
    pyom.K_22[2:-2,1:-2,:] = sumz / (4.*pyom.dzt[None,None,:])

    """
    compute Ai_bx, Ai_by and K33 on top face of T cell.
    """
    # eastward slopes at the top of T cells
    sumx = np.zeros((pyom.nx,pyom.ny,pyom.nz-1))
    for ip in xrange(2):
        for kr in xrange(2):
            drodxb = drdTS[2:-2,2:-2,kr:-1+kr or None,0] * ddxt[1+ip:-3+ip,2:-2,kr:-1+kr or None,0] + drdTS[2:-2,2:-2,kr:-1+kr or None,1] * ddxt[1+ip:-3+ip,2:-2,kr:-1+kr or None,1]
            drodzb = drdTS[2:-2,2:-2,kr:-1+kr or None,0] * ddzt[2:-2,2:-2,:-1,0] + drdTS[2:-2,2:-2,kr:-1+kr or None,1] * ddzt[2:-2,2:-2,:-1,1]
            sxb = -drodxb / (np.minimum(0.,drodzb) - epsln)
            taper = dm_taper(sxb,pyom.iso_slopec,pyom.iso_dslope)
            sumx += pyom.dxu[1+ip:-3+ip,None,None] * pyom.K_iso[2:-2,2:-2,:-1] * taper * sxb**2 * pyom.maskW[2:-2,2:-2,:-1]
            pyom.Ai_bx[2:-2,2:-2,:-1,ip,kr] = taper * sxb * pyom.maskW[2:-2,2:-2,:-1]

    # northward slopes at the top of T cells
    sumy = np.zeros((pyom.nx,pyom.ny,pyom.nz-1))
    for jp in xrange(2): # jp=0,1
        facty = pyom.cosu[1+jp:-3+jp]*pyom.dyu[1+jp:-3+jp]
        for kr in xrange(2): # kr=0,1
            drodyb = drdTS[2:-2,2:-2,kr:-1+kr or None,0] * ddyt[2:-2,1+jp:-3+jp,kr:-1+kr or None,0] + drdTS[2:-2,2:-2,kr:-1+kr or None,1] * ddyt[2:-2,1+jp:-3+jp,kr:-1+kr or None,1]
            drodzb = drdTS[2:-2,2:-2,kr:-1+kr or None,0] * ddzt[2:-2,2:-2,:-1,0] + drdTS[2:-2,2:-2,kr:-1+kr or None,1] * ddzt[2:-2,2:-2,:-1,1]
            syb = -drodyb / (np.minimum(0.,drodzb) - epsln)
            taper = dm_taper(syb,pyom.iso_slopec,pyom.iso_dslope)
            sumy += facty[None,:,None] * pyom.K_iso[2:-2,2:-2,:-1] * taper * syb**2 * pyom.maskW[2:-2,2:-2,:-1]
            pyom.Ai_by[2:-2,2:-2,:-1,jp,kr] = taper * syb * pyom.maskW[2:-2,2:-2,:-1]

    pyom.K_33[2:-2,2:-2,:-1] = sumx / (4*pyom.dxt[2:-2,None,None]) + sumy / (4*pyom.dyt[None,2:-2,None] * pyom.cost[None,2:-2,None])
    pyom.K_33[2:-2,2:-2,-1] = 0.


def isoneutral_diag_streamfunction(pyom):
    """
    calculate hor. components of streamfunction for eddy driven velocity
    for diagnostics purpose only
    """

    """
    meridional component at east face of "T" cells
    """
    K_gm_pad = utilities.pad_z_edges(pyom.K_gm)

    diffloc = 0.25 * (K_gm_pad[1:-2, 2:-2, 1:-1] + K_gm_pad[1:-2, 2:-2, :-2] + K_gm_pad[2:-1, 2:-2, 1:-1] + K_gm_pad[2:-1, 2:-2, :-2])
    sumz = np.sum(diffloc[..., None, None] * pyom.Ai_ez[1:-2, 2:-2, ...], axis=(3,4))
    pyom.B2_gm[1:-2, 2:-2, :] = 0.25 * sumz

    """
    zonal component at north face of "T" cells
    """
    diffloc = 0.25 * (K_gm_pad[2:-2, 1:-2, 1:-1] + K_gm_pad[2:-2, 1:-2, :-2] + K_gm_pad[2:-2, 2:-1, 1:-1] + K_gm_pad[2:-2, 2:-1, :-2])
    sumz = np.sum(diffloc[..., None, None] * pyom.Ai_nz[2:-2, 1:-2, ...], axis=(3,4))
    pyom.B1_gm[2:-2, 1:-2, :] = -0.25 * sumz


def dm_taper(sx,iso_slopec,iso_dslope):
    """
    tapering function for isopycnal slopes
    """
    return 0.5 * (1. + np.tanh((-np.abs(sx) + iso_slopec) / iso_dslope))


def check_isoneutral_slope_crit(pyom):
    """
    check linear stability criterion from Griffies et al
    """
    epsln = 1e-20
    if pyom.enable_neutral_diffusion:
        ft1 = 1.0 / (4.0 * pyom.K_iso_0 * pyom.dt_tracer + epsln)
        delta1a = np.min(pyom.dxt[2:-2, None, None] * np.abs(pyom.cost[None, 2:-2, None]) * pyom.dzt[None, None, :] * ft1)
        delta1b = np.min(pyom.dyt[None, 2:-2, None] * pyom.dzt[None, None, :] * ft1)
        delta_iso1 = min(pyom.dzt[0] * ft1 * pyom.dxt[-1] * np.abs(pyom.cost[-1]), delta1a, delta1b)

        print("diffusion grid factor delta_iso1 = {}".format(delta_iso1))
        if delta_iso1 < pyom.iso_slopec:
            raise RuntimeError("""
                   Without latitudinal filtering, delta_iso1 is the steepest
                   isoneutral slope available for linear stab of Redi and GM.
                   Maximum allowable isoneutral slope is specified as {}
                   integration will be unstable
                   """.format(pyom.iso_slopec))

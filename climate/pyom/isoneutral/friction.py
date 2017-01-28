import numpy as np

from climate.pyom import numerics

def isoneutral_friction(kbot, nz):
    """
    vertical friction using TEM formalism for eddy driven velocity
    """
    #integer :: i,j,k,ks
    #real*8 :: a_tri(nz),b_tri(nz),c_tri(nz),d_tri(nz),delta(nz),fxa
    #real*8 :: diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    #real*8 :: aloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)

    a_tri = np.zeros(pyom.nz)
    b_tri = np.zeros(pyom.nz)
    c_tri = np.zeros(pyom.nz)
    d_tri = np.zeros(pyom.nz)
    delta = np.zeros(pyom.nz)
    diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    aloc = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    if pyom.enable_implicit_vert_friction:
        aloc[...] = pyom.u[:,:,:,pyom.taup1]
    else:
        aloc[...] = pyom.u[:,:,:,pyom.tau]

    # implicit vertical friction of zonal momentum by GM
    for j in xrange(pyom.js_pe-1, pyom.je_pe): #j=js_pe-1,je_pe
        for i in xrange(pyom.is_pe-1, pyom.ie_pe): #i=is_pe-1,ie_pe
            ks = max(pyom.kbot[i,j], pyom.kbot[i+1,j])
            if ks > 0:
                for k in xrange(ks, pyom.nz-1): #k=ks,nz-1
                    fxa = 0.5 * (pyom.kappa_gm[i,j,k] + pyom.kappa_gm[i+1,j,k])
                    delta[k] = pyom.dt_mom / pyom.dzw[k] * fxa * pyom.maskU[i,j,k+1] * pyom.maskU[i,j,k]
                delta[pyom.nz-1] = 0.0
                a_tri[ks] = 0.0
                for k in xrange(ks+1, pyom.nz): #k=ks+1,nz
                    a_tri[k] = -delta[k-1] / pyom.dzt[k]
                b_tri[ks] = 1 + delta[ks] / pyom.dzt[ks]
                for k in xrange(ks+1, pyom.nz-1): #k=ks+1,nz-1
                    b_tri[k] = 1 + delta[k] / pyom.dzt[k] + delta[k-1] / pyom.dzt[k]
                b_tri[pyom.nz-1] = 1 + delta[pyom.nz-2] / dzt[pyom.nz-1]
                for k in xrange(ks, pyom.nz-1): #k=ks,nz-1
                    c_tri[k] = - delta[k] / pyom.dzt[k]
                c_tri[pyom.nz-1] = 0.0
                d_tri[ks:nz+1] = aloc[i,j,ks:] #  A u = d
                pyom.u[i,j,ks:,pyom.taup1] = numerics.solve_tridiag(a_tri[ks:],b_tri[ks:],c_tri[ks:],d_tri[ks:])
                pyom.du_mix[i,j,ks:] = pyom.du_mix[i,j,ks:] + (pyom.u[i,j,ks:,pyom.taup1] - aloc[i,j,ks:]) / pyom.dt_mom

    if pyom.enable_conserve_energy:
        # diagnose dissipation
        for k in xrange(pyom.nz-1): #k=1,nz-1
            for j in xrange(pyom.js_pe-1, pyom.je_pe): #j=js_pe-1,je_pe
                for i in xrange(pyom.is_pe-1, pyom.ie_pe): #i=is_pe-1,ie_pe
                    fxa = 0.5 * (pyom.kappa_gm[i,j,k] + pyom.kappa_gm[i+1,j,k])
                    pyom.flux_top[i,j,k] = fxa * (pyom.u[i,j,k+1,pyom.taup1] - pyom.u[i,j,k,pyom.taup1]) \
                                            / pyom.dzw[k] * pyom.maskU[i,j,k+1] * pyom.maskU[i,j,k]
        for k in xrange(pyom.nz-1): # k=1,nz-1
            for j in xrange(pyom.js_pe-1, pyom.je_pe): # j=js_pe-1,je_pe
                for i in xrange(pyom.is_pe-1, pyom.ie_pe): # i=is_pe-1,ie_pe
                    diss[i,j,k] = (pyom.u[i,j,k+1,pyom.tau] - pyom.u[i,j,k,pyom.tau]) * pyom.flux_top[i,j,k] / pyom.dzw[k]
        diss[:,:,-1] = 0.0
        numerics.ugrid_to_tgrid(diss,diss)
        pyom.K_diss_gm[...] = diss

    if pyom.enable_implicit_vert_friction:
        aloc[...] = pyom.v[:,:,:,pyom.taup1]
    else:
        aloc[...] = pyom.v[:,:,:,pyom.tau]

    # implicit vertical friction of meridional momentum by GM
    for j in xrange(pyom.js_pe-1, pyom.je_pe): # j=js_pe-1,je_pe
        for i in xrange(pyom.is_pe-1, pyom.ie_pe): # i=is_pe-1,ie_pe
            ks = max(pyom.kbot[i,j], pyom.kbot[i,j+1])
            if ks > 0:
                for k in xrange(ks, pyom.nz-1): # k=ks,nz-1
                    fxa = 0.5 * (pyom.kappa_gm[i,j,k] + pyom.kappa_gm[i,j+1,k])
                    delta[k] = pyom.dt_mom / pyom.dzw[k] * fxa * pyom.maskV[i,j,k+1] * pyom.maskV[i,j,k]
                delta[-1] = 0.0
                a_tri[ks] = 0.0
                for k in xrange(ks+1, pyom.nz): # k=ks+1,nz
                    a_tri[k] = - delta[k-1] / pyom.dzt[k]
                b_tri[ks] = 1 + delta[ks] / pyom.dzt[ks]
                for k in xrange(ks+1, pyom.nz-1): # k=ks+1,nz-1
                    b_tri[k] = 1 + delta[k] / pyom.dzt[k] + delta[k-1] / pyom.dzt[k]
                b_tri[-1] = 1 + delta[-2] / pyom.dzt[-1]
                for k in xrange(ks, nz-1): # k=ks,nz-1
                    c_tri[k] = -delta[k] / pyom.dzt[k]
                c_tri[-1] = 0.0
                d_tri[ks:] = aloc[i,j,ks:]
                pyom.v[i,j,ks:,pyom.taup1] = numerics.solve_tridiag(a_tri[ks:],b_tri[ks:],c_tri[ks:],d_tri[ks:])
                pyom.dv_mix[i,j,ks:] = pyom.dv_mix[i,j,ks:] + (pyom.v[i,j,ks:,pyom.taup1] - aloc[i,j,ks:]) / pyom.dt_mom

    if pyom.enable_conserve_energy:
        # diagnose dissipation
        for k in xrange(pyom.nz-1): # k=1,nz-1
            for j in xrange(pyom.js_pe-1, pyom.je_pe+1): # j=js_pe-1,je_pe
                for i in xrange(pyom.is_pe-1, pyom.ie_pe+1): # i=is_pe-1,ie_pe
                    fxa = 0.5 * (pyom.kappa_gm[i,j,k] + pyom.kappa_gm[i,j+1,k])
                    pyom.flux_top[i,j,k] = fxa * (pyom.v[i,j,k+1,pyom.taup1] - pyom.v[i,j,k,pyom.taup1]) \
                                            / pyom.dzw[k] * pyom.maskV[i,j,k+1] * pyom.maskV[i,j,k]
        for k in xrange(pyom.nz-1): #k=1,nz-1
            for j in xrange(pyom.js_pe-1, pyom.je_pe+1): # j=js_pe-1,je_pe
                for i in xrange(pyom.is_pe-1, pyom.ie_pe+1): # i=is_pe-1,ie_pe
                    diss[i,j,k] = (pyom.v[i,j,k+1,pyom.tau] - pyom.v[i,j,k,pyom.tau]) * pyom.flux_top[i,j,k] / pyom.dzw[k]
        diss[:,:,-1] = 0.0
        numerics.vgrid_to_tgrid(diss,diss)
        pyom.K_diss_gm = pyom.K_diss_gm + diss

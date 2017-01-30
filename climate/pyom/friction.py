import math
import numpy as np

from climate.pyom import numerics


def explicit_vert_friction(pyom):
    """
    explicit vertical friction
    dissipation is calculated and added to K_diss_v
    """

    # real*8 :: diss(pyom.is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz),fxa
    diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    """
    vertical friction of zonal momentum
    """
    for k in xrange(pyom.nz-1): # k = 1,nz-1
        for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
            for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                fxa = 0.5 * (pyom.kappaM[i,j,k] + pyom.kappaM[i+1,j,k])
                pyom.flux_top[i,j,k] = fxa * (pyom.u[i,j,k+1,pyom.tau] - pyom.u[i,j,k,pyom.tau]) \
                                        / pyom.dzw[k]*pyom.maskU[i,j,k+1]*pyom.maskU[i,j,k]
    pyom.flux_top[:,:,pyom.nz-1] = 0.0
    k = 0
    pyom.du_mix[:,:,k] = pyom.flux_top[:,:,k] / pyom.dzt[k] * pyom.maskU[:,:,k]
    for k in xrange(1,pyom.nz): # k = 2,nz
        pyom.du_mix[:,:,k] = (pyom.flux_top[:,:,k] - pyom.flux_top[:,:,k-1]) / pyom.dzt[k] * pyom.maskU[:,:,k]

    """
    diagnose dissipation by vertical friction of zonal momentum
    """
    for k in xrange(pyom.nz-1): # k = 1,nz-1
        for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
            for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                diss[i,j,k] = (pyom.u[i,j,k+1,pyom.tau] - pyom.u[i,j,k,pyom.tau]) * pyom.flux_top[i,j,k] / pyom.dzw[k]
    diss[:,:,pyom.nz-1] = 0.0
    diss[...] = numerics.ugrid_to_tgrid(diss,pyom)
    pyom.K_diss_v += diss

    """
    vertical friction of meridional momentum
    """
    for k in xrange(pyom.nz-1): # k = 1,nz-1
        for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
            for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                fxa = 0.5 * (pyom.kappaM[i,j,k]+pyom.kappaM[i,j+1,k])
                pyom.flux_top[i,j,k] = fxa * (pyom.v[i,j,k+1,pyom.tau] - pyom.v[i,j,k,pyom.tau]) \
                                        / pyom.dzw[k] * pyom.maskV[i,j,k+1] * pyom.maskV[i,j,k]
    pyom.flux_top[:,:,pyom.nz-1] = 0.0
    k = 0
    pyom.dv_mix[:,:,k] = pyom.flux_top[:,:,k] / pyom.dzt[k] * pyom.maskV[:,:,k]
    for k in xrange(1,pyom.nz): # k = 2,nz
        pyom.dv_mix[:,:,k] = (pyom.flux_top[:,:,k]  -pyom.flux_top[:,:,k-1]) / pyom.dzt[k] * pyom.maskV[:,:,k]

    """
    diagnose dissipation by vertical friction of meridional momentum
    """
    for k in xrange(pyom.nz-1): # k = 1,nz-1
        for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
            for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                diss[i,j,k] = (pyom.v[i,j,k+1,pyom.tau] - pyom.v[i,j,k,pyom.tau]) \
                                * pyom.flux_top[i,j,k] / pyom.dzw[k]
    diss[:,:,pyom.nz-1] = 0.0
    diss[...] = numerics.vgrid_to_tgrid(diss,pyom)
    pyom.K_diss_v += diss

    if not pyom.enable_hydrostatic:
        """
        vertical friction of vertical momentum
        """
        for k in xrange(pyom.nz-1): # k = 1,nz-1
            for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
                for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                    fxa = 0.5 * (pyom.kappaM[i,j,k] + pyom.kappaM[i,j,k+1])
                    pyom.flux_top[i,j,k] = fxa * (pyom.w[i,j,k+1,pyom.tau] - pyom.w[i,j,k,pyom.tau]) \
                                            / pyom.dzt[k+1] * pyom.maskW[i,j,k+1] * pyom.maskW[i,j,k]
        pyom.flux_top[:,:,pyom.nz] = 0.0
        k = 0
        pyom.dw_mix[:,:,k] = pyom.flux_top[:,:,k] / pyom.dzw[k] * pyom.maskW[:,:,k]
        for k in xrange(1,pyom.nz): # k = 2,nz
            pyom.dw_mix[:,:,k] = (pyom.flux_top[:,:,k] - pyom.flux_top[:,:,k-1]) / pyom.dzw[k] * pyom.maskW[:,:,k]

        """
        diagnose dissipation by vertical friction of vertical momentum
        """
        # to be implemented


def implicit_vert_friction(pyom):
    """
    vertical friction
    dissipation is calculated and added to K_diss_v
    """

    # real*8 :: a_tri(nz),b_tri(nz),c_tri(nz),d_tri(nz),delta(nz),fxa
    # real*8 :: diss(pyom.is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    a_tri = np.zeros(pyom.nz)
    b_tri = np.zeros(pyom.nz)
    c_tri = np.zeros(pyom.nz)
    d_tri = np.zeros(pyom.nz)
    delta = np.zeros(pyom.nz)
    diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    """
    implicit vertical friction of zonal momentum
    """
    kss = np.maximum(pyom.kbot[1:-2, 1:-2], pyom.kbot[2:-1, 1:-2]) - 1
    for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
        for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
            #ks = max(pyom.kbot[i,j],pyom.kbot[i+1,j]) - 1
            ks = kss[i-1, j-1]
            if ks >= 0:
                fxa = 0.5 * (pyom.kappaM[i,j,ks:pyom.nz-1] + pyom.kappaM[i+1,j,ks:pyom.nz-1])
                delta[ks:pyom.nz-1] = pyom.dt_mom / pyom.dzw[ks:pyom.nz-1] * fxa * pyom.maskU[i,j,ks+1:pyom.nz] * pyom.maskU[i,j,ks:pyom.nz-1]
                #for k in xrange(ks,pyom.nz-1): # k = ks,nz-1
                #    fxa = 0.5 * (pyom.kappaM[i,j,k] + pyom.kappaM[i+1,j,k])
                #    delta[k] = pyom.dt_mom / pyom.dzw[k] * fxa * pyom.maskU[i,j,k+1] * pyom.maskU[i,j,k]
                delta[pyom.nz-1] = 0.0
                a_tri[ks] = 0.0
                a_tri[ks+1:] = -delta[ks:-1] / pyom.dzt[ks+1:]
                #for k in xrange(ks+1,pyom.nz): # k = ks+1,nz
                #    a_tri[k] = -delta[k-1] / pyom.dzt[k]
                tmp1 = delta[ks:-1] / pyom.dzt[ks:-1]
                tmp2 = delta[ks:-1] / pyom.dzt[ks+1:]
                b_tri[ks:] = 1
                b_tri[ks:-1] += tmp1
                b_tri[ks+1:] += tmp2
                #b_tri[ks] = 1 + delta[ks] / pyom.dzt[ks]
                #for k in xrange(ks+1,pyom.nz-1): # k = ks+1,nz-1
                #    b_tri[k] = 1 + delta[k] / pyom.dzt[k] + delta[k-1] / pyom.dzt[k]
                #b_tri[pyom.nz-1] = 1 + delta[pyom.nz-2] / pyom.dzt[pyom.nz-1]
                c_tri[ks:-1] = -delta[ks:-1] / pyom.dzt[ks:-1]
                #for k in xrange(ks,pyom.nz-1): # k = ks,nz-1
                #    c_tri[k] = -delta[k] / pyom.dzt[k]
                c_tri[pyom.nz-1] = 0.0
                d_tri[ks:] = pyom.u[i,j,ks:,pyom.tau]
                pyom.u[i,j,ks:,pyom.taup1] = numerics.solve_tridiag(a_tri[ks:],b_tri[ks:],c_tri[ks:],d_tri[ks:])
            #pyom.du_mix[i,j,:] = (pyom.u[i,j,:,pyom.taup1] - pyom.u[i,j,:,pyom.tau]) / pyom.dt_mom

    pyom.du_mix[1:-2, 1:-2] = (pyom.u[1:-2,1:-2,:,pyom.taup1] - pyom.u[1:-2,2:-1,:,pyom.tau]) / pyom.dt_mom

    """
    diagnose dissipation by vertical friction of zonal momentum
    """
    fxa = 0.5 * (pyom.kappaM[1:-2, 1:-2, :-1] + pyom.kappaM[2:-1, 1:-2, :-1])
    pyom.flux_top[1:-2, 1:-2, :-1] = fxa * (pyom.u[1:-2, 1:-2, 1:, pyom.taup1] - pyom.u[1:-2, 1:-2, :-1, pyom.taup1]) \
            / pyom.dzw[:-1] * pyom.maskU[1:-2, 1:-2, 1:] * pyom.maskU[1:-2, 1:-2, :-1]
    #for k in xrange(pyom.nz-1): # k = 1,nz-1
    #    for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
    #        for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
    #            fxa = 0.5 * (pyom.kappaM[i,j,k]+pyom.kappaM[i+1,j,k])
    #            pyom.flux_top[i,j,k] = fxa * (pyom.u[i,j,k+1,pyom.taup1] - pyom.u[i,j,k,pyom.taup1]) \
    #                                   / pyom.dzw[k] * pyom.maskU[i,j,k+1] * pyom.maskU[i,j,k]
    diss[1:-2, 1:-2, :-1] = (pyom.u[1:-2, 1:-2, 1:, pyom.tau] - pyom.u[1:-2, 1:-2, :-1, pyom.tau]) * pyom.flux_top[1:-2, 1:-2, :-1] / pyom.dzw[1:]
    #for k in xrange(pyom.nz-1): # k = 1,nz-1
    #    for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
    #        for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
    #            diss[i,j,k] = (pyom.u[i,j,k+1,pyom.tau] - pyom.u[i,j,k,pyom.tau]) * pyom.flux_top[i,j,k] / pyom.dzw[k]
    diss[:,:,pyom.nz-1] = 0.0
    diss[...] = numerics.ugrid_to_tgrid(diss,pyom)
    pyom.K_diss_v += diss

    """
    implicit vertical friction of meridional momentum
    """
    kss = np.maximum(pyom.kbot[1:-2, 1:-2], pyom.kbot[1:-2, 2:-1]) - 1
    for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
        for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
            #ks = max(pyom.kbot[i,j],pyom.kbot[i,j+1]) - 1
            ks = kss[i-1, j-1]
            if ks >= 0:
                fxa = 0.5 * (pyom.kappaM[i, j, ks:-1] + pyom.kappaM[i, j+2, ks:-1])
                delta[ks:-1] = pyom.dt_mom / pyom.dzw[ks:-1] * fxa * pyom.maskV[i,j,ks+1:] * pyom.maskV[i,j,ks:-1]
                #for k in xrange(ks,pyom.nz-1): # k = ks,nz-1
                #    fxa = 0.5 * (pyom.kappaM[i,j,k] + pyom.kappaM[i,j+1,k])
                #    delta[k] = pyom.dt_mom / pyom.dzw[k] * fxa * pyom.maskV[i,j,k+1] * pyom.maskV[i,j,k]
                delta[pyom.nz-1] = 0.0
                a_tri[ks] = 0.0
                a_tri[ks+1:] = -delta[ks:-1] / pyom.dzt[ks+1:]
                #for k in xrange(ks+1,pyom.nz): # k = ks+1,nz
                #    a_tri[k] = -delta[k-1] / pyom.dzt[k]
                tmp1 = delta[ks:-1] / pyom.dzt[ks:-1]
                tmp2 = delta[ks:-1] / pyom.dzt[ks+1:]
                b_tri[ks:] = 1
                b_tri[ks:-1] += tmp1
                b_tri[ks+1:] += tmp2
                #b_tri[ks] = 1 + delta[ks] / pyom.dzt[ks]
                #for k in xrange(ks+1,pyom.nz-1): # k = ks+1,nz-1
                #    b_tri[k] = 1 + delta[k] / pyom.dzt[k] + delta[k-1] / pyom.dzt[k]
                #b_tri[pyom.nz-1] = 1 + delta[pyom.nz-2] / pyom.dzt[pyom.nz-1]
                c_tri[ks:-1] = -delta[ks:-1] / pyom.dzt[ks:-1]
                #for k in xrange(ks,pyom.nz-1): # k = ks,nz-1
                #    c_tri[k] = -delta[k] / pyom.dzt[k]
                c_tri[pyom.nz-1] = 0.0
                d_tri[ks:] = pyom.v[i,j,ks:,pyom.tau]
                pyom.v[i,j,ks:,pyom.taup1] = numerics.solve_tridiag(a_tri[ks:],b_tri[ks:],c_tri[ks:],d_tri[ks:])
            #pyom.dv_mix[i,j,:] = (pyom.v[i,j,:,pyom.taup1] - pyom.v[i,j,:,pyom.tau]) / pyom.dt_mom
    pyom.dv_mix[1:-2, 1:-2] = (pyom.v[1:-2, 1:-2, :, pyom.taup1] - pyom.v[1:-2, 1:-2, :, pyom.tau]) / pyom.dt_mom

    """
    diagnose dissipation by vertical friction of meridional momentum
    """
    fxa = 0.5*(pyom.kappaM[1:-2, 1:-2, :-1] + pyom.kappaM[1:-2, 2:-1, :-1])
    pyom.flux_top[1:-2, 1:-2, :-1] = fxa * (pyom.v[1:-2, 1:-2, 1:, pyom.taup1] - pyom.v[1:-2, 1:-2, :-1, pyom.taup1]) \
            / pyom.dzw[:-1] * pyom.maskV[1:-2, 1:-2, 1:] * pyom.maskV[1:-2, 1:-2, :-1]
    #for k in xrange(pyom.nz-1): # k = 1,nz-1
    #    for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
    #        for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
    #            fxa = 0.5 * (pyom.kappaM[i,j,k] + pyom.kappaM[i,j+1,k])
    #            pyom.flux_top[i,j,k] = fxa * (pyom.v[i,j,k+1,pyom.taup1] - pyom.v[i,j,k,pyom.taup1]) \
    #                                    / pyom.dzw[k] * pyom.maskV[i,j,k+1] * pyom.maskV[i,j,k]
    diss[1:-2, 1:-2, :-1] = (pyom.v[1:-2, 1:-2, 1:, pyom.tau] - pyom.v[1:-2, 1:-2, :-1, pyom.tau]) * pyom.flux_top[1:-2, 1:-2, :-1] / pyom.dzw[:-1]
    #for k in xrange(pyom.nz-1): # k = 1,nz-1
    #    for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
    #        for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
    #            diss[i,j,k] = (pyom.v[i,j,k+1,pyom.tau] - pyom.v[i,j,k,pyom.tau]) * pyom.flux_top[i,j,k] / pyom.dzw[k]
    diss[:,:,pyom.nz-1] = 0.0
    diss = numerics.vgrid_to_tgrid(diss,pyom)
    pyom.K_diss_v += diss

    if not pyom.enable_hydrostatic:
        # !if (my_pe==0) print'(/a/)','ERROR: implicit vertical friction for vertical velocity not implemented'
        # !halt_stop(' in implicit_vert_friction')
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                ks = pyom.kbot[i,j] - 1
                if ks >= 0:
                    delta[ks:-1] = pyom.dt_mom / pyom.dzt[ks+1:] * 0.5 * (pyom.kappaM[i,k,ks:-1] + pyom.kappaM[i,j,ks+1])
                    #for k in xrange(ks,pyom.nz-1): # k = ks,nz-1
                    #    delta[k] = pyom.dt_mom / pyom.dzt[k+1] * 0.5 * (pyom.kappaM[i,j,k] + pyom.kappaM[i,j,k+1])
                    delta[pyom.nz-1] = 0.0
                    a_tri[ks+1:-1] = -delta[ks:-2] / pyom.dzw[ks+1:-1]
                    #for k in xrange(ks+1,pyom.nz-1): # k = ks+1,nz-1
                    #    a_tri[k] = -delta[k-1] / pyom.dzw[k]
                    a_tri[ks] = 0.0
                    a_tri[pyom.nz-1] = -delta[pyom.nz-2]/(0.5*pyom.dzw[pyom.nz-1])
                    tmp1 = pyom.delta[ks:-1] / pyom.dzw[ks:-1]
                    b_tri[ks:] = 1
                    b_tri[ks:-1] += tmp1
                    b_tri[ks+1:-1] += delta[ks:-2] / pyom.dzw[ks+1:-1]
                    b_tri[-1] += delta[pyom.nz-2]/(0.5*pyom.dzw[nz-1])
                    #for k in xrange(ks+1,pyom.nz-1): # k = ks+1,nz-1
                    #    b_tri[k] = 1 + delta[k] / pyom.dzw[k] + delta[k-1] / pyom.dzw[k]
                    #b_tri[pyom.nz-1] = 1 + delta[pyom.nz-2]/(0.5*pyom.dzw[nz-1])
                    #b_tri[ks] = 1 + delta[ks] / pyom.dzw[ks]
                    c_tri[ks:-1] = - delta[ks:-1] / pyom.dzw[ks:-1]
                    #for k in xrange(ks,pyom.nz-1): # k = ks,nz-1
                    #    c_tri[k] = - delta[k] / pyom.dzw[k]
                    c_tri[pyom.nz-1] = 0.0
                    d_tri[ks:] = pyom.w[i,j,ks:,pyom.tau]
                    pyom.w[i,j,ks:,pyom.taup1] = numerics.solve_tridiag(a_tri[ks:],b_tri[ks:],c_tri[ks:],d_tri[ks:])
                #pyom.dw_mix[i,j,:] = (pyom.w[i,j,:,pyom.taup1] - pyom.w[i,j,:,pyom.tau]) / pyom.dt_mom
        pyom.dw_mix[2:-2, 2:-2] = (pyom.w[2:-2,2:-2,:,pyom.taup1] - pyom.w[2:-2,2:-2,:,pyom.tau]) / pyom.dt_mom

        """
        diagnose dissipation by vertical friction of vertical momentum
        """
        fxa = 0.5 * (pyom.kappaM[1:-2, 1:-2, :-1] + pyom.kappaM[1:-2,1:-2,1:])
        pyom.flux_top[1:-2,1:-2,:-1] = fxa * (pyom.w[1:-2,1:-2,1:,pyom.taup1] - pyom.w[1:-2,1:-2,:-1,pyom.taup1]) \
                / pyom.dzt[1:] * pyom.maskW[1:-2, 1:-2, 1:] * pyom.maskW[1:-2, 1:-2, :-1]
        #for k in xrange(pyom.nz-1): # k = 1,nz-1
        #    for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
        #        for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
        #            fxa = 0.5 * (pyom.kappaM[i,j,k] + pyom.kappaM[i,j,k+1])
        #            pyom.flux_top[i,j,k] = fxa * (pyom.w[i,j,k+1,pyom.taup1] - pyom.w[i,j,k,pyom.taup1]) \
        #                                    / pyom.dzt[k+1] * pyom.maskW[i,j,k+1] * pyom.maskW[i,j,k]
        diss[1:-2, 1:-2,:-1] = (pyom.w[1:-2,1:-2,1:,pyom.tau] - pyom.w[1:-2,1:-2,:-1,pyom.tau]) * pyom.flux_top[1:-2,1:-2,:-1] / pyom.dzt[1:]
        #for k in xrange(pyom.nz-1): # k = 1,nz-1
        #    for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
        #        for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
        #            diss[i,j,k] = (pyom.w[i,j,k+1,pyom.tau] - pyom.w[i,j,k,pyom.tau]) * pyom.flux_top[i,j,k] / pyom.dzt[k+1]
        diss[:,:,pyom.nz-1] = 0.0
        K_diss_v += diss


def rayleigh_friction(pyom):
    """
    interior Rayleigh friction
    dissipation is calculated and added to K_diss_bot
    """
    # real*8 :: diss(pyom.is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    for k in xrange(pyom.nz): # k = 1,nz
        pyom.du_mix[:,:,k] = pyom.du_mix[:,:,k] - pyom.maskU[:,:,k] * pyom.r_ray * pyom.u[:,:,k,pyom.tau]
    if pyom.enable_conserve_energy:
        for k in xrange(pyom.nz): # k = 1,nz
            diss[:,:,k] = pyom.maskU[:,:,k] * pyom.r_ray * pyom.u[:,:,k]**2
        pyom.K_diss_bot[...] = numerics.calc_diss(diss,pyom.K_diss_bot,'U',pyom)
    for k in xrange(pyom.nz): # k = 1,nz
        pyom.dv_mix[:,:,k] = pyom.dv_mix[:,:,k] - pyom.maskV[:,:,k] * pyom.r_ray * pyom.v[:,:,k,pyom.tau]
    if pyom.enable_conserve_energy:
        for k in xrange(pyom.nz): # k = 1,nz
            diss[:,:,k] = pyom.maskV[:,:,k] * pyom.r_ray * pyom.v[:,:,k]**2
        pyom.K_diss_bot[...] = numerics.calc_diss(diss,pyom.K_diss_bot,'V',pyom)
    if not pyom.enable_hydrostatic:
        raise NotImplementedError("Rayleigh friction for vertical velocity not implemented")


def linear_bottom_friction(pyom):
    """
    linear bottom friction
    dissipation is calculated and added to K_diss_bot
    """
    # real*8 :: diss(pyom.is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    if pyom.enable_bottom_friction_var:
        """
        with spatially varying coefficient
        """
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                k = max(pyom.kbot[i,j],pyom.kbot[i+1,j]) - 1
                if k >= 0:
                    pyom.du_mix[i,j,k] -= pyom.maskU[i,j,k] * pyom.r_bot_var_u[i,j] * pyom.u[i,j,k,pyom.tau]
        if pyom.enable_conserve_energy:
            diss[...] = 0.0
            for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
                for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                    k = max(pyom.kbot(i,j),pyom.kbot(i+1,j)) - 1
                    if k >= 0:
                        diss[i,j,k] = pyom.maskU[i,j,k] * pyom.r_bot_var_u[i,j] * pyom.u[i,j,k,pyom.tau]**2
            pyom.K_diss_bot[...] = numerics.calc_diss(diss,pyom.K_diss_bot,'U',pyom)

        for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                k = max(pyom.kbot(i,j+1),pyom.kbot(i,j)) - 1
                if k >= 0:
                    pyom.dv_mix[i,j,k] -= pyom.maskV[i,j,k] * pyom.r_bot_var_v(i,j) * pyom.v[i,j,k,pyom.tau]
        if pyom.enable_conserve_energy:
            diss[...] = 0.0
            for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
                for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                    k = max(pyom.kbot(i,j+1),pyom.kbot(i,j)) - 1
                    if k >= 0:
                        diss[i,j,k] = pyom.maskV[i,j,k] * pyom.r_bot_var_v(i,j) * pyom.v[i,j,k,pyom.tau]**2
            pyom.K_diss_bot[...] = numerics.calc_diss(diss,pyom.K_diss_bot,'V',pyom)
    else:
        """
        with constant coefficient
        """
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                k = max(pyom.kbot[i,j],pyom.kbot[i+1,j]) - 1
                if k >= 0:
                    pyom.du_mix[i,j,k] -= pyom.maskU[i,j,k] * pyom.r_bot * pyom.u[i,j,k,pyom.tau]
        if pyom.enable_conserve_energy:
            diss[...] = 0.0
            for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
                for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                    k = max(pyom.kbot[i,j],pyom.kbot[i+1,j]) - 1
                    if k>= 0:
                        diss[i,j,k] = pyom.maskU[i,j,k] * pyom.r_bot * pyom.u[i,j,k,pyom.tau]**2
            pyom.K_diss_bot[...] = numerics.calc_diss(diss,pyom.K_diss_bot,'U',pyom)

        for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                k = max(pyom.kbot[i,j+1],pyom.kbot[i,j]) - 1
                if k >= 0:
                    pyom.dv_mix[i,j,k] -= pyom.maskV[i,j,k] * pyom.r_bot * pyom.v[i,j,k,pyom.tau]
        if pyom.enable_conserve_energy:
            diss[...] = 0.0
            for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
                for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                    k = max(pyom.kbot[i,j+1],pyom.kbot[i,j]) - 1
                    if k >= 0:
                        diss[i,j,k] = pyom.maskV[i,j,k] * pyom.r_bot * pyom.v[i,j,k,pyom.tau]**2
            pyom.K_diss_bot[...] = numerics.calc_diss(diss,pyom.K_diss_bot,'V',pyom)

    if not pyom.enable_hydrostatic:
        raise NotImplementedError("bottom friction for vertical velocity not implemented")


def quadratic_bottom_friction(pyom):
    """
    quadratic bottom friction
    dissipation is calculated and added to K_diss_bot
    """
    # real*8 :: diss(pyom.is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz),fxa
    # real*8 :: aloc(pyom.is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    aloc = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    # we might want to account for EKE in the drag, also a tidal residual
    aloc[...] = 0.0
    for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
        for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
            k = max(pyom.kbot[i,j],pyom.kbot[i+1,j]) - 1
            if k >= 0:
                fxa = pyom.maskV[i,j,k] * pyom.v[i,j,k,pyom.tau]**2 + pyom.maskV[i,j-1,k] * pyom.v[i,j-1,k,pyom.tau]**2 \
                      + pyom.maskV[i+1,j,k] * pyom.v[i+1,j,k,pyom.tau]**2 + pyom.maskV[i+1,j-1,k] * pyom.v[i+1,j-1,k,pyom.tau]**2
                fxa = np.sqrt(pyom.u[i,j,k,pyom.tau]**2 + 0.25*fxa)
                aloc[i,j] = pyom.maskU[i,j,k] * pyom.r_quad_bot * pyom.u[i,j,k,pyom.tau] * fxa / pyom.dzt[k]
                pyom.du_mix[i,j,k] -= aloc[i,j]

    if pyom.enable_conserve_energy:
        diss[...] = 0.0
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                k = max(pyom.kbot[i,j],pyom.kbot[i+1,j]) - 1
                if k >= 0:
                    diss[i,j,k] = aloc(i,j) * pyom.u[i,j,k,pyom.tau]
        pyom.K_diss_bot[...] = numerics.calc_diss(diss,pyom.K_diss_bot,'U',pyom)

    aloc = 0.0
    for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
        for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
            k = max(pyom.kbot[i,j+1],pyom.kbot[i,j]) - 1
            if k >= 0:
                fxa = pyom.maskU[i,j,k] * pyom.u[i,j,k,pyom.tau]**2 + pyom.maskU[i-1,j,k] * pyom.u[i-1,j,k,pyom.tau]**2 \
                      + pyom.maskU[i,j+1,k] * pyom.u[i,j+1,k,pyom.tau]**2 + pyom.maskU[i-1,j+1,k] * pyom.u[i-1,j+1,k,pyom.tau]**2
                fxa = np.sqrt(pyom.v[i,j,k,pyom.tau]**2 + 0.25*fxa)
                aloc[i,j] = pyom.maskV[i,j,k] * pyom.r_quad_bot * pyom.v[i,j,k,pyom.tau] * fxa / pyom.dzt[k]
                pyom.dv_mix[i,j,k] -= aloc[i,j]

    if pyom.enable_conserve_energy:
        diss[...] = 0.0
        for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                k = max(pyom.kbot[i,j+1],pyom.kbot[i,j]) - 1
                if k >= 0:
                    diss[i,j,k] = aloc[i,j] * pyom.v[i,j,k,pyom.tau]
        pyom.K_diss_bot[...] = numerics.calc_diss(diss,pyom.K_diss_bot,'V',pyom)

    if not pyom.enable_hydrostatic:
        raise NotImplementedError("bottom friction for vertical velocity not implemented")


def harmonic_friction(pyom):
    """
    horizontal harmonic friction
    dissipation is calculated and added to K_diss_h
    """
    # real*8 :: diss(pyom.is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz),fxa
    diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    is_ = pyom.is_pe - pyom.onx
    ie_ = pyom.ie_pe + pyom.onx
    js_ = pyom.js_pe - pyom.onx
    je_ = pyom.je_pe + pyom.onx

    """
    Zonal velocity
    """
    if pyom.enable_hor_friction_cos_scaling:
        fxa = (pyom.cost**pyom.hor_friction_cosPower) * np.ones(pyom.nx+3)[:, np.newaxis]
        pyom.flux_east[:-1] = pyom.A_h * fxa[:,:,np.newaxis] * (pyom.u[1:,:,:,pyom.tau] - pyom.u[:-1,:,:,pyom.tau]) \
                / (pyom.cost * pyom.dxt[1:, np.newaxis])[:,:,np.newaxis] * pyom.maskU[1:] * pyom.maskU[:-1]
        #for j in xrange(js_,je_): # j = js,je
        #    fxa = pyom.cost[j]**pyom.hor_friction_cosPower
        #    for i in xrange(is_,ie_-1): # i = is,ie-1
        #        pyom.flux_east[i,j,:] = fxa * pyom.A_h * (pyom.u[i+1,j,:,pyom.tau] - pyom.u[i,j,:,pyom.tau]) \
        #                                 / (pyom.cost[j] * pyom.dxt[i+1]) * pyom.maskU[i+1,j,:] * pyom.maskU[i,j,:]
        fxa = (pyom.cosu[:-1]**pyom.hor_friction_cosPower) * np.ones(pyom.nx+4)[:,np.newaxis]
        pyom.flux_north[:,:-1] = pyom.A_h * fxa[:,:,np.newaxis] * (pyom.u[:,1:,:,pyom.tau] - pyom.u[:,:-1,:,pyom.tau]) \
                / pyom.dyu[np.newaxis,:-1,np.newaxis] * pyom.maskU[:,1:] * pyom.maskU[:,:-1] * pyom.cosu[np.newaxis,:-1,np.newaxis]
        #for j in xrange(js_,je_-1): # j = js,je-1
        #    fxa = pyom.cosu[j]**pyom.hor_friction_cosPower
        #    pyom.flux_north[:,j,:] = fxa * pyom.A_h * (pyom.u[:,j+1,:,pyom.tau] - pyom.u[:,j,:,pyom.tau]) \
        #                             / pyom.dyu[j] * pyom.maskU[:,j+1,:] * pyom.maskU[:,j,:] * pyom.cosu[j]
    else:
        pyom.flux_east[:-1, :] = pyom.A_h * (pyom.u[1:,:,:,pyom.tau] - pyom[:-1,:,:,pyom.tau]) \
                / (pyom.cost * pyom.dxt[1:, np.newaxis])[:,:,np.newaxis] * pyom.maskU[1:] * pyom.maskU[:-1]
        #for j in xrange(js_,je_): # j = js,je
        #    for i in xrange(is_,ie_-1): # i = is,ie-1
        #        pyom.flux_east[i,j,:] = pyom.A_h * (pyom.u[i+1,j,:,pyom.tau] - pyom.u[i,j,:,pyom.tau]) \
        #                                 / (pyom.cost[j] * pyom.dxt[i+1]) * pyom.maskU[i+1,j,:] * pyom.maskU[i,j,:]
        pyom.flux_north[:,:-1,:] = pyom.A_h * (pyom.u[:,1:,:,pyom.tau] - pyom.u[:,j,:,pyom.tau]) \
                / pyom.dyu[np.newaxis, :-1, np.newaxis] * pyom.maskU[:,1:] * pyom.maskU[:,:-1] * pyom.cosu[np.newaxis,:-1,np.newaxis]
        #for j in xrange(js_,je_-1): # j = js,je-1
        #    pyom.flux_north[:,j,:] = pyom.A_h * (pyom.u[:,j+1,:,pyom.tau] - pyom.u[:,j,:,pyom.tau]) \
        #                              / pyom.dyu[j] * pyom.maskU[:,j+1,:] * pyom.maskU[:,j,:] * pyom.cosu[j]
    pyom.flux_east[ie_-1,:,:] = 0.
    pyom.flux_north[:,je_-1,:] = 0.

    """
    update tendency
    """
    pyom.du_mix[2:-2, 2:-2] += pyom.maskU[2:-2,2:-2] * ((pyom.flux_east[2:-2,2:-2] - pyom.flux_east[1:-3,2:-2]) / (pyom.cost[2:-2] * pyom.dxt[2:-2, np.newaxis])[:,:,np.newaxis] \
            + (pyom.flux_north[2:-2,2:-2] - pyom.flux_north[2:-2,1:-3]) / (pyom.cost[2:-2] * pyom.dyt[2:-2])[np.newaxis, :, np.newaxis])
    #for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
    #    for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
    #        pyom.du_mix[i,j,:] += pyom.maskU[i,j,:] * ((pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:]) / (pyom.cost[j] * pyom.dxu[i]) \
    #                                                    + (pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:]) / (pyom.cost[j] * pyom.dyt[j]))

    if pyom.enable_conserve_energy:
        """
        diagnose dissipation by lateral friction
        """
        diss[1:-2, 2:-2] = 0.5*((pyom.u[2:-1,2:-2,:,pyom.tau] - pyom.u[1:-2,2:-2,:,pyom.tau]) * pyom.flux_east[1:-2,2:-2] \
                + (pyom.u[1:-2,2:-2,:,pyom.tau] - pyom.u[:-3,2:-2,:,pyom.tau]) * pyom.flux_east[:-3,2:-2]) / (pyom.cost[2:-2] * pyom.dxu[1:-2,np.newaxis])[:,:,np.newaxis]\
                + 0.5*((pyom.u[1:-2,3:-1,:,pyom.tau] - pyom.u[1:-2,2:-2,:,pyom.tau]) * pyom.flux_north[1:-2,2:-2] \
                + (pyom.u[1:-2,2:-2,:,pyom.tau] - pyom.u[1:-2,1:-3,:,pyom.tau]) * pyom.flux_north[1:-2,1:-3]) / (pyom.cost[2:-2] * pyom.dyt[2:-2])[np.newaxis,:,np.newaxis]
        #for k in xrange(pyom.nz): # k = 1,nz
        #    for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
        #        for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
        #            diss[i,j,k] = 0.5*((pyom.u[i+1,j,k,pyom.tau] - pyom.u[i,j,k,pyom.tau]) * pyom.flux_east[i,j,k] \
        #                            +(pyom.u[i,j,k,pyom.tau] - pyom.u[i-1,j,k,pyom.tau]) * pyom.flux_east[i-1,j,k]) / (pyom.cost[j]*pyom.dxu[i])  \
        #                          +0.5*((pyom.u[i,j+1,k,pyom.tau] - pyom.u[i,j,k,pyom.tau]) * pyom.flux_north[i,j,k] \
        #                            +(pyom.u[i,j,k,pyom.tau] - pyom.u[i,j-1,k,pyom.tau]) * pyom.flux_north[i,j-1,k]) / (pyom.cost[j] * pyom.dyt[j])
        pyom.K_diss_h[...] = numerics.calc_diss(diss,pyom.K_diss_h,'U',pyom)

    """
    Meridional velocity
    """
    if pyom.enable_hor_friction_cos_scaling:
        fxa = (pyom.cosu ** pyom.hor_friction_cosPower) * np.ones(pyom.nx+3)[:,np.newaxis]
        pyom.flux_east[:-1] = pyom.A_h * fxa[:, :, np.newaxis] * (pyom.v[1:,:,:,pyom.tau] - pyom.v[:-1,:,:,pyom.tau]) \
                / (pyom.cosu * pyom.dxu[:-1, np.newaxis])[:,:,np.newaxis] * pyom.maskV[1:] * pyom.maskV[:-1]
        #for j in xrange(js_,je_): # j = js,je
        #    fxa = pyom.cosu[j]**pyom.hor_friction_cosPower
        #    for i in xrange(is_,ie_-1): # i = is,ie-1
        #        pyom.flux_east[i,j,:] = fxa * pyom.A_h * (pyom.v[i+1,j,:,pyom.tau] - pyom.v[i,j,:,pyom.tau]) \
        #                                 / (pyom.cosu[j]*pyom.dxu[i]) * pyom.maskV[i+1,j,:] * pyom.maskV[i,j,:]
        fxa = (pyom.cost[1:] ** pyom.hor_friction_cosPower) * np.ones(pyom.nx+4)[:, np.newaxis]
        pyom.flux_north[:,:-1] = pyom.A_h * fxa[:,:,np.newaxis] * (pyom.v[:,1:,:,pyom.tau] - pyom.v[:,:-1,:,pyom.tau]) \
                / pyom.dyt[np.newaxis,1:,np.newaxis] * pyom.cost[np.newaxis,1:,np.newaxis] * pyom.maskV[:,:-1] * pyom.maskV[:,1:]
        #for j in xrange(js_,je_-1): # j = js,je-1
        #    fxa = pyom.cost[j+1]**pyom.hor_friction_cosPower
        #    pyom.flux_north[:,j,:] = fxa * pyom.A_h * (pyom.v[:,j+1,:,pyom.tau] - pyom.v[:,j,:,pyom.tau]) \
        #                              / pyom.dyt[j+1] * pyom.cost[j+1] * pyom.maskV[:,j,:] * pyom.maskV[:,j+1,:]
    else:
        pyom.flux_east[:-1] = pyom.A_h * (pyom.v[1:,:,:,pyom.tau] - pyom.v[:-1,:,:,pyom.tau]) \
                / (pyom.cosu * pyom.dxu[:-1, np.newaxis])[:,:,np.newaxis] * pyom.maskV[1:] * pyom.maskV[:-1]
        #for j in xrange(js_,je_): # j = js,je
        #    for i in xrange(is_,ie_-1): # i = is,ie-1
        #        pyom.flux_east[i,j,:] = pyom.A_h * (pyom.v[i+1,j,:,pyom.tau] - pyom.v[i,j,:,pyom.tau]) \
        #                                 / (pyom.cosu[j] * pyom.dxu[i]) * pyom.maskV[i+1,j,:] * pyom.maskV[i,j,:]
        pyom.flux_north[:,:-1] = pyom.A_h * (pyom.v[:,1:,:,pyom.tau] - pyom.v[:,:-1,:,pyom.tau]) \
                / pyom.dyt[np.newaxis,1:,np.newaxis] * pyom.cost[np.newaxis,1:,np.newaxis] * pyom.maskV[:,:-1] * pyom.maskV[:,1:]
        #for j in xrange(js_,je_-1): # j = js,je-1
        #    pyom.flux_north[:,j,:] = pyom.A_h * (pyom.v[:,j+1,:,pyom.tau] - pyom.v[:,j,:,pyom.tau]) \
        #                              / pyom.dyt[j+1] * pyom.cost[j+1] * pyom.maskV[:,j,:] * pyom.maskV[:,j+1,:]
    pyom.flux_east[ie_-1,:,:] = 0.
    pyom.flux_north[:,je_-1,:] = 0.

    """
    update tendency
    """
    pyom.dv_mix[2:-2,2:-2] += pyom.maskV[2:-2,2:-2] * ((pyom.flux_east[2:-2,2:-2] - pyom.flux_east[1:-3,2:-2]) / (pyom.cosu[2:-2] * pyom.dxt[2:-2,np.newaxis])[:,:,np.newaxis] \
            + (pyom.flux_north[2:-2,2:-2] - pyom.flux_north[2:-2,1:-3]) / (pyom.dyu[2:-2] * pyom.cosu[2:-2])[np.newaxis,:,np.newaxis])
    #for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
    #    for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
    #        pyom.dv_mix[i,j,:] += pyom.maskV[i,j,:] * ((pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:]) / (pyom.cosu[j] * pyom.dxt[i]) \
    #                                                  +(pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:]) / (pyom.dyu[j] * pyom.cosu[j]))

    if pyom.enable_conserve_energy:
        """
        diagnose dissipation by lateral friction
        """
        diss[2:-2,1:-2] = 0.5 * ((pyom.v[3:-1,1:-2,:,pyom.tau] - pyom.v[2:-2,1:-2,:,pyom.tau]) * pyom.flux_east[2:-2,1:-2]\
                + (pyom.v[2:-2,1:-2,:,pyom.tau] - pyom.v[1:-3,1:-2,:,pyom.tau]) * pyom.flux_east[1:-3,1:-2]) \
                / (pyom.cosu[1:-2] * pyom.dxt[2:-2,np.newaxis])[:,:,np.newaxis] \
                + 0.5*((pyom.v[2:-2,2:-1,:,pyom.tau] - pyom.v[2:-2,1:-2,:,pyom.tau]) * pyom.flux_north[2:-2,1:-2] \
                + (pyom.v[2:-2,1:-2,:,pyom.tau] - pyom.v[2:-2,:-3,:,pyom.tau]) * pyom.flux_north[2:-2,:-3]) \
                / (pyom.cosu[1:-2] * pyom.dyu[1:-2])[np.newaxis,:,np.newaxis]
        #for k in xrange(pyom.nz): # k = 1,nz
        #    for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
        #        for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
        #            diss[i,j,k] = 0.5*((pyom.v[i+1,j,k,pyom.tau] - pyom.v[i,j,k,pyom.tau]) * pyom.flux_east[i,j,k]+ \
        #                                (pyom.v[i,j,k,pyom.tau] - pyom.v[i-1,j,k,pyom.tau]) * pyom.flux_east[i-1,j,k]) \
        #                              / (pyom.cosu[j] * pyom.dxt[i]) \
        #                        + 0.5*((pyom.v[i,j+1,k,pyom.tau] - pyom.v[i,j,k,pyom.tau]) * pyom.flux_north[i,j,k]+ \
        #                                (pyom.v[i,j,k,pyom.tau] - pyom.v[i,j-1,k,pyom.tau]) * pyom.flux_north[i,j-1,k]) \
        #                              / (pyom.cosu[j] * pyom.dyu[j])
        pyom.K_diss_h[...] = numerics.calc_diss(diss,pyom.K_diss_h,'V',pyom)

    if not pyom.enable_hydrostatic:
        if pyom.enable_hor_friction_cos_scaling:
            raise NotImplementedError("scaling of lateral friction for vertical velocity not implemented")

        pyom.flux_east[:-1] = pyom.A_h * (pyom.w[1:,:,:,pyom.tau] - pyom.w[:-1,:,:,pyom.tau]) \
                / (pyom.cost * pyom.dxu[:,np.newaxis])[:,:,np.newaxis] * pyom.maskW[1:] * pyom.maskW[:-1]
        #for j in xrange(js_,je_): # j = js,je
        #    for i in xrange(is_,ie_-1): # i = is,ie-1
        #        pyom.flux_east[i,j,:] = pyom.A_h * (pyom.w[i+1,j,:,pyom.tau] - pyom.w[i,j,:,pyom.tau]) \
        #                                 / (pyom.cost[j] * pyom.dxu[i]) * pyom.maskW[i+1,j,:] * pyom.maskW[i,j,:]
        pyom.flux_north[:,:-1] = pyom.A_h * (pyom.w[:,1:,:,pyom.tau] - pyom.w[:,:-1,:,pyom.tau]) \
                / pyom.dyu[np.newaxis,:-1,np.newaxis] * pyom.maskW[:,1:] * pyom.maskW[:,:-1] * pyom.cosu[np.newaxis,:-1,np.newaxis]
        #for j in xrange(js_,je_-1): # j = js,je-1
        #    pyom.flux_north[:,j,:] = pyom.A_h * (pyom.w[:,j+1,:,pyom.tau] - pyom.w[:,j,:,pyom.tau]) \
        #                              / pyom.dyu[j] * pyom.maskW[:,j+1,:] * pyom.maskW[:,j,:] * pyom.cosu[j]
        pyom.flux_east[ie_-1,:,:] = 0.
        pyom.flux_north[:,je_-1,:] = 0.

        """
        update tendency
        """
        pyom.dw_mix[2:-2,2:-2] += pyom.maskW[2:-2,2:-2]*((pyom.flux_east[2:-2,2:-2] - pyom.flux_east[1:-3,2:-2]) \
                / (pyom.cost[2:-2] * pyom.dxt[2:-2,np.newaxis])[:,:,np.newaxis] \
                + (pyom.flux_north[2:-2,2:-2] - pyom.flux_north[2:-2,1:-3]) \
                / (pyom.dyt[2:-2] * pyom.cost[2:-2])[np.newaxis,:,np.newaxis])
        #for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
        #    for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
        #        pyom.dw_mix[i,j,:] += pyom.maskW[i,j,:]*((pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:]) \
        #                                                    / (pyom.cost[j] * pyom.dxt[i]) \
        #                                                +(pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:]) \
        #                                                    / (pyom.dyt[j] * pyom.cost[j]))

        """
        diagnose dissipation by lateral friction
        """
        # to be implemented


def biharmonic_friction(pyom):
    """
    horizontal biharmonic friction
    dissipation is calculated and added to K_diss_h
    """
    # real*8 :: del2(pyom.is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz),fxa
    # real*8 :: diss(pyom.is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    del2 = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))
    diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    if not pyom.enable_hydrostatic:
        raise NotImplementedError("biharmonic mixing for non-hydrostatic not yet implemented")

    is_ = pyom.is_pe - pyom.onx
    ie_ = pyom.ie_pe + pyom.onx
    js_ = pyom.js_pe - pyom.onx
    je_ = pyom.je_pe + pyom.onx
    fxa = math.sqrt(abs(pyom.A_hbi))

    """
    Zonal velocity
    """
    for j in xrange(js_,je_): # j = js,je
        for i in xrange(is_,ie_-1): # i = is,ie-1
            pyom.flux_east[i,j,:] = fxa * (pyom.u[i+1,j,:,pyom.tau] - pyom.u[i,j,:,pyom.tau]) \
                                    / (pyom.cost[j] * pyom.dxt[i+1]) * pyom.maskU[i+1,j,:] * pyom.maskU[i,j,:]
    for j in xrange(js_,je_-1): # j = js,je-1
        pyom.flux_north[:,j,:] = fxa * (pyom.u[:,j+1,:,pyom.tau] - pyom.u[:,j,:,pyom.tau]) \
                                 / pyom.dyu[j] * pyom.maskU[:,j+1,:] * pyom.maskU[:,j,:] * pyom.cosu[j]
    pyom.flux_east[ie_-1,:,:] = 0.
    pyom.flux_north[:,je_-1,:] = 0.

    for j in xrange(js_+1,je_): # j = js+1,je
        for i in xrange(is_+1,ie_): # i = is+1,ie
            del2[i,j,:] = (pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:]) / (pyom.cost[j] * pyom.dxu[i]) \
                        + (pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:]) / (pyom.cost[j] * pyom.dyt[j])

    for j in xrange(js_,je_): # j = js,je
        for i in xrange(is_,ie_-1): # i = is,ie-1
            pyom.flux_east[i,j,:] = fxa * (del2[i+1,j,:] - del2[i,j,:]) \
                                    / (pyom.cost[j] * pyom.dxt[i+1]) * pyom.maskU[i+1,j,:] * pyom.maskU[i,j,:]
    for j in xrange(js_,je_-1): # j = js,je-1
        pyom.flux_north[:,j,:] = fxa * (del2[:,j+1,:] - del2[:,j,:]) \
                                 / pyom.dyu[j] * pyom.maskU[:,j+1,:] * pyom.maskU[:,j,:] * pyom.cosu[j]
    pyom.flux_east[ie_-1,:,:] = 0.
    pyom.flux_north[:,je_-1,:] = 0.

    """
    update tendency
    """
    for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
        for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
            du_mix[i,j,:] -= pyom.maskU[i,j,:] * ((pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:]) \
                                                    / (pyom.cost[j] * pyom.dxu[i]) \
                                                 +(pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:]) \
                                                    / (pyom.cost[j] * pyom.dyt[j]))
    if pyom.enable_conserve_energy:
        """
        diagnose dissipation by lateral friction
        """
        if pyom.enable_cyclic_x:
            cyclic.setcyclic_x(pyom.flux_east)
            cyclic.setcyclic_x(pyom.flux_north)
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                diss[i,j,:] = -0.5*((pyom.u[i+1,j,:,pyom.tau] - pyom.u[i,j,:,pyom.tau]) * pyom.flux_east[i,j,:] \
                                   +(pyom.u[i,j,:,pyom.tau] - pyom.u[i-1,j,:,pyom.tau]) * pyom.flux_east[i-1,j,:]) \
                                  / (pyom.cost[j] * pyom.dxu[i])  \
                              -0.5*((pyom.u[i,j+1,:,pyom.tau] - pyom.u[i,j,:,pyom.tau]) * pyom.flux_north[i,j,:] \
                                   +(pyom.u[i,j,:,pyom.tau] - pyom.u[i,j-1,:,pyom.tau]) * pyom.flux_north[i,j-1,:]) \
                                  / (pyom.cost[j] * pyom.dyt[j])
        pyom.K_diss_h[...] = numerics.calc_diss(diss,pyom.K_diss_h,'U',pyom)

    """
    Meridional velocity
    """
    for j in xrange(js_,je_): # j = js,je
        for i in xrange(is_,ie_-1): # i = is,ie-1
            pyom.flux_east[i,j,:] = fxa * (pyom.v[i+1,j,:,pyom.tau] - pyom.v[i,j,:,pyom.tau]) \
                                     / (pyom.cosu[j] * pyom.dxu[i]) \
                                     * pyom.maskV[i+1,j,:] * pyom.maskV[i,j,:]
    for j in xrange(js_,je_-1): # j = js,je-1
        pyom.flux_north[:,j,:] = fxa * (pyom.v[:,j+1,:,pyom.tau] - pyom.v[:,j,:,pyom.tau]) \
                                 / pyom.dyt[j+1] * pyom.cost[j+1] * pyom.maskV[:,j,:] * pyom.maskV[:,j+1,:]
    pyom.flux_east[ie-1,:,:] = 0.
    pyom.flux_north[:,je-1,:] = 0.

    for j in xrange(js_+1,je_): # j = js+1,je
        for i in xrange(is_+1,ie_): # i = is+1,ie
            del2[i,j,:] = (pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:]) / (pyom.cosu[j] * pyom.dxt[i])  \
                         +(pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:]) / (pyom.dyu[j] * pyom.cosu[j])
    for j in xrange(js_,je_): # j = js,je
        for i in xrange(is_,ie_-1): # i = is,ie-1
            pyom.flux_east[i,j,:] = fxa * (del2[i+1,j,:] - del2[i,j,:]) \
                                    / (pyom.cosu[j] * pyom.dxu[i]) \
                                    * pyom.maskV[i+1,j,:] * pyom.maskV[i,j,:]
    for j in xrange(js_,je_-1): # j = js,je-1
        pyom.flux_north[:,j,:] = fxa * (del2[:,j+1,:] - del2[:,j,:]) \
                                 / pyom.dyt[j+1] * pyom.cost[j+1] * pyom.maskV[:,j,:] * pyom.maskV[:,j+1,:]
    pyom.flux_east[ie_-1,:,:] = 0.
    pyom.flux_north[:,je_-1,:] = 0.

    """
    update tendency
    """
    for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
        for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
            pyom.dv_mix[i,j,:] -= pyom.maskV[i,j,:] * ((pyom.flux_east[i,j,:] - pyom.flux_east[i-1,j,:]) \
                                                        / (pyom.cosu[j] * pyom.dxt[i]) \
                                                     + (pyom.flux_north[i,j,:] - pyom.flux_north[i,j-1,:]) \
                                                        / (pyom.dyu[j] * pyom.cosu[j]))

    if pyom.enable_conserve_energy:
        """
        diagnose dissipation by lateral friction
        """
        if pyom.enable_cyclic_x:
            cyclic.setcyclic_x(pyom.flux_east)
            cyclic.setcyclic_x(pyom.flux_north)
        for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                diss[i,j,:] = -0.5*((pyom.v[i+1,j,:,pyom.tau] - pyom.v[i,j,:,pyom.tau]) * pyom.flux_east[i,j,:] \
                                  + (pyom.v[i,j,:,pyom.tau] - pyom.v[i-1,j,:,pyom.tau]) * pyom.flux_east[i-1,j,:]) \
                                 / (pyom.cosu[j]*pyom.dxt[i]) \
                             - 0.5*((pyom.v[i,j+1,:,pyom.tau] - pyom.v[i,j,:,pyom.tau]) * pyom.flux_north[i,j,:] \
                                  + (pyom.v[i,j,:,pyom.tau] - pyom.v[i,j-1,:,pyom.tau]) * pyom.flux_north[i,j-1,:]) \
                                 / (pyom.cosu[j] * pyom.dyu[j])
        pyom.K_diss_h[...] = numerics.calc_diss(diss,pyom.K_diss_h,'V',pyom)


def momentum_sources(pyom):
    """
    other momentum sources
    dissipation is calculated and added to K_diss_bot
    """
    # real*8 :: diss(pyom.is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    for k in xrange(pyom.nz): # k = 1,nz
        pyom.du_mix[:,:,k] += pyom.maskU[:,:,k] * pyom.u_source[:,:,k]
    if pyom.enable_conserve_energy:
        for k in xrange(pyom.nz): # k = 1,nz
            diss[:,:,k] = -pyom.maskU[:,:,k] * pyom.u[:,:,k] * pyom.u_source[:,:,k]
        pyom.K_diss_bot[...] = numerics.calc_diss(diss,pyom.K_diss_bot,'U',pyom)
    for k in xrange(pyom.nz): # k = 1,nz
        pyom.dv_mix[:,:,k] += pyom.maskV[:,:,k] * pyom.v_source[:,:,k]
    if pyom.enable_conserve_energy:
        for k in xrange(pyom.nz): # k = 1,nz
            diss[:,:,k] = -pyom.maskV[:,:,k] * pyom.v[:,:,k] * pyom.v_source[:,:,k]
        pyom.K_diss_bot[...] = numerics.calc_diss(diss,pyom.K_diss_bot,'V',pyom)

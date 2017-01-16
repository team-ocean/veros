import numpy as np

from climate.pyom import numerics


def explicit_vert_friction(pyom):
    """
    explicit vertical friction
    dissipation is calculated and added to K_diss_v
    """

    # real*8 :: diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz),fxa
    diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

     """
     vertical friction of zonal momentum
     """
     for k in xrange(pyom.nz-1): # k = 1,nz-1
        for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
            for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                fxa = 0.5 * (pyom.kappaM[i,j,k] + pyom.kappaM[i+1,j,k])
                pyom.flux_top[i,j,k] = fxa * (pyom.u[i,j,k+1] - pyom.u[i,j,k])/pyom.dzw(k)*pyom.maskU[i,j,k+1]*pyom.maskU[i,j,k]
    flux_top[]:,:,pyom.nz-1] = 0.0
    k = 0
    pyom.du_mix[:,:,k] = pyom.flux_top[:,:,k]/pyom.dzt[k]*pyom.maskU[:,:,k]
    for k in xrange(1,pyom.nz): # k = 2,nz
        pyom.du_mix[:,:,k] = (pyom.flux_top[:,:,k] - pyom.flux_top[:,:,k-1])/pyom.dzt[k]*pyom.maskU[:,:,k]

     """
     diagnose dissipation by vertical friction of zonal momentum
     """
     for k in xrange(pyom.nz-1): # k = 1,nz-1
        for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
            for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                diss[i,j,k] = (pyom.u[i,j,k+1] - pyom.u[i,j,k]) * pyom.flux_top[i,j,k]/pyom.dzw[k]
    diss[:,:,pyom.nz-1] = 0.0
    ugrid_to_tgrid(diss,diss)
    pyom.K_diss_v += diss

     """
     vertical friction of meridional momentum
     """
     for k in xrange(pyom.nz-1): # k = 1,nz-1
        for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
            for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                fxa = 0.5 * (pyom.kappaM[i,j,k]+pyom.kappaM[i,j+1,k])
                pyom.flux_top[i,j,k] = fxa * (pyom.v[i,j,k+1] - pyom.v[i,j,k])/pyom.dzw[k]*pyom.maskV[i,j,k+1]*pyom.maskV[i,j,k]
                flux_top[:,:,pyom.nz-1] = 0.0
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
                diss[i,j,k] = (pyom.v[i,j,k+1] - pyom.v[i,j,k]) * pyom.flux_top[i,j,k] / pyom.dzw(k)
    diss[:,:,pyom.nz-1] = 0.0
    vgrid_to_tgrid(diss)
    pyom.K_diss_v += diss

    if not pyom.enable_hydrostatic:
        """
        vertical friction of vertical momentum
        """
        for k in xrange(pyom.nz-1): # k = 1,nz-1
            for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
                for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                    fxa = 0.5 * (pyom.kappaM[i,j,k] + pyom.kappaM[i,j,k+1])
                    pyom.flux_top[i,j,k] = fxa * (pyom.w[i,j,k+1] - pyom.w[i,j,k]) / pyom.dzt[k+1] * pyom.maskW[i,j,k+1] * pyom.maskW[i,j,k]
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
    # real*8 :: diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    a_tri = np.zeros(pyom.nz)
    b_tri = np.zeros(pyom.nz)
    c_tri = np.zeros(pyom.nz)
    d_tri = np.zeros(pyom.nz)
    delta = np.zeros(pyom.nz)
    diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    """
    implicit vertical friction of zonal momentum
    """
    for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
        for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
            ks = max(pyom.kbot[i,j],pyom.kbot[i+1,j]) - 1
            if ks >= 0:
                for k in xrange(ks,pyom.nz-1): # k = ks,nz-1
                    fxa = 0.5 * (pyom.kappaM[i,j,k] + pyom.kappaM[i+1,j,k])
                    delta[k] = pyom.dt_mom / pyom.dzw[k] * fxa * pyom.maskU[i,j,k+1] * pyom.maskU[i,j,k]
                delta[pyom.nz-1] = 0.0
                a_tri[ks] = 0.0
                for k in xrange(ks+1,pyom.nz): # k = ks+1,nz
                    a_tri[k] = -delta[k-1] / pyom.dzt[k]
                b_tri[ks] = 1 + delta[ks] / pyom.dzt[ks]
                for k in xrange(ks+1,pyom.nz-1): # k = ks+1,nz-1
                    b_tri[k] = 1 + delta[k] / pyom.dzt[k] + delta[k-1] / pyom.dzt[k]
                b_tri[pyom.nz-1] = 1 + delta[pyom.nz-2] / pyom.dzt[pyom.nz-1]
                for k in xrange(ks,pyom.nz-1): # k = ks,nz-1
                    c_tri[k] = -delta[k] / pyom.dzt[k]
                c_tri[pyom.nz-1] = 0.0
                d_tri[ks:] = pyom.u[i,j,ks:,pyom.tau]
                numerics.solve_tridiag(a_tri[ks:],b_tri[ks:],c_tri[ks:],d_tri[ks:],pyom.u[i,j,ks:,pyom.taup1],pyom.nz-ks)
            pyom.du_mix[i,j,:] = (pyom.u[i,j,:,pyom.taup1] - pyom.u[i,j,:]) / pyom.dt_mom

    """
    diagnose dissipation by vertical friction of zonal momentum
    """
    for k in xrange(pyom.nz-1): # k = 1,nz-1
        for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
            for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                fxa = 0.5 * (pyom.kappaM[i,j,k]+pyom.kappaM[i+1,j,k])
                pyom.flux_top[i,j,k] = fxa * (pyom.u[i,j,k+1,pyom.taup1] - pyom.u[i,j,k,pyom.taup1]) \
                                       / pyom.dzw[k] * pyom.maskU[i,j,k+1] * pyom.maskU[i,j,k]
    for k in xrange(pyom.nz-1): # k = 1,nz-1
        for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
            for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                diss[i,j,k] = (pyom.u[i,j,k+1] - pyom.u[i,j,k]) * pyom.flux_top[i,j,k] / pyom.dzw[k]
    diss[:,:,pyom.nz-1] = 0.0
    ugrid_to_tgrid(diss)
    K_diss_v += diss

    """
    implicit vertical friction of meridional momentum
    """
    for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
        for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
            ks = max(pyom.kbot[i,j],pyom.kbot[i,j+1]) - 1
            if ks >= 0:
                for k in xrange(ks,pyom.nz-1): # k = ks,nz-1
                    fxa = 0.5 * (pyom.kappaM[i,j,k] + pyom.kappaM[i,j+1,k])
                    delta[k] = pyom.dt_mom / pyom.dzw[k] * fxa * pyom.maskV[i,j,k+1] * pyom.maskV[i,j,k]
                delta[pyom.nz-1] = 0.0
                a_tri[ks] = 0.0
                for k in xrange(ks+1,pyom.nz): # k = ks+1,nz
                    a_tri[k] = -delta[k-1] / pyom.dzt[k]
                b_tri[ks] = 1 + delta[ks] / pyom.dzt[ks]
                for k in xrange(ks+1,pyom.nz-1): # k = ks+1,nz-1
                    b_tri[k] = 1 + delta[k] / pyom.dzt[k] + delta[k-1] / pyom.dzt[k]
                b_tri[pyom.nz-1] = 1 + delta[pyom.nz-2] / pyom.dzt[pyom.nz-1]
                for k in xrange(ks,pyom.nz-1): # k = ks,nz-1
                    c_tri[k] = -delta[k] / pyom.dzt[k]
                c_tri[pyom.nz-1] = 0.0
                d_tri[ks:] = pyom.v[i,j,ks:,pyom.tau]
                numerics.solve_tridiag(a_tri[ks:],b_tri[ks:],c_tri[ks:],d_tri[ks:],pyom.v[i,j,ks:,pyom.taup1],pyom.nz-ks)
            pyom.dv_mix[i,j,:] = (pyom.v[i,j,:,pyom.taup1] - pyom.v[i,j,:]) / pyom.dt_mom

    """
    diagnose dissipation by vertical friction of meridional momentum
    """
    for k in xrange(pyom.nz-1): # k = 1,nz-1
        for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
            for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                fxa = 0.5 * (pyom.kappaM[i,j,k] + pyom.kappaM[i,j+1,k])
                pyom.flux_top[i,j,k] = fxa * (pyom.v[i,j,k+1,pyom.taup1] - pyom.v[i,j,k,pyom.taup1]) \
                                        / pyom.dzw[k] * pyom.maskV[i,j,k+1] * pyom.maskV[i,j,k]
    for k in xrange(pyom.nz-1): # k = 1,nz-1
        for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
            for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                diss[i,j,k] = (pyom.v[i,j,k+1] - pyom.v[i,j,k]) * pyom.flux_top[i,j,k] / pyom.dzw(k)
    diss[:,:,pyom.nz-1] = 0.0
    vgrid_to_tgrid(diss)
    pyom.K_diss_v += diss

    if not pyom.enable_hydrostatic:
        # !if (my_pe==0) print'(/a/)','ERROR: implicit vertical friction for vertical velocity not implemented'
        # !halt_stop(' in implicit_vert_friction')
        for j in xrange(pyom.js_pe,pyom.je_pe): # j = js_pe,je_pe
            for i in xrange(pyom.is_pe,pyom.ie_pe): # i = is_pe,ie_pe
                ks = pyom.kbot[i,j] - 1
                if ks >= 0:
                    for k in xrange(ks,pyom.nz-1): # k = ks,nz-1
                        delta[k] = pyom.dt_mom / pyom.dzt[k+1] * 0.5 * (pyom.kappaM[i,j,k] + pyom.kappaM[i,j,k+1])
                    delta[pyom.nz-1] = 0.0
                    for k in xrange(ks+1,pyom.nz-1): # k = ks+1,nz-1
                        a_tri[k] = -delta[k-1] / pyom.dzw[k]
                    a_tri[ks] = 0.0
                    a_tri[pyom.nz-1] = -delta[pyom.nz-2]/(0.5*pyom.dzw[pyom.nz-1])
                    for k in xrange(ks+1,pyom.nz-1): # k = ks+1,nz-1
                        b_tri[k] = 1 + delta[k] / pyom.dzw[k] + delta[k-1] / pyom.dzw[k]
                    b_tri[pyom.nz-1] = 1 + delta[pyom.nz-2]/(0.5*pyom.dzw[nz-1])
                    b_tri[ks] = 1 + delta[ks] / pyom.dzw[ks]
                    for k in xrange(ks,pyom.nz-1): # k = ks,nz-1
                        c_tri[k] = - delta[k] / pyom.dzw[k]
                    c_tri[pyom.nz-1] = 0.0
                    d_tri[ks:] = pyom.w[i,j,ks:,pyom.tau]
                    numerics.solve_tridiag(a_tri[ks:],b_tri[ks:],c_tri[ks:],d_tri[ks:],pyom.w[i,j,ks:,pyom.taup1],pyom.nz-ks)
                pyom.dw_mix[i,j,:] = (pyom.w(i,j,:,pyom.taup1) - pyom.w[i,j,:]) / pyom.dt_mom

        """
        diagnose dissipation by vertical friction of vertical momentum
        """
        for k in xrange(pyom.nz-1): # k = 1,nz-1
            for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
                for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                    fxa = 0.5 * (pyom.kappaM[i,j,k] + pyom.kappaM[i,j,k+1])
                    pyom.flux_top[i,j,k] = fxa * (pyom.w[i,j,k+1,pyom.taup1] - pyom.w[i,j,k,pyom.taup1]) \
                                            / pyom.dzt[k+1] * pyom.maskW[i,j,k+1] * pyom.maskW[i,j,k]
        for k in xrange(pyom.nz-1): # k = 1,nz-1
            for j in xrange(pyom.js_pe-1,pyom.je_pe): # j = js_pe-1,je_pe
                for i in xrange(pyom.is_pe-1,pyom.ie_pe): # i = is_pe-1,ie_pe
                    diss[i,j,k] = (pyom.w[i,j,k+1] - pyom.w[i,j,k]) * pyom.flux_top[i,j,k] / pyom.dzt[k+1]
        diss[:,:,pyom.nz-1] = 0.0
        K_diss_v += diss


def rayleigh_friction(pyom):
    """
    interior Rayleigh friction
    dissipation is calculated and added to K_diss_bot
    """
    # real*8 :: diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    diss = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))

    for k in xrange(pyom.nz): # k = 1,nz
        pyom.du_mix[:,:,k] = pyom.du_mix[:,:,k] - pyom.maskU[:,:,k] * pyom.r_ray * pyom.u[:,:,k]
    if pyom.enable_conserve_energy:
        for k in xrange(pyom.nz): # k = 1,nz
            diss[:,:,k] = pyom.maskU[:,:,k] * pyom.r_ray * pyom.u[:,:,k]**2
        calc_diss(diss,K_diss_bot,'U')
    for k in xrange(pyom.nz): # k = 1,nz
        pyom.dv_mix[:,:,k] = pyom.dv_mix[:,:,k] - pyom.maskV[:,:,k] * pyom.r_ray * pyom.v[:,:,k]
    if pyom.enable_conserve_energy:
        for k in xrange(pyom.nz): # k = 1,nz
            diss[:,:,k] = pyom.maskV[:,:,k] * pyom.r_ray * pyom.v[:,:,k]**2
        calc_diss(diss,K_diss_bot,'V')
    if not pyom.enable_hydrostatic:
        raise NotImplementedError("Rayleigh friction for vertical velocity not implemented")


def linear_bottom_friction(pyom):
"""
!   linear bottom friction
!   dissipation is calculated and added to K_diss_bot
"""
 use main_module
 implicit none
 integer :: i,j,k
 real*8 :: diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)

 if enable_bottom_friction_var:

 """
 ! with spatially varying coefficient
 """
  for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
   for i in xrange(is_pe-1,ie_pe): # i = is_pe-1,ie_pe
    k = max(kbot(i,j),kbot(i+1,j))
    if (k>0) du_mix[i,j,k] = du_mix[i,j,k] - maskU[i,j,k]*r_bot_var_u(i,j)*u[i,j,k]
   enddo
  enddo
  if enable_conserve_energy:
   diss = 0.0
   for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
    for i in xrange(is_pe-1,ie_pe): # i = is_pe-1,ie_pe
     k = max(kbot(i,j),kbot(i+1,j))
     if (k>0) diss[i,j,k] = maskU[i,j,k]*r_bot_var_u(i,j)*u[i,j,k]**2
    enddo
   enddo
   calc_diss(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,K_diss_bot,'U')
  endif

  for j in xrange(js_pe-1,je_pe): # j = js_pe-1,je_pe
   for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
    k = max(kbot(i,j+1),kbot(i,j))
    if (k>0) dv_mix[i,j,k] = dv_mix[i,j,k] - maskV[i,j,k]*r_bot_var_v(i,j)*v[i,j,k]
   enddo
  enddo
  if enable_conserve_energy:
   diss = 0.0
   for j in xrange(js_pe-1,je_pe): # j = js_pe-1,je_pe
    for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
     k = max(kbot(i,j+1),kbot(i,j))
     if (k>0) diss[i,j,k] = maskV[i,j,k]*r_bot_var_v(i,j)*v[i,j,k]**2
    enddo
   enddo
   calc_diss(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,K_diss_bot,'V')
  endif

 else
 """
 ! with constant coefficient
 """
  for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
   for i in xrange(is_pe-1,ie_pe): # i = is_pe-1,ie_pe
    k = max(kbot(i,j),kbot(i+1,j))
    if (k>0) du_mix[i,j,k] = du_mix[i,j,k] - maskU[i,j,k]*r_bot*u[i,j,k]
   enddo
  enddo
  if enable_conserve_energy:
   diss = 0.0
   for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
    for i in xrange(is_pe-1,ie_pe): # i = is_pe-1,ie_pe
     k = max(kbot(i,j),kbot(i+1,j))
     if (k>0) diss[i,j,k] = maskU[i,j,k]*r_bot*u[i,j,k]**2
    enddo
   enddo
   calc_diss(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,K_diss_bot,'U')
  endif

  for j in xrange(js_pe-1,je_pe): # j = js_pe-1,je_pe
   for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
    k = max(kbot(i,j+1),kbot(i,j))
    if (k>0) dv_mix[i,j,k] = dv_mix[i,j,k] - maskV[i,j,k]*r_bot*v[i,j,k]
   enddo
  enddo
  if enable_conserve_energy:
   diss = 0.0
   for j in xrange(js_pe-1,je_pe): # j = js_pe-1,je_pe
    for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
     k = max(kbot(i,j+1),kbot(i,j))
     if (k>0) diss[i,j,k] = maskV[i,j,k]*r_bot*v[i,j,k]**2
    enddo
   enddo
   calc_diss(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,K_diss_bot,'V')
  endif
 endif

 if .not.enable_hydrostatic:
  if (my_pe==0) print'(/a/)','ERROR: bottom friction for vertical velocity not implemented'
  halt_stop(' in bottom_friction')
 endif
end def linear_bottom_friction(pyom):


def quadratic_bottom_friction(pyom):
"""
! quadratic bottom friction
! dissipation is calculated and added to K_diss_bot
"""
 use main_module
 implicit none
 integer :: i,j,k
 real*8 :: diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz),fxa
 real*8 :: aloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)


 ! we might want to account for EKE in the drag, also a tidal residual
 aloc = 0.0
 for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
   for i in xrange(is_pe-1,ie_pe): # i = is_pe-1,ie_pe
    k = max(kbot(i,j),kbot(i+1,j))
    if k>0:
      fxa = maskV[i,j,k]*v[i,j,k]**2 + maskV[i,j-1,k]*v[i,j-1,k]**2
      fxa = fxa + maskV[i+1,j,k]*v[i+1,j,k]**2 + maskV[i+1,j-1,k]*v[i+1,j-1,k]**2
      fxa = sqrt(u[i,j,k]**2+ 0.25*fxa)
      aloc(i,j) = maskU[i,j,k]*r_quad_bot*u[i,j,k]*fxa/dzt(k)
      du_mix[i,j,k] = du_mix[i,j,k] - aloc(i,j)
    endif
   enddo
 enddo

 if enable_conserve_energy:
   diss = 0.0
   for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
    for i in xrange(is_pe-1,ie_pe): # i = is_pe-1,ie_pe
     k = max(kbot(i,j),kbot(i+1,j))
     if (k>0) diss[i,j,k] = aloc(i,j)*u[i,j,k]
    enddo
   enddo
   calc_diss(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,K_diss_bot,'U')
 endif

 aloc = 0.0
 for j in xrange(js_pe-1,je_pe): # j = js_pe-1,je_pe
   for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
    k = max(kbot(i,j+1),kbot(i,j))
    if k>0:
      fxa = maskU[i,j,k]*u[i,j,k]**2 + maskU[i-1,j,k]*u[i-1,j,k]**2
      fxa = fxa + maskU[i,j+1,k]*u[i,j+1,k]**2 + maskU[i-1,j+1,k]*u[i-1,j+1,k]**2
      fxa = sqrt(v[i,j,k]**2+ 0.25*fxa)
      aloc(i,j) = maskV[i,j,k]*r_quad_bot*v[i,j,k]*fxa/dzt(k)
      dv_mix[i,j,k] = dv_mix[i,j,k] - aloc(i,j)
    endif
   enddo
 enddo

 if enable_conserve_energy:
   diss = 0.0
   for j in xrange(js_pe-1,je_pe): # j = js_pe-1,je_pe
    for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
     k = max(kbot(i,j+1),kbot(i,j))
     if (k>0) diss[i,j,k] = aloc(i,j)*v[i,j,k]
    enddo
   enddo
   calc_diss(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,K_diss_bot,'V')
 endif

 if .not.enable_hydrostatic:
  if (my_pe==0) print'(/a/)','ERROR: bottom friction for vertical velocity not implemented'
  halt_stop(' in quadratic_bottom_friction')
 endif
end def quadratic_bottom_friction(pyom):





def harmonic_friction(pyom):
"""
! horizontal harmonic friction
! dissipation is calculated and added to K_diss_h
"""
 use main_module
 implicit none
 integer :: i,j,k
 integer :: is,ie,js,je
 real*8 :: diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz),fxa

 is = is_pe-onx; ie = ie_pe+onx; js = js_pe-onx; je = je_pe+onx

 """
 ! Zonal velocity
 """
 if enable_hor_friction_cos_scaling:
  for j in xrange(js,je): # j = js,je
   fxa = cost(j)**hor_friction_cosPower
   for i in xrange(is,ie-1): # i = is,ie-1
    flux_east[i,j,:] = fxa*A_h*(u[i+1,j,:]-u[i,j,:])/(cost(j)*dxt(i+1))*maskU[i+1,j,:]*maskU[i,j,:]
   enddo
  enddo
  for j in xrange(js,je-1): # j = js,je-1
   fxa = cosu(j)**hor_friction_cosPower
   flux_north[:,j,:] = fxa*A_h*(u[:,j+1,:]-u[:,j,:])/dyu(j)*maskU[:,j+1,:]*maskU[:,j,:]*cosu(j)
  enddo
 else
  for j in xrange(js,je): # j = js,je
   for i in xrange(is,ie-1): # i = is,ie-1
    flux_east[i,j,:] = A_h*(u[i+1,j,:]-u[i,j,:])/(cost(j)*dxt(i+1))*maskU[i+1,j,:]*maskU[i,j,:]
   enddo
  enddo
  for j in xrange(js,je-1): # j = js,je-1
    flux_north[:,j,:] = A_h*(u[:,j+1,:]-u[:,j,:])/dyu(j)*maskU[:,j+1,:]*maskU[:,j,:]*cosu(j)
  enddo
 endif
 flux_east(ie,:,:) = 0.
 flux_north(:,je,:) = 0.

 """
 ! update tendency
 """
 for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
   for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
    du_mix[i,j,:] = du_mix[i,j,:] + maskU[i,j,:]*((flux_east[i,j,:] - flux_east[i-1,j,:])/(cost(j)*dxu(i)) \
                                                +(flux_north[i,j,:] - flux_north[i,j-1,:])/(cost(j)*dyt(j)))
  enddo
 enddo

 if enable_conserve_energy:
 """
 ! diagnose dissipation by lateral friction
 """
  for k in xrange(1,nz): # k = 1,nz
   for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
    for i in xrange(is_pe-1,ie_pe): # i = is_pe-1,ie_pe
     diss[i,j,k] = 0.5*((u[i+1,j,k]-u[i,j,k])*flux_east[i,j,k] \
                      +(u[i,j,k]-u[i-1,j,k])*flux_east[i-1,j,k])/(cost(j)*dxu(i))  \
                 +0.5*((u[i,j+1,k]-u[i,j,k])*flux_north[i,j,k]+ \
                       (u[i,j,k]-u[i,j-1,k])*flux_north[i,j-1,k])/(cost(j)*dyt(j))
    enddo
   enddo
  enddo
  K_diss_h = 0
  calc_diss(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,K_diss_h,'U')
 endif

 """
 ! Meridional velocity
 """
 if enable_hor_friction_cos_scaling:
  for j in xrange(js,je): # j = js,je
   fxa = cosu(j)**hor_friction_cosPower
   for i in xrange(is,ie-1): # i = is,ie-1
    flux_east[i,j,:] = fxa*A_h*(v[i+1,j,:]-v[i,j,:])/(cosu(j)*dxu(i)) *maskV[i+1,j,:]*maskV[i,j,:]
   enddo
  enddo
  for j in xrange(js,je-1): # j = js,je-1
   fxa = cost(j+1)**hor_friction_cosPower
   flux_north[:,j,:] = fxa*A_h*(v[:,j+1,:]-v[:,j,:])/dyt(j+1)*cost(j+1)*maskV[:,j,:]*maskV[:,j+1,:]
  enddo
 else
  for j in xrange(js,je): # j = js,je
   for i in xrange(is,ie-1): # i = is,ie-1
    flux_east[i,j,:] = A_h*(v[i+1,j,:]-v[i,j,:])/(cosu(j)*dxu(i)) *maskV[i+1,j,:]*maskV[i,j,:]
   enddo
  enddo
  for j in xrange(js,je-1): # j = js,je-1
   flux_north[:,j,:] = A_h*(v[:,j+1,:]-v[:,j,:])/dyt(j+1)*cost(j+1)*maskV[:,j,:]*maskV[:,j+1,:]
  enddo
 endif
 flux_east(ie,:,:) = 0.
 flux_north(:,je,:) = 0.

 """
 ! update tendency
 """
 for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
   for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
    dv_mix[i,j,:] = dv_mix[i,j,:] + maskV[i,j,:]*((flux_east[i,j,:] - flux_east[i-1,j,:])/(cosu(j)*dxt(i))  \
                                                 +(flux_north[i,j,:] - flux_north[i,j-1,:])/(dyu(j)*cosu(j)))
  enddo
 enddo

 if enable_conserve_energy:
 """
 ! diagnose dissipation by lateral friction
 """
  for k in xrange(1,nz): # k = 1,nz
   for j in xrange(js_pe-1,je_pe): # j = js_pe-1,je_pe
    for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
     diss[i,j,k] = 0.5*((v[i+1,j,k]-v[i,j,k])*flux_east[i,j,k]+ \
                       (v[i,j,k]-v[i-1,j,k])*flux_east[i-1,j,k])/(cosu(j)*dxt(i)) \
                + 0.5*((v[i,j+1,k]-v[i,j,k])*flux_north[i,j,k]+ \
                       (v[i,j,k]-v[i,j-1,k])*flux_north[i,j-1,k])/(cosu(j)*dyu(j))
    enddo
   enddo
  enddo
  calc_diss(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,K_diss_h,'V')
 endif

 if .not.enable_hydrostatic:

  if enable_hor_friction_cos_scaling:
   if (my_pe==0) print'(/a/)','ERROR: scaling of lateral friction for vertical velocity not implemented'
   halt_stop(' in hamronic_friction')
  endif

  for j in xrange(js,je): # j = js,je
   for i in xrange(is,ie-1): # i = is,ie-1
    flux_east[i,j,:] = A_h*(w[i+1,j,:]-w[i,j,:])/(cost(j)*dxu(i)) *maskW[i+1,j,:]*maskW[i,j,:]
   enddo
  enddo
  for j in xrange(js,je-1): # j = js,je-1
   flux_north[:,j,:] = A_h*(w[:,j+1,:]-w[:,j,:])/dyu(j)*maskW[:,j+1,:]*maskW[:,j,:]*cosu(j)
  enddo
  flux_east(ie,:,:) = 0.
  flux_north(:,je,:) = 0.

 """
 ! update tendency
 """
  for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
   for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
    dw_mix[i,j,:] = dw_mix[i,j,:] + maskW[i,j,:]*((flux_east[i,j,:] - flux_east[i-1,j,:])/(cost(j)*dxt(i))  \
                                                 +(flux_north[i,j,:] - flux_north[i,j-1,:])/(dyt(j)*cost(j)))
  enddo
  enddo

 """
 ! diagnose dissipation by lateral friction
 """
  ! to be implemented
 endif
end def harmonic_friction(pyom):






def biharmonic_friction(pyom):
"""
! horizontal biharmonic friction
! dissipation is calculated and added to K_diss_h
"""
 use main_module
 implicit none
 integer :: i,j,is,ie,js,je
 real*8 :: del2(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz),fxa
 real*8 :: diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)

 if (.not.enable_hydrostatic) halt_stop('biharmonic mixing for non-hydrostatic not yet implemented')

 is = is_pe-onx; ie = ie_pe+onx; js = js_pe-onx; je = je_pe+onx
 fxa = sqrt(abs(A_hbi))

 """
 ! Zonal velocity
 """
 for j in xrange(js,je): # j = js,je
   for i in xrange(is,ie-1): # i = is,ie-1
    flux_east[i,j,:] = fxa*(u[i+1,j,:]-u[i,j,:])/(cost(j)*dxt(i+1))*maskU[i+1,j,:]*maskU[i,j,:]
   enddo
 enddo
 for j in xrange(js,je-1): # j = js,je-1
    flux_north[:,j,:] = fxa*(u[:,j+1,:]-u[:,j,:])/dyu(j)*maskU[:,j+1,:]*maskU[:,j,:]*cosu(j)
 enddo
 flux_east(ie,:,:) = 0.
 flux_north(:,je,:) = 0.

 for j in xrange(js+1,je): # j = js+1,je
   for i in xrange(is+1,ie): # i = is+1,ie
    del2[i,j,:] = (flux_east[i,j,:] - flux_east[i-1,j,:])/(cost(j)*dxu(i)) \
                +(flux_north[i,j,:] - flux_north[i,j-1,:])/(cost(j)*dyt(j))
  enddo
 enddo

 for j in xrange(js,je): # j = js,je
   for i in xrange(is,ie-1): # i = is,ie-1
    flux_east[i,j,:] = fxa*(del2[i+1,j,:]-del2[i,j,:])/(cost(j)*dxt(i+1))*maskU[i+1,j,:]*maskU[i,j,:]
   enddo
 enddo
 for j in xrange(js,je-1): # j = js,je-1
    flux_north[:,j,:] = fxa*(del2[:,j+1,:]-del2[:,j,:])/dyu(j)*maskU[:,j+1,:]*maskU[:,j,:]*cosu(j)
 enddo
 flux_east(ie,:,:) = 0.
 flux_north(:,je,:) = 0.

 """
 ! update tendency
 """
 for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
   for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
    du_mix[i,j,:] = du_mix[i,j,:] - maskU[i,j,:]*((flux_east[i,j,:] - flux_east[i-1,j,:])/(cost(j)*dxu(i)) \
                                                +(flux_north[i,j,:] - flux_north[i,j-1,:])/(cost(j)*dyt(j)))
  enddo
 enddo

 if enable_conserve_energy:
 """
 ! diagnose dissipation by lateral friction
 """
  border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_east)
  setcyclic_xyz   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_east)
  border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_north)
  setcyclic_xyz   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_north)
  for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
    for i in xrange(is_pe-1,ie_pe): # i = is_pe-1,ie_pe
     diss[i,j,:] = -0.5*((u[i+1,j,:]-u[i,j,:])*flux_east[i,j,:] \
                       +(u[i,j,:]-u[i-1,j,:])*flux_east[i-1,j,:])/(cost(j)*dxu(i))  \
                 -0.5*((u[i,j+1,:]-u[i,j,:])*flux_north[i,j,:]+ \
                       (u[i,j,:]-u[i,j-1,:])*flux_north[i,j-1,:])/(cost(j)*dyt(j))
    enddo
  enddo
  K_diss_h = 0
  calc_diss(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,K_diss_h,'U')
 endif

 """
 ! Meridional velocity
 """
 for j in xrange(js,je): # j = js,je
   for i in xrange(is,ie-1): # i = is,ie-1
    flux_east[i,j,:] = fxa*(v[i+1,j,:]-v[i,j,:])/(cosu(j)*dxu(i)) *maskV[i+1,j,:]*maskV[i,j,:]
   enddo
 enddo
 for j in xrange(js,je-1): # j = js,je-1
   flux_north[:,j,:] = fxa*(v[:,j+1,:]-v[:,j,:])/dyt(j+1)*cost(j+1)*maskV[:,j,:]*maskV[:,j+1,:]
 enddo
 flux_east(ie,:,:) = 0.
 flux_north(:,je,:) = 0.

 for j in xrange(js+1,je): # j = js+1,je
  for i in xrange(is+1,ie): # i = is+1,ie
    del2[i,j,:] = (flux_east[i,j,:] - flux_east[i-1,j,:])/(cosu(j)*dxt(i))  \
                 +(flux_north[i,j,:] - flux_north[i,j-1,:])/(dyu(j)*cosu(j))
  enddo
 enddo

 for j in xrange(js,je): # j = js,je
   for i in xrange(is,ie-1): # i = is,ie-1
    flux_east[i,j,:] = fxa*(del2[i+1,j,:]-del2[i,j,:])/(cosu(j)*dxu(i)) *maskV[i+1,j,:]*maskV[i,j,:]
   enddo
 enddo
 for j in xrange(js,je-1): # j = js,je-1
   flux_north[:,j,:] = fxa*(del2[:,j+1,:]-del2[:,j,:])/dyt(j+1)*cost(j+1)*maskV[:,j,:]*maskV[:,j+1,:]
 enddo
 flux_east(ie,:,:) = 0.
 flux_north(:,je,:) = 0.

 """
 ! update tendency
 """
 for j in xrange(js_pe,je_pe): # j = js_pe,je_pe
   for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
    dv_mix[i,j,:] = dv_mix[i,j,:] - maskV[i,j,:]*((flux_east[i,j,:] - flux_east[i-1,j,:])/(cosu(j)*dxt(i))  \
                                                 +(flux_north[i,j,:] - flux_north[i,j-1,:])/(dyu(j)*cosu(j)))
  enddo
 enddo

 if enable_conserve_energy:
 """
 ! diagnose dissipation by lateral friction
 """
  border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_east)
  setcyclic_xyz   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_east)
  border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_north)
  setcyclic_xyz   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_north)
  for j in xrange(js_pe-1,je_pe): # j = js_pe-1,je_pe
    for i in xrange(is_pe,ie_pe): # i = is_pe,ie_pe
     diss[i,j,:] = -0.5*((v[i+1,j,:]-v[i,j,:])*flux_east[i,j,:]+ \
                       (v[i,j,:]-v[i-1,j,:])*flux_east[i-1,j,:])/(cosu(j)*dxt(i)) \
                - 0.5*((v[i,j+1,:]-v[i,j,:])*flux_north[i,j,:]+ \
                       (v[i,j,:]-v[i,j-1,:])*flux_north[i,j-1,:])/(cosu(j)*dyu(j))
    enddo
  enddo
  calc_diss(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,K_diss_h,'V')
 endif

end def biharmonic_friction(pyom):


def momentum_sources(pyom):
"""
! other momentum sources
! dissipation is calculated and added to K_diss_bot
"""
 use main_module
 implicit none
 integer :: k
 real*8 :: diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)

 for k in xrange(1,nz): # k = 1,nz
  du_mix[:,:,k] = du_mix[:,:,k] + maskU[:,:,k]*u_source[:,:,k]
 enddo
 if enable_conserve_energy:
  for k in xrange(1,nz): # k = 1,nz
   diss[:,:,k] = -maskU[:,:,k]*u[:,:,k]*u_source[:,:,k]
  enddo
  calc_diss(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,K_diss_bot,'U')
 endif

 for k in xrange(1,nz): # k = 1,nz
   dv_mix[:,:,k] = dv_mix[:,:,k] + maskV[:,:,k]*v_source[:,:,k]
 enddo
 if enable_conserve_energy:
  for k in xrange(1,nz): # k = 1,nz
   diss[:,:,k] = -maskV[:,:,k]*v[:,:,k]*v_source[:,:,k]
  enddo
  calc_diss(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,K_diss_bot,'V')
 endif
end def momentum_sources(pyom):


!=======================================================================
!  test for isopycnal mixing
!======================================================================= 


module config_module
 ! use this module only locally in this file
 implicit none
 real*8 :: y0 = -40, y1 = 28-60, y2=32-60
 real*8 :: hresolv = 1.0, vresolv = 2.0, yt_1,yt_ny,xt_1,xt_nx
end module config_module
 
 

subroutine set_parameter
 ! ----------------------------------
 !       set here main parameter
 ! ----------------------------------
 use main_module   
 use eke_module   
 use tke_module   
 use idemix_module   
 use isoneutral_module   
 use diagnostics_module   
 use config_module
 implicit none
 real*8 :: freq = 86400.*365/12.

  nx   = int(20*hresolv); nz   = int(10*vresolv); ny  = int(20*hresolv)
  !dt_mom    = 3600/2.0 /hresolv
  dt_tracer = 86400.0/4.0  /hresolv
  dt_mom    = 3600.0
  !dt_tracer = 3600.0

  coord_degree     = .true.
  enable_cyclic_x  = .true.

  runlen = 365*86400.*50

  !enable_diag_ts_monitor = .true.; ts_monint = dt_tracer !86400.*365!/12.
  enable_diag_snapshots  = .true.; snapint  =  freq
  enable_diag_energy  = .true.; energfreq =  dt_tracer*10
  energint  = freq

! enable_diag_averages   = .true.
! aveint  = 365*86400
! avefreq = dt_tracer*10

  congr_epsilon = 1e-9
  enable_streamfunction = .true.
  congr_max_iterations = 15000

  enable_hor_friction = .true.; A_h = (1.*degtom)**3*2e-11    
  enable_hor_friction_cos_scaling = .true.; hor_friction_cosPower=1
  enable_bottom_friction = .true.; r_bot = 1e-5*vresolv

  !enable_quadratic_bottom_friction = .true.; r_quad_bot = 2e-3
  !enable_store_bottom_friction_tke = .true.;
 
  enable_conserve_energy = .true.
  enable_store_cabbeling_heat = .true.

 enable_hor_diffusion = .true.; K_h = 1000.0
 !enable_biharmonic_mixing = .true.
 !K_hbi  = 1e11!/hresolv**4
 !enable_biharmonic_friction  = .true. ! for eddy resolving version
 !A_hbi  = 5e11!/hresolv**4

  enable_neutral_diffusion = .true.; 
      K_iso_0 = 1000.0
      K_iso_steep = 1000.0
      iso_dslope=4./1000.0
      iso_slopec=4./1000.0
  enable_skew_diffusion = .true.

  enable_implicit_vert_friction = .true.; 
  enable_tke = .true.
  c_k = 0.1
  c_eps = 0.7
  alpha_tke = 30.0
  mxl_min = 1d-8
  tke_mxl_choice = 2
  enable_tke_superbee_advection = .false.

  K_gm_0 = 1000
  enable_eke = .true.
  eke_k_max  = 1e4
  eke_c_k    = 0.4
  eke_c_eps  = 0.5
  eke_cross  = 2.
  eke_crhin  = 1.0
  eke_lmin   = 100.0
  enable_eke_superbee_advection  = .true.
  !enable_eke_isopycnal_diffusion = .true.
  enable_eke_diss_surfbot = .true.
  eke_diss_surfbot_frac = 0.2

  !enable_eke_isopycnal_diffusion = .true.
  !enable_eke_leewave_dissipation = .true.
  !eke_int_diss0 = 1./(10*86400.)
  !c_lee0 = 5.
  !eke_Ri0 = 300.
  !eke_Ri1 = 50.
  !alpha_eke = 20.0
  !eke_r_bot = 2e-3
  !eke_hrms_k0_min = 5e-3

  enable_idemix = .true.
  enable_idemix_hor_diffusion = .true.; 
  enable_idemix_superbee_advection = .true.

  enable_conserve_energy = .true.
  eq_of_state_type = 5 
end subroutine set_parameter



subroutine set_grid
  use main_module   
  use config_module   
  implicit none
  dxt = 1.0/hresolv
  dyt = 1.0/hresolv
  x_origin= 0.0
  y_origin= y0
  dzt = 1000.0/nz

 yt_1  = y0-dyt(js_pe)/2.0
 yt_ny = yt_1 + dyt(js_pe)*ny
 xt_1  = 0.-dxt(is_pe)/2.0
 xt_nx = xt_1 + dxt(is_pe)*nx
end subroutine set_grid


subroutine set_coriolis
 use main_module   
 use config_module   
 implicit none
 real*8 :: phi0,betaloc
 integer :: j
 phi0 = (y2+y1)/2.0 /180. *pi
 betaloc = 2*omega*cos(phi0)/radius
 do j=js_pe-onx,je_pe+onx
  coriolis_t(:,j) = 2*omega*sin(phi0) +betaloc*yt(j)
 enddo

end subroutine set_coriolis


subroutine set_initial_conditions
 use main_module   
 use config_module   
 use idemix_module   
 use tke_module   
 implicit none
 integer :: i,j,k
 real*8 :: y
 do k=1,nz
  do j=js_pe-onx,je_pe+onx
   do i=is_pe-onx,ie_pe+onx
     y = yt(j)/yt_ny
    salt(i,j,k,:) = 35
    temp(i,j,k,:) =  (1-zt(k)/zw(1))*15 + &
        2*(tanh((y-0.5)/0.1)+1.0) +  3*(tanh(-(y-0.5)/0.2)+1.0)
   enddo
  enddo
 enddo


 do j=js_pe-onx,je_pe+onx
   do i=is_pe-onx,ie_pe+onx
     if ( yt(j)< y1) surface_taux(i,j) = &
         .1e-3*sin(pi*(yu(j)-y0)/(y1-yt_1))*maskU(i,j,nz)
   enddo
 enddo

 if (enable_idemix ) then
  do j=js_pe-onx,je_pe+onx
   do i=is_pe-onx,ie_pe+onx
     forc_iw_bottom(i,j) =  3e-6
     forc_iw_surface(i,j) = 0.5e-6
   enddo
  enddo
 endif

 if (enable_tke ) then
  do j=js_pe-onx+1,je_pe+onx
   do i=is_pe-onx+1,ie_pe+onx
     forc_tke_surface(i,j) = sqrt( (0.5*(surface_taux(i,j)+surface_taux(i-1,j)))**2  &
                                  +(0.5*(surface_tauy(i,j)+surface_tauy(i,j-1)))**2 )**(3./2.) 
   enddo
  enddo
 endif

end subroutine set_initial_conditions




function tstar(j)
 use main_module   
 use config_module   
 implicit none
 integer :: j
 real*8 :: tstar
 tstar=15
 if (yt(j)<y1) tstar=15*(yt(j)-yt_1)/(y1-yt_1)
 if (yt(j)>y2) tstar=15*(1-(yt(j)-y2)/(yt_ny-y2) )
end function tstar



subroutine set_forcing
 use main_module   
 implicit none
 integer :: i,j
 real*8 :: tstar

 do j=js_pe-onx,je_pe+onx
  do i=is_pe-onx,ie_pe+onx
    forc_temp_surface(i,j)=dzt(nz)/(30.*86400.)*(tstar(j)-temp(i,j,nz,tau)) 
  enddo
 enddo
end subroutine set_forcing


subroutine set_topography
 use main_module   
 use config_module   
 implicit none
 integer :: i,j
 !kbot=0
 kbot=1
 do i=is_pe,ie_pe
   do j=js_pe,je_pe
     !if ( yt(j)>=y1.and.xt(i)<=int(xt_nx*0.1) )  kbot(i,j)=0
     if ( yt(j)>=y1.and.i>=nx/2-hresolv.and.i<=nx/2+hresolv)  kbot(i,j)=0
   enddo
 enddo
end subroutine set_topography



subroutine set_diagnostics
end subroutine set_diagnostics

subroutine set_particles
end subroutine set_particles


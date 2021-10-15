
!=======================================================================
!  test for convergence of energetics in simple setup
!======================================================================= 


module config_module
 ! use this module only locally in this file
 implicit none
 !real*8 :: y0 = 20, y1 = 28, y2=32
 real*8 :: y0 = -20, y1 = -12, y2=5
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
 implicit none

  !nx   = 5; nz   = 8; ny  = 18
  nx   = 5; nz   = 8; ny  = 42
  dt_mom    = 3600/2.0
  dt_tracer = 86400*0.5

  coord_degree     = .true.
  enable_cyclic_x  = .true.

  runlen = 365*86400.*2000  

  enable_diag_ts_monitor = .true.; ts_monint = 86400.*365!/12.
  enable_diag_snapshots  = .true.; snapint  =  86400.*365!/12.
  enable_diag_tracer_content = .true.; trac_cont_int = 86400.*365
  enable_diag_energy  = .true.; energint  =  86400.*365; energfreq =  86400.*365

  congr_epsilon = 1e-9
  enable_streamfunction = .true.
  congr_max_iterations = 15000

  enable_hor_friction = .true.; A_h = (1*degtom)**3*2e-11    
  enable_hor_friction_cos_scaling = .true.; hor_friction_cosPower=1
  enable_bottom_friction = .true.; r_bot = 1e-5

  !enable_superbee_advection = .true.

  !enable_hor_diffusion = .true.; K_h = 1000.0
  enable_neutral_diffusion = .true.; 
      K_iso_0 = 1000.0
      K_iso_steep = 1000.0
      iso_dslope=4./1000.0
      iso_slopec=4./1000.0
  enable_skew_diffusion = .true.
  !enable_TEM_friction = .true.

  enable_implicit_vert_friction = .true.; 
  enable_tke = .true.
  c_k = 0.1
  c_eps = 0.7
  alpha_tke = 30.0
  mxl_min = 1d-8
  tke_mxl_choice = 2

  K_gm_0 = 1000.0
  enable_idemix = .true.

  eq_of_state_type = 3 
end subroutine set_parameter



subroutine set_grid
  use main_module   
  use config_module   
  implicit none
  real*8 :: ddz(15)  = (/50.,70.,100.,140.,190.,240.,290.,340.,390.,440.,490.,540.,590.,640.,690./)
    dxt = 1.0
    dyt = 1.0
    x_origin=  0.0
    y_origin= y0
    ddz = ddz(15:1:-1)/2.5
    dzt = ddz(8:15)
end subroutine set_grid


subroutine set_coriolis
 use main_module   
 implicit none
 integer :: j
 do j=js_pe-onx,je_pe+onx
     coriolis_t(:,j) = 2*omega*sin( yt(j)/180.*pi ) 
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
     y = yt(j)/yt(ny)
    salt(i,j,k,:) = 35
    temp(i,j,k,:) =  (1-zt(k)/zw(1))*15 +  2*(tanh((y-0.5)/0.1)+1.0) +  3*(tanh(-(y-0.5)/0.2)+1.0)
   enddo
  enddo
 enddo


 do j=js_pe-onx,je_pe+onx
   do i=is_pe-onx,ie_pe+onx
     if ( yt(j)< y1) surface_taux(i,j) =.02e-3*sin(pi*(yu(j)-yu(1))/(y1-yt(1)))*maskU(i,j,nz)
   enddo
 enddo

 if (enable_idemix ) then
  do j=js_pe-onx,je_pe+onx
   do i=is_pe-onx,ie_pe+onx
     forc_iw_bottom(i,j) =  1e-6
     forc_iw_surface(i,j) = 0.1e-6
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
 if (yt(j)<y1) tstar=15*(yt(j)-yt(1))/(y1-yt(1))
 if (yt(j)>y2) tstar=15*(1-(yt(j)-y2)/(yt(ny)-y2) )
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
 kbot=0
 do i=is_pe,ie_pe
   do j=js_pe,je_pe
     if ( (yt(j)<y1).or.(xt(i)>xt(1)))  kbot(i,j)=1
   enddo
 enddo
end subroutine set_topography





subroutine set_diagnostics
end subroutine set_diagnostics

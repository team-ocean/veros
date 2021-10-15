
!=======================================================================
!  test for isopycnal mixing
!======================================================================= 



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

  nx   = 2; nz   = 15; ny  = 30
  dt_mom    = 3600/2.0
  dt_tracer = 86400*0.5

  coord_degree     = .true.
  enable_cyclic_x  = .true.

  runlen = 365*86400.*2000  

  enable_diag_ts_monitor = .true.; ts_monint = 86400.*365!/12.
  enable_diag_snapshots  = .true.; snapint  =  86400.*365!/12.
  enable_diag_tracer_content = .true.; trac_cont_int = 86400.*365

  congr_epsilon = 1e-9
  enable_streamfunction = .true.
  congr_max_iterations = 15000

  enable_hor_friction = .true.; A_h = (1*degtom)**3*2e-11    
  enable_hor_friction_cos_scaling = .true.; hor_friction_cosPower=1

  !enable_hor_diffusion = .true.; K_h = 1000.0
  enable_neutral_diffusion = .true.; 
      K_iso_0 = 1000.0
      K_iso_steep = 1000.0
      iso_dslope=4./1000.0
      iso_slopec=4./1000.0
  enable_skew_diffusion = .true.
  !enable_TEM_friction = .true.
  K_gm_0 = 1000.0
  enable_conserve_energy = .false.
  eq_of_state_type = 5 
end subroutine set_parameter



subroutine set_grid
  use main_module   
  implicit none
  real*8 :: ddz(15)  = (/50.,70.,100.,140.,190.,240.,290.,340.,390.,440.,490.,540.,590.,640.,690./)
    dxt = 1.0
    dyt = 1.0
    x_origin=  0.0
    y_origin= 20.0
    dzt = ddz(15:1:-1)/2.5
end subroutine set_grid


subroutine set_coriolis
 use main_module   
 implicit none
 real*8 :: phi0,betaloc
 integer :: j
 phi0 = 25.0 /180. *pi
 betaloc = 2*omega*cos(phi0)/radius
 do j=js_pe-onx,je_pe+onx
  coriolis_t(:,j) = 1e-4 +betaloc*yt(j)
 enddo
end subroutine set_coriolis


subroutine set_initial_conditions
 use main_module   
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
    !salt(i,j,k,:) = exp( -(zt(k) - zt(1)/2)**2/300**2 - (yt(j)-yt(ny)/2)**2/50e3**2 )
    temp(i,j,k,:) =  (1-zt(k)/zw(1))*15 +  2*(tanh((y-0.5)/0.1)+1.0) +  3*(tanh(-(y-0.5)/0.2)+1.0)
   enddo
  enddo
 enddo


 !do j=js_pe-onx,je_pe+onx
 ! do i=is_pe-onx,ie_pe+onx
 !  if (j<=15) temp(i,j,nz-7:nz,tau)   = temp(i,j,nz-8,tau)
 !  if (j<=15) temp(i,j,nz-7:nz,taum1)   = temp(i,j,nz-8,taum1)
 ! enddo
 !enddo
end subroutine set_initial_conditions

subroutine set_momentum_forcing
end subroutine set_momentum_forcing

subroutine set_forcing
end subroutine set_forcing

subroutine set_topography
 use main_module   
 implicit none
 !kbot=0
 !kbot(1:nx,1:ny)=1
 kbot=1
end subroutine set_topography

subroutine set_diagnostics
end subroutine set_diagnostics

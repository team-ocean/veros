
!=======================================================================
!     Templates for setup
!     Exp. Disturb in Olbers and Eden, JPO, 2003
!     Long Rossby wave propagation
!=======================================================================


module config_module
 ! use this module only locally in this file
 implicit none
 real*8, parameter   :: N_0  = 2.6e-3
 !real*8, allocatable :: temp_back(:,:,:),dtemp_back(:,:,:,:)
 real*8, parameter   :: x0=30.0, y0=48.0, x_len = 8.0
end module config_module


subroutine set_parameter
 ! ----------------------------------
 !       set here main parameter
 ! ----------------------------------
 use main_module
 use config_module
 use diagnostics_module
 implicit none

 nx=20;nz=15;ny=20
 dt_mom    = 3600.0
 dt_tracer = 3600.0

 enable_superbee_advection    = .true.
 enable_hor_friction          = .true.
 A_h=2e3

 !enable_hor_friction = .true.; A_h = (2*degtom)**3*2e-11    
 !enable_hor_friction_cos_scaling = .true.; hor_friction_cosPower=1

 kappah_0=1.e-4
 kappam_0=1.e-3

 congr_epsilon = 1e-12
 congr_max_iterations = 50000
 enable_streamfunction = .true.

 enable_conserve_energy   = .false.
 coord_degree             = .true.
 enable_hydrostatic       = .true.
 eq_of_state_type = 1 
 !enable_tempsalt_sources =  .true.

 runlen=5*365.0*86400.
 !enable_diag_ts_monitor = .true.; ts_monint = dt_mom
 enable_diag_snapshots  = .true.; snapint  =  3*86400
end subroutine set_parameter


subroutine set_grid
 use main_module
 implicit none
 real*8 :: ddz(15)  = (/50.,70.,100.,140.,190.,240.,290.,340.,390.,440.,490.,540.,590.,640.,690./)
 dxt=2.0
 dyt=2.0
 if (nz == 15) then
   dzt = ddz(15:1:-1)
 else
   call halt_stop('in set grid')
 endif
 y_origin = 30.0
end subroutine set_grid


subroutine set_coriolis
  use main_module   
  implicit none
  integer :: j
  real*8 :: phi0 = 30.0 /180. *pi,betaloc 
  betaloc = 2*omega*cos(phi0)/radius
  do j=js_pe-onx,je_pe+onx
    !coriolis_t(:,j) = 2*omega*sin(phi0) + betaloc*(yt(j)-y_origin)*degtom
    coriolis_t(:,j) = 2*omega*sin(yt(j)/180.*pi) 
  enddo
end subroutine set_coriolis


subroutine set_initial_conditions
 use main_module
 use config_module
 use linear_eq_of_state
 implicit none
 integer :: i,j,k
 real*8 :: alpha, t0

 alpha = linear_eq_of_state_drhodt()
 !allocate( temp_back(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); temp_back=0.0
 !allocate( dtemp_back(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3) ); dtemp_back=0.0

 do k=1,nz
  do j=js_pe-onx,je_pe+onx
   do i=is_pe-onx,ie_pe+onx
    !temp_back(i,j,k) = 20.0-N_0**2*zt(k)*maskT(i,j,k)*rho_0/(grav*alpha)
    t0= 20.0-N_0**2*zt(k)*rho_0/(grav*alpha)
    temp(i,j,k,:)    = (t0-0.5*exp(-(xt(i)-x0)**2/x_len**2-(yt(j)-y0)**2/x_len**2)*exp(zt(k)/500.))*maskT(i,j,k)
   enddo
  enddo
 enddo
end subroutine set_initial_conditions


subroutine set_forcing
 use main_module
 use config_module
 implicit none

 ! update density, etc of last time step
 !temp(:,:,:,tau) = temp(:,:,:,tau) + temp_back
 !call calc_eq_of_state(tau)
 !temp(:,:,:,tau) = temp(:,:,:,tau) - temp_back
      
 ! advection of background temperature
 !call  advect_tracer(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,temp_back,dtemp_back(:,:,:,tau)) 
 !temp_source = (1.5+ab_eps)*dtemp_back(:,:,:,tau) - ( 0.5+ab_eps)*dtemp_back(:,:,:,taum1)
end subroutine set_forcing


subroutine set_topography
 use main_module   
 use config_module   
 implicit none
 kbot=0
 kbot(is_pe:ie_pe,js_pe:je_pe)=1
end subroutine set_topography


subroutine set_diagnostics
end subroutine set_diagnostics







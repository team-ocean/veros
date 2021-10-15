
!=======================================================================
! Kelvin Helmholtz instability
!=======================================================================

module config_module
 real*8 :: fac=1.0,mix=5e-3
end module config_module

subroutine set_parameter
 use main_module   
 use config_module   
 use diagnostics_module   
 use tke_module   
 implicit none
  ny=1
  nx=int(1.5*64*fac)
  nz=int(40*fac)

  dt_mom=0.04/fac
  dt_tracer = dt_mom

  enable_conserve_energy = .false. 
  coord_degree         =.false.
  enable_cyclic_x      =.true.
  enable_hydrostatic   =.false.
  eq_of_state_type = 1 

  congr_epsilon = 1d-12
  congr_max_iterations = 5000
  !enable_streamfunction = .true.

  congr_epsilon_non_hydro=   1d-6
  congr_max_itts_non_hydro = 5000    

  enable_tempsalt_sources = .true.
  enable_momentum_sources = .true.

  enable_explicit_vert_friction = .true.; kappaM_0 = mix/fac**2
  enable_hor_friction = .true.;            A_h = mix/fac**2
  enable_superbee_advection       =.true.
  !kappaH_0 = mix/fac**2
  !enable_hor_diffusion = .true.;       K_h = mix/fac**2

  runlen =  86400.0
  enable_diag_ts_monitor = .true.; ts_monint =0.5!dt_mom
  enable_diag_snapshots  = .true.; snapint  = 0.5!dt_mom

  enable_diag_particles = .true.; particles_int = 0.5
end subroutine set_parameter


subroutine set_grid
 use main_module   
 use config_module   
 implicit none
 dxt(:)=0.25/fac 
 dyt(:)=0.25/fac 
 dzt(:)=0.25/fac 
end subroutine set_grid

subroutine set_coriolis
 !use main_module   
 !use config_module   
 !implicit none
 !coriolis_t = 2*omega*sin( 30./180.*pi)
 !coriolis_h = 2*omega*cos( 30./180.*pi)
end subroutine set_coriolis


real*8  function t_star(k)
 use main_module   
 implicit none
 integer :: k
 t_star=9.85-6.5*tanh( (zt(k)-zt(nz/2) ) /zt(1)*100 )
end function t_star


real*8  function u_star(k)
 use main_module   
 implicit none
 integer :: k
 u_star=0.6+0.5*tanh( (zt(k)-zt(nz/2))/zt(1)*100)
end function u_star



subroutine set_initial_conditions
  use main_module   
  implicit none
  integer :: i,j,k
  real*8 :: fxa,t_star,u_star
  do k=1,nz
   do j=js_pe-onx,je_pe+onx
    do i=is_pe-onx,ie_pe+onx
       fxa=1e-3*zt(1)*sin(xt(i)/(20*dxt(is_pe))*pi)
       temp(i,j,k,:)=( fxa+t_star(k)  )*maskT(i,j,k)
       u(i,j,k,:)   = u_star(k)*maskU(i,j,k)
    enddo
   enddo
  enddo
end subroutine set_initial_conditions




subroutine set_forcing
 use main_module
 implicit none
 integer :: i,k
 real*8 :: T_rest,t_star,u_star
  T_rest=1./(15.*dt_mom)
  do k=1,nz
   do i=2,nx/8
     if (i>=is_pe .and. i<=ie_pe) then
       temp_source(i,:,k)=maskT(i,:,k)*T_rest*(t_star(k)-temp(i,:,k,tau))
       u_source(i,:,k)  = maskU(i,:,k)*T_rest*(u_star(k)-u(i,:,k,tau))
     endif
   enddo
  enddo
end subroutine set_forcing



subroutine set_topography
 use main_module   
 implicit none
 kbot =1
end subroutine set_topography


subroutine set_diagnostics
end subroutine set_diagnostics


subroutine set_particles
 use main_module   
 use particles_module   
 implicit none
 integer :: n
 real :: fxa,xs,xe,zs,ze

 call allocate_particles(2000)
 xs=0;xe=nx*dxt(is_pe);
 zs=zt(1);ze=zt(nz)
 do n=1,nptraj
        call random_number(fxa)
        pxyz(1,n) = xs+fxa*(xe-xs)
        pxyz(2,n) = yt(1)
        call random_number(fxa)
        pxyz(3,n) = zs+fxa*(ze-zs)
 enddo

 !call allocate_particles(2)
 !pxyz(:,1) = (/10.d0 , yt(js_pe), -8.d0/)
 !pxyz(:,2) = (/14.d0 , yt(js_pe), -8.d0/)

end subroutine set_particles

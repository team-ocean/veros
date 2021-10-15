
!=======================================================================
! Wide eddying channel with restoring zones at side walls
! Experiment CHANNEL in Eden (2010), Ocean modeling 32 (2010) 58-71
!=======================================================================

module config_module
 ! use this module only locally in this file
 implicit none
 real*8,parameter :: hRESOLVE = 0.5  ! 2 in original model 
 real*8,parameter :: vRESOLVE = 0.5  ! 2 in original model
 real*8,parameter :: M_0     = sqrt(1e-5*0.1/1024.*9.81)
 real*8,parameter :: N_0     = 0.004
 integer :: spg_width=int(3*hRESOLVE)
 real*8 :: T_rest=1./(5.*86400)
 !real*8 :: DX  = 32094.1729769
 !real*8,parameter :: DX  = 30e3
 !real*8,parameter :: Lx  = DX*128
 real*8,parameter :: Lx  = 3800e3
 real*8,parameter :: H   = 1800.0
end module config_module





subroutine set_parameter
 ! ----------------------------------
 !       set here main parameter
 ! ----------------------------------
 use main_module
 use config_module
 use diagnostics_module   
 use tke_module   
 use idemix_module   
 implicit none

 nx = int(128*hRESOLVE) ; nz = int(18*vRESOLVE); ny = int(128*hRESOLVE)
 dt_tracer = 1800.0/hRESOLVE
 dt_mom    = 1800.0/hRESOLVE

 coord_degree           = .false.
 enable_cyclic_x        = .true.
 enable_hydrostatic     = .true.
 eq_of_state_type       = 1
     
 congr_epsilon = 1e-12
 congr_max_iterations = 5000
 enable_streamfunction = .true.
     
 enable_ray_friction      = .true.
 r_ray = 0.5e-7 
     
 !enable_hor_friction = .false.
 !A_h = 100/HRESOLVE**2  

 enable_biharmonic_mixing    = .true.  ! was enabled in original model
 K_hbi  = 1e11/hRESOLVE**4
 !enable_superbee_advection = .true.

 enable_biharmonic_friction  = .true.  ! was enabled in original model
 A_hbi  = 1e11/hRESOLVE**4
     
 enable_tempsalt_sources =  .true.

 enable_conserve_energy = .false.
 kappah_0=1e-4/VRESOLVE**2
 kappam_0=1e-3/VRESOLVE**2

 !enable_implicit_vert_friction = .true.;
 !enable_tke = .true.
 c_k = 0.1
 c_eps = 0.7
 alpha_tke = 30.0
 mxl_min = 1d-8
 tke_mxl_choice = 2
 enable_tke_superbee_advection = .true.

 !enable_idemix = .true.
 enable_idemix_hor_diffusion = .true.;
 enable_idemix_superbee_advection = .true.

 runlen =  365*86400.*10
 enable_diag_ts_monitor = .true.; ts_monint = dt_tracer
 enable_diag_snapshots  = .true.; snapint  =  3*86400
 !enable_diag_averages   = .true.
 !aveint  = 365*86400./12!*100
 !avefreq = dt_tracer*10
 !enable_diag_energy     = .true.; energint =   86400*3.
 !energfreq = dt_tracer*10

  enable_diag_particles = .true.; particles_int = 3*86400.0

end subroutine set_parameter


subroutine set_grid
 use main_module   
 use config_module   
 implicit none
 !dxt = 1./3.*degtom*cos(30./180.*pi)/hRESOLVE ! original grid
 !dzt = 100.0 /vRESOLVE
 dxt = Lx/nx
 dyt = Lx/ny
 dzt = H/nz
end subroutine set_grid


subroutine set_coriolis
 use main_module   
 use config_module
 implicit none
 integer :: j
 real*8 :: phi0 , betaloc
 phi0 = 10.0 /180. *pi
 betaloc = 2*omega*cos(phi0)/radius
 do j=js_pe-onx,je_pe+onx
    coriolis_t(:,j) = 2*omega*sin(phi0) +betaloc*yt(j)
 enddo
end subroutine set_coriolis


subroutine set_initial_conditions
 ! ----------------------------------
 !      add here initial conditions
 ! ----------------------------------
 use main_module
 use config_module
 implicit none
 integer :: i,j,k
 real*8 :: B0,alpha,get_drhodT
 do k=1,nz
   do j=js_pe,je_pe
     do i=is_pe,ie_pe
       alpha = get_drhodT(salt(i,j,k,tau),temp(i,j,k,tau),zt(k) ) 
       B0=M_0**2*yt(j)+0.5e-3*sin(xt(i)/Lx*8.5*pi)!*exp(-(y-0.5)**2/0.5**2)
       temp(i,j,k,:)  = ( 32+rho_0/grav/alpha*(B0-N_0**2*zt(k))  )*maskT(i,j,k)
     enddo
   enddo
 enddo
end subroutine set_initial_conditions



subroutine set_forcing
 ! ----------------------------------
 !      add here restoring zones
 ! ----------------------------------
 use main_module
 use config_module
 implicit none
 integer :: i,j,k 
 real*8 :: B0, alpha, get_drhodT
 if (enable_tempsalt_sources) then
  do k=1,nz
   do j=2,spg_width+1
     if (j>=js_pe .and.  j<=je_pe) then
       do i=is_pe,ie_pe
         alpha = get_drhodT(salt(i,j,k,tau),temp(i,j,k,tau),zt(k) ) 
         B0=32+( yt(j)*M_0**2-N_0**2*zt(k) )*rho_0/grav/alpha
         temp_source(i,j,k)=maskT(i,j,k)*t_rest/(j-1.)*(B0-temp(i,j,k,tau))
       enddo
      endif
   enddo
   do j=ny-1,ny-spg_width,-1
     if (j>=js_pe .and.  j<=je_pe) then
       do i=is_pe,ie_pe
         alpha = get_drhodT(salt(i,j,k,tau),temp(i,j,k,tau),zt(k) ) 
         B0=32+( yt(j)*M_0**2-N_0**2*zt(k) )*rho_0/grav/alpha
         temp_source(i,j,k)=maskT(i,j,k)*t_rest/(-1.*(j-ny))*(B0-temp(i,j,k,tau))
       enddo
      endif
   enddo
  enddo
 endif
end subroutine set_forcing


subroutine set_topography
 use main_module   
 implicit none
 kbot=1
end subroutine set_topography


subroutine set_diagnostics
 use main_module   
 use eke_module   
 use tke_module   
 use idemix_module   
 use isoneutral_module   
 implicit none
 call register_average('taux','Zonal wind stress','m^2/s','UT',surface_taux,0D0,.false.)
 call register_average('tauy','Meridional wind stress','m^2/s','TU',surface_tauy,0D0,.false.)
 call register_average('forc_temp_surface','Surface temperature flux','m K/s','TT',forc_temp_surface,0D0,.false.)
 call register_average('forc_salt_surface','Surface salinity flux','m g/s kg','TT',forc_salt_surface,0D0,.false.)
 if (enable_streamfunction) then
   call register_average('psi','Barotropic streamfunction','m^2/s','UU',psi(:,:,tau),0D0,.false.)
 else
   call register_average('psi','Surface pressure','m^2/s','TT',psi(:,:,tau),0D0,.false.)
 endif
 call register_average('temp','Temperature','deg C','TTT',0d0,temp(:,:,:,tau),.true.)
 call register_average('salt','Salinity','g/kg','TTT',0d0,salt(:,:,:,tau),.true.)
 call register_average('u','Zonal velocity','m/s','UTT',0d0,u(:,:,:,tau),.true.)
 call register_average('v','Meridional velocity','m/s','TUT',0d0,v(:,:,:,tau),.true.)
 call register_average('w','Vertical velocity','m/s','TTU',0d0,w(:,:,:,tau),.true.)
 call register_average('Nsqr','Square of stability frequency','1/s^2','TTU',0d0,Nsqr(:,:,:,tau),.true.)
 call register_average('Hd','Dynamic enthalpy','m^2/s^2','TTT',0d0,Hd(:,:,:,tau),.true.)

 call register_average('K_diss_v','Dissipation by vertical friction','m^2/s^3','TTU',0d0,K_diss_v,.true.)
 call register_average('K_diss_h','Dissipation by lateral friction','m^2/s^3','TTU',0d0,K_diss_h,.true.)
 call register_average('K_diss_bot','Dissipation by bottom friction','m^2/s^3','TTU',0d0,K_diss_bot,.true.)
 call register_average('P_diss_v','Dissipation by vertical mixing','m^2/s^3','TTU',0d0,P_diss_v,.true.)
 call register_average('P_diss_nonlin','Dissipation by nonlinear vert. mix.','m^2/s^3','TTU',0d0,P_diss_nonlin,.true.)
 call register_average('P_diss_iso','Dissipation by Redi mixing tensor','m^2/s^3','TTU',0d0,P_diss_iso,.true.)

 call register_average('kappaH','Vertical diffusivity','m^2/s','TTU',0d0,kappaH,.true.)
 if (enable_skew_diffusion)  then
   call register_average('B1_gm','Zonal component of GM streamfct.','m^2/s','TUT',0d0,B1_gm,.true.)
   call register_average('B2_gm','Meridional component of GM streamfct.','m^2/s','UTT',0d0,B2_gm,.true.)
 else
   call register_average('kappa_gm','Vertical GM viscosity','m^2/s','TTU',0d0,kappa_gm,.true.)
   call register_average('K_diss_gm','Dissipation by GM friction','m^2/s^3','TTU',0d0,K_diss_gm,.true.)
 endif

 if (enable_tke)  then
   call register_average('TKE','Turbulent kinetic energy','m^2/s^2','TTU',0d0,tke(:,:,:,tau),.true.)
   call register_average('Prandtl','Prandtl number',' ','TTU',0d0,Prandtlnumber,.true.)
   call register_average('mxl','Mixing length',' ','TTU',0d0,mxl,.true.)
   call register_average('tke_diss','Dissipation of TKE','m^2/s^3','TTU',0d0,tke_diss,.true.)
   call register_average('forc_tke_surface','TKE surface forcing','m^3/s^2','TT',forc_tke_surface,0D0,.false.)
   call register_average('tke_surface_corr','TKE surface flux correction','m^3/s^2','TT',tke_surf_corr,0D0,.false.)
 endif
 if (enable_idemix)  then
   call register_average('E_iw','Internal wave energy','m^2/s^2','TTU',0d0,e_iw(:,:,:,tau),.true.)
   call register_average('forc_iw_surface','IW surface forcing','m^3/s^2','TT',forc_iw_surface,0D0,.false.)
   call register_average('forc_iw_bottom','IW bottom forcing','m^3/s^2','TT',forc_iw_bottom,0D0,.false.)
   call register_average('iw_diss','Dissipation of E_iw','m^2/s^3','TTU',0d0,iw_diss,.true.)
   call register_average('c0','Vertical IW group velocity','m/s','TTU',0d0,c0,.true.)
   call register_average('v0','Horizontal IW group velocity','m/s','TTU',0d0,v0,.true.)
 endif
 if (enable_eke)  then
  call register_average('EKE','Eddy energy','m^2/s^2','TTU',0d0,eke(:,:,:,tau),.true.)
  call register_average('K_gm','Lateral diffusivity','m^2/s','TTU',0d0,K_gm,.true.)
  call register_average('eke_diss','Eddy energy dissipation','m^2/s^3','TTU',0d0,eke_diss,.true.)
  call register_average('L_Rossby','Rossby radius','m','TT',L_rossby,0d0,.false.)
  call register_average('L_Rhines','Rhines scale','m','TTU',0d0,L_Rhines,.true.)

 endif

end subroutine set_diagnostics




subroutine set_particles
 use main_module   
 use particles_module   
 implicit none
 integer :: n
 real :: fxa
 real*8 :: xs,xe,zs,ze,ys,ye

 call allocate_particles(1000)
 xs=0;xe=nx*dxt(is_pe);
 ys=dyt(js_pe);ye=(ny-2)*dyt(js_pe);
 !zs=zt(1);ze=zt(nz)
 zs=-500;ze=-500
 do n=1,nptraj
    call random_number(fxa)
    pxyz(1,n) = xs+fxa*(xe-xs)
    call random_number(fxa)
    pxyz(2,n) = ys+fxa*(ye-ys)
    call random_number(fxa)
    pxyz(3,n) = zs+fxa*(ze-zs)
 enddo
end subroutine set_particles




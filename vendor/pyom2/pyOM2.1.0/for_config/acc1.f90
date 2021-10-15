
!=======================================================================
!  idealised Southern Ocean, same as in Viebahn and Eden (2010) Ocean modeling
!======================================================================= 


module config_module
 ! use this module only locally in this file
 implicit none
 real*8,parameter :: hRESOLVE = 1.0 ! 1 in original model
 real*8,parameter :: vRESOLVE = 1.0 ! 1 in original model
 real*8,parameter :: N_0     = 0.004
 real*8 :: L_y,L_x
 real*8 :: t_rest=30*86400
 real*8:: phi0 = -25.0 /180. *3.1415
end module config_module


subroutine set_parameter
 ! ----------------------------------
 !       set here main parameter
 ! ----------------------------------
 use main_module   
 use config_module
 use diagnostics_module   
 use eke_module   
 use tke_module   
 use isoneutral_module   
 use idemix_module   

 implicit none
 nx   = int( 128*hRESOLVE ); nz   = int( 18 *vRESOLVE ); ny   = int( 128*hRESOLVE )
 dt_mom     = 1200.0/hRESOLVE
 dt_tracer  = 1200.0/hRESOLVE !*5

 coord_degree           = .false.
 enable_cyclic_x        = .true.
 enable_hydrostatic     = .true.
 eq_of_state_type       = 5
     
 congr_epsilon = 1e-6
 congr_max_iterations = 5000
 enable_streamfunction = .true.
     
 !enable_superbee_advection = .true.
 !enable_hor_diffusion = .true.
 !K_h = 200
 enable_biharmonic_mixing = .true.
 K_hbi  = 5e11/hRESOLVE**4

 !enable_implicit_vert_friction = .true.; 
 !enable_TEM_friction = .true.
 !K_gm_0 = 1000.0

 !enable_hor_friction  = .true.
 !A_h = 5e4!/hRESOLVE  ! for coarse model version

 enable_biharmonic_friction  = .true. ! for eddy resolving version
 A_hbi  = 5e11/hRESOLVE**4

 enable_bottom_friction = .true.
 r_bot = 1e-5*vRESOLVE

 !kappah_0=1.e-4/vRESOLVE
 !kappam_0=1.e-3/vRESOLVE
 !enable_conserve_energy = .false.

 enable_implicit_vert_friction = .true.;
 enable_tke = .true.
  c_k = 0.1
  c_eps = 0.7
  alpha_tke = 30.0
  mxl_min = 1d-8
  tke_mxl_choice = 2
  enable_tke_superbee_advection = .false.

 enable_idemix = .true.
 enable_idemix_hor_diffusion = .true.;
 enable_idemix_superbee_advection = .true.

 runlen =  365*86400.*0.5

 enable_diag_ts_monitor = .true.; ts_monint = dt_tracer
 enable_diag_snapshots  = .true.; snapint  =  86400*3.
 !enable_diag_overturning= .true.; overint  =  86400*3; overfreq = overint
 enable_diag_energy     = .true.; energint =   86400*3.
 energfreq = dt_tracer*10
 enable_diag_averages   = .true.
 aveint  = 365*86400
 avefreq = dt_tracer*10
end subroutine set_parameter



subroutine set_grid
 use main_module   
 use config_module   
 implicit none
 dxt    = 20e3/hRESOLVE
 dyt    = 20e3/hRESOLVE
 dzt    = 50.0/vRESOLVE
end subroutine set_grid

subroutine set_coriolis
 use main_module   
 use config_module   
 implicit none
 integer :: j
 do j=js_pe-onx,je_pe+onx
   coriolis_t(:,j) = 2*omega*sin(phi0) + 2*omega*cos(phi0)/radius*yt(j)
 enddo
end subroutine set_coriolis


subroutine set_initial_conditions
 use main_module
 use tke_module
 use idemix_module
 use config_module
 implicit none
 integer :: i,j
 real*8 :: y

 salt = 35.0
 temp = 0.0

 do j=js_pe-onx, je_pe+onx
   do i=is_pe-onx, ie_pe+onx
     if (yt(j)<L_y/2.0) then
         y = yu(j)/L_y
         surface_taux(i,j) = .1e-3*sin(2*pi*y)*maskU(i,j,nz)
     endif
   enddo
 enddo

 if (enable_tke ) then
  do j=js_pe-onx+1,je_pe+onx
   do i=is_pe-onx+1,ie_pe+onx
     forc_tke_surface(i,j) = sqrt( (0.5*(surface_taux(i,j)+surface_taux(i-1,j)))**2  &
                                  +(0.5*(surface_tauy(i,j)+surface_tauy(i,j-1)))**2 )**(3./2.)
   enddo
  enddo
 endif

 if (enable_idemix ) then
   forc_iw_bottom =  1e-6
   forc_iw_surface = 0.1e-6
 endif

end subroutine set_initial_conditions



subroutine b_surface(bstar,j)
 use main_module   
 use config_module   
 implicit none
 integer :: j
 real*8 :: db,bstar,y2
 db = -30e-3
 bstar=db
 if (yt(j)<L_y/2.0) then
       bstar=db*yt(j)/(L_y/2.0)
 endif
 y2=L_y*0.75
 if (yt(j)>y2) then
       bstar=db*(1-(yt(j)-y2)/(L_y-y2) )
 endif
end subroutine b_surface


subroutine set_forcing
 use main_module
 use config_module
 implicit none
 integer :: i,j
 real*8 :: bstar,alpha,p0=0.0,get_drhodt
 do j=js_pe, je_pe
   call b_surface(bstar,j)
   do i=is_pe, ie_pe
     alpha = get_drhodT(salt(i,j,nz,tau),temp(i,j,nz,tau),p0) 
     forc_temp_surface(i,j)=dzt(nz)/t_rest*(bstar*rho_0/grav/alpha-temp(i,j,nz,tau))
   enddo
 enddo
end subroutine set_forcing

subroutine set_topography
 use main_module   
 use config_module   
 implicit none
 integer :: i,j

 L_y = 0.0; if (my_blk_j == n_pes_j) L_y = yu(ny)
 call global_max(L_y)
 L_x = 0.0; if (my_blk_i == n_pes_i) L_x = xu(nx)
 call global_max(L_x)
 if (my_pe==0) print*,' domain size is ',L_x,' m x ',L_y,' m'

 kbot=1
 do j=js_pe,je_pe
   do i=is_pe,ie_pe
      if ((yt(j)>L_y/2.0).and.(xt(i)>L_x*.75.or.xt(i)<L_x*.25))  kbot(i,j)=0
   enddo
 enddo
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







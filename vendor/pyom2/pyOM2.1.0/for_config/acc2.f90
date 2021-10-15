
!=======================================================================
!  idealised global ocean, 2x2 deg and 15 vertical levels
!======================================================================= 

module config_module
 implicit none
 real*8 :: yt_start = -39.0
 real*8 :: yt_end   = 43
 real*8 :: yu_start = -40.0
 real*8 :: yu_end   = 42
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
  nx   = 30; nz   = 15; ny  = 42
  dt_mom    = 4800  
  dt_tracer = 86400/2.0

  coord_degree     = .true.
  enable_cyclic_x  = .true.

  runlen =  365*86400.*100

  enable_diag_ts_monitor = .true.; ts_monint = 365*86400./12.
  enable_diag_snapshots  = .true.; snapint  =  365*86400./12.0
  !enable_diag_overturning= .true.; overint  =  365*86400./48.0; overfreq = dt_tracer
  enable_diag_energy     = .true.; energint =   365*86400./48; energfreq = dt_tracer*10
  !enable_diag_averages   = .true.; aveint  = 365*86400.; avefreq = dt_tracer*10
  !enable_diag_particles = .true.; particles_int = snapint

  congr_epsilon = 1e-12
  congr_max_iterations = 5000
  enable_streamfunction = .true.

  !enable_hor_diffusion = .true.;  K_h=2000
  enable_neutral_diffusion = .true.; 
      K_iso_0 = 1000.0
      K_iso_steep = 500.0
      iso_dslope=0.005
      iso_slopec=0.01
  !enable_skew_diffusion = .true.
  enable_TEM_friction = .true.

  enable_hor_friction = .true.; A_h = (2*degtom)**3*2e-11    
  enable_hor_friction_cos_scaling = .true.; hor_friction_cosPower=1

  !enable_bottom_friction = .true.; r_bot = 1e-5
  !enable_bottom_friction_var = .true.; 
  enable_quadratic_bottom_friction = .true.; r_quad_bot = 2e-3
  enable_store_bottom_friction_tke = .true.; 

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
  eke_r_bot = 2e-3
  enable_eke_superbee_advection = .true.
  enable_eke_isopycnal_diffusion = .true.

  enable_eke_leewave_dissipation = .true.
  eke_int_diss0 = 1./(10*86400.)
  c_lee0 = 5.
  eke_Ri0 = 300.
  eke_Ri1 = 50.
  alpha_eke = 20.0



  enable_idemix = .true.
  enable_idemix_hor_diffusion = .true.; 
  !enable_eke_diss_surfbot = .true.
  !eke_diss_surfbot_frac = 0.2
  enable_idemix_superbee_advection = .true.
  
  eq_of_state_type = 3 
end subroutine set_parameter


subroutine set_grid
  use main_module   
  implicit none
  real*8 :: ddz(15)  = (/50.,70.,100.,140.,190.,240.,290.,340.,390.,440.,490.,540.,590.,640.,690./)
  if (nz == 15) then
    dzt = ddz(15:1:-1)/2.5
  else
    call halt_stop('in set grid')
  endif
  dxt = 2.0
  dyt = 2.0
  x_origin=  0.0
  y_origin= -40.0
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
 use eke_module   
 use tke_module   
 use idemix_module   
 use config_module   
 implicit none
 integer :: i,j,k

 do k=1,nz
  !salt(:,:,k,:) = 34+(1-zt(k)/zw(1))*2
  salt(:,:,k,:) = 35.0
  temp(:,:,k,:) =  (1-zt(k)/zw(1))*15
 enddo

 do j=js_pe-onx,je_pe+onx
  if ( yt(j)< -20.0) surface_taux(:,j) =.1e-3*sin(pi*(yu(j)-yu_start)/(-20.0-yt_start))*maskU(:,j,nz)
  if ( yt(j)> 10.0)  surface_taux(:,j) =.1e-3*(1-cos(2*pi*(yu(j)-10.0)/(yu_end-10.0)))*maskU(:,j,nz)
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

 if (enable_bottom_friction_var) then
  do j=js_pe-onx,je_pe+onx
     if ( yt(j)< -20.0) r_bot_var_u(:,j) = r_bot
     if ( yt(j)< -20.0) r_bot_var_v(:,j) = r_bot
  enddo
 endif

 if (enable_eke .and. enable_eke_leewave_dissipation ) then
  eke_topo_hrms = 100.0
  eke_topo_lam =1e3/0.2 !1e3/lam = 0.2 -> lam = 1e3/0.2
 endif

end subroutine set_initial_conditions



function t_star(j)
 use main_module   
 use config_module   
 implicit none
 integer :: j
 real*8 :: t_star
 t_star=15
 if (yt(j)<-20.0) t_star=15*(yt(j)-yt_start)/(-20.0-yt_start)
 if (yt(j)> 20.0) t_star=15*(1-(yt(j)-20)/(yt_end -20) )
end function t_star


function s_star(j)
 use main_module   
 use config_module   
 implicit none
 integer :: j
 real*8 :: s_star
 s_star=35
 if (yt(j)<-30.0) s_star=5*(yt(j)-yt_start)/(-30.0-yt_start)+30
 if (yt(j)> 30.0) s_star=5*(1-(yt(j)-30)/(yt_end-30) )+30
end function s_star




subroutine set_forcing
 use main_module   
 implicit none
 integer :: i,j
 real*8 :: t_rest,t_star,fxa,fxb!,s_star
 ! P-E = 50 mg/m^2/s  = 50 10^-6 kg/m^2/s , salt flux = - S (P-E)   kg/m^2 /s
 ! dS/dt (1/s) =  d/dz  F  , F (m/s ) = S (E-P) /rho ( m/s)
 real*8, parameter :: s_flux0 = -50e-6 *0.035

 t_rest=30*86400
 do j=js_pe-onx,je_pe+onx
    forc_temp_surface(:,j)=dzt(nz)/t_rest*(t_star(j)-temp(:,j,nz,tau)) 
    !forc_salt_surface(:,j)=dzt(nz)/t_rest*(s_star(j)-salt(:,j,nz,tau)) 
 enddo


 !forc_salt_surface = 0.0
 !do j=js_pe,je_pe
 ! if (yt(j)>30.0 .or. yt(j)< -30.0) forc_salt_surface(:,j)= s_flux0*maskT(:,j,nz)
 !enddo
 !fxa = 0.0; fxb = 0.0
 !do j=js_pe,je_pe
 ! do i=is_pe,ie_pe
 !   fxa = fxa + forc_salt_surface(i,j)*area_t(i,j)*maskT(i,j,nz)
 !   fxb = fxb + area_t(i,j)*maskT(i,j,nz)
 ! enddo
 !enddo
 !call global_sum(fxa); call global_sum(fxb)
 !fxa = fxa/fxb
 !forc_salt_surface = forc_salt_surface  -fxa

end subroutine set_forcing



subroutine set_topography
 use main_module   
 implicit none
 integer :: i,j
 
 kbot=0
 do i=is_pe,ie_pe
   do j=js_pe,je_pe
     if ( (yt(j)<-20.0).or.(xt(i)<59.0 .and. xt(i)>1.0 ))  kbot(i,j)=1
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
  !call register_average('eke_diss','Eddy energy dissipation','m^2/s^3','TTU',0d0,eke_diss,.true.)
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
 xs=20;xe=20;
 ys=-40;ye=40;
 zs=-200;ze=-200
 do n=1,nptraj
    call random_number(fxa)
    pxyz(1,n) = xs+fxa*(xe-xs)
    call random_number(fxa)
    pxyz(2,n) = ys+fxa*(ye-ys)
    call random_number(fxa)
    pxyz(3,n) = zs+fxa*(ze-zs)
 enddo
end subroutine set_particles





!=======================================================================
!  global 1 deg model with 115 levels
!======================================================================= 

module config_module
 ! use this module only locally in this file
 implicit none
 real*8, allocatable :: t_star(:,:,:)
 real*8, allocatable :: s_star(:,:,:)
 real*8, allocatable :: qnec(:,:,:)
 real*8, allocatable :: qnet(:,:,:)
 real*8, allocatable :: qsol(:,:,:)
 real*8, allocatable :: divpen_shortwave(:)
 real*8, allocatable :: taux(:,:,:)
 real*8, allocatable :: tauy(:,:,:)
end module config_module


subroutine set_parameter
 ! ----------------------------------
 !       set here main parameter
 ! ----------------------------------
 use main_module   
 use config_module
 use eke_module   
 use tke_module   
 use idemix_module   
 use isoneutral_module   
 use diagnostics_module   
 implicit none
  nx   = 360
  ny   = 160
  nz   = 115
  dt_mom    = 3600.0!/2.0
  dt_tracer = 3600.0!/2.0

  coord_degree     = .true.
  enable_cyclic_x  = .true.

  runlen = 365.*86400*2

  enable_diag_ts_monitor = .true.; ts_monint = 86400.0
  enable_diag_snapshots  = .true.; snapint  = 365*86400.0!/12.

  enable_diag_overturning= .true.; overint = 365*86400; 
  overfreq = overint/24.
  enable_diag_energy     = .true.; energint = 365*86400; energfreq =overfreq
  enable_diag_averages   = .true.; aveint  = 365*86400; avefreq = overfreq


  congr_epsilon = 1e-6
  congr_max_iterations = 10000
  enable_streamfunction = .true.
  !enable_congrad_verbose = .true.

  enable_neutral_diffusion = .true.; 
      K_iso_0 = 1000.0
      K_iso_steep = 50.0
      iso_dslope=0.005
      iso_slopec=0.005
  enable_skew_diffusion = .true.

  enable_hor_friction = .true.; A_h = 5e4; 
  enable_hor_friction_cos_scaling = .true.; 
  hor_friction_cosPower=1
  enable_tempsalt_sources = .true.

  enable_implicit_vert_friction = .true.
  enable_tke = .true.
  c_k = 0.1
  c_eps = 0.7
  alpha_tke = 30.0
  mxl_min = 1d-8
  tke_mxl_choice = 2
  enable_tke_superbee_advection = .true.

  !K_gm_0 = 1000.0
  enable_eke = .true.
  eke_k_max  = 1e4
  eke_c_k    = 0.4
  eke_c_eps  = 0.5
  eke_cross  = 2.
  eke_crhin  = 1.0
  eke_lmin   = 100.0
  enable_eke_superbee_advection = .true.
  enable_eke_isopycnal_diffusion = .true.

  enable_idemix = .true.
  enable_eke_diss_surfbot = .true.
  eke_diss_surfbot_frac = 0.2 ! fraction which goes into bottom
  enable_idemix_superbee_advection = .true.

  enable_idemix_hor_diffusion = .true.; 
  !np=17+2
  !enable_idemix_M2 = .true.
  !enable_idemix_niw = .true.
  !omega_M2 =  2*pi/( 12*60*60 +  25.2 *60 )   ! M2 frequency in 1/s

  eq_of_state_type = 5
end subroutine set_parameter



subroutine set_grid
 use main_module   
 use config_module   
 implicit none
 real*4 :: dz4(nz)
 open(10,file='dz.bin',access='direct',recl=4*nz,status='old')
 read(10,rec=1) dz4
 close(10);
 dzt = dz4(nz:1:-1)
 dxt = 1.0
 dyt = 1.0
 y_Origin=-79.
 x_Origin=91.

end subroutine set_grid


subroutine set_coriolis
 use main_module   
 use config_module   
 implicit none
 integer :: j
 do j=js_pe-onx,je_pe+onx
   coriolis_t(:,j) = 2*omega*sin( yt(j)/180.*pi ) 
 enddo
end subroutine set_coriolis


subroutine set_initial_conditions
 use main_module   
 use config_module   
 use eke_module   
 use tke_module   
 use idemix_module   
 use isoneutral_module   
 implicit none
 integer :: i,j,k,kk,n
 real*4 :: dat4(nx,ny)
 real*8 :: pen(0:nz),swarg1,swarg2
 real*8 ::  rpart_shortwave  = 0.58
 real*8 ::  efold1_shortwave = 0.35
 real*8 ::  efold2_shortwave = 23.0
 include "netcdf.inc"
 integer :: iret, ncid,id

 allocate( t_star(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); t_star=0.0
 allocate( s_star(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); s_star=0.0
 allocate( qnec(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); qnec=0.0
 allocate( qnet(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); qnet=0.0
 allocate( qsol(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); qsol=0.0
 allocate( divpen_shortwave(nz) ); divpen_shortwave=0.0
 allocate( taux(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); taux=0.0
 allocate( tauy(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); tauy=0.0

 ! initital conditions
 open(10,file='lev_clim_temp.bin',access='direct',recl=4*nx*ny,status='old')
 open(20,file='lev_clim_salt.bin',access='direct',recl=4*nx*ny,status='old')
 kk=nz
 do k=1,nz
  read(10,rec=kk) dat4
  temp(is_pe:ie_pe,js_pe:je_pe,k,taum1) = dat4(is_pe:ie_pe,js_pe:je_pe)*maskT(is_pe:ie_pe,js_pe:je_pe,k)
  temp(is_pe:ie_pe,js_pe:je_pe,k,tau)   = dat4(is_pe:ie_pe,js_pe:je_pe)*maskT(is_pe:ie_pe,js_pe:je_pe,k)
  read(20,rec=kk) dat4
  salt(is_pe:ie_pe,js_pe:je_pe,k,tau)   = dat4(is_pe:ie_pe,js_pe:je_pe)*maskT(is_pe:ie_pe,js_pe:je_pe,k)
  salt(is_pe:ie_pe,js_pe:je_pe,k,taum1) = dat4(is_pe:ie_pe,js_pe:je_pe)*maskT(is_pe:ie_pe,js_pe:je_pe,k)
  kk=kk-1
 enddo
 close(10); close(20)

 ! wind stress on MIT grid
 open(10,file='ECMWFBB_taux.bin',access='direct',recl=4*nx*ny,status='old')
 open(20,file='ECMWFBB_tauy.bin',access='direct',recl=4*nx*ny,status='old')
 do n=1,12
  read(10,rec=n) dat4
  taux(is_pe:ie_pe,js_pe:je_pe,n)= dat4(is_pe:ie_pe,js_pe:je_pe)/rho_0
  read(20,rec=n) dat4
  tauy(is_pe:ie_pe,js_pe:je_pe,n)= dat4(is_pe:ie_pe,js_pe:je_pe)/rho_0
  call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,taux(:,:,n)) 
  call setcyclic_xy   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,taux(:,:,n))
  call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,tauy(:,:,n)) 
  call setcyclic_xy   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,tauy(:,:,n))
 enddo
 close(10); close(20)
 ! check for special values
 where( taux < -99.9) taux = 0
 where( tauy < -99.9) tauy = 0


 ! Qnet and dQ/dT and Qsol
 open(10,file='ECMWFBB_qnet.bin',access='direct',recl=4*nx*ny,status='old')
 open(20,file='ECMWFBB_dqdt.bin',access='direct',recl=4*nx*ny,status='old')
 open(30,file='ECMWFBB_swf.bin', access='direct',recl=4*nx*ny,status='old')
 do n=1,12
  read(10,rec=n) dat4
  qnet(is_pe:ie_pe,js_pe:je_pe,n) = -dat4(is_pe:ie_pe,js_pe:je_pe)*maskT(is_pe:ie_pe,js_pe:je_pe,nz)
  read(20,rec=n) dat4
  qnec(is_pe:ie_pe,js_pe:je_pe,n) = +dat4(is_pe:ie_pe,js_pe:je_pe)*maskT(is_pe:ie_pe,js_pe:je_pe,nz)
  read(30,rec=n) dat4
  qsol(is_pe:ie_pe,js_pe:je_pe,n) = -dat4(is_pe:ie_pe,js_pe:je_pe)*maskT(is_pe:ie_pe,js_pe:je_pe,nz)
 enddo
 close(10); close(20); close(30)


 ! SST and SSS
 open(10,file='ECMWFBB_target_sst.bin',access='direct',recl=4*nx*ny,status='old')
 open(20,file='lev_sss.bin',access='direct',recl=4*nx*ny,status='old')
 do n=1,12
  read(10,rec=n) dat4
  t_star(is_pe:ie_pe,js_pe:je_pe,n) = dat4(is_pe:ie_pe,js_pe:je_pe)*maskT(is_pe:ie_pe,js_pe:je_pe,nz)
  read(20,rec=n) dat4
  s_star(is_pe:ie_pe,js_pe:je_pe,n) = dat4(is_pe:ie_pe,js_pe:je_pe)*maskT(is_pe:ie_pe,js_pe:je_pe,nz)
 enddo
 close(10); close(20)


 if (enable_idemix ) then


  open(10,file='tidal_energy.bin',access='direct',recl=4*nx*ny,status='old')
  read(10,rec=1) dat4
  close(10)
  do j=js_pe,je_pe
   do i=is_pe,ie_pe
      k=max(1,kbot(i,j))
      dat4(i,j) = dat4(i,j)*maskW(i,j,k)/rho_0
   enddo
  enddo

  if (enable_idemix_M2) then
   do j=js_pe,je_pe
     do k=2,np-1
      forc_M2(is_pe:ie_pe,j,k) = 0.5*dat4(is_pe:ie_pe,j)/(2*pi)
     enddo
     forc_iw_bottom(is_pe:ie_pe,j) = 0.5*dat4(is_pe:ie_pe,j)
   enddo
  else
   forc_iw_bottom(is_pe:ie_pe,js_pe:je_pe) = dat4(is_pe:ie_pe,js_pe:je_pe)
  endif

  open(10,file='wind_energy.bin',access='direct',recl=4*nx*ny,status='old')
  read(10,rec=1) dat4
  close(10)
  do j=js_pe,je_pe
   do i=is_pe,ie_pe
      dat4(i,j) = dat4(i,j)*maskW(i,j,nz)/rho_0*0.2
   enddo
  enddo

  if (enable_idemix_niw) then
   do j=js_pe,je_pe
     do k=2,np-1
      forc_niw(is_pe:ie_pe,j,k) = 1.0*dat4(is_pe:ie_pe,j)/(2*pi)
     enddo
     forc_iw_surface(is_pe:ie_pe,j) = 0.0*dat4(is_pe:ie_pe,j)
   enddo
  else
   forc_iw_surface(is_pe:ie_pe,js_pe:je_pe) = dat4(is_pe:ie_pe,js_pe:je_pe)
  endif

  if (enable_idemix_niw) then
    do j=js_pe-onx,je_pe+onx
     omega_niw(j) = max(1d-8, abs( 1.05 * coriolis_t(j) ) )
    enddo
  endif

  if (enable_idemix_niw .or. enable_idemix_M2) then
   iret = nf_open('hrms_1deg.nc',NF_nowrite,ncid)
   iret = nf_inq_varid(ncid,'HRMS',id)
   iret = nf_get_vara_double(ncid,id ,(/is_pe,js_pe/),(/ie_pe-is_pe+1,je_pe-js_pe+1/),topo_hrms(is_pe:ie_pe,js_pe:je_pe))
   iret = nf_inq_varid(ncid,'LAM',id)
   iret = nf_get_vara_double(ncid,id ,(/is_pe,js_pe/),(/ie_pe-is_pe+1,je_pe-js_pe+1/),topo_lam(is_pe:ie_pe,js_pe:je_pe))
   call ncclos (ncid, iret)

   call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,topo_hrms) 
   call setcyclic_xy   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,topo_hrms)
   call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,topo_lam) 
   call setcyclic_xy   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,topo_lam)
  endif

 endif

!     Initialize penetration profile for solar radiation
!     and store divergence in divpen
!     note that pen(nz) is set 0.0 instead of 1.0 to compensate for the
!     shortwave part of the total surface flux 
  pen = 0.0
  do k=1,nz-1
     swarg1 = zw(k)/efold1_shortwave
     swarg2 = zw(k)/efold2_shortwave
     pen(k) = rpart_shortwave*exp(swarg1)+ (1.0-rpart_shortwave)*exp(swarg2)
  enddo
  do k=1,nz
     divpen_shortwave(k) = (pen(k) - pen(k-1))/dzt(k)
  enddo
end subroutine set_initial_conditions



subroutine get_periodic_interval(currentTime,cycleLength,recSpacing,nbrec,trec1,trec2,wght1,wght2)
 ! interpolation routine taken from mitgcm
 implicit none
 real*8, intent(in) :: currentTime,recSpacing,cycleLength
 integer, intent(in) :: nbrec
 real*8, intent(out) :: wght1,wght2
 integer, intent(out) :: tRec1,tRec2
 real*8 :: locTime,tmpTime
 locTime = currentTime - recSpacing*0.5 + cycleLength*( 2 - NINT(currentTime/cycleLength) )
 tmpTime = MOD( locTime, cycleLength )
 tRec1 = 1 + INT( tmpTime/recSpacing )
 tRec2 = 1 + MOD( tRec1, nbRec )
 wght2 = ( tmpTime - recSpacing*(tRec1 - 1) )/recSpacing
 wght1 = 1d0 - wght2
end subroutine



subroutine set_forcing
 use main_module   
 use config_module   
 use tke_module   
 implicit none
 integer :: i,j,k,n1,n2
 real*8 :: t_rest= 30*86400, cp_0 = 3991.86795711963  ! J/kg /K
 real*8 :: f1,f2,fxa,qqnet,qqnec,ice(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)

 fxa = 365*86400d0
 call get_periodic_interval((itt-1)*dt_tracer,fxa,fxa/12.,12,n1,n2,f1,f2)

 ! linearly interpolate wind stress and shift from MITgcm U/V grid to this grid
 do j=js_pe-onx,je_pe+onx-1
  do i=is_pe-onx,ie_pe+onx-1
   surface_taux(i,j) = f1*taux(i+1,j,n1) + f2*taux(i+1,j,n2)
   surface_tauy(i,j) = f1*tauy(i,j+1,n1) + f2*tauy(i,j+1,n2)
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

 !  W/m^2 K kg/J m^3/kg = K m/s 
 do j=js_pe,je_pe 
  do i=is_pe,ie_pe
   fxa   =  f1*t_star(i,j,n1)+f2*t_star(i,j,n2)
   qqnec =  f1*qnec(i,j,n1)+f2*qnec(i,j,n2)
   qqnet =  f1*qnet(i,j,n1)+f2*qnet(i,j,n2)
   forc_temp_surface(i,j)=(qqnet+qqnec*(fxa -temp(i,j,nz,tau)) )*maskT(i,j,nz)/cp_0/rho_0

   fxa  =  f1*s_star(i,j,n1)+f2*s_star(i,j,n2)
   forc_salt_surface(i,j)=dzt(nz)/t_rest*(fxa-salt(i,j,nz,tau))*maskT(i,j,nz)
  enddo
 enddo

 ! apply simple ice mask  
 ice = 1.0
 do j=js_pe,je_pe 
  do i=is_pe,ie_pe
   if (temp(i,j,nz,tau)*maskT(i,j,nz) <= -1.8 .and. forc_temp_surface(i,j) <=0 ) then
       forc_temp_surface(i,j) = 0.0
       forc_salt_surface(i,j) = 0.0
       ice(i,j)               = 0.0
   endif
  enddo
 enddo

 ! solar radiation
 do k=1,nz
  do j=js_pe,je_pe 
   do i=is_pe,ie_pe
    temp_source(i,j,k) =  (f1*qsol(i,j,n1)+f2*qsol(i,j,n2))*divpen_shortwave(k)*ice(i,j)*maskT(i,j,k)/cp_0/rho_0
   enddo
  enddo
 enddo
end subroutine set_forcing



subroutine set_topography
 use main_module   
 implicit none
 integer :: i,j,k,kk
 real*4 :: lev(nx,ny,nz),bathy(nx,ny)
 kbot=0
 open(10,file='bathymetry.bin',access='direct',recl=4*nx*ny,status='old')
 read(10,rec=1) bathy
 close(10)

 open(10,file='lev_clim_salt.bin',access='direct',recl=4*nx*ny,status='old')
 kk=nz
 do k=1,nz
  read(10,rec=kk) lev(:,:,k)
  kk=kk-1
 enddo
 close(10)

 do j=js_pe,je_pe
  do i=is_pe,ie_pe
   do k=nz,1,-1
    if (lev(i,j,k) .ne. 0.0 ) kbot(i,j)=k
   enddo
  enddo
  if (bathy(i,j) == 0.0) kbot(i,j) =0
 enddo
 where ( kbot == nz) kbot = 0
 !where (bathy == 0.0) kbot(1:nx,1:ny) =0

 do i=208,214
  do j=1,5
    if (i>=is_pe.and.i<=ie_pe.and.j>=js_pe.and.j<=je_pe) kbot(i,j)=0
  enddo
 enddo
 !kbot(208:214,1:5)=0
 i=105; j=135
 if (i>=is_pe.and.i<=ie_pe.and.j>=js_pe.and.j<=je_pe) kbot(i,j)=0
 !kbot(105,135)=0 ! Aleuten island
 do i=270,271
  j=131
    if (i>=is_pe.and.i<=ie_pe.and.j>=js_pe.and.j<=je_pe) kbot(i,j)=0
 enddo
 !kbot(270:271,131) = 0 ! Engl Kanal
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
 call register_average('rho','Density','kg/m^3','TTT',0d0,rho(:,:,:,tau),.true.)

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
 endif
 if (enable_TEM_friction)  then
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

 if (enable_idemix_M2)  then
   call register_average('E_M2','M2 tidal energy','m^2/s^2','TT',E_M2_int,0d0,.false.)
   call register_average('cg_M2','M2 group velocity','m/s','TT',cg_M2,0d0,.false.)
   call register_average('tau_M2','Decay scale','1/s','TT',tau_M2,0d0,.false.)
   call register_average('alpha_M2_cont','Interaction coeff.','s/m^3','TT',alpha_M2_cont,0d0,.false.)
 endif

 if (enable_idemix_niw)  then
   call register_average('E_niw','NIW energy','m^2/s^2','TT',E_niw_int,0d0,.false.)
   call register_average('cg_niw','NIW group velocity','m/s','TT',cg_niw,0d0,.false.)
   call register_average('tau_niw','Decay scale','1/s','TT',tau_niw,0d0,.false.)
 endif
end subroutine set_diagnostics

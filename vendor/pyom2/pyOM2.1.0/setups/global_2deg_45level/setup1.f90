
!=======================================================================
! global 2x2 deg model with 45 levels
!======================================================================= 


module config_module
 ! use this module only locally in this file
 implicit none
 real*8, allocatable :: t_star(:,:,:)
 real*8, allocatable :: s_star(:,:,:)
 real*8, allocatable :: qnec(:,:,:)
 real*8, allocatable :: qnet(:,:,:)
 real*8, allocatable :: qsol(:,:,:)
 real*8, allocatable :: taux(:,:,:),tauy(:,:,:)
 real*8, allocatable :: divpen_shortwave(:)
end module config_module
 

subroutine set_parameter
 ! ----------------------------------
 !       set here main parameter
 ! ----------------------------------
 use main_module   
 use config_module
 implicit none
  nx   = 128
  ny   = 64
  nz   = 45
  dt_mom    = 3600.0
  dt_tracer = 3600.0*5

  coord_degree     = .true.
  enable_cyclic_x  = .true.

  runlen = 365*86400.*50 
  enable_diag_ts_monitor = .true.; ts_monint = 365*86400./24.
  enable_diag_snapshots  = .true.; snapint  =  365*86400.*10
  enable_diag_energy     = .true.; energint  = 365*86400.; 
  energfreq = dt_tracer*10
  enable_diag_averages   = .true.; aveint  = 365*86400.*10
  avefreq = dt_tracer*10
  enable_diag_overturning  = .true.; overint  = 365*86400.
  avefreq = dt_tracer*10

  congr_epsilon = 1e-8
  congr_max_iterations = 20000
  enable_streamfunction = .true.

  !enable_hor_diffusion = .true.;  K_h = 2000.0
  enable_neutral_diffusion = .true.; 
      K_iso_0 = 1000.0
      K_iso_steep = 500.0
      iso_dslope=0.001
      iso_slopec=0.001
  enable_skew_diffusion = .true.

  enable_hor_friction = .true.; A_h = (2.8*degtom)**3*2e-11    
  enable_hor_friction_cos_scaling = .true. 
  hor_friction_cosPower = 1
  enable_tempsalt_sources = .true.

  enable_implicit_vert_friction = .true.
  enable_tke = .true.
  c_k = 0.1
  c_eps = 0.7
  alpha_tke = 30.0
  mxl_min = 1d-8
  tke_mxl_choice = 2
  enable_tke_superbee_advection = .true.

  K_gm_0 = 1000.0
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
  enable_idemix_hor_diffusion = .true.; 
  enable_eke_diss_surfbot = .true.
  eke_diss_surfbot_frac = 0.2 ! fraction which goes into bottom
  enable_idemix_superbee_advection = .true.

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

  dxt = 2.8125
  dyt = 2.8125
  y_origin = -90.0+2.8125
  x_origin = 0+2.8125
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
 implicit none
 integer :: i,j,k,n
 real*4 :: dat4(nx,ny)
 include "netcdf.inc"
 integer :: iret, ncid,id
 real*8 :: pen(0:nz),swarg1,swarg2
 real*8 ::  rpart_shortwave  = 0.58
 real*8 ::  efold1_shortwave = 0.35
 real*8 ::  efold2_shortwave = 23.0

 allocate( s_star(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); s_star=0
 allocate( t_star(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); t_star=0
 allocate( qnec(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ) ; qnec=0
 allocate( qnet(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); qnet=0
 allocate( qsol(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); qsol=0
 allocate( divpen_shortwave(nz) ); divpen_shortwave=0.0
 allocate( taux(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); taux=0
 allocate( tauy(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); tauy=0

 iret=nf_open('forcing_2deg.cdf',NF_NOWRITE,ncid)
 if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif

 iret=nf_inq_varid(ncid,'DQDT2',id)
 iret= nf_get_vara_double(ncid,id,(/is_pe,js_pe,1/), (/ie_pe-is_pe+1,je_pe-js_pe+1,12/),qnec(is_pe:ie_pe,js_pe:je_pe,1:12))
 iret=nf_inq_varid(ncid,'QNET2',id)
 iret= nf_get_vara_double(ncid,id,(/is_pe,js_pe,1/), (/ie_pe-is_pe+1,je_pe-js_pe+1,12/),qnet(is_pe:ie_pe,js_pe:je_pe,1:12))
 iret=nf_inq_varid(ncid,'SWF2',id)
 iret= nf_get_vara_double(ncid,id,(/is_pe,js_pe,1/), (/ie_pe-is_pe+1,je_pe-js_pe+1,12/),qsol(is_pe:ie_pe,js_pe:je_pe,1:12))

 iret=nf_inq_varid(ncid,'TAUX2',id)
 iret= nf_get_vara_double(ncid,id,(/is_pe,js_pe,1/), (/ie_pe-is_pe+1,je_pe-js_pe+1,12/),taux(is_pe:ie_pe,js_pe:je_pe,1:12))
 iret=nf_inq_varid(ncid,'TAUY2',id)
 iret= nf_get_vara_double(ncid,id,(/is_pe,js_pe,1/), (/ie_pe-is_pe+1,je_pe-js_pe+1,12/),tauy(is_pe:ie_pe,js_pe:je_pe,1:12))

 iret=nf_inq_varid(ncid,'SST2',id)
 iret= nf_get_vara_double(ncid,id,(/is_pe,js_pe,1/), (/ie_pe-is_pe+1,je_pe-js_pe+1,12/),t_star(is_pe:ie_pe,js_pe:je_pe,1:12))
 iret=nf_inq_varid(ncid,'SSS2',id)
 iret= nf_get_vara_double(ncid,id,(/is_pe,js_pe,1/), (/ie_pe-is_pe+1,je_pe-js_pe+1,12/),s_star(is_pe:ie_pe,js_pe:je_pe,1:12))

 iret=nf_inq_varid(ncid,'TEMP2',id)
 do k=1,nz
  iret= nf_get_vara_double(ncid,id,(/is_pe,js_pe,k,1/), (/ie_pe-is_pe+1,je_pe-js_pe+1,1,1/),temp(is_pe:ie_pe,js_pe:je_pe,k,tau))
 enddo
 temp(:,:,:,tau) = temp(:,:,:,tau)*maskT
 where( temp(:,:,:,tau) <= -1e33) temp(:,:,:,tau)=0.0
 call border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,temp(:,:,:,tau)) 
 call setcyclic_xyz   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,temp(:,:,:,tau))
 temp(:,:,:,taum1) = temp(:,:,:,tau)

 iret=nf_inq_varid(ncid,'SALT2',id)
 do k=1,nz
  iret= nf_get_vara_double(ncid,id,(/is_pe,js_pe,k,1/), (/ie_pe-is_pe+1,je_pe-js_pe+1,1,1/),salt(is_pe:ie_pe,js_pe:je_pe,k,tau))
 enddo
 salt(:,:,:,tau) = salt(:,:,:,tau)*maskT
 where( salt(:,:,:,tau) <= -1e33) salt(:,:,:,tau)=35.0
 call border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,salt(:,:,:,tau)) 
 call setcyclic_xyz   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,salt(:,:,:,tau))
 salt(:,:,:,taum1) = salt(:,:,:,tau)

 iret = nf_close (ncid)

 do n=1,12
  qnec(:,:,n)=qnec(:,:,n)*maskT(:,:,nz)
  qnet(:,:,n)=-qnet(:,:,n)*maskT(:,:,nz)
  qsol(:,:,n)=-qsol(:,:,n)*maskT(:,:,nz)
  taux(:,:,n)=taux(:,:,n)*maskU(:,:,nz)/rho_0
  tauy(:,:,n)=tauy(:,:,n)*maskV(:,:,nz)/rho_0
  t_star(:,:,n)=t_star(:,:,n)*maskT(:,:,nz)
  s_star(:,:,n)=s_star(:,:,n)*maskT(:,:,nz)
 enddo

 if (enable_idemix ) then
  open(10,file='tidal_energy.bin',access='direct',recl=4*nx*ny,status='old')
  read(10,rec=1) dat4
  close(10)
  do j=js_pe,je_pe
   do i=is_pe,ie_pe
      k=max(1,kbot(i,j))
      forc_iw_bottom(i,j) = dat4(i,j)*maskW(i,j,k)/rho_0
   enddo
  enddo
  open(10,file='wind_energy.bin',access='direct',recl=4*nx*ny,status='old')
  read(10,rec=1) dat4
  close(10)
  do j=js_pe,je_pe
   do i=is_pe,ie_pe
      forc_iw_surface(i,j) = dat4(i,j)*maskW(i,j,nz)/rho_0*0.2
   enddo
  enddo
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
 implicit none
 integer :: i,j,k,n1,n2
 real*8 :: t_rest= 30*86400, cp_0 = 3991.86795711963  ! J/kg /K
 real*8 :: f1,f2,fxa,qqnet,qqnec,ice(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)

 fxa = 365*86400d0
 call get_periodic_interval((itt-1)*dt_tracer,fxa,fxa/12.,12,n1,n2,f1,f2)

 ! linearly interpolate wind stress 
 do j=js_pe-onx,je_pe+onx-1
  do i=is_pe-onx,ie_pe+onx-1
   surface_taux(i,j) = f1*taux(i,j,n1) + f2*taux(i,j,n2)
   surface_tauy(i,j) = f1*tauy(i,j,n1) + f2*tauy(i,j,n2)
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
 include "netcdf.inc"
 integer :: iret, ncid,id, i,j,k
 real*8 :: bathy(nx,ny)

 kbot=0

 iret=nf_open('topo_2deg.cdf',NF_NOWRITE,ncid)
 if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret=nf_inq_varid(ncid,'TOPO3',id)
 iret= nf_get_vara_double(ncid,id,(/1,1/), (/nx,ny/),bathy)
 iret = nf_close (ncid)

 do j=js_pe,je_pe
  do i=is_pe,ie_pe
   if (bathy(i,j) >= 0.0) then
     kbot(i,j) = 0  
   else if (bathy(i,j) <= zw(1) ) then
    kbot(i,j) = 1  
   else
    k= minloc( (zw-bathy(i,j))**2,1)-1
    kbot(i,j) = max(1,min(nz,k))
   endif
  enddo
 enddo
end subroutine set_topography








subroutine set_diagnostics
 use main_module   
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


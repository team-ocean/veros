
!=======================================================================
! global 4x4 deg model with 15 levels
!======================================================================= 


module config_module
 ! use this module only locally in this file
 implicit none
 real*8, allocatable :: t_star(:,:,:)
 real*8, allocatable :: s_star(:,:,:)
 real*8, allocatable :: qnec(:,:,:)
 real*8, allocatable :: qnet(:,:,:)
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
  nx   = 90
  ny   = 40
  nz   = 15
  dt_mom    = 1800.0
  dt_tracer = 86400.!dt_mom

  coord_degree     = .true.
  enable_cyclic_x  = .true.

  runlen = 365*86400*100.
!  runlen = 365*86400/12.0
  enable_diag_ts_monitor = .true.; ts_monint = 365*86400./24.
  enable_diag_snapshots  = .true.; snapint  =  365*86400. /24.0
  enable_diag_overturning= .true.; overint  =  365*86400./24.0; overfreq = dt_tracer
  enable_diag_energy     = .true.; 
  energint = 365*86400./24.
  energfreq = 86400
  enable_diag_averages   = .true.
  aveint  = 365*86400.*10
  avefreq = 86400


  congr_epsilon = 1e-8
  congr_max_iterations = 20000
  enable_streamfunction = .true.
!  enable_congrad_verbose = .true.

  enable_neutral_diffusion = .true.; 
      K_iso_0 = 1000.0
      K_iso_steep = 1000.0
      iso_dslope=4./1000.0
      iso_slopec=1./1000.0
  enable_skew_diffusion = .true.

  enable_hor_friction  = .true.; A_h = (4*degtom)**3*2e-11
  enable_hor_friction_cos_scaling = .true.; hor_friction_cosPower=1
 
  enable_implicit_vert_friction = .true.
  enable_tke = .true.
  c_k = 0.1
  c_eps = 0.7
  alpha_tke = 30.0
  mxl_min = 1d-8
  tke_mxl_choice = 2
  enable_tke_superbee_advection = .true.

  enable_eke = .true.
  eke_k_max  = 1e4
  eke_c_k    = 0.4
  eke_c_eps  = 0.5
  eke_cross  = 2.
  eke_crhin  = 1.0
  eke_lmin   = 100.0
  enable_eke_superbee_advection = .true.

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
  real*8 :: ddz(nz)
  ddz = (/50.,70.,100.,140.,190.,240.,290.,340.,390.,440.,490.,540.,590.,640.,690./)
  dzt=ddz(nz:1:-1)
  dxt = 4.0
  dyt = 4.0
  y_origin = -76.0
  x_origin = 4.0
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




subroutine set_topography
 use main_module   
 implicit none
 integer :: i,j,k,kk
 real*4 :: lev_salt(nx,ny,nz),bathy(nx,ny)
 kbot=0

 open(10,file='bathymetry.bin',access='direct',recl=4*nx*ny,status='old')
 read(10,rec=1) bathy
 close(10)

 open(10,file='lev_s.bin',access='direct',recl=4*nx*ny,status='old')
 kk=nz
 do k=1,nz
  read(10,rec=kk) lev_salt(:,:,k)
  kk=kk-1
 enddo
 close(10)

 do i=is_pe,ie_pe
  do j=js_pe,je_pe
   do k=nz,1,-1
    if (lev_salt(i,j,k) .ne. 0.0 ) kbot(i,j)=k
   enddo
   if (bathy(i,j) == 0.0) kbot(i,j) =0
  enddo
 enddo
 where ( kbot == nz) kbot = 0
 !where (bathy == 0.0) kbot(1:nx,1:ny) =0
end subroutine set_topography


subroutine set_initial_conditions
 use main_module   
 use config_module   
 use eke_module   
 use tke_module   
 use idemix_module   
 implicit none
 integer :: i,j,k,kk,n
 real*4 :: dat4(nx,ny)
 include "netcdf.inc"
 integer :: iret, ncid,id
 real*8 :: fxa,fxb

 allocate( s_star(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); s_star=0
 allocate( t_star(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); t_star=0
 allocate( qnec(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ) ; qnec=0
 allocate( qnet(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); qnet=0
 allocate( taux(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); taux=0
 allocate( tauy(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); tauy=0

 iret=nf_open('ecmwf_4deg_monthly.cdf',NF_NOWRITE,ncid) ! ECMWF monthly heat flux forcing
 !iret=nf_open('ecmwf_4deg.cdf',NF_NOWRITE,ncid) ! annual mean
 if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret=nf_inq_varid(ncid,'Q3',id)
 iret= nf_get_vara_double(ncid,id,(/is_pe,js_pe,1/), (/ie_pe-is_pe+1,je_pe-js_pe+1,12/),qnec(is_pe:ie_pe,js_pe:je_pe,1:12))
 iret = nf_close (ncid)
 do n=1,12
  qnec(:,:,n)=qnec(:,:,n)*maskT(:,:,nz)
 enddo


 ! use NCEP net heat flux instead of ECMWF
 open(10,file='ncep_qnet.bin',access='direct',recl=4*nx*ny,status='old')
 do n=1,12
  read(10,rec=n) dat4
  qnet(is_pe:ie_pe,js_pe:je_pe,n) = - dat4(is_pe:ie_pe,js_pe:je_pe)*maskT(is_pe:ie_pe,js_pe:je_pe,nz)
 enddo
 close(10)

 fxa = 0.; fxb = 0.
 do n=1,12
  do i=is_pe,ie_pe
   do j=js_pe,je_pe
    fxa = fxa + qnet(i,j,n)*area_t(i,j)
    fxb = fxb + area_t(i,j)
   enddo
  enddo
 enddo

 call global_sum(fxa); call global_sum(fxb)
 if (my_pe==0) print*,' removing a heat flux imbalance of ',fxa/fxb,'W/m^2'
 do n=1,12
  qnet(:,:,n) = (qnet(:,:,n) - fxa/fxb)*maskT(:,:,nz)
 enddo

 ! use Trenberth wind stress from MITgcm instead of ECMWF (also contained in ecmwf_4deg.cdf)
 open(10,file='trenberth_taux.bin',access='direct',recl=4*nx*ny,status='old')
 open(20,file='trenberth_tauy.bin',access='direct',recl=4*nx*ny,status='old')
 do n=1,12
  read(10,rec=n) dat4
  taux(is_pe:ie_pe,js_pe:je_pe,n) = dat4(is_pe:ie_pe,js_pe:je_pe)/rho_0
  read(20,rec=n) dat4
  tauy(is_pe:ie_pe,js_pe:je_pe,n) = dat4(is_pe:ie_pe,js_pe:je_pe)/rho_0
  call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,taux(:,:,n)) 
  call setcyclic_xy   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,taux(:,:,n))
  call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,tauy(:,:,n)) 
  call setcyclic_xy   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,tauy(:,:,n))
 enddo
 close(10); close(20)
 ! check for special values
 where( taux < -99.9) taux = 0
 where( tauy < -99.9) tauy = 0

 ! initial conditions for T and S
 open(10,file='lev_t.bin',access='direct',recl=4*nx*ny,status='old')
 open(20,file='lev_s.bin',access='direct',recl=4*nx*ny,status='old')
 kk=nz
 do k=1,nz
  read(10,rec=kk) dat4
  temp(is_pe:ie_pe,js_pe:je_pe,k,tau  )= dat4(is_pe:ie_pe,js_pe:je_pe)*maskT(is_pe:ie_pe,js_pe:je_pe,k)
  temp(is_pe:ie_pe,js_pe:je_pe,k,taum1)= dat4(is_pe:ie_pe,js_pe:je_pe)*maskT(is_pe:ie_pe,js_pe:je_pe,k)
  read(20,rec=kk) dat4
  salt(is_pe:ie_pe,js_pe:je_pe,k,tau  )= dat4(is_pe:ie_pe,js_pe:je_pe)*maskT(is_pe:ie_pe,js_pe:je_pe,k)
  salt(is_pe:ie_pe,js_pe:je_pe,k,taum1)= dat4(is_pe:ie_pe,js_pe:je_pe)*maskT(is_pe:ie_pe,js_pe:je_pe,k)
  kk=kk-1
 enddo
 close(10); close(20)

 ! SST and SSS
 open(10,file='lev_sst.bin',access='direct',recl=4*nx*ny,status='old')
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
 use eke_module   
 use tke_module   
 use idemix_module   
 implicit none
 integer :: i,j,n1,n2
 real*8 :: t_rest= 30*86400, cp_0 = 3991.86795711963  ! J/kg /K
 real*8 :: fxa,qqnet,qqnec,f1,f2

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
   fxa  =  f1*t_star(i,j,n1)+f2*t_star(i,j,n2)
   qqnec =  f1*qnec(i,j,n1)+f2*qnec(i,j,n2)
   qqnet =  f1*qnet(i,j,n1)+f2*qnet(i,j,n2)
   forc_temp_surface(i,j)=(qqnet+ qqnec*(fxa -temp(i,j,nz,tau)) )*maskT(i,j,nz)/cp_0/rho_0

   fxa  =  f1*s_star(i,j,n1)+f2*s_star(i,j,n2)
   forc_salt_surface(i,j)=dzt(nz)/t_rest*(fxa-salt(i,j,nz,tau))*maskT(i,j,nz)
  enddo
 enddo

 ! apply simple ice mask  
 do j=js_pe,je_pe 
  do i=is_pe,ie_pe
   if (temp(i,j,nz,tau)*maskT(i,j,nz) <= -1.8 .and. forc_temp_surface(i,j) <=0 ) then
       forc_temp_surface(i,j) = 0.0
       forc_salt_surface(i,j) = 0.0
   endif
  enddo
 enddo

end subroutine set_forcing






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
 call register_average('kappaH','Vertical diffusivity','m^2/s','TTU',0d0,kappaH,.true.)
 if (enable_skew_diffusion)  then
   call register_average('B1_gm','Zonal component of GM streamfct.','m^2/s','TUT',0d0,B1_gm,.true.)
   call register_average('B2_gm','Meridional component of GM streamfct.','m^2/s','UTT',0d0,B2_gm,.true.)
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
 endif
 if (enable_eke)  then
  call register_average('EKE','Eddy energy','m^2/s^2','TTU',0d0,eke(:,:,:,tau),.true.)
  call register_average('K_gm','Lateral diffusivity','m^2/s','TTU',0d0,K_gm,.true.)
  call register_average('eke_diss_tke','Eddy energy dissipation','m^2/s^3','TTU',0d0,eke_diss_tke,.true.)
  call register_average('eke_diss_iw','Eddy energy dissipation','m^2/s^3','TTU',0d0,eke_diss_iw,.true.)
 endif

end subroutine set_diagnostics


subroutine set_particles
end subroutine set_particles


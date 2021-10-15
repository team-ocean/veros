
!=======================================================================
!  FLAME 4/3 deg model
!======================================================================= 



module config_module
 ! use this module only locally in this file
 implicit none
 real*8, allocatable :: t_clim(:,:,:)
 real*8, allocatable :: s_clim(:,:,:)
 real*8, allocatable :: t_rest(:,:,:)
 real*8, allocatable :: s_rest(:,:,:)
 real*8, allocatable :: taux(:,:,:),tauy(:,:,:)
 real*8, allocatable :: t_star(:,:,:,:),s_star(:,:,:,:),tscl(:,:,:)
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
  nx   = 87
  ny   = 89
  nz   = 45
  dt_mom    = 3600.0
  dt_tracer = 3600.0

  coord_degree     = .true.
  runlen = 365.*86400 *5

  enable_diag_ts_monitor = .true.; ts_monint = 86400.0
  enable_diag_snapshots  = .true.; snapint  =  365*86400. /20.0
  enable_diag_energy     = .true.; energint = dt_tracer !365*86400./12.0
  energfreq = dt_tracer
  !enable_diag_averages   = .true.
  !aveint  = 365*86400. /12.
  !avefreq = 86400.0

  congr_epsilon = 1e-6
  congr_max_iterations = 20000
  enable_streamfunction = .true.
  !enable_congrad_verbose = .true.

!  enable_hor_diffusion = .true.; K_h = 2000.0
  enable_neutral_diffusion = .true.; 
      K_iso_0 = 1000.0
      K_iso_steep = 200.0
      iso_dslope=1./1000.0
      iso_slopec=4./1000.0
  enable_skew_diffusion = .true.

  enable_hor_friction = .true.; A_h = 5e4; 
  enable_hor_friction_cos_scaling = .true.; hor_friction_cosPower = 3
  enable_tempsalt_sources = .true.

  !kappaH_0 = 1e-4; kappaM_0 = 10e-4
  enable_implicit_vert_friction = .true.
  enable_tke = .true.
  c_k = 0.1
  c_eps = 0.7
  alpha_tke = 30.0
  mxl_min = 1d-8
  tke_mxl_choice = 2

  K_gm_0 = 1000.0
  !enable_eke = .true.

  enable_idemix = .true.
  enable_idemix_hor_diffusion = .true.; 

  eq_of_state_type = 5
end subroutine set_parameter



subroutine set_grid
 use main_module   
 use config_module   
 implicit none
 include "netcdf.inc"
 integer :: iret,id,ncid

 iret=nf_open('forcing.cdf',NF_NOWRITE,ncid)
 if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif

 !if (n_pes>1) then
 !  call halt_stop(' cannot read grid')
 !endif
 
 iret=nf_inq_varid(ncid,'dxtdeg',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 !iret= nf_get_vara_double(ncid,id,1, nx, dxt(1:nx) )
 !dxt(1-onx:0)=dxt(1); dxt(nx+1:nx+onx)=dxt(nx)
 iret= nf_get_vara_double(ncid,id,is_pe, ie_pe-is_pe+1, dxt(is_pe:ie_pe) )

 iret=nf_inq_varid(ncid,'dytdeg',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 !iret= nf_get_vara_double(ncid,id,1, ny, dyt(1:ny) )
 !dyt(1-onx:0)=dyt(1); dyt(ny+1:ny+onx)=dyt(ny)
 iret= nf_get_vara_double(ncid,id,js_pe, je_pe-js_pe+1, dyt(js_pe:je_pe) )

 iret=nf_inq_varid(ncid,'dzt',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret= nf_get_vara_double(ncid,id,1,nz, dzt )
 dzt(1:nz) = dzt(nz:1:-1)/100.0

 iret=nf_inq_varid(ncid,'xu',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret= nf_get_vara_double(ncid,id,1,1, x_origin )

 iret=nf_inq_varid(ncid,'yu',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret= nf_get_vara_double(ncid,id,1,1, y_origin )

 iret = nf_close (ncid)

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
 include "netcdf.inc"
 integer :: iret,id,ncid
 integer :: kmt(nx,ny),i,j

 iret=nf_open('forcing.cdf',NF_NOWRITE,ncid)
 if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret=nf_inq_varid(ncid,'kmt',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret= nf_get_vara_int(ncid,id,(/1,1/),(/nx,ny/), kmt )
 iret = nf_close (ncid)

 kbot=1
 do j=js_pe,je_pe
   do i=is_pe,ie_pe
    kbot(i,j)=min(nz,nz-kmt(i,j)+1)
    if ( kmt(i,j) == 0) kbot(i,j)=0
   enddo
 enddo

end subroutine set_topography





subroutine set_initial_conditions
 use main_module   
 use config_module   
 use eke_module   
 use tke_module   
 use idemix_module   
 use isoneutral_module   
 implicit none
 include "netcdf.inc"
 integer :: iret,id,ncid
 integer :: i,j,k,n
 real*8 :: dat8(nx,ny,12)
 real*4 :: dat4(nx,ny)

 allocate( t_clim(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); t_clim=0.0
 allocate( s_clim(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); s_clim=0.0
 allocate( t_rest(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); t_rest=0.0
 allocate( s_rest(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); s_rest=0.0
 allocate( taux(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); taux=0.0
 allocate( tauy(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); tauy=0.0
 allocate( t_star(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,12) ); t_star=0.0
 allocate( s_star(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,12) ); s_star=0.0
 allocate( tscl(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); tscl=0.0

 ! initial conditions
 iret=nf_open('forcing.cdf',NF_NOWRITE,ncid)
 if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif

 iret=nf_inq_varid(ncid,'temp_ic',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret= nf_get_vara_double(ncid,id,(/is_pe,js_pe,1/),(/ie_pe-is_pe+1,je_pe-js_pe+1,nz/), temp(is_pe:ie_pe,js_pe:je_pe,1:nz,1) )
 if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif

 iret=nf_inq_varid(ncid,'salt_ic',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret= nf_get_vara_double(ncid,id,(/is_pe,js_pe,1/),(/ie_pe-is_pe+1,je_pe-js_pe+1,nz/), salt(is_pe:ie_pe,js_pe:je_pe,1:nz,1) )
 if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif

 temp(:,:,1:nz,1) = temp(:,:,nz:1:-1,1)
 salt(:,:,1:nz,1) = salt(:,:,nz:1:-1,1)

 do k=1,nz
   do j=js_pe,je_pe
    do i=is_pe,ie_pe
      temp(i,j,k,1:3) = temp(i,j,k,1)*maskT(i,j,k)
      salt(i,j,k,1:3) = (salt(i,j,k,1)*1000+35)*maskT(i,j,k)
    enddo
   enddo
 enddo

 ! wind stress
 iret=nf_inq_varid(ncid,'taux',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret=nf_get_vara_double(ncid,id,(/1,1,1/),(/nx,ny,12/), dat8 )
 where (dat8 <= -1e20 ) dat8 = 0
 do n=1,12
  taux(is_pe:ie_pe,js_pe:je_pe,n) = dat8(is_pe:ie_pe,js_pe:je_pe,n)/10/rho_0*maskZ(is_pe:ie_pe,js_pe:je_pe,nz)
  call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,taux(:,:,n)); 
  call setcyclic_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,taux(:,:,n))
 enddo

 iret=nf_inq_varid(ncid,'tauy',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret= nf_get_vara_double(ncid,id,(/1,1,1/),(/nx,ny,12/), dat8 )
 where (dat8 <= -1e20 ) dat8 = 0
 do n=1,12
  tauy(is_pe:ie_pe,js_pe:je_pe,n) = dat8(is_pe:ie_pe,js_pe:je_pe,n)/10/rho_0*maskZ(is_pe:ie_pe,js_pe:je_pe,nz)
  call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,tauy(:,:,n)); 
  call setcyclic_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,tauy(:,:,n))
 enddo

 ! heat flux and salinity restoring
 iret=nf_inq_varid(ncid,'sst_clim',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret= nf_get_vara_double(ncid,id,(/1,1,1/),(/nx,ny,12/), dat8 )
 where (dat8 <= -1e20 ) dat8 = 0
 t_clim(is_pe:ie_pe,js_pe:je_pe,:) = dat8(is_pe:ie_pe,js_pe:je_pe,:)

 iret=nf_inq_varid(ncid,'sst_rest',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret= nf_get_vara_double(ncid,id,(/1,1,1/),(/nx,ny,12/), dat8 )
 where (dat8 <= -1e20 ) dat8 = 0
 t_rest(is_pe:ie_pe,js_pe:je_pe,:) = dat8(is_pe:ie_pe,js_pe:je_pe,:)*41868.

 iret=nf_inq_varid(ncid,'sss_clim',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret= nf_get_vara_double(ncid,id,(/1,1,1/),(/nx,ny,12/), dat8 )
 where (dat8 <= -1e20 ) dat8 = 0
 s_clim(is_pe:ie_pe,js_pe:je_pe,:) = dat8(is_pe:ie_pe,js_pe:je_pe,:)*1000+35

 iret=nf_inq_varid(ncid,'sss_rest',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret= nf_get_vara_double(ncid,id,(/1,1,1/),(/nx,ny,12/), dat8 )
 where (dat8 <= -1e20 ) dat8 = 0
 s_rest(is_pe:ie_pe,js_pe:je_pe,:) = dat8(is_pe:ie_pe,js_pe:je_pe,:)/100.0

 iret = nf_close (ncid)

 ! Restoring zone

 iret=nf_open('restoring_zone.cdf',NF_NOWRITE,ncid)
 if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif

 iret=nf_inq_varid(ncid,'t_star',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret= nf_get_vara_double(ncid,id,(/is_pe,js_pe,1,1/),(/i_blk,j_blk,nz,12/), t_star(is_pe:ie_pe,js_pe:je_pe,:,:) )

 iret=nf_inq_varid(ncid,'s_star',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret= nf_get_vara_double(ncid,id,(/is_pe,js_pe,1,1/),(/i_blk,j_blk,nz,12/), s_star(is_pe:ie_pe,js_pe:je_pe,:,:) )

 iret=nf_inq_varid(ncid,'tscl',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret= nf_get_vara_double(ncid,id,(/is_pe,js_pe,1,1/),(/i_blk,j_blk,nz,1/), tscl(is_pe:ie_pe,js_pe:je_pe,:) )

 iret = nf_close (ncid)

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
 use isoneutral_module   
 implicit none
 integer :: i,j,n1,n2
 real*8 :: t1,t2,f1,f2,fxa,cp_0 = 3991.86795711963  ! J/kg /K
 !   q(Tstar-T),  q     (W/m^2/K)   q T/cp0 /rho0  ( W/m^2  K kg /J  m^3/kg = K m /s  )

 fxa = 365*86400d0
 call get_periodic_interval((itt-1)*dt_tracer,fxa,fxa/12.,12,n1,n2,f1,f2)

 do j=js_pe-1,je_pe+1
  do i=is_pe-1,ie_pe+1
    t1 = (taux(i,j-1,n1)*maskZ(i,j-1,nz)+taux(i,j,n1)*maskZ(i,j,nz)) / (maskZ(i,j,nz)+maskZ(i,j-1,nz)+1d-20)
    t2 = (taux(i,j-1,n2)*maskZ(i,j-1,nz)+taux(i,j,n2)*maskZ(i,j,nz)) / (maskZ(i,j,nz)+maskZ(i,j-1,nz)+1d-20)
    surface_taux(i,j)=(f1*t1+f2*t2)*maskU(i,j,nz)
    t1 = (tauy(i-1,j,n1)*maskZ(i-1,j,nz)+tauy(i,j,n1)*maskZ(i,j,nz)) / (maskZ(i,j,nz)+maskZ(i-1,j,nz)+1d-20)
    t2 = (tauy(i-1,j,n2)*maskZ(i-1,j,nz)+tauy(i,j,n2)*maskZ(i,j,nz)) / (maskZ(i,j,nz)+maskZ(i-1,j,nz)+1d-20)
    surface_tauy(i,j)=( f1*t1 +f2*t2 )*maskV(i,j,nz)
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

 do j=js_pe,je_pe 
  do i=is_pe,ie_pe
   forc_temp_surface(i,j)=(f1*t_rest(i,j,n1)+f2*t_rest(i,j,n2))* &
                          (f1*t_clim(i,j,n1)+f2*t_clim(i,j,n2)-temp(i,j,nz,tau))*maskT(i,j,nz) /cp_0/rho_0
   forc_salt_surface(i,j)=(f1*s_rest(i,j,n1)+f2*s_rest(i,j,n2))* &
                          (f1*s_clim(i,j,n1)+f2*s_clim(i,j,n2)-salt(i,j,nz,tau))*maskT(i,j,nz)
     if (temp(i,j,nz,tau)*maskT(i,j,nz) <= -1.8 .and. forc_temp_surface(i,j) <=0 ) then ! apply simple ice mask
       forc_temp_surface(i,j) = 0.0
       forc_salt_surface(i,j) = 0.0
     endif
  enddo
 enddo

 if (enable_tempsalt_sources) then
  do j=js_pe,je_pe 
   do i=is_pe,ie_pe
      temp_source(i,j,:) = maskT(i,j,:)*tscl(i,j,:)*(f1*t_star(i,j,:,n1)+f2*t_star(i,j,:,n2) - temp(i,j,:,tau) )
      salt_source(i,j,:) = maskT(i,j,:)*tscl(i,j,:)*(f1*s_star(i,j,:,n1)+f2*s_star(i,j,:,n2) - salt(i,j,:,tau) )
   enddo
  enddo
 endif

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







!=======================================================================
!  idealised Southern Ocean, same as in Viebahn and Eden (2010) Ocean modeling
!======================================================================= 



subroutine set_parameter
 ! ----------------------------------
 !       set here main parameter
 ! ----------------------------------
 use main_module   
 use tke_module   
 use idemix_module   
 use isoneutral_module   
 use diagnostics_module   
 implicit none
  nx   = 5; nz   = 18; ny   = 30
  dt_mom    = 4800/2.
  dt_tracer    = dt_mom!86400.0/2.0

  !coord_degree     = .true.
  enable_cyclic_x  = .true.

  runlen = 365*86400.*200   ! was  one year in original model
  snapint =  10.*86400.     ! was 3 days in original model

  congr_epsilon = 1e-12
  congr_max_iterations = 5000

  enable_hor_diffusion = .true.; K_h = 1000.0
  enable_hor_friction = .true.; A_h = 1**2*5e4     ! 1/T = A_1 /dx^2 = A_2 /(f dx)^2
  enable_bottom_friction = .true.; r_bot = 1e-5
  !kappaH_0 = 1.0e-4; kappaM_0=10.e-4
  K_gm_0 = 1000
  enable_implicit_vert_friction = .true.; 
  enable_tke = .true.
  enable_idemix = .true.
  eq_of_state_type = 3 ! simple non-linear with compressibility
end subroutine set_parameter



subroutine set_grid
  use main_module   
  implicit none
  dxt = 80e3*1; dyt = 80e3*1
  if (coord_degree) then
    dxt = dxt/degtom
    dyt = dyt/degtom
    x_origin= 0.0
    y_origin= -20.0
  endif
  dzt = 50.0 
end subroutine set_grid


subroutine set_coriolis
 use main_module   
 implicit none
 real*8 :: phi0,betaloc
 integer :: j 
 if (coord_degree) then
   do j=js_pe-onx,je_pe+onx
     coriolis_t(:,j) = 2*omega*sin( yt(j)/180.*pi ) 
   enddo
 else 
  phi0 =- 25.0 /180. *pi
  betaloc = 2*omega*cos(phi0)/radius
  do j=js_pe-onx,je_pe+onx
    coriolis_t(:,j) = 2*omega*sin(phi0) +betaloc*yt(j)
  enddo
 endif
end subroutine set_coriolis


subroutine set_initial_conditions
 use main_module   
 use tke_module   
 use idemix_module   
 implicit none
 integer :: i,j,k
 real*8 :: L_y

 do k=1,nz
  do j=js_pe-onx,je_pe+onx
   do i=is_pe-onx,ie_pe+onx
     !salt(i,j,k,:) = 34+(1-zt(k)/zw(1))*2
    salt(i,j,k,:) = 35.0
    temp(i,j,k,:) = (1-zt(k)/zw(1))*15
   enddo
  enddo
 enddo

 L_y = yt(ny)-yt(1)
 do j=js_pe-onx,je_pe+onx
   do i=is_pe-onx,ie_pe+onx
     if ( (yt(j)-yu(1))<L_y/2.0) then
         surface_taux(i,j) = .1e-3*sin(2*pi*(yu(j)-yu(1))/L_y)*maskU(i,j,nz)
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
  do j=js_pe-onx,je_pe+onx
   do i=is_pe-onx,ie_pe+onx
      forc_iw_bottom = 1e-6
   enddo
  enddo
 endif

end subroutine set_initial_conditions



function t_star(j)
 use main_module   
 implicit none
 integer :: j
 real*8 :: t_star,y2,L_y
 L_y = yt(ny )-yt(1)
 t_star=15
 if (yt(j)<L_y/2.0) t_star=15*yt(j)/(L_y/2.0)
 y2=L_y*0.75
 if (yt(j)>y2) t_star= 15*(1-(yt(j)-y2)/(yt(ny )-y2) )
end function t_star



function s_star(j)
 use main_module   
 implicit none
 integer :: j
 real*8 :: s_star,y2,L_y
 L_y = yt(ny )-yt(1)
 s_star=36
 if (yt(j)<L_y/2.0) s_star=34+2*yt(j)/(L_y/2.0)
 y2=L_y*0.75
 if (yt(j)>y2) s_star= 34+2*(1-(yt(j)-y2)/(yt(ny )-y2) )
end function s_star




subroutine set_forcing
 use main_module   
 implicit none
 integer :: i,j
 real*8 :: t_rest,t_star,get_rho


 t_rest=30*86400
 do j=js_pe-onx,je_pe+onx
  do i=is_pe-onx,ie_pe+onx
    forc_temp_surface(i,j)=dzt(nz)/t_rest*(t_star(j)-temp(i,j,nz,tau)) 
    forc_rho_surface(i,j)=dzt(nz)/t_rest*(get_rho(35D0,t_star(j),0D0)-rho(i,j,nz,tau)) 
    !forc_salt_surface(i,j)=dzt(nz)/t_rest*(s_star(j)-salt(i,j,nz,tau)) 
  enddo
 enddo
end subroutine set_forcing



subroutine set_topography
 use main_module   
 implicit none
 integer :: i,j
 real*8 :: L_x,L_y
 kbot=0
 L_x = xt(nx )-xt(1)
 L_y = yt(ny )-yt(1)
 do i=is_pe,ie_pe
   do j=je_pe,je_pe
     if (( (yt(j)-yu(1))<L_y/2.0).or.((xt(i)-xu(1))<L_x*.75.and. (xt(i)-xu(1))>L_x*.25))  kbot(i,j)=1
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

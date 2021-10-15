

subroutine set_spectral_parameter
!=======================================================================
! calculate spectral parameter for low mode wave propagation
!=======================================================================
 use main_module   
 implicit none
 real*8 :: fxa

  call calc_wave_speed
  call group_velocity(fxa)
  call set_time_scales
  call calc_vertical_struct_fct

  !if (enable_diag_ts_monitor .and.  modulo(itt*dt_tracer,ts_monint) < dt_tracer ) then
    call global_max(fxa)
    !if (my_pe==0) print'(a,f12.8)',' max. low mode CFL number = ',fxa
  !endif
  if (fxa > 0.6 .and. my_pe ==0) print'(a,f12.8)',' WARNING: low mode CFL number =',fxa
end subroutine set_spectral_parameter



subroutine calc_wave_speed
!=======================================================================
! calculate barolinic wave speed 
!=======================================================================
  use main_module   
  use idemix_module   
  implicit none
  integer :: i,j,k

  cn=0.0 ! calculate cn = int_(-h)^0 N/pi dz 
  do k=1,nz   
   do j=js_pe-onx,je_pe+onx
    do i=is_pe-onx,ie_pe+onx
      cn(i,j)=cn(i,j)+sqrt(max(0d0,Nsqr(i,j,k,tau)))*dzt(k)*maskT(i,j,k)/pi
    enddo
   enddo
  enddo
end subroutine calc_wave_speed



subroutine get_shelf
  use main_module   
  use idemix_module   
  implicit none
  integer :: i,j,k
  real*8 :: map2(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx),fxa
  topo_shelf=0; where( ht==0.0 ) topo_shelf=1

  fxa = 0
  if (js_pe>= ny/2 .and. je_pe <= ny/2) fxa = dyt(ny/2)
  call global_max(fxa)

  do k=1,max(1,int( 300e3/fxa ))
     map2=topo_shelf
     do j=js_pe,je_pe
      do i=is_pe,ie_pe
        if (map2(i,j)==1) topo_shelf(i-1:i+1,j-1:j+1) =1
      enddo   
     enddo
     call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,topo_shelf) 
     call setcyclic_xy   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,topo_shelf)
  enddo
end subroutine get_shelf


subroutine set_time_scales
!=======================================================================
! set decay and interaction time scales
!=======================================================================
  use main_module   
  use idemix_module   
  implicit none
  integer :: i,j
  real*8 :: N0,fxc,fxb
  real*8 :: mstar=0.01,M2_f=2*pi/(12.42*60*60)

  if (enable_idemix_niw) then
   do j=js_pe,je_pe
    do i=is_pe,ie_pe
      if (ht(i,j)>0) then
        N0=cn(i,j)*pi/ht(i,j)
        if (  N0> abs(coriolis_t(i,j)) .and. omega_niw(i,j)> abs(coriolis_t(i,j))  ) then
         fxc =  topo_hrms(i,j)**2* 2*pi/(1d-12+topo_lam(i,j) ) ! Goff
         fxb = 0.5* N0*( (omega_niw(i,j)**2+coriolis_t(i,j)**2)/omega_niw(i,j)**2 )**2  &
                        *(omega_niw(i,j)**2-coriolis_t(i,j)**2)**0.5/omega_niw(i,j)
         tau_niw(i,j) = min(0.5/dt_tracer, fxb*fxc/ht(i,j)  ) 
        endif
       endif
    enddo
   enddo
   where (topo_shelf == 1.0) tau_niw = 1./(3.*86400)
   tau_niw = max(1d0/(50.*86400),  tau_niw )*maskT(:,:,nz)
  endif

  if (enable_idemix_M2) then
   do j=js_pe,je_pe
    do i=is_pe,ie_pe
      if (ht(i,j)>0) then
        N0=cn(i,j)*pi/ht(i,j)
        if (  N0> abs(coriolis_t(i,j)) .and. omega_M2> abs(coriolis_t(i,j))  ) then
         fxc =  topo_hrms(i,j)**2* 2*pi/(1d-12+topo_lam(i,j) ) ! Goff
         fxb = 0.5* N0*( (omega_M2**2+coriolis_t(i,j)**2)/omega_M2**2 )**2  &
                        *(omega_M2**2-coriolis_t(i,j)**2)**0.5/omega_M2
         tau_m2(i,j) = min(0.5/dt_tracer, fxc*fxb/ht(i,j)  ) 
        endif
      endif
    enddo
    where (topo_shelf == 1.0) tau_m2  = 1./(3.*86400)
    tau_m2 = max(1d0/(50.*86400),  tau_m2 )*maskT(:,:,nz)
   enddo

   alpha_M2_cont = 0.0
   do i=is_pe,ie_pe
    do j=js_pe,je_pe
     if (ht(i,j) > 0.) then 
      N0 = cn(i,j)*pi/ht(i,j)+1D-20
      if (abs(yt(j)) < 28.5 ) then
       !    lambda+/M2=15*E*mstar/N * (sin(phi-28.5)/sin(28.5))^1/2
       alpha_M2_cont(i,j)=alpha_M2_cont(i,j)+ M2_f*15*mstar/N0* (sin( abs(abs(yt(j)) -28.5)/180.*pi )/sin(28.5/180.*pi) )**0.5
      endif
      if (abs(yt(j)) < 74.5 ) then
       !         !lambda-/M2 =  0.7*E*mstar/N *sin^2(phi)
       alpha_M2_cont(i,j)=alpha_M2_cont(i,j)+ M2_f*0.7*mstar/N0* sin( abs(yt(j))/180.*pi )**2
      endif
      alpha_M2_cont(i,j)=alpha_M2_cont(i,j)/ht(i,j)
     endif
    enddo
   enddo
   alpha_M2_cont = max(0D0, min( 1d-5,alpha_M2_cont ))*maskT(:,:,nz)
  endif
end subroutine set_time_scales




subroutine group_velocity(cfl)
!=======================================================================
! calculate (modulus of) group velocity of long gravity waves and change of wavenumber angle phi
!=======================================================================
  use main_module   
  use idemix_module   
  implicit none
  real*8, intent(out) :: cfl
  integer :: i,j,k
  real*8 :: gradx(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx),fxa
  real*8 :: grady(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)

  if (enable_idemix_M2) omega_M2 =  2*pi/( 12.*60*60 +  25.2 *60 )   ! M2 frequency in 1/s

  if (enable_idemix_niw) then
     omega_niw = max(1D-8, abs( 1.05 * coriolis_t ) )
  endif

  if (enable_idemix_M2) then
   cg_M2=sqrt( max(0d0,omega_M2**2 - coriolis_t**2 )  )*cn/omega_M2
  endif

  if (enable_idemix_niw) then
   cg_niw=sqrt(  max(0d0,omega_niw**2 - coriolis_t**2 )  )*cn/omega_niw
  endif

  grady = 0.0
  do j=js_pe,je_pe
   grady(:,j) = ( coriolis_t(:,j+1)-coriolis_t(:,j-1) )/(dyu(j)+dyu(j-1) )
  enddo

  if (enable_idemix_M2) then
   do j=js_pe,je_pe
    do i=is_pe,ie_pe
     fxa = max(1d-10,omega_M2**2 - coriolis_t(i,j)**2 )
     kdot_y_M2(i,j)= -cn(i,j)/sqrt(fxa)*coriolis_t(i,j)/omega_M2*grady(i,j)
    enddo
   enddo
  endif

  if (enable_idemix_niw) then
   do j=js_pe,je_pe
    do i=is_pe,ie_pe
     fxa = max(1d-10,omega_niw(i,j)**2 - coriolis_t(i,j)**2 )
     kdot_y_niw(i,j)= -cn(i,j)/sqrt(fxa)*coriolis_t(i,j)/omega_niw(i,j)*grady(i,j)
    enddo
   enddo
  endif

  grady = 0.0
  do j=js_pe,je_pe
   grady(:,j) = 0.5*(cn(:,j+1)-cn(:,j  ))/dyu(j  )*maskTp(:,j  ,1)*maskTp(:,j+1,1) &
              + 0.5*(cn(:,j  )-cn(:,j-1))/dyu(j-1)*maskTp(:,j-1,1)*maskTp(:,j  ,1)
  enddo

  if (enable_idemix_M2) then
   do j=js_pe,je_pe
    do i=is_pe,ie_pe
     fxa = max(0d0,omega_M2**2 - coriolis_t(i,j)**2 )
     kdot_y_M2(i,j)= kdot_y_M2(i,j)-sqrt(fxa)/omega_M2*grady(i,j)
    enddo
   enddo
  endif

  if (enable_idemix_niw) then
   do j=js_pe,je_pe
    do i=is_pe,ie_pe
     fxa = max(0d0,omega_niw(i,j)**2 - coriolis_t(i,j)**2 )
     kdot_y_niw(i,j)= kdot_y_niw(i,j)-sqrt(fxa)/omega_niw(i,j)*grady(i,j)
    enddo
   enddo
  endif

  gradx = 0.0
  do j=js_pe,je_pe
   do i=is_pe,ie_pe
    gradx(i,j) = 0.5*(cn(i+1,j)-cn(i  ,j))/(dxu(i  )*cost(j))*maskTp(i,j  ,1)*maskTp(i+1,j,1) &
               + 0.5*(cn(i,j  )-cn(i-1,j))/(dxu(i-1)*cost(j))*maskTp(i-1,j,1)*maskTp(i  ,j,1)
   enddo
  enddo

  if (enable_idemix_M2) then
   do j=js_pe,je_pe
    do i=is_pe,ie_pe
     fxa = max(0d0,omega_M2**2 - coriolis_t(i,j)**2 )
     kdot_x_M2(i,j)= sqrt(fxa)/omega_M2*gradx(i,j)
    enddo
   enddo
  endif

  if (enable_idemix_niw) then
   do j=js_pe,je_pe
    do i=is_pe,ie_pe
     fxa = max(0d0,omega_niw(i,j)**2 - coriolis_t(i,j)**2 )
     kdot_x_niw(i,j)= sqrt(fxa)/omega_niw(i,j)*gradx(i,j)
    enddo
   enddo
  endif

  if (enable_idemix_M2) then

   do k=2,np-1
    do j=js_pe,je_pe
     do i=is_pe-1,ie_pe
      u_M2(i,j,k) = 0.5*(cg_M2(i+1,j)+cg_M2(i,j))*cos( phit(k) ) *maskUp(i,j,k)
     enddo
    enddo
   enddo
   do k=2,np-1
    do j=js_pe-1,je_pe
     do i=is_pe,ie_pe
     v_M2(i,j,k) = 0.5*(cg_M2(i,j)+cg_M2(i,j+1))*sin( phit(k) ) *cosu(j) *maskVp(i,j,k)
     enddo
    enddo
   enddo
   do k=1,np-1
    do j=js_pe,je_pe
     do i=is_pe,ie_pe
     w_M2(i,j,k) =  (kdot_y_M2(i,j)*cos(phiu(k)) + kdot_x_M2(i,j)*sin(phiu(k)) )*maskWp(i,j,k)
     enddo
    enddo
   enddo
  endif

  if (enable_idemix_niw) then
   do k=2,np-1
    do j=js_pe,je_pe
     do i=is_pe-1,ie_pe
       u_niw(i,j,k) = 0.5*(cg_niw(i+1,j)+cg_niw(i,j))*cos( phit(k) )  *maskUp(i,j,k)
     enddo
    enddo
   enddo
   do k=2,np-1
    do j=js_pe-1,je_pe
     do i=is_pe,ie_pe
      v_niw(i,j,k) = 0.5*(cg_niw(i,j)+cg_niw(i,j+1))*sin( phit(k) ) *cosu(j)*maskVp(i,j,k)
     enddo
    enddo
   enddo
   do k=1,np-1
    do j=js_pe,je_pe
     do i=is_pe,ie_pe
      w_niw(i,j,k) =  (kdot_y_niw(i,j)*cos(phiu(k)) + kdot_x_niw(i,j)*sin(phiu(k))) *maskWp(i,j,k)
     enddo
    enddo
   enddo
  endif

  cfl = 0.0
  if (enable_idemix_M2) then
    do j=js_pe,je_pe
       do i=is_pe,ie_pe
         cfl = max( cfl, 0.5*(cg_M2(i,j)+cg_M2(i+1,j))*dt_tracer/(cost(j)*dxt(i)) )
         cfl = max( cfl, 0.5*(cg_M2(i,j)+cg_M2(i,j+1))*dt_tracer/(dyt(j)) )
         cfl = max( cfl,  kdot_y_M2(i,j)*dt_tracer/dphit(1) )
         cfl = max( cfl,  kdot_x_M2(i,j)*dt_tracer/dphit(1) )
         !if (cfl>0.5) print*,' WARNING: CFL =',cfl,' at i=',i,' j=',j
       enddo
    enddo
  endif
  if (enable_idemix_niw) then
    do j=js_pe,je_pe
       do i=is_pe,ie_pe
         cfl = max( cfl, 0.5*(cg_niw(i,j)+cg_niw(i+1,j))*dt_tracer/(cost(j)*dxt(i)) )
         cfl = max( cfl, 0.5*(cg_niw(i,j)+cg_niw(i,j+1))*dt_tracer/(dyt(j)) )
         cfl = max( cfl,  kdot_y_niw(i,j)*dt_tracer/dphit(1) )
         cfl = max( cfl,  kdot_x_niw(i,j)*dt_tracer/dphit(1) )
       enddo
    enddo
  endif
end subroutine group_velocity






subroutine calc_vertical_struct_fct
!=======================================================================
! calculate vertical structure function for low modes
!=======================================================================
  use main_module   
  use idemix_module   
  implicit none
  integer :: i,j,k,km1
  real*8 :: fxa
  real*8 :: norm(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
  real*8 :: Nsqr_lim(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
  real*8, parameter :: small = 1d-12

  Nsqr_lim = max(small,Nsqr(:,:,:,tau))

  ! calculate int_(-h)^z N dz
  phin=0
  do k=1,nz   
   km1 = max(1,k-1)
   do j=js_pe-onx,je_pe+onx
    do i=is_pe-onx,ie_pe+onx
     !fxa = (Nsqr_lim(i,j,k)*maskT(i,j,k)+Nsqr_lim(i,j,km1)*maskT(i,j,km1))/(maskT(i,j,k)+maskT(i,j,km1)+1d-22)
     fxa = Nsqr_lim(i,j,k)*maskW(i,j,k)
     phin(i,j,k) = phin(i,j,km1)*maskT(i,j,km1)+sqrt(fxa)*dzw(km1)!*maskT(i,j,k) 
    enddo
   enddo
  enddo

  ! calculate phi_n    =    cos( int_(-h)^z N/c_n dz )*N^0.5
  !    and   dphi_n/dz =    sin( int_(-h)^z N/c_n dz )/N^0.5
  do k=1,nz   
   do j=js_pe-onx,je_pe+onx
    do i=is_pe-onx,ie_pe+onx
     fxa = phin(i,j,k)/(small+cn(i,j))
     phinz(i,j,k) =  sin(fxa)/Nsqr_lim(i,j,k)**0.25 
     phin(i,j,k)  =  cos(fxa)*Nsqr_lim(i,j,k)**0.25
    enddo
   enddo
  enddo

  ! normalisation with int_(-h)^0 dz (dphi_n/dz )^2 /N^2  = 1
  norm=0
  !do k=1,nz   
    !norm = norm+ phinz(:,:,k)**2/Nsqr_lim(:,:,k)*dzt(k)*maskT(:,:,k)
  do k=1,nz-1
    norm = norm+ phinz(:,:,k)**2/Nsqr_lim(:,:,k)*dzw(k)*maskW(:,:,k)
  enddo
  k=nz;  norm = norm+ phinz(:,:,k)**2/Nsqr_lim(:,:,k)*0.5*dzw(k)*maskW(:,:,k)
  do k=1,nz   
    where( norm>0D0) phinz(:,:,k) = phinz(:,:,k)/norm**0.5
  enddo

  ! normalisation with int_(-h)^0 dz phi_n^2 /c_n^2  = 1
  norm=0
  !do k=1,nz   
    !norm = norm+ phin(:,:,k)**2/max(1d-22,cn)**2*dzt(k)*maskT(:,:,k)
  do k=1,nz-1
    norm = norm+ phin(:,:,k)**2/(small+cn**2)*dzw(k)*maskW(:,:,k)
  enddo
  k=nz; norm = norm+ phin(:,:,k)**2/(small+cn**2)*0.5*dzw(k)*maskW(:,:,k)
  do k=1,nz   
    where( norm>0d0) phin(:,:,k) = phin(:,:,k)/norm**0.5
  enddo

  if (enable_idemix_M2) then
    ! calculate structure function for energy: 
    ! E(z) = E_0 0.5( (1+f^2/om^2) phi_n^2/c_n^2 + (1-f^2/om^2) (dphi_n/dz)^2/N^2)
    do k=1,nz   
     do j=js_pe-onx,je_pe+onx
      do i=is_pe-onx,ie_pe+onx
       E_struct_M2(i,j,k) = 0.5*( (1+coriolis_t(i,j)**2/omega_M2**2)*phin(i,j,k)**2 /(small+cn(i,j)**2) &
                                 +(1-coriolis_t(i,j)**2/omega_M2**2)*phinz(i,j,k)**2/Nsqr_lim(i,j,k) &
                                      !)*maskT(i,j,k)
                                      )*maskW(i,j,k)
      enddo
     enddo
    enddo
  endif

  if (enable_idemix_niw) then
    do k=1,nz   
     do j=js_pe-onx,je_pe+onx
      do i=is_pe-onx,ie_pe+onx
       E_struct_niw(i,j,k) = 0.5*( (1+coriolis_t(i,j)**2/omega_niw(i,j)**2)*phin(i,j,k)**2/(small+cn(i,j)**2) &
                                  !+(1-coriolis_t(i,j)**2/omega_niw(i,j)**2)*phinz(i,j,k)**2/Nsqr_lim(i,j,k))*maskT(i,j,k)
                                  +(1-coriolis_t(i,j)**2/omega_niw(i,j)**2)*phinz(i,j,k)**2/Nsqr_lim(i,j,k))*maskW(i,j,k)
      enddo
     enddo
    enddo
  endif
end subroutine calc_vertical_struct_fct







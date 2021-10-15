

subroutine set_idemix_parameter
!=======================================================================
! set main IDEMIX parameter 
!=======================================================================
  use main_module   
  use idemix_module   
  implicit none
  integer :: i,j,k
  real*8 :: fxa,cstar,gofx2,bN0,hofx1
  !include "mass.include"  ! include this on AIX which does not know function acosh, also link with -lmass
  do j=js_pe-onx,je_pe+onx
   do i=is_pe-onx,ie_pe+onx
    bN0=0.0
    do k=1,nz-1
     bN0 = bN0 + max(0d0,Nsqr(i,j,k,tau))**0.5*dzw(k)*maskW(i,j,k) 
    enddo
    bN0 = bN0 + max(0d0,Nsqr(i,j,nz,tau))**0.5*0.5*dzw(nz)*maskW(i,j,nz) 
    do k=1,nz
     fxa = max(0d0,Nsqr(i,j,k,tau))**0.5/(1d-22 + abs(coriolis_t(i,j)) )
     cstar = max(1d-2,bN0/(pi*jstar) )
     c0(i,j,k)=max(0d0, gamma*cstar*gofx2(fxa)*maskW(i,j,k) )
     v0(i,j,k)=max(0d0, gamma*cstar*hofx1(fxa)*maskW(i,j,k) )
     alpha_c(i,j,k) = max( 1d-4, mu0*acosh(max(1d0,fxa))*abs(coriolis_t(i,j))/cstar**2 )*maskW(i,j,k) 
    enddo
  enddo
 enddo
end subroutine set_idemix_parameter


subroutine integrate_idemix
!=======================================================================
! integrate idemix on W grid
!=======================================================================
 use main_module   
 use eke_module   
 use idemix_module   
 implicit none
 integer :: i,j,k,ks,ke
 real*8 :: a_tri(nz),b_tri(nz),c_tri(nz),d_tri(nz),delta(nz)
 real*8 :: forc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
 real*8 :: maxE_iw(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
 real*8 :: a_loc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
 ke = nz

 !---------------------------------------------------------------------------------
 ! forcing by EKE dissipation
 !---------------------------------------------------------------------------------
 if (enable_eke) then
      forc = eke_diss_iw
 else ! short cut without EKE model
      if (enable_store_cabbeling_heat)  then
            forc = K_diss_gm + K_diss_h - P_diss_skew - P_diss_hmix  - P_diss_iso
      else
            forc = K_diss_gm + K_diss_h - P_diss_skew
      endif
 endif

 if (enable_eke.and. (enable_eke_diss_bottom.or.enable_eke_diss_surfbot)) then
 !---------------------------------------------------------------------------------
 ! vertically integrate EKE dissipation and inject at bottom and/or surface
 !---------------------------------------------------------------------------------
   a_loc = 0d0
   do k=1,nz-1
     a_loc = a_loc + dzw(k)*forc(:,:,k)*maskW(:,:,k)
   enddo
   k=nz; a_loc = a_loc + 0.5*dzw(k)*forc(:,:,k)*maskW(:,:,k)
   forc = 0 
   if (enable_eke_diss_bottom) then 
    do j=js_pe,je_pe
     do i=is_pe,ie_pe
      ks = max(1,kbot(i,j))
      forc(i,j,ks)=a_loc(i,j)/dzw(ks)
     enddo
    enddo
   else
    do j=js_pe,je_pe
     do i=is_pe,ie_pe
      ks = max(1,kbot(i,j))
      forc(i,j,ks) =      eke_diss_surfbot_frac*a_loc(i,j)/dzw(ks)
      forc(i,j,ke) = (1.-eke_diss_surfbot_frac)*a_loc(i,j)/(0.5*dzw(ke))
     enddo
    enddo
   endif
 endif

 !---------------------------------------------------------------------------------
 ! forcing by bottom friction
 !---------------------------------------------------------------------------------
 if (.not.enable_store_bottom_friction_tke) forc = forc + K_diss_bot

 !---------------------------------------------------------------------------------
 !prevent negative dissipation of IW energy
 !---------------------------------------------------------------------------------
 maxE_iw = max(0D0, E_iw(:,:,:,tau) )

 !---------------------------------------------------------------------------------
 ! vertical diffusion and dissipation is solved implicitely
 !---------------------------------------------------------------------------------
 do j=js_pe,je_pe
  do i=is_pe,ie_pe
    ks=kbot(i,j)
    if (ks>0) then
     do k=ks,ke-1
      delta(k) = dt_tracer*tau_v/dzt(k+1)*0.5*(c0(i,j,k)+c0(i,j,k+1))
     enddo
     delta(ke)=0.0
     do k=ks+1,ke-1
       a_tri(k) = - delta(k-1)*c0(i,j,k-1)/dzw(k)
     enddo
     a_tri(ks)=0.0
     a_tri(ke) = - delta(ke-1)/(0.5*dzw(ke))*c0(i,j,ke-1)
     do k=ks+1,ke-1
      b_tri(k) = 1+ delta(k)*c0(i,j,k)/dzw(k) + delta(k-1)*c0(i,j,k)/dzw(k)+ dt_tracer*alpha_c(i,j,k)*maxE_iw(i,j,k)
     enddo
     b_tri(ke) = 1+ delta(ke-1)/(0.5*dzw(ke))*c0(i,j,ke) + dt_tracer*alpha_c(i,j,ke)*maxE_iw(i,j,ke)
     b_tri(ks) = 1+ delta(ks)/dzw(ks)*c0(i,j,ks)         + dt_tracer*alpha_c(i,j,ks)*maxE_iw(i,j,ks)
     do k=ks,ke-1
      c_tri(k) = - delta(k)/dzw(k)*c0(i,j,k+1)
     enddo
     c_tri(ke)=0.0
     d_tri(ks:ke)=E_iw(i,j,ks:ke,tau)  + dt_tracer*forc(i,j,ks:ke)
     d_tri(ks) = d_tri(ks) + dt_tracer*forc_iw_bottom(i,j)/dzw(ks) 
     d_tri(ke) = d_tri(ke) + dt_tracer*forc_iw_surface(i,j)/(0.5*dzw(ke))
     call solve_tridiag(a_tri(ks:ke),b_tri(ks:ke),c_tri(ks:ke),d_tri(ks:ke),E_iw(i,j,ks:ke,taup1),ke-ks+1)
    endif
  enddo
 enddo

 !---------------------------------------------------------------------------------
 ! store IW dissipation 
 !---------------------------------------------------------------------------------
 iw_diss = alpha_c*maxE_iw(:,:,:)*E_iw(:,:,:,taup1)

 !---------------------------------------------------------------------------------
 ! add tendency due to lateral diffusion
 !---------------------------------------------------------------------------------
 if (enable_idemix_hor_diffusion) then
  do j=js_pe-onx,je_pe+onx
   do i=is_pe-onx,ie_pe+onx-1
    flux_east(i,j,:)=tau_h*0.5*(v0(i+1,j,:)+v0(i,j,:)) * &
                     (v0(i+1,j,:)*E_iw(i+1,j,:,tau)-v0(i,j,:)*E_iw(i,j,:,tau))/(cost(j)*dxu(i))*maskU(i,j,:)
   enddo
  enddo
  flux_east(ie_pe-onx,:,:)=0.
  do j=js_pe-onx,je_pe+onx-1
    flux_north(:,j,:)= tau_h*0.5*(v0(:,j+1,:)+v0(:,j,:)) * &
                       (v0(:,j+1,:)*E_iw(:,j+1,:,tau)-v0(:,j,:)*E_iw(:,j,:,tau))/dyu(j)*maskV(:,j,:)*cosu(j)
  enddo
  flux_north(:,je_pe+onx,:)=0.
  do j=js_pe,je_pe
    do i=is_pe,ie_pe
     E_iw(i,j,:,taup1)= E_iw(i,j,:,taup1) + dt_tracer*maskW(i,j,:)* &
                                  ((flux_east(i,j,:) - flux_east(i-1,j,:))/(cost(j)*dxt(i)) &
                                  +(flux_north(i,j,:) -flux_north(i,j-1,:))/(cost(j)*dyt(j)) )
   enddo
  enddo
 endif

 !---------------------------------------------------------------------------------
 ! add tendency due to advection
 !---------------------------------------------------------------------------------
 if (enable_idemix_superbee_advection) then
  call adv_flux_superbee_wgrid(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_east,flux_north,flux_top,E_iw(:,:,:,tau))
 endif
 if (enable_idemix_upwind_advection) then
  call adv_flux_upwind_wgrid(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_east,flux_north,flux_top,E_iw(:,:,:,tau))
 endif
 if (enable_idemix_superbee_advection .or. enable_idemix_upwind_advection) then
  do j=js_pe,je_pe
   do i=is_pe,ie_pe
      dE_iw(i,j,:,tau)=maskW(i,j,:)* (-( flux_east(i,j,:)-  flux_east(i-1,j,:))/(cost(j)*dxt(i)) &
                                      -(flux_north(i,j,:)- flux_north(i,j-1,:))/(cost(j)*dyt(j)) )
   enddo
  enddo
  k=1; dE_iw(:,:,k,tau)=dE_iw(:,:,k,tau)-flux_top(:,:,k)/dzw(k)   
  do k=2,nz-1
   dE_iw(:,:,k,tau)=dE_iw(:,:,k,tau)-(flux_top(:,:,k)- flux_top(:,:,k-1))/dzw(k)   
  enddo
  k=nz
  dE_iw(:,:,k,tau)=dE_iw(:,:,k,tau)-(flux_top(:,:,k)- flux_top(:,:,k-1))/(0.5*dzw(k))   
 !---------------------------------------------------------------------------------
 ! Adam Bashforth time stepping
 !---------------------------------------------------------------------------------
  E_iw(:,:,:,taup1)=E_iw(:,:,:,taup1)+dt_tracer*( (1.5+AB_eps)*dE_iw(:,:,:,tau) - ( 0.5+AB_eps)*dE_iw(:,:,:,taum1))
 endif

end subroutine integrate_idemix





function gofx2(x)
!=======================================================================
! a function g(x) 
!=======================================================================
 implicit none
 real*8 :: gofx2,x,c
 real*8, parameter :: pi = 3.14159265358979323846264338327950588
 x=max(3d0,x)
 c= 1.-(2./pi)*asin(1./x)
 gofx2 = 2/pi/c*0.9*x**(-2./3.)*(1-exp(-x/4.3))
end function gofx2

function hofx1(x)
!=======================================================================
! a function h(x) 
!=======================================================================
 implicit none
 real*8 :: hofx1,x
 real*8, parameter :: pi = 3.14159265358979323846264338327950588
 hofx1 = (2./pi)/(1.-(2./pi)*asin(1./x)) * (x-1.)/(x+1.)
end function hofx1

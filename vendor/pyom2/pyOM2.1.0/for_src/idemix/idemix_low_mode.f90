


subroutine integrate_idemix_M2
!=======================================================================
! integrate M2 wave compartment in time
!=======================================================================
 use main_module   
 use idemix_module   
 implicit none
 integer :: i,j,k
 real*8 :: advp_fe(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np) 
 real*8 :: advp_fn(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np) 
 real*8 :: advp_ft(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np) 
 call adv_flux_superbee_spectral(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,np,advp_fe,advp_fn,advp_ft,E_M2,u_M2,v_M2,w_M2)
 call reflect_flux(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,np,advp_fe,advp_fn)
 do k=2,np-1
  do j=js_pe,je_pe
   do i=is_pe,ie_pe
     dE_M2p(i,j,k,tau)=maskTp(i,j,k)* &
                 (-(advp_fe(i,j,k)- advp_fe(i-1,j,k))/(cost(j)*dxt(i)) &
                  -(advp_fn(i,j,k)- advp_fn(i,j-1,k))/(cost(j)*dyt(j)) &
                  -(advp_ft(i,j,k)- advp_ft(i,j,k-1))/dphit(k)        ) 
   enddo
  enddo
 enddo
 do k=2,np-1
  E_M2(:,:,k,taup1)=E_M2(:,:,k,tau)+dt_tracer*(forc_M2(:,:,k) -tau_M2*E_M2(:,:,k,tau) + &
                                   (1.5+AB_eps)*dE_M2p(:,:,k,tau)-(0.5+AB_eps)*dE_M2p(:,:,k,taum1)) 
 enddo
 ! physical advection
end subroutine integrate_idemix_M2





subroutine integrate_idemix_niw
!=======================================================================
! integrate NIW wave compartment in time
!=======================================================================
 use main_module   
 use idemix_module   
 implicit none
 integer :: i,j,k
 real*8 :: advp_fe(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np) 
 real*8 :: advp_fn(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np) 
 real*8 :: advp_ft(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np) 


 call adv_flux_superbee_spectral(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,np,advp_fe,advp_fn,advp_ft,E_niw,u_niw,v_niw,w_niw)
 call reflect_flux(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,np,advp_fe,advp_fn)
 do k=2,np-1
  do j=js_pe,je_pe
   do i=is_pe,ie_pe
     dE_niwp(i,j,k,tau)=maskTp(i,j,k)* &
                 (-(advp_fe(i,j,k)- advp_fe(i-1,j,k))/(cost(j)*dxt(i)) &
                  -(advp_fn(i,j,k)- advp_fn(i,j-1,k))/(cost(j)*dyt(j)) &
                  -(advp_ft(i,j,k)- advp_ft(i,j,k-1))/dphit(k)         )  
   enddo
  enddo
 enddo
 do k=2,np-1
   E_niw(:,:,k,taup1)=E_niw(:,:,k,tau)+dt_tracer*(forc_niw(:,:,k) -tau_niw(:,:)*E_niw(:,:,k,tau) +&
                                (1.5+AB_eps)*dE_niwp(:,:,k,tau)-(0.5+AB_eps)*dE_niwp(:,:,k,taum1)) 
 enddo
end subroutine integrate_idemix_niw




subroutine wave_interaction
!=======================================================================
! interaction of wave components
!=======================================================================
  use main_module   
  use idemix_module   
  implicit none
  integer :: i,j,k
  real*8 :: fmin,cont(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)

  if (enable_idemix) then 
   cont=0.0
   do k=1,nz
    do j=js_pe-onx,je_pe+onx
     do i=is_pe-onx,ie_pe+onx
        cont(i,j) = cont(i,j) + E_iw(i,j,k,tau)*dzt(k)*maskT(i,j,k)
     enddo
    enddo
   enddo
  endif

  if (enable_idemix_M2) then 
   ! integrate M2 energy over angle
   E_M2_int = 0
   do k=2,np-1
    do j=js_pe-onx,je_pe+onx
     do i=is_pe-onx,ie_pe+onx
      E_M2_int(i,j) = E_M2_int(i,j) + E_M2(i,j,k,tau)*dphit(k)*maskTp(i,j,k)
     enddo
    enddo
   enddo
  endif

  if (enable_idemix_niw) then 
   ! integrate niw energy over angle
   E_niw_int = 0
   do k=2,np-1
    do j=js_pe-onx,je_pe+onx
     do i=is_pe-onx,ie_pe+onx
      E_niw_int(i,j) = E_niw_int(i,j) + E_niw(i,j,k,tau)*dphit(k)*maskTp(i,j,k)
     enddo
    enddo
   enddo
  endif


  if (enable_idemix_M2.and.enable_idemix) then 
   ! update M2 energy: interaction of M2 and continuum
   do k=2,np-1
    do j=js_pe-onx,je_pe+onx
     do i=is_pe-onx,ie_pe+onx
      fmin = min(0.5d0/dt_tracer,alpha_M2_cont(i,j)*cont(i,j) ) ! flux limiter
      M2_psi_diss(i,j,k) = fmin*E_M2(i,j,k,tau)*maskTp(i,j,k)
      E_M2(i,j,k,taup1)= E_M2(i,j,k,taup1)-dt_tracer*M2_psi_diss(i,j,k) 
     enddo
    enddo
   enddo
  endif

  if (enable_idemix.and.enable_idemix_M2) then
   do k=1,nz
    do j=js_pe-onx,je_pe+onx
     do i=is_pe-onx,ie_pe+onx
       fmin = min(0.5d0/dt_tracer,alpha_M2_cont(i,j)*cont(i,j) ) ! flux limiter
       E_iw(i,j,k,taup1) = E_iw(i,j,k,taup1)+dt_tracer*tau_M2(i,j)*E_M2_int(i,j)*E_struct_M2(i,j,k)*maskT(i,j,k) &
                                            + dt_tracer*fmin*E_M2_int(i,j)*E_struct_M2(i,j,k)*maskT(i,j,k)
     enddo
    enddo
   enddo
  endif

  if (enable_idemix.and.enable_idemix_niw) then
   do k=1,nz
    do j=js_pe-onx,je_pe+onx
     do i=is_pe-onx,ie_pe+onx
       E_iw(i,j,k,taup1) = E_iw(i,j,k,taup1)+dt_tracer*tau_niw(i,j)*E_niw_int(i,j)*E_struct_niw(i,j,k)*maskT(i,j,k)
     enddo
    enddo
   enddo
  endif

end subroutine wave_interaction





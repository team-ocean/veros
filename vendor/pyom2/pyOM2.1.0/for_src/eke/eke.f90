

subroutine init_eke
!=======================================================================
! Initialize EKE
!=======================================================================
 use main_module   
 use eke_module   

 if (enable_eke_leewave_dissipation ) then
    hrms_k0 = max(eke_hrms_k0_min, 2/pi*eke_topo_hrms**2/max(1d-12,eke_topo_lam)**1.5  )
 endif
end subroutine init_eke


subroutine set_eke_diffusivities
!=======================================================================
! set skew diffusivity K_gm and isopycnal diffusivity K_iso
! set also vertical viscosity if TEM formalism is chosen
!=======================================================================
 use main_module   
 use isoneutral_module   
 use eke_module   
 implicit none
 integer :: i,j,k
 real*8 :: C_rossby(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)

 if (enable_eke) then
 !---------------------------------------------------------------------------------
 ! calculate Rossby radius as minimum of mid-latitude and equatorial R. rad.
 !---------------------------------------------------------------------------------
  C_rossby = 0.0
  do k=1,nz
   C_Rossby(:,:) = C_Rossby(:,:)   + sqrt(max(0D0,Nsqr(:,:,k,tau)))*dzw(k)*maskW(:,:,k)/pi      
  enddo
  do j=js_pe-onx,je_pe+onx
   do i=is_pe-onx,ie_pe+onx
    L_Rossby(i,j) = min( C_Rossby(i,j)/max(abs(coriolis_t(i,j)),1D-16), &
                        sqrt( C_Rossby(i,j)/ max(2d0*beta(i,j),1.D-16) )  )
   enddo
  enddo
 !---------------------------------------------------------------------------------
 ! calculate vertical viscosity and skew diffusivity  
 !---------------------------------------------------------------------------------
  sqrteke = sqrt(max(0d0,eke(:,:,:,tau)))
  do k=1,nz
    L_Rhines(:,:,k)= sqrt(sqrteke(:,:,k)/max(beta,1D-16))
    eke_len(:,:,k) = max(eke_lmin, min(eke_cross*L_Rossby,eke_crhin*L_Rhines(:,:,k)))
  enddo
  K_gm  = min(eke_k_max,eke_c_k*eke_len*sqrteke )
 else ! enable_eke
 !---------------------------------------------------------------------------------
 ! use fixed GM diffusivity
 !---------------------------------------------------------------------------------
  K_gm = K_gm_0
 endif

 if (enable_TEM_friction) then 
  do j=js_pe-onx,je_pe+onx
   do i=is_pe-onx,ie_pe+onx
    kappa_gm(i,j,:)  = K_gm(i,j,:)*min(0.01D0,coriolis_t(i,j)**2/max(1d-9,Nsqr(i,j,:,tau)) )*maskW(i,j,:)
   enddo
  enddo
 endif

 if (enable_eke .and.enable_eke_isopycnal_diffusion) then
   K_iso = K_gm 
 else
   K_iso = K_iso_0 ! always constant
 endif
end subroutine set_eke_diffusivities



subroutine integrate_eke
!=======================================================================
! integrate EKE equation on W grid
!=======================================================================
 use main_module   
 use isoneutral_module   
 use eke_module   
 implicit none
 integer :: i,j,k,ks,ke
 real*8 :: a_tri(nz),b_tri(nz),c_tri(nz),d_tri(nz),delta(nz)
 real*8 :: forc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz),fxa,uz,vz,Ri
 real*8 :: c_int(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
 real*8 :: a_loc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
 real*8 :: b_loc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)

 !---------------------------------------------------------------------------------
 ! forcing by dissipation by lateral friction and GM using TRM formalism or skew diffusion
 !---------------------------------------------------------------------------------
 forc = K_diss_h + K_diss_gm  - P_diss_skew
 
 !---------------------------------------------------------------------------------
 ! store transfer due to isopycnal and horizontal mixing from dyn. enthalpy 
 ! by non-linear eq.of state either to EKE or to heat
 !---------------------------------------------------------------------------------
 if (.not. enable_store_cabbeling_heat)  forc = forc - P_diss_hmix  - P_diss_iso

 !---------------------------------------------------------------------------------
 ! coefficient for dissipation of EKE:
 ! by lee wave generation, Ri-dependent interior loss of balance and bottom friction
 !---------------------------------------------------------------------------------
 if (enable_eke_leewave_dissipation ) then

 !---------------------------------------------------------------------------------
 !by lee wave generation
 !---------------------------------------------------------------------------------
  c_lee = 0d0
  do j=js_pe,je_pe
   do i=is_pe,ie_pe
     k=kbot(i,j)
     if (k>0.and. k<nz) then ! could be surface: factor 0.5
        fxa = max(0d0,Nsqr(i,j,k,tau))**0.25 
        fxa = fxa *(1.5*fxa/sqrt(max( 1D-6,abs(coriolis_t(i,j)) )  ) -2)
        c_lee(i,j) = c_lee0*hrms_k0(i,j)*sqrt( sqrteke(i,j,k))  * max(0D0,fxa ) /dzw(k)
     endif
    enddo
  enddo

 !---------------------------------------------------------------------------------
 !Ri-dependent dissipation by interior loss of balance
 !---------------------------------------------------------------------------------
  c_Ri_diss=0d0
  do k=1,nz-1
   do j=js_pe-onx+1,je_pe+onx
    do i=is_pe-onx+1,ie_pe+onx
     uz = ( ((u(i  ,j,k+1,tau)-u(i  ,j,k,tau))/dzt(k)*maskU(i  ,j,k))**2 &
         +  ((u(i-1,j,k+1,tau)-u(i-1,j,k,tau))/dzt(k)*maskU(i-1,j,k))**2 )/(maskU(i,j,k)+maskU(i-1,j,k) +1d-18 )
     vz = ( ((v(i,j  ,k+1,tau)-v(i,j  ,k,tau))/dzt(k)*maskV(i,j  ,k))**2 &
         +  ((v(i,j-1,k+1,tau)-v(i,j-1,k,tau))/dzt(k)*maskV(i,j-1,k))**2 )/(maskV(i,j,k)+maskV(i,j-1,k) +1d-18 )
     Ri = max(1d-8, Nsqr(i,j,k,tau)) /(uz+vz+1d-18)
     fxa=1-0.5*(1.+tanh((Ri-eke_Ri0)/eke_Ri1))   
     c_Ri_diss(i,j,k) = maskW(i,j,k)*fxa*eke_int_diss0
    enddo
   enddo
  enddo
  c_Ri_diss(:,:,nz)=c_Ri_diss(:,:,nz-1)*maskW(:,:,nz)

 !---------------------------------------------------------------------------------
 ! vertically integrate Ri-dependent dissipation and EKE 
 !---------------------------------------------------------------------------------
  a_loc=0d0; b_loc=0d0
  do k=1,nz-1
     a_loc = a_loc + c_Ri_diss(:,:,k)*eke(:,:,k,tau)*maskW(:,:,k)*dzw(k) 
     b_loc = b_loc + eke(:,:,k,tau)*maskW(:,:,k)*dzw(k)
  enddo
  k=nz
  a_loc = a_loc + c_Ri_diss(:,:,k)*eke(:,:,k,tau)*maskW(:,:,k)*dzw(k)*0.5
  b_loc = b_loc + eke(:,:,k,tau)*maskW(:,:,k)*dzw(k)*0.5

 !---------------------------------------------------------------------------------
 ! add bottom fluxes by lee waves and bottom friction to a_loc
 !---------------------------------------------------------------------------------
  do j=js_pe,je_pe
   do i=is_pe,ie_pe
     k=kbot(i,j)
     if (k>0.and.k<nz)  a_loc(i,j) = a_loc(i,j) + c_lee(i,j)*eke(i,j,k,tau)*maskW(i,j,k)*dzw(k) ! could be surface: factor 0.5
     if (k>0.and.k<nz)  a_loc(i,j) = a_loc(i,j) +  & 
             2*eke_r_bot*eke(i,j,k,tau)*sqrt(2.0)*sqrteke(i,j,k)*maskW(i,j,k) ! /dzw(k)*dzw(k) ! could be surface: factor 0.5
   enddo
  enddo

 !---------------------------------------------------------------------------------
 ! dissipation constant is vertically integrated forcing divided by 
 ! vertically integrated EKE to account for vertical EKE radiation 
 !---------------------------------------------------------------------------------
  where (b_loc > 0.0)  
      a_loc = a_loc/b_loc 
  elsewhere
      a_loc=0.0
  end where
  do k=1,nz
     c_int(:,:,k) = a_loc
  enddo

 else 
 !---------------------------------------------------------------------------------
 ! dissipation by local interior loss of balance with constant coefficient
 !---------------------------------------------------------------------------------
  do k=1,nz
   do j=js_pe-onx,je_pe+onx
    do i=is_pe-onx,ie_pe+onx
      c_int(i,j,k) = eke_c_eps*sqrteke(i,j,k)/eke_len(i,j,k)*maskW(i,j,k)
    enddo
   enddo
  enddo
 endif


 !---------------------------------------------------------------------------------
 ! vertical diffusion of EKE,forcing and dissipation
 !---------------------------------------------------------------------------------
 ke = nz
 do j=js_pe,je_pe
  do i=is_pe,ie_pe
    ks=kbot(i,j)
    if (ks>0) then
     do k=ks,ke-1
      delta(k) = dt_tracer/dzt(k+1)*0.5*(kappaM(i,j,k)+KappaM(i,j,k+1))*alpha_eke
     enddo
     delta(ke)=0.0
     do k=ks+1,ke-1
       a_tri(k) = - delta(k-1)/dzw(k)
     enddo
     a_tri(ks)=0.0
     a_tri(ke) = - delta(ke-1)/(0.5*dzw(ke))
     do k=ks+1,ke-1
      b_tri(k) = 1+ delta(k)/dzw(k) + delta(k-1)/dzw(k) + dt_tracer*c_int(i,j,k)
     enddo
     b_tri(ke) = 1+ delta(ke-1)/(0.5*dzw(ke)) + dt_tracer*c_int(i,j,ke) 
     b_tri(ks) = 1+ delta(ks)/dzw(ks)         + dt_tracer*c_int(i,j,ks) 
     do k=ks,ke-1
      c_tri(k) = - delta(k)/dzw(k)
     enddo
     c_tri(ke)=0.0
     d_tri(ks:ke)=eke(i,j,ks:ke,tau)  + dt_tracer*forc(i,j,ks:ke)
     d_tri(ks) = d_tri(ks) 
     d_tri(ke) = d_tri(ke) !+ dt_tracer*forc_eke_surfac(i,j)/(0.5*dzw(ke))
     call solve_tridiag(a_tri(ks:ke),b_tri(ks:ke),c_tri(ks:ke),d_tri(ks:ke),eke(i,j,ks:ke,taup1),ke-ks+1)
    endif
  enddo
 enddo



 !---------------------------------------------------------------------------------
 ! store eke dissipation 
 !---------------------------------------------------------------------------------
 if (enable_eke_leewave_dissipation ) then

  eke_diss_iw = 0d0
  eke_diss_tke = c_Ri_diss*eke(:,:,:,taup1)

 !---------------------------------------------------------------------------------
 ! flux by lee wave generation and bottom friction
 !---------------------------------------------------------------------------------
  do j=js_pe,je_pe
   do i=is_pe,ie_pe
    k=kbot(i,j)
    if (k>0.and.k<nz) eke_diss_iw(i,j,k)  = eke_diss_iw(i,j,k) + c_lee(i,j)*eke(i,j,k,taup1)*maskW(i,j,k)
    if (k>0.and.k<nz) eke_diss_tke(i,j,k) = eke_diss_tke(i,j,k)+ &
                       2*eke_r_bot*eke(i,j,k,taup1)*sqrt(2.0)*sqrteke(i,j,k)*maskW(i,j,k)/dzw(k)
   enddo
  enddo

 !---------------------------------------------------------------------------------
 ! account for sligthly incorrect integral of dissipation due to time stepping
 !---------------------------------------------------------------------------------
  a_loc=0d0; b_loc=0d0
  do k=1,nz-1
   a_loc=a_loc + (eke_diss_iw(:,:,k)+eke_diss_tke(:,:,k))*dzw(k)
   b_loc=b_loc + c_int(:,:,k)*eke(:,:,k,taup1)*dzw(k)
  enddo
  k=nz
  a_loc=a_loc + (eke_diss_iw(:,:,k)+eke_diss_tke(:,:,k))*dzw(k)*0.5
  b_loc=b_loc + c_int(:,:,k)*eke(:,:,k,taup1)*dzw(k)*0.5
  where (a_loc/=0d0)
     b_loc=b_loc/a_loc
  elsewhere
     b_loc=0.0
  end where
  ! F= eke_diss ,  a = sum F,  b = sum c_int e
  ! G = F*b/a -> sum G = (sum c_int e) /(sum F )  sum F
  do k=1,nz
   eke_diss_iw(:,:,k) = eke_diss_iw(:,:,k)*b_loc
   eke_diss_tke(:,:,k) = eke_diss_tke(:,:,k)*b_loc
  enddo

 !---------------------------------------------------------------------------------
 ! store diagnosed flux by lee waves and bottom friction
 !---------------------------------------------------------------------------------
  do j=js_pe,je_pe
   do i=is_pe,ie_pe
     k=kbot(i,j)
     if (k>0.and.k<nz) eke_lee_flux(i,j)=c_lee(i,j)*eke(i,j,k,taup1)*dzw(k)
     if (k>0.and.k<nz) eke_bot_flux(i,j)=2*eke_r_bot*eke(i,j,k,taup1)*sqrt(2.0)*sqrteke(i,j,k)
   enddo
  enddo

 else
   eke_diss_iw = c_int*eke(:,:,:,taup1)
   eke_diss_tke = 0d0
 endif


 !---------------------------------------------------------------------------------
 ! add tendency due to lateral diffusion
 !---------------------------------------------------------------------------------
 do j=js_pe-onx,je_pe+onx
  do i=is_pe-onx,ie_pe+onx-1
   flux_east(i,j,:)=0.5*max(500d0,K_gm(i,j,:)+K_gm(i+1,j,:))*(eke(i+1,j,:,tau)-eke(i,j,:,tau))/(cost(j)*dxu(i))*maskU(i,j,:)
  enddo
 enddo
 flux_east(ie_pe+onx,:,:)=0.
 do j=js_pe-onx,je_pe+onx-1
   flux_north(:,j,:)=0.5*max(500d0,K_gm(:,j,:)+K_gm(:,j+1,:))*(eke(:,j+1,:,tau)-eke(:,j,:,tau))/dyu(j)*maskV(:,j,:)*cosu(j)
 enddo
 flux_north(:,je_pe+onx,:)=0.
 do j=js_pe,je_pe
   do i=is_pe,ie_pe
    eke(i,j,:,taup1)= eke(i,j,:,taup1) + dt_tracer*maskW(i,j,:)*&
                       ((flux_east(i,j,:) - flux_east(i-1,j,:))/(cost(j)*dxt(i)) &
                       +(flux_north(i,j,:) -flux_north(i,j-1,:))/(cost(j)*dyt(j)) )
  enddo
 enddo

 !---------------------------------------------------------------------------------
 ! add tendency due to advection
 !---------------------------------------------------------------------------------
 if (enable_eke_superbee_advection) then
  call adv_flux_superbee_wgrid(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_east,flux_north,flux_top,eke(:,:,:,tau))
 endif
 if (enable_eke_upwind_advection) then
  call adv_flux_upwind_wgrid(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_east,flux_north,flux_top,eke(:,:,:,tau))
 endif
 if (enable_eke_superbee_advection .or. enable_eke_upwind_advection) then
  do j=js_pe,je_pe
   do i=is_pe,ie_pe
      deke(i,j,:,tau)=maskW(i,j,:)* (-( flux_east(i,j,:)-  flux_east(i-1,j,:))/(cost(j)*dxt(i)) &
                                     -(flux_north(i,j,:)- flux_north(i,j-1,:))/(cost(j)*dyt(j)) )
   enddo
  enddo
  k=1; deke(:,:,k,tau)=deke(:,:,k,tau)-flux_top(:,:,k)/dzw(k)   
  do k=2,nz-1
   deke(:,:,k,tau)=deke(:,:,k,tau)-(flux_top(:,:,k)- flux_top(:,:,k-1))/dzw(k)   
  enddo
  k=nz
  deke(:,:,k,tau)=deke(:,:,k,tau)-(flux_top(:,:,k)- flux_top(:,:,k-1))/(0.5*dzw(k))   
 !---------------------------------------------------------------------------------
 ! Adam Bashforth time stepping
 !---------------------------------------------------------------------------------
  eke(:,:,:,taup1)=eke(:,:,:,taup1)+dt_tracer*( (1.5+AB_eps)*deke(:,:,:,tau) - ( 0.5+AB_eps)*deke(:,:,:,taum1))
 endif

end subroutine integrate_eke


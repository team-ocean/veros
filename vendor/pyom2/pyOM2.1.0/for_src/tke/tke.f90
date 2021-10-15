


subroutine set_tke_diffusivities
!=======================================================================
!  set vertical diffusivities based on TKE model
!=======================================================================
 use main_module   
 use tke_module   
 use idemix_module   
 implicit none
 integer :: k!,kp1,km1
 real*8 :: Rinumber(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)

 if (enable_tke) then

  sqrttke = sqrt(max(0d0,tke(:,:,:,tau)))

 !---------------------------------------------------------------------------------
 ! calculate buoyancy length scale
 !---------------------------------------------------------------------------------
  mxl = sqrt(2D0)*sqrttke/sqrt(max(1d-12,Nsqr(:,:,:,tau)))*maskW

 !---------------------------------------------------------------------------------
 ! apply limits for mixing length 
 !---------------------------------------------------------------------------------
  if (tke_mxl_choice == 1) then
 !---------------------------------------------------------------------------------
 ! bounded by the distance to surface/bottom
 !---------------------------------------------------------------------------------
    do k=1,nz
     mxl(:,:,k) = min(-zw(k)+dzw(k)*0.5,mxl(:,:,k),ht+zw(k))
    enddo
    mxl= max(mxl,mxl_min)
  elseif (tke_mxl_choice == 2) then
 !---------------------------------------------------------------------------------
 ! bound length scale as in mitgcm/OPA code
 !---------------------------------------------------------------------------------
    do k=nz-1,1,-1
     mxl(:,:,k) = MIN(mxl(:,:,k),mxl(:,:,k+1)+dzt(k+1) )
    enddo
    mxl(:,:,nz) = MIN( mxl(:,:,nz),mxl_min+dzt(nz) )
    do k=2,nz
     mxl(:,:,k) = MIN(mxl(:,:,k), mxl(:,:,k-1)+dzt(k))
    enddo
    mxl= max(mxl,mxl_min)
  else
   call halt_stop(' unknown mixing length choice for tke_mxl_choice')
  endif

 !---------------------------------------------------------------------------------
 ! calculate viscosity and diffusivity based on Prandtl number
 !---------------------------------------------------------------------------------
  call border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,K_diss_v) 
  call setcyclic_xyz   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,K_diss_v)
  KappaM  = min(kappaM_max, c_k*mxl*sqrttke )
  RiNumber = Nsqr(:,:,:,tau)/max(K_diss_v/max(1d-12,kappaM),1d-12)
  if (enable_idemix) RiNumber= min(RiNumber,KappaM*Nsqr(:,:,:,tau)/max(1d-12,alpha_c*E_iw(:,:,:,tau)**2))
  PrandtlNumber = max(1d0, min(10d0,6.6* Rinumber))
  KappaH = KappaM/Prandtlnumber
  kappaM = max( kappaM_min , kappaM)

 else
  kappaM=kappaM_0
  kappaH=kappaH_0
  if (enable_hydrostatic) then
 !---------------------------------------------------------------------------------
 !  simple convective adjustment
 !---------------------------------------------------------------------------------
    where (Nsqr(:,:,:,tau)< 0.0) kappaH = 1.0
  endif
 endif
end subroutine set_tke_diffusivities



subroutine integrate_tke
!=======================================================================
! integrate Tke equation on W grid with surface flux boundary condition
!=======================================================================
 use main_module   
 use eke_module   
 use tke_module   
 use idemix_module   
 implicit none
 integer :: i,j,k,ks,ke
 real*8 :: a_tri(nz),b_tri(nz),c_tri(nz),d_tri(nz),delta(nz)
 real*8 :: forc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)

 dt_tke = dt_mom  ! use momentum time step to prevent spurious oscillations

 !---------------------------------------------------------------------------------
 ! Sources and sinks by vertical friction, vertical mixing, and non-conservative advection
 !---------------------------------------------------------------------------------
 forc = K_diss_v - P_diss_v - P_diss_adv

 !---------------------------------------------------------------------------------
 ! store transfer due to vertical mixing from dyn. enthalpy by non-linear eq.of 
 ! state either to TKE or to heat
 !---------------------------------------------------------------------------------
 if (.not. enable_store_cabbeling_heat)  forc = forc - P_diss_nonlin

 !---------------------------------------------------------------------------------
 ! transfer part of dissipation of EKE to TKE
 !---------------------------------------------------------------------------------
 if (enable_eke) forc = forc + eke_diss_tke

 if (enable_idemix) then
 !---------------------------------------------------------------------------------
 ! transfer dissipation of internal waves to TKE
 !---------------------------------------------------------------------------------
  forc = forc + iw_diss
 !---------------------------------------------------------------------------------
 ! store bottom friction either in TKE or internal waves
 !---------------------------------------------------------------------------------
  if (enable_store_bottom_friction_tke) forc = forc + K_diss_bot
 else ! short-cut without idemix
  if (enable_eke) then
   forc = forc + eke_diss_iw 
  else ! and without EKE model
      if (enable_store_cabbeling_heat)  then
            forc = forc + K_diss_gm + K_diss_h - P_diss_skew - P_diss_hmix  - P_diss_iso
      else
            forc = forc + K_diss_gm + K_diss_h - P_diss_skew
      endif
  endif
  forc = forc + K_diss_bot
 endif


 !---------------------------------------------------------------------------------
 ! vertical mixing and dissipation of TKE
 !---------------------------------------------------------------------------------
 ke = nz
 do j=js_pe,je_pe
  do i=is_pe,ie_pe
    ks=kbot(i,j)
    if (ks>0) then
     do k=ks,ke-1
      delta(k) = dt_tke/dzt(k+1)*alpha_tke*0.5*(kappaM(i,j,k)+kappaM(i,j,k+1))
     enddo
     delta(ke)=0.0
     do k=ks+1,ke-1
       a_tri(k) = - delta(k-1)/dzw(k)
     enddo
     a_tri(ks)=0.0
     a_tri(ke) = - delta(ke-1)/(0.5*dzw(ke))
     do k=ks+1,ke-1
      b_tri(k) = 1+ delta(k)/dzw(k) + delta(k-1)/dzw(k) + dt_tke*c_eps*sqrttke(i,j,k)/mxl(i,j,k)
     enddo
     b_tri(ke) = 1+ delta(ke-1)/(0.5*dzw(ke))           + dt_tke*c_eps/mxl(i,j,ke)*sqrttke(i,j,ke) 
     b_tri(ks) = 1+ delta(ks)/dzw(ks)                   + dt_tke*c_eps/mxl(i,j,ks)*sqrttke(i,j,ks) 
     do k=ks,ke-1
      c_tri(k) = - delta(k)/dzw(k)
     enddo
     c_tri(ke)=0.0
     d_tri(ks:ke)=tke(i,j,ks:ke,tau)  + dt_tke*forc(i,j,ks:ke)
     d_tri(ks) = d_tri(ks) 
     d_tri(ke) = d_tri(ke) + dt_tke*forc_tke_surface(i,j)/(0.5*dzw(ke))
     call solve_tridiag(a_tri(ks:ke),b_tri(ks:ke),c_tri(ks:ke),d_tri(ks:ke),tke(i,j,ks:ke,taup1),ke-ks+1)
    endif
  enddo
 enddo

 !  E_n - E_(n-1) = dt( F - c_eps*sqrt(E)*E_n/mxl  )
 !  E_n = (E_(n-1) + dt F )/  (1+ dt*c_eps*sqrt(E) /mxl)
 !tke(:,:,:,taup1) = (tke(:,:,:,taup1) + dt_tke*forc)/(1d0+dt_tke*c_eps*sqrttke/mxl)*maskW

 !---------------------------------------------------------------------------------
 ! store tke dissipation for diagnostics
 !---------------------------------------------------------------------------------
 tke_diss = c_eps/mxl*sqrttke*tke(:,:,:,taup1)

 !---------------------------------------------------------------------------------
 ! Add TKE if surface density flux drains TKE in uppermost box
 !---------------------------------------------------------------------------------
 tke_surf_corr = 0.0
 do j=js_pe,je_pe
  do i=is_pe,ie_pe
    if (tke(i,j,nz,taup1) < 0.0 ) then
      tke_surf_corr(i,j) = -tke(i,j,nz,taup1)*(0.5*dzw(ke)) /dt_tke
      tke(i,j,nz,taup1) = 0.0
    endif
  enddo
 enddo

 if (enable_tke_hor_diffusion) then
 !---------------------------------------------------------------------------------
 ! add tendency due to lateral diffusion
 !---------------------------------------------------------------------------------
  do j=js_pe-onx,je_pe+onx
   do i=is_pe-onx,ie_pe+onx-1
    flux_east(i,j,:)=K_h_tke*(tke(i+1,j,:,tau)-tke(i,j,:,tau))/(cost(j)*dxu(i))*maskU(i,j,:)
   enddo
  enddo
  flux_east(ie_pe-onx,:,:)=0.
  do j=js_pe-onx,je_pe+onx-1
    flux_north(:,j,:)=K_h_tke*(tke(:,j+1,:,tau)-tke(:,j,:,tau))/dyu(j)*maskV(:,j,:)*cosu(j)
  enddo
  flux_north(:,je_pe+onx,:)=0.
  do j=js_pe,je_pe
    do i=is_pe,ie_pe
     tke(i,j,:,taup1)= tke(i,j,:,taup1) + dt_tke*maskW(i,j,:)* &
                                 ((flux_east(i,j,:) - flux_east(i-1,j,:))/(cost(j)*dxt(i)) &
                                 +(flux_north(i,j,:) -flux_north(i,j-1,:))/(cost(j)*dyt(j)) )
   enddo
  enddo
 endif

 !---------------------------------------------------------------------------------
 ! add tendency due to advection
 !---------------------------------------------------------------------------------
 if (enable_tke_superbee_advection) then
  call adv_flux_superbee_wgrid(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_east,flux_north,flux_top,tke(:,:,:,tau))
 endif
 if (enable_tke_upwind_advection) then
  call adv_flux_upwind_wgrid(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_east,flux_north,flux_top,tke(:,:,:,tau))
 endif
 if (enable_tke_superbee_advection .or. enable_tke_upwind_advection) then
  do j=js_pe,je_pe
   do i=is_pe,ie_pe
      dtke(i,j,:,tau)=maskW(i,j,:)* (-( flux_east(i,j,:)-  flux_east(i-1,j,:))/(cost(j)*dxt(i)) &
                                     -(flux_north(i,j,:)- flux_north(i,j-1,:))/(cost(j)*dyt(j)) )
    enddo
  enddo
  k=1; dtke(:,:,k,tau)=dtke(:,:,k,tau)-flux_top(:,:,k)/dzw(k)   
  do k=2,nz-1
   dtke(:,:,k,tau)=dtke(:,:,k,tau)-(flux_top(:,:,k)- flux_top(:,:,k-1))/dzw(k)   
  enddo
  k=nz
  dtke(:,:,k,tau)=dtke(:,:,k,tau)-(flux_top(:,:,k)- flux_top(:,:,k-1))/(0.5*dzw(k))   
 !---------------------------------------------------------------------------------
 ! Adam Bashforth time stepping
 !---------------------------------------------------------------------------------
  tke(:,:,:,taup1)=tke(:,:,:,taup1)+dt_tracer*( (1.5+AB_eps)*dtke(:,:,:,tau) - ( 0.5+AB_eps)*dtke(:,:,:,taum1))
 endif

end subroutine integrate_tke












subroutine isoneutral_diffusion(is_,ie_,js_,je_,nz_,tr,istemp)
!=======================================================================
!   Isopycnal diffusion for tracer, 
!   following functional formulation by Griffies et al 
!   Dissipation is calculated and stored in P_diss_iso
!   T/S changes are added to dtemp_iso/dsalt_iso
!=======================================================================
 use main_module   
 use isoneutral_module
 implicit none
 integer :: is_,ie_,js_,je_,nz_
 real*8 :: tr(is_:ie_,js_:je_,nz_,3)
 logical, intent(in) :: istemp
 integer :: i,j,k,kr,ip,jp,km1kr,kpkr,ks
 real*8 :: sumz,sumx,sumy
 real*8 :: a_tri(nz),b_tri(nz),c_tri(nz),d_tri(nz),delta(nz),sol(nz)
 real*8 :: aloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
 real*8 :: bloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
 real*8 :: fxa,diffloc

!-----------------------------------------------------------------------
!     construct total isoneutral tracer flux at east face of "T" cells 
!-----------------------------------------------------------------------
 do k=1,nz
  do j=js_pe,je_pe
   do i=is_pe-1,ie_pe
     diffloc = 0.25*(K_iso(i,j,k)+K_iso(i,j,max(1,k-1)) + K_iso(i+1,j,k)+K_iso(i+1,j,max(1,k-1)) ) 
     sumz = 0.
     do kr=0,1
       km1kr = max(k-1+kr,1)
       kpkr  = min(k+kr,nz)
       do ip=0,1
         sumz = sumz + diffloc*Ai_ez(i,j,k,ip,kr) *(tr(i+ip,j,kpkr,tau)-tr(i+ip,j,km1kr,tau))
       enddo
     enddo
     flux_east(i,j,k) = sumz/(4*dzt(k)) + (tr(i+1,j,k,tau)-tr(i,j,k,tau))/(cost(j)*dxu(i)) *K_11(i,j,k)
   enddo
  enddo
 enddo
!-----------------------------------------------------------------------
!     construct total isoneutral tracer flux at north face of "T" cells 
!-----------------------------------------------------------------------
 do k=1,nz
  do j=js_pe-1,je_pe
   do i=is_pe,ie_pe
     diffloc = 0.25*(K_iso(i,j,k)+K_iso(i,j,max(1,k-1)) + K_iso(i,j+1,k)+K_iso(i,j+1,max(1,k-1)) ) 
     sumz    = 0.
     do kr=0,1
       km1kr = max(k-1+kr,1)
       kpkr  = min(k+kr,nz)
       do jp=0,1
         sumz = sumz + diffloc*Ai_nz(i,j,k,jp,kr) *(tr(i,j+jp,kpkr,tau)-tr(i,j+jp,km1kr,tau))
       enddo
     enddo
     flux_north(i,j,k) = cosu(j)*( sumz/(4*dzt(k))+ (tr(i,j+1,k,tau)-tr(i,j,k,tau))/dyu(j)*K_22(i,j,k) )
   enddo
  enddo
 enddo
!-----------------------------------------------------------------------
!     compute the vertical tracer flux "flux_top" containing the K31
!     and K32 components which are to be solved explicitly. The K33
!     component will be treated implicitly. Note that there are some
!     cancellations of dxu(i-1+ip) and dyu(jrow-1+jp) 
!-----------------------------------------------------------------------
 do k=1,nz-1
  do j=js_pe,je_pe
   do i=is_pe,ie_pe
    diffloc = K_iso(i,j,k) 
    sumx = 0.
    do ip=0,1
     do kr=0,1
      sumx = sumx + diffloc*Ai_bx(i,j,k,ip,kr)/cost(j)*(tr(i+ip,j,k+kr,tau) - tr(i-1+ip,j,k+kr,tau))
     enddo
    enddo
    sumy    = 0.
    do jp=0,1
     do kr=0,1
      sumy = sumy + diffloc*Ai_by(i,j,k,jp,kr)*cosu(j-1+jp)* (tr(i,j+jp,k+kr,tau)-tr(i,j-1+jp,k+kr,tau))
     enddo
    enddo
    flux_top(i,j,k) = sumx/(4*dxt(i)) +sumy/(4*dyt(j)*cost(j) ) 
   enddo
  enddo
 enddo
 flux_top(:,:,nz)=0.0
!---------------------------------------------------------------------------------
!     add explicit part 
!---------------------------------------------------------------------------------
 do j=js_pe,je_pe
   do i=is_pe,ie_pe
      aloc(i,j,:)=maskT(i,j,:)*( (flux_east(i,j,:)-  flux_east(i-1,j,:))/(cost(j)*dxt(i)) &
                                +(flux_north(i,j,:)- flux_north(i,j-1,:))/(cost(j)*dyt(j)) )
   enddo
 enddo
 k=1; aloc(:,:,k)=aloc(:,:,k)+maskT(:,:,k)*flux_top(:,:,k)/dzt(k)   
 do k=2,nz
   aloc(:,:,k)=aloc(:,:,k)+maskT(:,:,k)*(flux_top(:,:,k)- flux_top(:,:,k-1))/dzt(k)   
 enddo

 if (istemp) then
      dtemp_iso = dtemp_iso + aloc
 else
      dsalt_iso = dsalt_iso + aloc
 endif

 do j=js_pe,je_pe
   do i=is_pe,ie_pe
      tr(i,j,:,taup1)=tr(i,j,:,taup1)+dt_tracer*aloc(i,j,:)
   enddo
 enddo
!---------------------------------------------------------------------------------
!     add implicit part 
!---------------------------------------------------------------------------------
 aloc = tr(:,:,:,taup1)
 a_tri=0.0;b_tri=0.0; c_tri=0.0; d_tri=0.0; delta=0.0
 do j=js_pe,je_pe
    do i=is_pe,ie_pe
        ks=kbot(i,j)
        if (ks>0) then
         do k=ks,nz-1
          delta(k) = dt_tracer/dzw(k)*K_33(i,j,k)
         enddo
         delta(nz)=0.0
         do k=ks+1,nz
           a_tri(k) = - delta(k-1)/dzt(k)
         enddo
         a_tri(ks)=0.0
         do k=ks+1,nz-1
          b_tri(k) = 1+ delta(k)/dzt(k) + delta(k-1)/dzt(k) 
         enddo
         b_tri(nz) = 1+ delta(nz-1)/dzt(nz) 
         b_tri(ks) = 1+ delta(ks)/dzt(ks)   
         do k=ks,nz-1
          c_tri(k) = - delta(k)/dzt(k)
         enddo
         c_tri(nz)=0.0
         d_tri(ks:nz)=tr(i,j,ks:nz,taup1) 
         call solve_tridiag(a_tri(ks:nz),b_tri(ks:nz),c_tri(ks:nz),d_tri(ks:nz),sol(ks:nz),nz-ks+1)
         tr(i,j,ks:nz,taup1) = sol(ks:nz)
        endif
   enddo
 enddo
 if (istemp) then
      dtemp_iso = dtemp_iso + (tr(:,:,:,taup1)-aloc)/dt_tracer
 else
      dsalt_iso = dsalt_iso + (tr(:,:,:,taup1)-aloc)/dt_tracer
 endif
!---------------------------------------------------------------------------------
! dissipation by isopycnal mixing
!---------------------------------------------------------------------------------
if (enable_conserve_energy) then

 if (istemp) then
  bloc(:,:,:) = int_drhodT(:,:,:,tau)
 else
  bloc(:,:,:) = int_drhodS(:,:,:,tau)
 endif

 do k=1,nz
   do j=js_pe-onx+1,je_pe+onx-1
    do i=is_pe-onx+1,ie_pe+onx-1
     fxa = bloc(i,j,k)
     aloc(i,j,k) =+0.5*grav/rho_0*( (bloc(i+1,j,k)-fxa)*flux_east(i  ,j,k) &
                                   +(fxa-bloc(i-1,j,k))*flux_east(i-1,j,k) ) /(dxt(i)*cost(j))  &
                  +0.5*grav/rho_0*( (bloc(i,j+1,k)-fxa)*flux_north(i,j  ,k) &
                                   +(fxa-bloc(i,j-1,k))*flux_north(i,j-1,k) ) /(dyt(j)*cost(j)) 
    enddo
   enddo
 end do
 !---------------------------------------------------------------------------------
 ! dissipation interpolated on W-grid
 !---------------------------------------------------------------------------------
 do j=js_pe-onx,je_pe+onx
   do i=is_pe-onx,ie_pe+onx
     ks=kbot(i,j)
     if (ks>0) then
      k=ks; P_diss_iso(i,j,k) = P_diss_iso(i,j,k)+ &
                         0.5*(aloc(i,j,k)+aloc(i,j,k+1)) + 0.5*aloc(i,j,k)*dzw(max(1,k-1))/dzw(k)
      do k=ks+1,nz-1
       P_diss_iso(i,j,k) = P_diss_iso(i,j,k)+ 0.5*(aloc(i,j,k) +aloc(i,j,k+1))
      enddo
      k=nz; P_diss_iso(i,j,k) = P_diss_iso(i,j,k)+ aloc(i,j,k)
     endif
   enddo
 enddo
 !---------------------------------------------------------------------------------
 ! diagnose dissipation of dynamic enthalpy by explicit and implicit vertical mixing
 !---------------------------------------------------------------------------------
 if (istemp) then
  do k=1,nz-1
   do j=js_pe,je_pe
    do i=is_pe,ie_pe
     fxa = (-bloc(i,j,k+1) +bloc(i,j,k))/dzw(k)
     P_diss_iso(i,j,k)=P_diss_iso(i,j,k)  -grav/rho_0*fxa*flux_top(i,j,k)*maskW(i,j,k)  &
                    -grav/rho_0*fxa*K_33(i,j,k)*(temp(i,j,k+1,taup1)-temp(i,j,k,taup1))/dzw(k)*maskW(i,j,k)
    enddo
   enddo
  end do
 else
  do k=1,nz-1
   do j=js_pe,je_pe
    do i=is_pe,ie_pe
     fxa = (-bloc(i,j,k+1) +bloc(i,j,k))/dzw(k)
     P_diss_iso(i,j,k)= P_diss_iso(i,j,k)  -grav/rho_0*fxa*flux_top(i,j,k)*maskW(i,j,k) &
                    -grav/rho_0*fxa*K_33(i,j,k)*(salt(i,j,k+1,taup1)-salt(i,j,k,taup1))/dzw(k)*maskW(i,j,k)
    enddo
   enddo
  end do
 endif
endif
end subroutine isoneutral_diffusion








subroutine isoneutral_skew_diffusion(is_,ie_,js_,je_,nz_,tr,istemp)
!=======================================================================
!   Isopycnal skew diffusion for tracer, 
!   following functional formulation by Griffies et al 
!   Dissipation is calculated and stored in P_diss_skew
!   T/S changes are added to dtemp_iso/dsalt_iso
!=======================================================================
 use main_module   
 use isoneutral_module
 implicit none
 integer :: is_,ie_,js_,je_,nz_
 real*8 :: tr(is_:ie_,js_:je_,nz_,3)
 logical, intent(in) :: istemp
 integer :: i,j,k,kr,ip,jp,km1kr,kpkr,ks
 real*8 :: sumz,sumx,sumy
 real*8 :: aloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
 real*8 :: bloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
 real*8 :: fxa,diffloc

!-----------------------------------------------------------------------
!     construct total isoneutral tracer flux at east face of "T" cells 
!-----------------------------------------------------------------------
 do k=1,nz
  do j=js_pe,je_pe
   do i=is_pe-1,ie_pe
     diffloc =-0.25*(K_gm(i,j,k)+K_gm(i,j,max(1,k-1)) + K_gm(i+1,j,k)+K_gm(i+1,j,max(1,k-1)) ) 
     sumz = 0.
     do kr=0,1
       km1kr = max(k-1+kr,1)
       kpkr  = min(k+kr,nz)
       do ip=0,1
         sumz = sumz + diffloc*Ai_ez(i,j,k,ip,kr) *(tr(i+ip,j,kpkr,tau)-tr(i+ip,j,km1kr,tau))
       enddo
     enddo
     flux_east(i,j,k) = sumz/(4*dzt(k)) + (tr(i+1,j,k,tau)-tr(i,j,k,tau))/(cost(j)*dxu(i)) *K_11(i,j,k)
   enddo
  enddo
 enddo
!-----------------------------------------------------------------------
!     construct total isoneutral tracer flux at north face of "T" cells 
!-----------------------------------------------------------------------
 do k=1,nz
  do j=js_pe-1,je_pe
   do i=is_pe,ie_pe
     diffloc =-0.25*(K_gm(i,j,k)+K_gm(i,j,max(1,k-1)) + K_gm(i,j+1,k)+K_gm(i,j+1,max(1,k-1)) ) 
     sumz    = 0.
     do kr=0,1
       km1kr = max(k-1+kr,1)
       kpkr  = min(k+kr,nz)
       do jp=0,1
         sumz = sumz + diffloc*Ai_nz(i,j,k,jp,kr) *(tr(i,j+jp,kpkr,tau)-tr(i,j+jp,km1kr,tau))
       enddo
     enddo
     flux_north(i,j,k) = cosu(j)*( sumz/(4*dzt(k))+ (tr(i,j+1,k,tau)-tr(i,j,k,tau))/dyu(j)*K_22(i,j,k) )
   enddo
  enddo
 enddo
!-----------------------------------------------------------------------
!     compute the vertical tracer flux "flux_top" containing the K31
!     and K32 components which are to be solved explicitly. 
!-----------------------------------------------------------------------
 do k=1,nz-1
  do j=js_pe,je_pe
   do i=is_pe,ie_pe
    diffloc = K_gm(i,j,k) 
    sumx = 0.
    do ip=0,1
     do kr=0,1
      sumx = sumx + diffloc*Ai_bx(i,j,k,ip,kr)/cost(j)*(tr(i+ip,j,k+kr,tau) - tr(i-1+ip,j,k+kr,tau))
     enddo
    enddo
    sumy    = 0.
    do jp=0,1
     do kr=0,1
      sumy = sumy + diffloc*Ai_by(i,j,k,jp,kr)*cosu(j-1+jp)* (tr(i,j+jp,k+kr,tau)-tr(i,j-1+jp,k+kr,tau))
     enddo
    enddo
    flux_top(i,j,k) = sumx/(4*dxt(i)) +sumy/(4*dyt(j)*cost(j) ) 
   enddo
  enddo
 enddo
 flux_top(:,:,nz)=0.0
!---------------------------------------------------------------------------------
!     add explicit part 
!---------------------------------------------------------------------------------
 do j=js_pe,je_pe
   do i=is_pe,ie_pe
      aloc(i,j,:)=maskT(i,j,:)*( (flux_east(i,j,:)-  flux_east(i-1,j,:))/(cost(j)*dxt(i)) &
                                +(flux_north(i,j,:)- flux_north(i,j-1,:))/(cost(j)*dyt(j)) )
   enddo
 enddo
 k=1; aloc(:,:,k)=aloc(:,:,k)+maskT(:,:,k)*flux_top(:,:,k)/dzt(k)   
 do k=2,nz
   aloc(:,:,k)=aloc(:,:,k)+maskT(:,:,k)*(flux_top(:,:,k)- flux_top(:,:,k-1))/dzt(k)   
 enddo

 if (istemp) then
      dtemp_iso = dtemp_iso + aloc
 else
      dsalt_iso = dsalt_iso + aloc
 endif

 do j=js_pe,je_pe
   do i=is_pe,ie_pe
      tr(i,j,:,taup1)=tr(i,j,:,taup1)+dt_tracer*aloc(i,j,:)
   enddo
 enddo

!---------------------------------------------------------------------------------
! dissipation by isopycnal mixing
!---------------------------------------------------------------------------------
if (enable_conserve_energy) then

 if (istemp) then
  bloc(:,:,:) = int_drhodT(:,:,:,tau)
 else
  bloc(:,:,:) = int_drhodS(:,:,:,tau)
 endif

 do k=1,nz
   do j=js_pe-onx+1,je_pe+onx-1
    do i=is_pe-onx+1,ie_pe+onx-1
     fxa = bloc(i,j,k)
     aloc(i,j,k) =+0.5*grav/rho_0*( (bloc(i+1,j,k)-fxa)*flux_east(i  ,j,k) &
                                   +(fxa-bloc(i-1,j,k))*flux_east(i-1,j,k) ) /(dxt(i)*cost(j))  &
                  +0.5*grav/rho_0*( (bloc(i,j+1,k)-fxa)*flux_north(i,j  ,k) &
                                   +(fxa-bloc(i,j-1,k))*flux_north(i,j-1,k) ) /(dyt(j)*cost(j)) 
    enddo
   enddo
 end do
 !---------------------------------------------------------------------------------
 ! dissipation interpolated on W-grid
 !---------------------------------------------------------------------------------
 do j=js_pe-onx,je_pe+onx
   do i=is_pe-onx,ie_pe+onx
     ks=kbot(i,j)
     if (ks>0) then
      k=ks; P_diss_skew(i,j,k) = P_diss_skew(i,j,k)+ &
                         0.5*(aloc(i,j,k)+aloc(i,j,k+1)) + 0.5*aloc(i,j,k)*dzw(max(1,k-1))/dzw(k)
      do k=ks+1,nz-1
       P_diss_skew(i,j,k) = P_diss_skew(i,j,k)+ 0.5*(aloc(i,j,k) +aloc(i,j,k+1))
      enddo
      k=nz; P_diss_skew(i,j,k) = P_diss_skew(i,j,k)+ aloc(i,j,k)
     endif
   enddo
 enddo
 !---------------------------------------------------------------------------------
 ! dissipation by vertical component of skew mixing
 !---------------------------------------------------------------------------------
  do k=1,nz-1
   do j=js_pe,je_pe
    do i=is_pe,ie_pe
     fxa = (-bloc(i,j,k+1) +bloc(i,j,k))/dzw(k)
     P_diss_skew(i,j,k) = P_diss_skew(i,j,k)  -grav/rho_0*fxa*flux_top(i,j,k)*maskW(i,j,k)  
    enddo
   enddo
  end do

endif
end subroutine isoneutral_skew_diffusion







subroutine isoneutral_diffusion_all(is_,ie_,js_,je_,nz_,tr,istemp)
!=======================================================================
!   Isopycnal diffusion plus skew diffusion for tracer, 
!   following functional formulation by Griffies et al 
!   Dissipation is calculated and stored in P_diss_iso
!=======================================================================
 use main_module   
 use isoneutral_module
 implicit none
 integer :: is_,ie_,js_,je_,nz_
 !real*8 :: tr(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3)
 real*8 :: tr(is_:ie_,js_:je_,nz_,3)
 logical, intent(in) :: istemp
 integer :: i,j,k,kr,ip,jp,km1kr,kpkr,ks
 real*8 :: sumz,sumx,sumy
 real*8 :: a_tri(nz),b_tri(nz),c_tri(nz),d_tri(nz),delta(nz),sol(nz)
 real*8 :: aloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
 real*8 :: bloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
 real*8 :: fxa,diffloc

 if (enable_skew_diffusion) then
  aloc = K_gm
 else
  aloc = 0.0
 endif

!-----------------------------------------------------------------------
!     construct total isoneutral tracer flux at east face of "T" cells 
!-----------------------------------------------------------------------
 do k=1,nz
  do j=js_pe,je_pe
   do i=is_pe-1,ie_pe
     diffloc = 0.25*(K_iso(i,j,k)+K_iso(i,j,max(1,k-1)) + K_iso(i+1,j,k)+K_iso(i+1,j,max(1,k-1)) ) &
             - 0.25*(aloc(i,j,k)+aloc(i,j,max(1,k-1)) + aloc(i+1,j,k)+aloc(i+1,j,max(1,k-1)) )
     sumz = 0.
     do kr=0,1
       km1kr = max(k-1+kr,1)
       kpkr  = min(k+kr,nz)
       do ip=0,1
         sumz = sumz + diffloc*Ai_ez(i,j,k,ip,kr) *(tr(i+ip,j,kpkr,tau)-tr(i+ip,j,km1kr,tau))
       enddo
     enddo
     flux_east(i,j,k) = sumz/(4*dzt(k)) + (tr(i+1,j,k,tau)-tr(i,j,k,tau))/(cost(j)*dxu(i)) *K_11(i,j,k)
   enddo
  enddo
 enddo
!-----------------------------------------------------------------------
!     construct total isoneutral tracer flux at north face of "T" cells 
!-----------------------------------------------------------------------
 do k=1,nz
  do j=js_pe-1,je_pe
   do i=is_pe,ie_pe
     diffloc = 0.25*(K_iso(i,j,k)+K_iso(i,j,max(1,k-1)) + K_iso(i,j+1,k)+K_iso(i,j+1,max(1,k-1)) ) &
             - 0.25*(aloc(i,j,k)+aloc(i,j,max(1,k-1)) + aloc(i,j+1,k)+aloc(i,j+1,max(1,k-1)) )
     sumz    = 0.
     do kr=0,1
       km1kr = max(k-1+kr,1)
       kpkr  = min(k+kr,nz)
       do jp=0,1
         sumz = sumz + diffloc*Ai_nz(i,j,k,jp,kr) *(tr(i,j+jp,kpkr,tau)-tr(i,j+jp,km1kr,tau))
       enddo
     enddo
     flux_north(i,j,k) = cosu(j)*( sumz/(4*dzt(k))+ (tr(i,j+1,k,tau)-tr(i,j,k,tau))/dyu(j)*K_22(i,j,k) )
   enddo
  enddo
 enddo
!-----------------------------------------------------------------------
!     compute the vertical tracer flux "flux_top" containing the K31
!     and K32 components which are to be solved explicitly. The K33
!     component will be treated implicitly. Note that there are some
!     cancellations of dxu(i-1+ip) and dyu(jrow-1+jp) 
!-----------------------------------------------------------------------
 do k=1,nz-1
  do j=js_pe,je_pe
   do i=is_pe,ie_pe
    diffloc = K_iso(i,j,k) + aloc(i,j,k)
    sumx = 0.
    do ip=0,1
     do kr=0,1
      sumx = sumx + diffloc*Ai_bx(i,j,k,ip,kr)/cost(j)*(tr(i+ip,j,k+kr,tau) - tr(i-1+ip,j,k+kr,tau))
     enddo
    enddo
    sumy    = 0.
    do jp=0,1
     do kr=0,1
      sumy = sumy + diffloc*Ai_by(i,j,k,jp,kr)*cosu(j-1+jp)* (tr(i,j+jp,k+kr,tau)-tr(i,j-1+jp,k+kr,tau))
     enddo
    enddo
    flux_top(i,j,k) = sumx/(4*dxt(i)) +sumy/(4*dyt(j)*cost(j) ) 
   enddo
  enddo
 enddo
 flux_top(:,:,nz)=0.0
!---------------------------------------------------------------------------------
!     add explicit part 
!---------------------------------------------------------------------------------
 do j=js_pe,je_pe
   do i=is_pe,ie_pe
      aloc(i,j,:)=maskT(i,j,:)*( (flux_east(i,j,:)-  flux_east(i-1,j,:))/(cost(j)*dxt(i)) &
                                +(flux_north(i,j,:)- flux_north(i,j-1,:))/(cost(j)*dyt(j)) )
   enddo
 enddo
 k=1; aloc(:,:,k)=aloc(:,:,k)+maskT(:,:,k)*flux_top(:,:,k)/dzt(k)   
 do k=2,nz
   aloc(:,:,k)=aloc(:,:,k)+maskT(:,:,k)*(flux_top(:,:,k)- flux_top(:,:,k-1))/dzt(k)   
 enddo

 if (istemp) then
      dtemp_iso = aloc
 else
      dsalt_iso = aloc
 endif

 do j=js_pe,je_pe
   do i=is_pe,ie_pe
      tr(i,j,:,taup1)=tr(i,j,:,taup1)+dt_tracer*aloc(i,j,:)
   enddo
 enddo
!---------------------------------------------------------------------------------
!     add implicit part 
!---------------------------------------------------------------------------------
 aloc = tr(:,:,:,taup1)
 a_tri=0.0;b_tri=0.0; c_tri=0.0; d_tri=0.0; delta=0.0
 do j=js_pe,je_pe
    do i=is_pe,ie_pe
        ks=kbot(i,j)
        if (ks>0) then
         do k=ks,nz-1
          delta(k) = dt_tracer/dzw(k)*K_33(i,j,k)
         enddo
         delta(nz)=0.0
         do k=ks+1,nz
           a_tri(k) = - delta(k-1)/dzt(k)
         enddo
         a_tri(ks)=0.0
         do k=ks+1,nz-1
          b_tri(k) = 1+ delta(k)/dzt(k) + delta(k-1)/dzt(k) 
         enddo
         b_tri(nz) = 1+ delta(nz-1)/dzt(nz) 
         b_tri(ks) = 1+ delta(ks)/dzt(ks)   
         do k=ks,nz-1
          c_tri(k) = - delta(k)/dzt(k)
         enddo
         c_tri(nz)=0.0
         d_tri(ks:nz)=tr(i,j,ks:nz,taup1) 
         call solve_tridiag(a_tri(ks:nz),b_tri(ks:nz),c_tri(ks:nz),d_tri(ks:nz),sol(ks:nz),nz-ks+1)
         tr(i,j,ks:nz,taup1) = sol(ks:nz)
        endif
   enddo
 enddo
 if (istemp) then
      dtemp_iso = dtemp_iso + (tr(:,:,:,taup1)-aloc)/dt_tracer
 else
      dsalt_iso = dsalt_iso + (tr(:,:,:,taup1)-aloc)/dt_tracer
 endif
!---------------------------------------------------------------------------------
! dissipation by isopycnal mixing
!---------------------------------------------------------------------------------
if (enable_conserve_energy) then

 if (istemp) then
  bloc(:,:,:) = int_drhodT(:,:,:,tau)
 else
  bloc(:,:,:) = int_drhodS(:,:,:,tau)
 endif

 do k=1,nz
   do j=js_pe-onx+1,je_pe+onx-1
    do i=is_pe-onx+1,ie_pe+onx-1
     fxa = bloc(i,j,k)
     aloc(i,j,k) =+0.5*grav/rho_0*( (bloc(i+1,j,k)-fxa)*flux_east(i  ,j,k) &
                                   +(fxa-bloc(i-1,j,k))*flux_east(i-1,j,k) ) /(dxt(i)*cost(j))  &
                  +0.5*grav/rho_0*( (bloc(i,j+1,k)-fxa)*flux_north(i,j  ,k) &
                                   +(fxa-bloc(i,j-1,k))*flux_north(i,j-1,k) ) /(dyt(j)*cost(j)) 
    enddo
   enddo
 end do
 !---------------------------------------------------------------------------------
 ! dissipation interpolated on W-grid
 !---------------------------------------------------------------------------------
 do j=js_pe-onx,je_pe+onx
   do i=is_pe-onx,ie_pe+onx
     ks=kbot(i,j)
     if (ks>0) then
      k=ks; P_diss_iso(i,j,k) = P_diss_iso(i,j,k)+ &
                         0.5*(aloc(i,j,k)+aloc(i,j,k+1)) + 0.5*aloc(i,j,k)*dzw(max(1,k-1))/dzw(k)
      do k=ks+1,nz-1
       P_diss_iso(i,j,k) = P_diss_iso(i,j,k)+ 0.5*(aloc(i,j,k) +aloc(i,j,k+1))
      enddo
      k=nz; P_diss_iso(i,j,k) = P_diss_iso(i,j,k)+ aloc(i,j,k)
     endif
   enddo
 enddo
 !---------------------------------------------------------------------------------
 ! diagnose dissipation of dynamic enthalpy by explicit and implicit vertical mixing
 !---------------------------------------------------------------------------------
 if (istemp) then
  do k=1,nz-1
   do j=js_pe,je_pe
    do i=is_pe,ie_pe
     fxa = (-bloc(i,j,k+1) +bloc(i,j,k))/dzw(k)
     P_diss_iso(i,j,k)=P_diss_iso(i,j,k)  -grav/rho_0*fxa*flux_top(i,j,k)*maskW(i,j,k)  &
                    -grav/rho_0*fxa*K_33(i,j,k)*(temp(i,j,k+1,taup1)-temp(i,j,k,taup1))/dzw(k)*maskW(i,j,k)
    enddo
   enddo
  end do
 else
  do k=1,nz-1
   do j=js_pe,je_pe
    do i=is_pe,ie_pe
     fxa = (-bloc(i,j,k+1) +bloc(i,j,k))/dzw(k)
     P_diss_iso(i,j,k)= P_diss_iso(i,j,k)  -grav/rho_0*fxa*flux_top(i,j,k)*maskW(i,j,k) &
                    -grav/rho_0*fxa*K_33(i,j,k)*(salt(i,j,k+1,taup1)-salt(i,j,k,taup1))/dzw(k)*maskW(i,j,k)
    enddo
   enddo
  end do
 endif
endif
end subroutine isoneutral_diffusion_all











subroutine adv_flux_superbee_spectral(is_,ie_,js_,je_,np_,adv_fe,adv_fn,adv_ft,var,uvel,vvel,wvel)
!=======================================================================
! Calculates advection of a tracer in spectral space
!=======================================================================
      use main_module   
      use idemix_module   
      implicit none
      integer, intent(in) :: is_,ie_,js_,je_,np_
      real*8, intent(inout) :: adv_fe(is_:ie_,js_:je_,np_), adv_fn(is_:ie_,js_:je_,np_)
      real*8, intent(inout) :: adv_ft(is_:ie_,js_:je_,np_),    var(is_:ie_,js_:je_,np_,3)
      real*8, intent(in) :: uvel(is_:ie_,js_:je_,np_),vvel(is_:ie_,js_:je_,np_),wvel(is_:ie_,js_:je_,np_)
      integer :: i,j,k,km1,kp2
      real*8 :: Rjp,Rj,Rjm,uCFL=0.5,Cr
      real*8 :: Limiter
      Limiter(Cr)=max(0.D0,max(min(1.D0,2.D0*Cr), min(2.D0,Cr))) 
      
      do k=2,np-1
       do j=js_pe,je_pe
        do i=is_pe-1,ie_pe
         uCFL = ABS( uvel(i,j,k)*dt_tracer/(cost(j)*dxt( min(nx,max(1,i)) )) )
         Rjp=(var(i+2,j,k,tau)-var(i+1,j,k,tau))*maskUp(i+1,j,k)
         Rj =(var(i+1,j,k,tau)-var(i  ,j,k,tau))*maskUp(i  ,j,k)
         Rjm=(var(i  ,j,k,tau)-var(i-1,j,k,tau))*maskUp(i-1,j,k)
         IF (Rj.NE.0.) THEN
          IF (uvel(i,j,k).GT.0) THEN; Cr=Rjm/Rj; ELSE; Cr=Rjp/Rj; ENDIF
         ELSE
          IF (uvel(i,j,k).GT.0) THEN; Cr=Rjm*1.E20; ELSE; Cr=Rjp*1.E20; ENDIF
         ENDIF
         Cr=Limiter(Cr)
         adv_fe(i,j,k) = uvel(i,j,k)*(var(i+1,j,k,tau)+var(i,j,k,tau))*0.5d0   &
                                -ABS(uvel(i,j,k))*((1.-Cr)+uCFL*Cr)*Rj*0.5d0
        enddo
       enddo
      enddo

      do k=2,np-1
       do j=js_pe-1,je_pe
        do i=is_pe,ie_pe
         Rjp=(var(i,j+2,k,tau)-var(i,j+1,k,tau))*maskVp(i,j+1,k)
         Rj =(var(i,j+1,k,tau)-var(i,j  ,k,tau))*maskVp(i,j  ,k)
         Rjm=(var(i,j  ,k,tau)-var(i,j-1,k,tau))*maskVp(i,j-1,k)
         uCFL = ABS( vvel(i,j,k)*dt_tracer/dyt( min(ny,max(1,j)) ) )
         IF (Rj.NE.0.) THEN
          IF (vvel(i,j,k).GT.0) THEN; Cr=Rjm/Rj; ELSE; Cr=Rjp/Rj; ENDIF
         ELSE
          IF (vvel(i,j,k).GT.0) THEN; Cr=Rjm*1.E20; ELSE; Cr=Rjp*1.E20; ENDIF
         ENDIF
         Cr=Limiter(Cr)
         adv_fn(i,j,k) = vvel(i,j,k)*(var(i,j+1,k,tau)+var(i,j,k,tau))*0.5d0   &
                                -ABS(vvel(i,j,k))*((1.-Cr)+uCFL*Cr)*Rj*0.5d0
        enddo
       enddo
      enddo
 
      do k=1,np-1
       kp2=k+2; if (kp2>np) kp2=3
       km1=k-1; if (km1<1) km1=np-2
       do j=js_pe,je_pe
        do i=is_pe,ie_pe
         Rjp=(var(i,j,kp2,tau)-var(i,j,k+1,tau))*maskWp(i,j,k+1)
         Rj =(var(i,j,k+1,tau)-var(i,j,k  ,tau))*maskWp(i,j,k  )
         Rjm=(var(i,j,k  ,tau)-var(i,j,km1,tau))*maskWp(i,j,km1)
         uCFL = ABS( wvel(i,j,k)*dt_tracer/dphit(k) )
         IF (Rj.NE.0.) THEN
          IF (wvel(i,j,k).GT.0) THEN; Cr=Rjm/Rj; ELSE; Cr=Rjp/Rj; ENDIF
         ELSE
          IF (wvel(i,j,k).GT.0) THEN; Cr=Rjm*1.E20; ELSE; Cr=Rjp*1.E20; ENDIF
         ENDIF
         Cr=Limiter(Cr)
         adv_ft(i,j,k) = wvel(i,j,k)*(var(i,j,k+1,tau)+var(i,j,k,tau))*0.5d0   &
                                -ABS(wvel(i,j,k))*((1.-Cr)+uCFL*Cr)*Rj*0.5d0
        enddo
       enddo
      enddo
end subroutine adv_flux_superbee_spectral





subroutine reflect_flux(is_,ie_,js_,je_,np_,adv_fe,adv_fn)
!=======================================================================
! refection boundary condition for advective flux in spectral space
!=======================================================================
 use main_module   
 use idemix_module   
 implicit none
 integer, intent(in) :: is_,ie_,js_,je_,np_
 real*8, intent(inout) :: adv_fe(is_:ie_,js_:je_,np_),adv_fn(is_:ie_,js_:je_,np_)
 integer :: i,j,k,kk
 real*8 :: flux

  do k=2,np-1
   ! reflexion at southern boundary
   do j=js_pe-1,je_pe
    do i=is_pe,ie_pe
      kk=bc_south(i,j,k)
      if (kk>0 ) then
        flux = adv_fn(i,j+1,k)
        adv_fn(i,j,k)  = adv_fn(i,j,k)  + flux
        adv_fn(i,j,kk) = adv_fn(i,j,kk) - flux
      endif
    enddo
   enddo
   ! reflexion at northern boundary
   do j=js_pe-1,je_pe
    do i=is_pe,ie_pe
     kk=bc_north(i,j,k)
     if (kk>0 ) then
       flux = adv_fn(i,j-1,k)
       adv_fn(i,j,k)  = adv_fn(i,j,k)  + flux
       adv_fn(i,j,kk) = adv_fn(i,j,kk) - flux
     endif
    enddo
   enddo
   ! reflexion at western boundary
   do j=js_pe,je_pe
    do i=is_pe-1,ie_pe
     kk=bc_west(i,j,k)
     if (kk>0 ) then
      flux   = adv_fe(i+1,j,k)
      adv_fe(i,j,k)  = adv_fe(i,j,k)  + flux
      adv_fe(i,j,kk) = adv_fe(i,j,kk) - flux
     endif
    enddo
   enddo
   ! reflexion at eastern boundary
   do j=js_pe,je_pe
    do i=is_pe-1,ie_pe
     kk=bc_east(i,j,k)
     if (kk>0 ) then
      flux   = adv_fe(i-1,j,k)
      adv_fe(i,j,k)  = adv_fe(i,j,k)  + flux
      adv_fe(i,j,kk) = adv_fe(i,j,kk) - flux
     endif
    enddo
   enddo
  enddo
end subroutine reflect_flux






subroutine reflect_ini
!=======================================================================
! initialize indexing for reflection boundary conditions
!=======================================================================
 use main_module   
 use idemix_module   
 implicit none
 integer :: i,j,k,kk
 real*8 :: fxa

  if (my_pe==0) print'(/a/)','preparing reflection boundary conditions'
  do k=2,np-1
   

    ! southern boundary from pi to 2 pi
    if (phit(k) >= pi .and. phit(k) < 2*pi ) then   
      fxa=2*pi-phit(k)
      if (fxa < 0.) fxa = fxa + 2*pi
      if (fxa > 2*pi) fxa = fxa - 2*pi
      kk = minloc( (phit  - fxa)**2,1 )
      do j=js_pe-1,je_pe
        where (maskTp(is_pe:ie_pe,j,k) == 0.0 .and. maskTp(is_pe:ie_pe,j+1,k)== 1.0) bc_south(is_pe:ie_pe,j,k)=kk
      enddo
    !endif
    ! northern boundary von 0 bis pi
    !if ( phit(k) >= 0. .and. phit(k) <= pi ) then 
    else
      fxa=2*pi-phit(k)
      if (fxa < 0.) fxa = fxa + 2*pi
      if (fxa > 2*pi) fxa = fxa - 2*pi
      kk = minloc( (phit  - fxa)**2,1 )
      do j=js_pe-1,je_pe
        where (maskTp(is_pe:ie_pe,j,k) == 1.0 .and. maskTp(is_pe:ie_pe,j+1,k)== 0.0) bc_north(is_pe:ie_pe,j,k)=kk
      enddo
    endif

  enddo

  do k=2,np-1

    ! western boundary:  from 0.5 pi to 0.75 pi
    if (phit(k) >= pi/2 .and. phit(k) < 3*pi/2. ) then 

      fxa=pi- phit(k)
      if (fxa < 0.) fxa = fxa + 2*pi
      if (fxa > 2*pi) fxa = fxa - 2*pi
      kk = minloc( (phit  - fxa)**2,1 )
      do i=is_pe-1,ie_pe
        where (maskTp(i,js_pe:je_pe,k) == 0.0 .and. maskTp(i+1,js_pe:je_pe,k)== 1.0) bc_west(i,js_pe:je_pe,k)=kk
      enddo
    !endif
    ! eastern boundary:  from 0 to 0.5 pi   and from 0.75 pi to 2 pi 
    !if ( ( phit(k) >= 0. .and. phit(k) <= pi/2 )  .or. (phit(k) >= 3*pi/2. .and. phit(k) <= 2*pi ) ) then 
    else
      fxa=pi-phit(k)
      if (fxa < 0.) fxa = fxa + 2*pi
      if (fxa > 2*pi) fxa = fxa - 2*pi
      kk = minloc( (phit  - fxa)**2,1 )
      do i=is_pe-1,ie_pe
          where (maskTp(i,js_pe:je_pe,k) == 1.0 .and. maskTp(i+1,js_pe:je_pe,k)== 0.0) bc_east(i,js_pe:je_pe,k)=kk
      enddo
    endif
  enddo
end subroutine reflect_ini





subroutine calc_spectral_topo
!=======================================================================
!  spectral stuff related to topography
!=======================================================================
 use main_module   
 use idemix_module   
 implicit none
 integer :: i,j,k

  if (enable_idemix_M2 .or. enable_idemix_niw) then
   ! wavenumber grid  
   dphit=2.*pi/(np-2);  dphiu=dphit
   phit(1)=0.0-dphit(1); phiu(1)=phit(1)+dphit(1)/2.
   do i=2,np
    phit(i)=phit(i-1)+dphit(i); phiu(i)=phiu(i-1)+dphiu(i)
   enddo
   ! topographic mask for waves
   maskTp=0.0
   do j=js_pe,je_pe
    do i=is_pe,ie_pe
      if ( kbot(i,j) /=0 ) maskTp(i,j,:)=1.0
    enddo
   enddo
   call border_exchg_xyp(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,np,maskTp) 
   call setcyclic_xyp   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,np,maskTp)
   maskUp=maskTp
   do i=is_pe-onx,ie_pe+onx-1
    maskUp(i,:,:)=min(maskTp(i,:,:),maskTp(i+1,:,:))
   enddo
   call border_exchg_xyp(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,np,maskUp)
   call setcyclic_xyp   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,np,maskUp)
   maskVp=maskTp
   do j=js_pe-onx,je_pe+onx-1
    maskVp(:,j,:)=min(maskTp(:,j,:),maskTp(:,j+1,:))
   enddo
   call border_exchg_xyp(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,np,maskVp)
   call setcyclic_xyp   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,np,maskVp)
   maskWp=maskTp
   do k=1,np-1
     maskWp(:,:,k)=min(maskTp(:,:,k),maskTp(:,:,k+1))
   enddo
   ! precalculate mirrow boundary conditions
   call reflect_ini
   ! mark shelf for wave interaction
   call get_shelf
  endif
end subroutine calc_spectral_topo

 




subroutine adv_flux_2nd(is_,ie_,js_,je_,nz_,adv_fe,adv_fn,adv_ft,var)
!---------------------------------------------------------------------------------
!      2th order advective tracer flux
!---------------------------------------------------------------------------------
      use main_module   
      implicit none
      integer, intent(in) :: is_,ie_,js_,je_,nz_
      real*8, intent(inout) :: adv_fe(is_:ie_,js_:je_,nz_), adv_fn(is_:ie_,js_:je_,nz_)
      real*8, intent(inout) :: adv_ft(is_:ie_,js_:je_,nz_),    var(is_:ie_,js_:je_,nz_)
      integer :: i,j,k

      do k=1,nz
       do j=js_pe,je_pe
        do i=is_pe-1,ie_pe
         adv_fe(i,j,k)=0.5*(var(i,j,k) + var(i+1,j,k) )*u(i,j,k,tau)*maskU(i,j,k)
        enddo
       enddo
      enddo
      do k=1,nz
       do j=js_pe-1,je_pe
        do i=is_pe,ie_pe
         adv_fn(i,j,k)=cosu(j)*0.5*( var(i,j,k) + var(i,j+1,k) )*v(i,j,k,tau)*maskV(i,j,k)
        enddo
       enddo
      enddo
      do k=1,nz-1
       do j=js_pe,je_pe
        do i=is_pe,ie_pe
         adv_ft(i,j,k)=0.5*( var(i,j,k) + var(i,j,k+1) )*w(i,j,k,tau)*maskW(i,j,k)
        enddo
       enddo
      enddo
      adv_ft(:,:,nz)=0.0
end subroutine adv_flux_2nd




subroutine adv_flux_superbee(is_,ie_,js_,je_,nz_,adv_fe,adv_fn,adv_ft,var)
!---------------------------------------------------------------------------------
! from MITgcm
! Calculates advection of a tracer
! using second-order interpolation with a flux limiter:
! \begin{equation*}
! F^x_{adv} = U \overline{ \theta }^i
! - \frac{1}{2} \left(
!     [ 1 - \psi(C_r) ] |U|
!    + U \frac{u \Delta t}{\Delta x_c} \psi(C_r)
!              \right) \delta_i \theta
! \end{equation*}
! where the $\psi(C_r)$ is the limiter function and $C_r$ is
! the slope ratio.
!---------------------------------------------------------------------------------
      use main_module   
      implicit none
      integer, intent(in) :: is_,ie_,js_,je_,nz_
      real*8, intent(inout) :: adv_fe(is_:ie_,js_:je_,nz_), adv_fn(is_:ie_,js_:je_,nz_)
      real*8, intent(inout) :: adv_ft(is_:ie_,js_:je_,nz_),    var(is_:ie_,js_:je_,nz_)
      integer :: i,j,k,km1,kp2
      real*8 :: Rjp,Rj,Rjm,uCFL=0.5,Cr
     ! Statement function to describe flux limiter
     ! Upwind        Limiter(Cr)=0.
     ! Lax-Wendroff  Limiter(Cr)=1.
     ! Suberbee      Limiter(Cr)=max(0.,max(min(1.,2*Cr),min(2.,Cr)))
     ! Sweby         Limiter(Cr)=max(0.,max(min(1.,1.5*Cr),min(1.5.,Cr)))
      real*8 :: Limiter
      Limiter(Cr)=max(0.D0,max(min(1.D0,2.D0*Cr), min(2.D0,Cr))) 
     ! Limiter(Cr)=max(0.D0,max(min(1.D0,1.5D0*Cr), min(1.5D0,Cr))) 

      do k=1,nz
       do j=js_pe,je_pe
        do i=is_pe-1,ie_pe
         uCFL = ABS( u(i,j,k,tau)*dt_tracer/(cost(j)*dxt(i)) )
         Rjp=(var(i+2,j,k)-var(i+1,j,k))*maskU(i+1,j,k)
         Rj =(var(i+1,j,k)-var(i  ,j,k))*maskU(i  ,j,k)
         Rjm=(var(i  ,j,k)-var(i-1,j,k))*maskU(i-1,j,k)
         IF (Rj.NE.0.) THEN
          IF (u(i,j,k,tau).GT.0) THEN; Cr=Rjm/Rj; ELSE; Cr=Rjp/Rj; ENDIF
         ELSE
          IF (u(i,j,k,tau).GT.0) THEN; Cr=Rjm*1.E20; ELSE; Cr=Rjp*1.E20; ENDIF
         ENDIF
         Cr=Limiter(Cr)
         adv_fe(i,j,k) = u(i,j,k,tau)*(var(i+1,j,k)+var(i,j,k))*0.5d0   &
                                -ABS(u(i,j,k,tau))*((1.-Cr)+uCFL*Cr)*Rj*0.5d0
        enddo
       enddo
      enddo

      do k=1,nz
       do j=js_pe-1,je_pe
        do i=is_pe,ie_pe
         Rjp=(var(i,j+2,k)-var(i,j+1,k))*maskV(i,j+1,k)
         Rj =(var(i,j+1,k)-var(i,j  ,k))*maskV(i,j  ,k)
         Rjm=(var(i,j  ,k)-var(i,j-1,k))*maskV(i,j-1,k)
         uCFL = ABS( cosu(j)*v(i,j,k,tau)*dt_tracer/(cost(j)*dyt(j)) )
         IF (Rj.NE.0.) THEN
          IF (v(i,j,k,tau).GT.0) THEN; Cr=Rjm/Rj; ELSE; Cr=Rjp/Rj; ENDIF
         ELSE
          IF (v(i,j,k,tau).GT.0) THEN; Cr=Rjm*1.E20; ELSE; Cr=Rjp*1.E20; ENDIF
         ENDIF
         Cr=Limiter(Cr)
         adv_fn(i,j,k) = cosu(j)*v(i,j,k,tau)*(var(i,j+1,k)+var(i,j,k))*0.5d0   &
                    -ABS(cosu(j)*v(i,j,k,tau))*((1.-Cr)+uCFL*Cr)*Rj*0.5d0
        enddo
       enddo
      enddo
 
      do k=1,nz-1
       kp2=min(nz,k+2); !if (kp2>np) kp2=3
       km1=max(1,k-1) !if (km1<1) km1=np-2
       do j=js_pe,je_pe
        do i=is_pe,ie_pe
         Rjp=(var(i,j,kp2)-var(i,j,k+1))*maskW(i,j,k+1)
         Rj =(var(i,j,k+1)-var(i,j,k  ))*maskW(i,j,k  )
         Rjm=(var(i,j,k  )-var(i,j,km1))*maskW(i,j,km1)
         uCFL = ABS( w(i,j,k,tau)*dt_tracer/dzt(k) )
         IF (Rj.NE.0.) THEN
          IF (w(i,j,k,tau).GT.0) THEN; Cr=Rjm/Rj; ELSE; Cr=Rjp/Rj; ENDIF
         ELSE
          IF (w(i,j,k,tau).GT.0) THEN; Cr=Rjm*1.E20; ELSE; Cr=Rjp*1.E20; ENDIF
         ENDIF
         Cr=Limiter(Cr)
         adv_ft(i,j,k) = w(i,j,k,tau)*(var(i,j,k+1)+var(i,j,k))*0.5d0   &
                                -ABS(w(i,j,k,tau))*((1.-Cr)+uCFL*Cr)*Rj*0.5d0
        enddo
       enddo
      enddo
      adv_ft(:,:,nz)=0.0
end subroutine adv_flux_superbee




subroutine calculate_velocity_on_wgrid
!---------------------------------------------------------------------------------
! calculates advection velocity for tracer on W grid
!---------------------------------------------------------------------------------
 use main_module
 implicit none
 integer :: i,j,k
 !real*8 :: fxa,fxb

 ! lateral advection velocities on W grid
 do k=1,nz-1
  u_wgrid(:,:,k) = u(:,:,k+1,tau)*maskU(:,:,k+1)*0.5*dzt(k+1)/dzw(k) + u(:,:,k,tau)*maskU(:,:,k)*0.5*dzt(k)/dzw(k)
  v_wgrid(:,:,k) = v(:,:,k+1,tau)*maskV(:,:,k+1)*0.5*dzt(k+1)/dzw(k) + v(:,:,k,tau)*maskV(:,:,k)*0.5*dzt(k)/dzw(k)
 enddo
 k=nz
 u_wgrid(:,:,k) = u(:,:,k,tau)*maskU(:,:,k)*0.5*dzt(k)/dzw(k)
 v_wgrid(:,:,k) = v(:,:,k,tau)*maskV(:,:,k)*0.5*dzt(k)/dzw(k)

 ! redirect velocity at bottom and at topography
 k=1
 u_wgrid(:,:,k) = u_wgrid(:,:,k) + u(:,:,k,tau)*maskU(:,:,k)*0.5*dzt(k)/dzw(k)
 v_wgrid(:,:,k) = v_wgrid(:,:,k) + v(:,:,k,tau)*maskV(:,:,k)*0.5*dzt(k)/dzw(k) 
 do k=1,nz-1
  do j=js_pe-onx,je_pe+onx
    do i=is_pe-onx,ie_pe+onx-1
      if (maskW(i,j,k)*maskW(i+1,j,k) == 0d0 ) then
         u_wgrid(i,j,k+1) = u_wgrid(i,j,k+1)+u_wgrid(i,j,k)*dzw(k)/dzw(k+1)
         u_wgrid(i,j,k) = 0d0
      endif
   enddo
  enddo
  do j=js_pe-onx,je_pe+onx-1
    do i=is_pe-onx,ie_pe+onx
      if (maskW(i,j,k)*maskW(i,j+1,k) == 0d0 ) then
         v_wgrid(i,j,k+1) = v_wgrid(i,j,k+1)+v_wgrid(i,j,k)*dzw(k)/dzw(k+1)
         v_wgrid(i,j,k) = 0d0
      endif
   enddo
  enddo
 enddo

 ! vertical advection velocity on W grid from continuity
 w_wgrid(:,:,1)=0d0
 do k=1,nz
  do j=js_pe-onx+1,je_pe+onx
    do i=is_pe-onx+1,ie_pe+onx
      w_wgrid(i,j,k) = w_wgrid(i,j,max(1,k-1))-dzw(k)* &
               ((        u_wgrid(i,j,k)          -u_wgrid(i-1,j,k))/(cost(j)*dxt(i)) &
               +(cosu(j)*v_wgrid(i,j,k)-cosu(j-1)*v_wgrid(i,j-1,k))/(cost(j)*dyt(j)) )
   enddo
  enddo
 enddo

 ! test continuity
 !if ( modulo(itt*dt_tracer,ts_monint) < dt_tracer .and. .false.) then
 ! fxa=0;fxb=0;
 ! do j=js_pe,je_pe
 !  do i=is_pe,ie_pe
 !    fxa = fxa + w_wgrid(i,j,nz) *area_t(i,j)
 !    fxb = fxb +   w(i,j,nz,tau) *area_t(i,j)
 !  enddo
 ! enddo
 ! call global_sum(fxa); call global_sum(fxb); 
 ! if (my_pe==0) print'(a,e12.6,a)',' transport at sea surface on t grid = ',fxb,' m^3/s'
 ! if (my_pe==0) print'(a,e12.6,a)',' transport at sea surface on w grid = ',fxa,' m^3/s'
!
!
!  fxa=0;fxb=0;
!  do j=js_pe,je_pe
!   do i=is_pe,ie_pe
!     fxa = fxa + w_wgrid(i,j,nz)**2 *area_t(i,j)
!     fxb = fxb +   w(i,j,nz,tau)**2 *area_t(i,j)
!   enddo
!  enddo
!  call global_sum(fxa); call global_sum(fxb); 
!  if (my_pe==0) print'(a,e12.6,a)',' w variance on t grid = ',fxb,' (m^3/s)^2'
!  if (my_pe==0) print'(a,e12.6,a)',' w variance on w grid = ',fxa,' (m^3/s)^2'
!
! endif

end subroutine calculate_velocity_on_wgrid






subroutine adv_flux_superbee_wgrid(is_,ie_,js_,je_,nz_,adv_fe,adv_fn,adv_ft,var)
!---------------------------------------------------------------------------------
! Calculates advection of a tracer defined on Wgrid
!---------------------------------------------------------------------------------
      use main_module   
      implicit none
      integer, intent(in) :: is_,ie_,js_,je_,nz_
      real*8, intent(inout) :: adv_fe(is_:ie_,js_:je_,nz_), adv_fn(is_:ie_,js_:je_,nz_)
      real*8, intent(inout) :: adv_ft(is_:ie_,js_:je_,nz_),    var(is_:ie_,js_:je_,nz_)
      integer :: i,j,k,km1,kp2,kp1
      real*8 :: Rjp,Rj,Rjm,uCFL=0.5,Cr
     ! Statement function to describe flux limiter
     ! Upwind        Limiter(Cr)=0.
     ! Lax-Wendroff  Limiter(Cr)=1.
     ! Suberbee      Limiter(Cr)=max(0.,max(min(1.,2*Cr),min(2.,Cr)))
     ! Sweby         Limiter(Cr)=max(0.,max(min(1.,1.5*Cr),min(1.5.,Cr)))
      real*8 :: Limiter
      Limiter(Cr)=max(0.D0,max(min(1.D0,2.D0*Cr), min(2.D0,Cr))) 
      real*8 :: maskUtr,maskVtr,maskWtr
      maskUtr(i,j,k) = maskW(i+1,j,k)*maskW(i,j,k)
      maskVtr(i,j,k) = maskW(i,j+1,k)*maskW(i,j,k)
      maskWtr(i,j,k) = maskW(i,j,k+1)*maskW(i,j,k)

      do k=1,nz
       do j=js_pe,je_pe
        do i=is_pe-1,ie_pe
         uCFL = ABS( u_wgrid(i,j,k)*dt_tracer/(cost(j)*dxt(i)) )
         Rjp=(var(i+2,j,k)-var(i+1,j,k))*maskUtr(i+1,j,k)
         Rj =(var(i+1,j,k)-var(i  ,j,k))*maskUtr(i  ,j,k)
         Rjm=(var(i  ,j,k)-var(i-1,j,k))*maskUtr(i-1,j,k)
         IF (Rj.NE.0.) THEN
          IF (u_wgrid(i,j,k).GT.0) THEN; Cr=Rjm/Rj; ELSE; Cr=Rjp/Rj; ENDIF
         ELSE
          IF (u_wgrid(i,j,k).GT.0) THEN; Cr=Rjm*1.E20; ELSE; Cr=Rjp*1.E20; ENDIF
         ENDIF
         Cr=Limiter(Cr)
         adv_fe(i,j,k) = u_wgrid(i,j,k)*(var(i+1,j,k)+var(i,j,k))*0.5d0   &
                                -ABS(u_wgrid(i,j,k))*((1.-Cr)+uCFL*Cr)*Rj*0.5d0
        enddo
       enddo
      enddo

      do k=1,nz
       do j=js_pe-1,je_pe
        do i=is_pe,ie_pe
         Rjp=(var(i,j+2,k)-var(i,j+1,k))*maskVtr(i,j+1,k)
         Rj =(var(i,j+1,k)-var(i,j  ,k))*maskVtr(i,j  ,k)
         Rjm=(var(i,j  ,k)-var(i,j-1,k))*maskVtr(i,j-1,k)
         uCFL = ABS( cosu(j)*v_wgrid(i,j,k)*dt_tracer/(cost(j)*dyt(j)) )
         IF (Rj.NE.0.) THEN
          IF (v_wgrid(i,j,k).GT.0) THEN; Cr=Rjm/Rj; ELSE; Cr=Rjp/Rj; ENDIF
         ELSE
          IF (v_wgrid(i,j,k).GT.0) THEN; Cr=Rjm*1.E20; ELSE; Cr=Rjp*1.E20; ENDIF
         ENDIF
         Cr=Limiter(Cr)
         adv_fn(i,j,k) = cosu(j)*v_wgrid(i,j,k)*(var(i,j+1,k)+var(i,j,k))*0.5d0   &
                    -ABS(cosu(j)*v_wgrid(i,j,k))*((1.-Cr)+uCFL*Cr)*Rj*0.5d0
        enddo
       enddo
      enddo
 
      do k=1,nz-1
       kp1=min(nz-1,k+1) 
       kp2=min(nz,k+2); 
       km1=max(1,k-1) 
       do j=js_pe,je_pe
        do i=is_pe,ie_pe
         Rjp=(var(i,j,kp2)-var(i,j,k+1))*maskWtr(i,j,kp1)
         Rj =(var(i,j,k+1)-var(i,j,k  ))*maskWtr(i,j,k  )
         Rjm=(var(i,j,k  )-var(i,j,km1))*maskWtr(i,j,km1)
         uCFL = ABS( w_wgrid(i,j,k)*dt_tracer/dzw(k) )
         IF (Rj.NE.0.) THEN
          IF (w_wgrid(i,j,k).GT.0) THEN; Cr=Rjm/Rj; ELSE; Cr=Rjp/Rj; ENDIF
         ELSE
          IF (w_wgrid(i,j,k).GT.0) THEN; Cr=Rjm*1.E20; ELSE; Cr=Rjp*1.E20; ENDIF
         ENDIF
         Cr=Limiter(Cr)
         adv_ft(i,j,k) = w_wgrid(i,j,k)*(var(i,j,k+1)+var(i,j,k))*0.5d0   &
                                -ABS(w_wgrid(i,j,k))*((1.-Cr)+uCFL*Cr)*Rj*0.5d0
        enddo
       enddo
      enddo
      adv_ft(:,:,nz)=0.0
end subroutine adv_flux_superbee_wgrid







subroutine adv_flux_upwind_wgrid(is_,ie_,js_,je_,nz_,adv_fe,adv_fn,adv_ft,var)
!---------------------------------------------------------------------------------
! Calculates advection of a tracer defined on Wgrid
!---------------------------------------------------------------------------------
      use main_module   
      implicit none
      integer, intent(in) :: is_,ie_,js_,je_,nz_
      real*8, intent(inout) :: adv_fe(is_:ie_,js_:je_,nz_), adv_fn(is_:ie_,js_:je_,nz_)
      real*8, intent(inout) :: adv_ft(is_:ie_,js_:je_,nz_),    var(is_:ie_,js_:je_,nz_)
      integer :: i,j,k
      real*8 :: Rj
      real*8 :: maskUtr,maskVtr,maskWtr
      maskUtr(i,j,k) = maskW(i+1,j,k)*maskW(i,j,k)
      maskVtr(i,j,k) = maskW(i,j+1,k)*maskW(i,j,k)
      maskWtr(i,j,k) = maskW(i,j,k+1)*maskW(i,j,k)

      do k=1,nz
       do j=js_pe,je_pe
        do i=is_pe-1,ie_pe
         Rj =(var(i+1,j,k)-var(i  ,j,k))*maskUtr(i  ,j,k)
         adv_fe(i,j,k) = u_wgrid(i,j,k)*(var(i+1,j,k)+var(i,j,k))*0.5d0  -ABS(u_wgrid(i,j,k))*Rj*0.5d0
        enddo
       enddo
      enddo

      do k=1,nz
       do j=js_pe-1,je_pe
        do i=is_pe,ie_pe
         Rj =(var(i,j+1,k)-var(i,j  ,k))*maskVtr(i,j  ,k)
         adv_fn(i,j,k) = cosu(j)*v_wgrid(i,j,k)*(var(i,j+1,k)+var(i,j,k))*0.5d0 -ABS(cosu(j)*v_wgrid(i,j,k))*Rj*0.5d0
        enddo
       enddo
      enddo
 
      do k=1,nz-1
       do j=js_pe,je_pe
        do i=is_pe,ie_pe
         Rj =(var(i,j,k+1)-var(i,j,k  ))*maskWtr(i,j,k  )
         adv_ft(i,j,k) = w_wgrid(i,j,k)*(var(i,j,k+1)+var(i,j,k))*0.5d0 -ABS(w_wgrid(i,j,k))*Rj*0.5d0
        enddo
       enddo
      enddo
      adv_ft(:,:,nz)=0.0
end subroutine adv_flux_upwind_wgrid


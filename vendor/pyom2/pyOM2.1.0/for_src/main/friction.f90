



subroutine explicit_vert_friction
!=======================================================================
!  explicit vertical friction
!  dissipation is calculated and added to K_diss_v
!=======================================================================
 use main_module   
 implicit none
 integer :: i,j,k
 real*8 :: diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz),fxa
 
 !---------------------------------------------------------------------------------
 ! vertical friction of zonal momentum
 !---------------------------------------------------------------------------------
 do k=1,nz-1
  do j=js_pe-1,je_pe
   do i=is_pe-1,ie_pe
    fxa = 0.5*(kappaM(i,j,k)+kappaM(i+1,j,k))
    flux_top(i,j,k)=fxa*(u(i,j,k+1,tau)-u(i,j,k,tau))/dzw(k)*maskU(i,j,k+1)*maskU(i,j,k)
   enddo
  enddo
 enddo
 flux_top(:,:,nz)=0d0
 k=1; du_mix(:,:,k) = flux_top(:,:,k)/dzt(k)*maskU(:,:,k)
 do k=2,nz
   du_mix(:,:,k) = (flux_top(:,:,k)-flux_top(:,:,k-1))/dzt(k)*maskU(:,:,k)
 enddo

 !---------------------------------------------------------------------------------
 ! diagnose dissipation by vertical friction of zonal momentum 
 !---------------------------------------------------------------------------------
  do k=1,nz-1
   do j=js_pe-1,je_pe
    do i=is_pe-1,ie_pe
     diss(i,j,k) = (u(i,j,k+1,tau)-u(i,j,k,tau))*flux_top(i,j,k)/dzw(k)  
    enddo
   enddo
  enddo
  diss(:,:,nz)=0.0
  call ugrid_to_tgrid(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,diss)
  K_diss_v = K_diss_v + diss 

 !---------------------------------------------------------------------------------
 ! vertical friction of meridional momentum
 !---------------------------------------------------------------------------------
 do k=1,nz-1
  do j=js_pe-1,je_pe
   do i=is_pe-1,ie_pe
    fxa = 0.5*(kappaM(i,j,k)+kappaM(i,j+1,k))
    flux_top(i,j,k)=fxa*(v(i,j,k+1,tau)-v(i,j,k,tau))/dzw(k)*maskV(i,j,k+1)*maskV(i,j,k)
   enddo
  enddo
 enddo
 flux_top(:,:,nz)=0d0
 k=1; dv_mix(:,:,k) = flux_top(:,:,k)/dzt(k)*maskV(:,:,k)
 do k=2,nz
   dv_mix(:,:,k) = (flux_top(:,:,k)-flux_top(:,:,k-1))/dzt(k)*maskV(:,:,k)
 enddo

 !---------------------------------------------------------------------------------
 ! diagnose dissipation by vertical friction of meridional momentum
 !---------------------------------------------------------------------------------
  do k=1,nz-1
   do j=js_pe-1,je_pe
    do i=is_pe-1,ie_pe
     diss(i,j,k) = (v(i,j,k+1,tau)-v(i,j,k,tau))*flux_top(i,j,k)/dzw(k)  
    enddo
   enddo
  enddo
  diss(:,:,nz)=0.0
  call vgrid_to_tgrid(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,diss)
  K_diss_v = K_diss_v + diss 

 if (.not.enable_hydrostatic) then
 !---------------------------------------------------------------------------------
 ! vertical friction of vertical momentum
 !---------------------------------------------------------------------------------
   do k=1,nz-1
    do j=js_pe-1,je_pe
     do i=is_pe-1,ie_pe
      fxa = 0.5*(kappaM(i,j,k)+kappaM(i,j,k+1))
      flux_top(i,j,k)=fxa*(w(i,j,k+1,tau)-w(i,j,k,tau))/dzt(k+1)*maskW(i,j,k+1)*maskW(i,j,k)
     enddo
    enddo
   enddo
   flux_top(:,:,nz)=0d0
   k=1; dw_mix(:,:,k) = flux_top(:,:,k)/dzw(k)*maskW(:,:,k)
   do k=2,nz
    dw_mix(:,:,k) = (flux_top(:,:,k)-flux_top(:,:,k-1))/dzw(k)*maskW(:,:,k)
   enddo

 !---------------------------------------------------------------------------------
 ! diagnose dissipation by vertical friction of vertical momentum
 !---------------------------------------------------------------------------------
 ! to be implemented
 endif

end subroutine explicit_vert_friction






subroutine implicit_vert_friction
!=======================================================================
!  vertical friction
!  dissipation is calculated and added to K_diss_v
!=======================================================================
 use main_module   
 implicit none
 integer :: i,j,k,ks
 real*8 :: a_tri(nz),b_tri(nz),c_tri(nz),d_tri(nz),delta(nz),fxa
 real*8 :: diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
 
 !---------------------------------------------------------------------------------
 ! implicit vertical friction of zonal momentum
 !---------------------------------------------------------------------------------
 do j=js_pe-1,je_pe
   do i=is_pe-1,ie_pe
    ks=max(kbot(i,j),kbot(i+1,j))
    if (ks>0) then
     do k=ks,nz-1
      fxa = 0.5*(kappaM(i,j,k)+kappaM(i+1,j,k))
      delta(k) = dt_mom/dzw(k)*fxa*maskU(i,j,k+1)*maskU(i,j,k)
     enddo
     delta(nz)=0.0
     a_tri(ks)=0.0
     do k=ks+1,nz
       a_tri(k) = - delta(k-1)/dzt(k)
     enddo
     b_tri(ks) = 1+ delta(ks)/dzt(ks)   
     do k=ks+1,nz-1
      b_tri(k) = 1+ delta(k)/dzt(k) + delta(k-1)/dzt(k) 
     enddo
     b_tri(nz) = 1+ delta(nz-1)/dzt(nz) 
     do k=ks,nz-1
      c_tri(k) = - delta(k)/dzt(k)
     enddo
     c_tri(nz)=0.0
     d_tri(ks:nz)=u(i,j,ks:nz,tau) 
     call solve_tridiag(a_tri(ks:nz),b_tri(ks:nz),c_tri(ks:nz),d_tri(ks:nz),u(i,j,ks:nz,taup1),nz-ks+1)
    endif
    du_mix(i,j,:)=(u(i,j,:,taup1)-u(i,j,:,tau))/dt_mom
   enddo
 enddo

 !---------------------------------------------------------------------------------
 ! diagnose dissipation by vertical friction of zonal momentum 
 !---------------------------------------------------------------------------------
  do k=1,nz-1
   do j=js_pe-1,je_pe
    do i=is_pe-1,ie_pe
     fxa = 0.5*(kappaM(i,j,k)+kappaM(i+1,j,k))
     flux_top(i,j,k)=fxa*(u(i,j,k+1,taup1)-u(i,j,k,taup1))/dzw(k)*maskU(i,j,k+1)*maskU(i,j,k)
    enddo
   enddo
  enddo
  do k=1,nz-1
   do j=js_pe-1,je_pe
    do i=is_pe-1,ie_pe
     diss(i,j,k) = (u(i  ,j,k+1,tau)-u(i  ,j,k,tau))*flux_top(i  ,j,k)/dzw(k)  
    enddo
   enddo
  enddo
  diss(:,:,nz)=0.0
  call ugrid_to_tgrid(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,diss)
  K_diss_v = K_diss_v + diss 

 !---------------------------------------------------------------------------------
 ! implicit vertical friction of meridional momentum
 !---------------------------------------------------------------------------------
 do j=js_pe-1,je_pe
   do i=is_pe-1,ie_pe
    ks=max(kbot(i,j),kbot(i,j+1))
    if (ks>0) then
     do k=ks,nz-1
      fxa = 0.5*(kappaM(i,j,k)+kappaM(i,j+1,k))
      delta(k) = dt_mom/dzw(k)*fxa*maskV(i,j,k+1)*maskV(i,j,k)
     enddo
     delta(nz)=0.0
     a_tri(ks)=0.0
     do k=ks+1,nz
       a_tri(k) = - delta(k-1)/dzt(k)
     enddo
     b_tri(ks) = 1+ delta(ks)/dzt(ks)   
     do k=ks+1,nz-1
      b_tri(k) = 1+ delta(k)/dzt(k) + delta(k-1)/dzt(k) 
     enddo
     b_tri(nz) = 1+ delta(nz-1)/dzt(nz) 
     do k=ks,nz-1
      c_tri(k) = - delta(k)/dzt(k)
     enddo
     c_tri(nz)=0.0
     d_tri(ks:nz)=v(i,j,ks:nz,tau) 
     call solve_tridiag(a_tri(ks:nz),b_tri(ks:nz),c_tri(ks:nz),d_tri(ks:nz),v(i,j,ks:nz,taup1),nz-ks+1)
    endif
    dv_mix(i,j,:)=(v(i,j,:,taup1)-v(i,j,:,tau))/dt_mom
   enddo
 enddo

 !---------------------------------------------------------------------------------
 ! diagnose dissipation by vertical friction of meridional momentum
 !---------------------------------------------------------------------------------
  do k=1,nz-1
   do j=js_pe-1,je_pe
    do i=is_pe-1,ie_pe
     fxa = 0.5*(kappaM(i,j,k)+kappaM(i,j+1,k))
     flux_top(i,j,k)=fxa*(v(i,j,k+1,taup1)-v(i,j,k,taup1))/dzw(k)*maskV(i,j,k+1)*maskV(i,j,k)
    enddo
   enddo
  enddo
  do k=1,nz-1
   do j=js_pe-1,je_pe
    do i=is_pe-1,ie_pe
     diss(i,j,k) = (v(i,j,k+1,tau)-v(i,j,k,tau))*flux_top(i,j,k)/dzw(k)  
    enddo
   enddo
  enddo
  diss(:,:,nz)=0.0
  call vgrid_to_tgrid(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,diss)
  K_diss_v = K_diss_v + diss 

 if (.not.enable_hydrostatic) then
  !if (my_pe==0) print'(/a/)','ERROR: implicit vertical friction for vertical velocity not implemented'
  !call halt_stop(' in implicit_vert_friction')

  do j=js_pe,je_pe
   do i=is_pe,ie_pe
    ks=kbot(i,j)
    if (ks>0) then
     do k=ks,nz-1
      delta(k) = dt_mom/dzt(k+1)*0.5*(kappaM(i,j,k)+kappaM(i,j,k+1))
     enddo
     delta(nz)=0.0
     do k=ks+1,nz-1
       a_tri(k) = - delta(k-1)/dzw(k)
     enddo
     a_tri(ks)=0.0
     a_tri(nz) = - delta(nz-1)/(0.5*dzw(nz))
     do k=ks+1,nz-1
      b_tri(k) = 1+ delta(k)/dzw(k) + delta(k-1)/dzw(k) 
     enddo
     b_tri(nz) = 1+ delta(nz-1)/(0.5*dzw(nz))         
     b_tri(ks) = 1+ delta(ks)/dzw(ks)                
     do k=ks,nz-1
      c_tri(k) = - delta(k)/dzw(k)
     enddo
     c_tri(nz)=0.0
     d_tri(ks:nz)=w(i,j,ks:nz,tau) 
     call solve_tridiag(a_tri(ks:nz),b_tri(ks:nz),c_tri(ks:nz),d_tri(ks:nz),w(i,j,ks:nz,taup1),nz-ks+1)
    endif
    dw_mix(i,j,:)=(w(i,j,:,taup1)-w(i,j,:,tau))/dt_mom
   enddo
  enddo

 !---------------------------------------------------------------------------------
 ! diagnose dissipation by vertical friction of vertical momentum
 !---------------------------------------------------------------------------------
  do k=1,nz-1
   do j=js_pe-1,je_pe
    do i=is_pe-1,ie_pe
     fxa = 0.5*(kappaM(i,j,k)+kappaM(i,j,k+1))
     flux_top(i,j,k)=fxa*(w(i,j,k+1,taup1)-w(i,j,k,taup1))/dzt(k+1)*maskW(i,j,k+1)*maskW(i,j,k)
    enddo
   enddo
  enddo
  do k=1,nz-1
   do j=js_pe-1,je_pe
    do i=is_pe-1,ie_pe
     diss(i,j,k) = (w(i,j,k+1,tau)-w(i,j,k,tau))*flux_top(i,j,k)/dzt(k+1)  
    enddo
   enddo
  enddo
  diss(:,:,nz)=0.0
  K_diss_v = K_diss_v + diss 

 endif

end subroutine implicit_vert_friction




subroutine rayleigh_friction
!=======================================================================
!  interior Rayleigh friction   
!  dissipation is calculated and added to K_diss_bot
!=======================================================================
 use main_module   
 implicit none
 integer :: k
 real*8 :: diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)

 do k=1,nz
  du_mix(:,:,k)=du_mix(:,:,k) - maskU(:,:,k)*r_ray*u(:,:,k,tau)
 enddo
 if (enable_conserve_energy) then
  do k=1,nz
   diss(:,:,k) = maskU(:,:,k)*r_ray*u(:,:,k,tau)**2
  enddo
  call calc_diss(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,K_diss_bot,'U')
 endif

 do k=1,nz
   dv_mix(:,:,k)=dv_mix(:,:,k) - maskV(:,:,k)*r_ray*v(:,:,k,tau)
 enddo
 if (enable_conserve_energy) then
  do k=1,nz
   diss(:,:,k) = maskV(:,:,k)*r_ray*v(:,:,k,tau)**2
  enddo
  call calc_diss(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,K_diss_bot,'V')
 endif

 if (.not.enable_hydrostatic) then
  if (my_pe==0) print'(/a/)','ERROR: rayleigh friction for vertical velocity not implemented'
  call halt_stop(' in rayleigh_friction')
 endif
end subroutine rayleigh_friction






subroutine linear_bottom_friction
!=======================================================================
!   linear bottom friction   
!   dissipation is calculated and added to K_diss_bot
!=======================================================================
 use main_module   
 implicit none
 integer :: i,j,k
 real*8 :: diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)

 if (enable_bottom_friction_var) then

 !---------------------------------------------------------------------------------
 ! with spatially varying coefficient
 !---------------------------------------------------------------------------------
  do j=js_pe,je_pe
   do i=is_pe-1,ie_pe
    k=max(kbot(i,j),kbot(i+1,j))
    if (k>0) du_mix(i,j,k)=du_mix(i,j,k) - maskU(i,j,k)*r_bot_var_u(i,j)*u(i,j,k,tau)
   enddo
  enddo
  if (enable_conserve_energy) then
   diss=0.0
   do j=js_pe,je_pe
    do i=is_pe-1,ie_pe
     k=max(kbot(i,j),kbot(i+1,j))
     if (k>0) diss(i,j,k)= maskU(i,j,k)*r_bot_var_u(i,j)*u(i,j,k,tau)**2
    enddo
   enddo
   call calc_diss(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,K_diss_bot,'U')
  endif

  do j=js_pe-1,je_pe
   do i=is_pe,ie_pe
    k=max(kbot(i,j+1),kbot(i,j))
    if (k>0) dv_mix(i,j,k)=dv_mix(i,j,k) - maskV(i,j,k)*r_bot_var_v(i,j)*v(i,j,k,tau)
   enddo
  enddo
  if (enable_conserve_energy) then
   diss=0.0
   do j=js_pe-1,je_pe
    do i=is_pe,ie_pe
     k=max(kbot(i,j+1),kbot(i,j))
     if (k>0) diss(i,j,k) = maskV(i,j,k)*r_bot_var_v(i,j)*v(i,j,k,tau)**2
    enddo
   enddo
   call calc_diss(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,K_diss_bot,'V')
  endif

 else
 !---------------------------------------------------------------------------------
 ! with constant coefficient
 !---------------------------------------------------------------------------------
  do j=js_pe,je_pe
   do i=is_pe-1,ie_pe
    k=max(kbot(i,j),kbot(i+1,j))
    if (k>0) du_mix(i,j,k)=du_mix(i,j,k) - maskU(i,j,k)*r_bot*u(i,j,k,tau)
   enddo
  enddo
  if (enable_conserve_energy) then
   diss=0.0
   do j=js_pe,je_pe
    do i=is_pe-1,ie_pe
     k=max(kbot(i,j),kbot(i+1,j))
     if (k>0) diss(i,j,k)= maskU(i,j,k)*r_bot*u(i,j,k,tau)**2
    enddo
   enddo
   call calc_diss(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,K_diss_bot,'U')
  endif
 
  do j=js_pe-1,je_pe
   do i=is_pe,ie_pe
    k=max(kbot(i,j+1),kbot(i,j))
    if (k>0) dv_mix(i,j,k)=dv_mix(i,j,k) - maskV(i,j,k)*r_bot*v(i,j,k,tau)
   enddo
  enddo
  if (enable_conserve_energy) then
   diss=0.0
   do j=js_pe-1,je_pe
    do i=is_pe,ie_pe
     k=max(kbot(i,j+1),kbot(i,j))
     if (k>0) diss(i,j,k) = maskV(i,j,k)*r_bot*v(i,j,k,tau)**2
    enddo
   enddo
   call calc_diss(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,K_diss_bot,'V')
  endif
 endif

 if (.not.enable_hydrostatic) then
  if (my_pe==0) print'(/a/)','ERROR: bottom friction for vertical velocity not implemented'
  call halt_stop(' in bottom_friction')
 endif
end subroutine linear_bottom_friction





subroutine quadratic_bottom_friction
!=======================================================================
! quadratic bottom friction   
! dissipation is calculated and added to K_diss_bot
!=======================================================================
 use main_module   
 implicit none
 integer :: i,j,k
 real*8 :: diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz),fxa
 real*8 :: aloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)


 ! we might want to account for EKE in the drag, also a tidal residual
 aloc=0.0
 do j=js_pe,je_pe
   do i=is_pe-1,ie_pe
    k=max(kbot(i,j),kbot(i+1,j))
    if (k>0) then
      fxa =       maskV(i  ,j,k)*v(i  ,j,k,tau)**2 + maskV(i  ,j-1,k)*v(i  ,j-1,k,tau)**2
      fxa = fxa + maskV(i+1,j,k)*v(i+1,j,k,tau)**2 + maskV(i+1,j-1,k)*v(i+1,j-1,k,tau)**2
      fxa = sqrt(u(i,j,k,tau)**2+ 0.25*fxa ) 
      aloc(i,j) =  maskU(i,j,k)*r_quad_bot*u(i,j,k,tau)*fxa/dzt(k)
      du_mix(i,j,k) = du_mix(i,j,k) - aloc(i,j)
    endif
   enddo
 enddo

 if (enable_conserve_energy) then
   diss=0.0
   do j=js_pe,je_pe
    do i=is_pe-1,ie_pe
     k=max(kbot(i,j),kbot(i+1,j))
     if (k>0) diss(i,j,k)= aloc(i,j)*u(i,j,k,tau)
    enddo
   enddo
   call calc_diss(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,K_diss_bot,'U')
 endif
 
 aloc=0.0
 do j=js_pe-1,je_pe
   do i=is_pe,ie_pe
    k=max(kbot(i,j+1),kbot(i,j))
    if (k>0) then
      fxa =       maskU(i,j  ,k)*u(i,j  ,k,tau)**2 + maskU(i-1,j  ,k)*u(i-1,j  ,k,tau)**2
      fxa = fxa + maskU(i,j+1,k)*u(i,j+1,k,tau)**2 + maskU(i-1,j+1,k)*u(i-1,j+1,k,tau)**2
      fxa = sqrt(v(i,j,k,tau)**2+ 0.25*fxa ) 
      aloc(i,j)= maskV(i,j,k)*r_quad_bot*v(i,j,k,tau)*fxa/dzt(k)
      dv_mix(i,j,k)=dv_mix(i,j,k) - aloc(i,j)
    endif
   enddo
 enddo

 if (enable_conserve_energy) then
   diss=0.0
   do j=js_pe-1,je_pe
    do i=is_pe,ie_pe
     k=max(kbot(i,j+1),kbot(i,j))
     if (k>0) diss(i,j,k) = aloc(i,j)*v(i,j,k,tau)
    enddo
   enddo
   call calc_diss(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,K_diss_bot,'V')
 endif

 if (.not.enable_hydrostatic) then
  if (my_pe==0) print'(/a/)','ERROR: bottom friction for vertical velocity not implemented'
  call halt_stop(' in quadratic_bottom_friction')
 endif
end subroutine quadratic_bottom_friction





subroutine harmonic_friction
!=======================================================================
! horizontal harmonic friction   
! dissipation is calculated and added to K_diss_h
!=======================================================================
 use main_module   
 implicit none
 integer :: i,j,k
 integer :: is,ie,js,je
 real*8 :: diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz),fxa

 is = is_pe-onx; ie = ie_pe+onx; js = js_pe-onx; je = je_pe+onx

 !---------------------------------------------------------------------------------
 ! Zonal velocity
 !---------------------------------------------------------------------------------
 if (enable_hor_friction_cos_scaling) then
  do j=js,je
   fxa = cost(j)**hor_friction_cosPower
   do i=is,ie-1
    flux_east(i,j,:)=fxa*A_h*(u(i+1,j,:,tau)-u(i,j,:,tau))/(cost(j)*dxt(i+1))*maskU(i+1,j,:)*maskU(i,j,:)
   enddo
  enddo
  do j=js,je-1
   fxa = cosu(j)**hor_friction_cosPower
   flux_north(:,j,:)=fxa*A_h*(u(:,j+1,:,tau)-u(:,j,:,tau))/dyu(j)*maskU(:,j+1,:)*maskU(:,j,:)*cosu(j)
  enddo 
 else
  do j=js,je
   do i=is,ie-1
    flux_east(i,j,:)=A_h*(u(i+1,j,:,tau)-u(i,j,:,tau))/(cost(j)*dxt(i+1))*maskU(i+1,j,:)*maskU(i,j,:)
   enddo
  enddo
  do j=js,je-1
    flux_north(:,j,:)=A_h*(u(:,j+1,:,tau)-u(:,j,:,tau))/dyu(j)*maskU(:,j+1,:)*maskU(:,j,:)*cosu(j)
  enddo 
 endif
 flux_east(ie,:,:)=0.
 flux_north(:,je,:)=0.

 !---------------------------------------------------------------------------------
 ! update tendency 
 !---------------------------------------------------------------------------------
 do j=js_pe,je_pe
   do i=is_pe,ie_pe
    du_mix(i,j,:)= du_mix(i,j,:) + maskU(i,j,:)*((flux_east(i,j,:) - flux_east(i-1,j,:))/(cost(j)*dxu(i)) &
                                                +(flux_north(i,j,:) - flux_north(i,j-1,:))/(cost(j)*dyt(j)) )
  enddo
 enddo

 if (enable_conserve_energy) then
 !---------------------------------------------------------------------------------
 ! diagnose dissipation by lateral friction
 !---------------------------------------------------------------------------------
  do k=1,nz
   do j=js_pe,je_pe
    do i=is_pe-1,ie_pe
     diss(i,j,k) =0.5*((u(i+1,j,k,tau)-u(i,j,k,tau))*flux_east(i,j,k) &
                      +(u(i,j,k,tau)-u(i-1,j,k,tau))*flux_east(i-1,j,k))/(cost(j)*dxu(i))  &
                 +0.5*((u(i,j+1,k,tau)-u(i,j,k,tau))*flux_north(i,j,k)+ &
                       (u(i,j,k,tau)-u(i,j-1,k,tau))*flux_north(i,j-1,k))/(cost(j)*dyt(j)) 
    enddo
   enddo
  enddo
  K_diss_h=0
  call calc_diss(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,K_diss_h,'U')
 endif

 !---------------------------------------------------------------------------------
 ! Meridional velocity
 !---------------------------------------------------------------------------------
 if (enable_hor_friction_cos_scaling) then
  do j=js,je
   fxa = cosu(j)**hor_friction_cosPower
   do i=is,ie-1
    flux_east(i,j,:)=fxa*A_h*(v(i+1,j,:,tau)-v(i,j,:,tau))/(cosu(j)*dxu(i)) *maskV(i+1,j,:)*maskV(i,j,:)
   enddo
  enddo
  do j=js,je-1
   fxa = cost(j+1)**hor_friction_cosPower
   flux_north(:,j,:)=fxa*A_h*(v(:,j+1,:,tau)-v(:,j,:,tau) )/dyt(j+1)*cost(j+1)*maskV(:,j,:)*maskV(:,j+1,:)
  enddo
 else
  do j=js,je
   do i=is,ie-1
    flux_east(i,j,:)=A_h*(v(i+1,j,:,tau)-v(i,j,:,tau))/(cosu(j)*dxu(i)) *maskV(i+1,j,:)*maskV(i,j,:)
   enddo
  enddo
  do j=js,je-1
   flux_north(:,j,:)=A_h*(v(:,j+1,:,tau)-v(:,j,:,tau) )/dyt(j+1)*cost(j+1)*maskV(:,j,:)*maskV(:,j+1,:)
  enddo
 endif
 flux_east(ie,:,:)=0.
 flux_north(:,je,:)=0.

 !---------------------------------------------------------------------------------
 ! update tendency 
 !---------------------------------------------------------------------------------
 do j=js_pe,je_pe
   do i=is_pe,ie_pe
    dv_mix(i,j,:)= dv_mix(i,j,:) + maskV(i,j,:)*( (flux_east(i,j,:) - flux_east(i-1,j,:))/(cosu(j)*dxt(i))  &
                                                 +(flux_north(i,j,:) - flux_north(i,j-1,:))/(dyu(j)*cosu(j)) )
  enddo
 enddo

 if (enable_conserve_energy) then
 !---------------------------------------------------------------------------------
 ! diagnose dissipation by lateral friction
 !---------------------------------------------------------------------------------
  do k=1,nz
   do j=js_pe-1,je_pe
    do i=is_pe,ie_pe
     diss(i,j,k) =0.5*((v(i+1,j,k,tau)-v(i,j,k,tau))*flux_east(i,j,k)+ &
                       (v(i,j,k,tau)-v(i-1,j,k,tau))*flux_east(i-1,j,k))/(cosu(j)*dxt(i)) &
                + 0.5*((v(i,j+1,k,tau)-v(i,j,k,tau))*flux_north(i,j,k)+ &
                       (v(i,j,k,tau)-v(i,j-1,k,tau))*flux_north(i,j-1,k))/(cosu(j)*dyu(j)) 
    enddo
   enddo
  enddo
  call calc_diss(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,K_diss_h,'V')
 endif

 if (.not.enable_hydrostatic) then

  if (enable_hor_friction_cos_scaling) then
   if (my_pe==0) print'(/a/)','ERROR: scaling of lateral friction for vertical velocity not implemented'
   call halt_stop(' in hamronic_friction')
  endif

  do j=js,je
   do i=is,ie-1
    flux_east(i,j,:)=A_h*(w(i+1,j,:,tau)-w(i,j,:,tau))/(cost(j)*dxu(i)) *maskW(i+1,j,:)*maskW(i,j,:)
   enddo
  enddo
  do j=js,je-1
   flux_north(:,j,:)=A_h*(w(:,j+1,:,tau)-w(:,j,:,tau))/dyu(j)*maskW(:,j+1,:)*maskW(:,j,:)*cosu(j)
  enddo
  flux_east(ie,:,:)=0.
  flux_north(:,je,:)=0.

 !---------------------------------------------------------------------------------
 ! update tendency 
 !---------------------------------------------------------------------------------
  do j=js_pe,je_pe
   do i=is_pe,ie_pe
    dw_mix(i,j,:)= dw_mix(i,j,:) + maskW(i,j,:)*( (flux_east(i,j,:) - flux_east(i-1,j,:))/(cost(j)*dxt(i))  &
                                                 +(flux_north(i,j,:) - flux_north(i,j-1,:))/(dyt(j)*cost(j)) )
  enddo
  enddo

 !---------------------------------------------------------------------------------
 ! diagnose dissipation by lateral friction
 !---------------------------------------------------------------------------------
  ! to be implemented
 endif
end subroutine harmonic_friction






subroutine biharmonic_friction
!=======================================================================
! horizontal biharmonic friction   
! dissipation is calculated and added to K_diss_h
!=======================================================================
 use main_module   
 implicit none
 integer :: i,j,is,ie,js,je
 real*8 :: del2(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz),fxa
 real*8 :: diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)

 if (.not.enable_hydrostatic) call halt_stop('biharmonic mixing for non-hydrostatic not yet implemented')

 is = is_pe-onx; ie = ie_pe+onx; js = js_pe-onx; je = je_pe+onx
 fxa = sqrt(abs(A_hbi))

 !---------------------------------------------------------------------------------
 ! Zonal velocity
 !---------------------------------------------------------------------------------
 do j=js,je
   do i=is,ie-1
    flux_east(i,j,:)=fxa*(u(i+1,j,:,tau)-u(i,j,:,tau))/(cost(j)*dxt(i+1))*maskU(i+1,j,:)*maskU(i,j,:)
   enddo
 enddo
 do j=js,je-1
    flux_north(:,j,:)=fxa*(u(:,j+1,:,tau)-u(:,j,:,tau))/dyu(j)*maskU(:,j+1,:)*maskU(:,j,:)*cosu(j)
 enddo 
 flux_east(ie,:,:)=0.
 flux_north(:,je,:)=0.

 do j=js+1,je
   do i=is+1,ie
    del2(i,j,:)= (flux_east(i,j,:) - flux_east(i-1,j,:))/(cost(j)*dxu(i)) &
                +(flux_north(i,j,:) - flux_north(i,j-1,:))/(cost(j)*dyt(j)) 
  enddo
 enddo

 do j=js,je
   do i=is,ie-1
    flux_east(i,j,:)=fxa*(del2(i+1,j,:)-del2(i,j,:))/(cost(j)*dxt(i+1))*maskU(i+1,j,:)*maskU(i,j,:)
   enddo
 enddo
 do j=js,je-1
    flux_north(:,j,:)=fxa*(del2(:,j+1,:)-del2(:,j,:))/dyu(j)*maskU(:,j+1,:)*maskU(:,j,:)*cosu(j)
 enddo 
 flux_east(ie,:,:)=0.
 flux_north(:,je,:)=0.

 !---------------------------------------------------------------------------------
 ! update tendency 
 !---------------------------------------------------------------------------------
 do j=js_pe,je_pe
   do i=is_pe,ie_pe
    du_mix(i,j,:)= du_mix(i,j,:) - maskU(i,j,:)*((flux_east(i,j,:) - flux_east(i-1,j,:))/(cost(j)*dxu(i)) &
                                                +(flux_north(i,j,:) - flux_north(i,j-1,:))/(cost(j)*dyt(j)) )
  enddo
 enddo

 if (enable_conserve_energy) then
 !---------------------------------------------------------------------------------
 ! diagnose dissipation by lateral friction
 !---------------------------------------------------------------------------------
  call border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_east) 
  call setcyclic_xyz   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_east)
  call border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_north) 
  call setcyclic_xyz   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_north)
  do j=js_pe,je_pe
    do i=is_pe-1,ie_pe
     diss(i,j,:) =-0.5*((u(i+1,j,:,tau)-u(i,j,:,tau))*flux_east(i,j,:) &
                       +(u(i,j,:,tau)-u(i-1,j,:,tau))*flux_east(i-1,j,:))/(cost(j)*dxu(i))  &
                 -0.5*((u(i,j+1,:,tau)-u(i,j,:,tau))*flux_north(i,j,:)+ &
                       (u(i,j,:,tau)-u(i,j-1,:,tau))*flux_north(i,j-1,:))/(cost(j)*dyt(j)) 
    enddo
  enddo
  K_diss_h=0
  call calc_diss(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,K_diss_h,'U')
 endif

 !---------------------------------------------------------------------------------
 ! Meridional velocity
 !---------------------------------------------------------------------------------
 do j=js,je
   do i=is,ie-1
    flux_east(i,j,:)=fxa*(v(i+1,j,:,tau)-v(i,j,:,tau))/(cosu(j)*dxu(i)) *maskV(i+1,j,:)*maskV(i,j,:)
   enddo
 enddo
 do j=js,je-1
   flux_north(:,j,:)=fxa*(v(:,j+1,:,tau)-v(:,j,:,tau) )/dyt(j+1)*cost(j+1)*maskV(:,j,:)*maskV(:,j+1,:)
 enddo
 flux_east(ie,:,:)=0.
 flux_north(:,je,:)=0.

 do j=js+1,je
  do i=is+1,ie
    del2(i,j,:)=  (flux_east(i,j,:) - flux_east(i-1,j,:))/(cosu(j)*dxt(i))  &
                 +(flux_north(i,j,:) - flux_north(i,j-1,:))/(dyu(j)*cosu(j)) 
  enddo
 enddo

 do j=js,je
   do i=is,ie-1
    flux_east(i,j,:)=fxa*(del2(i+1,j,:)-del2(i,j,:))/(cosu(j)*dxu(i)) *maskV(i+1,j,:)*maskV(i,j,:)
   enddo
 enddo
 do j=js,je-1
   flux_north(:,j,:)=fxa*(del2(:,j+1,:)-del2(:,j,:) )/dyt(j+1)*cost(j+1)*maskV(:,j,:)*maskV(:,j+1,:)
 enddo
 flux_east(ie,:,:)=0.
 flux_north(:,je,:)=0.

 !---------------------------------------------------------------------------------
 ! update tendency 
 !---------------------------------------------------------------------------------
 do j=js_pe,je_pe
   do i=is_pe,ie_pe
    dv_mix(i,j,:)= dv_mix(i,j,:) - maskV(i,j,:)*( (flux_east(i,j,:) - flux_east(i-1,j,:))/(cosu(j)*dxt(i))  &
                                                 +(flux_north(i,j,:) - flux_north(i,j-1,:))/(dyu(j)*cosu(j)) )
  enddo
 enddo

 if (enable_conserve_energy) then
 !---------------------------------------------------------------------------------
 ! diagnose dissipation by lateral friction
 !---------------------------------------------------------------------------------
  call border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_east) 
  call setcyclic_xyz   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_east)
  call border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_north) 
  call setcyclic_xyz   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,flux_north)
  do j=js_pe-1,je_pe
    do i=is_pe,ie_pe
     diss(i,j,:) =-0.5*((v(i+1,j,:,tau)-v(i,j,:,tau))*flux_east(i,j,:)+ &
                       (v(i,j,:,tau)-v(i-1,j,:,tau))*flux_east(i-1,j,:))/(cosu(j)*dxt(i)) &
                - 0.5*((v(i,j+1,:,tau)-v(i,j,:,tau))*flux_north(i,j,:)+ &
                       (v(i,j,:,tau)-v(i,j-1,:,tau))*flux_north(i,j-1,:))/(cosu(j)*dyu(j)) 
    enddo
  enddo
  call calc_diss(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,K_diss_h,'V')
 endif

end subroutine biharmonic_friction







subroutine momentum_sources
!=======================================================================
! other momentum sources
! dissipation is calculated and added to K_diss_bot
!=======================================================================
 use main_module   
 implicit none
 integer :: k
 real*8 :: diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)

 do k=1,nz
  du_mix(:,:,k)=du_mix(:,:,k) + maskU(:,:,k)*u_source(:,:,k)
 enddo
 if (enable_conserve_energy) then
  do k=1,nz
   diss(:,:,k) = -maskU(:,:,k)*u(:,:,k,tau)*u_source(:,:,k)
  enddo
  call calc_diss(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,K_diss_bot,'U')
 endif

 do k=1,nz
   dv_mix(:,:,k)=dv_mix(:,:,k) + maskV(:,:,k)*v_source(:,:,k)
 enddo
 if (enable_conserve_energy) then
  do k=1,nz
   diss(:,:,k) = -maskV(:,:,k)*v(:,:,k,tau)*v_source(:,:,k)
  enddo
  call calc_diss(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,K_diss_bot,'V')
 endif
end subroutine momentum_sources




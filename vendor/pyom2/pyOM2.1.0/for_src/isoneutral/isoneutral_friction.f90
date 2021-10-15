



subroutine isoneutral_friction
!=======================================================================
!  vertical friction using TEM formalism for eddy driven velocity
!=======================================================================
 use main_module   
 use isoneutral_module   
 use eke_module   
 implicit none
 integer :: i,j,k,ks
 real*8 :: a_tri(nz),b_tri(nz),c_tri(nz),d_tri(nz),delta(nz),fxa
 real*8 :: diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
 real*8 :: aloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
 
 if (enable_implicit_vert_friction) then
   aloc=u(:,:,:,taup1)
 else
   aloc=u(:,:,:,tau)
 endif

 ! implicit vertical friction of zonal momentum by GM
 do j=js_pe-1,je_pe
   do i=is_pe-1,ie_pe
    ks=max(kbot(i,j),kbot(i+1,j))
    if (ks>0) then
     do k=ks,nz-1
      fxa = 0.5*(kappa_gm(i,j,k)+kappa_gm(i+1,j,k))
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
     d_tri(ks:nz)=aloc(i,j,ks:nz)!  A u = d
     call solve_tridiag(a_tri(ks:nz),b_tri(ks:nz),c_tri(ks:nz),d_tri(ks:nz),u(i,j,ks:nz,taup1),nz-ks+1)
     du_mix(i,j,ks:nz)=du_mix(i,j,ks:nz)+ (u(i,j,ks:nz,taup1)-aloc(i,j,ks:nz))/dt_mom
    endif
   enddo
 enddo

 if (enable_conserve_energy) then
  ! diagnose dissipation 
  do k=1,nz-1
   do j=js_pe-1,je_pe
    do i=is_pe-1,ie_pe
     fxa = 0.5*(kappa_gm(i,j,k)+kappa_gm(i+1,j,k))
     flux_top(i,j,k)=fxa*(u(i,j,k+1,taup1)-u(i,j,k,taup1))/dzw(k)*maskU(i,j,k+1)*maskU(i,j,k)
    enddo
   enddo
  enddo
  do k=1,nz-1
   do j=js_pe-1,je_pe
    do i=is_pe-1,ie_pe
     diss(i,j,k) = (u(i,j,k+1,tau)-u(i,j,k,tau))*flux_top(i,j,k)/dzw(k)  
    enddo
   enddo
  enddo
  diss(:,:,nz)=0.0
   call ugrid_to_tgrid(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,diss)
  K_diss_gm = diss 
 endif

 if (enable_implicit_vert_friction) then
   aloc=v(:,:,:,taup1)
 else
   aloc=v(:,:,:,tau)
 endif

 ! implicit vertical friction of meridional momentum by GM
 do j=js_pe-1,je_pe
   do i=is_pe-1,ie_pe
    ks=max(kbot(i,j),kbot(i,j+1))
    if (ks>0) then
     do k=ks,nz-1
      fxa = 0.5*(kappa_gm(i,j,k)+kappa_gm(i,j+1,k))
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
     d_tri(ks:nz)=aloc(i,j,ks:nz)
     call solve_tridiag(a_tri(ks:nz),b_tri(ks:nz),c_tri(ks:nz),d_tri(ks:nz),v(i,j,ks:nz,taup1),nz-ks+1)
     dv_mix(i,j,ks:nz)=dv_mix(i,j,ks:nz)+ (v(i,j,ks:nz,taup1)-aloc(i,j,ks:nz))/dt_mom
    endif
   enddo
 enddo

 if (enable_conserve_energy) then
  ! diagnose dissipation 
  do k=1,nz-1
   do j=js_pe-1,je_pe
    do i=is_pe-1,ie_pe
     fxa = 0.5*(kappa_gm(i,j,k)+kappa_gm(i,j+1,k))
     flux_top(i,j,k)=fxa*(v(i,j,k+1,taup1)-v(i,j,k,taup1))/dzw(k)*maskV(i,j,k+1)*maskV(i,j,k)
    enddo
   enddo
  enddo
  do k=1,nz-1
   do j=js_pe-1,je_pe
    do i=is_pe-1,ie_pe
     diss(i,j,k) =(v(i,j  ,k+1,tau)-v(i,j  ,k,tau))*flux_top(i,j  ,k)/dzw(k) 
    enddo
   enddo
  enddo
  diss(:,:,nz)=0.0
  call vgrid_to_tgrid(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss,diss)
  K_diss_gm = K_diss_gm + diss 
 endif

end subroutine isoneutral_friction



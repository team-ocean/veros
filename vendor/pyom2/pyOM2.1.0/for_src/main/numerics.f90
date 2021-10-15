




subroutine u_centered_grid(dyt,dyu,yt,yu,n)
!---------------------------------------------------------------------------------
! setup u-centered grid based in Delta yt and the relations
! dyt_i = yu_i - yu_i-1 , yu_i = 0.5(yt_i+yt_(i+1)) , dyu_i = yt_(i+1)-yt_i
!---------------------------------------------------------------------------------
  implicit none
  integer, intent(in) :: n
  real*8, intent(in) :: dyt(n)
  real*8, intent(out) :: yu(n),yt(n),dyu(n)
  integer :: i 
  yu(1)=0
  do i=2,n  
   yu(i)=yu(i-1)+dyt(i)
  enddo
  yt(1)=yu(1)-dyt(1)*0.5
  do i=2,n
   yt(i) = 2*yu(i-1) - yt(i-1)
  enddo
  do i=1,n-1 
   dyu(i)= yt(i+1)-yt(i)
  enddo
  dyu(n)=2*dyt(n)- dyu(n-1)
end subroutine u_centered_grid





subroutine calc_grid
!---------------------------------------------------------------------------------
!    setup grid based on dxt,dyt,dzt and x_origin, y_origin
!---------------------------------------------------------------------------------
 use main_module   
 implicit none
 integer :: i,j
 real*8 :: aloc(nx,ny)
 real*8, dimension(1-onx:nx+onx) :: dxt_gl,dxu_gl,xt_gl,xu_gl
 real*8, dimension(1-onx:ny+onx) :: dyt_gl,dyu_gl,yt_gl,yu_gl

  aloc=0.
!--------------------------------------------------------------
! transfer from locally defined variables to global ones
!--------------------------------------------------------------
  aloc(is_pe:ie_pe,1) = dxt(is_pe:ie_pe)
  call pe0_recv_2D(nx,ny,aloc)
  call pe0_bcast(aloc,nx*ny)
  dxt_gl(1:nx) = aloc(:,1)

  if (enable_cyclic_x) then
   do i=1,onx
      dxt_gl(nx+i)=dxt_gl(i); dxt_gl(1-i)=dxt_gl(nx-i+1) 
   enddo
  else
   do i=1,onx
      dxt_gl(nx+i)=dxt_gl(nx); dxt_gl(1-i)=dxt_gl(1) 
   enddo
  endif

  aloc(1,js_pe:je_pe) = dyt(js_pe:je_pe)
  call pe0_recv_2D(nx,ny,aloc)
  call pe0_bcast(aloc,nx*ny)
  dyt_gl(1:ny) = aloc(1,:)

  do i=1,onx
      dyt_gl(ny+i)=dyt_gl(ny); dyt_gl(1-i)=dyt_gl(1) 
  enddo
!--------------------------------------------------------------
! grid in east/west direction
!--------------------------------------------------------------
  call u_centered_grid(dxt_gl,dxu_gl,xt_gl,xu_gl,nx+2*onx)
  xt_gl=xt_gl-xu_gl(1)+x_origin
  xu_gl=xu_gl-xu_gl(1)+x_origin

  if (enable_cyclic_x) then
   do i=1,onx
       xt_gl(nx+i)=xt_gl(i); xt_gl(1-i)=xt_gl(nx-i+1) 
       xu_gl(nx+i)=xt_gl(i); xu_gl(1-i)=xu_gl(nx-i+1) 
       dxu_gl(nx+i)=dxu_gl(i); dxu_gl(1-i)=dxu_gl(nx-i+1) 
   enddo
  endif

!--------------------------------------------------------------
! grid in north/south direction
!--------------------------------------------------------------
  call u_centered_grid(dyt_gl,dyu_gl,yt_gl,yu_gl,ny+2*onx)
  yt_gl=yt_gl-yu_gl(1)+y_origin
  yu_gl=yu_gl-yu_gl(1)+y_origin

  if (coord_degree) then
!--------------------------------------------------------------
! convert from degrees to pseudo cartesian grid
!--------------------------------------------------------------
    dxt_gl=dxt_gl*degtom; dxu_gl=dxu_gl*degtom;
    dyt_gl=dyt_gl*degtom; dyu_gl=dyu_gl*degtom;
  endif

!--------------------------------------------------------------
!  transfer to locally defined variables
!--------------------------------------------------------------
  xt(is_pe-onx:ie_pe+onx)  = xt_gl(is_pe-onx:ie_pe+onx)
  xu(is_pe-onx:ie_pe+onx)  = xu_gl(is_pe-onx:ie_pe+onx)
  dxu(is_pe-onx:ie_pe+onx) = dxu_gl(is_pe-onx:ie_pe+onx)
  dxt(is_pe-onx:ie_pe+onx) = dxt_gl(is_pe-onx:ie_pe+onx)

  yt(js_pe-onx:je_pe+onx)  = yt_gl(js_pe-onx:je_pe+onx)
  yu(js_pe-onx:je_pe+onx)  = yu_gl(js_pe-onx:je_pe+onx)
  dyu(js_pe-onx:je_pe+onx) = dyu_gl(js_pe-onx:je_pe+onx)
  dyt(js_pe-onx:je_pe+onx) = dyt_gl(js_pe-onx:je_pe+onx)

!--------------------------------------------------------------
! grid in vertical direction
!--------------------------------------------------------------
  call u_centered_grid(dzt,dzw,zt,zw,nz)
  !dzw(nz)=dzt(nz) !*0.5 ! this is account for in the model directly
  zt = zt - zw(nz); zw = zw - zw(nz)  ! zero at zw(nz) 

!--------------------------------------------------------------
! metric factors
!--------------------------------------------------------------
  if (coord_degree) then
   do j=js_pe-onx,je_pe+onx
    cost(j) = cos( yt(j)/180.*pi ) 
    cosu(j) = cos( yu(j)/180.*pi ) 
    tantr(j) = tan( yt(j)/180.*pi ) /radius
   enddo
  else
   cost=1.0;cosu=1.0;tantr=0.0
  endif

!--------------------------------------------------------------
! precalculate area of boxes
!--------------------------------------------------------------
  do j=js_pe-onx,je_pe+onx
   do i=is_pe-onx,ie_pe+onx
     area_t(i,j) = dxt(i)*cost(j)*dyt(j)
     area_u(i,j) = dxu(i)*cost(j)*dyt(j)
     area_v(i,j) = dxt(i)*cosu(j)*dyu(j)
   enddo
  enddo
end subroutine calc_grid



subroutine calc_beta
!--------------------------------------------------------------
! calculate beta = df/dy
!--------------------------------------------------------------
 use main_module   
 implicit none
 integer :: j
 do j=js_pe,je_pe
   beta(:,j) = 0.5*(  (coriolis_t(:,j+1)-coriolis_t(:,j))/dyu(j) + (coriolis_t(:,j)-coriolis_t(:,j-1))/dyu(j-1) )
 enddo
 call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,beta) 
 call setcyclic_xy   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,beta)
end subroutine calc_beta


subroutine calc_topo
!--------------------------------------------------------------
! calulate masks, total depth etc
!--------------------------------------------------------------
 use main_module   
 implicit none
 integer :: i,j,k

!--------------------------------------------------------------
! close domain
!--------------------------------------------------------------
  if (my_blk_j == 1)         kbot(:,1-onx:0)=0
  if (my_blk_j == n_pes_j)   kbot(:,ny+1:ny+onx)=0  
  if (.not. enable_cyclic_x) then
    if (my_blk_i == 1)         kbot(1-onx:0,:)=0
    if (my_blk_i == n_pes_i)   kbot(nx+1:nx+onx,:)=0  
  endif
  call border_exchg_xy_int(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,kbot) 
  call setcyclic_xy_int   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,kbot)
  
!--------------------------------------------------------------
! Land masks
!--------------------------------------------------------------
  maskT = 0.0
  do k=1,nz
   do j=js_pe-onx,je_pe+onx
      do i=is_pe-onx,ie_pe+onx
        if ( kbot(i,j)/=0 .and. kbot(i,j) <= k ) maskT(i,j,k)=1.0
      enddo
   enddo
  enddo
  call border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,maskT) 
  call setcyclic_xyz   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,maskT)
  maskU=maskT
  do i=is_pe-onx,ie_pe+onx-1
     maskU(i,:,:)=min(maskT(i,:,:),maskT(i+1,:,:))
  enddo
  call border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,maskU) 
  call setcyclic_xyz   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,maskU)
  maskV=maskT
  do j=js_pe-onx,je_pe+onx-1
      maskV(:,j,:)=min(maskT(:,j,:),maskT(:,j+1,:))
  enddo
  call border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,maskV) 
  call setcyclic_xyz   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,maskV)
  maskZ=maskT
  do j=js_pe-onx,je_pe+onx-1
   do i=is_pe-onx,ie_pe+onx-1
     maskZ(i,j,:)=min(maskT(i,j,:),maskT(i,j+1,:),maskT(i+1,j,:))
   enddo
  enddo
  call border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,maskZ) 
  call setcyclic_xyz   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,maskZ)
  maskW=maskT
  do k=1,nz-1
    maskW(:,:,k)=min(maskT(:,:,k),maskT(:,:,k+1))
  enddo
!--------------------------------------------------------------
! total depth
!--------------------------------------------------------------
  ht=0.0;hu=0.0;hv=0.0 
  do k=1,nz
    ht=ht+maskT(:,:,k)*dzt(k)
    hu=hu+maskU(:,:,k)*dzt(k)
    hv=hv+maskV(:,:,k)*dzt(k)
  enddo
  where ( hu /= 0.0) hur = 1./hu
  where ( hv /= 0.0) hvr = 1./hv
end subroutine calc_topo




subroutine calc_initial_conditions
!--------------------------------------------------------------
! calculate dyn. enthalp, etc
!--------------------------------------------------------------
 use main_module   
 implicit none
 integer :: i,j,k,n
 real*8 :: fxa,get_rho,get_dyn_enthalpy,get_int_drhodT,get_int_drhodS
  do n=1,3
    ! boundary exchange
     call border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,temp(:,:,:,n)) 
     call setcyclic_xyz   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,temp(:,:,:,n))
     call border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,salt(:,:,:,n)) 
     call setcyclic_xyz   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,salt(:,:,:,n))
    ! calculate density, etc
    do k=1,nz
     do j=js_pe-onx,je_pe+onx
      do i=is_pe-onx,ie_pe+onx
       if (salt(i,j,k,n) < 0.0) then
         if (my_pe==0) print*,' salinity <0 at i=',i,'j=',j,'k=',k
         call halt_stop('in main')
       endif
       rho(i,j,k,n) = get_rho(salt(i,j,k,n),temp(i,j,k,n),abs(zt(k)))*maskT(i,j,k)
       Hd(i,j,k,n) = get_dyn_enthalpy(salt(i,j,k,n),temp(i,j,k,n),abs(zt(k)))*maskT(i,j,k)
       int_drhodT(i,j,k,n) = get_int_drhodT(salt(i,j,k,n),temp(i,j,k,n),abs(zt(k)))
       int_drhodS(i,j,k,n) = get_int_drhodS(salt(i,j,k,n),temp(i,j,k,n),abs(zt(k)))
      enddo
     enddo
    enddo
    ! stability frequency
    do k=1,nz-1
     do j=js_pe-onx,je_pe+onx
      do i=is_pe-onx,ie_pe+onx
       fxa =  -grav/rho_0/dzw(k)*maskW(i,j,k)
       Nsqr(i,j,k,n) =fxa*(get_rho(salt(i,j,k+1,n),temp(i,j,k+1,n),abs(zt(k)))-rho(i,j,k,n))
      enddo
     enddo
    enddo
    Nsqr(:,:,nz,n)=Nsqr(:,:,nz-1,n)
  enddo
end subroutine calc_initial_conditions



subroutine ugrid_to_tgrid( is_,ie_,js_,je_,nz_,A,B)  ! for U-centered boxes, A and B can be identical
!---------------------------------------------------------------------------------
!---------------------------------------------------------------------------------
 use main_module   
 implicit none
 integer :: is_,ie_,js_,je_,nz_,i
 !real*8 :: A(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) 
 !real*8 :: B(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) 
 real*8, dimension(is_:ie_,js_:je_,nz_) :: A,B
 do i=ie_pe,is_pe,-1
  !B(i,:,:)=(dxu(i)*A(i,:,:)+dxu(i-1)*A(i-1,:,:))/(dxu(i)+dxu(i-1))  ! error should be 2*dxt(i) below fraction !!!!
  B(i,:,:)=(dxu(i)*A(i,:,:)+dxu(i-1)*A(i-1,:,:))/(2*dxt(i))  ! correct version
 enddo
end subroutine ugrid_to_tgrid


subroutine vgrid_to_tgrid( is_,ie_,js_,je_,nz_,A,B)  ! for V-centered boxes, A and B can be identical
!---------------------------------------------------------------------------------
!---------------------------------------------------------------------------------
 use main_module   
 implicit none
 integer :: is_,ie_,js_,je_,nz_,j,k
 !real*8 :: A(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) 
 !real*8 :: B(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) 
 real*8, dimension(is_:ie_,js_:je_,nz_) :: A,B
 do k=1,nz
  do j=je_pe,js_pe,-1
   B(:,j,k)=(area_v(:,j)*A(:,j,k)+area_v(:,j-1)*A(:,j-1,k))/(2*area_t(:,j))
  enddo
 enddo
end subroutine vgrid_to_tgrid




subroutine solve_tridiag(a,b,c,d,x,n)
      implicit none
!---------------------------------------------------------------------------------
!        a - sub-diagonal (means it is the diagonal below the main diagonal)
!        b - the main diagonal
!        c - sup-diagonal (means it is the diagonal above the main diagonal)
!        d - right part
!        x - the answer
!        n - number of equations
!---------------------------------------------------------------------------------
        integer,intent(in) :: n
        real*8,dimension(n),intent(in) :: a,b,c,d
        real*8,dimension(n),intent(out) :: x
        real*8,dimension(n) :: cp,dp
        real*8 :: m,fxa
        integer i
 
! initialize c-prime and d-prime
        cp(1) = c(1)/b(1)
        dp(1) = d(1)/b(1)
! solve for vectors c-prime and d-prime
         do i = 2,n
           m = b(i)-cp(i-1)*a(i)
           fxa = 1D0/m
           cp(i) = c(i)*fxa
           dp(i) = (d(i)-dp(i-1)*a(i))*fxa
         enddo
! initialize x
         x(n) = dp(n)
! solve for x from the vectors c-prime and d-prime
        do i = n-1, 1, -1
          x(i) = dp(i)-cp(i)*x(i+1)
        end do
end subroutine solve_tridiag





 subroutine calc_diss( is_,ie_,js_,je_,nz_,diss,K_diss,tag)
!---------------------------------------------------------------------------------
!---------------------------------------------------------------------------------
 use main_module   
 implicit none
 integer :: is_,ie_,js_,je_,nz_
 integer :: i,j,k,ks
 !real*8 :: diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
 !real*8 :: K_diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
 !real*8 :: diss_u(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
 real*8, dimension(is_:ie_,js_:je_,nz_) :: diss,K_diss,diss_u
 character*1 :: tag

 diss_u=0.0

 if (tag == 'U') then
  ! dissipation interpolated on W-grid
  do j=js_pe,je_pe
   do i=is_pe-1,ie_pe
     ks=max(kbot(i,j),kbot(i+1,j))
     if (ks>0) then
      k=ks; diss_u(i,j,k) = 0.5*(diss(i,j,k)+diss(i,j,k+1)) + 0.5*diss(i,j,k)*dzw(max(1,k-1))/dzw(k)
      do k=ks+1,nz-1
       diss_u(i,j,k) =  0.5*(diss(i,j,k) +diss(i,j,k+1))
      enddo
      k=nz; diss_u(i,j,k) = diss(i,j,k)
     endif
   enddo
  enddo
  ! dissipation interpolated from U-grid to T-grid
  call ugrid_to_tgrid(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss_u,diss_u)
  K_diss = K_diss + diss_u 
 else if (tag == 'V') then
  ! dissipation interpolated on W-grid
  do j=js_pe-1,je_pe
   do i=is_pe,ie_pe
     ks=max(kbot(i,j),kbot(i,j+1))
     if (ks>0) then
      k=ks; diss_u(i,j,k) = 0.5*(diss(i,j,k)+diss(i,j,k+1)) + 0.5*diss(i,j,k)*dzw(max(1,k-1))/dzw(k)
      do k=ks+1,nz-1
       diss_u(i,j,k) =  0.5*(diss(i,j,k) +diss(i,j,k+1))
      enddo
      k=nz; diss_u(i,j,k) = diss(i,j,k)
     endif
   enddo
  enddo
  ! dissipation interpolated from V-grid to T-grid
  call vgrid_to_tgrid(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,diss_u,diss_u)
  K_diss = K_diss + diss_u 
 else
  call halt_stop(' unknown tag in subr. vert_diss')
 endif
 end subroutine calc_diss









subroutine solve_non_hydrostatic
!=======================================================================
! solve for non hydrostatic pressure
!=======================================================================
 use main_module
 implicit none
 real*8 :: forc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
 integer :: i,j,k

 !---------------------------------------------------------------------------------
 ! integrate forward in time
 !---------------------------------------------------------------------------------
 do k=1,nz-1
   w(:,:,k,taup1)=w(:,:,k,tau)+dt_mom*(dw_mix(:,:,k)+(1.5+AB_eps)*dw(:,:,k,tau)-(0.5+AB_eps)*dw(:,:,k,taum1))*maskW(:,:,k)
 enddo
 !---------------------------------------------------------------------------------
 !        forcing for non-hydrostatic pressure
 !---------------------------------------------------------------------------------
 call border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,u(:,:,:,taup1)) 
 call setcyclic_xyz   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,u(:,:,:,taup1))
 call border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,v(:,:,:,taup1)) 
 call setcyclic_xyz   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,v(:,:,:,taup1))
 call border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,w(:,:,:,taup1)) 
 call setcyclic_xyz   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,w(:,:,:,taup1))
 do j=js_pe,je_pe
  do i=is_pe,ie_pe
    forc(i,j,:)= (        u(i,j,:,taup1)-          u(i-1,j,:,taup1) )/(cost(j)*dxt(i)) + &
                 (cosu(j)*v(i,j,:,taup1)-cosu(j-1)*v(i,j-1,:,taup1) )/(cost(j)*dyt(j))
  enddo
 enddo
 k=1; forc(:,:,k)= forc(:,:,k) + w(:,:,k,taup1)/dzt(k) 
 do k=2,nz
  forc(:,:,k)= forc(:,:,k) + (w(:,:,k,taup1)-w(:,:,k-1,taup1) )/dzt(k) 
 enddo
 forc=forc/dt_mom
 !---------------------------------------------------------------------------------
 !        solve for non-hydrostatic pressure
 !---------------------------------------------------------------------------------
 p_non_hydro(:,:,:,taup1)=2*p_non_hydro(:,:,:,tau)-p_non_hydro(:,:,:,taum1) ! first guess
 call congrad_non_hydro(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,forc,congr_itts_non_hydro)
 call border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,p_non_hydro(:,:,:,taup1)) 
 call setcyclic_xyz   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,p_non_hydro(:,:,:,taup1))
 if (itt==0) then 
       p_non_hydro(:,:,:,tau)  =p_non_hydro(:,:,:,taup1)
       p_non_hydro(:,:,:,taum1)=p_non_hydro(:,:,:,taup1)
 endif
 !---------------------------------------------------------------------------------
 ! add non-hydrostatic pressure gradient to tendencies
 !---------------------------------------------------------------------------------
 do j=js_pe,je_pe
  do i=is_pe,ie_pe
   u(i,j,:,taup1) = u(i,j,:,taup1) - dt_mom*( p_non_hydro(i+1,j,:,taup1)-p_non_hydro(i,j,:,taup1))/(dxu(i)*cost(j)) *maskU(i,j,:) 
   v(i,j,:,taup1) = v(i,j,:,taup1) - dt_mom*( p_non_hydro(i,j+1,:,taup1)-p_non_hydro(i,j,:,taup1)) /dyu(j)*maskV(i,j,:) 
  enddo
 enddo
 do k=1,nz-1
   w(:,:,k,taup1) = w(:,:,k,taup1) - dt_mom*( p_non_hydro(:,:,k+1,taup1)-p_non_hydro(:,:,k,taup1)) /dzw(k)*maskW(:,:,k) 
 enddo
end subroutine solve_non_hydrostatic






subroutine make_coeff_non_hydro(is_,ie_,js_,je_,nz_,cf)
!=======================================================================
!             A * dpsi = forc
!                       res = A * p
!          res = res +  cf(...,ii,jj,kk) * p(i+ii,j+jj,k+kk) 
!
!          forc = p_xx + p_yy + p_zz
!         forc = (p(i+1) - 2p(i) + p(i-1))  /dx^2 ...
!              = [ (p(i+1) - p(i))/dx - (p(i)-p(i-1))/dx ] /dx 
!=======================================================================
 use main_module   
 implicit none
 integer :: is_,ie_,js_,je_,nz_
 !real*8 :: cf(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3,3,3)
 real*8 :: cf(is_:ie_,js_:je_,nz_,3,3,3)
 real*8 :: mp,mm
 integer :: i,j,k

 cf=0d0
 do k=1,nz
  do j=js_pe,je_pe
   do i=is_pe,ie_pe
    mp=maskU(i,j,k)
    mm=maskU(i-1,j,k)
    cf(i,j,k, 0+2, 0+2, 0+2)= cf(i,j,k, 0+2, 0+2, 0+2)-mp/dxu(i  )/dxt(i) /cost(j)**2
    cf(i,j,k, 1+2, 0+2, 0+2)= cf(i,j,k, 1+2, 0+2, 0+2)+mp/dxu(i  )/dxt(i) /cost(j)**2
    cf(i,j,k, 0+2, 0+2, 0+2)= cf(i,j,k, 0+2, 0+2, 0+2)-mm/dxu(i-1)/dxt(i) /cost(j)**2
    cf(i,j,k,-1+2, 0+2, 0+2)= cf(i,j,k,-1+2, 0+2, 0+2)+mm/dxu(i-1)/dxt(i) /cost(j)**2

    mp=maskV(i,j,k)
    mm=maskV(i,j-1,k)
    cf(i,j,k, 0+2, 0+2, 0+2)= cf(i,j,k, 0+2, 0+2, 0+2)-mp/dyu(j  )/dyt(j) *cosu(j  )/cost(j)
    cf(i,j,k, 0+2, 1+2, 0+2)= cf(i,j,k, 0+2, 1+2, 0+2)+mp/dyu(j  )/dyt(j) *cosu(j  )/cost(j)
    cf(i,j,k, 0+2, 0+2, 0+2)= cf(i,j,k, 0+2, 0+2, 0+2)-mm/dyu(j-1)/dyt(j) *cosu(j-1)/cost(j)
    cf(i,j,k, 0+2,-1+2, 0+2)= cf(i,j,k, 0+2,-1+2, 0+2)+mm/dyu(j-1)/dyt(j) *cosu(j-1)/cost(j)
   enddo
  enddo
 enddo

 k=1
 do j=js_pe,je_pe
  do i=is_pe,ie_pe
    mp=maskW(i,j,k)
    cf(i,j,k, 0+2, 0+2, 0+2)= cf(i,j,k, 0+2, 0+2, 0+2)-mp/dzw(k  )/dzt(k) 
    cf(i,j,k, 0+2, 0+2, 1+2)= cf(i,j,k, 0+2, 0+2, 1+2)+mp/dzw(k  )/dzt(k) 
  enddo
 enddo
 do k=2,nz-1
  do j=js_pe,je_pe
   do i=is_pe,ie_pe
    mp=maskW(i,j,k)
    mm=maskW(i,j,k-1)
    cf(i,j,k, 0+2, 0+2, 0+2)= cf(i,j,k, 0+2, 0+2, 0+2)-mp/dzw(k  )/dzt(k) 
    cf(i,j,k, 0+2, 0+2, 1+2)= cf(i,j,k, 0+2, 0+2, 1+2)+mp/dzw(k  )/dzt(k) 
    cf(i,j,k, 0+2, 0+2, 0+2)= cf(i,j,k, 0+2, 0+2, 0+2)-mm/dzw(k-1)/dzt(k)
    cf(i,j,k, 0+2, 0+2,-1+2)= cf(i,j,k, 0+2, 0+2,-1+2)+mm/dzw(k-1)/dzt(k)
   enddo
  enddo
 enddo
 k=nz
 do j=js_pe,je_pe
  do i=is_pe,ie_pe
    mm=maskW(i,j,k-1)
    cf(i,j,k, 0+2, 0+2, 0+2)= cf(i,j,k, 0+2, 0+2, 0+2)-mm/dzw(k-1)/dzt(k)
    cf(i,j,k, 0+2, 0+2,-1+2)= cf(i,j,k, 0+2, 0+2,-1+2)+mm/dzw(k-1)/dzt(k)
  enddo
 enddo

end subroutine make_coeff_non_hydro







 subroutine congrad_non_hydro(is_,ie_,js_,je_,nz_,forc,iterations)
!=======================================================================
!  simple conjugate gradient solver
!=======================================================================
    use main_module   
    implicit none
    integer :: is_,ie_,js_,je_,nz_
    integer :: iterations,n,i,j,k
    !real*8  :: forc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    real*8  :: forc(is_:ie_,js_:je_,nz_)
    logical, save :: first = .true.
    real*8 , allocatable,save :: cf(:,:,:,:,:,:)
    real*8  :: res(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    real*8  :: p(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    real*8  :: Ap(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
    real*8  :: rsold,alpha,rsnew,dot_3D,absmax_3D
    real*8  :: step,step1=0,convergence_rate,estimated_error,smax,rs_min=0

    if (first) then
        allocate( cf(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3,3,3) ); cf=0d0
        call make_coeff_non_hydro(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,cf)
        first =.false.
    endif

    res=0d0
    call apply_op_3D(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,cf, p_non_hydro(:,:,:,taup1), res) !  res = A *psi
    do k=1,nz
     do j=js_pe,je_pe
      do i=is_pe,ie_pe
        res(i,j,k)=forc(i,j,k)-res(i,j,k)
      enddo
     enddo
    enddo

    p=res
    call border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,p)
    call setcyclic_xyz   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,p)
    rsold =  dot_3d(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,res,res)
    do n=1,congr_max_itts_non_hydro
!----------------------------------------------------------------------
!       key algorithm
!----------------------------------------------------------------------
        call apply_op_3D(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,cf, p, Ap) !  Ap = A *p
        alpha=rsold/dot_3D(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,p,Ap)
        p_non_hydro(:,:,:,taup1)=p_non_hydro(:,:,:,taup1)+alpha*p
        res=res-alpha*Ap
        rsnew=dot_3D(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,res,res)
        p=res+rsnew/rsold*p
        call border_exchg_xyz(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,p)
        call setcyclic_xyz   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,p)
        rsold=rsnew
!-----------------------------------------------------------------------
!       test for divergence
!-----------------------------------------------------------------------
        if (n .eq. 1) then
          rs_min = abs(rsnew)
        elseif (n .gt. 2) then
          rs_min = min(rs_min, abs(rsnew))
          if (abs(rsnew) .gt. 100.0*rs_min) then
           if (my_pe==0) print'(a,i5,a)','WARNING: non hydrostatic solver diverging after ',n,' iterations'
           goto 99
          endif
        endif
!-----------------------------------------------------------------------
!       test for convergence
!-----------------------------------------------------------------------
        smax = absmax_3D(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,nz,p)
        step = abs(alpha) * smax
        if (n .eq. 1) then
          step1 = step
          estimated_error = step
          if (step .lt. congr_epsilon_non_hydro) goto 101 
        else if (step .lt. congr_epsilon_non_hydro) then
          convergence_rate = exp(log(step/step1)/(n-1))
          estimated_error = step*convergence_rate/(1.0-convergence_rate)
          if (estimated_error .lt. congr_epsilon_non_hydro) goto 101
        end if
    enddo
    if (my_pe==0) print*,' WARNING: max iterations exceeded at itt=',itt
    goto 99

101 iterations = n  ! normal end
    if (my_pe==0 .and. enable_congrad_verbose) then
        print*,' estimated error=',estimated_error,'/',congr_epsilon_non_hydro
        print*,' iterations=',n
    endif
    return

99  iterations = n  ! error end
    if (my_pe==0) then
        print*,' estimated error=',estimated_error,'/',congr_epsilon_non_hydro
        print*,' iterations=',n
    endif

end subroutine congrad_non_hydro




subroutine apply_op_3D(is_,ie_,js_,je_,nz_,cf, p1, res)
      use main_module   
      implicit none
!-----------------------------------------------------------------------
!     apply operator A,  res = A *p1
!-----------------------------------------------------------------------
      integer :: is_,ie_,js_,je_,nz_
      !real*8 :: cf(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3,3,3) 
      !real*8 :: p1(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
      !real*8 :: res(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
      real*8 :: cf(is_:ie_,js_:je_,nz_,3,3,3) 
      real*8, dimension(is_:ie_,js_:je_,nz_) :: p1,res
      integer :: i,j,k,ii,jj,kk,kpkk

      res=0.
      do kk=-1,1
       do jj=-1,1
        do ii=-1,1
         do k=1,nz
          kpkk = min(nz,max(1,k+kk))
          do j=js_pe,je_pe
           do i=is_pe,ie_pe
            res(i,j,k) = res(i,j,k) + cf(i,j,k,ii+2,jj+2,kk+2)*p1(i+ii,j+jj,kpkk) 
           end do
          end do
         end do
        end do
       end do
      end do
end subroutine apply_op_3D


function absmax_3D(is_,ie_,js_,je_,nz_,p1)
      use main_module   
      implicit none
      integer :: is_,ie_,js_,je_,nz_
      real*8 :: absmax_3D,s2
      !real*8 :: p1(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
      real*8 :: p1(is_:ie_,js_:je_,nz_)
      integer :: i,j,k
      s2=0
      do k=1,nz
       do j=js_pe,je_pe
        do i=is_pe,ie_pe
         s2 = max( abs(p1(i,j,k)*maskT(i,j,k)), s2 )
         !s2 = max( abs(p1(i,j,k)), s2 )
        enddo
       enddo
      enddo
      call global_max(s2)
      absmax_3D=s2
end function absmax_3D


function dot_3D(is_,ie_,js_,je_,nz_,p1,p2)
      use main_module   
      implicit none
      integer :: is_,ie_,js_,je_,nz_
      real*8 :: dot_3D,s2
      !real*8 :: p1(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
      !real*8 :: p2(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
      real*8, dimension(is_:ie_,js_:je_,nz_) :: p1,p2
      integer :: i,j,k
      s2=0
      do k=1,nz
       do j=js_pe,je_pe
        do i=is_pe,ie_pe
         s2 = s2+p1(i,j,k)*p2(i,j,k)*maskT(i,j,k)
         !s2 = s2+p1(i,j,k)*p2(i,j,k)
        enddo
       enddo
      enddo
      call global_sum(s2)
      dot_3D=s2
end function dot_3D


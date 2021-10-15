
!=======================================================================
!      solve two dimensional Possion equation
!           A * dpsi = forc,  where A = nabla_h^2  
!      with Dirichlet boundary conditions
!      used for streamfunction
!=======================================================================


subroutine solve_streamfunction
!=======================================================================
!  solve for barotropic streamfunction
!=======================================================================
 use main_module   
 implicit none
 integer :: i,j,k,isle
 real*8 :: fxa,line_forc(nisle),line_psi0(nisle),aloc(nisle,nisle)
 real*8 :: fpx(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
 real*8 :: fpy(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
 real*8 :: forc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
 integer :: ipiv(nisle),info
 logical :: converged 

 !hydrostatic pressure
 fxa = grav/rho_0
 p_hydro(:,:,nz) = 0.5*rho(:,:,nz,tau)*fxa*dzw(nz)*maskT(:,:,nz)
 do k=nz-1,1,-1
   p_hydro(:,:,k)= maskT(:,:,k)*(p_hydro(:,:,k+1)+ 0.5*(rho(:,:,k+1,tau)+rho(:,:,k,tau))*fxa*dzw(k))
 enddo

 ! add hydrostatic pressure gradient
 do j=js_pe,je_pe
  do i=is_pe,ie_pe
   du(i,j,:,tau) = du(i,j,:,tau) - ( p_hydro(i+1,j,:)-p_hydro(i,j,:)  )/(dxu(i)*cost(j)) *maskU(i,j,:) 
   dv(i,j,:,tau) = dv(i,j,:,tau) - ( p_hydro(i,j+1,:)-p_hydro(i,j,:)  ) /dyu(j)*maskV(i,j,:) 
  enddo
 enddo
 ! forcing for barotropic streamfunction
 fpx=0.;fpy=0.
 do k=1,nz
   fpx=fpx+(du(:,:,k,tau)+du_mix(:,:,k))*maskU(:,:,k)*dzt(k)
   fpy=fpy+(dv(:,:,k,tau)+dv_mix(:,:,k))*maskV(:,:,k)*dzt(k)
 enddo
 fpx = fpx*hur; fpy = fpy*hvr
 call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,fpx); 
 call setcyclic_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,fpx)
 call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,fpy); 
 call setcyclic_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,fpy)
 do j=js_pe,je_pe
   do i=is_pe,ie_pe
    forc(i,j)=(fpy(i+1,j)-fpy(i,j))/(cosu(j)*dxu(i))-(cost(j+1)*fpx(i,j+1)-cost(j)*fpx(i,j))/(cosu(j)*dyu(j))
   enddo
 enddo

 ! solve for interior streamfunction
 dpsi(:,:,taup1)=2*dpsi(:,:,tau)-dpsi(:,:,taum1) ! first guess, we need three time levels here
 call congrad_streamfunction(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,forc,congr_itts,dpsi(:,:,taup1),converged)
 call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,dpsi(:,:,taup1))
 call setcyclic_xy   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,dpsi(:,:,taup1))

 if (nisle>1) then
   ! calculate island integrals of forcing, keep psi constant on island 1
   do k=2,nisle
     call line_integral(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx, k,fpx,fpy,line_forc(k))
   enddo
   ! calculate island integrals of interior streamfunction
   do k=2,nisle
     fpx=0;fpy=0
     do j=js_pe-onx+1,je_pe+onx
      do i=is_pe-onx+1,ie_pe+onx
       fpx(i,j) =-maskU(i,j,nz)*( dpsi(i,j,taup1)-dpsi(i,j-1,taup1))/dyt(j)*hur(i,j)
       fpy(i,j) = maskV(i,j,nz)*( dpsi(i,j,taup1)-dpsi(i-1,j,taup1))/(cosu(j)*dxt(i))*hvr(i,j)
      enddo
     enddo
     call line_integral(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx, k,fpx,fpy,line_psi0(k))
   enddo
   line_forc=line_forc-line_psi0

   ! solve for time dependent boundary values 
   aloc =line_psin ! will be changed in lapack routine
   CALL DGESV(nisle-1 , 1, aloc(2:nisle,2:nisle), nisle-1, IPIV, line_forc(2:nisle), nisle-1, INFO )
   if (info.ne.0) then
     print*,'info = ',info
     print*,' line_forc=',line_forc(2:nisle)
     call halt_stop(' in solve_streamfunction, lapack info not zero ')
   endif
   dpsin(2:nisle,tau) =line_forc(2:nisle)
 endif

 ! integrate barotropic and baroclinic velocity forward in time
 psi(:,:,taup1)= psi(:,:,tau)+ dt_mom*( (1.5+AB_eps)*dpsi(:,:,taup1) - (0.5+AB_eps)*dpsi(:,:,tau) )
 do isle=2,nisle
  psi(:,:,taup1)= psi(:,:,taup1)+ dt_mom*( (1.5+AB_eps)*dpsin(isle,tau) - (0.5+AB_eps)*dpsin(isle,taum1))*psin(:,:,isle)   
 enddo
 u(:,:,:,taup1)   = u(:,:,:,tau)   + dt_mom*( du_mix+ (1.5+AB_eps)*du(:,:,:,tau) - (0.5+AB_eps)*du(:,:,:,taum1) )*maskU
 v(:,:,:,taup1)   = v(:,:,:,tau)   + dt_mom*( dv_mix+ (1.5+AB_eps)*dv(:,:,:,tau) - (0.5+AB_eps)*dv(:,:,:,taum1) )*maskV

 ! subtract incorrect vertical mean from baroclinic velocity
 fpx=0.;fpy=0.
 do k=1,nz
   fpx=fpx+u(:,:,k,taup1)*maskU(:,:,k)*dzt(k)
   fpy=fpy+v(:,:,k,taup1)*maskV(:,:,k)*dzt(k)
 enddo
 do k=1,nz
   u(:,:,k,taup1) = u(:,:,k,taup1)-fpx*maskU(:,:,k)*hur
   v(:,:,k,taup1) = v(:,:,k,taup1)-fpy*maskV(:,:,k)*hvr
 enddo

 ! add barotropic mode to baroclinic velocity
 do j=js_pe,je_pe
   do i=is_pe,ie_pe
     u(i,j,:,taup1) = u(i,j,:,taup1)-maskU(i,j,:)*( psi(i,j,taup1)-psi(i,j-1,taup1))/dyt(j)*hur(i,j)
     v(i,j,:,taup1) = v(i,j,:,taup1)+maskV(i,j,:)*( psi(i,j,taup1)-psi(i-1,j,taup1))/(cosu(j)*dxt(i))*hvr(i,j)
   enddo
 enddo

end subroutine 


subroutine make_coeff_streamfunction(is_,ie_,js_,je_,cf)
!=======================================================================
!         A * p = forc
!         res = A * p
!         res = res +  cf(...,ii,jj,kk) * p(i+ii,j+jj,k+kk) 
!=======================================================================
      use main_module   
      implicit none
      integer :: is_,ie_,js_,je_
      !real*8 :: cf(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,3,3)
      real*8 :: cf(is_:ie_,js_:je_,3,3)
      integer :: i,j
      cf=0.
      do j=js_pe,je_pe
       do i=is_pe,ie_pe
         cf(i,j, 0+2, 0+2)= cf(i,j, 0+2, 0+2)-hvr(i+1,j)/dxu(i)/dxt(i+1) /cosu(j)**2
         cf(i,j, 1+2, 0+2)= cf(i,j, 1+2, 0+2)+hvr(i+1,j)/dxu(i)/dxt(i+1) /cosu(j)**2
         cf(i,j, 0+2, 0+2)= cf(i,j, 0+2, 0+2)-hvr(i  ,j)/dxu(i)/dxt(i  ) /cosu(j)**2
         cf(i,j,-1+2, 0+2)= cf(i,j,-1+2, 0+2)+hvr(i  ,j)/dxu(i)/dxt(i  ) /cosu(j)**2

         cf(i,j, 0+2, 0+2)= cf(i,j, 0+2, 0+2)-hur(i,j+1)/dyu(j)/dyt(j+1)*cost(j+1)/cosu(j)
         cf(i,j, 0+2, 1+2)= cf(i,j, 0+2, 1+2)+hur(i,j+1)/dyu(j)/dyt(j+1)*cost(j+1)/cosu(j)
         cf(i,j, 0+2, 0+2)= cf(i,j, 0+2, 0+2)-hur(i,j  )/dyu(j)/dyt(j  )*cost(j  )/cosu(j)
         cf(i,j, 0+2,-1+2)= cf(i,j, 0+2,-1+2)+hur(i,j  )/dyu(j)/dyt(j  )*cost(j  )/cosu(j)
       end do
      end do
end subroutine 




subroutine congrad_streamfunction(is_,ie_,js_,je_,forc,iterations,sol,converged)
!=======================================================================
!  conjugate gradient solver with preconditioner from MOM
!=======================================================================
      use main_module   
      implicit none
      integer :: is_,ie_,js_,je_
      integer :: iterations, n,i,j
      !real*8  :: forc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
      !real*8  :: sol(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
      real*8  :: forc(is_:ie_,js_:je_)
      real*8  :: sol(is_:ie_,js_:je_)
      real*8  :: res(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
      real*8  :: Z(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
      real*8  :: Zres(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
      real*8  :: ss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
      real*8  :: As(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
      real*8  :: estimated_error
      real*8  :: zresmax,betakm1,betak,betak_min=0,betaquot,s_dot_As,smax
      real*8  :: alpha,step,step1=0,convergence_rate
      real*8 , external :: absmax_sfc,dot_sfc
      logical, save :: first = .true.
      real*8 , allocatable,save :: cf(:,:,:,:)
      logical :: converged 

      if (first) then
        allocate( cf(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,3,3) ); cf=0
        call make_coeff_streamfunction(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx, cf)
        first =.false.
      endif

      Z=0.;Zres=0.;ss=0.;As=0.
!-----------------------------------------------------------------------
!     make approximate inverse operator Z (always even symmetry)
!-----------------------------------------------------------------------
      call make_inv_sfc(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,cf, Z)
!-----------------------------------------------------------------------
!     impose boundary conditions on guess
!     sol(0) = guess
!-----------------------------------------------------------------------
      call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,sol)
      call setcyclic_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,sol)
!-----------------------------------------------------------------------
!     res(0)  = forc - A * eta(0)
!-----------------------------------------------------------------------
      call apply_op(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx, cf, sol, res) 
      do j=js_pe,je_pe
       do i=is_pe,ie_pe
        res(i,j) = forc(i,j) - res(i,j)
       enddo
      enddo
!-----------------------------------------------------------------------
!     Zres(k-1) = Z * res(k-1)
!     see if guess is a solution, bail out to avoid division by zero
!-----------------------------------------------------------------------
      n = 0
      call inv_op_sfc(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,Z, res, Zres)
      Zresmax = absmax_sfc(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,Zres)
!     Assume convergence rate of 0.99 to extrapolate error
      if (100.0 * Zresmax .lt. congr_epsilon) then
         estimated_error = 100.0 * Zresmax 
         goto 101
      endif
!-----------------------------------------------------------------------
!     beta(0) = 1
!     ss(0)    = zerovector()
!-----------------------------------------------------------------------
      betakm1 = 1.0
      ss=0.
!-----------------------------------------------------------------------
!     begin iteration loop
!----------------------------------------------------------------------
      do n = 1,congr_max_iterations
!-----------------------------------------------------------------------
!       Zres(k-1) = Z * res(k-1)
!-----------------------------------------------------------------------
        call inv_op_sfc(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,Z, res, Zres)
!-----------------------------------------------------------------------
!       beta(k)   = res(k-1) * Zres(k-1)
!-----------------------------------------------------------------------
        betak = dot_sfc(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,Zres, res)
        if (n .eq. 1) then
          betak_min = abs(betak)
        elseif (n .gt. 2) then
          betak_min = min(betak_min, abs(betak))
          if (abs(betak) .gt. 100.0*betak_min) then
           if (my_pe==0) print'(a,i8)','WARNING: solver diverging at itt=',itt
           goto 99
          endif
        endif
!-----------------------------------------------------------------------
!       ss(k)      = Zres(k-1) + (beta(k)/beta(k-1)) * ss(k-1)
!-----------------------------------------------------------------------
        betaquot = betak/betakm1
        do j=js_pe,je_pe
         do i=is_pe,ie_pe
          ss(i,j) = Zres(i,j) + betaquot*ss(i,j)
         enddo
        enddo
        call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,ss)
        call setcyclic_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,ss)
!-----------------------------------------------------------------------
!       As(k)     = A * ss(k)
!-----------------------------------------------------------------------
        call apply_op(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,cf, ss, As)
!-----------------------------------------------------------------------
!       If ss=0 then the division for alpha(k) gives a float exception.
!       Assume convergence rate of 0.99 to extrapolate error.
!       Also assume alpha(k) ~ 1.
!-----------------------------------------------------------------------
        s_dot_As = dot_sfc(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,ss, As)
        if (abs(s_dot_As) .lt. abs(betak)*1.e-10) then
          smax = absmax_sfc(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,ss)
          estimated_error = 100.0 * smax 
           goto 101
        endif
!-----------------------------------------------------------------------
!       alpha(k)  = beta(k) / (ss(k) * As(k))
!-----------------------------------------------------------------------
        alpha = betak / s_dot_As
!-----------------------------------------------------------------------
!       update values:
!       eta(k)   = eta(k-1) + alpha(k) * ss(k)
!       res(k)    = res(k-1) - alpha(k) * As(k)
!-----------------------------------------------------------------------
        do j=js_pe,je_pe
         do i=is_pe,ie_pe
          sol(i,j)  = sol(i,j)  + alpha * ss(i,j)
          res(i,j) = res(i,j) - alpha * As(i,j)
         enddo
        enddo

        smax = absmax_sfc(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,ss)
!-----------------------------------------------------------------------
!       test for convergence
!       if (estimated_error) < congr_epsilon) exit
!-----------------------------------------------------------------------
        step = abs(alpha) * smax
        if (n .eq. 1) then
          step1 = step
          estimated_error = step
          if (step .lt. congr_epsilon) goto 101
        else if (step .lt. congr_epsilon) then
          convergence_rate = exp(log(step/step1)/(n-1))
          estimated_error = step*convergence_rate/(1.0-convergence_rate)
          if (estimated_error .lt. congr_epsilon) goto 101
        end if
        betakm1 = betak
      end do
!-----------------------------------------------------------------------
!     end of iteration loop
!-----------------------------------------------------------------------
    if (my_pe==0) print*,' WARNING: max iterations exceeded at itt=',itt
    goto 99


101 iterations = n  ! normal end
    converged = .true.
    if (my_pe==0 .and. enable_congrad_verbose) then
        print*,' estimated error=',estimated_error,'/',congr_epsilon
        print*,' iterations=',n
    endif
    return

99  iterations = n  ! error end
    converged = .false.
    if (my_pe==0) then
        print*,' estimated error=',estimated_error,'/',congr_epsilon
        print*,' iterations=',n
    endif
    ! check for NaN
    if (estimated_error/=estimated_error) then
        if (my_pe==0) print'(/a/)',' error is NaN, stopping integration '
        call panic_snap
        call halt_stop(' in solve_streamfunction')
    endif
end subroutine congrad_streamfunction



real*8 function absmax_sfc(is_,ie_,js_,je_,p1)
      use main_module   
      implicit none
      integer :: is_,ie_,js_,je_
      !real*8 :: s2,p1(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
      real*8 :: s2,p1(is_:ie_,js_:je_)
      integer :: i,j
      s2=0
      do j=js_pe,je_pe
       do i=is_pe,ie_pe
        s2 = max( abs(p1(i,j)), s2 )
       enddo
      enddo
      call global_max(s2)
      absmax_sfc=s2
end function 


real*8 function dot_sfc(is_,ie_,js_,je_,p1,p2)
      use main_module   
      implicit none
      integer :: is_,ie_,js_,je_
      !real*8 :: s2,p1(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx),p2(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
      real*8 :: s2,p1(is_:ie_,js_:je_),p2(is_:ie_,js_:je_)
      integer :: i,j
      s2=0
      do j=js_pe,je_pe
       do i=is_pe,ie_pe
        s2 = s2+p1(i,j)*p2(i,j)
       enddo
      enddo
      call global_sum(s2)
      dot_sfc=s2
end function 

subroutine inv_op_sfc(is_,ie_,js_,je_,Z, res, Zres)
      use main_module   
      implicit none
!-----------------------------------------------------------------------
!     apply approximate inverse Z of the operator A
!-----------------------------------------------------------------------
      integer :: is_,ie_,js_,je_
      !real*8 :: Z(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
      !real*8 :: res(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
      !real*8 :: Zres(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
      real*8, dimension(is_:ie_,js_:je_) :: Z,res,Zres
      integer :: i,j
      do j=js_pe,je_pe
       do i=is_pe,ie_pe
        Zres(i,j) = Z(i,j) * res(i,j)
       end do
      end do
end subroutine 


subroutine make_inv_sfc(is_,ie_,js_,je_,cf, Z)
      use main_module   
      implicit none
!-----------------------------------------------------------------------
!     construct an approximate inverse Z to A
!-----------------------------------------------------------------------
      integer :: is_,ie_,js_,je_
      !real*8 :: cf(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,3,3) 
      !real*8 ::  Z(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
      real*8 :: cf(is_:ie_,js_:je_,3,3) 
      real*8 :: Z (is_:ie_,js_:je_) 
      integer :: i,j,isle,n
!
!     copy diagonal coefficients of A to Z
!
      Z=0
      do j=js_pe,je_pe
       do i=is_pe,ie_pe
         Z(i,j) = cf(i,j,0+2,0+2)
       end do
      end do
!
!     now invert Z
!
      do j=js_pe,je_pe
       do i=is_pe,ie_pe
          if (Z(i,j) .ne. 0.0) then
            Z(i,j) = 1./Z(i,j)
          else
            Z(i,j) = 0.0
          end if
        end do
      end do
!
!     make inverse zero on island perimeters that are not integrated
!
      do  isle=1,nisle
       do n=1,nr_boundary(isle)      
         i = boundary(isle,n,1)
         j = boundary(isle,n,2)
         if ( i>=is_pe-onx .and. i<=ie_pe+onx .and. j>=js_pe-onx .and. j<=je_pe+onx ) Z(i,j)=0.0
       enddo       
      enddo       
end subroutine 






module particles_module
!=======================================================================
! module for particles
!=======================================================================
 implicit none 
 integer :: mmax      ! fraction of time step for integration
 integer :: nptraj=-1 ! number of particles
 real*8, allocatable :: pxyz(:,:) ! position of particles
 real*8, allocatable :: puvw(:,:) ! velocity of particles
 real*8, allocatable :: pts(:,:)  ! temperature and salinity of particles
 integer, allocatable :: pijk(:,:) ! index of tracer box
 logical, allocatable :: particle_active(:)
 integer, allocatable :: particle_pe(:)
 real*8 :: lenx_periodic  ! zonal length of domain
end module particles_module


subroutine allocate_particles(n_in)
!=======================================================================
! allocate variables for particles
!=======================================================================
 use main_module
 use particles_module
 implicit none 
 integer, intent(in) ::  n_in
 mmax = 1
 nptraj = n_in
 allocate( pijk(3,nptraj) ); pijk=0
 allocate( pxyz(3,nptraj) ); pxyz=0.0
 allocate( puvw(3,nptraj) ); puvw=0.0
 allocate( pts(2,nptraj) ); pts=0.0
 allocate( particle_active( nptraj) ); particle_active = .false.
 allocate( particle_pe( nptraj) );     particle_pe=-1
end subroutine allocate_particles



subroutine init_diag_particles
!=======================================================================
! initialize everything, user defined function to seed particles is
! already called using allocate_particles
!=======================================================================
 use main_module
 use particles_module
 implicit none 
 integer :: i,n

 lenx_periodic = 0.0
 do i=is_pe,ie_pe
   lenx_periodic = lenx_periodic + dxt(i)
 enddo
 call zonal_sum_vec(lenx_periodic,1)
 call pe0_bcast(lenx_periodic,1)

 ! find positions of particles
 do n=1,nptraj
  pijk(1,n) = minloc(  (pxyz(1,n) - xt(is_pe:ie_pe) )**2  ,1)+is_pe-1
  pijk(2,n) = minloc(  (pxyz(2,n) - yt(js_pe:je_pe) )**2  ,1)+js_pe-1
  pijk(3,n) = minloc(  (pxyz(3,n) - zt)**2  ,1)
 enddo
 call particle_pe_domain()
 call particle_distribute()
end subroutine init_diag_particles




subroutine particle_pe_domain()
!=======================================================================
!  is particle inside domain of this pe?
!=======================================================================
 use main_module
 use particles_module
 implicit none 
 integer :: n
 do n=1,nptraj
  if (pijk(1,n)>=is_pe .and. pijk(1,n)<=ie_pe .and. pijk(2,n)>=js_pe .and. pijk(2,n)<=je_pe ) then
        particle_active(n) = .true. 
        particle_pe(n)=my_pe
  else
        particle_active(n) = .false.
        particle_pe(n)=-1
  endif
  call global_max_int(particle_pe(n))
 enddo
end subroutine particle_pe_domain


subroutine particle_distribute
!=======================================================================
!  distribute particles to all pes
!=======================================================================
 use main_module
 use particles_module
 implicit none 
 integer :: n
 do n=1,nptraj
  if (particle_pe(n)>=0) then
     call bcast_real(   pxyz(:,n),3,particle_pe(n))
     call bcast_real(   puvw(:,n),3,particle_pe(n))
     call bcast_real(   pts(:,n),2,particle_pe(n))
     call bcast_integer(pijk(:,n),3,particle_pe(n))
  elseif (my_pe==0) then
     print*,' WARNING : particle #',n,' is out of domain.', &
               ' pxyz= ',pxyz(:,n),' pijk= ',pijk(:,n),  &
               ' particle PE=',particle_pe(n)
  endif
 enddo
end subroutine particle_distribute



subroutine integrate_particles
!=======================================================================
!       integrate particles
!=======================================================================
 use main_module
 use particles_module
 implicit none 
 integer :: i,j,k,n,m
 real*8 :: xe,xw,yn,ys,zu,zl,dvol,th,tf
 real*8 :: xeyszu,xeyszl,xeynzu,xeynzl
 real*8 :: xwyszu,xwyszl,xwynzu,xwynzl
 real*8 :: uuh,vvh,wwh,xold,yold,zold,tth,ttf
 real*8 :: uuf,vvf,wwf,uu,vv,ww,rcos,fac


 fac = 1.0
 if (coord_degree) fac = mtodeg


 do m=1,mmax

   call particle_pe_domain()

   do n=1,nptraj
     th= (m-1.)/mmax;  tf = 1.-th;

     if (particle_active(n)) then
!-----------------------------------------------------------------------
!      interpolate T/S on particle position
!-----------------------------------------------------------------------
       i  = pijk(1,n); 
       if (pxyz(1,n) > xt(i) ) then
        i=i+1
        xe = (xt(i-1)+dxu(i)*fac - pxyz(1,n)); xw = (pxyz(1,n)-xt(i-1))
       else
        xe = (xt(i) - pxyz(1,n)); xw = (pxyz(1,n)-(xt(i)-dxu(i-1)*fac) )
       endif

       j  = pijk(2,n);
       if (pxyz(2,n) > yt(j) )  j=j+1
       yn = (yt(j) - pxyz(2,n)); ys = (pxyz(2,n) - yt(j-1))

       k  = pijk(3,n)
       if (pxyz(3,n) > zt(k) )  k=k+1
       if (k>nz) then
        k=nz
        zu=0;zl=dzt(k)
       elseif (k<=1) then
        k=2 
        zu=dzt(k);zl=0
       else 
        zu = (zt(k) - pxyz(3,n)); zl = (pxyz(3,n)-zt(k-1))
        if (maskT(i,j,k-1)==0) then; zu=0;zl=dzt(k); endif
       endif

       if (maskT(i+1,j,k)==0) then; xe=dxt(i)*fac;xw=0; endif
       if (maskT(i-1,j,k)==0) then; xe=0;xw=dxt(i)*fac; endif
       if (maskT(i,j+1,k)==0) then; yn=dyt(j)*fac;ys=0; endif
       if (maskT(i,j-1,k)==0) then; yn=0;ys=dyt(j)*fac; endif

       dvol = 1./(dxu(i)*dyt(j)*dzt(k)*fac**2)
       xeyszu = xe*ys*zu*dvol; xwyszu = xw*ys*zu*dvol
       xeyszl = xe*ys*zl*dvol; xwyszl = xw*ys*zl*dvol
       xeynzu = xe*yn*zu*dvol; xwynzu = xw*yn*zu*dvol
       xeynzl = xe*yn*zl*dvol; xwynzl = xw*yn*zl*dvol

       tth=temp(i-1,j  ,k-1,tau  )*xeyszu+temp(i,j  ,k-1,tau  )*xwyszu   &
          +temp(i-1,j  ,k  ,tau  )*xeyszl+temp(i,j  ,k  ,tau  )*xwyszl   &
          +temp(i-1,j-1,k-1,tau  )*xeynzu+temp(i,j-1,k-1,tau  )*xwynzu   &
          +temp(i-1,j-1,k  ,tau  )*xeynzl+temp(i,j-1,k  ,tau  )*xwynzl
       ttf=temp(i-1,j  ,k-1,taup1)*xeyszu+temp(i,j  ,k-1,taup1)*xwyszu  &
          +temp(i-1,j  ,k  ,taup1)*xeyszl+temp(i,j  ,k  ,taup1)*xwyszl  &
          +temp(i-1,j-1,k-1,taup1)*xeynzu+temp(i,j-1,k-1,taup1)*xwynzu  &
          +temp(i-1,j-1,k  ,taup1)*xeynzl+temp(i,j-1,k  ,taup1)*xwynzl   
       pts(1,n)=th*tth+tf*ttf

       tth=salt(i-1,j  ,k-1,tau  )*xeyszu+salt(i,j  ,k-1,tau  )*xwyszu   &
          +salt(i-1,j  ,k  ,tau  )*xeyszl+salt(i,j  ,k  ,tau  )*xwyszl   &
          +salt(i-1,j-1,k-1,tau  )*xeynzu+salt(i,j-1,k-1,tau  )*xwynzu   &
          +salt(i-1,j-1,k  ,tau  )*xeynzl+salt(i,j-1,k  ,tau  )*xwynzl
       ttf=salt(i-1,j  ,k-1,taup1)*xeyszu+salt(i,j  ,k-1,taup1)*xwyszu  &
          +salt(i-1,j  ,k  ,taup1)*xeyszl+salt(i,j  ,k  ,taup1)*xwyszl  &
          +salt(i-1,j-1,k-1,taup1)*xeynzu+salt(i,j-1,k-1,taup1)*xwynzu  &
          +salt(i-1,j-1,k  ,taup1)*xeynzl+salt(i,j-1,k  ,taup1)*xwynzl   
       pts(2,n)=th*tth+tf*ttf

!-----------------------------------------------------------------------
!      pijk gives tracer box of the particle, 
!      find u-box and distances to borders, account for free slip
!-----------------------------------------------------------------------
       i  = pijk(1,n);
       xe = (xu(i) - pxyz(1,n)); xw = (pxyz(1,n)-(xu(i)-dxt(i)*fac )   )

       j  = pijk(2,n);
       if (pxyz(2,n) > yt(j) )  j=j+1
       yn = (yt(j) - pxyz(2,n)); ys = (pxyz(2,n) - yt(j-1))

       k  = pijk(3,n)
       if (pxyz(3,n) > zt(k) )  k=k+1
       if (k>nz) then
        k=nz
        zu=0;zl=dzt(k)
       elseif (k<=1) then
        k=2 
        zu=dzt(k);zl=0
       else 
        zu = (zt(k) - pxyz(3,n)); zl = (pxyz(3,n)-zt(k-1))
        if (maskT(i,j,k-1)==0) then; zu=0;zl=dzt(k); endif
       endif

       if (maskT(i,j+1,k)==0) then; yn=dyt(j)*fac;ys=0; endif
       if (maskT(i,j-1,k)==0) then; yn=0;ys=dyt(j)*fac; endif

       dvol = 1./(dxu(i)*dyt(j)*dzt(k)*fac**2)
       xeyszu = xe*ys*zu*dvol; xwyszu = xw*ys*zu*dvol
       xeyszl = xe*ys*zl*dvol; xwyszl = xw*ys*zl*dvol
       xeynzu = xe*yn*zu*dvol; xwynzu = xw*yn*zu*dvol
       xeynzl = xe*yn*zl*dvol; xwynzl = xw*yn*zl*dvol
!-----------------------------------------------------------------------
!      interpolate u on particle position
!-----------------------------------------------------------------------
       uuh=u(i-1,j  ,k-1,tau  )*xeyszu+u(i,j  ,k-1,tau  )*xwyszu   &
          +u(i-1,j  ,k  ,tau  )*xeyszl+u(i,j  ,k  ,tau  )*xwyszl   &
          +u(i-1,j-1,k-1,tau  )*xeynzu+u(i,j-1,k-1,tau  )*xwynzu   &
          +u(i-1,j-1,k  ,tau  )*xeynzl+u(i,j-1,k  ,tau  )*xwynzl
       uuf=u(i-1,j  ,k-1,taup1)*xeyszu+u(i,j  ,k-1,taup1)*xwyszu  &
          +u(i-1,j  ,k  ,taup1)*xeyszl+u(i,j  ,k  ,taup1)*xwyszl  &
          +u(i-1,j-1,k-1,taup1)*xeynzu+u(i,j-1,k-1,taup1)*xwynzu  &
          +u(i-1,j-1,k  ,taup1)*xeynzl+u(i,j-1,k  ,taup1)*xwynzl   
       uu=th*uuh+tf*uuf
!-----------------------------------------------------------------------
!      find v-box and distances to borders, account for free slip
!-----------------------------------------------------------------------
       i  = pijk(1,n); 
       if (pxyz(1,n) > xt(i) ) then
        i=i+1
        xe = (xt(i-1)+dxu(i)*fac - pxyz(1,n)); xw = (pxyz(1,n)-xt(i-1))
       else
        xe = (xt(i) - pxyz(1,n)); xw = (pxyz(1,n)-(xt(i)-dxu(i-1)*fac) )
       endif

       j  = pijk(2,n);
       yn = (yu(j) - pxyz(2,n)); ys = (pxyz(2,n) - yu(j-1))

       k  = pijk(3,n)
       if (pxyz(3,n) > zt(k) )  k=k+1
       if (k>nz) then
        k=nz
        zu=0;zl=dzt(k)
       elseif (k<=1) then
        k=2 
        zu=dzt(k);zl=0
       else 
        zu = (zt(k) - pxyz(3,n)); zl = (pxyz(3,n)-zt(k-1))
        if (maskT(i,j, k-1)==0) then; zu=0;zl=dzt(k); endif
       endif

       if (maskT(i+1,j,k)==0) then; xe=dxt(i)*fac;xw=0; endif
       if (maskT(i-1,j,k)==0) then; xe=0;xw=dxt(i)*fac; endif

       dvol = 1./(dxt(i)*fac*dyu(j)*fac*dzt(k))
       xeyszu = xe*ys*zu*dvol; xwyszu = xw*ys*zu*dvol
       xeyszl = xe*ys*zl*dvol; xwyszl = xw*ys*zl*dvol
       xeynzu = xe*yn*zu*dvol; xwynzu = xw*yn*zu*dvol
       xeynzl = xe*yn*zl*dvol; xwynzl = xw*yn*zl*dvol
!-----------------------------------------------------------------------
!      interpolate v on particle position
!-----------------------------------------------------------------------
       vvh=v(i-1,j  ,k-1,tau  )*xeyszu+v(i,j  ,k-1,tau  )*xwyszu   &
          +v(i-1,j  ,k  ,tau  )*xeyszl+v(i,j  ,k  ,tau  )*xwyszl   &
          +v(i-1,j-1,k-1,tau  )*xeynzu+v(i,j-1,k-1,tau  )*xwynzu   &
          +v(i-1,j-1,k  ,tau  )*xeynzl+v(i,j-1,k  ,tau  )*xwynzl
       vvf=v(i-1,j  ,k-1,taup1)*xeyszu+v(i,j  ,k-1,taup1)*xwyszu   &
          +v(i-1,j  ,k  ,taup1)*xeyszl+v(i,j  ,k  ,taup1)*xwyszl   &
          +v(i-1,j-1,k-1,taup1)*xeynzu+v(i,j-1,k-1,taup1)*xwynzu   &
          +v(i-1,j-1,k  ,taup1)*xeynzl+v(i,j-1,k  ,taup1)*xwynzl
       vv=th*vvh+tf*vvf
!-----------------------------------------------------------------------
!      find w-box and distances to borders, account for free slip
!-----------------------------------------------------------------------
       i  = pijk(1,n); 
       if (pxyz(1,n) > xt(i) ) then
        i=i+1
        xe = (xt(i-1)+dxu(i)*fac - pxyz(1,n)); xw = (pxyz(1,n)-xt(i-1))
       else
        xe = (xt(i) - pxyz(1,n)); xw = (pxyz(1,n)-(xt(i)-dxu(i-1)*fac) )
       endif

       j  = pijk(2,n); 
       if (pxyz(2,n) > yt(j) )  j=j+1
       yn = (yt(j) - pxyz(2,n)); ys = (pxyz(2,n) - yt(j-1))

       k  = pijk(3,n)
       if (k<=1) then
        k=2 
        zu=dzw(k);zl=0
       else 
        zu = (zw(k) - pxyz(3,n)); zl = (pxyz(3,n)-zw(k-1))
       endif

       if (maskT(i+1,j,k)==0) then; xe=dxt(i)*fac;xw=0; endif
       if (maskT(i-1,j,k)==0) then; xe=0;xw=dxt(i)*fac; endif
       if (maskT(i,j+1,k)==0) then; yn=dyt(j)*fac;ys=0; endif
       if (maskT(i,j-1,k)==0) then; yn=0;ys=dyt(j)*fac; endif

       dvol = 1./(dxt(i)*fac*dyt(j)*fac*dzw(k))
       xeyszu = xe*ys*zu*dvol; xwyszu = xw*ys*zu*dvol
       xeyszl = xe*ys*zl*dvol; xwyszl = xw*ys*zl*dvol
       xeynzu = xe*yn*zu*dvol; xwynzu = xw*yn*zu*dvol
       xeynzl = xe*yn*zl*dvol; xwynzl = xw*yn*zl*dvol
!-----------------------------------------------------------------------
!      interpolate w on particle position
!-----------------------------------------------------------------------
       wwh=w(i-1,j  ,k-1,tau  )*xeyszu+w(i,j  ,k-1,tau  )*xwyszu   &
          +w(i-1,j  ,k  ,tau  )*xeyszl+w(i,j  ,k  ,tau  )*xwyszl   &
          +w(i-1,j-1,k-1,tau  )*xeynzu+w(i,j-1,k-1,tau  )*xwynzu   &
          +w(i-1,j-1,k  ,tau  )*xeynzl+w(i,j-1,k  ,tau  )*xwynzl
       if (.not. enable_hydrostatic ) then
         wwf=w(i-1,j  ,k-1,taup1)*xeyszu+w(i,j  ,k-1,taup1)*xwyszu   &
            +w(i-1,j  ,k  ,taup1)*xeyszl+w(i,j  ,k  ,taup1)*xwyszl   &
            +w(i-1,j-1,k-1,taup1)*xeynzu+w(i,j-1,k-1,taup1)*xwynzu   &
            +w(i-1,j-1,k  ,taup1)*xeynzl+w(i,j-1,k  ,taup1)*xwynzl
         ww=th*wwh+tf*wwf
       else
          ww=wwh
       endif
!-----------------------------------------------------------------------
!      integrate the particle trajectory forward for one time step
!-----------------------------------------------------------------------
       xold=pxyz(1,n); yold=pxyz(2,n); zold=pxyz(3,n)
       if (coord_degree) then
         rcos      = mtodeg/cos(pxyz(2,n)/180.*pi)
         pxyz(1,n) = pxyz(1,n) + dt_tracer*uu/mmax *rcos
         pxyz(2,n) = pxyz(2,n) + dt_tracer*vv/mmax *mtodeg
         pxyz(3,n) = pxyz(3,n) + dt_tracer*ww/mmax
       else
         pxyz(1,n) = pxyz(1,n) + dt_tracer*uu/mmax 
         pxyz(2,n) = pxyz(2,n) + dt_tracer*vv/mmax
         pxyz(3,n) = pxyz(3,n) + dt_tracer*ww/mmax
       endif
       puvw(1,n)=uu; puvw(2,n)=vv; puvw(3,n)=ww
!-----------------------------------------------------------------------
!      update index of bounding tracer volume
!-----------------------------------------------------------------------
       i  = pijk(1,n); j  = pijk(2,n); k  = pijk(3,n)
       if (pxyz(1,n) >= xu(i)) then
            pijk(1,n) = i + 1
       else if (pxyz(1,n) < (xu(i)-dxt(i)*fac)  ) then
            pijk(1,n) = i - 1
       endif

       if (pxyz(2,n) >= yu(j)) then
            pijk(2,n) = j + 1
       else if (pxyz(2,n) < yu(j-1)) then
            pijk(2,n) = j - 1
       endif

       if (k<nz) then
           if (pxyz(3,n) >= zw(k) ) pijk(3,n) = k + 1
       endif
       if ( k>1) then
          if (pxyz(3,n) < zw(k-1) ) pijk(3,n) = k - 1
       endif
       
!-----------------------------------------------------------------------
!      periodic boundary conditions
!-----------------------------------------------------------------------
       if (enable_cyclic_x .and. pijk(1,n)>nx) then
          pijk(1,n)=pijk(1,n)-nx; pxyz(1,n)=pxyz(1,n)-lenx_periodic*fac
       endif
       if (enable_cyclic_x .and. pijk(1,n)<1) then
          pijk(1,n)=pijk(1,n)+nx; pxyz(1,n)=pxyz(1,n)+lenx_periodic*fac
       endif

     endif ! particle_active
   enddo ! nptraj
   call particle_distribute
 enddo ! m


end subroutine integrate_particles



subroutine particles_read_restart
!=======================================================================
! read unfinished averages from file
!=======================================================================
 use main_module
 use particles_module
 implicit none
 character (len=80) :: filename
 logical :: file_exists
 integer :: io,ierr,nptraj_
 
 write(filename,'(a,i5,a)')  'particles_restart.dta'
 inquire ( FILE=filename, EXIST=file_exists )
 if (.not. file_exists) then
      if (my_pe==0) then
         print'(a,a,a)',' file ',filename(1:len_trim(filename)),' not present'
         print'(a)',' reading no restart for particles'
      endif
      return
 endif
 if (my_pe==0) print'(2a)',' reading particles from ',filename(1:len_trim(filename))
 call get_free_iounit(io,ierr)
 if (ierr/=0) goto 10
 open(io,file=filename,form='unformatted',status='old',err=10)
 read(io,err=10) nptraj_
 if (nptraj_ /= nptraj) then
       if (my_pe==0) then
        print*,' read number of particles ',nptraj_
        print*,' which does not match ',nptraj
       endif
       goto 10
 endif
 read(io,err=10) pijk,pxyz,puvw,pts
 close(io)
 call particle_pe_domain()
 call particle_distribute()
 return
 10 continue
 print'(a)',' Warning: error reading file'
end subroutine particles_read_restart



subroutine particles_write_restart
!=======================================================================
! write unfinished averages to restart file
!=======================================================================
 use main_module
 use particles_module
 implicit none
 character (len=80) :: filename
 integer :: io,ierr

 write(filename,'(a,i5,a)')  'particles_restart.dta'
 if (my_pe==0) print'(a,a)',' writing particles to ',filename(1:len_trim(filename))
 call get_free_iounit(io,ierr)
 if (ierr/=0) goto 10
 open(io,file=filename,form='unformatted',status='unknown')
 write(io,err=10) nptraj
 write(io,err=10) pijk,pxyz,puvw,pts
 close(io)
 return
 10 continue
 print'(a)',' Warning: error writing file'
end subroutine particles_write_restart




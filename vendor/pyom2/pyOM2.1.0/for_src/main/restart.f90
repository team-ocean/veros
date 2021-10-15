

 subroutine write_restart(itt_)
!=======================================================================
!      write a restart file
!=======================================================================
     use main_module   
     use tke_module   
     use eke_module   
     use idemix_module   
     implicit none
     integer, intent(in) :: itt_
     character*80 :: filename
     integer :: ierr,io,is,ie,js,je

     is=is_pe-onx; ie=ie_pe+onx; js=js_pe-onx; je=je_pe+onx

     write(filename,'(a,i5,a)')  'restart_PE_',my_pe,'.dta'
     call replace_space_zero(filename)
     if (my_pe==0) print'(a,a,a,i8)',' writing restart file ',filename(1:len_trim(filename)),' at itt=',itt_

     call get_free_iounit(io,ierr)
     if (ierr/=0) goto 10
     open(io,file=filename,form='unformatted',status='unknown')
     write(io,err=10) nx,ny,nz,itt_
     write(io,err=10) is,ie,js,je

     write(io,err=10)  rho(is:ie,js:je,:,tau),  rho(is:ie,js:je,:,taum1)
     write(io,err=10) Nsqr(is:ie,js:je,:,tau), Nsqr(is:ie,js:je,:,taum1)
     write(io,err=10)   Hd(is:ie,js:je,:,tau),   Hd(is:ie,js:je,:,taum1)
     write(io,err=10) int_drhodT(is:ie,js:je,:,tau), int_drhodT(is:ie,js:je,:,taum1)
     write(io,err=10) int_drhodS(is:ie,js:je,:,tau), int_drhodS(is:ie,js:je,:,taum1)
     write(io,err=10)   dHd(is:ie,js:je,:,tau),   dHd(is:ie,js:je,:,taum1)
     write(io,err=10)  temp(is:ie,js:je,:,tau),  temp(is:ie,js:je,:,taum1)
     write(io,err=10) dtemp(is:ie,js:je,:,tau), dtemp(is:ie,js:je,:,taum1)
     write(io,err=10)  salt(is:ie,js:je,:,tau),  salt(is:ie,js:je,:,taum1)
     write(io,err=10) dsalt(is:ie,js:je,:,tau), dsalt(is:ie,js:je,:,taum1)
     write(io,err=10)     u(is:ie,js:je,:,tau),     u(is:ie,js:je,:,taum1)
     write(io,err=10)    du(is:ie,js:je,:,tau),    du(is:ie,js:je,:,taum1)
     write(io,err=10)     v(is:ie,js:je,:,tau),     v(is:ie,js:je,:,taum1)
     write(io,err=10)    dv(is:ie,js:je,:,tau),    dv(is:ie,js:je,:,taum1)
     write(io,err=10)     w(is:ie,js:je,:,tau),     w(is:ie,js:je,:,taum1)
     write(io,err=10) psi(is:ie,js:je,taum1) , psi(is:ie,js:je,tau)
     if (enable_streamfunction) then
       write(io,err=10) dpsi(is:ie,js:je,taum1),dpsi(is:ie,js:je,tau)
       write(io,err=10) dpsi(is:ie,js:je,taup1)
       write(io,err=10) dpsin(1:nisle,tau) ,dpsin(1:nisle,taum1)
     endif
     if (enable_eke) write(io,err=10) eke(is:ie,js:je,:,tau),eke(is:ie,js:je,:,taum1)
     if (enable_eke) write(io,err=10) deke(is:ie,js:je,:,tau),deke(is:ie,js:je,:,taum1)
     if (enable_tke) write(io,err=10) tke(is:ie,js:je,:,tau),tke(is:ie,js:je,:,taum1),K_diss_v(is:ie,js:je,:)
     if (enable_tke) write(io,err=10) dtke(is:ie,js:je,:,tau),dtke(is:ie,js:je,:,taum1)
     if (enable_idemix) write(io,err=10) E_iw(is:ie,js:je,:,tau),E_iw(is:ie,js:je,:,taum1)
     if (enable_idemix) write(io,err=10) dE_iw(is:ie,js:je,:,tau),dE_iw(is:ie,js:je,:,taum1)
     if (enable_idemix_M2) write(io,err=10) E_M2(is:ie,js:je,:,tau),E_M2(is:ie,js:je,:,taum1)
     if (enable_idemix_M2) write(io,err=10) dE_M2p(is:ie,js:je,:,tau),dE_M2p(is:ie,js:je,:,taum1)
     if (enable_idemix_niw) write(io,err=10) E_niw(is:ie,js:je,:,tau),E_niw(is:ie,js:je,:,taum1)
     if (enable_idemix_niw) write(io,err=10) dE_niwp(is:ie,js:je,:,tau),dE_niwp(is:ie,js:je,:,taum1)
     close(io)
     call fortran_barrier()
     return
     10 continue
     print'(a)',' Warning: error writing restart file'
 end subroutine write_restart



 subroutine read_restart(itt_)
!=======================================================================
!       read the restart file
!=======================================================================
     use main_module   
     use tke_module   
     use eke_module   
     use idemix_module   
     implicit none
     integer, intent(out) :: itt_
     integer :: ierr,is_,ie_,js_,je_,is,ie,js,je
     integer :: io,nx_,ny_,nz_
     character*80 :: filename
     logical :: file_exists
     is=is_pe-onx; ie=ie_pe+onx; js=js_pe-onx; je=je_pe+onx

     write(filename,'(a,i5,a)')  'restart_PE_',my_pe,'.dta'
     call replace_space_zero(filename)
     inquire ( FILE=filename, EXIST=file_exists )
     if (.not. file_exists) then
       if (my_pe==0) then
         print'(a,a)',' found no restart file ',filename(1:len_trim(filename))
         print'(a)',' proceeding with initial conditions'
       endif
       return
     endif

     if (my_pe==0) print'(a,a)',' reading from restart file ',filename(1:len_trim(filename))
     call get_free_iounit(io,ierr)
     if (ierr/=0) goto 10
     open(io,file=filename,form='unformatted',status='old',err=10)
     read(io,err=10) nx_,ny_,nz_,itt_
     if (nx/=nx_ .or. ny/=ny_ .or. nz/= nz_) then 
       if (my_pe==0) then
        print*,' read from restart dimensions: ',nx_,ny_,nz_
        print*,' does not match dimensions   : ',nx,ny,nz
       endif
       goto 10
     endif
     read(io,err=10) is_,ie_,js_,je_
     if (is_/=is.or.ie_/=ie.or.js_/=js.or.je_/=je) then
       if (my_pe==0) then
        print*,' read from restart PE boundaries: ',is_,ie_,js_,je_
        print*,' which does not match           : ',is,ie,js,je
       endif
       goto 10
     endif

     read(io,err=10)  rho(is:ie,js:je,:,tau),  rho(is:ie,js:je,:,taum1)
     read(io,err=10) Nsqr(is:ie,js:je,:,tau), Nsqr(is:ie,js:je,:,taum1)
     read(io,err=10)   Hd(is:ie,js:je,:,tau),   Hd(is:ie,js:je,:,taum1)
     read(io,err=10) int_drhodT(is:ie,js:je,:,tau), int_drhodT(is:ie,js:je,:,taum1)
     read(io,err=10) int_drhodS(is:ie,js:je,:,tau), int_drhodS(is:ie,js:je,:,taum1)
     read(io,err=10)   dHd(is:ie,js:je,:,tau),   dHd(is:ie,js:je,:,taum1)
     read(io,err=10)  temp(is:ie,js:je,:,tau),  temp(is:ie,js:je,:,taum1)
     read(io,err=10) dtemp(is:ie,js:je,:,tau), dtemp(is:ie,js:je,:,taum1)
     read(io,err=10)  salt(is:ie,js:je,:,tau),  salt(is:ie,js:je,:,taum1)
     read(io,err=10) dsalt(is:ie,js:je,:,tau), dsalt(is:ie,js:je,:,taum1)
     read(io,err=10)     u(is:ie,js:je,:,tau),     u(is:ie,js:je,:,taum1)
     read(io,err=10)    du(is:ie,js:je,:,tau),    du(is:ie,js:je,:,taum1)
     read(io,err=10)     v(is:ie,js:je,:,tau),     v(is:ie,js:je,:,taum1)
     read(io,err=10)    dv(is:ie,js:je,:,tau),    dv(is:ie,js:je,:,taum1)
     read(io,err=10)     w(is:ie,js:je,:,tau),     w(is:ie,js:je,:,taum1)
     read(io,err=10) psi(is:ie,js:je,taum1), psi(is:ie,js:je,tau)
     if (enable_streamfunction) then
       read(io,err=10) dpsi(is:ie,js:je,taum1), dpsi(is:ie,js:je,tau)
       read(io,err=10) dpsi(is:ie,js:je,taup1) ! we need this also for first guess of streamfunction
       read(io,err=10) dpsin(1:nisle,tau) ,dpsin(1:nisle,taum1)
     endif
     if (enable_eke) read(io,err=10) eke(is:ie,js:je,:,tau), eke(is:ie,js:je,:,taum1)
     if (enable_eke) read(io,err=10) deke(is:ie,js:je,:,tau), deke(is:ie,js:je,:,taum1)
     if (enable_tke) read(io,err=10) tke(is:ie,js:je,:,tau),tke(is:ie,js:je,:,taum1),K_diss_v(is:ie,js:je,:)  ! we need this to calculate kappaM
     if (enable_tke) read(io,err=10) dtke(is:ie,js:je,:,tau),dtke(is:ie,js:je,:,taum1)
     if (enable_idemix) read(io,err=10) E_iw(is:ie,js:je,:,tau), E_iw(is:ie,js:je,:,taum1)
     if (enable_idemix) read(io,err=10) dE_iw(is:ie,js:je,:,tau), dE_iw(is:ie,js:je,:,taum1)
     if (enable_idemix_M2) read(io,err=10) E_M2(is:ie,js:je,:,tau), E_M2(is:ie,js:je,:,taum1)
     if (enable_idemix_M2) read(io,err=10) dE_M2p(is:ie,js:je,:,tau), dE_M2p(is:ie,js:je,:,taum1)
     if (enable_idemix_niw) read(io,err=10) E_niw(is:ie,js:je,:,tau), E_niw(is:ie,js:je,:,taum1)
     if (enable_idemix_niw) read(io,err=10) dE_niwp(is:ie,js:je,:,tau), dE_niwp(is:ie,js:je,:,taum1)
     close(io)
     call fortran_barrier()
   
     return
     10 continue
     print'(a)',' Warning: error reading restart file'
     call halt_stop(' in read_restart')     
 end subroutine read_restart




 subroutine get_free_iounit (nu,ierr)
!-----------------------------------------------------------------------
!     returns the first free IO unit number in nu
!-----------------------------------------------------------------------
      implicit none
      integer nu,n,ierr
      logical in_use
      character (len=80) :: name
      ierr=0
      do n=7,99
        inquire (n, OPENED=in_use, NAME=name)
        if (.not. in_use) then
          nu = n
          go to 10
         endif
      enddo
      print *,'Error: exhausted available fortran unit numbers'
      print *,'             Are you forgetting to close units?'
      ierr=-1
10    continue
 end subroutine get_free_iounit

 subroutine replace_space_zero(name)
      implicit none
      character (len=*) :: name
      integer  :: i
      do i=1,len_trim(name)
          if (name(i:i)==' ')name(i:i)='0'
      enddo
 end subroutine replace_space_zero





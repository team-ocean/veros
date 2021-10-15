


subroutine pe_decomposition
  use main_module   
  implicit none
  integer :: n,tag=0,iloc(20),ierr
  include "mpif.h"
  integer,dimension(MPI_STATUS_SIZE)  :: Status

! ----------------------------------
!      domain decomposition for each PE
! ----------------------------------
   if (n_pes>1) then
      if (n_pes_i*n_pes_j /= n_pes ) call halt_stop(' n_pes_i times n_pes_j not equal number of PEs')
      i_blk = (nx-1)/n_pes_i + 1    ! i-extent of each block
      j_blk = (ny-1)/n_pes_j + 1    ! j-extent of each block
      my_blk_i = mod(my_pe,n_pes_i)+1! number of PE in i-dir.
      my_blk_j = (my_pe)/n_pes_i + 1 ! number of PE in j-dir.
      is_pe = (my_blk_i-1)*i_blk + 1 ! start index in i-dir of this PE
      ie_pe = min(my_blk_i*i_blk,nx)
      js_pe = (my_blk_j-1)*j_blk + 1
      je_pe = min(my_blk_j*j_blk,ny)
! ----------------------------------
!     check for incorrect domain decomposition
! ----------------------------------
      !if (my_blk_j==n_pes_j .and. js_pe>=je_pe-2) then
      !if (my_blk_j==n_pes_j .and. js_pe>=je_pe) then
      if (my_blk_j==n_pes_j .and. js_pe>je_pe) then
       print*,' ERROR:'
       print*,' domain decompositon impossible in j-direction'
       print*,' choose other number of PEs in j-direction'
       call halt_stop(' in pe_decomposition')
      endif
      !if (my_blk_i==n_pes_i .and. is_pe>=ie_pe-2) then
      !if (my_blk_i==n_pes_i .and. is_pe>=ie_pe) then
      if (my_blk_i==n_pes_i .and. is_pe>ie_pe) then
       print*,' ERROR:'
       print*,' domain decompositon impossible in i-direction'
       print*,' choose other number of PEs in i-direction'
       call halt_stop(' in pe_decomposition')
      endif
   else
       n_pes_j = 1; n_pes_i = 1
       i_blk = nx; j_blk = ny
       my_blk_j = 1 ; my_blk_i = 1 
       js_pe = 1; je_pe = ny
       is_pe = 1; ie_pe = nx
   endif
! ----------------------------------
!      print out the PE decomposition, let PE 0 talk
! ----------------------------------
   do n=0,n_pes-1
     if (n==0) then
       iloc(1:6) = (/my_blk_i,my_blk_j,is_pe,ie_pe,js_pe,je_pe/)
     else
       if (my_pe==n) then
          iloc(1:6) = (/my_blk_i,my_blk_j,is_pe,ie_pe,js_pe,je_pe/)
          call mpi_send(iloc,6,mpi_integer,0,tag,my_comm,ierr)
       endif
       if (my_pe==0) call mpi_recv(iloc,6,mpi_integer,n,tag,my_comm,status,ierr)
     endif
     if (my_pe==0) print'(a,i4,a,i4,a,i4,a,i4,a,i4)','domain of PE #',n,' i=',iloc(3),':',iloc(4),'   j=',iloc(5),':',iloc(6)
   enddo
   if (my_pe==0) print*,' '
   call fortran_barrier
end subroutine pe_decomposition




 subroutine my_mpi_init(comm_)
!--------------------------------------------------------------
!     intitialize mpi system for model
!--------------------------------------------------------------
  use main_module   
  implicit none
  integer :: comm_,nlen,ierr
  include "mpif.h"
      character (len=MPI_MAX_PROCESSOR_NAME) :: pname
      if (comm_ == MPI_COMM_NULL) then
        print *, 'You passed MPI_COMM_NULL !!!'
        return
       end if
       my_comm=comm_
       call MPI_Comm_rank(my_comm, my_pe, ierr)
       call MPI_Comm_size(my_comm, n_pes, ierr)
       call MPI_Get_processor_name(pname, nlen, ierr)
       call my_mpi_test(my_comm)
 end subroutine my_mpi_init


subroutine halt_stop(string)
!--------------------------------------------------------------
!     controlled stop, should not be called from python
!--------------------------------------------------------------
      implicit none
      character*(*) :: string
      integer :: ierr,code,my_pe
      include "mpif.h"
      call mpi_comm_rank(MPI_COMM_WORLD,my_pe,ierr)
      print*,' global pe #',my_pe,' : ',string
      print*,' global pe #',my_pe,' aborting '
      code=99
      call MPI_ABORT(mpi_comm_world, code, IERR)
end subroutine halt_stop



subroutine fortran_barrier
!--------------------------------------------------------------
!     A barrier for the local sub domain
!     for use in fortran part only
!--------------------------------------------------------------
  use main_module   
  implicit none
  integer :: ierr
  call mpi_barrier(my_comm, ierr)
end subroutine fortran_barrier



subroutine my_mpi_test(my_comm)
!--------------------------------------------------------------
!     test some basic mpi routines
!--------------------------------------------------------------
      implicit none
      integer :: my_comm
      integer :: my_pe=-1,all_pes,xint,xint2,ierr
      real*8    :: xreal,xreal2
      include "mpif.h"
!   get some mpi infos first
      call mpi_comm_rank(my_comm       ,my_pe,ierr)
      if (my_pe==0) print*,' testing mpi routines'
      call mpi_comm_size(my_comm       ,all_pes,ierr)
!   try first global barrier
      call mpi_barrier(my_comm       , ierr)
!   try broadcasting
      xreal = 1.0
      call mpi_bcast(xreal,1,mpi_real8,0,my_comm       ,ierr)
      xint = 1
      call mpi_bcast(xint,1,mpi_integer,0,my_comm       ,ierr)
!   check results of broadcasting
      if (xreal /= 1.0 ) then
       print*,'fatal: MPI test failed on broadcasting reals for PE #',my_pe
       stop
      endif
      if (xint /= 1 ) then
       print*,'fatal: MPI test failed on broadcasting integer for PE #',my_pe
       stop
      endif
      call mpi_barrier(my_comm       , ierr)
!   try global sum
      xreal = 2.0
      call mpi_allreduce(xreal,xreal2,1,mpi_real8,MPI_SUM,my_comm       ,ierr)
      xint = 2
      call mpi_allreduce(xint,xint2,1,mpi_integer,MPI_SUM,my_comm       ,ierr)
!   check results 
      xreal = xreal2/all_pes
      if (xreal /= 2.0 ) then
       print*,'fatal: MPI test failed on global sum (real) for PE #',my_pe
       stop
      endif
      xint = xint2/all_pes
      if (xint /= 2.0 ) then
       print*,'fatal: MPI test failed on global sum (int) for PE #',my_pe
       stop
      endif
      call mpi_barrier(my_comm       , ierr)
end subroutine my_mpi_test


subroutine pe0_bcast_int(a,len)
!--------------------------------------------------------------
!     Broadcast an integer vector from pe0 to all other pe
!--------------------------------------------------------------
      use main_module   
      implicit none
      integer, intent(in) :: len
      integer, intent(inout) :: a(len)
      integer :: ierr
      include "mpif.h"
      call mpi_bcast(a,len,mpi_integer,0,my_comm,ierr)
end subroutine pe0_bcast_int


subroutine pe0_bcast(a,len)
!--------------------------------------------------------------
!     Broadcast a vector from pe0 to all other pe
!--------------------------------------------------------------
      use main_module   
      implicit none
      integer, intent(in) :: len
      real*8, intent(inout) :: a(len)
      integer :: ierr
      include "mpif.h"
      call mpi_bcast(a,len,mpi_real8,0,my_comm,ierr)
end subroutine pe0_bcast



subroutine bcast_real(x,len,pe)
!--------------------------------------------------------------
!     Broadcast a real vector from PE pe to others
!--------------------------------------------------------------
      use main_module   
      implicit none
      integer :: len,ierr,pe
      real*8 :: x(len)
      include "mpif.h"
      call mpi_barrier(my_comm, ierr)
      call mpi_bcast(x,len,mpi_real8,pe,my_comm,ierr)
end subroutine bcast_real

subroutine bcast_integer(x,len,pe)
!--------------------------------------------------------------
!     Broadcast an integer vector from PE pe to others
!--------------------------------------------------------------
      use main_module
      implicit none
      integer :: len,ierr,pe
      integer :: x(len)
      include "mpif.h"
      call mpi_barrier(my_comm, ierr)
      call mpi_bcast(x,len,mpi_integer,pe,my_comm,ierr)
end subroutine bcast_integer



subroutine global_max(x)
!--------------------------------------------------------------
!     Get the max of real x over all PEs in sub domain
!--------------------------------------------------------------
      use main_module   
      implicit none
      real*8,intent(inout)    :: x
      real*8    :: x_sym,x_sym2
      integer :: ierr
      include "mpif.h"
      x_sym = x
      call mpi_allreduce(x_sym,x_sym2,1,mpi_real8,MPI_MAX,my_comm       ,ierr)
      x = x_sym2
 end subroutine global_max


subroutine global_min(x)
!--------------------------------------------------------------
!     Get the min of real x over all PEs in sub domain
!--------------------------------------------------------------
      use main_module   
      implicit none
      real*8,intent(inout)    :: x
      real*8    :: x_sym,x_sym2
      integer :: ierr
      include "mpif.h"
      x_sym = x
      call mpi_allreduce(x_sym,x_sym2,1,mpi_real8,MPI_MIN,my_comm       ,ierr)
      x = x_sym2
end subroutine global_min


subroutine global_sum(x)
!--------------------------------------------------------------
!     Do a sum of real x over all PEs in sub domain
!--------------------------------------------------------------
      use main_module   
      implicit none
      real*8,intent(inout)    :: x
      real*8    :: x_sym,x_sym2
      integer :: ierr
      include "mpif.h"
      x_sym = x
      call mpi_allreduce(x_sym,x_sym2,1,mpi_real8,MPI_SUM,my_comm       ,ierr)
      x = x_sym2
end subroutine global_sum






subroutine global_max_int(x)
!--------------------------------------------------------------
!     Get the max of integer x over all PEs in sub domain
!--------------------------------------------------------------
      use main_module   
      implicit none
      integer,intent(inout)    :: x
      integer    :: x_sym,x_sym2,ierr
      include "mpif.h"
      x_sym = x
      call mpi_allreduce(x_sym,x_sym2,1,mpi_integer,MPI_MAX,my_comm       ,ierr)
      x = x_sym2
 end subroutine global_max_int


subroutine global_min_int(x)
!--------------------------------------------------------------
!     Get the min of integer x over all PEs in sub domain
!--------------------------------------------------------------
      use main_module   
      implicit none
      integer,intent(inout)    :: x
      integer    :: x_sym,x_sym2,ierr
      include "mpif.h"
      x_sym = x
      call mpi_allreduce(x_sym,x_sym2,1,mpi_integer,MPI_MIN,my_comm       ,ierr)
      x = x_sym2
end subroutine global_min_int


subroutine global_sum_int(x)
!--------------------------------------------------------------
!     Do a sum of integer x over all PEs in sub domain
!--------------------------------------------------------------
      use main_module   
      implicit none
      integer,intent(inout)    :: x
      integer    :: x_sym,x_sym2,ierr
      include "mpif.h"
      x_sym = x
      call mpi_allreduce(x_sym,x_sym2,1,mpi_integer,MPI_SUM,my_comm       ,ierr)
      x = x_sym2
end subroutine global_sum_int







subroutine border_exchg_xy(is_,ie_,js_,je_,a)
!--------------------------------------------------------------
! Exchange overlapping areas of 2D array a in all PEs of sub domain. 
!--------------------------------------------------------------
  use main_module   
  implicit none
  integer, intent(in) :: is_,ie_,js_,je_
  real*8, intent(inout)  :: a(is_:ie_,js_:je_)
  integer  ::  tag=0, ierr,i,j,len
  include "mpif.h"
  integer,dimension(MPI_STATUS_SIZE)  :: Status

  
      call mpi_barrier(my_comm,ierr)
      if ( n_pes_j > 1) then
!       from north to south
        len=ie_pe-is_pe+1+2*onx
        if (my_blk_j /=1 ) then
         do j=1,onx
          call mpi_send(a(:,js_pe+j-1),len,mpi_real8,my_pe-n_pes_i,tag,my_comm,ierr)
         enddo
        endif
        if (my_blk_j /= n_pes_j) then 
         do j=1,onx
          call mpi_recv(a(:,je_pe+j),len,mpi_real8,my_pe+n_pes_i,tag,my_comm,Status,ierr)
         enddo
        endif
!       from south to north
        if (my_blk_j /= n_pes_j) then 
         do j=1,onx
          call mpi_send(a(:,je_pe-j+1),len,mpi_real8,my_pe+n_pes_i,tag,my_comm,ierr)
         enddo
        endif
        if (my_blk_j /=1 ) then
         do j=1,onx
          call mpi_recv(a(:,js_pe-j),len,mpi_real8,my_pe-n_pes_i,tag,my_comm,Status,ierr)
         enddo
        endif
      endif
      call mpi_barrier(my_comm,ierr)

      if ( n_pes_i > 1) then
        len=je_pe-js_pe+1+2*onx
!       from east to west
        if (my_blk_i /=1 ) then
         do i=1,onx
          call mpi_send(a(is_pe+i-1,:),len,mpi_real8,my_pe-1,tag,my_comm,ierr)
         enddo
        endif
        if (my_blk_i /= n_pes_i) then 
         do i=1,onx
          call mpi_recv(a(ie_pe+i,:),len,mpi_real8,my_pe+1,tag,my_comm,Status,ierr)
         enddo
        endif
!       from west to east
        if (my_blk_i /= n_pes_i) then 
         do i=1,onx
          call mpi_send(a(ie_pe-i+1,:),len,mpi_real8,my_pe+1,tag,my_comm,ierr)
         enddo
        endif
        if (my_blk_i /=1 ) then
         do i=1,onx
          call mpi_recv(a(is_pe-i,:),len,mpi_real8,my_pe-1,tag,my_comm,Status,ierr)
         enddo
        endif
      endif
      call mpi_barrier(my_comm,ierr)

end subroutine border_exchg_xy





subroutine border_exchg_xy_int(is_,ie_,js_,je_,a)
!--------------------------------------------------------------
! Exchange overlapping areas of 2D array a in all PEs of sub domain. 
!--------------------------------------------------------------
  use main_module   
  implicit none
  integer, intent(in) :: is_,ie_,js_,je_
  integer, intent(inout)  :: a(is_:ie_,js_:je_)
  integer  ::  tag=0, ierr,i,j,len
  include "mpif.h"
  integer,dimension(MPI_STATUS_SIZE)  :: Status

  
      call mpi_barrier(my_comm,ierr)
      if ( n_pes_j > 1) then
!       from north to south
        len=ie_pe-is_pe+1+2*onx
        if (my_blk_j /=1 ) then
         do j=1,onx
          call mpi_send(a(:,js_pe+j-1),len,mpi_integer,my_pe-n_pes_i,tag,my_comm,ierr)
         enddo
        endif
        if (my_blk_j /= n_pes_j) then 
         do j=1,onx
          call mpi_recv(a(:,je_pe+j),len,mpi_integer,my_pe+n_pes_i,tag,my_comm,Status,ierr)
         enddo
        endif
!       from south to north
        if (my_blk_j /= n_pes_j) then 
         do j=1,onx
          call mpi_send(a(:,je_pe-j+1),len,mpi_integer,my_pe+n_pes_i,tag,my_comm,ierr)
         enddo
        endif
        if (my_blk_j /=1 ) then
         do j=1,onx
          call mpi_recv(a(:,js_pe-j),len,mpi_integer,my_pe-n_pes_i,tag,my_comm,Status,ierr)
         enddo
        endif
      endif
      call mpi_barrier(my_comm,ierr)

      if ( n_pes_i > 1) then
        len=je_pe-js_pe+1+2*onx
!       from east to west
        if (my_blk_i /=1 ) then
         do i=1,onx
          call mpi_send(a(is_pe+i-1,:),len,mpi_integer,my_pe-1,tag,my_comm,ierr)
         enddo
        endif
        if (my_blk_i /= n_pes_i) then 
         do i=1,onx
          call mpi_recv(a(ie_pe+i,:),len,mpi_integer,my_pe+1,tag,my_comm,Status,ierr)
         enddo
        endif
!       from west to east
        if (my_blk_i /= n_pes_i) then 
         do i=1,onx
          call mpi_send(a(ie_pe-i+1,:),len,mpi_integer,my_pe+1,tag,my_comm,ierr)
         enddo
        endif
        if (my_blk_i /=1 ) then
         do i=1,onx
          call mpi_recv(a(is_pe-i,:),len,mpi_integer,my_pe-1,tag,my_comm,Status,ierr)
         enddo
        endif
      endif
      call mpi_barrier(my_comm,ierr)

end subroutine border_exchg_xy_int




subroutine setcyclic_xy(is_,ie_,js_,je_,p1)
!--------------------------------------------------------------
!       set cyclic boundary conditions for 2D array
!--------------------------------------------------------------
  use main_module   
  implicit none
  integer, intent(in) :: is_,ie_,js_,je_
  real*8,intent(inout) :: p1(is_:ie_,js_:je_)
  integer  ::  tag, ierr,i,len
  include "mpif.h"
  integer,dimension(MPI_STATUS_SIZE)  :: Status

  if (enable_cyclic_x) then
    if (n_pes_i == 1) then
      do i=1,onx
        p1(nx+i,:)=p1(i  ,:)
        p1(1-i,:)=p1(nx-i+1,:) 
      enddo
    else
      len=(je_pe-js_pe+1+2*onx)
      tag=1
      do i=1,onx
        if (my_blk_i ==1        ) call mpi_send(p1(i   ,:),len,mpi_real8,my_pe+(n_pes_i-1),tag,my_comm,ierr)
        if (my_blk_i == n_pes_i ) call mpi_recv(p1(nx+i,:),len,mpi_real8,my_pe-(n_pes_i-1),tag,my_comm,status,ierr)
      enddo
      tag=2
      do i=1,onx
        if (my_blk_i ==n_pes_i )  call mpi_send(p1(nx-i+1,:),len,mpi_real8,my_pe-(n_pes_i-1),tag,my_comm,ierr)
        if (my_blk_i == 1 )       call mpi_recv(p1(1   -i,:),len,mpi_real8,my_pe+(n_pes_i-1),tag,my_comm,status,ierr)
      enddo
    endif
  endif
end subroutine setcyclic_xy





subroutine setcyclic_xy_int(is_,ie_,js_,je_,p1)
!--------------------------------------------------------------
!       set cyclic boundary conditions for 2D array
!--------------------------------------------------------------
  use main_module   
  implicit none
  integer, intent(in) :: is_,ie_,js_,je_
  integer,intent(inout) :: p1(is_:ie_,js_:je_)
  integer  ::  tag, ierr,i,len
  include "mpif.h"
  integer,dimension(MPI_STATUS_SIZE)  :: Status

  if (enable_cyclic_x) then
    if (n_pes_i == 1) then
      do i=1,onx
        p1(nx+i,:)=p1(i  ,:)
        p1(1-i,:)=p1(nx-i+1,:) 
      enddo
    else
      len=(je_pe-js_pe+1+2*onx)
      tag=1
      do i=1,onx
        if (my_blk_i ==1        ) call mpi_send(p1(i   ,:),len,mpi_integer,my_pe+(n_pes_i-1),tag,my_comm,ierr)
        if (my_blk_i == n_pes_i ) call mpi_recv(p1(nx+i,:),len,mpi_integer,my_pe-(n_pes_i-1),tag,my_comm,status,ierr)
      enddo
      tag=2
      do i=1,onx
        if (my_blk_i ==n_pes_i )  call mpi_send(p1(nx-i+1,:),len,mpi_integer,my_pe-(n_pes_i-1),tag,my_comm,ierr)
        if (my_blk_i == 1 )       call mpi_recv(p1(1   -i,:),len,mpi_integer,my_pe+(n_pes_i-1),tag,my_comm,status,ierr)
      enddo
    endif
  endif
end subroutine setcyclic_xy_int




subroutine border_exchg_xyz(is_,ie_,js_,je_,nz_,a)
!--------------------------------------------------------------
! Exchange overlapping areas of 3D array a in all PEs 
!--------------------------------------------------------------
  use main_module   
  implicit none
  integer, intent(in) :: is_,ie_,js_,je_,nz_
  real*8, intent(inout)  :: a(is_:ie_,js_:je_,nz_)
  integer :: k
  do k=1,nz
   call border_exchg_xy(is_,ie_,js_,je_,a(:,:,k)) 
  enddo
end subroutine border_exchg_xyz
  


subroutine setcyclic_xyz(is_,ie_,js_,je_,nz_,a)
!--------------------------------------------------------------
!       set cyclic boundary conditions for 3D array
!--------------------------------------------------------------
  use main_module   
  implicit none
  integer, intent(in) :: is_,ie_,js_,je_,nz_
  real*8, intent(inout)  :: a(is_:ie_,js_:je_,nz_)
  integer :: k
  do k=1,nz
   call setcyclic_xy   (is_,ie_,js_,je_,a(:,:,k))
  enddo
end subroutine setcyclic_xyz




subroutine border_exchg_xyp(is_,ie_,js_,je_,np,a)
!--------------------------------------------------------------
! Exchange overlapping areas of spectral array a in all PEs  
!--------------------------------------------------------------
  use main_module   
  implicit none
  integer, intent(in) :: is_,ie_,js_,je_,np
  real*8, intent(inout)  :: a(is_:ie_,js_:je_,np)
  integer :: k
  do k=1,np
   call border_exchg_xy(is_,ie_,js_,je_,a(:,:,k)) 
  enddo
end subroutine border_exchg_xyp


subroutine setcyclic_xyp(is_,ie_,js_,je_,np,p1)
!--------------------------------------------------------------
!       set cyclic boundary conditions for 3D array
!--------------------------------------------------------------
  use main_module   
  implicit none
  integer, intent(in) :: is_,ie_,js_,je_,np
  real*8, intent(inout) :: p1(is_:ie_,js_:je_,np)
  integer  ::  k

  p1(:,:,1 )=p1(:,:,np-1) 
  p1(:,:,np)=p1(:,:,2)
  do k=1,np
   call setcyclic_xy   (is_,ie_,js_,je_,p1(:,:,k))
  enddo
end subroutine setcyclic_xyp





subroutine pe0_recv_2D(nx_,ny_,a)
!--------------------------------------------------------------
!     all PEs send their data of a 2D array to PE0
!--------------------------------------------------------------
      use main_module   
      implicit none
      integer, intent(in) :: nx_,ny_
      real*8, intent(inout) :: a(nx_,ny_)
      integer                     :: js,je,iproc,is,ie
      integer                     :: tag=0, ierr,len
      include "mpif.h"
      integer, dimension(MPI_STATUS_SIZE) :: Status
      js=js_pe; je=je_pe; is=is_pe; ie=ie_pe
      do iproc=1,n_pes-1
       call mpi_barrier(my_comm       ,ierr)
       if ( my_pe == iproc ) then
        call mpi_send(js,1,mpi_integer,0,tag,my_comm,ierr)
        call mpi_send(je,1,mpi_integer,0,tag,my_comm,ierr)
        call mpi_send(is,1,mpi_integer,0,tag,my_comm,ierr)
        call mpi_send(ie,1,mpi_integer,0,tag,my_comm,ierr)
        len=(ie-is+1)*(je-js+1)
        call mpi_send(a(is:ie,js:je),len,mpi_real8,0,tag,my_comm,ierr)
       endif
       if ( my_pe == 0 ) then
        call mpi_recv(js,1,mpi_integer,iproc,tag,my_comm,Status,ierr)
        call mpi_recv(je,1,mpi_integer,iproc,tag,my_comm,Status,ierr)
        call mpi_recv(is,1,mpi_integer,iproc,tag,my_comm,Status,ierr)
        call mpi_recv(ie,1,mpi_integer,iproc,tag,my_comm,Status,ierr)
        len=(ie-is+1)*(je-js+1)
        call mpi_recv(a(is:ie,js:je),len,mpi_real8,iproc,tag,my_comm,Status,ierr)
       endif
       call mpi_barrier(my_comm       ,ierr)
      enddo
end subroutine pe0_recv_2D





subroutine pe0_recv_2D_int(nx_,ny_,a)
!--------------------------------------------------------------
!     all PEs send their data of a 2D array to PE0
!--------------------------------------------------------------
      use main_module   
      implicit none
      integer, intent(in) :: nx_,ny_
      integer, intent(inout) :: a(nx_,ny_)
      integer                     :: js,je,iproc,is,ie
      integer                     :: tag=0, ierr,len
      include "mpif.h"
      integer, dimension(MPI_STATUS_SIZE) :: Status
      js=js_pe; je=je_pe; is=is_pe; ie=ie_pe
      do iproc=1,n_pes-1
       call mpi_barrier(my_comm       ,ierr)
       if ( my_pe == iproc ) then
        call mpi_send(js,1,mpi_integer,0,tag,my_comm,ierr)
        call mpi_send(je,1,mpi_integer,0,tag,my_comm,ierr)
        call mpi_send(is,1,mpi_integer,0,tag,my_comm,ierr)
        call mpi_send(ie,1,mpi_integer,0,tag,my_comm,ierr)
        len=(ie-is+1)*(je-js+1)
        call mpi_send(a(is:ie,js:je),len,mpi_integer,0,tag,my_comm,ierr)
       endif
       if ( my_pe == 0 ) then
        call mpi_recv(js,1,mpi_integer,iproc,tag,my_comm,Status,ierr)
        call mpi_recv(je,1,mpi_integer,iproc,tag,my_comm,Status,ierr)
        call mpi_recv(is,1,mpi_integer,iproc,tag,my_comm,Status,ierr)
        call mpi_recv(ie,1,mpi_integer,iproc,tag,my_comm,Status,ierr)
        len=(ie-is+1)*(je-js+1)
        call mpi_recv(a(is:ie,js:je),len,mpi_integer,iproc,tag,my_comm,Status,ierr)
       endif
       call mpi_barrier(my_comm       ,ierr)
      enddo
end subroutine pe0_recv_2D_int





subroutine zonal_sum_vec(a,len)
!--------------------------------------------------------------
!    sum vector along zonal row of PEs
!    result is stored in first PE of row
!--------------------------------------------------------------
      use main_module   
      implicit none
      integer, intent(in) :: len
      real*8, intent(inout) :: a(len)
      real*8  :: b(len)
      integer     :: tag=0, ierr,n
      include "mpif.h"
      integer, dimension(MPI_STATUS_SIZE) :: Status

      if (n_pes_i>1) then
        do n=2,n_pes_i
          if (my_blk_i == 1) then
            call mpi_recv(b,len,mpi_real8,my_pe+(n-1),tag,my_comm,Status,ierr)
            a = a+b
          else
            if (my_blk_i==n) call mpi_send(a,len,mpi_real8,my_pe-(my_blk_i-1),tag,my_comm,ierr)
          endif
        enddo
      endif 
end subroutine zonal_sum_vec






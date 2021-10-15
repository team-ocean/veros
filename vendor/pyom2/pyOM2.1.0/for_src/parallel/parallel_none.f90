

subroutine pe_decomposition
  use main_module   
  implicit none
! ----------------------------------
!      domain decomposition for each PE
! ----------------------------------
     n_pes = 1
     n_pes_j = 1
     n_pes_i = 1
     j_blk = ny
     i_blk = nx
     my_blk_j = 1 
     my_blk_i = 1 
     js_pe = 1
     je_pe = ny
     is_pe = 1
     ie_pe = nx
     my_comm = 0
end subroutine pe_decomposition


subroutine mpi_finalize(ierr)
  use main_module   
  implicit none
  integer, intent(out) :: ierr
  ierr=0
end subroutine mpi_finalize

subroutine mpi_init(ierr)
  use main_module   
  implicit none
  integer, intent(out) :: ierr
  ierr=0
end subroutine mpi_init


 subroutine my_mpi_init(comm_)
!--------------------------------------------------------------
!     intitialize mpi system for model
!--------------------------------------------------------------
  use main_module   
  implicit none
  integer :: comm_
  comm_=0
 end subroutine my_mpi_init



subroutine halt_stop(string)
!--------------------------------------------------------------
!     controlled stop, should not be called from python
!--------------------------------------------------------------
      implicit none
      character*(*) :: string
      print*,string
      stop
end subroutine halt_stop


subroutine fortran_barrier
end subroutine fortran_barrier


subroutine pe0_bcast_int(a,len)
!--------------------------------------------------------------
!     Broadcast an integer vector from pe0 to all other pe
!--------------------------------------------------------------
      implicit none
      integer, intent(in) :: len
      integer, intent(inout) :: a(len)
end subroutine pe0_bcast_int


subroutine pe0_bcast(a,len)
!--------------------------------------------------------------
!     Broadcast a vector from pe0 to all other pe
!--------------------------------------------------------------
      implicit none
      integer, intent(in) :: len
      real*8, intent(inout) :: a(len)
end subroutine pe0_bcast



subroutine bcast_real(x,len,pe)
!--------------------------------------------------------------
!     Broadcast a real vector from PE pe to others
!--------------------------------------------------------------
      use main_module   
      implicit none
      integer :: len,ierr,pe
      real*8 :: x(len)
end subroutine bcast_real

subroutine bcast_integer(x,len,pe)
!--------------------------------------------------------------
!     Broadcast an integer vector from PE pe to others
!--------------------------------------------------------------
      use main_module
      implicit none
      integer :: len,ierr,pe
      integer :: x(len)
end subroutine bcast_integer


subroutine global_max(x)
!--------------------------------------------------------------
!     Get the max of real x over all PEs in sub domain
!--------------------------------------------------------------
      implicit none
      real*8, intent(inout)    :: x
end subroutine global_max


subroutine global_min(x)
!--------------------------------------------------------------
!     Get the min of real x over all PEs in sub domain
!--------------------------------------------------------------
      implicit none
      real*8, intent(inout)    :: x
end subroutine global_min


subroutine global_sum(x)
!--------------------------------------------------------------
!     Do a sum of real x over all PEs in sub domain
!--------------------------------------------------------------
      implicit none
      real*8, intent(inout)    :: x
end subroutine global_sum


subroutine global_max_int(x)
!--------------------------------------------------------------
!     Get the max of integer x over all PEs in sub domain
!--------------------------------------------------------------
      implicit none
      integer,intent(inout)    :: x
 end subroutine global_max_int


subroutine global_min_int(x)
!--------------------------------------------------------------
!     Get the min of integer x over all PEs in sub domain
!--------------------------------------------------------------
      implicit none
      integer,intent(inout)    :: x
end subroutine global_min_int


subroutine global_sum_int(x)
!--------------------------------------------------------------
!     Do a sum of integer x over all PEs in sub domain
!--------------------------------------------------------------
      implicit none
      integer,intent(inout)    :: x
end subroutine global_sum_int


subroutine border_exchg_xy(is_,ie_,js_,je_,a)
!--------------------------------------------------------------
!     Exchange overlapping areas of 3D array a in all PEs of sub 
!     domain. Number of overlapping indicees are given by jx.
!--------------------------------------------------------------
  use main_module   
  implicit none
  integer, intent(in) :: is_,ie_,js_,je_
  real*8, intent(inout)  :: a
end subroutine border_exchg_xy


subroutine border_exchg_xy_int(is_,ie_,js_,je_,a)
!--------------------------------------------------------------
!     Exchange overlapping areas of 3D array a in all PEs of sub 
!     domain. Number of overlapping indicees are given by jx.
!--------------------------------------------------------------
  use main_module   
  implicit none
  integer, intent(in) :: is_,ie_,js_,je_
  integer, intent(inout)  :: a
end subroutine border_exchg_xy_int




subroutine setcyclic_xy(is_,ie_,js_,je_,p1)
!--------------------------------------------------------------
!       set cyclic boundary conditions for 2D array
!--------------------------------------------------------------
  use main_module   
  implicit none
  integer, intent(in) :: is_,ie_,js_,je_
  real*8, intent(inout) :: p1(is_:ie_,js_:je_)
  integer :: j,i

  if (enable_cyclic_x) then
      do i=1,onx
        p1(nx+i,:)=p1(i  ,:)
        p1(1-i,:)=p1(nx-i+1,:) 
      enddo
  endif
end subroutine setcyclic_xy



subroutine setcyclic_xy_int(is_,ie_,js_,je_,p1)
!--------------------------------------------------------------
!       set cyclic boundary conditions for 2D array
!--------------------------------------------------------------
  use main_module   
  implicit none
  integer, intent(in) :: is_,ie_,js_,je_
  integer, intent(inout) :: p1(is_:ie_,js_:je_)
  integer :: j,i

  if (enable_cyclic_x) then
      do i=1,onx
        p1(nx+i,:)=p1(i  ,:)
        p1(1-i,:)=p1(nx-i+1,:) 
      enddo
  endif
end subroutine setcyclic_xy_int




subroutine border_exchg_xyz(is_,ie_,js_,je_,nz_,a)
!--------------------------------------------------------------
! Exchange overlapping areas of 3D array a in all PEs of sub domain. 
!--------------------------------------------------------------
  use main_module   
  implicit none
  integer :: is_,ie_,js_,je_,nz_
  real*8  :: a(is_:ie_,js_:je_,nz_)
end subroutine border_exchg_xyz


subroutine setcyclic_xyz(is_,ie_,js_,je_,nz_,a)
!--------------------------------------------------------------
!       set cyclic boundary conditions for 3D array
!--------------------------------------------------------------
  use main_module   
  implicit none
  integer:: is_,ie_,js_,je_,nz_
  real*8 :: a(is_:ie_,js_:je_,nz_)
  integer :: k
  do k=1,nz
   call setcyclic_xy(is_,ie_,js_,je_,a(:,:,k))
  enddo
end subroutine setcyclic_xyz



subroutine border_exchg_xyp(is_,ie_,js_,je_,np,a)
!--------------------------------------------------------------
! Exchange overlapping areas of 3D array a in all PEs of sub domain. 
!--------------------------------------------------------------
  use main_module   
  implicit none
  integer:: is_,ie_,js_,je_,np
  real*8 :: a
end subroutine border_exchg_xyp



subroutine setcyclic_xyp(is_,ie_,js_,je_,np,p1)
!--------------------------------------------------------------
!       set cyclic boundary conditions for 3D array
!--------------------------------------------------------------
  use main_module   
  implicit none
  integer:: is_,ie_,js_,je_,np
  real*8 :: p1(is_:ie_,js_:je_,np)
  integer :: i

  p1(:,:,1 )=p1(:,:,np-1) 
  p1(:,:,np)=p1(:,:,2)
  if (enable_cyclic_x) then
     do i=1,onx
       p1(nx+i,:,:)=p1(i  ,:,:)
       p1(1-i,:,:)=p1(nx-i+1,:,:) 
      enddo
  endif
end subroutine setcyclic_xyp



subroutine pe0_recv_2D(nx_,ny_,a)
!--------------------------------------------------------------
!     all PEs send their data of a 2D array to PE0
!--------------------------------------------------------------
      use main_module   
      implicit none
      integer, intent(in) :: nx_,ny_
      real*8, intent(inout)  :: a(nx_,ny_)
end subroutine pe0_recv_2D


subroutine pe0_recv_2D_int(nx_,ny_,a)
!--------------------------------------------------------------
!     all PEs send their data of a 2D array to PE0
!--------------------------------------------------------------
      use main_module   
      implicit none
      integer, intent(in) :: nx_,ny_
      integer, intent(inout)  :: a(nx_,ny_)
end subroutine pe0_recv_2D_int


subroutine zonal_sum_vec(a,len)
!--------------------------------------------------------------
!    sum vector along zonal row of PEs
!--------------------------------------------------------------
      use main_module   
      implicit none
      real*8  :: a(len)
      integer       :: len
end subroutine zonal_sum_vec





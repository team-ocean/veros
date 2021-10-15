


module diag_averages_module
!=======================================================================
! Module for time averages
!=======================================================================
  implicit none
  integer :: nitts=0,number_diags = 0
  integer, parameter :: max_number_diags = 500
  type type_var2D
    real*8,pointer  :: a(:,:)
  end type type_var2D
  type type_var3D
    real*8,pointer  :: a(:,:,:)
  end type type_var3D
  character (len=80) :: diag_name(max_number_diags), diag_longname(max_number_diags)
  character (len=80) :: diag_units(max_number_diags), diag_grid(max_number_diags)
  type(type_var2d)   :: diag_var2D(max_number_diags), diag_sum_var2D(max_number_diags)
  type(type_var3d)   :: diag_var3D(max_number_diags), diag_sum_var3D(max_number_diags)
  logical            :: diag_is3D(max_number_diags)
end module diag_averages_module


subroutine register_average(name,longname,units,grid,var2D,var3D,is3D)
!=======================================================================
! register a variables to be averaged 
! this routine may be called by user in set_diagnostics
! name : NetCDF variables name (must be unique)
! longname:  long name
! units : units
! grid : three digits, either 'T' for T grid or 'U' for shifted grid
!        applies for the 3 dimensions, U is shifted by 1/2 grid point
!        if .not. is3D, third digit is not referenced
! var2D : variables to be averaged, 2D, not referenced if is3D = .true.
! var3D : variables to be averaged, 3D, not referenced if is3D = .false.
! is3D  : true if variable to be averaged is 3D else false
!=======================================================================
 use main_module
 use diagnostics_module
 use diag_averages_module
 implicit none
 character (len=*) :: name,longname,units,grid
 real*8,target :: var2D(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
 real*8,target :: var3D(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)
 logical :: is3D
 integer :: n
 if (.not. enable_diag_averages) then 
    if (my_pe==0) print*,' switch on enable_diag_averages to use time averaging'
    return
 endif
 if (my_pe==0) then
   print'(2a)',' time averaging variable ',name(1:len_trim(name))
   print'(6a)',' long name ',longname(1:len_trim(longname)),' units ',units(1:len_trim(units)), &
          ' grid  ',grid(1:len_trim(grid))
 endif
 ! check if name is in use
 do n=1,number_diags
   if (name(1:len_trim(name)) == diag_name(n)(1:len_trim(name)) ) then
     if (my_pe==0) print*,' name already in use'
     call halt_stop(' in register_average')
   endif
 enddo
 number_diags = number_diags + 1 
 ! check for overflow
 if (number_diags > max_number_diags) then
     if (my_pe==0) print*,' too many diagnostics, increase max_number_diags'
     call halt_stop(' in register_average')
 endif

 diag_name(number_diags)     = name
 diag_longname(number_diags) = longname
 diag_units(number_diags)    = units
 diag_grid(number_diags)     = grid
 diag_is3D(number_diags)     = is3D
 if (is3D) then
  diag_var2D(number_diags)%a => null()
  diag_var3D(number_diags)%a => var3D
  allocate( diag_sum_var3D(number_diags)%a(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) )
  diag_sum_var3D(number_diags)%a(:,:,:) = 0
 else
  diag_var2D(number_diags)%a => var2D
  diag_var3D(number_diags)%a => null()
  allocate( diag_sum_var2D(number_diags)%a(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) )
  diag_sum_var2D(number_diags)%a(:,:) = 0
 endif
end subroutine register_average





subroutine diag_averages
!=======================================================================
! do the average
!=======================================================================
 use main_module
 use diag_averages_module
 implicit none
 integer :: n
 nitts = nitts+1
 do n=1,number_diags
   if (diag_is3D(n) ) then
    diag_sum_var3D(n)%a(:,:,:) = diag_sum_var3D(n)%a(:,:,:) + diag_var3D(n)%a(:,:,:)
   else
    diag_sum_var2D(n)%a(:,:) = diag_sum_var2D(n)%a(:,:) + diag_var2D(n)%a(:,:)
   endif
 enddo
end subroutine diag_averages



subroutine write_averages
!=======================================================================
! write averages to netcdf file and zero array
!=======================================================================
 use main_module
 use diag_averages_module
 implicit none
 include "netcdf.inc"
 integer :: ncid,iret,n,ilen,k
 integer :: lon_tdim,lon_udim,itimedim
 integer :: lat_tdim,lat_udim,id,z_tdim,z_udim
 integer :: dims(4),itimeid
 character (len=80) :: file
 real*8, parameter :: spval = -1.0d33
 real*8 :: bloc(nx,ny),fxa

 write(file,'(a,i12,a)')  'averages_',itt,'.cdf'
 call replace_space_zero(file)
 if (my_pe==0) print'(2a)',' writing averages to file ',file(1:len_trim(file))
 call def_grid_cdf(file)

 if (my_pe==0) then
    iret=nf_open(file,NF_WRITE, ncid)
    iret=nf_set_fill(ncid, NF_NOFILL, iret)
    call ncredf(ncid, iret)
    iret=nf_inq_dimid(ncid,'xt',lon_tdim)
    iret=nf_inq_dimid(ncid,'xu',lon_udim)
    iret=nf_inq_dimid(ncid,'yt',lat_tdim)
    iret=nf_inq_dimid(ncid,'yu',lat_udim)
    iret=nf_inq_dimid(ncid,'zt',z_tdim)
    iret=nf_inq_dimid(ncid,'zu',z_udim)
    iret=nf_inq_dimid(ncid,'Time',itimedim)
    do n=1,number_diags
       dims = (/Lon_tdim,lat_tdim,z_tdim,iTimedim/)
       if (diag_grid(n)(1:1) == 'U') dims(1) = Lon_udim
       if (diag_grid(n)(2:2) == 'U') dims(2) = Lat_udim
       if (diag_grid(n)(3:3) == 'U') dims(3) = z_udim
       if (diag_is3D(n)) then
          id  = ncvdef (ncid,diag_name(n),NCFLOAT,4,dims,iret)
       else
          dims(3)=iTimedim
          id  = ncvdef (ncid,diag_name(n),NCFLOAT,3,dims,iret)
       endif
       call dvcdf(ncid,id,diag_longname(n),len_trim(diag_longname(n)),diag_units(n),len_trim(diag_units(n)),spval)
    enddo
    call ncendf(ncid, iret)
    iret=nf_inq_dimlen(ncid, itimedim,ilen)
    ilen=ilen+1
    fxa = itt*dt_tracer/86400.0
    iret=nf_inq_varid(ncid,'Time',itimeid)
    iret= nf_put_vara_double(ncid,itimeid,ilen,1,fxa)
 endif
 do n=1,number_diags
   if (diag_is3D(n)) then
    do k=1,nz
     bloc(is_pe:ie_pe,js_pe:je_pe) = diag_sum_var3D(n)%a(is_pe:ie_pe,js_pe:je_pe,k)/nitts
     if (diag_grid(n)(1:3) =='TTT') where( maskT(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
     if (diag_grid(n)(1:3) =='UTT') where( maskU(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
     if (diag_grid(n)(1:3) =='TUT') where( maskV(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
     if (diag_grid(n)(1:3) =='TTU') where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
     call pe0_recv_2D(nx,ny,bloc)
     if (my_pe==0) then
      iret=nf_inq_varid(ncid,diag_name(n),id)
      iret= nf_put_vara_double(ncid,id,(/1,1,k,1/), (/nx,ny,1,1/),bloc)
     endif
    enddo
    diag_sum_var3D(n)%a(:,:,:) = 0.0
   else
    bloc(is_pe:ie_pe,js_pe:je_pe) = diag_sum_var2D(n)%a(is_pe:ie_pe,js_pe:je_pe)/nitts
    if (diag_grid(n)(1:2) =='TT') where( maskT(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
    if (diag_grid(n)(1:2) =='UT') where( maskU(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
    if (diag_grid(n)(1:2) =='TU') where( maskV(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
    call pe0_recv_2D(nx,ny,bloc)
    if (my_pe==0) then
      iret=nf_inq_varid(ncid,diag_name(n),id)
      iret= nf_put_vara_double(ncid,id,(/1,1,1/), (/nx,ny,1/),bloc)
    endif
    diag_sum_var2D(n)%a(:,:) = 0.0
   endif
 enddo
 nitts = 0
 if (my_pe==0) iret=nf_close(ncid)
end subroutine write_averages





subroutine diag_averages_read_restart
!=======================================================================
! read unfinished averages from file
!=======================================================================
 use main_module
 use diag_averages_module
 implicit none
 character (len=80) :: filename
 logical :: file_exists
 integer :: io,nx_,ny_,nz_,is_,ie_,js_,je_,is,ie,js,je,n,ierr

 is=is_pe-onx; ie=ie_pe+onx; js=js_pe-onx; je=je_pe+onx
 write(filename,'(a,i5,a)')  'unfinished_averages_PE_',my_pe,'.dta'
 call replace_space_zero(filename)
 inquire ( FILE=filename, EXIST=file_exists )
 if (.not. file_exists) then
       if (my_pe==0) then
         print'(a,a,a)',' file ',filename(1:len_trim(filename)),' not present'
         print'(a)',' reading no unfinished time averages'
       endif
       return
 endif

 if (my_pe==0) print'(2a)',' reading unfinished averages from ',filename(1:len_trim(filename))
 call get_free_iounit(io,ierr)
 if (ierr/=0) goto 10
 open(io,file=filename,form='unformatted',status='old',err=10)
 read(io,err=10) nx_,ny_,nz_
 if (nx/=nx_ .or. ny/=ny_ .or. nz/= nz_) then 
       if (my_pe==0) then
        print*,' read dimensions: ',nx_,ny_,nz_
        print*,' does not match dimensions   : ',nx,ny,nz
       endif
       goto 10
 endif
 read(io,err=10) is_,ie_,js_,je_
 if (is_/=is.or.ie_/=ie.or.js_/=js.or.je_/=je) then
       if (my_pe==0) then
        print*,' read PE boundaries   ',is_,ie_,js_,je_
        print*,' which does not match ',is,ie,js,je
       endif
       goto 10
 endif
 read(io,err=10) nitts,is_
 if (number_diags /= is_) then
     if (my_pe==0) print*,' read number_diags=',is_
     if (my_pe==0) print*,' but number_diags is ',number_diags
     goto 10
 endif
 ! check for overflow
 if (number_diags > max_number_diags) then
     if (my_pe==0) print*,' too many diagnostics, increase max_number_diags'
     goto 10
 endif
 do n=1,number_diags
   if (diag_is3D(n)) then
     read(io,err=10) diag_sum_var3D(n)%a(:,:,:) 
   else
     read(io,err=10) diag_sum_var2D(n)%a(:,:)
   endif
 enddo
 close(io)
 call fortran_barrier()
 return
 10 continue
 print'(a)',' Warning: error reading file'
end subroutine diag_averages_read_restart





subroutine diag_averages_write_restart
!=======================================================================
! write unfinished averages to restart file
!=======================================================================
 use main_module
 use diag_averages_module
 implicit none
 character (len=80) :: filename
 integer :: io,is,ie,js,je,n,ierr

 is=is_pe-onx; ie=ie_pe+onx; js=js_pe-onx; je=je_pe+onx
 write(filename,'(a,i5,a)')  'unfinished_averages_PE_',my_pe,'.dta'
 call replace_space_zero(filename)

 if (my_pe==0) print'(2a)',' writing unfinished averages to ',filename(1:len_trim(filename))
 call get_free_iounit(io,ierr)
 if (ierr/=0) goto 10
 open(io,file=filename,form='unformatted',status='unknown')
 write(io,err=10) nx,ny,nz
 write(io,err=10) is,ie,js,je
 write(io,err=10) nitts,number_diags
 do n=1,number_diags
   if (diag_is3D(n)) then
     write(io,err=10) diag_sum_var3D(n)%a(is:ie,js:je,:) 
   else
     write(io,err=10) diag_sum_var2D(n)%a(is:ie,js:je)
   endif
 enddo
 close(io)
 call fortran_barrier()
 return
 10 continue
 print'(a)',' Warning: error writing file'
end subroutine diag_averages_write_restart



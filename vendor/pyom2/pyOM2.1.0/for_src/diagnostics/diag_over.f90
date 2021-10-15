


module module_diag_overturning
!=======================================================================
!  diagnose meridional overturning on isopycnals and depth
!=======================================================================
 implicit none
 integer :: nitts
 integer :: nlevel 
 real*8,allocatable :: sig(:),zarea(:,:)
 character (len=80) :: over_file
 real*8 :: p_ref = 0d0 ! in dbar
 real*8,allocatable :: mean_trans(:,:)
 real*8,allocatable :: mean_vsf_iso(:,:)
 real*8,allocatable :: mean_bolus_iso(:,:)
 real*8,allocatable :: mean_vsf_depth(:,:)
 real*8,allocatable :: mean_bolus_depth(:,:)
end module module_diag_overturning



subroutine init_diag_overturning
 use main_module
 use isoneutral_module
 use module_diag_overturning
 implicit none
 real*8 :: dsig,sigs,sige,get_rho
 integer :: i,j,k,n
 include "netcdf.inc"
 integer :: ncid,iret
 integer :: itimedim,itimeid,z_tdim,z_tid,z_udim,z_uid
 integer :: lat_udim,lat_uid,lat_tdim,lat_tid
 integer :: sig_dim,sig_id,id
 character (len=80) :: name,unit

 nitts = 0
 nlevel = nz*4
 over_file = 'over.cdf'

 allocate( sig(nlevel) );sig=0
 allocate( zarea(js_pe-onx:je_pe+onx,nz) ); zarea=0d0
 allocate( mean_trans(js_pe-onx:je_pe+onx,nlevel) );mean_trans=0
 allocate( mean_vsf_iso(js_pe-onx:je_pe+onx,nz) );mean_vsf_iso=0
 allocate( mean_vsf_depth(js_pe-onx:je_pe+onx,nz) );mean_vsf_depth=0

 if (enable_neutral_diffusion .and. enable_skew_diffusion) then
  allocate( mean_bolus_iso(js_pe-onx:je_pe+onx,nz) );mean_bolus_iso=0
  allocate( mean_bolus_depth(js_pe-onx:je_pe+onx,nz) );mean_bolus_depth=0
 endif

 ! sigma levels
 p_ref=2000.0
 sige = get_rho(35d0,-2d0,p_ref)
 sigs = get_rho(35d0,30d0,p_ref)
 dsig = (sige-sigs)/(nlevel-1.)
 if (my_pe==0) then
   print'(a)',      ' sigma ranges for overturning diagnostic:' 
   print'(a,f12.6)',' start sigma0 = ',sigs
   print'(a,f12.6)',' end sigma0   = ',sige
   print'(a,f12.6)',' Delta sigma0 = ',dsig
   if (enable_neutral_diffusion .and. enable_skew_diffusion) &
      print'(a)',      ' also calculating overturning by eddy-driven velocities' 
 endif
 do k=1,nlevel
   sig(k) = sigs + dsig*(k-1)
 enddo

 ! precalculate Area below z levels
 do j=js_pe,je_pe
  do i=is_pe,ie_pe
    zarea(j,:)=zarea(j,:)+dxt(i)*cosu(j)*maskV(i,j,:)
  enddo
 enddo
 do k=2,nz
  zarea(:,k) = zarea(:,k-1) + zarea(:,k)*dzt(k)
 enddo
 call zonal_sum_vec(zarea(js_pe:je_pe,:),nz*(je_pe-js_pe+1))

 ! prepare cdf file for output
 if (my_pe==0) then
    print'(2a)',' preparing file ',over_file(1:len_trim(over_file))
    iret = nf_create (over_file, nf_clobber, ncid)
    iret=nf_set_fill(ncid, NF_NOFILL, iret)
    iTimedim  = ncddef(ncid, 'Time', nf_unlimited, iret)
    itimeid  = ncvdef (ncid,'Time', NCFLOAT,1,itimedim,iret)
    name = 'Time '; unit = 'days'
    call ncaptc(ncid, itimeid, 'long_name', NCCHAR, len_trim(name), name, iret) 
    call ncaptc(ncid, itimeid, 'units',   NCCHAR, len_trim(unit), unit, iret) 
    call ncaptc(ncid, iTimeid,'time_origin',NCCHAR, 20,'01-JAN-1900 00:00:00', iret)
    sig_dim = ncddef(ncid, 'sigma', nlevel , iret)
    sig_id  = ncvdef (ncid,'sigma',NCFLOAT,1,sig_dim,iret)
    name = 'Sigma axis'; unit = 'kg/m^3'
    call ncaptc(ncid, sig_id, 'long_name', NCCHAR, len_trim(name), name, iret) 
    call ncaptc(ncid, sig_id, 'units',     NCCHAR, len_trim(unit), unit, iret) 

    z_tdim    = ncddef(ncid, 'zt',  nz, iret)
    z_udim    = ncddef(ncid, 'zu',  nz, iret)
    z_tid  = ncvdef (ncid,'zt', NCFLOAT,1,z_tdim,iret)
    z_uid  = ncvdef (ncid,'zu', NCFLOAT,1,z_udim,iret)
    name = 'Height on T grid     '; unit = 'm'
    call ncaptc(ncid, z_tid, 'long_name', NCCHAR, len_trim(name), name, iret) 
    call ncaptc(ncid, z_tid, 'units',     NCCHAR, len_trim(unit), unit, iret) 
    name = 'Height on U grid     '; unit = 'm'
    call ncaptc(ncid, z_uid, 'long_name', NCCHAR, len_trim(name), name, iret) 
    call ncaptc(ncid, z_uid, 'units',     NCCHAR, len_trim(unit), unit, iret) 


    Lat_udim  = ncddef(ncid,'yu', ny , iret)
    Lat_uid  = ncvdef (ncid,'yu',NCFLOAT,1,lat_udim,iret)
    Lat_tdim  = ncddef(ncid,'yt', ny , iret)
    Lat_tid  = ncvdef (ncid,'yt',NCFLOAT,1,lat_udim,iret)
    if (coord_degree) then
       name = 'Latitude on T grid     '; unit = 'degrees N'
       call ncaptc(ncid, Lat_tid, 'long_name', NCCHAR, len_trim(name), name, iret) 
       call ncaptc(ncid, Lat_tid, 'units',     NCCHAR, len_trim(unit), unit, iret) 
       name = 'Latitude on U grid     '; unit = 'degrees N'
       call ncaptc(ncid, Lat_uid, 'long_name', NCCHAR, len_trim(name), name, iret) 
       call ncaptc(ncid, Lat_uid, 'units',     NCCHAR, len_trim(unit), unit, iret) 
    else
       name = 'meridional axis T grid'; unit = 'km'
       call ncaptc(ncid, Lat_tid, 'long_name', NCCHAR, len_trim(name), name, iret) 
       call ncaptc(ncid, Lat_tid, 'units',     NCCHAR, len_trim(unit), unit, iret) 
       name = 'meridional axis U grid'; unit = 'km'
       call ncaptc(ncid, Lat_uid, 'long_name', NCCHAR, len_trim(name), name, iret) 
       call ncaptc(ncid, Lat_uid, 'units',     NCCHAR, len_trim(unit), unit, iret) 
    endif

    id  = ncvdef (ncid,'trans',NCFLOAT,3,(/lat_udim,sig_dim,itimedim/),iret)
    name = 'Meridional transport'; unit = 'm^3/s'
    call ncaptc(ncid, id, 'long_name', NCCHAR, len_trim(name), name, iret) 
    call ncaptc(ncid, id, 'units',     NCCHAR, len_trim(unit), unit, iret) 
    call ncapt (ncid,id, 'missing_value',NCDOUBLE,1,-1d33,iret)
    call ncapt (ncid,id, '_FillValue', NCDOUBLE, 1,-1d33, iret)

    id  = ncvdef (ncid,'vsf_iso',NCFLOAT,3,(/lat_udim,z_udim,itimedim/),iret)
    name = 'Meridional transport'; unit = 'm^3/s'
    call ncaptc(ncid, id, 'long_name', NCCHAR, len_trim(name), name, iret) 
    call ncaptc(ncid, id, 'units',     NCCHAR, len_trim(unit), unit, iret) 
    call ncapt (ncid,id, 'missing_value',NCDOUBLE,1,-1d33,iret)
    call ncapt (ncid,id, '_FillValue', NCDOUBLE, 1,-1d33, iret)


    id  = ncvdef (ncid,'vsf_depth',NCFLOAT,3,(/lat_udim,z_udim,itimedim/),iret)
    name = 'Meridional transport'; unit = 'm^3/s'
    call ncaptc(ncid, id, 'long_name', NCCHAR, len_trim(name), name, iret) 
    call ncaptc(ncid, id, 'units',     NCCHAR, len_trim(unit), unit, iret) 
    call ncapt (ncid,id, 'missing_value',NCDOUBLE,1,-1d33,iret)
    call ncapt (ncid,id, '_FillValue', NCDOUBLE, 1,-1d33, iret)

    if (enable_neutral_diffusion .and. enable_skew_diffusion) then
      id  = ncvdef (ncid,'bolus_iso',NCFLOAT,3,(/lat_udim,z_udim,itimedim/),iret)
      name = 'Meridional transport'; unit = 'm^3/s'
      call ncaptc(ncid, id, 'long_name', NCCHAR, len_trim(name), name, iret) 
      call ncaptc(ncid, id, 'units',     NCCHAR, len_trim(unit), unit, iret) 
      call ncapt (ncid,id, 'missing_value',NCDOUBLE,1,-1d33,iret)
      call ncapt (ncid,id, '_FillValue', NCDOUBLE, 1,-1d33, iret)

      id  = ncvdef (ncid,'bolus_depth',NCFLOAT,3,(/lat_udim,z_udim,itimedim/),iret)
      name = 'Meridional transport'; unit = 'm^3/s'
      call ncaptc(ncid, id, 'long_name', NCCHAR, len_trim(name), name, iret) 
      call ncaptc(ncid, id, 'units',     NCCHAR, len_trim(unit), unit, iret) 
      call ncapt (ncid,id, 'missing_value',NCDOUBLE,1,-1d33,iret)
      call ncapt (ncid,id, '_FillValue', NCDOUBLE, 1,-1d33, iret)
    endif


    call ncendf(ncid, iret)
    iret= nf_put_vara_double(ncid,z_tid,1,nz,zt)
    iret= nf_put_vara_double(ncid,z_uid,1,nz,zw)
    iret= nf_put_vara_double(ncid,sig_id,1,nlevel,sig)
    iret=nf_close(ncid)
 endif


 do n=0,n_pes-1
   call fortran_barrier
   if (my_pe==n) then
     iret=nf_open(over_file,NF_WRITE,ncid)
     iret=nf_inq_varid(ncid,'yt',lat_tid)
     iret=nf_inq_varid(ncid,'yu',lat_uid)
     if (coord_degree) then
       iret= nf_put_vara_double(ncid,lat_Tid,js_pe,je_pe-js_pe+1 ,yt(js_pe:je_pe))
       iret= nf_put_vara_double(ncid,lat_uid,js_pe,je_pe-js_pe+1 ,yu(js_pe:je_pe))
     else
       iret= nf_put_vara_double(ncid,lat_Tid,js_pe,je_pe-js_pe+1 ,yt(js_pe:je_pe)/1e3)
       iret= nf_put_vara_double(ncid,lat_uid,js_pe,je_pe-js_pe+1 ,yu(js_pe:je_pe)/1e3)
     endif
     iret=nf_close(ncid)
   endif
 enddo
end subroutine init_diag_overturning



subroutine diag_overturning
 use main_module
 use isoneutral_module
 use module_diag_overturning
 implicit none
 integer :: i,j,k,m,m1,m2,mm,mmp,mmm
 real*8 :: get_rho
 real*8 :: trans(js_pe-onx:je_pe+onx,nlevel),fxa
 real*8 :: z_sig(js_pe-onx:je_pe+onx,nlevel) 
 real*8 :: bolus_trans(js_pe-onx:je_pe+onx,nlevel)
 real*8 :: bolus_iso(js_pe-onx:je_pe+onx,nz)
 real*8 :: vsf_iso(js_pe-onx:je_pe+onx,nz)
 real*8 :: vsf_depth(js_pe-onx:je_pe+onx,nz)
 real*8 :: bolus_depth(js_pe-onx:je_pe+onx,nz)
 real*8 :: sig_loc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz)

 ! sigma at p_ref
 do k=1,nz
  do j=js_pe,je_pe+1
    do i=is_pe,ie_pe
      sig_loc(i,j,k) =  get_rho(salt(i,j,k,tau),temp(i,j,k,tau),p_ref)
    enddo
  enddo
 enddo

 ! transports below isopycnals and area below isopycnals
 trans=0d0; z_sig=0d0
 do j=js_pe,je_pe
  do m=1,nlevel
   do k=1,nz
     do i=is_pe,ie_pe
       fxa = 0.5*( sig_loc(i,j,k) + sig_loc(i,j+1,k))
       if (fxa > sig(m) ) then 
         trans(j,m) = trans(j,m) + v(i,j,k,tau)*dxt(i)*cosu(j)*dzt(k)*maskV(i,j,k)
         z_sig(j,m) = z_sig(j,m) + dzt(k)*dxt(i)*cosu(j)*maskV(i,j,k)
       endif
     enddo
   enddo
  enddo 
 enddo
 call zonal_sum_vec(trans(js_pe:je_pe,:),nlevel*(je_pe-js_pe+1))
 call zonal_sum_vec(z_sig(js_pe:je_pe,:),nlevel*(je_pe-js_pe+1))

 if (enable_neutral_diffusion .and. enable_skew_diffusion) then
   ! eddy driven transports below isopycnals
   bolus_trans=0d0;
   do j=js_pe,je_pe
    do m=1,nlevel
     k=1
     do i=is_pe,ie_pe
      fxa = 0.5*( sig_loc(i,j,k) + sig_loc(i,j+1,k))
      if (fxa > sig(m) ) bolus_trans(j,m) = bolus_trans(j,m) + b1_gm(i,j,k)*dxt(i)*cosu(j)*maskV(i,j,k)
     enddo
     do k=2,nz
      do i=is_pe,ie_pe
       fxa = 0.5*( sig_loc(i,j,k) + sig_loc(i,j+1,k))
       if (fxa > sig(m) ) bolus_trans(j,m) = bolus_trans(j,m) + (b1_gm(i,j,k)-b1_gm(i,j,k-1))*dxt(i)*cosu(j)*maskV(i,j,k)
      enddo
     enddo
    enddo 
   enddo
   call zonal_sum_vec(bolus_trans(js_pe:je_pe,:),nlevel*(je_pe-js_pe+1))
 endif

 ! streamfunction on geopotentials
 vsf_depth = 0d0
 do j=js_pe,je_pe
   do i=is_pe,ie_pe
    vsf_depth(j,:) = vsf_depth(j,:) + dxt(i)*cosu(j)*v(i,j,:,tau)*maskV(i,j,:)
   enddo
   do k=2,nz
     vsf_depth(j,k) = vsf_depth(j,k-1) + vsf_depth(j,k)*dzt(k)
   enddo     
 enddo     
 call zonal_sum_vec(vsf_depth(js_pe:je_pe,:),nz*(je_pe-js_pe+1))

 if (enable_neutral_diffusion .and. enable_skew_diffusion) then
   ! streamfunction for eddy driven velocity on geopotentials
   bolus_depth = 0d0
   if (enable_neutral_diffusion .and. enable_skew_diffusion) then
    do j=js_pe,je_pe
     do i=is_pe,ie_pe
      bolus_depth(j,:) = bolus_depth(j,:) + dxt(i)*cosu(j)*b1_gm(i,j,:)
     enddo
    enddo
   endif
   call zonal_sum_vec(bolus_depth(js_pe:je_pe,:),nz*(je_pe-js_pe+1))
 endif

 ! interpolate from isopcnals to depth
 if (my_blk_i==1) then
  vsf_iso = 0d0
  do j=js_pe,je_pe
   do k=1,nz
     mm= minloc( (zarea(j,k)-z_sig(j,:))**2,1 )
     mmp = min(mm+1,nlevel)
     mmm = max(mm-1,1)
     if     (z_sig(j,mm)>zarea(j,k) .and. z_sig(j,mmm) <= zarea(j,k) ) then
         m1=mmm; m2=mm
     elseif (z_sig(j,mm)>zarea(j,k) .and. z_sig(j,mmp) <= zarea(j,k) ) then
         m1=mmp; m2=mm
     elseif  (z_sig(j,mm)<zarea(j,k) .and. z_sig(j,mmp) >= zarea(j,k) ) then
         m1=mm; m2=mmp
     elseif  (z_sig(j,mm)<zarea(j,k) .and. z_sig(j,mmm) >= zarea(j,k) ) then
         m1=mm; m2=mmm
     else 
         m1=mm;m2=mm
     endif

     fxa =  z_sig(j,m2)-z_sig(j,m1)
     if (fxa /=0d0) then
      if (zarea(j,k)-z_sig(j,m1) > z_sig(j,m2)-zarea(j,k) ) then
       fxa = (zarea(j,k)-z_sig(j,m1))/fxa
       vsf_iso(j,k)=trans(j,m1)*(1-fxa) + trans(j,m2)*fxa 
       bolus_iso(j,k)=bolus_trans(j,m1)*(1-fxa) + bolus_trans(j,m2)*fxa  ! to save time
      else
       fxa = (z_sig(j,m2)-zarea(j,k))/fxa
       vsf_iso(j,k)=trans(j,m1)*fxa + trans(j,m2)*(1-fxa) 
       bolus_iso(j,k)=bolus_trans(j,m1)*fxa + bolus_trans(j,m2)*(1-fxa)
      endif
     else
      vsf_iso(j,k)=trans(j,m1) 
      bolus_iso(j,k)=bolus_trans(j,m1) 
     endif
   enddo
  enddo
 
 endif

 ! average in time
 nitts = nitts + 1
 mean_trans = mean_trans + trans
 mean_vsf_iso = mean_vsf_iso + vsf_iso
 mean_vsf_depth = mean_vsf_depth + vsf_depth
 if (enable_neutral_diffusion .and. enable_skew_diffusion) then
  mean_bolus_iso = mean_bolus_iso + bolus_iso
  mean_bolus_depth = mean_bolus_depth + bolus_depth
 endif
end subroutine diag_overturning


subroutine write_overturning
 use main_module
 use isoneutral_module
 use module_diag_overturning
 implicit none
 include "netcdf.inc"
 integer :: ncid,iret,n
 integer :: itdimid,ilen,itimeid,id
 real*8 :: fxa

 if (my_pe==0) then
   print'(a,a)',' writing overturning diagnostics to file ',over_file(1:len_trim(over_file))
   iret=nf_open(over_file,NF_WRITE,ncid)
   iret=nf_set_fill(ncid, NF_NOFILL, iret)
   iret=nf_inq_dimid(ncid,'Time',itdimid)
   iret=nf_inq_dimlen(ncid, itdimid,ilen)
   ilen=ilen+1
   fxa = itt*dt_tracer/86400.0
   iret=nf_inq_varid(ncid,'Time',itimeid)
   iret= nf_put_vara_double(ncid,itimeid,ilen,1,fxa)
   iret=nf_close(ncid)
 endif

 if (nitts/=0) then
  mean_trans = mean_trans /nitts
  mean_vsf_iso = mean_vsf_iso /nitts
  mean_vsf_depth = mean_vsf_depth /nitts
  if (enable_neutral_diffusion .and. enable_skew_diffusion) then
   mean_bolus_iso = mean_bolus_iso /nitts
   mean_bolus_depth = mean_bolus_depth /nitts
  endif
 endif

 do n=1,n_pes_j
  call fortran_barrier
  if (my_blk_j==n .and. my_blk_i==1) then
   iret=nf_open(over_file,NF_WRITE,ncid)
   iret=nf_inq_dimid(ncid,'Time',itdimid)
   iret=nf_inq_dimlen(ncid,itdimid,ilen)
   iret=nf_inq_varid(ncid,'trans',id)
   iret= nf_put_vara_double(ncid,id,(/js_pe,1,ilen/), (/je_pe-js_pe+1,nlevel,1/),mean_trans(js_pe:je_pe,:))
   iret=nf_inq_varid(ncid,'vsf_iso',id)
   iret= nf_put_vara_double(ncid,id,(/js_pe,1,ilen/), (/je_pe-js_pe+1,nz,1/),mean_vsf_iso(js_pe:je_pe,:))
   iret=nf_inq_varid(ncid,'vsf_depth',id)
   iret= nf_put_vara_double(ncid,id,(/js_pe,1,ilen/), (/je_pe-js_pe+1,nz,1/),mean_vsf_depth(js_pe:je_pe,:))
   if (enable_neutral_diffusion .and. enable_skew_diffusion) then
     iret=nf_inq_varid(ncid,'bolus_iso',id)
     iret= nf_put_vara_double(ncid,id,(/js_pe,1,ilen/), (/je_pe-js_pe+1,nz,1/),mean_bolus_iso(js_pe:je_pe,:))
     iret=nf_inq_varid(ncid,'bolus_depth',id)
     iret= nf_put_vara_double(ncid,id,(/js_pe,1,ilen/), (/je_pe-js_pe+1,nz,1/),mean_bolus_depth(js_pe:je_pe,:))
   endif
   iret=nf_close(ncid)
  endif
  call fortran_barrier
 enddo

 nitts = 0
 mean_trans = 0d0
 mean_vsf_iso = 0d0
 mean_vsf_depth =0d0
 if (enable_neutral_diffusion .and. enable_skew_diffusion) then
   mean_bolus_iso = 0d0
   mean_bolus_depth = 0d0
 endif
end subroutine write_overturning



subroutine diag_over_read_restart
!=======================================================================
! read unfinished averages from file
!=======================================================================
 use main_module
 use isoneutral_module
 use module_diag_overturning
 implicit none
 character (len=80) :: filename
 logical :: file_exists
 integer :: io,ierr,ny_,nz_,nl_,js_,je_

 if (my_blk_i>1) return ! no need to read anything
 
 write(filename,'(a,i5,a)')  'unfinished_over_PE_',my_pe,'.dta'
 call replace_space_zero(filename)
 inquire ( FILE=filename, EXIST=file_exists )
 if (.not. file_exists) then
      if (my_pe==0) then
         print'(a,a,a)',' file ',filename(1:len_trim(filename)),' not present'
         print'(a)',' reading no unfinished overturning diagnostics'
      endif
      return
 endif

 if (my_pe==0) print'(2a)',' reading unfinished averages from ',filename(1:len_trim(filename))
 call get_free_iounit(io,ierr)
 if (ierr/=0) goto 10
 open(io,file=filename,form='unformatted',status='old',err=10)
 read(io,err=10) nitts,ny_,nz_,nl_
 if (ny/=ny_ .or. nz/= nz_ .or. nl_ /=nlevel) then 
       if (my_pe==0) then
        print*,' read dimensions: ',ny_,nz_,nl_
        print*,' does not match dimensions   : ',ny,nz,nlevel
       endif
       goto 10
 endif
 read(io,err=10) js_,je_
 if (js_/=js_pe.or.je_/=je_pe) then
       if (my_pe==0) then
        print*,' read PE boundaries   ',js_,je_
        print*,' which does not match ',js_pe,je_pe
       endif
       goto 10
 endif
 if (enable_neutral_diffusion .and. enable_skew_diffusion) then
   read(io,err=10) mean_trans,mean_vsf_iso,mean_bolus_iso,mean_vsf_depth,mean_bolus_depth
 else
   read(io,err=10) mean_trans,mean_vsf_iso,mean_vsf_depth
 endif
 close(io)
 return
 10 continue
 print'(a)',' Warning: error reading file'
end subroutine diag_over_read_restart




subroutine diag_over_write_restart
!=======================================================================
! write unfinished averages to restart file
!=======================================================================
 use main_module
 use isoneutral_module
 use module_diag_overturning
 implicit none
 character (len=80) :: filename
 integer :: io,ierr

 if (my_blk_i>1) return ! no need to write anything

 write(filename,'(a,i5,a)')  'unfinished_over_PE_',my_pe,'.dta'
 call replace_space_zero(filename)
 if (my_pe==0) print'(a,a)',' writing unfinished averages to ',filename(1:len_trim(filename))
 call get_free_iounit(io,ierr)
 if (ierr/=0) goto 10
 open(io,file=filename,form='unformatted',status='unknown')
 write(io,err=10) nitts,ny,nz,nlevel
 write(io,err=10) js_pe,je_pe
 if (enable_neutral_diffusion .and. enable_skew_diffusion) then
   write(io,err=10) mean_trans,mean_vsf_iso,mean_bolus_iso,mean_vsf_depth,mean_bolus_depth
 else
   write(io,err=10) mean_trans,mean_vsf_iso,mean_vsf_depth
 endif
 close(io)
 return
 10 continue
 print'(a)',' Warning: error writing file'
end subroutine diag_over_write_restart



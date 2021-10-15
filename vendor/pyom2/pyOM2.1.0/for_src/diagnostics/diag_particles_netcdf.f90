






subroutine init_write_particles
!=======================================================================
! initialize netcdf output
!=======================================================================
 use main_module
 use particles_module
 implicit none 
 include "netcdf.inc"
 integer :: ncid,tdim,pdim,tid,xid,yid,zid,iret,vid
 character :: name*24, unit*16

 if (my_pe==0) then
   ncid = nccre ('float.cdf', NCCLOB, iret)
   iret=nf_set_fill(ncid, NF_NOFILL, iret)
   tdim = ncddef(ncid, 'Time', nf_unlimited, iret)
   pdim = ncddef(ncid, 'Number', max(1,nptraj) , iret)
   tid  = ncvdef (ncid,'Time', NCFLOAT,1,tdim,iret)
   xid  = ncvdef (ncid,'x_pos', NCFLOAT,2,(/pdim,tdim/),iret)
   yid  = ncvdef (ncid,'y_pos', NCFLOAT,2,(/pdim,tdim/),iret)
   zid  = ncvdef (ncid,'z_pos', NCFLOAT,2,(/pdim,tdim/),iret)
   call ncaptc(ncid, tid, 'long_name', NCCHAR, 4, 'Time', iret) 
   call ncaptc(ncid, tid, 'units',     NCCHAR, 4, 'days', iret) 
   call ncaptc(ncid, Tid,'time_origin',NCCHAR, 20,'01-JAN-1900 00:00:00', iret)

   if (coord_degree) then
       name = 'Longitude'; unit = 'degrees E'
   else
       name = 'zonal position'; unit = 'km'
   endif
   call ncaptc(ncid, xid, 'long_name', NCCHAR, 24, name, iret) 
   call ncaptc(ncid, xid, 'units',     NCCHAR, 16, unit, iret) 

   if (coord_degree) then
       name = 'Latitude'; unit = 'degrees N'
   else
       name = 'meridional position'; unit = 'km'
   endif
   call ncaptc(ncid, yid, 'long_name', NCCHAR, 24, name, iret) 
   call ncaptc(ncid, yid, 'units',     NCCHAR, 16, unit, iret) 

   name = 'Height'; unit = 'm'
   call ncaptc(ncid, zid, 'long_name', NCCHAR, 24, name, iret) 
   call ncaptc(ncid, zid, 'units',     NCCHAR, 16, unit, iret) 

   vid  = ncvdef (ncid,'u', NCFLOAT,2,(/pdim,tdim/),iret)
   name = 'Zonal velocity'; unit = 'm/s'
   call ncaptc(ncid, vid, 'long_name', NCCHAR, 24, name, iret) 
   call ncaptc(ncid, vid, 'units',     NCCHAR, 16, unit, iret) 

   vid  = ncvdef (ncid,'v', NCFLOAT,2,(/pdim,tdim/),iret)
   name = 'Meridional velocity'; unit = 'm/s'
   call ncaptc(ncid, vid, 'long_name', NCCHAR, 24, name, iret) 
   call ncaptc(ncid, vid, 'units',     NCCHAR, 16, unit, iret) 

   vid  = ncvdef (ncid,'w', NCFLOAT,2,(/pdim,tdim/),iret)
   name = 'Vertical velocity'; unit = 'm/s'
   call ncaptc(ncid, vid, 'long_name', NCCHAR, 24, name, iret) 
   call ncaptc(ncid, vid, 'units',     NCCHAR, 16, unit, iret) 

   vid  = ncvdef (ncid,'temp', NCFLOAT,2,(/pdim,tdim/),iret)
   name = 'Temperature'; unit = 'deg C'
   call ncaptc(ncid, vid, 'long_name', NCCHAR, 24, name, iret) 
   call ncaptc(ncid, vid, 'units',     NCCHAR, 16, unit, iret) 

   vid  = ncvdef (ncid,'salt', NCFLOAT,2,(/pdim,tdim/),iret)
   name = 'Salinity'; unit = 'g/kg'
   call ncaptc(ncid, vid, 'long_name', NCCHAR, 24, name, iret) 
   call ncaptc(ncid, vid, 'units',     NCCHAR, 16, unit, iret) 

   iret = nf_close (ncid)
 endif
end subroutine init_write_particles



subroutine write_particles
!=======================================================================
! write to netcdf file
!=======================================================================
 use main_module
 use particles_module
 implicit none 
 include "netcdf.inc"
 integer :: ncid,tdim,tid,xid,yid,zid,iret,ilen,n,uid,vid,wid,teid,sid
 real*8 :: fxa

 if (my_pe==0) then
  print*,'writing particles to file float.cdf'
  iret=nf_open('float.cdf',NF_WRITE,ncid)
  iret=nf_set_fill(ncid, NF_NOFILL, iret)
  iret=nf_inq_varid(ncid,'x_pos',xid)
  iret=nf_inq_varid(ncid,'y_pos',yid)
  iret=nf_inq_varid(ncid,'z_pos',zid)
  iret=nf_inq_varid(ncid,'u',uid)
  iret=nf_inq_varid(ncid,'v',vid)
  iret=nf_inq_varid(ncid,'w',wid)
  iret=nf_inq_varid(ncid,'temp',teid)
  iret=nf_inq_varid(ncid,'salt',sid)
  iret=nf_inq_dimid(ncid,'Time',tdim)
  iret=nf_inq_dimlen(ncid, tdim,ilen)
  iret=nf_inq_varid(ncid,'Time',tid)
  ilen=ilen+1
  fxa = itt*dt_tracer/86400.0
  iret= nf_put_vara_double(ncid,tid,ilen,1,fxa)
  do n=1,nptraj
   if (coord_degree) then
      iret= nf_put_vara_double(ncid,xid,(/n,ilen/),(/1,1/),pxyz(1,n))
      iret= nf_put_vara_double(ncid,yid,(/n,ilen/),(/1,1/),pxyz(2,n))
   else
      iret= nf_put_vara_double(ncid,xid,(/n,ilen/),(/1,1/),pxyz(1,n)/1e3)
      iret= nf_put_vara_double(ncid,yid,(/n,ilen/),(/1,1/),pxyz(2,n)/1e3)
   endif
   iret= nf_put_vara_double(ncid,zid,(/n,ilen/),(/1,1/),pxyz(3,n))
   iret= nf_put_vara_double(ncid,uid,(/n,ilen/),(/1,1/),puvw(1,n))
   iret= nf_put_vara_double(ncid,vid,(/n,ilen/),(/1,1/),puvw(2,n))
   iret= nf_put_vara_double(ncid,wid,(/n,ilen/),(/1,1/),puvw(3,n))
   iret= nf_put_vara_double(ncid,teid,(/n,ilen/),(/1,1/),pts(1,n))
   iret= nf_put_vara_double(ncid,sid,(/n,ilen/),(/1,1/),pts(2,n))
  enddo
  call ncclos (ncid, iret)
 endif
end subroutine write_particles












subroutine init_snap_cdf
!=======================================================================
!     initialize NetCDF snapshot file
!=======================================================================
 use main_module   
 use isoneutral_module
 use tke_module   
 use eke_module   
 use idemix_module   
 use diagnostics_module
 implicit none
 include "netcdf.inc"
 integer :: ncid,iret,n
 integer :: lon_tdim,lon_udim,itimedim
 integer :: lat_tdim,lat_udim,id,z_tdim,z_udim
 integer :: dims(4)
 character :: name*32, unit*16
 real*8, parameter :: spval = -1.0d33
 real*8 :: bloc(nx,ny)

 if (my_pe==0) print'(a,a)',' preparing file ',snap_file(1:len_trim(snap_file))

 call def_grid_cdf(snap_file)

 if (my_pe==0) then
      iret=nf_open(snap_file,NF_WRITE, ncid)
      iret=nf_set_fill(ncid, NF_NOFILL, iret)
      call ncredf(ncid, iret)
      iret=nf_inq_dimid(ncid,'xt',lon_tdim)
      iret=nf_inq_dimid(ncid,'xu',lon_udim)
      iret=nf_inq_dimid(ncid,'yt',lat_tdim)
      iret=nf_inq_dimid(ncid,'yu',lat_udim)
      iret=nf_inq_dimid(ncid,'zt',z_tdim)
      iret=nf_inq_dimid(ncid,'zu',z_udim)
      iret=nf_inq_dimid(ncid,'Time',itimedim)

      dims = (/Lon_tdim,lat_tdim,1,1/)
      id  = ncvdef (ncid,'ht', NCFLOAT,2,dims,iret)
      name = 'depth'; unit = 'm'
      call dvcdf(ncid,id,name,32,unit,16,spval)

      dims = (/Lon_tdim,lat_tdim,z_tdim,itimedim/)
      id  = ncvdef (ncid,'temp', NCFLOAT,4,dims,iret)
      name = 'Temperature'; unit = 'deg C'
      call dvcdf(ncid,id,name,32,unit,16,spval)

      dims = (/Lon_tdim,lat_tdim,z_tdim,itimedim/)
      id  = ncvdef (ncid,'salt', NCFLOAT,4,dims,iret)
      name = 'Salinity'; unit = 'g/kg'
      call dvcdf(ncid,id,name,32,unit,16,spval)

      dims = (/Lon_udim,lat_tdim,z_tdim,iTimedim/)
      id  = ncvdef (ncid,'u', NCFLOAT,4,dims,iret)
      name = 'Zonal velocity'; unit = 'm/s'
      call dvcdf(ncid,id,name,32,unit,16,spval)

      dims = (/Lon_tdim,lat_udim,z_tdim,iTimedim/)
      id  = ncvdef (ncid,'v', NCFLOAT,4,dims,iret)
      name = 'Meridional velocity'; unit = 'm/s'
      call dvcdf(ncid,id,name,32,unit,16,spval)

      dims = (/Lon_tdim,lat_tdim,z_udim,iTimedim/)
      id  = ncvdef (ncid,'w', NCFLOAT,4,dims,iret)
      name = 'Vertical velocity'; unit = 'm/s'
      call dvcdf(ncid,id,name,32,unit,16,spval)

      if (enable_streamfunction) then
       dims = (/Lon_udim,lat_udim,iTimedim,1/)
       id  = ncvdef (ncid,'psi', NCFLOAT,3,dims,iret)
       name = 'Streamfunction'; unit = 'm^3/s'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_udim,lat_udim,1,1/)
       do n=1,nisle
        write(name, '("psi",i3)') n
        call replace_space_zero(name)
        id  = ncvdef (ncid,name, NCFLOAT,2,dims,iret)
        write(name, '("Boundary streamfunction ",i3)') n
        unit = 'm^3/s'
        call dvcdf(ncid,id,name,32,unit,16,spval)   
       enddo
      else
       dims = (/Lon_tdim,lat_tdim,iTimedim,1/)
       id  = ncvdef (ncid,'surf_press', NCFLOAT,3,dims,iret)
       name = 'Surface pressure'; unit = 'm^2/s^2'
       call dvcdf(ncid,id,name,32,unit,16,spval)
      endif

      if (.not. enable_hydrostatic) then
       dims = (/Lon_tdim,lat_tdim,z_tdim,iTimedim/)
       id  = ncvdef (ncid,'p_hydro', NCFLOAT,4,dims,iret)
       name = 'Hydrostatic pressure'; unit = 'm^2/s^2'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_tdim,z_tdim,iTimedim/)
       id  = ncvdef (ncid,'p_non_hydro', NCFLOAT,4,dims,iret)
       name = 'Non hydrostatic pressure'; unit = 'm^2/s^2'
       call dvcdf(ncid,id,name,32,unit,16,spval)
      endif

      dims = (/Lon_tdim,lat_tdim,iTimedim,1/)
      id  = ncvdef (ncid,'forc_temp_surface', NCFLOAT,3,dims,iret)
      name = 'Surface temperature flux'; unit = 'deg C m/s'
      call dvcdf(ncid,id,name,32,unit,16,spval)

      dims = (/Lon_tdim,lat_tdim,iTimedim,1/)
      id  = ncvdef (ncid,'forc_salt_surface', NCFLOAT,3,dims,iret)
      name = 'Surface salinity flux'; unit = 'g/Kg m/s'
      call dvcdf(ncid,id,name,32,unit,16,spval)

      dims = (/Lon_udim,lat_tdim,itimedim,1/)
      id  = ncvdef (ncid,'taux', NCFLOAT,3,dims,iret)
      name = 'Surface wind stress'; unit = 'm^2/s^2'
      call dvcdf(ncid,id,name,32,unit,16,spval)

      dims = (/Lon_tdim,lat_udim,itimedim,1/)
      id  = ncvdef (ncid,'tauy', NCFLOAT,3,dims,iret)
      name = 'Surface wind stress'; unit = 'm^2/s^2'
      call dvcdf(ncid,id,name,32,unit,16,spval)

   if (enable_conserve_energy) then
      dims = (/Lon_tdim,lat_tdim,z_udim,itimedim/)
      id  = ncvdef (ncid,'K_diss_v', NCFLOAT,4,dims,iret)
      name = 'Dissipation of kin. En.'; unit = 'm^2/s^3'
      call dvcdf(ncid,id,name,32,unit,16,spval)

      dims = (/Lon_tdim,lat_tdim,z_udim,itimedim/)
      id  = ncvdef (ncid,'K_diss_bot', NCFLOAT,4,dims,iret)
      name = 'Dissipation of kin. En.'; unit = 'm^2/s^3'
      call dvcdf(ncid,id,name,32,unit,16,spval)

      dims = (/Lon_tdim,lat_tdim,z_udim,itimedim/)
      id  = ncvdef (ncid,'K_diss_h', NCFLOAT,4,dims,iret)
      name = 'Dissipation of kin. En.'; unit = 'm^2/s^3'
      call dvcdf(ncid,id,name,32,unit,16,spval)

      dims = (/Lon_tdim,lat_tdim,z_udim,itimedim/)
      id  = ncvdef (ncid,'P_diss_v', NCFLOAT,4,dims,iret)
      name = 'Dissipation of pot. En.'; unit = 'm^2/s^3'
      call dvcdf(ncid,id,name,32,unit,16,spval)

      dims = (/Lon_tdim,lat_tdim,z_udim,itimedim/)
      id  = ncvdef (ncid,'P_diss_nonlin', NCFLOAT,4,dims,iret)
      name = 'Dissipation of pot. En.'; unit = 'm^2/s^3'
      call dvcdf(ncid,id,name,32,unit,16,spval)

      dims = (/Lon_tdim,lat_tdim,z_udim,itimedim/)
      id  = ncvdef (ncid,'P_diss_iso', NCFLOAT,4,dims,iret)
      name = 'Dissipation of pot. En.'; unit = 'm^2/s^3'
      call dvcdf(ncid,id,name,32,unit,16,spval)

      dims = (/Lon_tdim,lat_tdim,z_udim,itimedim/)
      id  = ncvdef (ncid,'P_diss_skew', NCFLOAT,4,dims,iret)
      name = 'Dissipation of pot. En.'; unit = 'm^2/s^3'
      call dvcdf(ncid,id,name,32,unit,16,spval)

      dims = (/Lon_tdim,lat_tdim,z_udim,itimedim/)
      id  = ncvdef (ncid,'P_diss_hmix', NCFLOAT,4,dims,iret)
      name = 'Dissipation of pot. En.'; unit = 'm^2/s^3'
      call dvcdf(ncid,id,name,32,unit,16,spval)

      dims = (/Lon_tdim,lat_tdim,z_udim,itimedim/)
      id  = ncvdef (ncid,'K_diss_gm', NCFLOAT,4,dims,iret)
      name = 'Dissipation of mean en.'; unit = 'm^2/s^3'
      call dvcdf(ncid,id,name,32,unit,16,spval)
    endif

      dims = (/Lon_tdim,lat_tdim,z_udim,itimedim/)
      id  = ncvdef (ncid,'Nsqr', NCFLOAT,4,dims,iret)
      name = 'Square of stability frequency'; unit = '1/s^2'
      call dvcdf(ncid,id,name,32,unit,16,spval)

      dims = (/Lon_tdim,lat_tdim,z_udim,iTimedim/)
      id  = ncvdef (ncid,'kappaH', NCFLOAT,4,dims,iret)
      name = 'Vertical diffusivity'; unit = 'm^2/s'
      call dvcdf(ncid,id,name,32,unit,16,spval)

      if (enable_neutral_diffusion .and. enable_skew_diffusion) then
       dims = (/Lon_tdim,lat_udim,z_tdim,iTimedim/)
       id  = ncvdef (ncid,'B1_gm', NCFLOAT,4,dims,iret)
       name = 'Zonal comp. GM streamfct.'; unit = 'm^2/s'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_udim,lat_tdim,z_tdim,iTimedim/)
       id  = ncvdef (ncid,'B2_gm', NCFLOAT,4,dims,iret)
       name = 'Meridional comp. GM streamfct.'; unit = 'm^2/s'
       call dvcdf(ncid,id,name,32,unit,16,spval)
      endif

      if (enable_TEM_friction) then
       dims = (/Lon_tdim,lat_tdim,z_udim,iTimedim/)
       id  = ncvdef (ncid,'kappa_gm', NCFLOAT,4,dims,iret)
       name = 'Vertical diffusivity'; unit = 'm^2/s'
       call dvcdf(ncid,id,name,32,unit,16,spval)
      endif

      if (enable_tke) then

       dims = (/Lon_tdim,lat_tdim,z_udim,iTimedim/)
       id  = ncvdef (ncid,'tke', NCFLOAT,4,dims,iret)
       name = 'Turbulent kinetic energy'; unit = 'm^2/s^2'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_tdim,z_udim,iTimedim/)
       id  = ncvdef (ncid,'Prandtl', NCFLOAT,4,dims,iret)
       name = 'Prandtl number'; unit = ' '
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_tdim,z_udim,iTimedim/)
       id  = ncvdef (ncid,'mxl', NCFLOAT,4,dims,iret)
       name = 'Mixing length'; unit = 'm'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_tdim,z_udim,iTimedim/)
       id  = ncvdef (ncid,'tke_diss', NCFLOAT,4,dims,iret)
       name = 'TKE dissipation'; unit = 'm^2/s^3'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_tdim,iTimedim,1/)
       id  = ncvdef (ncid,'forc_tke', NCFLOAT,3,dims,iret)
       name = 'TKE surface flux'; unit = 'm^3/s^3'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_tdim,iTimedim,1/)
       id  = ncvdef (ncid,'tke_surf_corr', NCFLOAT,3,dims,iret)
       name = 'Correction of TKE surface flux'; unit = 'm^3/s^3'
       call dvcdf(ncid,id,name,32,unit,16,spval)

      endif

      if (enable_eke) then

       dims = (/Lon_tdim,lat_tdim,z_udim,iTimedim/)
       id  = ncvdef (ncid,'EKE', NCFLOAT,4,dims,iret)
       name = 'meso-scale energy'; unit = 'm^2/s^2'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_tdim,iTimedim,1/)
       id  = ncvdef (ncid,'L_Rossby', NCFLOAT,3,dims,iret)
       name = 'Rossby Radius'; unit = 'm'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_tdim,z_udim,iTimedim/)
       id  = ncvdef (ncid,'L_Rhines', NCFLOAT,4,dims,iret)
       name = 'Rhines scale'; unit = 'm'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_tdim,z_udim,iTimedim/)
       id  = ncvdef (ncid,'K_gm', NCFLOAT,4,dims,iret)
       name = 'skewness diffusivity'; unit = 'm^2/s'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_tdim,z_udim,iTimedim/)
       id  = ncvdef (ncid,'eke_diss_iw', NCFLOAT,4,dims,iret)
       name = 'Dissipation of EKE to IW'; unit = 'm^2/s^3'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_tdim,z_udim,iTimedim/)
       id  = ncvdef (ncid,'eke_diss_tke', NCFLOAT,4,dims,iret)
       name = 'Dissipation of EKE to TKE'; unit = 'm^2/s^3'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_tdim,iTimedim,1/)
       id  = ncvdef (ncid,'eke_bot_flux', NCFLOAT,3,dims,iret)
       name = 'flux by bottom friction'; unit = 'm^3/s^3'
       call dvcdf(ncid,id,name,len_trim(name),unit,16,spval)

       if (enable_eke_leewave_dissipation ) then

        !dims = (/Lon_tdim,lat_tdim,iTimedim,1/)
        !id  = ncvdef (ncid,'hrms_k0', NCFLOAT,3,dims,iret)
        !name = 'parameter'; unit = ' '
        !call dvcdf(ncid,id,name,32,unit,16,spval)

        dims = (/Lon_tdim,lat_tdim,iTimedim,1/)
        id  = ncvdef (ncid,'c_lee', NCFLOAT,3,dims,iret)
        name = 'Lee wave dissipation coefficient'; unit = '1/s'
        call dvcdf(ncid,id,name,len_trim(name),unit,16,spval)

        dims = (/Lon_tdim,lat_tdim,iTimedim,1/)
        id  = ncvdef (ncid,'eke_lee_flux', NCFLOAT,3,dims,iret)
        name = 'lee wave flux'; unit = 'm^3/s^3'
        call dvcdf(ncid,id,name,len_trim(name),unit,16,spval)

        dims = (/Lon_tdim,lat_tdim,z_udim,iTimedim/)
        id  = ncvdef (ncid,'c_Ri_diss', NCFLOAT,4,dims,iret)
        name = 'Interior dissipation coefficient'; unit = '1/s'
        call dvcdf(ncid,id,name,32,unit,16,spval)

       endif
      endif

      if (enable_idemix) then

       dims = (/Lon_tdim,lat_tdim,z_udim,iTimedim/)
       id  = ncvdef (ncid,'E_iw', NCFLOAT,4,dims,iret)
       name = 'Internal wave energy'; unit = 'm^2/s^2'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_tdim,z_udim,iTimedim/)
       id  = ncvdef (ncid,'c0', NCFLOAT,4,dims,iret)
       name = 'vertical IW group velocity'; unit = 'm/s'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_tdim,z_udim,iTimedim/)
       id  = ncvdef (ncid,'v0', NCFLOAT,4,dims,iret)
       name = 'hor. IW group velocity'; unit = 'm/s'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_tdim,z_udim,iTimedim/)
       id  = ncvdef (ncid,'iw_diss', NCFLOAT,4,dims,iret)
       name = 'IW dissipation'; unit = 'm^2/s^3'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_tdim,1,1/)
       id  = ncvdef (ncid,'forc_iw_surface', NCFLOAT,2,dims,iret)
       name = 'IW surface forcing'; unit = 'm^3/s^3'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_tdim,1,1/)
       id  = ncvdef (ncid,'forc_iw_bottom', NCFLOAT,2,dims,iret)
       name = 'IW bottom forcing'; unit = 'm^3/s^3'
       call dvcdf(ncid,id,name,32,unit,16,spval)

      endif

      if (enable_idemix_M2) then

       dims = (/Lon_tdim,lat_tdim,itimedim,1/)
       id  = ncvdef (ncid,'E_M2', NCFLOAT,3,dims,iret)
       name = 'M2 Energy'; unit = 'm^3/s^2'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_tdim,z_tdim,itimedim/)
       id  = ncvdef (ncid,'E_struct_M2', NCFLOAT,4,dims,iret)
       name = 'M2 structure function'; unit = ' '
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_tdim,itimedim,1/)
       id  = ncvdef (ncid,'cg_M2', NCFLOAT,3,dims,iret)
       name = 'M2 group velocity'; unit = 'm/s'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_udim,lat_tdim,itimedim,1/)
       id  = ncvdef (ncid,'kdot_x_M2', NCFLOAT,3,dims,iret)
       name = 'M2 refraction'; unit = '1/s'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_udim,itimedim,1/)
       id  = ncvdef (ncid,'kdot_y_M2', NCFLOAT,3,dims,iret)
       name = 'M2 refraction'; unit = '1/s'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_tdim,itimedim,1/)
       id  = ncvdef (ncid,'tau_M2', NCFLOAT,3,dims,iret)
       name = 'M2 decay time scale'; unit = '1/s'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_tdim,itimedim,1/)
       id  = ncvdef (ncid,'alpha_M2_cont', NCFLOAT,3,dims,iret)
       name = 'M2-continuum coupling coef'; unit = 's/m^3'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_tdim,itimedim,1/)
       id  = ncvdef (ncid,'forc_M2', NCFLOAT,3,dims,iret)
       name = 'M2 forcing'; unit = 'm^3/s^3'
       call dvcdf(ncid,id,name,32,unit,16,spval)

      endif

      if (enable_idemix_niw) then
       dims = (/Lon_tdim,lat_tdim,itimedim,1/)
       id  = ncvdef (ncid,'E_NIW', NCFLOAT,3,dims,iret)
       name = 'NIW Energy'; unit = 'm^3/s^2'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_tdim,z_tdim,itimedim/)
       id  = ncvdef (ncid,'E_struct_NIW', NCFLOAT,4,dims,iret)
       name = 'NIW structure function'; unit = ' '
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_tdim,itimedim,1/)
       id  = ncvdef (ncid,'cg_NIW', NCFLOAT,3,dims,iret)
       name = 'NIW group velocity'; unit = 'm/s'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_udim,lat_tdim,itimedim,1/)
       id  = ncvdef (ncid,'kdot_x_NIW', NCFLOAT,3,dims,iret)
       name = 'NIW refraction'; unit = '1/s'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_udim,itimedim,1/)
       id  = ncvdef (ncid,'kdot_y_NIW', NCFLOAT,3,dims,iret)
       name = 'NIW refraction'; unit = '1/s'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_tdim,itimedim,1/)
       id  = ncvdef (ncid,'tau_NIW', NCFLOAT,3,dims,iret)
       name = 'NIW decay time scale'; unit = '1/s'
       call dvcdf(ncid,id,name,32,unit,16,spval)

       dims = (/Lon_tdim,lat_tdim,itimedim,1/)
       id  = ncvdef (ncid,'forc_NIW', NCFLOAT,3,dims,iret)
       name = 'NIW forcing'; unit = 'm^3/s^3'
       call dvcdf(ncid,id,name,32,unit,16,spval)

      endif

      call ncclos (ncid, iret)
 endif
 call fortran_barrier


 if (my_pe==0) iret=nf_open(snap_file,NF_WRITE,ncid)

 bloc(is_pe:ie_pe,js_pe:je_pe) = ht(is_pe:ie_pe,js_pe:je_pe)
 where( maskT(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
 call pe0_recv_2D(nx,ny,bloc)
 if (my_pe==0) then
    iret=nf_inq_varid(ncid,'ht',id)
    iret= nf_put_vara_double(ncid,id,(/1,1/), (/nx,ny/),bloc)
 endif

 do n=1,nisle
   write(name, '("psi",i3)') n
   call replace_space_zero(name)
   bloc(is_pe:ie_pe,js_pe:je_pe) = psin(is_pe:ie_pe,js_pe:je_pe,n)
   call pe0_recv_2D(nx,ny,bloc)
   if (my_pe==0) then
    iret=nf_inq_varid(ncid,name,id)
    iret= nf_put_vara_double(ncid,id,(/1,1/), (/nx,ny/),bloc)
   endif
 enddo


 if (enable_idemix) then
   bloc(is_pe:ie_pe,js_pe:je_pe) = forc_iw_surface(is_pe:ie_pe,js_pe:je_pe)
   where( maskW(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
   call pe0_recv_2D(nx,ny,bloc)
   if (my_pe==0) then
    iret=nf_inq_varid(ncid,'forc_iw_surface',id)
    iret= nf_put_vara_double(ncid,id,(/1,1/), (/nx,ny/),bloc)
  endif

   bloc(is_pe:ie_pe,js_pe:je_pe) = forc_iw_bottom(is_pe:ie_pe,js_pe:je_pe)
   where( maskW(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
   call pe0_recv_2D(nx,ny,bloc)
   if (my_pe==0) then
    iret=nf_inq_varid(ncid,'forc_iw_bottom',id)
    iret= nf_put_vara_double(ncid,id,(/1,1/), (/nx,ny/),bloc)
   endif
 endif

 if (my_pe==0) iret=nf_close(ncid)

end subroutine init_snap_cdf


subroutine diag_snap
!=======================================================================
!     write to NetCDF snapshot file
!=======================================================================
 use main_module   
 use isoneutral_module
 use tke_module   
 use eke_module   
 use idemix_module   
 use diagnostics_module
 implicit none
 include "netcdf.inc"
 integer :: ncid,iret
 real*8 :: bloc(nx,ny),fxa
 integer :: itdimid,ilen,itimeid,id,k
 real*8, parameter :: spval = -1.0d33

 if (my_pe == 0 ) then
   iret=nf_open(snap_file,NF_WRITE,ncid)
   !iret=nf_open(snap_file,ior(NF_SHARE,NF_WRITE), ncid)
   iret=nf_set_fill(ncid, NF_NOFILL, iret)
   iret=nf_inq_dimid(ncid,'Time',itdimid)
   iret=nf_inq_dimlen(ncid, itdimid,ilen)
   ilen=ilen+1
   fxa = itt*dt_tracer/86400.0
   if (fxa <1.0) then
     print'(a,f12.2,a,i4)',' writing snapshot at ',fxa*86400,' s, time steps in file : ',ilen
   else
     print'(a,f12.2,a,i4)',' writing snapshot at ',fxa,' d, time steps in file : ',ilen
   endif
   iret=nf_inq_varid(ncid,'Time',itimeid)
   iret= nf_put_vara_double(ncid,itimeid,ilen,1,fxa)
   call ncclos (ncid, iret)
 endif
 call fortran_barrier

 if (my_pe == 0 ) then
    iret=nf_open(snap_file,NF_WRITE,ncid)
    iret=nf_inq_dimid(ncid,'Time',itdimid)
    iret=nf_inq_dimlen(ncid, itdimid,ilen)
 endif

 ! streamfunction or surface pressure
 bloc(is_pe:ie_pe,js_pe:je_pe) = psi(is_pe:ie_pe,js_pe:je_pe,tau)
 call pe0_recv_2D(nx,ny,bloc)
 if (enable_streamfunction.and.my_pe==0) then
    iret=nf_inq_varid(ncid,'psi',id)
    iret= nf_put_vara_double(ncid,id,(/1,1,ilen/), (/nx,ny,1/),bloc)
 endif
 if (.not.enable_streamfunction.and.my_pe==0) then
    iret=nf_inq_varid(ncid,'surf_press',id)
    iret= nf_put_vara_double(ncid,id,(/1,1,ilen/), (/nx,ny,1/),bloc)
 endif


 ! surface temperature flux
 bloc(is_pe:ie_pe,js_pe:je_pe) = forc_temp_surface(is_pe:ie_pe,js_pe:je_pe)
 where( maskT(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
 call pe0_recv_2D(nx,ny,bloc)
 if (my_pe==0) then
    iret=nf_inq_varid(ncid,'forc_temp_surface',id)
    iret= nf_put_vara_double(ncid,id,(/1,1,ilen/), (/nx,ny,1/),bloc)
 endif

 ! surface salinity flux
 bloc(is_pe:ie_pe,js_pe:je_pe) = forc_salt_surface(is_pe:ie_pe,js_pe:je_pe)
 where( maskT(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
 call pe0_recv_2D(nx,ny,bloc)
 if (my_pe==0) then
    iret=nf_inq_varid(ncid,'forc_salt_surface',id)
    iret= nf_put_vara_double(ncid,id,(/1,1,ilen/), (/nx,ny,1/),bloc)
 endif

 ! zonal wind stress
 bloc(is_pe:ie_pe,js_pe:je_pe) = surface_taux(is_pe:ie_pe,js_pe:je_pe)
 where( maskU(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
 call pe0_recv_2D(nx,ny,bloc)
 if (my_pe==0) then
    iret=nf_inq_varid(ncid,'taux',id)
    iret= nf_put_vara_double(ncid,id,(/1,1,ilen/), (/nx,ny,1/),bloc)
 endif

 ! meridional wind stress
 bloc(is_pe:ie_pe,js_pe:je_pe) = surface_tauy(is_pe:ie_pe,js_pe:je_pe)
 where( maskV(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
 call pe0_recv_2D(nx,ny,bloc)
 if (my_pe==0) then
    iret=nf_inq_varid(ncid,'tauy',id)
    iret= nf_put_vara_double(ncid,id,(/1,1,ilen/), (/nx,ny,1/),bloc)
 endif

 if (enable_tke) then
   ! TKE forcing
   bloc(is_pe:ie_pe,js_pe:je_pe) = forc_tke_surface(is_pe:ie_pe,js_pe:je_pe)
   where( maskW(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
   call pe0_recv_2D(nx,ny,bloc)
   if (my_pe==0) then
    iret=nf_inq_varid(ncid,'forc_tke',id)
    iret= nf_put_vara_double(ncid,id,(/1,1,ilen/), (/nx,ny,1/),bloc)
   endif

   ! surface correction of TKE forcing
   bloc(is_pe:ie_pe,js_pe:je_pe) = tke_surf_corr(is_pe:ie_pe,js_pe:je_pe)
   where( maskW(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
   call pe0_recv_2D(nx,ny,bloc)
   if (my_pe==0) then
    iret=nf_inq_varid(ncid,'tke_surf_corr',id)
    iret= nf_put_vara_double(ncid,id,(/1,1,ilen/), (/nx,ny,1/),bloc)
   endif
 endif

 if (enable_eke) then
  ! Rossby radius
  bloc(is_pe:ie_pe,js_pe:je_pe) = L_rossby(is_pe:ie_pe,js_pe:je_pe)
  where( maskW(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
  call pe0_recv_2D(nx,ny,bloc)
  if (my_pe==0) then
    iret=nf_inq_varid(ncid,'L_Rossby',id)
    iret= nf_put_vara_double(ncid,id,(/1,1,ilen/), (/nx,ny,1/),bloc)
  endif

  bloc(is_pe:ie_pe,js_pe:je_pe) = eke_bot_flux(is_pe:ie_pe,js_pe:je_pe)
  where( maskW(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
  call pe0_recv_2D(nx,ny,bloc)
  if (my_pe==0) then  
      iret=nf_inq_varid(ncid,'eke_bot_flux',id)
      iret= nf_put_vara_double(ncid,id,(/1,1,ilen/), (/nx,ny,1/),bloc)  
  endif

  if (enable_eke_leewave_dissipation ) then
    !bloc(is_pe:ie_pe,js_pe:je_pe) = hrms_k0(is_pe:ie_pe,js_pe:je_pe)
    !where( maskW(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
    !call pe0_recv_2D(nx,ny,bloc)
    !if (my_pe==0) then  
    !  iret=nf_inq_varid(ncid,'hrms_k0',id)
    !  iret= nf_put_vara_double(ncid,id,(/1,1,ilen/), (/nx,ny,1/),bloc)  
    !endif

    bloc(is_pe:ie_pe,js_pe:je_pe) = c_lee(is_pe:ie_pe,js_pe:je_pe)
    where( maskW(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
    call pe0_recv_2D(nx,ny,bloc)
    if (my_pe==0) then  
      iret=nf_inq_varid(ncid,'c_lee',id)
      iret= nf_put_vara_double(ncid,id,(/1,1,ilen/), (/nx,ny,1/),bloc)  
    endif

    bloc(is_pe:ie_pe,js_pe:je_pe) = eke_lee_flux(is_pe:ie_pe,js_pe:je_pe)
    where( maskW(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
    call pe0_recv_2D(nx,ny,bloc)
    if (my_pe==0) then  
      iret=nf_inq_varid(ncid,'eke_lee_flux',id)
      iret= nf_put_vara_double(ncid,id,(/1,1,ilen/), (/nx,ny,1/),bloc)  
    endif

  endif
 endif

 if (enable_idemix_M2) then

  bloc = 0d0
  do k=2,np-1
    bloc(is_pe:ie_pe,js_pe:je_pe) = bloc(is_pe:ie_pe,js_pe:je_pe) + &
           E_M2(is_pe:ie_pe,js_pe:je_pe,k,tau)*dphit(k)*maskTp(is_pe:ie_pe,js_pe:je_pe,k)
  enddo
  where( maskT(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
  call pe0_recv_2D(nx,ny,bloc)
  if (my_pe==0) then
    iret=nf_inq_varid(ncid,'E_M2',id)
    iret= nf_put_vara_double(ncid,id,(/1,1,ilen/), (/nx,ny,1/),bloc)
  endif

  bloc(is_pe:ie_pe,js_pe:je_pe) = cg_M2(is_pe:ie_pe,js_pe:je_pe)
  where( maskT(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
  call pe0_recv_2D(nx,ny,bloc)
  if (my_pe==0) then
    iret=nf_inq_varid(ncid,'cg_M2',id)
    iret= nf_put_vara_double(ncid,id,(/1,1,ilen/), (/nx,ny,1/),bloc)
  endif

  bloc(is_pe:ie_pe,js_pe:je_pe) = kdot_x_M2(is_pe:ie_pe,js_pe:je_pe)
  where( maskU(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
  call pe0_recv_2D(nx,ny,bloc)
  if (my_pe==0) then
    iret=nf_inq_varid(ncid,'kdot_x_M2',id)
    iret= nf_put_vara_double(ncid,id,(/1,1,ilen/), (/nx,ny,1/),bloc)
  endif

  bloc(is_pe:ie_pe,js_pe:je_pe) = kdot_y_M2(is_pe:ie_pe,js_pe:je_pe)
  where( maskV(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
  call pe0_recv_2D(nx,ny,bloc)
  if (my_pe==0) then
    iret=nf_inq_varid(ncid,'kdot_y_M2',id)
    iret= nf_put_vara_double(ncid,id,(/1,1,ilen/), (/nx,ny,1/),bloc)
  endif

  bloc(is_pe:ie_pe,js_pe:je_pe) = tau_M2(is_pe:ie_pe,js_pe:je_pe)
  where( maskT(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
  call pe0_recv_2D(nx,ny,bloc)
  if (my_pe==0) then
    iret=nf_inq_varid(ncid,'tau_M2',id)
    iret= nf_put_vara_double(ncid,id,(/1,1,ilen/), (/nx,ny,1/),bloc)
  endif

  bloc(is_pe:ie_pe,js_pe:je_pe) = alpha_M2_cont(is_pe:ie_pe,js_pe:je_pe)
  where( maskT(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
  call pe0_recv_2D(nx,ny,bloc)
  if (my_pe==0) then
    iret=nf_inq_varid(ncid,'alpha_M2_cont',id)
    iret= nf_put_vara_double(ncid,id,(/1,1,ilen/), (/nx,ny,1/),bloc)
  endif

  bloc = 0d0
  do k=2,np-1
    bloc(is_pe:ie_pe,js_pe:je_pe) = bloc(is_pe:ie_pe,js_pe:je_pe) + &
           forc_M2(is_pe:ie_pe,js_pe:je_pe,k)*dphit(k)*maskTp(is_pe:ie_pe,js_pe:je_pe,k)
  enddo
  where( maskT(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
  call pe0_recv_2D(nx,ny,bloc)
  if (my_pe==0) then
    iret=nf_inq_varid(ncid,'forc_M2',id)
    iret= nf_put_vara_double(ncid,id,(/1,1,ilen/), (/nx,ny,1/),bloc)
  endif



 endif

 if (enable_idemix_niw) then

  bloc = 0d0
  do k=2,np-1
    bloc(is_pe:ie_pe,js_pe:je_pe) = bloc(is_pe:ie_pe,js_pe:je_pe) + &
           E_niw(is_pe:ie_pe,js_pe:je_pe,k,tau)*dphit(k)*maskTp(is_pe:ie_pe,js_pe:je_pe,k)
  enddo
  where( maskT(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
  call pe0_recv_2D(nx,ny,bloc)
  if (my_pe==0) then
    iret=nf_inq_varid(ncid,'E_NIW',id)
    iret= nf_put_vara_double(ncid,id,(/1,1,ilen/), (/nx,ny,1/),bloc)
  endif

  bloc(is_pe:ie_pe,js_pe:je_pe) = cg_niw(is_pe:ie_pe,js_pe:je_pe)
  where( maskT(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
  call pe0_recv_2D(nx,ny,bloc)
  if (my_pe==0) then
    iret=nf_inq_varid(ncid,'cg_NIW',id)
    iret= nf_put_vara_double(ncid,id,(/1,1,ilen/), (/nx,ny,1/),bloc)
  endif

  bloc(is_pe:ie_pe,js_pe:je_pe) = kdot_x_niw(is_pe:ie_pe,js_pe:je_pe)
  where( maskU(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
  call pe0_recv_2D(nx,ny,bloc)
  if (my_pe==0) then
    iret=nf_inq_varid(ncid,'kdot_x_NIW',id)
    iret= nf_put_vara_double(ncid,id,(/1,1,ilen/), (/nx,ny,1/),bloc)
  endif

  bloc(is_pe:ie_pe,js_pe:je_pe) = kdot_y_niw(is_pe:ie_pe,js_pe:je_pe)
  where( maskV(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
  call pe0_recv_2D(nx,ny,bloc)
  if (my_pe==0) then
    iret=nf_inq_varid(ncid,'kdot_y_NIW',id)
    iret= nf_put_vara_double(ncid,id,(/1,1,ilen/), (/nx,ny,1/),bloc)
  endif

  bloc(is_pe:ie_pe,js_pe:je_pe) = tau_niw(is_pe:ie_pe,js_pe:je_pe)
  where( maskT(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
  call pe0_recv_2D(nx,ny,bloc)
  if (my_pe==0) then
    iret=nf_inq_varid(ncid,'tau_NIW',id)
    iret= nf_put_vara_double(ncid,id,(/1,1,ilen/), (/nx,ny,1/),bloc)
  endif

  bloc = 0d0
  do k=2,np-1
    bloc(is_pe:ie_pe,js_pe:je_pe) = bloc(is_pe:ie_pe,js_pe:je_pe) + &
           forc_niw(is_pe:ie_pe,js_pe:je_pe,k)*dphit(k)*maskTp(is_pe:ie_pe,js_pe:je_pe,k)
  enddo
  where( maskT(is_pe:ie_pe,js_pe:je_pe,nz) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
  call pe0_recv_2D(nx,ny,bloc)
  if (my_pe==0) then
    iret=nf_inq_varid(ncid,'forc_NIW',id)
    iret= nf_put_vara_double(ncid,id,(/1,1,ilen/), (/nx,ny,1/),bloc)
  endif


 endif




 do k=1,nz

   if (.not. enable_hydrostatic) then
    ! hydrostatic pressure 
    bloc(is_pe:ie_pe,js_pe:je_pe) = p_hydro(is_pe:ie_pe,js_pe:je_pe,k)
    where( maskT(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
    call pe0_recv_2D(nx,ny,bloc)
    if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'p_hydro',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
    endif

    ! non hydrostatic pressure 
    bloc(is_pe:ie_pe,js_pe:je_pe) = p_non_hydro(is_pe:ie_pe,js_pe:je_pe,k,taup1)
    where( maskT(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
    call pe0_recv_2D(nx,ny,bloc)
    if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'p_non_hydro',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
    endif
   endif

   ! zonal velocity
   bloc(is_pe:ie_pe,js_pe:je_pe) = u(is_pe:ie_pe,js_pe:je_pe,k,tau)
   where( maskU(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
   call pe0_recv_2D(nx,ny,bloc)
   if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'u',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
   endif

   ! meridional velocity
   bloc(is_pe:ie_pe,js_pe:je_pe) = v(is_pe:ie_pe,js_pe:je_pe,k,tau)
   where( maskV(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
   call pe0_recv_2D(nx,ny,bloc)
   if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'v',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
   endif

   ! vertical velocity
   bloc(is_pe:ie_pe,js_pe:je_pe) = w(is_pe:ie_pe,js_pe:je_pe,k,tau)
   where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
   call pe0_recv_2D(nx,ny,bloc)
   if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'w',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
   endif

   ! temperature
   bloc(is_pe:ie_pe,js_pe:je_pe) = temp(is_pe:ie_pe,js_pe:je_pe,k,tau)
   where( maskT(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
   call pe0_recv_2D(nx,ny,bloc)
   if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'temp',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
   endif

   ! salinity
   bloc(is_pe:ie_pe,js_pe:je_pe) = salt(is_pe:ie_pe,js_pe:je_pe,k,tau)
   where( maskT(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
   call pe0_recv_2D(nx,ny,bloc)
   if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'salt',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
   endif

 if (enable_conserve_energy) then
   ! dissipation
   bloc(is_pe:ie_pe,js_pe:je_pe) = K_diss_v(is_pe:ie_pe,js_pe:je_pe,k)
   where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
   call pe0_recv_2D(nx,ny,bloc)
   if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'K_diss_v',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
   endif

   ! dissipation
   bloc(is_pe:ie_pe,js_pe:je_pe) = K_diss_bot(is_pe:ie_pe,js_pe:je_pe,k)
   where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
   call pe0_recv_2D(nx,ny,bloc)
   if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'K_diss_bot',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
   endif

   ! dissipation
   bloc(is_pe:ie_pe,js_pe:je_pe) = K_diss_h(is_pe:ie_pe,js_pe:je_pe,k)
   where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
   call pe0_recv_2D(nx,ny,bloc)
   if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'K_diss_h',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
   endif

   ! dissipation
   bloc(is_pe:ie_pe,js_pe:je_pe) = K_diss_gm(is_pe:ie_pe,js_pe:je_pe,k)
   where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
   call pe0_recv_2D(nx,ny,bloc)
   if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'K_diss_gm',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
   endif

   ! dissipation
   bloc(is_pe:ie_pe,js_pe:je_pe) = P_diss_v(is_pe:ie_pe,js_pe:je_pe,k)
   where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
   call pe0_recv_2D(nx,ny,bloc)
   if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'P_diss_v',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
   endif

   ! dissipation
   bloc(is_pe:ie_pe,js_pe:je_pe) = P_diss_hmix(is_pe:ie_pe,js_pe:je_pe,k)
   where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
   call pe0_recv_2D(nx,ny,bloc)
   if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'P_diss_hmix',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
   endif

   ! dissipation
   bloc(is_pe:ie_pe,js_pe:je_pe) = P_diss_nonlin(is_pe:ie_pe,js_pe:je_pe,k)
   where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
   call pe0_recv_2D(nx,ny,bloc)
   if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'P_diss_nonlin',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
   endif

   ! dissipation
   bloc(is_pe:ie_pe,js_pe:je_pe) = P_diss_iso(is_pe:ie_pe,js_pe:je_pe,k)
   where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
   call pe0_recv_2D(nx,ny,bloc)
   if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'P_diss_iso',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
   endif

   ! dissipation
   bloc(is_pe:ie_pe,js_pe:je_pe) = P_diss_skew(is_pe:ie_pe,js_pe:je_pe,k)
   where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
   call pe0_recv_2D(nx,ny,bloc)
   if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'P_diss_skew',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
   endif
  endif
 
   ! stability frequency
   bloc(is_pe:ie_pe,js_pe:je_pe) = Nsqr(is_pe:ie_pe,js_pe:je_pe,k,tau)
   where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
   call pe0_recv_2D(nx,ny,bloc)
   if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'Nsqr',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
   endif

   ! vertical diffusivity
   bloc(is_pe:ie_pe,js_pe:je_pe) = kappaH(is_pe:ie_pe,js_pe:je_pe,k)
   where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
   call pe0_recv_2D(nx,ny,bloc)
   if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'kappaH',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
   endif

   if (enable_neutral_diffusion .and. enable_skew_diffusion) then
    bloc(is_pe:ie_pe,js_pe:je_pe) = B1_gm(is_pe:ie_pe,js_pe:je_pe,k)
    where( maskV(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
    call pe0_recv_2D(nx,ny,bloc)
    if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'B1_gm',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
    endif

    bloc(is_pe:ie_pe,js_pe:je_pe) = B2_gm(is_pe:ie_pe,js_pe:je_pe,k)
    where( maskU(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
    call pe0_recv_2D(nx,ny,bloc)
    if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'B2_gm',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
    endif
   endif

   if (enable_TEM_friction) then
    ! vertical GM diffusivity
    bloc(is_pe:ie_pe,js_pe:je_pe) = kappa_gm(is_pe:ie_pe,js_pe:je_pe,k)
    where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
    call pe0_recv_2D(nx,ny,bloc)
    if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'kappa_gm',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
    endif
   endif

   if (enable_tke) then
    ! TKE
    bloc(is_pe:ie_pe,js_pe:je_pe) = tke(is_pe:ie_pe,js_pe:je_pe,k,tau)
    where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
    call pe0_recv_2D(nx,ny,bloc)
    if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'tke',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
    endif
   
    ! Prandtl zahl
    bloc(is_pe:ie_pe,js_pe:je_pe) = Prandtlnumber(is_pe:ie_pe,js_pe:je_pe,k)
    where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
    call pe0_recv_2D(nx,ny,bloc)
    if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'Prandtl',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
    endif

    ! mixing length
    bloc(is_pe:ie_pe,js_pe:je_pe) = mxl(is_pe:ie_pe,js_pe:je_pe,k)
    where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
    call pe0_recv_2D(nx,ny,bloc)
    if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'mxl',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
    endif

    ! TKE dissipation
    bloc(is_pe:ie_pe,js_pe:je_pe) = tke_diss(is_pe:ie_pe,js_pe:je_pe,k)
    where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
    call pe0_recv_2D(nx,ny,bloc)
    if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'tke_diss',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
    endif


   endif

   if (enable_eke) then

    ! EKE
    bloc(is_pe:ie_pe,js_pe:je_pe) = eke(is_pe:ie_pe,js_pe:je_pe,k,tau)
    where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
    call pe0_recv_2D(nx,ny,bloc)
    if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'EKE',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
    endif

    ! Rhines scale
    bloc(is_pe:ie_pe,js_pe:je_pe) = L_Rhines(is_pe:ie_pe,js_pe:je_pe,k)
    where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
    call pe0_recv_2D(nx,ny,bloc)
    if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'L_Rhines',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
    endif

    ! GM diffusivity
    bloc(is_pe:ie_pe,js_pe:je_pe) = K_gm(is_pe:ie_pe,js_pe:je_pe,k)
    where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
    call pe0_recv_2D(nx,ny,bloc)
    if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'K_gm',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
    endif

    ! Eke dissipation
    bloc(is_pe:ie_pe,js_pe:je_pe) = eke_diss_iw(is_pe:ie_pe,js_pe:je_pe,k)
    where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
    call pe0_recv_2D(nx,ny,bloc)
    if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'eke_diss_iw',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
    endif

    ! Eke dissipation
    bloc(is_pe:ie_pe,js_pe:je_pe) = eke_diss_tke(is_pe:ie_pe,js_pe:je_pe,k)
    where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
    call pe0_recv_2D(nx,ny,bloc)
    if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'eke_diss_tke',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
    endif

    if (enable_eke_leewave_dissipation ) then
     bloc(is_pe:ie_pe,js_pe:je_pe) = c_Ri_diss(is_pe:ie_pe,js_pe:je_pe,k)
     where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
     call pe0_recv_2D(nx,ny,bloc)
     if (my_pe == 0 ) then 
        iret=nf_inq_varid(ncid,'c_Ri_diss',id)
        iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
     endif

    endif
   endif

   if (enable_idemix) then

    ! Internal wave energy
    bloc(is_pe:ie_pe,js_pe:je_pe) = E_iw(is_pe:ie_pe,js_pe:je_pe,k,tau)
    where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
    call pe0_recv_2D(nx,ny,bloc)
    if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'E_iw',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
    endif

    ! Internal wave dissipation
    bloc(is_pe:ie_pe,js_pe:je_pe) = iw_diss(is_pe:ie_pe,js_pe:je_pe,k)
    where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
    call pe0_recv_2D(nx,ny,bloc)
    if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'iw_diss',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
    endif

    ! Internal wave vertical group velocity
    bloc(is_pe:ie_pe,js_pe:je_pe) = c0(is_pe:ie_pe,js_pe:je_pe,k)
    where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
    call pe0_recv_2D(nx,ny,bloc)
    if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'c0',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
    endif

    ! Internal wave horizontal group velocity
    bloc(is_pe:ie_pe,js_pe:je_pe) = v0(is_pe:ie_pe,js_pe:je_pe,k)
    where( maskW(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
    call pe0_recv_2D(nx,ny,bloc)
    if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'v0',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
    endif
   endif

   if (enable_idemix_M2) then
    bloc(is_pe:ie_pe,js_pe:je_pe) = E_struct_M2(is_pe:ie_pe,js_pe:je_pe,k)
    where( maskT(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
    call pe0_recv_2D(nx,ny,bloc)
    if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'E_struct_M2',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
    endif
   endif

   if (enable_idemix_niw) then
    bloc(is_pe:ie_pe,js_pe:je_pe) = E_struct_niw(is_pe:ie_pe,js_pe:je_pe,k)
    where( maskT(is_pe:ie_pe,js_pe:je_pe,k) == 0.) bloc(is_pe:ie_pe,js_pe:je_pe) = spval
    call pe0_recv_2D(nx,ny,bloc)
    if (my_pe == 0 ) then 
       iret=nf_inq_varid(ncid,'E_struct_NIW',id)
       iret= nf_put_vara_double(ncid,id,(/1,1,k,ilen/), (/nx,ny,1,1/),bloc)
    endif
   endif


 enddo

 if (my_pe==0)   iret = nf_close (ncid)

end subroutine diag_snap



subroutine panic_snap
!=======================================================================
! write a snapshot before panic shutdown
!=======================================================================
 use main_module
 use diagnostics_module
 if (.not. enable_diag_snapshots) call init_snap_cdf
 call diag_snap
end subroutine panic_snap



subroutine def_grid_cdf(filename)
!=======================================================================
!      Define standard grid in netcdf file
!=======================================================================
 use main_module   
 implicit none
 include "netcdf.inc"
 character*(*) filename
 integer :: ncid,iret,n
 integer :: lon_tdim,lon_udim,itimedim
 integer :: lon_tid,lon_uid,itimeid
 integer :: lat_tdim,lat_udim,lat_uid,lat_tid
 integer :: z_tdim,z_udim,z_tid,z_uid
 character :: name*24, unit*16

 if (my_pe==0) then
      iret = nf_create (filename, nf_clobber, ncid)
      if (iret /= 0) call halt_stop('NETCDF:'//nf_strerror(iret))
      iret=nf_set_fill(ncid, NF_NOFILL, iret)
!     dimensions
      lon_tdim  = ncddef(ncid, 'xt', nx , iret)
      Lon_udim  = ncddef(ncid, 'xu', nx , iret)
      lat_tdim  = ncddef(ncid, 'yt', ny , iret)
      Lat_udim  = ncddef(ncid, 'yu', ny , iret)
      iTimedim  = ncddef(ncid, 'Time', nf_unlimited, iret)
!     grid variables
      Lon_tid  = ncvdef (ncid,'xt',NCFLOAT,1,lon_tdim,iret)
      Lon_uid  = ncvdef (ncid,'xu',NCFLOAT,1,lon_udim,iret)
      Lat_tid  = ncvdef (ncid,'yt',NCFLOAT,1,lat_tdim,iret)
      Lat_uid  = ncvdef (ncid,'yu',NCFLOAT,1,lat_udim,iret)
      itimeid  = ncvdef (ncid,'Time', NCFLOAT,1,itimedim,iret)
!     attributes of the grid
      if (coord_degree) then
       name = 'Longitude on T grid     '; unit = 'degrees E'
       call ncaptc(ncid, Lon_tid, 'long_name', NCCHAR, 24, name, iret) 
       call ncaptc(ncid, Lon_tid, 'units',     NCCHAR, 16, unit, iret) 
       name = 'Longitude on U grid     '; unit = 'degrees E'
       call ncaptc(ncid, Lon_uid, 'long_name', NCCHAR, 24, name, iret) 
       call ncaptc(ncid, Lon_uid, 'units',     NCCHAR, 16, unit, iret) 
       name = 'Latitude on T grid     '; unit = 'degrees N'
       call ncaptc(ncid, Lat_tid, 'long_name', NCCHAR, 24, name, iret) 
       call ncaptc(ncid, Lat_tid, 'units',     NCCHAR, 16, unit, iret) 
       name = 'Latitude on U grid     '; unit = 'degrees N'
       call ncaptc(ncid, Lat_uid, 'long_name', NCCHAR, 24, name, iret) 
       call ncaptc(ncid, Lat_uid, 'units',     NCCHAR, 16, unit, iret) 
      else
       name = 'zonal axis T grid     '; unit = 'km'
       call ncaptc(ncid, Lon_tid, 'long_name', NCCHAR, 24, name, iret) 
       call ncaptc(ncid, Lon_tid, 'units',     NCCHAR, 16, unit, iret) 
       name = 'zonal axis U grid     '; unit = 'km'
       call ncaptc(ncid, Lon_uid, 'long_name', NCCHAR, 24, name, iret) 
       call ncaptc(ncid, Lon_uid, 'units',     NCCHAR, 16, unit, iret) 
       name = 'meridional axis T grid'; unit = 'km'
       call ncaptc(ncid, Lat_tid, 'long_name', NCCHAR, 24, name, iret) 
       call ncaptc(ncid, Lat_tid, 'units',     NCCHAR, 16, unit, iret) 
       name = 'meridional axis U grid'; unit = 'km'
       call ncaptc(ncid, Lat_uid, 'long_name', NCCHAR, 24, name, iret) 
       call ncaptc(ncid, Lat_uid, 'units',     NCCHAR, 16, unit, iret) 
      endif

      name = 'Time '; unit = 'days'
      call ncaptc(ncid, itimeid, 'long_name', NCCHAR, 24, name, iret) 
      call ncaptc(ncid, itimeid, 'units',     NCCHAR, 16, unit, iret) 
      call ncaptc(ncid, iTimeid,'time_origin',NCCHAR, 20,'01-JAN-1900 00:00:00', iret)

      z_tdim    = ncddef(ncid, 'zt',  nz, iret)
      z_udim    = ncddef(ncid, 'zu',  nz, iret)
      z_tid  = ncvdef (ncid,'zt', NCFLOAT,1,z_tdim,iret)
      z_uid  = ncvdef (ncid,'zu', NCFLOAT,1,z_udim,iret)
      name = 'Height on T grid     '; unit = 'm'
      call ncaptc(ncid, z_tid, 'long_name', NCCHAR, 24, name, iret) 
      call ncaptc(ncid, z_tid, 'units',     NCCHAR, 16, unit, iret) 
      name = 'Height on U grid     '; unit = 'm'
      call ncaptc(ncid, z_uid, 'long_name', NCCHAR, 24, name, iret) 
      call ncaptc(ncid, z_uid, 'units',     NCCHAR, 16, unit, iret) 

      call ncendf(ncid, iret)
      iret= nf_put_vara_double(ncid,z_tid,1,nz ,zt)
      iret= nf_put_vara_double(ncid,z_uid,1,nz ,zw)
      call ncclos (ncid, iret)
 endif


 do n=0,n_pes-1
   call fortran_barrier
   if (my_pe==n) then
      iret=nf_open(filename,NF_WRITE,ncid)
      iret=nf_inq_varid(ncid,'xt',lon_tid)
      iret=nf_inq_varid(ncid,'xu',lon_uid)
      iret=nf_inq_varid(ncid,'yt',lat_tid)
      iret=nf_inq_varid(ncid,'yu',lat_uid)
      if (coord_degree) then
       iret= nf_put_vara_double(ncid,lon_Tid,is_pe,ie_pe-is_pe+1 ,xt(is_pe:ie_pe))
       iret= nf_put_vara_double(ncid,lon_uid,is_pe,ie_pe-is_pe+1 ,xu(is_pe:ie_pe))
       iret= nf_put_vara_double(ncid,lat_Tid,js_pe,je_pe-js_pe+1 ,yt(js_pe:je_pe))
       iret= nf_put_vara_double(ncid,lat_uid,js_pe,je_pe-js_pe+1 ,yu(js_pe:je_pe))
      else
       iret= nf_put_vara_double(ncid,lon_Tid,is_pe,ie_pe-is_pe+1 ,xt(is_pe:ie_pe)/1e3)
       iret= nf_put_vara_double(ncid,lon_uid,is_pe,ie_pe-is_pe+1 ,xu(is_pe:ie_pe)/1e3)
       iret= nf_put_vara_double(ncid,lat_Tid,js_pe,je_pe-js_pe+1 ,yt(js_pe:je_pe)/1e3)
       iret= nf_put_vara_double(ncid,lat_uid,js_pe,je_pe-js_pe+1 ,yu(js_pe:je_pe)/1e3)
      endif
      call ncclos (ncid, iret)
   endif
   call fortran_barrier
 enddo
end subroutine def_grid_cdf




subroutine dvcdf(ncid,ivarid,name,iname,unit,iunit,spval)
!=======================================================================
!     define some standard attributes of variable ivarid in NetCDF file ncid 
!=======================================================================
 implicit none
 integer ncid,ivarid,iname,iunit,iret
 character (len=*) :: name, unit
 real*8 :: spval, vv
 include "netcdf.inc"
 vv=spval
 call ncaptc(ncid,ivarid, 'long_name', NCCHAR,iname , name, iret) 
 if (iret.ne.0) print*,nf_strerror(iret)
 call ncaptc(ncid,ivarid, 'units',     NCCHAR,iunit, unit, iret) 
 if (iret.ne.0) print*,nf_strerror(iret)
 call ncapt (ncid,ivarid, 'missing_value',NCDOUBLE,1,vv,iret)
 if (iret.ne.0) print*,nf_strerror(iret)
 call ncapt (ncid,ivarid, '_FillValue', NCDOUBLE, 1,vv, iret)
 if (iret.ne.0) print*,nf_strerror(iret)
end subroutine dvcdf





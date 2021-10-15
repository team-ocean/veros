

!
!  test modules in density.f90 and write TSP diagrams in cdf file
!

program check_density
 use nonlin1_eq_of_state
 use nonlin2_eq_of_state
 use gsw_eq_of_state
 implicit none
 integer, parameter :: nx=40,ny=40,nz=40
 real*8 :: T(nx),S(ny),P(nz),rho1(nx,ny,nz),rho2(nx,ny,nz),rho3(nx,ny,nz)
 real*8 :: drho1dT(nx,ny,nz),drho2dT(nx,ny,nz),drho3dT(nx,ny,nz)
 real*8 :: drho1dS(nx,ny,nz),drho2dS(nx,ny,nz),drho3dS(nx,ny,nz)
 real*8 :: drho1dP(nx,ny,nz),drho2dP(nx,ny,nz),drho3dP(nx,ny,nz)
 real*8 :: Hd1(nx,ny,nz),Hd2(nx,ny,nz),Hd3(nx,ny,nz)
 real*8 :: dHd3dT(nx,ny,nz),dHd2dT(nx,ny,nz)
 real*8 :: dHd3dS(nx,ny,nz),dHd2dS(nx,ny,nz)
 integer :: i,j,k
 include "netcdf.inc"
 integer :: iret,ncid,tdim,sdim,pdim,tid,sid,pid,rho1id,rho2id,rho3id
 integer :: drho1dTid,drho2dTid,drho3dTid
 integer :: drho1dSid,drho2dSid,drho3dSid
 integer :: drho1dPid,drho2dPid,drho3dPid
 integer :: Hd1id,Hd2id,Hd3id
 integer :: dHd3dTid
 integer :: dHd2dTid
 integer :: dHd3dSid
 integer :: dHd2dSid


 do i=1,nx
  do j=1,ny
   do k=1,nz
    T(i) = -2.+27.*(i-1.0)/nx
    S(j) = 33+4*(j-1.0)/ny
    !S(j) = 35*(j-1.0)/ny
    P(k) = 5000*(k-1.)/nz
    rho3(i,j,k) = gsw_rho(S(j),T(i),P(k))
    rho2(i,j,k) = nonlin2_eq_of_state_rho(S(j),T(i),P(k))
    drho3dT(i,j,k) = gsw_drhodT(S(j),T(i),P(k))
    drho2dT(i,j,k) = nonlin2_eq_of_state_drhodT(T(i),P(k))
    drho3dS(i,j,k) = gsw_drhodS(S(j),T(i),P(k))
    drho2dS(i,j,k) = nonlin2_eq_of_state_drhodS()
    drho3dP(i,j,k) = gsw_drhodP(S(j),T(i),P(k))
    drho2dP(i,j,k) = nonlin2_eq_of_state_drhodP(T(i))
    Hd3(i,j,k) = gsw_dyn_enthalpy(S(j),T(i),P(k))
    Hd2(i,j,k) = nonlin2_eq_of_state_dyn_enthalpy(S(j),T(i),P(k))
    dHd3dT(i,j,k) = gsw_dHdT(S(j),T(i),P(k))
    dHd3dS(i,j,k) = gsw_dHdS(S(j),T(i),P(k))
    dHd2dT(i,j,k) = nonlin2_eq_of_state_int_drhodT(T(i),P(k))*(-9.81/1024.)
    dHd2dS(i,j,k) = nonlin2_eq_of_state_int_drhodS(P(k))*(-9.81/1024.)
   enddo
  enddo
 enddo


 iret = nf_create ('check_density.cdf', nf_clobber, ncid)
 tdim  = ncddef(ncid, 'T', nx , iret)
 sdim  = ncddef(ncid, 'S', ny , iret)
 pdim  = ncddef(ncid, 'P', nz , iret)
 tid  = ncvdef (ncid,'T',NCFLOAT,1,tdim,iret)
 sid  = ncvdef (ncid,'S',NCFLOAT,1,sdim,iret)
 pid  = ncvdef (ncid,'P',NCFLOAT,1,pdim,iret)
 rho1id  = ncvdef (ncid,'rho1',NCFLOAT,3,(/tdim,sdim,pdim/),iret)
 rho2id  = ncvdef (ncid,'rho2',NCFLOAT,3,(/tdim,sdim,pdim/),iret)
 rho3id  = ncvdef (ncid,'rho3',NCFLOAT,3,(/tdim,sdim,pdim/),iret)
 Hd1id  = ncvdef (ncid,'Hd1',NCFLOAT,3,(/tdim,sdim,pdim/),iret)
 Hd2id  = ncvdef (ncid,'Hd2',NCFLOAT,3,(/tdim,sdim,pdim/),iret)
 Hd3id  = ncvdef (ncid,'Hd3',NCFLOAT,3,(/tdim,sdim,pdim/),iret)
 drho1dTid  = ncvdef (ncid,'drho1dT',NCFLOAT,3,(/tdim,sdim,pdim/),iret)
 drho2dTid  = ncvdef (ncid,'drho2dT',NCFLOAT,3,(/tdim,sdim,pdim/),iret)
 drho3dTid  = ncvdef (ncid,'drho3dT',NCFLOAT,3,(/tdim,sdim,pdim/),iret)
 drho1dSid  = ncvdef (ncid,'drho1dS',NCFLOAT,3,(/tdim,sdim,pdim/),iret)
 drho2dSid  = ncvdef (ncid,'drho2dS',NCFLOAT,3,(/tdim,sdim,pdim/),iret)
 drho3dSid  = ncvdef (ncid,'drho3dS',NCFLOAT,3,(/tdim,sdim,pdim/),iret)
 drho1dPid  = ncvdef (ncid,'drho1dP',NCFLOAT,3,(/tdim,sdim,pdim/),iret)
 drho2dPid  = ncvdef (ncid,'drho2dP',NCFLOAT,3,(/tdim,sdim,pdim/),iret)
 drho3dPid  = ncvdef (ncid,'drho3dP',NCFLOAT,3,(/tdim,sdim,pdim/),iret)
 dHd3dTid  = ncvdef (ncid,'dHd3dT',NCFLOAT,3,(/tdim,sdim,pdim/),iret)
 dHd3dSid  = ncvdef (ncid,'dHd3dS',NCFLOAT,3,(/tdim,sdim,pdim/),iret)
 dHd2dTid  = ncvdef (ncid,'dHd2dT',NCFLOAT,3,(/tdim,sdim,pdim/),iret)
 dHd2dSid  = ncvdef (ncid,'dHd2dS',NCFLOAT,3,(/tdim,sdim,pdim/),iret)
 iret = NF_PUT_ATT_TEXT (ncid, tid,'name',len('temperature'),'temperature')
 iret = NF_PUT_ATT_TEXT (ncid, tid,'unit',len('deg C'),'deg C')
 iret = NF_PUT_ATT_TEXT (ncid, sid,'name',len('salinity'),'salinity')
 iret = NF_PUT_ATT_TEXT (ncid, sid,'unit',len('g/kg'),'g/kg')
 iret = NF_PUT_ATT_TEXT (ncid, pid,'name',len('pressure'),'pressure')
 iret = NF_PUT_ATT_TEXT (ncid, pid,'unit',len('dbar'),'dbar')
 iret = NF_PUT_ATT_TEXT (ncid, rho1id,'name',len('density'),'density')
 iret = NF_PUT_ATT_TEXT (ncid, rho1id,'unit',len('kg/m^3'),'kg/m^3')
 iret = NF_PUT_ATT_TEXT (ncid, rho2id,'name',len('density'),'density')
 iret = NF_PUT_ATT_TEXT (ncid, rho2id,'unit',len('kg/m^3'),'kg/m^3')
 iret = NF_PUT_ATT_TEXT (ncid, rho3id,'name',len('density'),'density')
 iret = NF_PUT_ATT_TEXT (ncid, rho3id,'unit',len('kg/m^3'),'kg/m^3')
 iret = nf_enddef(ncid)
 iret = nf_put_vara_double(ncid,tid,1,nx ,T)
 iret = nf_put_vara_double(ncid,sid,1,ny ,S)
 iret = nf_put_vara_double(ncid,pid,1,nz ,P)
 iret = nf_put_vara_double(ncid,rho1id,(/1,1,1/),(/nx,ny,nz/) ,rho1)
 iret = nf_put_vara_double(ncid,rho2id,(/1,1,1/),(/nx,ny,nz/) ,rho2)
 iret = nf_put_vara_double(ncid,rho3id,(/1,1,1/),(/nx,ny,nz/) ,rho3)
 iret = nf_put_vara_double(ncid,Hd1id,(/1,1,1/),(/nx,ny,nz/) ,Hd1)
 iret = nf_put_vara_double(ncid,Hd2id,(/1,1,1/),(/nx,ny,nz/) ,Hd2)
 iret = nf_put_vara_double(ncid,Hd3id,(/1,1,1/),(/nx,ny,nz/) ,Hd3)
 iret = nf_put_vara_double(ncid,drho1dTid,(/1,1,1/),(/nx,ny,nz/) ,drho1dT)
 iret = nf_put_vara_double(ncid,drho2dTid,(/1,1,1/),(/nx,ny,nz/) ,drho2dT)
 iret = nf_put_vara_double(ncid,drho3dTid,(/1,1,1/),(/nx,ny,nz/) ,drho3dT)
 iret = nf_put_vara_double(ncid,drho1dSid,(/1,1,1/),(/nx,ny,nz/) ,drho1dS)
 iret = nf_put_vara_double(ncid,drho2dSid,(/1,1,1/),(/nx,ny,nz/) ,drho2dS)
 iret = nf_put_vara_double(ncid,drho3dSid,(/1,1,1/),(/nx,ny,nz/) ,drho3dS)
 iret = nf_put_vara_double(ncid,drho1dPid,(/1,1,1/),(/nx,ny,nz/) ,drho1dP)
 iret = nf_put_vara_double(ncid,drho2dPid,(/1,1,1/),(/nx,ny,nz/) ,drho2dP)
 iret = nf_put_vara_double(ncid,drho3dPid,(/1,1,1/),(/nx,ny,nz/) ,drho3dP)
 iret = nf_put_vara_double(ncid,dHd3dTid,(/1,1,1/),(/nx,ny,nz/) ,dHd3dT)
 iret = nf_put_vara_double(ncid,dHd3dSid,(/1,1,1/),(/nx,ny,nz/) ,dHd3dS)
 iret = nf_put_vara_double(ncid,dHd2dTid,(/1,1,1/),(/nx,ny,nz/) ,dHd2dT)
 iret = nf_put_vara_double(ncid,dHd2dSid,(/1,1,1/),(/nx,ny,nz/) ,dHd2dS)
 iret = nf_close (ncid)
end program check_density

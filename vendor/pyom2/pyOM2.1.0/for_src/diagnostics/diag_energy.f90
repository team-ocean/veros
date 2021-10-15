



module diag_energy_module
!=======================================================================
! module to store globally averaged energy diagnostics
!=======================================================================
implicit none
 integer :: nitts = 0
 real*8 :: mean_e(50)
 real*8 :: mean_dedt(50)
 real*8 :: mean_diss(50)
 real*8 :: mean_forc(50)
 real*8 :: mean_exchg(50)
 real*8 :: mean_misc(50)
end module diag_energy_module



subroutine diagnose_energy
!=======================================================================
! Diagnose globally averaged energy cycle
!=======================================================================
 use main_module   
 use tke_module   
 use eke_module   
 use idemix_module   
 use isoneutral_module   
 use diag_energy_module
 implicit none
 integer :: i,j,k
 real*8 :: fxa,mdiss_vmix,wind,dKm,wrhom,Km,Pm,TKEm,dPm,dTKEm,TKEdiss,TKEforc,corm,dPvmix
 real*8 :: IWm,IWforc,dIWm,IWdiss,mdiss_v,mdiss_h,spm,dEKEm,EKEdiss,EKEm,mdiss_gm,mdiss_nonlin
 real*8 :: KEadv,dPhmix,mdiss_adv,mdiss_comp,mdiss_hmix,mdiss_iso,dP_iso,mdiss_sources,mdiss_bot
 real*8 :: NIWm,NIWforc,dNIWm,NIWdiss, M2m,M2forc,dM2m,M2diss,EKEdiss_tke,mdiss_skew,dPm_all

 !---------------------------------------------------------------------------------
 ! changes of dynamic enthalpy  
 !---------------------------------------------------------------------------------
  dPvmix=0; dPhmix=0; dPm=0.; dP_iso=0
  do k=1,nz 
    do j=js_pe,je_pe
     do i=is_pe,ie_pe
      fxa = area_t(i,j)*dzt(k)*maskT(i,j,k)
      dP_iso  = dP_iso + fxa*grav/rho_0*(-int_drhodT(i,j,k,tau)*dtemp_iso(i,j,k) -int_drhodS(i,j,k,tau)*dsalt_iso(i,j,k) )
      dPhmix = dPhmix + fxa*grav/rho_0*(-int_drhodT(i,j,k,tau)*dtemp_hmix(i,j,k)-int_drhodS(i,j,k,tau)*dsalt_hmix(i,j,k) )
      dPvmix = dPvmix + fxa*grav/rho_0*(-int_drhodT(i,j,k,taup1)*dtemp_vmix(i,j,k)-int_drhodS(i,j,k,taup1)*dsalt_vmix(i,j,k) )
      dPm = dPm + fxa*grav/rho_0*(  &  
            -int_drhodT(i,j,k,tau)*dtemp(i,j,k,tau) -int_drhodS(i,j,k,tau)*dsalt(i,j,k,tau) )! this should be identical to g rho w
     enddo
    enddo
  enddo
  call global_sum(dPvmix); call global_sum(dPhmix); call global_sum(dPm); call global_sum(dP_iso)
  dPm_all = dPm+dPvmix+dPhmix+dP_iso

 !---------------------------------------------------------------------------------
 ! changes of kinetic energy
 !---------------------------------------------------------------------------------
  Km = 0; Pm = 0; dKm=0; spm=0.; corm=0.; KEadv=0.;
  do k=1,nz 
    do j=js_pe,je_pe
     do i=is_pe,ie_pe
      fxa = area_t(i,j)*dzt(k)*maskT(i,j,k)
      Km = Km + fxa*0.5*( 0.5*(u(i,j,k,tau)**2+u(i-1,j,k,tau)**2) + 0.5*(v(i,j,k,tau)**2+v(i,j-1,k,tau)**2) ) 
      Pm = Pm + fxa*Hd(i,j,k,tau) ! grav/rho_0*rho(i,j,k,tau)*zt(k)
      dKm = dKm + u(i,j,k,tau)*du(i,j,k,tau)*area_u(i,j)*dzt(k) &
                 +v(i,j,k,tau)*dv(i,j,k,tau)*area_v(i,j)*dzt(k) &
                 +u(i,j,k,tau)*du_mix(i,j,k)*area_u(i,j)*dzt(k) &
                 +v(i,j,k,tau)*dv_mix(i,j,k)*area_v(i,j)*dzt(k)
      corm = corm + u(i,j,k,tau)*du_cor(i,j,k)*area_u(i,j)*dzt(k) &
                  + v(i,j,k,tau)*dv_cor(i,j,k)*area_v(i,j)*dzt(k)
      KEadv = KEadv + u(i,j,k,tau)*du_adv(i,j,k)*area_u(i,j)*dzt(k)*maskU(i,j,k) &  !  
                    + v(i,j,k,tau)*dv_adv(i,j,k)*area_v(i,j)*dzt(k)*maskV(i,j,k)    ! 
     enddo
    enddo
  enddo

 !---------------------------------------------------------------------------------
 ! spurious work by surface pressure
 !---------------------------------------------------------------------------------
  if (.not.enable_streamfunction) then
   do k=1,nz 
    do j=js_pe,je_pe
     do i=is_pe,ie_pe
      fxa = area_t(i,j)*dzt(k)*maskT(i,j,k)
      spm = spm - u(i,j,k,tau)*area_u(i,j)*(psi(i+1,j,tau)-psi(i,j,tau)) /(dxu(i)*cost(j)) *dzt(k) &
                - v(i,j,k,tau)*area_v(i,j)*(psi(i,j+1,tau)-psi(i,j,tau))  /dyu(j) *dzt(k)
     enddo
    enddo
   enddo
  endif
  call global_sum(Km); call global_sum(Pm); call global_sum(dKm)
  call global_sum(spm); call global_sum(corm); call global_sum(KEadv)

 !---------------------------------------------------------------------------------
 !  K*Nsqr and KE and dyn. Enthalpy dissipation
 !---------------------------------------------------------------------------------
   mdiss_gm=0; mdiss_v=0; mdiss_h=0; mdiss_vmix=0; mdiss_adv=0; mdiss_nonlin=0; mdiss_comp = 0; mdiss_hmix =0
   mdiss_iso=0; mdiss_sources = 0; mdiss_bot=0; mdiss_skew = 0.0
   do k=1,nz 
    do j=js_pe,je_pe
      do i=is_pe,ie_pe
        fxa = area_t(i,j)*dzw(k)*maskW(i,j,k)
        if (k==nz) fxa=fxa*0.5
        mdiss_vmix  = mdiss_vmix   + P_diss_v(i,j,k)*fxa
        mdiss_nonlin= mdiss_nonlin + P_diss_nonlin(i,j,k)*fxa
        mdiss_adv   = mdiss_adv + P_diss_adv(i,j,k)*fxa
        mdiss_hmix  = mdiss_hmix+ P_diss_hmix(i,j,k)*fxa
        mdiss_comp  = mdiss_comp+ P_diss_comp(i,j,k)*fxa
        mdiss_iso   = mdiss_iso+ P_diss_iso(i,j,k)*fxa
        mdiss_skew  = mdiss_skew+ P_diss_skew(i,j,k)*fxa
        mdiss_sources = mdiss_sources+ P_diss_sources(i,j,k)*fxa

        mdiss_h = mdiss_h + K_diss_h(i,j,k)*fxa
        mdiss_v = mdiss_v + K_diss_v(i,j,k)*fxa
        mdiss_gm= mdiss_gm+ K_diss_gm(i,j,k)*fxa
        mdiss_bot= mdiss_bot+ K_diss_bot(i,j,k)*fxa
      enddo
    enddo
   enddo
   call global_sum(mdiss_nonlin); call global_sum(mdiss_adv); call global_sum(mdiss_vmix)
   call global_sum(mdiss_h); call global_sum(mdiss_v); call global_sum(mdiss_gm)
   call global_sum(mdiss_comp); call global_sum(mdiss_hmix); call global_sum(mdiss_iso);  
   call global_sum(mdiss_sources); call global_sum(mdiss_bot); call global_sum(mdiss_skew)


 !---------------------------------------------------------------------------------
 ! wrhom = + g rho w
 !---------------------------------------------------------------------------------
   wrhom=0
   do j=js_pe,je_pe
    do i=is_pe,ie_pe 
     do k=1,nz-1
      fxa = area_t(i,j)*maskW(i,j,k)
      wrhom = wrhom - fxa*(p_hydro(i,j,k+1)-p_hydro(i,j,k))*w(i,j,k,tau)
     enddo
    enddo
   enddo
   call global_sum(wrhom); 


 !---------------------------------------------------------------------------------
 ! wind work
 !---------------------------------------------------------------------------------
   wind=0
   do j=js_pe,je_pe
    do i=is_pe,ie_pe
      wind    = wind + u(i,j,nz,tau)*surface_taux(i,j)*maskU(i,j,nz)*area_u(i,j)
      wind    = wind + v(i,j,nz,tau)*surface_tauy(i,j)*maskV(i,j,nz)*area_v(i,j)
    enddo
   enddo
   call global_sum(wind); 

 !---------------------------------------------------------------------------------
 ! internal wave energy
 !---------------------------------------------------------------------------------
   Iwm=0; dIWm=0; IWdiss=0; IWforc=0
   if (enable_idemix) then

     do k=1,nz 
      do j=js_pe,je_pe
       do i=is_pe,ie_pe
        fxa = area_t(i,j)*dzw(k)*maskW(i,j,k)
        if (k==nz) fxa=fxa*0.5
        IWm = IWm + fxa*E_iw(i,j,k,tau)
        dIWm = dIWm + fxa*(E_iw(i,j,k,taup1)-E_iw(i,j,k,tau))/dt_tracer
        IWdiss = IWdiss + fxa*iw_diss(i,j,k)
       enddo
      enddo
     enddo
     do j=js_pe,je_pe
      do i=is_pe,ie_pe
        k=max(1,kbot(i,j))
        IWforc=IWforc + area_t(i,j)*(forc_iw_surface(i,j)*maskW(i,j,nz) &
                                    +forc_iw_bottom(i,j)*maskW(i,j,k))
      enddo
     enddo
     call global_sum(IWm); call global_sum(dIWm); call global_sum(IWdiss); call global_sum(IWforc); 
   endif

 !---------------------------------------------------------------------------------
 ! NIW low mode  compartment
 !---------------------------------------------------------------------------------
   NIWm = 0; dNIWm = 0; NIWforc = 0; NIWdiss=0
   if (enable_idemix .and. enable_idemix_niw) then
      do k=2,np-1
       do j=js_pe,je_pe
        do i=is_pe,ie_pe
          fxa = area_t(i,j)*dphit(k)*maskTp(i,j,k)
          NIWm = NIWm + fxa*E_niw(i,j,k,tau)
          dNIWm = dNIWm + fxa*(E_niw(i,j,k,taup1)-E_niw(i,j,k,tau))/dt_tracer
          NIWforc=NIWforc + fxa*forc_niw(i,j,k)
          NIWdiss = NIWdiss + fxa*tau_niw(i,j)*E_niw(i,j,k,tau)
        enddo
       enddo
      enddo
     call global_sum(NIWm); call global_sum(dNIWm); call global_sum(NIWdiss); call global_sum(NIWforc); 
   endif

 !---------------------------------------------------------------------------------
 ! M2 low mode  compartment
 !---------------------------------------------------------------------------------
   M2m = 0; dM2m = 0; M2forc = 0; M2diss=0
   if (enable_idemix .and. enable_idemix_M2) then
      do k=2,np-1
       do j=js_pe,je_pe
        do i=is_pe,ie_pe
          fxa = area_t(i,j)*dphit(k)*maskTp(i,j,k)
          M2m = M2m + fxa*E_M2(i,j,k,tau)
          dM2m = dM2m + fxa*(E_M2(i,j,k,taup1)-E_M2(i,j,k,tau))/dt_tracer
          M2forc=M2forc + fxa*forc_M2(i,j,k)
          M2diss = M2diss + fxa*(tau_M2(i,j)*E_M2(i,j,k,tau)+M2_psi_diss(i,j,k))
        enddo
       enddo
      enddo
     call global_sum(M2m); call global_sum(dM2m); call global_sum(M2diss); call global_sum(M2forc); 
   endif

 !---------------------------------------------------------------------------------
 ! meso-scale energy
 !---------------------------------------------------------------------------------
   EKEm=0; dEKEm=0; EKEdiss=0; EKEdiss_tke=0
   if (enable_eke) then
     do k=1,nz 
      do j=js_pe,je_pe
       do i=is_pe,ie_pe
        fxa = area_t(i,j)*dzw(k)*maskW(i,j,k)
        if (k==nz) fxa=fxa*0.5
        EKEm = EKEm + fxa*eke(i,j,k,tau)
        dEKEm = dEKEm + fxa*(eke(i,j,k,taup1)-eke(i,j,k,tau))/dt_tracer
        EKEdiss = EKEdiss + fxa*eke_diss_iw(i,j,k)
        EKEdiss_tke = EKEdiss_tke + fxa*eke_diss_tke(i,j,k)
       enddo
      enddo
     enddo
     call global_sum(EKEm); call global_sum(dEKEm); call global_sum(EKEdiss); call global_sum(EKEdiss_tke); 
   endif

 !---------------------------------------------------------------------------------
 ! small-scale energy
 !---------------------------------------------------------------------------------
   TKEm=0; dTKEm=0; TKEdiss=0; TKEforc=0
   if (enable_tke) then
     do k=1,nz 
      do j=js_pe,je_pe
       do i=is_pe,ie_pe
        fxa = area_t(i,j)*dzw(k)*maskW(i,j,k)
        if (k==nz) fxa=fxa*0.5
        TKEm = TKEm + fxa*tke(i,j,k,tau)
        dTKEm = dTKEm + fxa*(tke(i,j,k,taup1)-tke(i,j,k,tau))/dt_tke
        TKEdiss = TKEdiss + fxa*tke_diss(i,j,k)
       enddo
      enddo
     enddo
     call global_sum(TKEm); call global_sum(dTKEm); call global_sum(TKEdiss); 
     do j=js_pe,je_pe
      do i=is_pe,ie_pe
        fxa = area_t(i,j)*maskW(i,j,nz)
        TKEforc=TKEforc + fxa*(forc_tke_surface(i,j)+tke_surf_corr(i,j))
      enddo
     enddo
     call global_sum(TKEforc); 
   endif

 !---------------------------------------------------------------------------------
 ! average and store
 !---------------------------------------------------------------------------------
   if (.not. enable_eke)    then ! short cut for EKE model
      EKEdiss = mdiss_gm + mdiss_h - mdiss_skew
      if (.not.enable_store_cabbeling_heat)  EKEdiss = EKEdiss  - mdiss_hmix  - mdiss_iso
   endif
   if (.not. enable_idemix) IWdiss  = EKEdiss ! short cut for IW model
 
   nitts = nitts + 1
   mean_e(1:7)    = mean_e(1:7)    + (/Km,Pm,EKEm,IWm,TKEm,NIWm,M2m/)
   mean_dedt(1:7) = mean_dedt(1:7) + (/dKm,dPm_all+mdiss_sources,dEKEm,dIWm,dTKEm,dNIWm,dM2m/)
   mean_forc(1:7) = mean_forc(1:7) + (/wind,mdiss_sources,0d0,iwforc,tkeforc,niwforc,M2forc/)

   mean_diss(1)   = mean_diss(1) + mdiss_h+mdiss_v+mdiss_gm+mdiss_bot
   mean_diss(2)   = mean_diss(2) + mdiss_vmix+mdiss_nonlin+mdiss_hmix+mdiss_adv+mdiss_iso+mdiss_skew
   mean_diss(3)   = mean_diss(3) + EKEdiss+EKEdiss_tke
   mean_diss(4:7) = mean_diss(4:7) + (/IWdiss,TKEdiss,NIWdiss,M2diss/)

   mean_exchg(1) = mean_exchg(1) + wrhom                ! KE -> HD
   mean_exchg(2) = mean_exchg(2) + mdiss_h + mdiss_gm   ! KE -> EKE
   mean_exchg(3) = mean_exchg(3) - mdiss_skew           ! Hd -> EKE
   mean_exchg(4) = mean_exchg(4) + mdiss_v              ! Ke -> TKE
   mean_exchg(5) = mean_exchg(5) - mdiss_vmix-mdiss_adv ! TKE-> Hd
   if (enable_store_bottom_friction_tke)  then 
     mean_exchg(4) = mean_exchg(4) + mdiss_bot          ! Ke -> TKE
   else
     mean_exchg(6) = mean_exchg(6) + mdiss_bot          ! KE -> IW
   endif
   mean_exchg(7) = mean_exchg(7) + EKEdiss_tke          ! EKE-> TKE
   mean_exchg(8) = mean_exchg(8) + EKEdiss              ! EKE-> IW
   if (.not.enable_store_cabbeling_heat) then
         mean_exchg(3) = mean_exchg(3)-mdiss_hmix-mdiss_iso
         mean_exchg(5) = mean_exchg(5)-mdiss_nonlin
   endif

   mean_misc(1:3) = mean_misc(1:3) + (/corm,spm,KEadv/)
   mean_misc(4) = mean_misc(4) + mdiss_nonlin
   mean_misc(5) = mean_misc(5) + mdiss_adv
   mean_misc(6) = mean_misc(6) + mdiss_bot
   mean_misc(7) = mean_misc(7) + EKEdiss
   mean_misc(8) = mean_misc(8) + EKEdiss_tke
   mean_misc(9) = mean_misc(9) + mdiss_hmix + mdiss_iso

   mean_misc(20) = mean_misc(20) + mdiss_h 
   mean_misc(21) = mean_misc(21) + mdiss_gm

end subroutine diagnose_energy


subroutine write_energy
!=======================================================================
! write energy diagnostics to standard out and to netcdf file
!=======================================================================
 use main_module   
 use eke_module   
 use tke_module   
 use idemix_module   
 use isoneutral_module   
 use diagnostics_module
 use diag_energy_module
 implicit none
 include "netcdf.inc"
 integer :: ncid,iret,id,timedim,i
 real*8 :: fxa,fxb

 mean_e = mean_e/nitts
 mean_dedt = mean_dedt/nitts
 mean_diss = mean_diss/nitts
 mean_forc = mean_forc/nitts
 mean_exchg = mean_exchg/nitts
 mean_misc  = mean_misc /nitts

 if (my_pe==0) then

    fxa = rho_0/1d15
    print'(/,a,f12.2)',' Energy content averaged to day ',itt*dt_tracer/86400.0
    print'(a,f12.3,a)',' kin. energy  ',mean_e(1)*fxa,' PetaJ ' 
    print'(a,f12.3,a)',' dyn. enthalpy',mean_e(2)*rho_0/1d21,' ZettaJ ' 
    print'(a,f12.3,a)',' eddy energy  ',mean_e(3)*fxa,' PetaJ ' 
    print'(a,f12.3,a)',' IW. energy   ',mean_e(4)*fxa,' PetaJ ' 
    if (enable_idemix .and. enable_idemix_niw) print'(a,f12.3,a)',' NIW. energy  ',mean_e(6)*fxa,' PetaJ ' 
    if (enable_idemix .and. enable_idemix_M2)  print'(a,f12.3,a)',' M2 tidal en. ',mean_e(7)*fxa,' PetaJ ' 
    print'(a,f12.3,a)',' tur. energy  ',mean_e(5)*fxa,' PetaJ ' 
    print*,'  '
    fxa = rho_0/1d12

    print'(a)',' Mean kin. energy budget:'
    print'(a,e12.6,a)',' wind work      ',mean_forc(1)*fxa,'  TW'
    print'(a,e12.6,a)',' dissipation    ',mean_diss(1)*fxa,'  TW'
    print'(a,e12.6,a)',' KE -> Hd       ',mean_exchg(1)*fxa,'  TW'
    print'(a,e12.6,a)',' KE ->EKE(fric) ',mean_misc(20)*fxa,'  TW'
    print'(a,e12.6,a)',' KE -> EKE (GM) ',mean_misc(21)*fxa,'  TW'
    print'(a,e12.6,a)',' KE -> TKE      ',mean_exchg(4)*fxa,'  TW'
    print'(a,e12.6,a)',' KE -> IW       ',mean_exchg(6)*fxa,'  TW'
    print'(a,e12.6,a)',' dKE/dT         ',mean_dedt(1)*fxa,'  TW'
    print'(a,e12.6,a)',' error          ',(mean_dedt(1)-mean_forc(1)+mean_diss(1)+mean_exchg(1))*fxa,'  TW'
    print'(a,e12.6,a)',' adv/Cor/Press  ',sum(mean_misc(1:3))*fxa,'  TW'
    print*,'  '

    print'(a)',' Mean dynamic enthalpy budget:'
    print'(a,e12.6,a)',' external forc. ',mean_forc(2)*fxa,'  TW'
    print'(a,e12.6,a)',' dissipation    ',mean_diss(2)*fxa,'  TW'
    print'(a,e12.6,a)',' cabb. (vert)   ',mean_misc(4)*fxa,'  TW'
    print'(a,e12.6,a)',' cabb. (iso)    ',mean_misc(9)*fxa,'  TW'
    print'(a,e12.6,a)',' Hd -> EKE      ',mean_exchg(3)*fxa,'  TW'
    print'(a,e12.6,a)',' Hd -> TKE      ',-mean_exchg(5)*fxa,'  TW'
    print'(a,e12.6,a)',' advection      ',mean_misc(5)*fxa,'  TW'
    print'(a,e12.6,a)',' dHd/dT         ',mean_dedt(2)*fxa,'  TW'
    print'(a,e12.6,a)',' error          ',(mean_dedt(2)-mean_forc(2)-mean_diss(2)-mean_exchg(1))*fxa,'  TW'
    print*,'  '

    if (enable_eke) then
      print'(a)',' EKE budget:'
      print'(a,e12.6,a)',' dEKE/dT        ',mean_dedt(3)*fxa,'  TW'
      print'(a,e12.6,a)',' dissipation    ',mean_diss(3)*fxa,'  TW'
      print'(a,e12.6,a)',' KE  -> EKE     ',mean_exchg(2)*fxa,'  TW'
      print'(a,e12.6,a)',' Hd  -> EKE     ',mean_exchg(3)*fxa,'  TW'
      print'(a,e12.6,a)',' EKE -> TKE     ',mean_exchg(7)*fxa,'  TW'
      print'(a,e12.6,a)',' EKE -> IW      ',mean_exchg(8)*fxa,'  TW'
      print'(a,e12.6,a)',' error          ',(mean_dedt(3)-mean_forc(3)+mean_diss(3)-mean_exchg(2)-mean_exchg(3) )*fxa,'  TW'
      print*,' '
    endif

    if (enable_idemix) then
      print'(a)',' Internal wave energy budget:'
      print'(a,e12.6,a)',' external forc. ',mean_forc(4)*fxa,'  TW'
      print'(a,e12.6,a)',' dE_iw/dT       ',mean_dedt(4)*fxa,'  TW'
      print'(a,e12.6,a)',' dissipation    ',mean_diss(4)*fxa,'  TW'
      print'(a,e12.6,a)',' EKE ->  IW     ',mean_exchg(8)*fxa,'  TW'
      print'(a,e12.6,a)',' KE  ->  IW     ',mean_exchg(6)*fxa,'  TW'
      fxb = mean_dedt(4)-mean_forc(4)+mean_diss(4)-mean_exchg(8)-mean_exchg(6) 
      if (enable_idemix_niw) fxb = fxb - mean_diss(6)
      print'(a,e12.6,a)',' error          ',fxb*fxa,'  TW'
      print*,' '
    endif

    if (enable_idemix .and. enable_idemix_niw) then
      print'(a)',' LOW mode NIW energy budget:'
      print'(a,e12.6,a)',' external forc. ',mean_forc(6)*fxa,'  TW'
      print'(a,e12.6,a)',' dE_niw/dT      ',mean_dedt(6)*fxa,'  TW'
      print'(a,e12.6,a)',' dissipation    ',mean_diss(6)*fxa,'  TW'
      print'(a,e12.6,a)',' error          ',(mean_dedt(6)-mean_forc(6)+mean_diss(6))*fxa,'  TW'
      print*,' '
    endif

    if (enable_idemix .and. enable_idemix_M2) then
      print'(a)',' LOW mode M2 tide energy budget:'
      print'(a,e12.6,a)',' external forc. ',mean_forc(7)*fxa,'  TW'
      print'(a,e12.6,a)',' dE_niw/dT      ',mean_dedt(7)*fxa,'  TW'
      print'(a,e12.6,a)',' dissipation    ',mean_diss(7)*fxa,'  TW'
      print'(a,e12.6,a)',' error          ',(mean_dedt(7)-mean_forc(7)+mean_diss(7))*fxa,'  TW'
      print*,' '
    endif

    if (enable_tke) then
      print'(a)',' TKE budget:'
      print'(a,e12.6,a)',' external forc. ',mean_forc(5)*fxa,'  TW'
      print'(a,e12.6,a)',' dTKE/dT        ',mean_dedt(5)*fxa,'  TW'
      print'(a,e12.6,a)',' dissipation    ',mean_diss(5)*fxa,'  TW'
      print'(a,e12.6,a)',' EKE -> TKE     ',mean_exchg(7)*fxa,'  TW'
      print'(a,e12.6,a)',' TKE -> Hd      ',mean_exchg(5)*fxa,'  TW'
      print'(a,e12.6,a)',' KE  -> TKE     ',mean_exchg(4)*fxa,'  TW'
      print'(a,e12.6,a)',' IW  -> TKE     ',mean_diss(4)*fxa,'  TW'
      fxb=mean_dedt(5)-mean_forc(5)+mean_diss(5)- mean_diss(4)-mean_exchg(4)-mean_exchg(5) -mean_exchg(7)
      print'(a,e12.6,a)',' error          ',fxb*fxa,'  TW'
      print*,' '
    endif

    if (enable_store_cabbeling_heat) then
      print'(a)',' Heat budget (partial):'
      print'(a,e12.6,a)',' Hd->Ho (vert) ',-mean_misc(4)*fxa,'  TW'
      print'(a,e12.6,a)',' Hd->Ho (lat)  ',-mean_misc(9)*fxa,'  TW'
      print'(a,e12.6,a)',' TKE->Ho       ',mean_diss(5)*fxa,'  TW'
      print*,' '
    endif

    print'(a,e12.6,a)',' total forcing  ',sum( mean_forc(1:50)) *fxa,'  TW'
    print'(a,e12.6,a)',' total change   ',sum( mean_dedt(1:50)) *fxa,'  TW'

    fxb = sum( mean_dedt(1:50)) - sum( mean_forc(1:50)) + mean_diss(5)
    if (enable_store_cabbeling_heat) fxb = fxb - mean_misc(4) - mean_misc(9)
    print'(a,e12.6,a)',' error          ',fxb*fxa,'  TW'
    print*,' '

    print'(a,a)',' writing energy diagnostics to file ',diag_energy_file(1:len_trim(diag_energy_file))
    iret=nf_open(diag_energy_file,NF_WRITE,ncid)
    iret=nf_set_fill(ncid, NF_NOFILL, iret)
    iret=nf_inq_dimid(ncid,'Time',timedim)
    iret=nf_inq_dimlen(ncid, timedim,i)
    i=i+1
    iret=nf_inq_varid(ncid,'Time',id)
    fxa = itt*dt_tracer/86400.0
    iret= nf_put_vara_double(ncid,id,i,1,fxa)
    
    iret=nf_inq_varid(ncid,'KE',id);   iret= nf_put_vara_double(ncid,id,i,1,mean_e(1)*rho_0)
    iret=nf_inq_varid(ncid,'Hd',id);   iret= nf_put_vara_double(ncid,id,i,1,mean_e(2)*rho_0)
    iret=nf_inq_varid(ncid,'EKE',id);  iret= nf_put_vara_double(ncid,id,i,1,mean_e(3)*rho_0)
    iret=nf_inq_varid(ncid,'E_iw',id); iret= nf_put_vara_double(ncid,id,i,1,mean_e(4)*rho_0)
    iret=nf_inq_varid(ncid,'TKE',id);  iret= nf_put_vara_double(ncid,id,i,1,mean_e(5)*rho_0)

    iret=nf_inq_varid(ncid,'dE_tot',id);iret= nf_put_vara_double(ncid,id,i,1,sum(mean_dedt(1:5))*rho_0)
    iret=nf_inq_varid(ncid,'dKE',id);   iret= nf_put_vara_double(ncid,id,i,1,mean_dedt(1)*rho_0)
    iret=nf_inq_varid(ncid,'dHd',id);   iret= nf_put_vara_double(ncid,id,i,1,mean_dedt(2)*rho_0)
    iret=nf_inq_varid(ncid,'dEKE',id);  iret= nf_put_vara_double(ncid,id,i,1,mean_dedt(3)*rho_0)
    iret=nf_inq_varid(ncid,'dE_iw',id); iret= nf_put_vara_double(ncid,id,i,1,mean_dedt(4)*rho_0)
    iret=nf_inq_varid(ncid,'dTKE',id);  iret= nf_put_vara_double(ncid,id,i,1,mean_dedt(5)*rho_0)

    iret=nf_inq_varid(ncid,'KE_diss' ,id);   iret= nf_put_vara_double(ncid,id,i,1,mean_diss(1)*rho_0)
    iret=nf_inq_varid(ncid,'Hd_diss' ,id);   iret= nf_put_vara_double(ncid,id,i,1,mean_diss(2)*rho_0)
    iret=nf_inq_varid(ncid,'EKE_diss' ,id);  iret= nf_put_vara_double(ncid,id,i,1,mean_diss(3)*rho_0)
    iret=nf_inq_varid(ncid,'E_iw_diss' ,id); iret= nf_put_vara_double(ncid,id,i,1,mean_diss(4)*rho_0)
    iret=nf_inq_varid(ncid,'TKE_diss' ,id);  iret= nf_put_vara_double(ncid,id,i,1,mean_diss(5)*rho_0)

    iret=nf_inq_varid(ncid,'Wind',id);        iret= nf_put_vara_double(ncid,id,i,1,mean_forc(1)*rho_0)
    iret=nf_inq_varid(ncid,'dHd_sources',id); iret= nf_put_vara_double(ncid,id,i,1,mean_forc(2)*rho_0)
    iret=nf_inq_varid(ncid,'E_iw_forc' ,id);  iret= nf_put_vara_double(ncid,id,i,1,mean_forc(4)*rho_0)
    iret=nf_inq_varid(ncid,'TKE_forc' ,id);   iret= nf_put_vara_double(ncid,id,i,1,mean_forc(5)*rho_0)

    iret=nf_inq_varid(ncid,'KE_Hd' ,id);   iret= nf_put_vara_double(ncid,id,i,1,mean_exchg(1)*rho_0)
    iret=nf_inq_varid(ncid,'KE_EKE' ,id);  iret= nf_put_vara_double(ncid,id,i,1,mean_exchg(2)*rho_0)
    iret=nf_inq_varid(ncid,'Hd_EKE' ,id);  iret= nf_put_vara_double(ncid,id,i,1,mean_exchg(3)*rho_0)
    iret=nf_inq_varid(ncid,'KE_TKE' ,id);  iret= nf_put_vara_double(ncid,id,i,1,mean_exchg(4)*rho_0)
    iret=nf_inq_varid(ncid,'TKE_Hd' ,id);  iret= nf_put_vara_double(ncid,id,i,1,mean_exchg(5)*rho_0)
    iret=nf_inq_varid(ncid,'KE_IW' ,id);   iret= nf_put_vara_double(ncid,id,i,1,mean_exchg(6)*rho_0)
    iret=nf_inq_varid(ncid,'EKE_TKE' ,id); iret= nf_put_vara_double(ncid,id,i,1,mean_exchg(7)*rho_0)
    iret=nf_inq_varid(ncid,'EKE_IW' ,id);  iret= nf_put_vara_double(ncid,id,i,1,mean_exchg(8)*rho_0)

    iret=nf_inq_varid(ncid,'cabb' ,id);    iret= nf_put_vara_double(ncid,id,i,1,mean_misc(4)*rho_0)
    iret=nf_inq_varid(ncid,'cabb_iso' ,id);iret= nf_put_vara_double(ncid,id,i,1,mean_misc(9)*rho_0)
    iret=nf_inq_varid(ncid,'adv'  ,id);    iret= nf_put_vara_double(ncid,id,i,1,mean_misc(5)*rho_0)

    call ncclos (ncid, iret)
 endif

 nitts = 0
 mean_e    = 0
 mean_dedt = 0
 mean_diss = 0
 mean_forc = 0
 mean_exchg = 0
 mean_misc = 0

end subroutine write_energy



subroutine init_diag_energy
!=======================================================================
!     initialize NetCDF snapshot file
!=======================================================================
 use main_module   
 use diagnostics_module
 use diag_energy_module
 implicit none
 include "netcdf.inc"
 integer :: ncid,iret
 integer :: timedim,timeid,id
 character :: name*80, unit*80
 real*8, parameter :: spval = -1.0d33


 nitts = 0
 mean_e    = 0
 mean_dedt = 0
 mean_diss = 0
 mean_forc = 0
 mean_exchg = 0
 mean_misc = 0

 if (my_pe==0) then

      print'(a,a)',' preparing file ',diag_energy_file(1:len_trim(diag_energy_file))
      iret = nf_create (diag_energy_file, nf_clobber, ncid)
      if (iret /= 0) call halt_stop('NETCDF:'//nf_strerror(iret))
      iret=nf_set_fill(ncid, NF_NOFILL, iret)
      Timedim  = ncddef(ncid, 'Time', nf_unlimited, iret)
      timeid  = ncvdef (ncid,'Time', NCFLOAT,1,timedim,iret)
      name = 'Time '; unit = 'days'
      call ncaptc(ncid, timeid, 'long_name', NCCHAR, 24, name, iret) 
      call ncaptc(ncid, timeid, 'units',     NCCHAR, 16, unit, iret) 
      call ncaptc(ncid, Timeid,'time_origin',NCCHAR, 20,'01-JAN-1900 00:00:00', iret)

      ! Energies
      id  = ncvdef (ncid,'KE', NCFLOAT,1,timedim,iret)
      name = 'Mean kinetic energy'; unit = 'J'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'Hd', NCFLOAT,1,timedim,iret)
      name = 'Mean dynamic enthalpy'; unit = 'J'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'EKE', NCFLOAT,1,timedim,iret)
      name = 'Meso-scale eddy energy'; unit = 'J'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'E_iw', NCFLOAT,1,timedim,iret)
      name = 'Internal wave energy'; unit = 'J'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'TKE', NCFLOAT,1,timedim,iret)
      name = 'Turbulent kinetic energy'; unit = 'J'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      ! changes in energies
      id  = ncvdef (ncid,'dE_tot', NCFLOAT,1,timedim,iret)
      name = 'Change of total energy'; unit = 'W'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'dKE', NCFLOAT,1,timedim,iret)
      name = 'Change of KE'; unit = 'W'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'dHd', NCFLOAT,1,timedim,iret)
      name = 'Change of dHd'; unit = 'W'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'dEKE', NCFLOAT,1,timedim,iret)
      name = 'Change of EKE'; unit = 'W'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'dE_iw', NCFLOAT,1,timedim,iret)
      name = 'Change of E_iw'; unit = 'W'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'dTKE', NCFLOAT,1,timedim,iret)
      name = 'Change of TKE'; unit = 'W'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      ! Dissipation of energies
      id  = ncvdef (ncid,'KE_diss', NCFLOAT,1,timedim,iret)
      name = 'Dissipation of KE'; unit = 'W'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'Hd_diss', NCFLOAT,1,timedim,iret)
      name = 'Dissipation of Hd'; unit = 'W'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'EKE_diss', NCFLOAT,1,timedim,iret)
      name = 'Dissipation of EKE'; unit = 'W'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'E_iw_diss', NCFLOAT,1,timedim,iret)
      name = 'Dissipation of E_iw'; unit = 'W'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'TKE_diss', NCFLOAT,1,timedim,iret)
      name = 'Dissipation of TKE'; unit = 'W'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      ! forcing of energies
      id  = ncvdef (ncid,'Wind', NCFLOAT,1,timedim,iret)
      name = 'Wind work'; unit = 'W'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'dHd_sources', NCFLOAT,1,timedim,iret)
      name = 'Hd production by ext. sources'; unit = 'W'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'E_iw_forc', NCFLOAT,1,timedim,iret)
      name = 'External forcing of E_iw'; unit = 'W'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'TKE_forc', NCFLOAT,1,timedim,iret)
      name = 'External forcing of TKE'; unit = 'W'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'KE_Hd', NCFLOAT,1,timedim,iret)
      name = 'Exchange KE -> Hd'; unit = 'W'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'KE_TKE', NCFLOAT,1,timedim,iret)
      name = 'Exchange KE -> TKE by vertical friction'; unit = 'W'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'KE_IW', NCFLOAT,1,timedim,iret)
      name = 'Exchange KE -> IW by bottom friction'; unit = 'W'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'TKE_Hd', NCFLOAT,1,timedim,iret)
      name = 'Exchange TKE -> Hd by vertical mixing'; unit = 'W'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'KE_EKE', NCFLOAT,1,timedim,iret)
      name = 'Exchange KE -> EKE by lateral friction'; unit = 'W'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'Hd_EKE', NCFLOAT,1,timedim,iret)
      name = 'Exchange Hd -> EKE by GM and lateral mixing'; unit = 'W'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'EKE_TKE', NCFLOAT,1,timedim,iret)
      name = 'Exchange EKE -> TKE'; unit = 'W'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'EKE_IW', NCFLOAT,1,timedim,iret)
      name = 'Exchange EKE -> IW'; unit = 'W'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'cabb', NCFLOAT,1,timedim,iret)
      name = 'Cabbeling by vertical mixing'; unit = 'W'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'cabb_iso', NCFLOAT,1,timedim,iret)
      name = 'Cabbeling by isopycnal mixing'; unit = 'W'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

      id  = ncvdef (ncid,'adv', NCFLOAT,1,timedim,iret)
      name = 'Dissipation by advection'; unit = 'W'
      call dvcdf(ncid,id,name(1:len_trim(name)),len_trim(name),unit(1:len_trim(unit)),len_trim(unit),spval)

     call ncclos (ncid, iret)
 endif
 call fortran_barrier
end subroutine







subroutine diag_energy_read_restart
!=======================================================================
! read unfinished averages from file
!=======================================================================
 use main_module
 use diag_energy_module
 implicit none
 character (len=80) :: filename
 logical :: file_exists
 integer :: io,ierr

 filename= 'unfinished_energy.dta'
 inquire ( FILE=filename, EXIST=file_exists )
 if (.not. file_exists) then
      if (my_pe==0) then
         print'(a,a,a)',' file ',filename(1:len_trim(filename)),' not present'
         print'(a)',' reading no unfinished energy diagnostics'
      endif
      return
 endif

 if (my_pe==0) print'(2a)',' reading unfinished averages from ',filename(1:len_trim(filename))
 call get_free_iounit(io,ierr)
 if (ierr/=0) goto 10
 open(io,file=filename,form='unformatted',status='old',err=10)
 read(io,err=10) nitts
 read(io,err=10) mean_e,mean_dedt,mean_diss,mean_forc,mean_exchg,mean_misc
 close(io)
 return
 10 continue
 print'(a)',' Warning: error reading file'
end subroutine diag_energy_read_restart




subroutine diag_energy_write_restart
!=======================================================================
! write unfinished averages to restart file
!=======================================================================
 use main_module
 use diag_energy_module
 implicit none
 character (len=80) :: filename
 integer :: io,ierr

 filename= 'unfinished_energy.dta'
 if (my_pe==0) print'(a,a)',' writing unfinished averages to ',filename(1:len_trim(filename))
 call get_free_iounit(io,ierr)
 if (ierr/=0) goto 10
 open(io,file=filename,form='unformatted',status='unknown')
 write(io,err=10) nitts
 write(io,err=10) mean_e,mean_dedt,mean_diss,mean_forc,mean_exchg,mean_misc
 close(io)
 return
 10 continue
 print'(a)',' Warning: error writing file'
end subroutine diag_energy_write_restart



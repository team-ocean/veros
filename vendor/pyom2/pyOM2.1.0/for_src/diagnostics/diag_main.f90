



subroutine init_diagnostics
!=======================================================================
! initialize diagnostic routines
!=======================================================================
 use main_module   
 use diagnostics_module   
 implicit none

 if (my_pe==0) print'(/,a)','Diagnostic setup:'

 if (enable_diag_ts_monitor) then
   if (my_pe==0) print'(a,e12.6,a,f10.2,a)',' time step monitor every ',ts_monint,' seconds/',ts_monint/dt_tracer,' time steps'
 endif

 if (enable_diag_tracer_content) then
   if (my_pe==0) print'(a,e12.6,a,f10.2,a)',' monitor tracer content every ',&
                  trac_cont_int,' seconds/',trac_cont_int/dt_tracer,' time steps'
 endif

 if (enable_diag_snapshots) then
    if (my_pe==0) print'(a,e12.6,a,f10.2,a)',' writing snapshots every ',snapint,' seconds/',snapint/dt_tracer,' time steps'
    call init_snap_cdf
 endif

 if (enable_diag_averages)  then
    if (my_pe==0) print'(a,e12.6,a,f10.2,a)',' writing time averages every ',aveint,' seconds/',aveint/dt_tracer,' time steps'
    if (my_pe==0) print'(a,f10.2,a)',' averaging every ',avefreq/dt_tracer,' time step'
 endif

 if (enable_diag_energy) then
    if (my_pe==0) print'(a,e12.6,a,f10.2,a)',' writing energetics every ',energint,' seconds/',energint/dt_tracer,' time steps'
    if (my_pe==0) print'(a,f10.2,a)',' diagnosing every ',energfreq/dt_tracer,' time step'
    call init_diag_energy
 endif

 if (enable_diag_overturning) then
    if (my_pe==0) print'(a,e12.6,a,f10.2,a)',' writing isopyc. overturning every ', &
                overint,' seconds/',overint/dt_tracer,' time steps'
    if (my_pe==0) print'(a,f10.2,a)',' diagnosing every ',overfreq/dt_tracer,' time step'
    call init_diag_overturning
 endif

 if (enable_diag_particles) then
    if (my_pe==0) print'(a,e12.6,a,f10.2,a)',' writing particles every ', &
        particles_int,' seconds/',particles_int/dt_tracer,' time steps'
    call set_particles()
    call init_diag_particles
    call init_write_particles
 endif
end subroutine init_diagnostics




subroutine diagnose
!=======================================================================
! call diagnostic routines
!=======================================================================
 use main_module   
 use diagnostics_module   
 use isoneutral_module   
 implicit none
 logical :: GM_strfct_diagnosed
 real*8 :: time


 GM_strfct_diagnosed = .false.
 time = itt*dt_tracer

 if ( enable_diag_ts_monitor .and.  modulo(time,ts_monint) < dt_tracer ) then
   if (my_pe==0) print'(a,i10.10,a,e10.4,a,i6)', ' itt=',itt,' time=',time,'s congr.itts=',congr_itts
   if (my_pe==0.and. .not.enable_hydrostatic) print'(a,i6)', ' congr. non hydro itts=',congr_itts_non_hydro
   call diag_cfl
 endif

 if ( enable_diag_tracer_content .and.  modulo(time,trac_cont_int) < dt_tracer ) then
   call diag_tracer_content
 endif


 if ( enable_diag_energy   .and.  modulo(time,energfreq) < dt_tracer  )  call diagnose_energy
 if ( enable_diag_energy   .and.  modulo(time,energint)  < dt_tracer  )  call write_energy

 if ( enable_diag_averages .and.  modulo(time,avefreq) < dt_tracer  ) then
    if (enable_neutral_diffusion .and. enable_skew_diffusion .and. .not.GM_strfct_diagnosed) then
      call isoneutral_diag_streamfunction
      GM_strfct_diagnosed = .true.
    endif
    call diag_averages
 endif

 if ( enable_diag_averages .and.  modulo(time,aveint) < dt_tracer )  call write_averages

 if ( enable_diag_snapshots.and.  modulo(time,snapint) < dt_tracer  )   then
    if (enable_neutral_diffusion .and. enable_skew_diffusion .and. .not.GM_strfct_diagnosed) then
      call isoneutral_diag_streamfunction
      GM_strfct_diagnosed = .true.
    endif
    call diag_snap
 endif

 if ( enable_diag_overturning .and.  modulo(time,overfreq) < dt_tracer )  then
    if (enable_neutral_diffusion .and. enable_skew_diffusion .and. .not.GM_strfct_diagnosed) then
      call isoneutral_diag_streamfunction
      GM_strfct_diagnosed = .true.
    endif
    call diag_overturning
 endif

 if ( enable_diag_overturning .and.  modulo(time,overint) < dt_tracer )  call write_overturning


 if ( enable_diag_particles)   then
    call integrate_particles
    if ( modulo(time,particles_int) < dt_tracer  )   call write_particles
 endif



end subroutine diagnose




subroutine diag_cfl
!=======================================================================
! check for CFL violation
!=======================================================================
 use main_module   
 use tke_module   
 use eke_module   
 use idemix_module   
 use diagnostics_module   
 implicit none
 integer :: i,j,k
 real*8 :: cfl,wcfl

 cfl = 0d0; wcfl=0d0
 do k=1,nz
  do j=js_pe,je_pe
   do i=is_pe,ie_pe
     cfl = max(cfl,  abs(u(i,j,k,tau))*maskU(i,j,k)/(cost(j)*dxt(i))*dt_tracer )
     cfl = max(cfl,  abs(v(i,j,k,tau))*maskV(i,j,k)/dyt(j)*dt_tracer )
     wcfl = max(wcfl,  abs(w(i,j,k,tau))*maskW(i,j,k)/dzt(k)*dt_tracer )
   enddo
  enddo
 enddo
 call global_max(cfl); call global_max(wcfl)
 !if (cfl > 0.5.or.wcfl > 0.5) then
 !  if (my_pe==0) print'(/a,f12.6)','ERROR:  maximal CFL number = ',max(cfl,wcfl)
 !  if (my_pe==0) print'(a,i9,a/)' ,' at itt = ',itt,' ... stopping integration '
 !  if (.not. enable_diag_snapshots )   call init_snap_cdf
 !  call diag_snap
 !  call halt_stop(' in diag_cfl')
 !endif
 ! check for NaN
 if (cfl/=cfl .or. wcfl /=wcfl) then
   if (my_pe==0) print'(/a)','ERROR:   CFL number is NaN '
   if (my_pe==0) print'(a,i9,a/)' ,' at itt = ',itt,' ... stopping integration '
   if (.not. enable_diag_snapshots )   call init_snap_cdf
   call diag_snap
   call halt_stop(' in diag_cfl')
 endif
 if (my_pe==0) print'(a,f12.6)', ' maximal hor. CFL number =',cfl
 if (my_pe==0) print'(a,f12.6)', ' maximal ver. CFL number =',wcfl

 if (enable_eke .or. enable_tke .or. enable_idemix) then
  cfl = 0d0; wcfl=0d0
  do k=1,nz
   do j=js_pe,je_pe
    do i=is_pe,ie_pe
      cfl = max(cfl,  abs(u_wgrid(i,j,k))*maskU(i,j,k)/(cost(j)*dxt(i))*dt_tracer )
      cfl = max(cfl,  abs(v_wgrid(i,j,k))*maskV(i,j,k)/dyt(j)*dt_tracer )
      wcfl = max(wcfl,  abs(w_wgrid(i,j,k))*maskW(i,j,k)/dzt(k)*dt_tracer )
    enddo
   enddo
  enddo
  call global_max(cfl); call global_max(wcfl)
  if (my_pe==0) print'(a,f12.6)', ' maximal hor. CFL number on w grid=',cfl
  if (my_pe==0) print'(a,f12.6)', ' maximal ver. CFL number on w grid=',wcfl
 endif

end subroutine diag_cfl






subroutine diag_tracer_content
!=======================================================================
! Diagnose tracer content
!=======================================================================
 use main_module   
 implicit none
 integer :: i,j,k
 real*8 :: fxa,tempm,saltm,volm,vtemp,vsalt
 real*8, save :: tempm1=0.,saltm1=0.,vtemp1=0.,vsalt1=0.

  volm=0;tempm=0;vtemp=0;saltm=0;vsalt=0
  do k=1,nz 
   do j=js_pe,je_pe
    do i=is_pe,ie_pe
     fxa = area_t(i,j)*dzt(k)*maskT(i,j,k)
     volm = volm + fxa
     tempm = tempm + fxa*temp(i,j,k,tau)
     saltm = saltm + fxa*salt(i,j,k,tau)
     vtemp = vtemp + temp(i,j,k,tau)**2*fxa
     vsalt = vsalt + salt(i,j,k,tau)**2*fxa
    enddo
   enddo
  enddo
  call global_sum(tempm); call global_sum(saltm); call global_sum(volm);  
  call global_sum(vtemp); call global_sum(vsalt);   

  if (my_pe==0) then
     print*,'  '
     print'(a,f20.15,a,e20.14)',' mean temperature ',tempm /volm,' change to last ',(tempm-tempm1)/volm
     print'(a,f20.15,a,e20.14)',' mean salinity    ',saltm /volm,' change to last ',(saltm-saltm1)/volm
     print'(a,e20.14,a,e20.14)',' temperature var. ',vtemp /volm,' change to last ',(vtemp-vtemp1)/volm
     print'(a,e20.14,a,e20.14)',' salinity var.    ',vsalt /volm,' change to last ',(vsalt-vsalt1)/volm
  endif

  tempm1=tempm; vtemp1=vtemp; saltm1=saltm; vsalt1=vsalt
end subroutine diag_tracer_content



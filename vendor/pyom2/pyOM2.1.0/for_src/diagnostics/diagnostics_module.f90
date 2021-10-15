

module diagnostics_module
      implicit none
!---------------------------------------------------------------------------------
!     diagnostic options
!---------------------------------------------------------------------------------
      logical :: enable_diag_ts_monitor    = .false. ! enable time step monitor
      logical :: enable_diag_energy        = .false. ! enable diagnostics for energy
      logical :: enable_diag_averages      = .false. ! enable time averages
      logical :: enable_diag_snapshots     = .false. ! enable snapshots
      logical :: enable_diag_overturning   = .false. ! enable isopycnal overturning diagnostic
      logical :: enable_diag_tracer_content= .false. ! enable tracer content and variance monitor
      logical :: enable_diag_particles     = .false. ! enable integration of particles
      character*80 :: snap_file   = 'pyOM.cdf'
      character*80 :: diag_energy_file   = 'energy.cdf'
      real*8  :: snapint=0.  ! intervall between snapshots to be written in seconds
      real*8  :: aveint=0.   ! intervall between time averages to be written in seconds
      real*8  :: energint=0. ! intervall between energy diag to be written in seconds
      real*8  :: energfreq=0.! diagnosing every energfreq seconds 
      real*8  :: ts_monint=0.! intervall between time step monitor in seconds
      real*8  :: avefreq=0.  ! averaging every ave_freq seconds 
      real*8  :: overint=0.  ! intervall between overturning averages to be written in seconds
      real*8  :: overfreq=0. ! averaging overturning every ave_freq seconds 
      real*8  :: trac_cont_int=0.! intervall between tracer content monitor in seconds
      real*8  :: particles_int=0. ! intervall 
end module diagnostics_module






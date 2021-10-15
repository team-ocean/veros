

module eke_module
     implicit none
!---------------------------------------------------------------------------------
!     EKE model
!---------------------------------------------------------------------------------
      logical :: enable_eke = .false.
      real*8, allocatable :: deke(:,:,:,:) ! tendency due to advection using Adam Bashforth
      real*8, allocatable :: eke(:,:,:,:)    ! meso-scale energy in m^2/s^2
      real*8, allocatable :: sqrteke(:,:,:)  ! square root of EKE
      real*8, allocatable :: L_rossby(:,:)   ! Rossby radius
      real*8, allocatable :: eke_len(:,:,:)  ! eddy length scale
      real*8, allocatable :: eke_diss_iw(:,:,:) ! Dissipation of EKE to internal waves
      real*8, allocatable :: eke_diss_tke(:,:,:) ! Dissipation of EKE to TKE
      real*8, allocatable :: L_rhines(:,:,:) ! Rhines scale
      real*8 :: eke_lmin  = 100.0  ! minimal length scale in m
      real*8 :: eke_c_k   = 1.0
      real*8 :: eke_cross = 1.0 ! Parameter for EKE model
      real*8 :: eke_crhin = 1.0 ! Parameter for EKE model
      real*8 :: eke_c_eps = 1.0 ! Parameter for EKE model
      real*8 :: eke_k_max = 1d4 ! maximum of K_gm
      real*8 :: alpha_eke = 1.0 ! factor for vertical friction
      logical :: enable_eke_superbee_advection   = .false.
      logical :: enable_eke_upwind_advection     = .false.
      logical :: enable_eke_isopycnal_diffusion  = .false. ! use K_gm also for isopycnal diffusivity

      real*8, allocatable :: eke_topo_hrms(:,:), eke_topo_lam(:,:), hrms_k0(:,:)
      real*8, allocatable :: c_lee(:,:)
      real*8, allocatable :: c_Ri_diss(:,:,:)
      real*8, allocatable :: eke_lee_flux(:,:),eke_bot_flux(:,:)
      logical :: enable_eke_leewave_dissipation  = .false.  
      real*8 :: c_lee0  = 1.0
      real*8 :: eke_Ri0 = 200.0
      real*8 :: eke_Ri1 = 50.0
      real*8 :: eke_int_diss0 = 1./(20*86400.)
      real*8 :: kappa_EKE0 = 0.1
      real*8 :: eke_r_bot = 0.0 ! bottom friction coefficient
      real*8 :: eke_hrms_k0_min = 0.0 ! minmal value for bottom roughness parameter

end module eke_module

subroutine allocate_eke_module
  use main_module
  use eke_module
  if (enable_eke) then
   allocate(deke(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3) );deke = 0.0
   allocate( eke(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3) ); eke = 0.0
   allocate( sqrteke(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); sqrteke = 0
   allocate( L_rossby(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); L_rossby = 0.0
   allocate( eke_len(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); eke_len = 0.0
   allocate( eke_diss_iw(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); eke_diss_iw = 0.0
   allocate( eke_diss_tke(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); eke_diss_tke = 0.0
   allocate( L_rhines(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); L_rhines = 0.0
   allocate( eke_bot_flux(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); eke_bot_flux = 0.0
   if (enable_eke_leewave_dissipation ) then
      allocate( eke_topo_hrms(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); eke_topo_hrms = 0.0
      allocate( eke_topo_lam(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); eke_topo_lam = 0.0
      allocate( hrms_k0(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); hrms_k0 = 0.0
      allocate( c_lee(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); c_lee = 0.0
      allocate( eke_lee_flux(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); eke_lee_flux = 0.0
      allocate( c_Ri_diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); c_Ri_diss = 0.0
   endif
  endif
end subroutine allocate_eke_module







module tke_module
      implicit none
!---------------------------------------------------------------------------------
!     TKE model
!---------------------------------------------------------------------------------
      logical :: enable_tke = .false.
      real*8, allocatable :: dtke(:,:,:,:) ! tendency due to advection using Adam Bashforth
      real*8, allocatable :: tke(:,:,:,:)          ! small-scale tke
      real*8, allocatable :: mxl(:,:,:)            ! eddy length scale
      real*8, allocatable :: sqrttke(:,:,:)        ! square root of TKE
      real*8, allocatable :: Prandtlnumber(:,:,:)          
      real*8, allocatable :: forc_tke_surface(:,:)          
      real*8, allocatable :: tke_surf_corr(:,:)          
      real*8, allocatable :: tke_diss(:,:,:)          
      real*8              :: c_k   = 0.1
      real*8              :: c_eps = 0.7
      real*8              :: alpha_tke = 1.0
      real*8              :: mxl_min = 1d-12
      real*8              :: kappaM_min = 0.d0
      real*8              :: kappaM_max = 100.d0
      integer             :: tke_mxl_choice = 1

      logical :: enable_tke_superbee_advection   = .false.
      logical :: enable_tke_upwind_advection     = .false.
      logical :: enable_tke_hor_diffusion = .false.
      real*8  :: K_h_tke = 2000.0     ! lateral diffusivity for tke
end module tke_module


subroutine allocate_tke_module
  use main_module
  use tke_module

  if (enable_tke) then
   allocate(dtke(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3) );dtke = 0.0
   allocate( tke(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3) ); tke = 0.0
   allocate( mxl(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); mxl = 0
   allocate( sqrttke(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); sqrttke = 0
   allocate( Prandtlnumber(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); Prandtlnumber = 0
   allocate( forc_tke_surface(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); forc_tke_surface = 0
   allocate( tke_diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); tke_diss = 0.0
   allocate( tke_surf_corr(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); tke_surf_corr = 0
  endif

end subroutine allocate_tke_module

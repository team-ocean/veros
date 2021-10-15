


module isoneutral_module
      implicit none
!---------------------------------------------------------------------------------
!     isopycnal mixing option
!---------------------------------------------------------------------------------
      logical :: enable_neutral_diffusion  = .false. ! enable isopycnal mixing
      logical :: enable_skew_diffusion     = .false. ! enable skew diffusion approach for eddy-driven velocities
      logical :: enable_TEM_friction       = .false. ! TEM approach for eddy-driven velocities
      real*8, allocatable :: K_11(:,:,:)         ! isopycnal mixing tensor component
      real*8, allocatable :: K_13(:,:,:)         ! isopycnal mixing tensor component
      real*8, allocatable :: K_22(:,:,:)         ! isopycnal mixing tensor component
      real*8, allocatable :: K_23(:,:,:)         ! isopycnal mixing tensor component
      real*8, allocatable :: K_31(:,:,:)         ! isopycnal mixing tensor component
      real*8, allocatable :: K_32(:,:,:)         ! isopycnal mixing tensor component
      real*8, allocatable :: K_33(:,:,:)         ! isopycnal mixing tensor component
      real*8, allocatable :: Ai_ez(:,:,:,:,:)    ! 
      real*8, allocatable :: Ai_nz(:,:,:,:,:)    ! 
      real*8, allocatable :: Ai_bx(:,:,:,:,:)    ! 
      real*8, allocatable :: Ai_by(:,:,:,:,:)    ! 
      real*8, allocatable :: B1_gm(:,:,:)    ! zonal streamfunction (for diagnostic purpose only)
      real*8, allocatable :: B2_gm(:,:,:)    ! meridional streamfunction (for diagnostic purpose only)
      real*8, allocatable :: K_gm(:,:,:)     ! GM diffusivity in m^2/s, either constant or from EKE model
      real*8, allocatable :: kappa_gm(:,:,:) ! vertical viscosity due to skew diffusivity K_gm in m^2/s
      real*8, allocatable :: K_iso(:,:,:)    ! along isopycnal diffusivity in m^2/s
      real*8 :: K_iso_0     = 0.0            ! constant for isopycnal diffusivity in m^2/s
      real*8 :: K_iso_steep = 0.0            ! lateral diffusivity for steep slopes in m^2/s
      real*8 :: K_gm_0      = 0.0            ! fixed value for K_gm which is set for no EKE model
      real*8 :: iso_dslope=0.0008            ! parameters controlling max allowed isopycnal slopes
      real*8 :: iso_slopec=0.001             ! parameters controlling max allowed isopycnal slopes

end module isoneutral_module




subroutine allocate_isoneutral_module
 use main_module
 use isoneutral_module
 implicit none

 if (enable_neutral_diffusion) then
  allocate( K_11(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); K_11 = 0
  allocate( K_13(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); K_13 = 0
  allocate( K_22(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); K_22 = 0
  allocate( K_23(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); K_23 = 0
  allocate( K_31(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); K_31 = 0
  allocate( K_32(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); K_32 = 0
  allocate( K_33(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); K_33 = 0
  allocate( Ai_ez(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,0:1,0:1) ); Ai_ez = 0
  allocate( Ai_nz(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,0:1,0:1) ); Ai_nz = 0
  allocate( Ai_bx(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,0:1,0:1) ); Ai_bx = 0
  allocate( Ai_by(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,0:1,0:1) ); Ai_by = 0
 endif

 allocate( B1_gm(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); B1_gm = 0
 allocate( B2_gm(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); B2_gm = 0

 allocate( kappa_gm(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); kappa_gm = 0
 allocate( K_gm(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); K_gm = 0.0
 allocate( K_iso(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); K_iso = 0.0
end subroutine allocate_isoneutral_module

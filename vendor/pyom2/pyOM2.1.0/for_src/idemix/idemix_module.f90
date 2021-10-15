
module idemix_module   
!=======================================================================
! module containing all relevant arrays and parameter for IDEMIX
!=======================================================================
      implicit none

!---------------------------------------------------------------------------------
!     Idemix 1.0
!---------------------------------------------------------------------------------
      logical :: enable_idemix = .false.
      real*8, allocatable :: dE_iw(:,:,:,:) ! tendency due to advection using Adam Bashforth
      real*8, allocatable :: E_iw(:,:,:,:),c0(:,:,:),v0(:,:,:),alpha_c(:,:,:)
      real*8, allocatable :: forc_iw_bottom(:,:),forc_iw_surface(:,:),iw_diss(:,:,:)
      real*8 :: tau_v=1.0*86400.0   ! time scale for vertical symmetrisation
      real*8 :: tau_h=15.0*86400.0  ! time scale for horizontal symmetrisation
      real*8 :: gamma=1.57          ! 
      real*8 :: jstar = 10.0        ! spectral bandwidth in modes
      real*8 :: mu0   = 4.0/3.0     ! dissipation parameter
      logical :: enable_idemix_hor_diffusion = .false.
      logical :: enable_eke_diss_bottom = .false.
      logical :: enable_eke_diss_surfbot = .false.
      real*8  :: eke_diss_surfbot_frac = 1.0 ! fraction which goes into bottom
      logical :: enable_idemix_superbee_advection   = .false.
      logical :: enable_idemix_upwind_advection     = .false.

!---------------------------------------------------------------------------------
!     IDEMIX 2.0
!---------------------------------------------------------------------------------
      logical :: enable_idemix_M2     = .false.
      logical :: enable_idemix_niw    = .false.

      integer :: np = 0
      real*8, allocatable :: phit(:),phiu(:),dphit(:),dphiu(:)
      real*8, allocatable :: maskTp(:,:,:)
      real*8, allocatable :: maskUp(:,:,:)
      real*8, allocatable :: maskVp(:,:,:)
      real*8, allocatable :: maskWp(:,:,:)
      integer, allocatable, dimension(:,:,:)  :: bc_south   ! index for southern reflection boundary condition
      integer, allocatable, dimension(:,:,:)  :: bc_north   ! same for northern boundaries
      integer, allocatable, dimension(:,:,:)  :: bc_east    ! same for eastern boundaries
      integer, allocatable, dimension(:,:,:)  :: bc_west    ! same for western boundaries

      real*8, allocatable :: topo_hrms(:,:),topo_lam(:,:),topo_shelf(:,:)
      real*8, allocatable :: E_M2(:,:,:,:),dE_M2p(:,:,:,:)
      real*8, allocatable :: E_niw(:,:,:,:),dE_niwp(:,:,:,:)

      real*8, allocatable :: tau_niw(:,:),forc_niw(:,:,:)
      real*8, allocatable :: tau_M2(:,:),forc_M2(:,:,:)
      real*8, allocatable :: alpha_M2_cont(:,:)
      real*8, allocatable :: M2_psi_diss(:,:,:)  ! dissipation by PSI 

      real*8, allocatable :: cn(:,:),phin(:,:,:),phinz(:,:,:)
      real*8, allocatable :: omega_niw(:,:)
      real*8   :: omega_M2 !=  2*pi/( 12*60*60 +  25.2 *60 )   ! M2 frequency in 1/s
      real*8, allocatable :: cg_niw(:,:),cg_M2(:,:)
      real*8, allocatable :: kdot_y_M2(:,:),kdot_y_niw(:,:)
      real*8, allocatable :: kdot_x_M2(:,:),kdot_x_niw(:,:)
      real*8, allocatable :: u_M2(:,:,:),v_M2(:,:,:),w_M2(:,:,:)
      real*8, allocatable :: u_niw(:,:,:),v_niw(:,:,:),w_niw(:,:,:)
      real*8, allocatable :: E_struct_niw(:,:,:),E_struct_M2(:,:,:)
      real*8, allocatable :: E_niw_int(:,:),E_M2_int(:,:)
end module idemix_module   



subroutine allocate_idemix_module
  use main_module
  use idemix_module
  implicit none

  if (enable_idemix) then
   allocate(dE_iw(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3) );dE_iw = 0
   allocate( E_iw(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3) ); E_iw = 0
   allocate( c0(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); c0 = 0
   allocate( v0(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); v0 = 0
   allocate( alpha_c(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); alpha_c = 0
   allocate( iw_diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); iw_diss = 0
   allocate( forc_iw_surface(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); forc_iw_surface = 0
   allocate( forc_iw_bottom(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); forc_iw_bottom = 0
  endif


  if (enable_idemix_M2 .or. enable_idemix_niw ) then

    allocate( topo_shelf(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); topo_shelf = 0
    allocate( topo_hrms(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); topo_hrms = 0
    allocate( topo_lam(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); topo_lam = 0
    allocate( phit(np), dphit(np), phiu(np), dphiu(np)) ; phit=0;dphit=0;phiu=0;dphiu=0
    allocate( maskTp(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np) ); maskTp = 0
    allocate( maskUp(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np) ); maskUp = 0
    allocate( maskVp(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np) ); maskVp = 0
    allocate( maskWp(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np) ); maskWp = 0
    allocate( cn(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); cn = 0.0
    allocate( phin(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); phin = 0.0
    allocate( phinz(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); phinz = 0.0
    allocate( tau_M2(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); tau_M2 = 0
    allocate( tau_niw(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); tau_niw = 0
    allocate( alpha_M2_cont(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); alpha_M2_cont = 0
    allocate( bc_south(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np) )
    allocate( bc_north(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np) )
    allocate( bc_west (is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np) )
    allocate( bc_east (is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np) )
    bc_south = 0; bc_north=0; bc_west=0; bc_east=0
    allocate( M2_psi_diss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np) ); M2_psi_diss = 0
  endif

  if (enable_idemix_M2) then
   allocate( E_M2(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np,3) ); E_M2=0
   allocate(dE_M2p(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np,3) );dE_M2p=0
   allocate( cg_M2(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); cg_M2 = 0.0; 
   allocate( kdot_x_M2(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); kdot_x_M2 = 0.0; 
   allocate( kdot_y_M2(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); kdot_y_M2 = 0.0; 
   allocate(  forc_M2(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np) ); forc_M2 = 0
   allocate( u_M2(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np) ); u_M2=0.
   allocate( v_M2(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np) ); v_M2=0.
   allocate( w_M2(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np) ); w_M2=0.
   allocate( E_struct_M2(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); E_struct_M2 = 0
   allocate( E_M2_int(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); E_M2_int = 0
  endif


  if (enable_idemix_niw) then
    allocate( omega_niw(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); omega_niw=0
    allocate( E_niw(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np,3) ); E_niw=0
    allocate(dE_niwp(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np,3) );dE_niwp=0
    allocate( cg_niw(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); cg_niw = 0.0; 
    allocate( kdot_x_niw(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); kdot_x_niw = 0.0; 
    allocate( kdot_y_niw(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); kdot_y_niw = 0.0; 
    allocate(  forc_niw(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np) ); forc_niw = 0
    allocate( u_niw(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np) ); u_niw=0.
    allocate( v_niw(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np) ); v_niw=0.
    allocate( w_niw(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,np) ); w_niw=0.
    allocate( E_struct_niw(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); E_struct_niw = 0
    allocate( E_niw_int(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); E_niw_int = 0
  endif




end subroutine allocate_idemix_module

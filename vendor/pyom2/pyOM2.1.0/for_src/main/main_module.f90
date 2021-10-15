

module main_module   
!=======================================================================
! main module containing most important arrays and parameter
! others can be found in specific modules
!=======================================================================
      implicit none
!---------------------------------------------------------------------------------
!     constants and parameter
!---------------------------------------------------------------------------------
      real*8, parameter :: version = 2.10
      real*8, parameter :: pi      = 3.14159265358979323846264338327950588
      real*8, parameter :: radius  = 6370.0e3        ! Earth radius in m
      real*8, parameter :: degtom  = radius/180.0*pi ! conversion degrees latitude to meters
      real*8, parameter :: mtodeg  = 1/degtom        ! revers conversion 
      real*8, parameter :: omega   = pi/43082.0      ! earth rotation frequency in 1/s
      real*8, parameter :: rho_0   = 1024.0          ! Boussinesq reference density in kg/m^3
      real*8, parameter :: grav    = 9.81            ! gravitational constant in m/s^2
!---------------------------------------------------------------------------------
!     Parallel domain setup
!---------------------------------------------------------------------------------
      integer :: n_pes     ! total number of processors
      integer :: my_pe     ! index of this processor from 0 to n_pes-1
      integer :: n_pes_i   ! total number of processors in x direction
      integer :: n_pes_j   ! total number of processors in y direction
      integer :: my_blk_i  ! index of this processor in x direction from 1 to n_pes_i
      integer :: my_blk_j  ! index of this processor in y direction from 1 to n_pes_j
      integer :: i_blk     ! grid points of domain decompostion in x direction 
      integer :: j_blk     ! grid points of domain decompostion in y direction
      integer :: is_pe     ! start index of grid points in x direction of this processor
      integer :: ie_pe     ! end index of grid points in x direction of this processor
      integer :: js_pe     ! start index of grid points in y direction of this processor
      integer :: je_pe     ! end index of grid points in y direction of this processor
      integer :: onx=2     ! number of overlapping points in x and y direction
      integer :: my_comm=0 ! communicator for MPI library
!---------------------------------------------------------------------------------
!     model parameter
!---------------------------------------------------------------------------------
      integer :: nx            ! grid points in zonal (x,i) direction
      integer :: ny            ! grid points in meridional (y,j) direction
      integer :: nz            ! grid points in vertical (z,k) direction
      integer :: taum1     = 1 ! pointer to last time step  
      integer :: tau       = 2 ! pointer to current time step
      integer :: taup1     = 3 ! pointer to next time step
      real*8  :: dt_mom    = 0 ! time step in seconds for momentum
      real*8  :: dt_tracer = 0 ! time step for tracer can be larger than for momentum
      real*8  :: dt_tke        ! should be time step for momentum (set in tke.f90)
      integer :: itt           ! time step number
      integer :: enditt        ! last time step of simulation
      real*8  :: runlen=0.     ! length of simulation in seconds
      real*8  :: AB_eps = 0.1  ! deviation from Adam-Bashforth weighting
!---------------------------------------------------------------------------------
!     logical switches for general model setup
!---------------------------------------------------------------------------------
      logical :: coord_degree                      = .false. ! either spherical (true) or cartesian (false) coordinates 
      logical :: enable_cyclic_x                   = .false. ! enable cyclic boundary conditions
      integer :: eq_of_state_type = 1                        ! equation of state: 1: linear, 3: nonlinear with comp., 5: TEOS
      logical :: enable_implicit_vert_friction     = .false. ! enable implicit vertical friction
      logical :: enable_explicit_vert_friction     = .false. ! enable explicit vertical friction
      logical :: enable_hor_friction               = .false. ! enable horizontal friction
      logical :: enable_hor_diffusion              = .false. ! enable horizontal diffusion
      logical :: enable_biharmonic_friction        = .false. ! enable biharmonic horizontal friction
      logical :: enable_biharmonic_mixing          = .false. ! enable biharmonic horizontal mixing
      logical :: enable_hor_friction_cos_scaling   = .false. ! scaling of hor. viscosity with cos(latitude)**cosPower
      logical :: enable_ray_friction               = .false. ! enable Rayleigh damping
      logical :: enable_bottom_friction            = .false. ! enable bottom friction
      logical :: enable_bottom_friction_var        = .false. ! enable bottom friction with lateral variations
      logical :: enable_quadratic_bottom_friction  = .false. ! enable quadratic bottom friction
      logical :: enable_tempsalt_sources           = .false. ! enable restoring zones, etc
      logical :: enable_momentum_sources           = .false. ! enable restoring zones, etc
      logical :: enable_superbee_advection         = .false. ! enable advection scheme with implicit mixing
      logical :: enable_conserve_energy            = .true.  ! exchange energy consistently
      logical :: enable_store_bottom_friction_tke  = .false. ! transfer dissipated energy by bottom/rayleig fric. to TKE
                                                             ! else transfer to internal waves
      logical :: enable_store_cabbeling_heat       = .false. ! transfer non-linear mixing terms to potential enthalpy
                                                             ! else transfer to TKE and EKE
!---------------------------------------------------------------------------------
!     variables related to numerical grid
!---------------------------------------------------------------------------------
      real*8, allocatable, dimension(:,:,:)   :: maskT     ! mask in physical space for tracer points
      real*8, allocatable, dimension(:,:,:)   :: maskU     ! mask in physical space for U points
      real*8, allocatable, dimension(:,:,:)   :: maskV     ! mask in physical space for V points
      real*8, allocatable, dimension(:,:,:)   :: maskW     ! mask in physical space for W points
      real*8, allocatable, dimension(:,:,:)   :: maskZ     ! mask in physical space for Zeta points
      integer, allocatable, dimension(:,:)    :: kbot       ! 0 denotes land, 0<kmt<=nz denotes deepest cell zt(kmt)
      real*8, allocatable, dimension(:)       :: xt,dxt     ! zonal (x) coordinate of T-grid point in meters
      real*8, allocatable, dimension(:)       :: xu,dxu     ! zonal (x) coordinate of U-grid point in meters
      real*8, allocatable, dimension(:)       :: yt,dyt     ! meridional (y) coordinate of T-grid point in meters
      real*8, allocatable, dimension(:)       :: yu,dyu     ! meridional (y) coordinate of V-grid point in meters
      real*8                                  :: x_origin,y_origin ! origin of grid in x and y direction, located at xu_1, yu_1
      real*8, allocatable, dimension(:)       :: zt,zw      ! vertical coordinate in m
      real*8, allocatable, dimension(:)       :: dzt,dzw    ! box thickness in m
      real*8, allocatable, dimension(:,:)     :: area_t     ! Area of T-box in m^2
      real*8, allocatable, dimension(:,:)     :: area_u     ! Area of U-box in m^2
      real*8, allocatable, dimension(:,:)     :: area_v     ! Area of V-box in m^2
      real*8, allocatable, dimension(:,:)     :: coriolis_t ! coriolis frequency at T grid point in 1/s
      real*8, allocatable, dimension(:,:)     :: coriolis_h ! horizontal coriolis frequency at T grid point in 1/s
      real*8, allocatable, dimension(:)       :: cost       ! metric factor for spherical coordinates on T grid
      real*8, allocatable, dimension(:)       :: cosu       ! metric factor for spherical coordinates on U grid
      real*8, allocatable, dimension(:)       :: tantr      ! metric factor for spherical coordinates 
      real*8, allocatable, dimension(:,:)     :: ht         ! total depth in m
      real*8, allocatable, dimension(:,:)     :: hu,hur     ! total depth in m at u-grid
      real*8, allocatable, dimension(:,:)     :: hv,hvr     ! total depth in m at v-grid
      real*8, allocatable, dimension(:,:)     :: beta       ! df/dy in 1/ms
!---------------------------------------------------------------------------------
!     variables related to thermodynamics
!---------------------------------------------------------------------------------
      real*8, allocatable, dimension(:,:,:,:) :: temp,dtemp            ! conservative temperature in deg C and its tendency
      real*8, allocatable, dimension(:,:,:,:) :: salt,dsalt            ! salinity in g/Kg and its tendency
      real*8, allocatable, dimension(:,:,:,:) :: rho                   ! density in kg/m^3
      real*8, allocatable, dimension(:,:,:,:) :: Hd                    ! dynamic enthalpy 
      real*8, allocatable, dimension(:,:,:,:) :: int_drhodT,int_drhodS ! partial derivatives of dyn. enthalpy
      real*8, allocatable, dimension(:,:,:,:) :: Nsqr                  ! Square of stability frequency in 1/s^2
      real*8, allocatable, dimension(:,:,:,:) :: dHd                   ! change of dynamic enthalpy due to advection
      real*8, allocatable, dimension(:,:,:)   :: dtemp_vmix            ! change temperature due to vertical mixing
      real*8, allocatable, dimension(:,:,:)   :: dtemp_hmix            ! change temperature due to lateral mixing
      real*8, allocatable, dimension(:,:,:)   :: dtemp_iso             ! change temperature due to isopynal mixing plus skew mixing
      real*8, allocatable, dimension(:,:,:)   :: dsalt_vmix            ! change salinity due to vertical mixing
      real*8, allocatable, dimension(:,:,:)   :: dsalt_hmix            ! change salinity due to lateral mixing
      real*8, allocatable, dimension(:,:,:)   :: dsalt_iso             ! change salinity due to isopynal mixing plus skew mixing
      real*8, allocatable, dimension(:,:,:)   :: temp_source           ! non conservative source of temperature in K/s
      real*8, allocatable, dimension(:,:,:)   :: salt_source           ! non conservative source of salinity in g/(kgs)
!---------------------------------------------------------------------------------
!     variables related to dynamics
!---------------------------------------------------------------------------------
      real*8, allocatable, dimension(:,:,:,:) :: u,du              ! zonal velocity and its tendency
      real*8, allocatable, dimension(:,:,:,:) :: v,dv              ! meridional velocity and its tendency
      real*8, allocatable, dimension(:,:,:,:) :: w                 ! vertical velocity
      real*8, allocatable, dimension(:,:,:)   :: du_cor            ! change of u due to Coriolis force
      real*8, allocatable, dimension(:,:,:)   :: dv_cor            ! change of v due to Coriolis force
      real*8, allocatable, dimension(:,:,:)   :: du_mix            ! change of v due to implicit vert. mixing
      real*8, allocatable, dimension(:,:,:)   :: dv_mix            ! change of v due to implicit vert. mixing
      real*8, allocatable, dimension(:,:,:)   :: du_adv            ! change of v due to advection
      real*8, allocatable, dimension(:,:,:)   :: dv_adv            ! change of v due to advection
      real*8, allocatable, dimension(:,:,:)   :: u_source          ! non conservative source of zonal velocity
      real*8, allocatable, dimension(:,:,:)   :: v_source          ! non conservative source of meridional velocity
      real*8, allocatable, dimension(:,:,:)   :: p_hydro           ! hydrostatic pressure
      real*8, allocatable, dimension(:,:,:)   :: psi               ! surface pressure or streamfunction
      real*8, allocatable, dimension(:,:,:)   :: dpsi              ! change of streamfunction
      real*8, allocatable, dimension(:,:,:)   :: psin              ! boundary contributions
      real*8, allocatable, dimension(:,:)     :: dpsin             ! boundary contributions
      real*8, allocatable, dimension(:,:)     :: line_psin         ! boundary contributions
      real*8, allocatable, dimension(:,:)     :: surface_taux      ! zonal wind stress
      real*8, allocatable, dimension(:,:)     :: surface_tauy      ! meridional wind stress
      real*8, allocatable, dimension(:,:)     :: forc_rho_surface  ! surface pot. density flux
      real*8, allocatable, dimension(:,:)     :: forc_temp_surface ! surface temperature flux
      real*8, allocatable, dimension(:,:)     :: forc_salt_surface ! surface salinity flux
      real*8, allocatable, dimension(:,:,:)   :: u_wgrid,v_wgrid,w_wgrid       ! velocity on W grid
      real*8, allocatable, dimension(:,:,:)   :: flux_east,flux_north,flux_top ! multi purpose fluxes
!---------------------------------------------------------------------------------
!     variables related to dissipation 
!---------------------------------------------------------------------------------
      real*8, allocatable, dimension(:,:,:)    :: K_diss_v          ! kinetic energy dissipation by vertical, rayleigh and bottom friction
      real*8, allocatable, dimension(:,:,:)    :: K_diss_h          ! kinetic energy dissipation by horizontal friction
      real*8, allocatable, dimension(:,:,:)    :: K_diss_gm         ! mean energy dissipation by GM (TRM formalism only)
      real*8, allocatable, dimension(:,:,:)    :: K_diss_bot        ! mean energy dissipation by bottom and rayleigh friction
      real*8, allocatable, dimension(:,:,:)    :: P_diss_v          ! potential energy dissipation by vertical diffusion
      real*8, allocatable, dimension(:,:,:)    :: P_diss_nonlin     ! potential energy dissipation by nonlinear equation of state
      real*8, allocatable, dimension(:,:,:)    :: P_diss_adv        ! potential energy dissipation by 
      real*8, allocatable, dimension(:,:,:)    :: P_diss_comp       ! potential energy dissipation by compress.
      real*8, allocatable, dimension(:,:,:)    :: P_diss_hmix       ! potential energy dissipation by horizontal mixing
      real*8, allocatable, dimension(:,:,:)    :: P_diss_iso        ! potential energy dissipation by isopycnal mixing 
      real*8, allocatable, dimension(:,:,:)    :: P_diss_skew       ! potential energy dissipation by GM (w/o TRM)
      real*8, allocatable, dimension(:,:,:)    :: P_diss_sources    ! potential energy dissipation by restoring zones, etc
!---------------------------------------------------------------------------------
!     external mode stuff
!---------------------------------------------------------------------------------
      logical :: enable_free_surface  = .false.   ! implicit free surface
      logical :: enable_streamfunction= .false.   ! solve for streamfct instead of surface pressure
      logical :: enable_congrad_verbose = .false. ! print some info
      integer :: congr_itts                       ! number of iterations of poisson solver
      real*8  :: congr_epsilon=1e-12              ! convergence criteria for poisson solver
      integer :: congr_max_iterations = 1000      ! max. number of iterations
      integer :: nisle                            ! number of islands
      integer, allocatable :: boundary(:,:,:),nr_boundary(:),line_dir(:,:,:)  ! positions and direction for island integrals
!---------------------------------------------------------------------------------
!     mixing parameter
!---------------------------------------------------------------------------------
      real*8 :: A_h=0.0    ! lateral viscosity in m^2/s
      real*8 :: K_h=0.0    ! lateral diffusivity in m^2/s
      real*8 :: r_ray=0.0  ! Rayleigh damping coefficient in 1/s  
      real*8 :: r_bot=0.0  ! bottom friction coefficient in 1/s
      real*8 :: r_quad_bot=0.0  ! qudratic bottom friction coefficient
      real*8, allocatable :: r_bot_var_u(:,:)     ! bottom friction coefficient in 1/s, on u points
      real*8, allocatable :: r_bot_var_v(:,:)     ! bottom friction coefficient in 1/s, on v points
      integer :: hor_friction_cosPower = 3
      real*8 :: A_hbi=0.0  ! lateral bihamronic viscosity in m^4/s
      real*8 :: K_hbi=0.0  ! lateral bihamronic diffusivity in m^4/s
      real*8 :: kappaH_0 = 0.0, kappaM_0 = 0.0   ! fixed values for vertical viscosity/diffusivity which are set for no TKE model
      real*8, allocatable :: kappaM(:,:,:)       ! vertical viscosity in m^2/s
      real*8, allocatable :: kappaH(:,:,:)       ! vertical diffusivity in m^2/s
!---------------------------------------------------------------------------------
!     non hydrostatic stuff
!---------------------------------------------------------------------------------
      logical :: enable_hydrostatic = .true.         ! enable hydrostatic approximation
      real*8,allocatable ::  p_non_hydro(:,:,:,:)    ! non-hydrostatic pressure
      real*8,allocatable ::  dw(:,:,:,:)             ! non-hydrostatic stuff
      real*8,allocatable ::  dw_cor(:,:,:)
      real*8,allocatable ::  dw_adv(:,:,:)
      real*8,allocatable ::  dw_mix(:,:,:)
      integer :: congr_itts_non_hydro                ! number of iterations of poisson solver
      real*8  :: congr_epsilon_non_hydro=1e-12       ! convergence criteria for poisson solver
      integer :: congr_max_itts_non_hydro = 1000     ! max. number of iterations
end module main_module   


subroutine allocate_main_module
!=======================================================================
! allocate all arrays within main module
!=======================================================================
 use main_module   
 implicit none

  allocate( xt(is_pe-onx:ie_pe+onx), xu(is_pe-onx:ie_pe+onx)) ; xt=0;xu=0
  allocate( yt(js_pe-onx:je_pe+onx), yu(js_pe-onx:je_pe+onx)) ; yt=0;yu=0
  allocate( dxt(is_pe-onx:ie_pe+onx), dxu(is_pe-onx:ie_pe+onx)) ; dxt=0;dxu=0
  allocate( dyt(js_pe-onx:je_pe+onx), dyu(js_pe-onx:je_pe+onx))  ; dyt=0;dyu=0

  allocate( zt(nz), dzt(nz), zw(nz), dzw(nz) ); zt=0; zw=0; dzt=0; dzw=0
  allocate( cost(js_pe-onx:je_pe+onx), cosu(js_pe-onx:je_pe+onx)); cost=1.0; cosu=1.0;
  allocate( tantr(js_pe-onx:je_pe+onx)); tantr = 0.0
  allocate( coriolis_t(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); coriolis_t=0
  allocate( coriolis_h(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); coriolis_h=0

  allocate( kbot(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); kbot=0
  allocate( ht(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); ht = 0
  allocate( hu(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); hu = 0
  allocate( hv(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); hv = 0
  allocate( hur(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); hur = 0
  allocate( hvr(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); hvr = 0
  allocate( beta(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); beta = 0
  allocate( area_t(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); area_t = 0
  allocate( area_u(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); area_u = 0
  allocate( area_v(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); area_v = 0

  allocate( maskT(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) )
  allocate( maskU(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) )
  allocate( maskV(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) )
  allocate( maskW(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) )
  allocate( maskZ(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) )
  maskW=0.; maskT=0.; maskU=0.; maskV=0.; maskZ =0.;

  allocate( rho(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3) ); rho = 0
  allocate(Nsqr(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3) ); Nsqr = 0
  allocate(  Hd(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3) ); Hd = 0
  allocate( dHd(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3) ); dHd = 0

  allocate( int_drhodT(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3) ); int_drhodT = 0
  allocate( int_drhodS(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3) ); int_drhodS = 0

  allocate(temp(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3) ); temp = 0
  allocate(dtemp(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3) ); dtemp = 0
  allocate(salt(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3) ); salt = 0
  allocate(dsalt(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3) ); dsalt = 0
  allocate(dtemp_vmix(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); dtemp_vmix = 0
  allocate(dsalt_vmix(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); dsalt_vmix = 0
  allocate(dtemp_hmix(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); dtemp_hmix = 0
  allocate(dsalt_hmix(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); dsalt_hmix = 0
  allocate(dsalt_iso(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); dsalt_iso = 0
  allocate(dtemp_iso(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); dtemp_iso = 0
  allocate( forc_temp_surface(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); forc_temp_surface = 0
  allocate( forc_salt_surface(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); forc_salt_surface = 0

  if (enable_tempsalt_sources) then
   allocate(temp_source(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); temp_source = 0
   allocate(salt_source(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); salt_source = 0
  endif
  if (enable_momentum_sources) then
   allocate(u_source(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); u_source = 0
   allocate(v_source(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); v_source = 0
  endif

  allocate( flux_east(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); flux_east = 0
  allocate(flux_north(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); flux_north = 0
  allocate(  flux_top(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); flux_top = 0

  allocate( u(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3) ); u = 0
  allocate( v(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3) ); v = 0
  allocate( w(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3) ); w = 0
  allocate( du(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3) ); du = 0
  allocate( dv(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3) ); dv = 0
  allocate(du_cor(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); du_cor = 0
  allocate(dv_cor(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); dv_cor = 0
  allocate(du_mix(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); du_mix = 0
  allocate(dv_mix(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); dv_mix = 0
  allocate(du_adv(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); du_adv = 0
  allocate(dv_adv(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); dv_adv = 0
  allocate( p_hydro(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); p_hydro = 0
  allocate( psi(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,3) ); psi = 0
  allocate( dpsi(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,3) ); dpsi = 0

  allocate( kappaM(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); kappaM = 0
  allocate( kappaH(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); kappaH = 0

  allocate( surface_taux(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); surface_taux = 0
  allocate( surface_tauy(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); surface_tauy = 0
  allocate( forc_rho_surface(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); forc_rho_surface = 0

  allocate( K_diss_v(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); K_diss_v = 0
  allocate( K_diss_h(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); K_diss_h = 0
  allocate( K_diss_gm(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); K_diss_gm = 0
  allocate( K_diss_bot(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); K_diss_bot = 0
  allocate( P_diss_v(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); P_diss_v = 0
  allocate( P_diss_nonlin(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); P_diss_nonlin = 0
  allocate( P_diss_adv(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); P_diss_adv = 0
  allocate( P_diss_comp(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); P_diss_comp = 0
  allocate( P_diss_hmix(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); P_diss_hmix = 0
  allocate( P_diss_iso(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); P_diss_iso = 0
  allocate( P_diss_skew(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); P_diss_skew = 0
  allocate( P_diss_sources(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); P_diss_sources = 0

  allocate( r_bot_var_u(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); r_bot_var_u = 0
  allocate( r_bot_var_v(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx) ); r_bot_var_v = 0

  if (.not. enable_hydrostatic) then
   allocate( p_non_hydro(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3) ); p_non_hydro = 0
   allocate( dw(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz,3) ); dw = 0
   allocate( dw_cor(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); dw_cor = 0
   allocate( dw_adv(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); dw_adv = 0
   allocate( dw_mix(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); dw_mix = 0
  endif

  allocate( u_wgrid(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); u_wgrid = 0.0
  allocate( v_wgrid(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); v_wgrid = 0.0
  allocate( w_wgrid(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nz) ); w_wgrid = 0.0


end subroutine allocate_main_module





